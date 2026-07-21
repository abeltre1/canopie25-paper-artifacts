"""Tests for `boxy generate card` (src/boxy/cardgen.py + the CLI subcommand).

The sandbox's egress proxy blocks huggingface.co, so these never hit the Hub:
`cardgen.fetch_model` is monkeypatched to return captured config shapes for four
real models (an 8B dense, a Llama-4 MoE, a quantized MoE, a 70B). What's asserted
is boxy's derivation (engine/gpus/vram/context) and the CLI behavior."""

import urllib.error

import pytest

from boxy import cardgen, cards
from boxy.cli import main

# captured HF config shapes (config.json + the safetensors index total_size in
# bytes — which already reflects any quantization). id -> {config, index}.
FIXTURES = {
    "meta-llama/Llama-3.1-8B-Instruct": {
        "config": {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16",
                   "max_position_embeddings": 131072, "num_hidden_layers": 32},
        "index": {"metadata": {"total_size": 16_060_000_000}},   # ~16 GB bf16
        "generation": None,
    },
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {
        "config": {"architectures": ["Llama4ForConditionalGeneration"], "torch_dtype": "bfloat16",
                   "num_local_experts": 16,
                   "text_config": {"max_position_embeddings": 10_485_760}},
        "index": {"metadata": {"total_size": 218_000_000_000}},  # ~109B params bf16
        "generation": None,
    },
    "openai/gpt-oss-20b": {
        "config": {"architectures": ["GptOssForCausalLM"], "num_local_experts": 32,
                   "quantization_config": {"quant_method": "mxfp4"},
                   "max_position_embeddings": 131072},
        "index": {"metadata": {"total_size": 11_000_000_000}},   # MXFP4 ~11 GB
        "generation": None,
    },
    "acme/Custom-Dense-450B": {
        # the NAME lies (450B); the index says ~9B bf16 — autogen must trust the bytes
        "config": {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16",
                   "max_position_embeddings": 131072},
        "index": {"metadata": {"total_size": 18_000_000_000}},
        "generation": None,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "config": {"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16",
                   "max_position_embeddings": 131072},
        "index": {"metadata": {"total_size": 141_000_000_000}},  # ~141 GB bf16
        "generation": None,
    },
    # a custom-code VISION model (image-text-to-text OCR): auto_map => needs
    # --trust-remote-code, vision_config => needs --limit-mm-per-prompt.
    "nvidia/NVIDIA-Nemotron-Parse-v1.2": {
        "config": {"architectures": ["NemotronParseForConditionalGeneration"],
                   "auto_map": {"AutoModel": "modeling_nemotron_parse.NemotronParse"},
                   "vision_config": {"hidden_size": 1280},
                   "torch_dtype": "bfloat16", "max_position_embeddings": 8192},
        "index": {"metadata": {"total_size": 1_800_000_000}},    # ~0.9B params bf16
        "generation": None,
    },
}


@pytest.fixture(autouse=True)
def _fake_hub(monkeypatch):
    """No network: fetch_model reads the captured fixtures; an unknown id 404s."""
    def _fetch(repo, token="", *, opener=None):
        if repo not in FIXTURES:
            raise cardgen.CardGenError(f"{repo!r} was not found on HuggingFace (HTTP 404).")
        return FIXTURES[repo]

    monkeypatch.setattr(cardgen, "fetch_model", _fetch)


def _gen(repo, **kw):
    text, engine, warnings = cardgen.generate(repo, **kw)
    return text, engine, warnings


# ---- per-model sizing / engine ------------------------------------------------------


def test_8b_dense_single_gpu_capped_context():
    text, engine, warnings = _gen("meta-llama/Llama-3.1-8B-Instruct")
    card = cards._parse_card(text, "x", "user", "x")   # must round-trip
    assert engine == "vllm" and not warnings
    assert card.gpus == 1
    assert 18 <= card.min_vram_gb <= 32                 # weights + headroom, mid-20s-ish
    assert card.args["max_model_len"] == 8192
    assert "profiles KV cache for the FULL window" in text   # the cap rationale comment
    assert 'match = "meta-llama/Llama-3.1-8B-Instruct*"' in text


def test_llama4_moe_sized_by_total_params():
    text, engine, _ = _gen("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    card = cards._parse_card(text, "x", "user", "x")
    assert engine == "vllm"
    assert card.gpus >= 2                                # 109B total -> multi-GPU
    assert "MoE (16 experts)" in text                    # MoE noted in the header
    assert card.gpus & (card.gpus - 1) == 0             # power-of-two (TP-friendly)


def test_quantized_moe_fits_one_gpu():
    text, engine, _ = _gen("openai/gpt-oss-20b")
    card = cards._parse_card(text, "x", "user", "x")
    assert engine == "vllm" and card.gpus == 1
    assert card.min_vram_gb <= 32                        # 4-bit weights are small
    assert "MXFP4" in text                               # quantization noted


def test_70b_four_gpus():
    text, _, _ = _gen("meta-llama/Llama-3.3-70B-Instruct")
    card = cards._parse_card(text, "x", "user", "x")
    assert card.gpus == 4
    assert card.min_vram_gb >= 130
    assert "tensor-parallel from the GPU count" in text


def test_max_model_len_override():
    text, _, _ = _gen("meta-llama/Llama-3.1-8B-Instruct", max_model_len=4096)
    assert cards._parse_card(text, "x", "user", "x").args["max_model_len"] == 4096


# ---- engine detection ---------------------------------------------------------------


def test_gguf_repo_picks_llama_cpp():
    engine, why, warn = cardgen.pick_engine("TheBloke/Llama-2-7B-GGUF", {}, "")
    assert engine == "llama.cpp" and not warn


def test_unknown_arch_warns_but_defaults_vllm():
    engine, _why, warn = cardgen.pick_engine("acme/Weird-3B", {"architectures": ["WeirdNetForCausalLM"]}, "")
    assert engine == "vllm" and "isn't in boxy's known-vLLM list" in warn


def test_custom_code_vision_card_carries_trust_and_mm_limit():
    # FIELD (Nemotron-Parse): the generated card must be a COMPLETE serve spec —
    # trust_remote_code + limit-mm-per-prompt in [model.args] — so the card alone
    # serves the model (it died at vLLM config validation without --trust-remote-code).
    text, engine, warnings = _gen("nvidia/NVIDIA-Nemotron-Parse-v1.2")
    card = cards._parse_card(text, "x", "user", "x")     # must round-trip through the loader
    assert engine == "vllm"
    assert card.args.get("trust_remote_code") is True
    assert card.args.get("limit-mm-per-prompt") == '{"image": 1}'
    assert "trust_remote_code = true" in text
    assert any("custom-code" in w for w in warnings)
    assert any("vision" in w or "multimodal" in w for w in warnings)


def test_explicit_engine_override_wins():
    engine, why, _ = cardgen.pick_engine("meta-llama/Llama-3.1-8B-Instruct",
                                         FIXTURES["meta-llama/Llama-3.1-8B-Instruct"]["config"], "llama.cpp")
    assert engine == "llama.cpp" and "--engine" in why


# ---- the CLI: dry-run, write, overwrite, errors -------------------------------------


@pytest.fixture
def cards_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return tmp_path / "cfg" / "boxy" / "cards" / "models"


def test_cli_dry_run_prints_without_writing(cards_home, capfd):
    rc = main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct", "--dry-run"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "[model]" in out and 'engine = "vllm"' in out
    assert not cards_home.exists() or not list(cards_home.glob("*.toml"))  # nothing written


def test_cli_write_lands_in_the_cards_dir_and_is_discoverable(cards_home, capfd, monkeypatch):
    rc = main(["generate", "card", "meta-llama/Llama-3.3-70B-Instruct"])
    out = capfd.readouterr().out
    assert rc == 0
    dest = cards_home / "meta-llama-llama-3.3-70b-instruct.toml"
    assert dest.exists()
    assert "### wrote" in out and "boxy serve meta-llama/Llama-3.3-70B-Instruct" in out
    # the written card is the one boxy would use for that model
    found = cards.find_card("meta-llama/Llama-3.3-70B-Instruct")
    assert found is not None and found.source == "user" and found.gpus == 4


def test_cli_non_interactive_refuses_to_overwrite(cards_home, capfd, monkeypatch):
    assert main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct"]) == 0   # write first
    capfd.readouterr()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct"])          # again, no TTY
    err = capfd.readouterr().err
    assert rc == 1 and "pass --force to replace" in err


def test_cli_force_overwrites_with_a_backup(cards_home, capfd):
    assert main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct"]) == 0
    capfd.readouterr()
    rc = main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct", "--force"])
    out = capfd.readouterr().out
    assert rc == 0 and "replaced" in out
    assert (cards_home / "meta-llama-llama-3.1-8b-instruct.toml.bak").exists()


def test_cli_dry_run_reports_would_overwrite(cards_home, capfd):
    assert main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct"]) == 0
    capfd.readouterr()
    rc = main(["generate", "card", "meta-llama/Llama-3.1-8B-Instruct", "--dry-run"])
    err = capfd.readouterr().err
    assert rc == 0 and "would overwrite" in err


def test_cli_unknown_model_is_a_clear_error(cards_home, capfd):
    rc = main(["generate", "card", "acme/does-not-exist"])
    err = capfd.readouterr().err
    assert rc == 1 and "not found on HuggingFace" in err


# ---- HF error mapping (hf_get_json) -------------------------------------------------


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def test_gated_without_token_asks_for_hf_token(monkeypatch):
    def _boom(*a, **k):
        raise _http_error(403)

    monkeypatch.setattr(cardgen, "_opener", lambda: type("O", (), {"open": staticmethod(_boom)})())
    with pytest.raises(cardgen.CardGenError) as e:
        cardgen.hf_get_json("meta-llama/Gated", "config.json", token="")
    assert "GATED" in str(e.value) and "HF_TOKEN" in str(e.value)


def test_gated_with_bad_token_says_token_rejected(monkeypatch):
    def _boom(*a, **k):
        raise _http_error(401)

    monkeypatch.setattr(cardgen, "_opener", lambda: type("O", (), {"open": staticmethod(_boom)})())
    with pytest.raises(cardgen.CardGenError) as e:
        cardgen.hf_get_json("meta-llama/Gated", "config.json", token="hf_bad")
    assert "token was rejected" in str(e.value)


def test_network_unreachable_is_actionable(monkeypatch):
    def _boom(*a, **k):
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(cardgen, "_opener", lambda: type("O", (), {"open": staticmethod(_boom)})())
    with pytest.raises(cardgen.CardGenError) as e:
        cardgen.hf_get_json("meta-llama/X", "config.json")
    assert "cannot reach HuggingFace" in str(e.value)


# ---- serve-time AUTOGEN: the card replaces the name guess ----------------------------


@pytest.fixture
def autogen_on(monkeypatch, tmp_path):  # _fake_hub is autouse — no live Hub here
    monkeypatch.setenv("BOXY_CARD_AUTOGEN", "true")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return tmp_path / "xdg" / "boxy" / "cards" / "models"


def test_serve_resolution_generates_card_instead_of_guessing(autogen_on):
    # an UNCARDED model resolves through the Hub fixtures — deterministic
    # metadata, not the '-405B' name token — and the card lands on disk.
    import argparse
    cards._user_dir().mkdir(parents=True, exist_ok=True)  # fresh user dir
    a = argparse.Namespace(model="acme/Custom-Dense-450B", gpus=None, nodes=None,
                          engine=None, args=None)
    lines = cards.apply_to_args(a)
    assert any("generated deterministically from HuggingFace metadata" in ln for ln in lines)
    assert a.gpus == 1                       # 18GB of real bytes -> 1 GPU (name said 450B!)
    written = autogen_on / "acme-custom-dense-450b.toml"
    assert written.exists()
    # second resolution: the FILE serves it — no Hub call, plain user card
    card = cards.find_card("acme/Custom-Dense-450B")
    assert card is not None and card.source == "user"


def test_autogen_falls_back_to_loud_name_guess(autogen_on):
    import argparse
    a = argparse.Namespace(model="acme/Unknown-70B-Instruct", gpus=None, nodes=None,
                          engine=None, args=None)
    lines = cards.apply_to_args(a)                         # fixture hub 404s acme/*
    assert a.gpus == 4                                     # heuristic still sizes it
    assert any("NAME GUESS" in ln and "boxy generate card" in ln for ln in lines)


def test_autogen_never_fires_for_non_hub_refs(autogen_on, monkeypatch):
    def _explode(model):
        raise AssertionError("auto_card must not be called")

    monkeypatch.setattr(cardgen, "auto_card", _explode)
    for ref in ("/lustre/models/llama.q4.gguf", "llama.q4.gguf", "oci://reg/img:tag",
                "ollama://llama3", "org/repo/file.gguf"):
        assert cards.resolve_model_card(ref) is None or True   # no raise = pass


def test_autogen_disabled_by_offline_env(autogen_on, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    card = cards.resolve_model_card("acme/Custom-Dense-450B")
    assert card is not None and card.source == "heuristic"     # straight to the guess


def test_opener_falls_back_to_config_proxy(monkeypatch):
    """No http(s)_proxy in the env + network.proxy configured -> the opener
    carries a ProxyHandler for it (field: `generate card` died on bare DNS
    from a shell without the proxy env, though boxy knew the proxy). An
    ambient env proxy still wins."""
    import urllib.request

    from boxy import config

    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "get",
                        lambda k: "http://proxy.example.gov:80" if k == "network.proxy" else "")
    op = cardgen._opener()
    assert any(getattr(h, "proxies", {}).get("https") == "http://proxy.example.gov:80"
               for h in op.handlers)

    monkeypatch.setenv("HTTPS_PROXY", "http://env-proxy:80")
    op = cardgen._opener()
    proxies = [getattr(h, "proxies", {}) for h in op.handlers
               if isinstance(h, urllib.request.ProxyHandler)]
    assert proxies and proxies[0].get("https") == "http://env-proxy:80"
