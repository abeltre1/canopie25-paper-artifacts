"""Model cards — the turnkey per-model deployment knowledge. Matching (user >
packaged, longest glob wins), the size heuristic, the flags-always-win rule,
and the end-to-end dryrun where `boxy serve <70B model> --scheduler slurm`
requests 4 GPUs with ZERO geometry flags."""

import argparse

import pytest

from boxy import cards
from boxy.cli import main

# ---- matching ---------------------------------------------------------------------


def test_packaged_card_matches_llama_70b():
    card = cards.find_card("meta-llama/Llama-3.3-70B-Instruct")
    assert card and card.source == "packaged"
    assert card.gpus == 4 and card.engine == "vllm"
    assert card.args.get("max_model_len") == 8192


def test_transport_scheme_is_stripped_before_matching():
    assert cards.model_key("hf://meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"
    card = cards.find_card("hf://meta-llama/Llama-3.1-8B-Instruct")
    assert card and card.gpus == 1


def test_longest_match_wins_gguf_over_safetensors_family():
    # both Qwen2.5-7B-Instruct* and Qwen2.5-7B-Instruct-GGUF* match the GGUF id;
    # the more specific (longer) pattern must win -> llama.cpp
    card = cards.find_card("Qwen/Qwen2.5-7B-Instruct-GGUF")
    assert card and card.engine == "llama.cpp"
    plain = cards.find_card("Qwen/Qwen2.5-7B-Instruct")
    assert plain and plain.engine == "vllm"


def test_user_card_beats_packaged(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    (d / "my-llama.toml").write_text(
        '[model]\nmatch = "meta-llama/Llama-3.3-70B-Instruct*"\ngpus = 8\n')
    card = cards.find_card("meta-llama/Llama-3.3-70B-Instruct")
    assert card and card.source == "user" and card.gpus == 8


def test_malformed_user_card_raises_with_path(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    (d / "broken.toml").write_text("[model]\n# no match key\n")
    with pytest.raises(ValueError, match="broken"):
        cards.load_cards()


# ---- size heuristic ---------------------------------------------------------------


@pytest.mark.parametrize("model,gpus", [
    ("someorg/CoolModel-3B-Instruct", 1),
    ("someorg/CoolModel-13B", 1),
    ("someorg/CoolModel-32B-Chat", 2),
    ("someorg/CoolModel-70B", 4),
    ("someorg/CoolModel-180B", 8),
    ("someorg/Mega-8x22B", 8),          # MoE: 8x22 = 176B effective
])
def test_size_heuristic_tiers(model, gpus):
    card = cards.size_heuristic(model)
    assert card is not None and card.source == "heuristic"
    assert card.gpus == gpus


def test_size_heuristic_none_without_size_token():
    assert cards.size_heuristic("someorg/whisper-large-v3") is None
    # 'b' inside a word must not parse as a size ("...-web", "bge-...")
    assert cards.size_heuristic("someorg/bge-reranker-base") is None


# ---- apply_to_args: flags always win ------------------------------------------------


def _args(model, gpus=None, nodes=None, engine=None):
    return argparse.Namespace(model=model, gpus=gpus, nodes=nodes, engine=engine)


def test_apply_fills_gpus_from_card_and_prints_provenance():
    a = _args("meta-llama/Llama-3.3-70B-Instruct")
    lines = cards.apply_to_args(a)
    assert a.gpus == 4
    assert any("gpus: 4" in ln and "card" in ln for ln in lines)


def test_apply_never_overrides_explicit_flags():
    a = _args("meta-llama/Llama-3.3-70B-Instruct", gpus=2, engine="llama.cpp")
    lines = cards.apply_to_args(a)
    assert a.gpus == 2 and a.engine == "llama.cpp"
    assert not any(ln.startswith("gpus:") or ln.startswith("engine:") for ln in lines)


def test_apply_uses_heuristic_for_unknown_model():
    a = _args("someorg/NewHotness-70B-Instruct")
    lines = cards.apply_to_args(a)
    assert a.gpus == 4
    assert any("heuristic" in ln for ln in lines)


def test_apply_no_card_no_size_is_a_noop():
    a = _args("someorg/whisper-large-v3")
    assert cards.apply_to_args(a) == []
    assert a.gpus is None


# ---- end-to-end: turnkey dryrun ------------------------------------------------------


def test_zero_flag_70b_submission_requests_4_gpus(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.3-70B-Instruct",
               "--scheduler", "slurm", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: gpus: 4 per node" in out          # card decision line printed
    assert "#SBATCH --gpus-per-node=4" in out       # ...and it reached the batch script


def test_explicit_gpus_still_wins_end_to_end(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.3-70B-Instruct",
               "--scheduler", "slurm", "--gpus", "8", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "#SBATCH --gpus-per-node=8" in out
    assert "auto: gpus:" not in out                 # the card stayed silent


def test_card_engine_args_reach_the_box(monkeypatch):
    # node-side: resolve() merges card args into box.args (tack-on-last keeps
    # user args winning at engine-command build time)
    from boxy import resolve as resolve_mod

    monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "cuda")
    monkeypatch.setattr(resolve_mod, "detect_runtime", lambda: ("podman", "test"))
    res = resolve_mod.resolve("hf://meta-llama/Llama-3.3-70B-Instruct",
                              require_exists=False, here=True)
    assert res.box.args.get("max_model_len") == 8192
    assert any("engine args" in d and "card" in d for d in res.decisions)
