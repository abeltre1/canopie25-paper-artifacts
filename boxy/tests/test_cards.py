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


def test_apply_merges_card_engine_args():
    # the 8B card caps context so vLLM doesn't OOM profiling the 128K window
    a = _args("hf://meta-llama/Llama-3.1-8B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a)
    assert a.args == ["--max-model-len", "8192"]
    assert any("engine args" in ln and "--max-model-len 8192" in ln for ln in lines)


def test_apply_card_engine_args_user_wins():
    a = _args("hf://meta-llama/Llama-3.1-8B-Instruct")
    a.args = ["--max-model-len", "32768"]           # user override
    cards.apply_to_args(a)
    # card flag FIRST, user AFTER -> the engine's argparse last-wins -> 32768
    assert a.args == ["--max-model-len", "8192", "--max-model-len", "32768"]


def test_engine_flags_bool_and_scalar():
    assert cards.engine_flags({"max_model_len": 8192}) == ["--max-model-len", "8192"]
    assert cards.engine_flags({"enforce_eager": True}) == ["--enforce-eager"]
    assert cards.engine_flags({"enforce_eager": False}) == []
    assert cards.engine_flags({}) == []


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


# ---- system cards -----------------------------------------------------------------


def test_system_cards_cover_every_type():
    types = {typ for _stem, typ, _r in cards.system_card_entries()}
    assert {"laptop", "hpc-slurm", "hpc-flux", "cloud", "openshift"} <= types
    # 3 examples per type (user direction)
    from collections import Counter
    counts = Counter(typ for _s, typ, _r in cards.system_card_entries())
    for typ in ("laptop", "hpc-slurm", "hpc-flux", "cloud", "openshift"):
        assert counts[typ] >= 3


def test_system_card_matches_by_location_name_and_stem():
    from boxy.location import Location
    # canonical [location].name is unique and self-describing
    assert Location.from_toml(cards.system_card_path("slurm-cuda")).scheduler == "slurm"
    assert Location.from_toml(cards.system_card_path("flux-rocm")).accelerator == "rocm"
    # a unique file stem also resolves (laptop-podman.toml)
    assert Location.from_toml(cards.system_card_path("podman")).runtime == "podman"


def test_unknown_system_card_lists_choices():
    with pytest.raises(ValueError, match="unknown system card"):
        cards.system_card_path("no-such-system")


def test_serve_with_system_card_dryrun(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.1-8B-Instruct",
               "--system", "slurm-cuda", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: system: slurm-cuda" in out
    assert "#SBATCH" in out and "boxy-llama-3.1-8b-instruct" in out


def test_boxy_cards_lists_models_and_systems(capsys):
    rc = main(["cards"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "meta-llama/Llama-3.3-70B-Instruct" in out
    assert "hpc-slurm" in out and "cloud" in out and "openshift" in out


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


def test_packaged_nemotron_parse_card_carries_trust_and_mm_limit():
    # FIELD (hops, repeatedly): serving nvidia/NVIDIA-Nemotron-Parse died at
    # vLLM config validation for want of --trust-remote-code. The packaged card
    # bakes the complete serve spec in, so the FIRST submit is right — no Hub
    # probe, no death-path resubmit needed.
    card = cards.find_card("nvidia/NVIDIA-Nemotron-Parse-v1.2")
    assert card and card.source == "packaged" and card.engine == "vllm"
    assert card.args.get("trust_remote_code") is True
    assert card.args.get("limit-mm-per-prompt") == '{"image": 1}'
    assert card.gpus == 1


def test_ensure_card_args_merges_missing_only():
    # the render-time guard: card args land on the FINAL box no matter how the
    # model reference mutated (bare-id rewrite, prestage path swap) — but a
    # value already on the box (user/explicit) is never overwritten.
    from boxy.box import Box
    from boxy.cli import _ensure_card_args

    box = Box(name="x", image="", engine="vllm",
              model="nvidia/NVIDIA-Nemotron-Parse-v1.2", ports=[8000],
              args={"max_model_len": 4096})                       # user override present
    healed, note = _ensure_card_args(box, "hf://nvidia/NVIDIA-Nemotron-Parse-v1.2")
    assert healed.args.get("trust_remote_code") is True           # merged from the card
    assert healed.args.get("limit-mm-per-prompt") == '{"image": 1}'
    assert healed.args["max_model_len"] == 4096                   # override untouched
    assert "trust_remote_code" in note and "merged into the final command" in note
    again, note2 = _ensure_card_args(healed, "hf://nvidia/NVIDIA-Nemotron-Parse-v1.2")
    assert again.args == healed.args and note2 == ""              # idempotent


def test_ensure_card_args_matches_by_box_model_when_cli_ref_is_a_path():
    # prestage rewrote args.model? the guard also tries the box's model value.
    from boxy.box import Box
    from boxy.cli import _ensure_card_args

    box = Box(name="x", image="", engine="vllm",
              model="nvidia/NVIDIA-Nemotron-Parse-v1.2", ports=[8000], args={})
    healed, _ = _ensure_card_args(box, "/scratch/staged/model-dir")
    assert healed.args.get("trust_remote_code") is True


def test_stale_user_card_inherits_packaged_safety_args(tmp_path, monkeypatch):
    # THE field failure: a stale `generate card` user card (pre-trust_remote_code
    # cardgen) shadowed the packaged Nemotron-Parse card and silently dropped
    # --trust-remote-code. Args now LAYER: user keys win, missing keys fall
    # through to the packaged card.
    d = tmp_path / "cfg" / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    (d / "nvidia-nvidia-nemotron-parse-v1.2.toml").write_text(
        '[model]\nmatch = "nvidia/NVIDIA-Nemotron-Parse-v1.2*"\nengine = "vllm"\n'
        'gpus = 1\n[model.args]\nmax_model_len = 4096\n')          # stale: no trust flag
    args, label = cards.layered_args("hf://nvidia/NVIDIA-Nemotron-Parse-v1.2")
    assert args["max_model_len"] == 4096                           # user value wins
    assert args["trust_remote_code"] is True                       # inherited from packaged
    assert args["limit-mm-per-prompt"] == '{"image": 1}'
    assert "inherited from the packaged" in label


def test_layered_args_user_only_and_packaged_only():
    args, label = cards.layered_args("nvidia/NVIDIA-Nemotron-Parse-v1.2")
    assert args.get("trust_remote_code") is True and "packaged" in label   # packaged only
    none_args, none_label = cards.layered_args("acme/NoCardAnywhere-3B")
    assert none_args == {} and none_label == ""


def test_card_pip_layered_and_missing_pkg_parser():
    # the packaged Nemotron-Parse card declares open_clip_torch; a user card
    # can add more but never erases the packaged deps.
    assert cards.layered_pip("hf://nvidia/NVIDIA-Nemotron-Parse-v1.2") == ["open_clip_torch"]

    from boxy.cli import _missing_py_packages

    err = ("ImportError: This modeling file requires the following packages that were "
           "not found in your environment: open_clip. Run `pip install open_clip`")
    assert _missing_py_packages(err) == ["open_clip_torch"]       # import -> PyPI name
    assert _missing_py_packages("CUDA out of memory") == []
    multi = ("This modeling file requires the following packages that were not found "
             "in your environment: einops, cv2. Run `pip install einops cv2`")
    assert _missing_py_packages(multi) == ["einops", "opencv-python-headless"]


def test_pip_wrapper_wraps_serve_command():
    from boxy import engines
    from boxy.box import Box
    from boxy.location import Location

    box = Box(name="x", image="", engine="vllm", model="nvidia/NVIDIA-Nemotron-Parse-v1.2",
              ports=[8000], args={"trust_remote_code": True}, pip=["open_clip_torch"])
    cmd = engines.build_serve_cmd(box, Location(name="l"), box.model)
    assert cmd[0:2] == ["sh", "-c"]
    assert cmd[2].startswith("pip install --no-cache-dir --quiet open_clip_torch && exec vllm serve")
    assert "--trust-remote-code" in cmd[2]
