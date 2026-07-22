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
    # FIELD (clustera, repeatedly): serving nvidia/NVIDIA-Nemotron-Parse died at
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


def test_packaged_llama4_scout_card():
    # shipped from the captured HF config: filtered-egress laptops can't run
    # `generate card`, so the MoE's geometry + context cap come packaged.
    card = cards.find_card("hf://meta-llama/Llama-4-Scout-17B-16E-Instruct")
    assert card and card.source == "packaged" and card.engine == "vllm"
    assert card.gpus == 4 and card.min_vram_gb == 228
    assert card.args["max_model_len"] == 8192


def test_single_node_multi_gpu_gets_tensor_parallel():
    # FIELD (Llama-4-Scout): a 4-GPU single-node allocation still ran vLLM with
    # its default tensor_parallel_size=1 — 218GB of MoE weights loaded onto GPU
    # 0 alone and OOM'd (uniproc executor in the traceback). Single-node
    # multi-GPU now shards across the allocation; user overrides still win.
    from boxy import engines
    from boxy.box import Box
    from boxy.location import Location, Resources

    box = Box(name="x", image="", engine="vllm", model="m", ports=[8000])
    loc4 = Location(name="l", resources=Resources(nodes=1, gpus_per_node=4))
    cmd = " ".join(engines.build_serve_cmd(box, loc4, "m"))
    assert "--tensor-parallel-size=4" in cmd

    override = Box(name="x", image="", engine="vllm", model="m", ports=[8000],
                   args={"tensor_parallel_size": 2})
    cmd2 = " ".join(engines.build_serve_cmd(override, loc4, "m"))
    assert "--tensor-parallel-size=2" in cmd2 and "--tensor-parallel-size=4" not in cmd2

    loc1 = Location(name="l", resources=Resources(nodes=1, gpus_per_node=1))
    assert "--tensor-parallel-size" not in " ".join(engines.build_serve_cmd(box, loc1, "m"))


# ---- fit_geometry: the card solver (demand x supply -> nodes/gpus) -------------------


def test_fit_geometry_parity_with_hand_sized_cards():
    # CALIBRATION CONTRACT: on the assumed 4x80GB shape the solver reproduces
    # every packaged card's hand-sized gpus exactly — so shipping the solver
    # changes NO existing deployment until a system card declares real hardware.
    for card in cards.load_cards():
        if card.source != "packaged" or not card.min_vram_gb or card.nodes:
            continue
        nodes, gpus, _ = cards.fit_geometry(card.min_vram_gb, 0, 0)
        if nodes == 1:
            assert gpus == card.gpus, card.card_name
        else:
            # a card too big for one assumed node spills to FULL nodes — its
            # advisory gpus must be the full assumed width
            assert gpus == card.gpus == 4, card.card_name


def test_fit_geometry_fat_vram_uses_fewer_gpus():
    # 70B (140GB weights): 4 GPUs on 80GB parts, but TWO on 140GB parts (clusterc)
    assert cards.fit_geometry(140, 4, 80)[:2] == (1, 4)
    assert cards.fit_geometry(140, 4, 140)[:2] == (1, 2)


def test_fit_geometry_spills_to_full_nodes_and_says_ray():
    nodes, gpus, why = cards.fit_geometry(810, 4, 80)     # 405B-class
    assert (nodes, gpus) == (4, 4)
    assert "exceeds one node" in why


def test_fit_geometry_states_assumptions_when_shape_unknown():
    _, _, why = cards.fit_geometry(24, 0, 0)
    assert "assuming 80GB-class GPUs" in why and "assuming 4 GPUs/node" in why


# ---- system cards carry the node shape ----------------------------------------------


def test_system_shape_from_user_cluster_card(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "systems"
    d.mkdir(parents=True)
    (d / "clusterc.toml").write_text(
        '[location]\nname = "clusterc"\nscheduler = "slurm"\n'
        '[location.resources]\ngpus_per_node = 4\ngpu_vram_gb = 140\n')
    assert cards.system_shape("clusterc") == (4, 140, "clusterc")
    assert cards.system_shape("no-such-cluster") is None


def test_apply_solves_geometry_from_shape():
    # same command, different metal: the 70B card needs 2 GPUs on a 4x140 node…
    a = _args("meta-llama/Llama-3.3-70B-Instruct")
    lines = cards.apply_to_args(a, shape=(4, 140, "system card 'clusterc' for clusterc"))
    assert a.gpus == 2 and a.nodes is None
    assert any("gpus: 2 per node" in ln and "clusterc" in ln for ln in lines)
    # …and becomes a 2-node Ray instance on skinny 2x80 nodes — zero flags either way
    b = _args("meta-llama/Llama-3.3-70B-Instruct")
    lines = cards.apply_to_args(b, shape=(2, 80, "system card 'small' for small"))
    assert b.gpus == 2 and b.nodes == 2
    assert any("nodes: 2" in ln and "Ray" in ln for ln in lines)


def test_apply_solver_bypassed_by_power_user_flags_and_card_nodes(tmp_path, monkeypatch):
    # explicit --gpus/--nodes: the solver never runs
    a = _args("meta-llama/Llama-3.3-70B-Instruct", gpus=8)
    cards.apply_to_args(a, shape=(4, 140, "x"))
    assert a.gpus == 8 and a.nodes is None
    a = _args("meta-llama/Llama-3.3-70B-Instruct", nodes=3)
    cards.apply_to_args(a, shape=(4, 140, "x"))
    assert a.nodes == 3 and a.gpus == 4                   # card copy, not the solver
    # a card that PINS nodes is author intent — also bypasses the solver
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    (d / "pinned.toml").write_text(
        '[model]\nmatch = "acme/Pinned-Geo-70B*"\nengine = "vllm"\n'
        'gpus = 4\nnodes = 2\nmin_vram_gb = 140\n')
    b = _args("acme/Pinned-Geo-70B-Instruct")
    cards.apply_to_args(b, shape=(4, 140, "x"))
    assert b.gpus == 4 and b.nodes == 2


def test_zero_flag_geometry_solved_end_to_end(monkeypatch, tmp_path, capsys):
    # config-pinned shape (the env power-user path): 4x140GB parts -> the same
    # zero-flag 70B serve now requests 2 GPUs, not 4
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOXY_GPUS_PER_NODE", "4")
    monkeypatch.setenv("BOXY_GPU_VRAM_GB", "140")
    rc = main(["serve", "hf://meta-llama/Llama-3.3-70B-Instruct",
               "--scheduler", "slurm", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: gpus: 2 per node" in out
    assert "#SBATCH --gpus-per-node=2" in out


# ---- card-pinned engine images ([model.images]) --------------------------------------


def test_card_pins_engine_image_per_accelerator():
    # FIELD (Nemotron-3-Nano on clusterb): a brand-new architecture dies with
    # 'Engine core initialization failed' in an older engine image — the card
    # now pins a CURRENT vLLM per accelerator; --image always wins.
    a = _args("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    a.image, a.accelerator = None, "rocm"
    lines = cards.apply_to_args(a)
    assert a.image == "docker.io/rocm/vllm:latest"
    assert any("pins a rocm image" in ln for ln in lines)
    b = _args("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    b.image, b.accelerator = None, "cuda"
    cards.apply_to_args(b)
    assert b.image == "docker.io/vllm/vllm-openai:latest"
    c = _args("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    c.image, c.accelerator = "my/own:img", "rocm"          # power user wins
    cards.apply_to_args(c)
    assert c.image == "my/own:img"


def test_nemotron_family_cards_resolve():
    # every family member hits its card, and the parity contract above already
    # validates each card's gpus against its min_vram_gb
    for mid, expect in (("nvidia/NVIDIA-Nemotron-Nano-9B-v2", "nvidia-nemotron-nano-v2"),
                        ("nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", "nvidia-nemotron-nano-v2"),
                        ("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "nvidia-llama-nemotron-70b"),
                        ("nvidia/Llama-3_3-Nemotron-Super-49B-v1_5", "nvidia-llama-nemotron-super-49b"),
                        ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nvidia-nemotron-3-nano"),
                        ("nvidia/NVIDIA-Nemotron-Parse-v1.2", "nvidia-nemotron-parse")):
        card = cards.find_card(mid)
        assert card is not None and card.card_name == expect, mid
    # the hybrid-Mamba members carry NVIDIA's recommended SSM cache dtype
    assert cards.find_card("nvidia/NVIDIA-Nemotron-Nano-9B-v2").args["mamba_ssm_cache_dtype"] == "float32"
    # the NAS-derived Super needs remote code
    assert cards.find_card("nvidia/Llama-3_3-Nemotron-Super-49B-v1").args["trust_remote_code"] is True


# ---- per-accelerator card knowledge (env / arg overlays / hardware constraints) ------


ULTRA_CARD = """
[model]
match = "acme/Test-Ultra-550B-NVFP4*"
engine = "vllm"
gpus = 8
min_vram_gb = 300
accelerators = ["cuda"]
unsupported_hint = "NVFP4 is Blackwell/CUDA-only - on ROCm serve the FP8 variant."
[model.env]
VLLM_USE_FLASHINFER_MOE_FP4 = "1"
[model.args]
max_model_len = 262144
trust_remote_code = true
[model.args.cuda]
enable_flashinfer_autotune = true
kv_cache_dtype = "fp8"
[model.args.rocm]
mamba_backend = "triton"
"""


@pytest.fixture
def ultra_card(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    (d / "ultra-test.toml").write_text(ULTRA_CARD)


def test_effective_args_overlays_per_accelerator(ultra_card):
    # NVIDIA reference commands carry CUDA-only knobs (FlashInfer) that would
    # crash a ROCm vLLM — one card stays honest on both kinds of metal.
    card = cards.find_card("acme/Test-Ultra-550B-NVFP4")
    cuda = cards.effective_args(card.args, "cuda")
    rocm = cards.effective_args(card.args, "rocm")
    assert cuda["enable_flashinfer_autotune"] is True and "mamba_backend" not in cuda
    assert rocm["mamba_backend"] == "triton" and "enable_flashinfer_autotune" not in rocm
    assert cuda["max_model_len"] == rocm["max_model_len"] == 262144   # shared base
    # nested overlay tables never leak into flags
    assert "--cuda" not in cards.engine_flags(card.args)


def test_card_env_layered_and_merged(ultra_card):
    assert cards.layered_env("acme/Test-Ultra-550B-NVFP4") == {"VLLM_USE_FLASHINFER_MOE_FP4": "1"}
    from boxy.cli import _ensure_card_args
    from boxy.box import Box
    box = Box(name="b", model="acme/Test-Ultra-550B-NVFP4", engine="vllm")
    box2, note = _ensure_card_args(box, "acme/Test-Ultra-550B-NVFP4", accel="cuda")
    assert box2.env["VLLM_USE_FLASHINFER_MOE_FP4"] == "1"
    assert box2.args["kv_cache_dtype"] == "fp8"                # cuda overlay flattened
    assert "mamba_backend" not in box2.args
    assert "env:" in note
    # user box.env wins over the card
    box3 = Box(name="b", model="acme/Test-Ultra-550B-NVFP4", engine="vllm",
               env={"VLLM_USE_FLASHINFER_MOE_FP4": "0"})
    box4, _ = _ensure_card_args(box3, "acme/Test-Ultra-550B-NVFP4", accel="cuda")
    assert box4.env["VLLM_USE_FLASHINFER_MOE_FP4"] == "0"


def test_hardware_bound_checkpoint_refuses_up_front(ultra_card):
    # an NVFP4 quant on a ROCm system fails deep in kernel init an hour into
    # the queue — the card refuses BEFORE submission, naming the alternative.
    a = _args("acme/Test-Ultra-550B-NVFP4")
    a.image, a.accelerator = None, "rocm"
    with pytest.raises(ValueError, match="cuda only, not rocm.*FP8 variant"):
        cards.apply_to_args(a)
    b = _args("acme/Test-Ultra-550B-NVFP4")
    b.image, b.accelerator = None, "cuda"
    lines = cards.apply_to_args(b)                            # cuda proceeds
    assert any("--enable-flashinfer-autotune" in ln for ln in lines)


def test_nemotron3_family_per_accelerator_serving():
    # Research-backed (vLLM day-0 blog + NVIDIA cookbooks + AMD ROCm blogs):
    # one card per checkpoint serves BOTH kinds of metal honestly.
    ultra = cards.find_card("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4")
    # cuda overlay = the cookbook's HOPPER/H200 configuration (this fleet has
    # no Blackwell): float16 mamba cache + stochastic rounding, NO FlashInfer
    # (FP4 MoE kernels are Blackwell-only), pinned v0.22.0 image (the
    # FlashInfer FP4 env switch was REMOVED in vLLM 0.24).
    cuda = cards.effective_args(ultra.args, "cuda")
    assert cuda["mamba_cache_dtype"] == "float16"
    assert cuda["enable_mamba_cache_stochastic_rounding"] is True
    assert cuda["max_num_seqs"] == 128
    assert "enable_flashinfer_autotune" not in cuda
    assert ultra.images["cuda"] == "docker.io/vllm/vllm-openai:v0.22.0"
    # rocm: NO flashinfer anywhere; AITER on; mamba stays on the portable
    # triton backend (the only ROCm-viable one)
    rocm = cards.effective_args(ultra.args, "rocm")
    assert "enable_flashinfer_autotune" not in rocm
    assert rocm["mamba_backend"] == "triton" and rocm["mamba_ssm_cache_dtype"] == "float32"
    renv = cards.effective_args(cards.layered_env(
        "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"), "rocm")
    assert renv == {"VLLM_ROCM_USE_AITER": "1"}
    # geometry on clusterb's MI300A shape (4x128): NVFP4 fits ONE node,
    # BF16 (~1.1TB) becomes a 3-node Ray instance
    assert cards.fit_geometry(ultra.min_vram_gb, 4, 128)[:2] == (1, 4)
    bf16 = cards.find_card("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16")
    assert cards.fit_geometry(bf16.min_vram_gb, 4, 128)[:2] == (3, 4)
    # Super FP8 — the best AMD path — needs 2 MI300A
    sup = cards.find_card("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")
    assert cards.fit_geometry(sup.min_vram_gb, 4, 128)[:2] == (1, 2)
    # variant-specific cards beat the generic Nano match
    assert cards.find_card("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8").min_vram_gb == 30
    assert cards.find_card("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4").min_vram_gb == 18


def test_cards_match_filesystem_paths(tmp_path, monkeypatch):
    """A shared-FS checkout served BY PATH must hit the same card as the
    hf:// id (field: a by-path Maverick serve missed its card's geometry and
    context cap, ran single-node, and OOMed)."""
    from boxy import cards

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    d = tmp_path / "cfg" / "boxy" / "cards" / "models"
    d.mkdir(parents=True)
    (d / "acme-big-moe.toml").write_text(
        '[model]\nmatch = "acme/Big-MoE-400B*"\nengine = "vllm"\n'
        'gpus = 16\nmin_vram_gb = 814\n[model.args]\nmax_model_len = 8192\n')
    card = cards.find_card("/pscratch/team/models/acme/Big-MoE-400B-Instruct")
    assert card is not None and card.gpus == 16
    args, _ = cards.layered_args("/pscratch/team/models/acme/Big-MoE-400B-Instruct")
    assert args.get("max_model_len") == 8192
    assert cards.find_card("acme/Big-MoE-400B-Instruct").gpus == 16   # plain id unchanged
    assert cards.find_card("acme/Other-Model") is None                # no false hits


def test_match_keys_shapes():
    from boxy import cards

    assert cards.match_keys("meta-llama/X") == ["meta-llama/X"]
    assert cards.match_keys("hf://meta-llama/X") == ["meta-llama/X"]
    keys = cards.match_keys("/fs/models/meta-llama/X")
    assert keys[0] == "/fs/models/meta-llama/X"
    assert "meta-llama/X" in keys and "X" in keys
