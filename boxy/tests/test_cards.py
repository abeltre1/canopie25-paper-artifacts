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


def test_packaged_kimi_k3_card():
    # 2.8T MoE, MXFP4, ~1.56TB of weights: the biggest thing boxy ships a card
    # for. The interesting assertion is the geometry — the card hand-sets
    # nothing, yet on the recipe's own hardware (8x256GB MI325X) the solver must
    # reproduce the recipe's shape: ONE node, TP8.
    card = cards.find_card("hf://moonshotai/Kimi-K3")
    assert card and card.source == "packaged" and card.engine == "vllm"
    assert card.min_vram_gb == 1560
    assert card.args["tool_call_parser"] == "kimi_k3"
    assert card.args["reasoning_parser"] == "kimi_k3"
    assert card.args["trust_remote_code"] is True
    assert card.args["max_model_len"] == 262144          # native 1M would OOM the KV profile
    # HYPHEN, not underscore: the underscore spelling doesn't exist on Docker Hub
    # ("access denied" = no such repo; field: 32 ranks retried it, job died in 4s)
    assert card.images["rocm"] == "docker.io/vllm/vllm-openai-rocm:latest"
    # AITER stays OFF on rocm: with it on, K3's 96 heads leave NO valid TP on
    # gfx942 (24 heads invalid; 12 heads selects the CDNA4-only Gluon kernel,
    # which asserted AFTER the full 87-minute weight load — field, 8x MI300A)
    assert card.env["rocm"]["VLLM_ROCM_USE_AITER"] == "0"
    # CUDA gets the K3-specific tag (ships ray — :latest doesn't, and the
    # runtime pip-install dropped an H200 worker mid-formation; field, cronus-
    # class 2x8 nodes) and the recipe's long-load engine-ready timeout.
    assert card.images["cuda"] == "docker.io/vllm/vllm-openai:kimi-k3"
    assert card.env["cuda"]["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"
    assert cards.fit_geometry(card.min_vram_gb, 8, 256)[:2] == (1, 8)      # MI325X node
    # smaller parts spill to one Ray instance across full nodes, never refuse
    nodes, gpus, why = cards.fit_geometry(card.min_vram_gb, 4, 128, unified=True)   # MI300A
    assert nodes > 1 and gpus == 4 and "exceeds one node" in why
    assert cards.derive_gpu_memory_utilization(card.min_vram_gb, nodes * gpus, 128) is not None


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


# ---- unified-memory (APU) pools: derived gpu-memory-utilization ---------------------


def test_derive_gpu_memory_utilization_field_calibration():
    # THE field failure: 140GB of 70B weights over 4 MI300A ranks on a 128GB
    # pool -> 0.7, exactly the hand-tuned value that ended the silent OOM kills
    assert cards.derive_gpu_memory_utilization(140, 4, 128) == 0.7
    # more ranks -> smaller shards -> the host needs less -> a bigger claim
    assert cards.derive_gpu_memory_utilization(140, 8, 128) == 0.85
    # small models never claim beyond vLLM's own 0.9 default
    assert cards.derive_gpu_memory_utilization(8, 1, 1024) == 0.9
    # 2 ranks x 70GB shards on a 128GB pool: NO value both fits weights+KV and
    # leaves the host the stream headroom -> None (the fix is more ranks)
    assert cards.derive_gpu_memory_utilization(140, 2, 128) is None
    # unknown inputs -> None
    assert cards.derive_gpu_memory_utilization(0, 4, 128) is None
    assert cards.derive_gpu_memory_utilization(140, 0, 128) is None
    assert cards.derive_gpu_memory_utilization(140, 4, 0) is None


def test_apply_unified_solves_wider_and_derives_util():
    # discrete 4x128 metal packs the 70B onto 2 GPUs (see the solver tests); the
    # SAME node as a unified APU pool must spread to 4 (35GB shards that load)…
    a = _args("meta-llama/Llama-3.3-70B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 128, "x"), unified=True)
    assert a.gpus == 4 and a.nodes is None
    # …and derive the claim from the footprint, appended LAST so it wins over
    # the card's static 0.7 fallback (engine argparse is last-wins)
    assert a.args[-2:] == ["--gpu-memory-utilization", "0.7"]
    assert any("derived" in ln and "unified" in ln for ln in lines)
    # a user's own post-`--` value lands after the derived one and still wins
    b = _args("meta-llama/Llama-3.3-70B-Instruct")
    b.args = ["--gpu-memory-utilization", "0.6"]
    cards.apply_to_args(b, shape=(4, 128, "x"), unified=True)
    assert b.args[-2:] == ["--gpu-memory-utilization", "0.6"]
    # smaller model, same pool: the derived claim scales UP (8B leaves plenty)
    c = _args("hf://meta-llama/Llama-3.1-8B-Instruct")
    c.args = None
    cards.apply_to_args(c, shape=(4, 128, "x"), unified=True)
    assert c.args[-2:] == ["--gpu-memory-utilization", "0.79"]


def test_unified_model_reproduces_both_observed_mi300a_configurations():
    """The two configurations actually run on MI300A hardware, as a table. These
    are the only real datapoints the constants have, so they are the calibration:
    changing _UNIFIED_HOST_CAP_GB / _LOAD_FACTOR / _FLOOR must keep both."""
    POOL = 128
    # Llama-3.3-70B, 140GB: 0.9 was OOM-killed at 4 ranks, 0.7 served.
    assert cards.derive_gpu_memory_utilization(140, 4, POOL) == 0.7
    # ...and 2x70GB shards must stay INFEASIBLE, or the solver would pack the
    # 70B onto 2 ranks and reproduce the silent kill.
    assert cards.derive_gpu_memory_utilization(140, 2, POOL) is None
    assert cards.fit_geometry(140, 4, POOL, unified=True)[:2] == (1, 4)
    # Llama-4-Scout-17B-16E, 228GB: ran at tensor_parallel_size=4 on ONE node
    # (hpc-workflow/3-start-vllm-llama4-scout.sh). Both halves used to be wrong —
    # the solver sent it to 2 nodes and the derivation refused to produce any
    # value for 4 ranks.
    assert cards.fit_geometry(228, 4, POOL, unified=True)[:2] == (1, 4)
    scout = cards.derive_gpu_memory_utilization(228, 4, POOL)
    assert scout is not None and 0.5 < scout < 0.9


def test_unified_solver_and_derivation_never_disagree():
    """Whatever geometry the solver picks, the derivation must produce a number
    for it — otherwise the serve runs on the one architecture that needs the flag
    without it.

    The two used different memory models: the solver discounted the pool by a
    fixed fraction while the derivation subtracted an unbounded shard-proportional
    reserve. Measured against the old code this sweep finds one violation
    (nemotron-3-ultra-bf16 on a 96GB pool, 1 GPU/node), so it is a guard rather
    than the reproducer for the Scout failure — that one is the field-configuration
    test above, where the user pins the rank count the solver would not have
    chosen. Both come from the same root cause: two disagreeing models."""
    for card in cards.load_cards():
        if card.source != "packaged" or not card.min_vram_gb:
            continue
        for pool in (64, 96, 128, 192, 256):
            for width in (1, 2, 4, 8):
                nodes, gpus, _ = cards.fit_geometry(card.min_vram_gb, width, pool, unified=True)
                world = nodes * gpus
                assert cards.derive_gpu_memory_utilization(card.min_vram_gb, world, pool) is not None, (
                    f"{card.label}: solver chose {nodes}x{gpus} on a {pool}GB pool "
                    f"but the derivation refuses {world} ranks")


def test_unified_ranks_needed_answers_the_gpu_need():
    # the GPU need, derived from the footprint alone
    assert cards.unified_ranks_needed(24, 128) == 1      # 8B fits one rank
    assert cards.unified_ranks_needed(140, 128) == 4     # 70B needs 4 (not 2)
    assert cards.unified_ranks_needed(228, 128) == 4     # Scout needs 4
    assert cards.unified_ranks_needed(810, 128) == 16    # 405B-class
    # a smaller pool needs more ranks for the same model
    assert cards.unified_ranks_needed(140, 64) > cards.unified_ranks_needed(140, 128)
    assert cards.unified_ranks_needed(0, 128) is None
    assert cards.unified_ranks_needed(140, 0) is None
    # bounded search: an absurd model against a tiny limit gives up rather than loops
    assert cards.unified_ranks_needed(10_000, 8, limit=4) is None


def test_unified_reserve_is_bounded_at_both_ends():
    # the floor: a tiny shard still leaves the host the OS + container + tokenizer
    assert cards.unified_host_reserve_gb(1) == cards._UNIFIED_HOST_FLOOR_GB
    # the cap is the fix — an UNBOUNDED reserve grows past the pool and makes big
    # models look impossible, which is exactly what refused Scout on 4 ranks
    assert cards.unified_host_reserve_gb(10_000) == cards._UNIFIED_HOST_CAP_GB
    # in between it tracks the shard
    assert cards.unified_host_reserve_gb(35) == pytest.approx(38.5)
    # claimable is the pool minus that reserve, and never exceeds the pool
    assert cards.unified_claimable_gb(35, 128) == pytest.approx(89.5)
    assert cards.unified_claimable_gb(1, 128) < 128


def test_apply_unified_scout_lands_on_one_node_with_a_real_claim():
    """End-to-end for the field configuration: 228GB MoE on a 4x128GB MI300A
    node. Used to resolve to 2 nodes and no derived claim."""
    a = _args("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 128, "x"), unified=True)
    assert a.gpus == 4 and a.nodes is None
    assert a.args[-2] == "--gpu-memory-utilization"
    assert 0.5 < float(a.args[-1]) < 0.9
    assert any("derived" in ln and "unified" in ln for ln in lines)
    assert not any("NOT derived" in ln for ln in lines)


def test_apply_unified_warns_when_user_geometry_is_too_tight():
    # a power user pinning --gpus 2 on a unified pool: 70GB shards leave no
    # feasible claim — boxy SAYS so instead of emitting a number that can't work
    a = _args("meta-llama/Llama-3.3-70B-Instruct", gpus=2)
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 128, "x"), unified=True)
    assert any("NOT derived" in ln and "spread wider" in ln for ln in lines)
    # only the card's static fallback remains — no derived pair was appended
    assert (a.args or []).count("--gpu-memory-utilization") == 1


def test_apply_unified_skips_non_vllm_engines():
    # llama-server exits 2 on --gpu-memory-utilization — never derive it there
    a = _args("hf://meta-llama/Llama-3.1-8B-Instruct", engine="llama.cpp")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 128, "x"), unified=True)
    assert "--gpu-memory-utilization" not in (a.args or [])
    assert not any("gpu-memory-utilization" in ln for ln in lines)


def test_system_unified_memory_from_user_card(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "systems"
    d.mkdir(parents=True)
    (d / "clusterc.toml").write_text(
        '[location]\nname = "clusterc"\nscheduler = "flux"\n'
        '[location.resources]\ngpus_per_node = 4\ngpu_vram_gb = 128\nunified_memory = true\n')
    assert cards.system_unified_memory("clusterc") is True
    assert cards.system_unified_memory("no-such-cluster") is False
    # the packaged MI300A example declares it too
    assert cards.system_unified_memory("flux-rocm") is True


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
    card = cards.find_card("/scratch/team/models/acme/Big-MoE-400B-Instruct")
    assert card is not None and card.gpus == 16
    args, _ = cards.layered_args("/scratch/team/models/acme/Big-MoE-400B-Instruct")
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


# ---- bare HuggingFace ids: say what to type, never guess -----------------------------


@pytest.mark.parametrize("model", [
    "thinkingmachines/Inkling",
    "meta-llama/Llama-3.3-70B-Instruct",     # version dots must not read as a suffix
    "Qwen/Qwen2.5-72B-Instruct",
])
def test_bare_repo_id_error_names_the_exact_command(model):
    """A bare id stays a local path on purpose (same command, same meaning, every
    machine). The error must then name the command that DOES work — the old text
    said 'hf://<org>/Inkling', making the reader substitute an org they had just
    typed."""
    from boxy import resolve

    with pytest.raises(RuntimeError) as e:
        resolve._classify_model(model, require_exists=True)
    msg = str(e.value)
    assert f"hf://{model}" in msg
    assert "<org>" not in msg


@pytest.mark.parametrize("model,is_id", [
    ("thinkingmachines/Inkling", True),
    ("meta-llama/Llama-3.3-70B-Instruct", True),   # dots are versions, not extensions
    ("./models/x", False),
    ("/abs/path", False),
    ("~/models/x", False),
    ("a/b/c", False),                              # two slashes = a path
    ("dir/model.gguf", False),                     # weight suffix = a path
    ("models/llama.safetensors", False),
    ("models/tiny-demo.ggu", False),               # a TYPO'd suffix is still a path
    ("Qwen/Qwen2.5-72B", True),                    # version dot, not an extension
    ("justaname", False),
])
def test_repo_id_shape_excludes_paths(model, is_id):
    from boxy import resolve

    assert resolve._looks_like_repo_id(model) is is_id


def test_non_repo_id_keeps_the_generic_hint():
    from boxy import resolve

    with pytest.raises(RuntimeError, match=r"did you mean ollama://"):
        resolve._classify_model("some-random-thing", require_exists=True)


def test_generate_card_suggests_a_command_that_works(tmp_path, monkeypatch, capsys):
    """The reported bug: `generate card <id>` printed `boxy serve <id>`, which
    serve then rejects — after resolving that very card and probing the cluster."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from boxy import cardgen
    from boxy.cli import main

    monkeypatch.setattr(cardgen, "generate", lambda repo, **kw: (
        '[model]\nmatch = "org/Thing*"\nengine = "vllm"\ngpus = 1\n', "vllm", []))
    assert main(["generate", "card", "org/Thing"]) == 0
    out = capsys.readouterr().out
    assert "boxy serve hf://org/Thing" in out
    assert "boxy serve org/Thing" not in out


# ---- derived context window (max_model_len) ------------------------------------------
#
# FIELD (Kimi-K3): the 1M-native-window model shipped with a static 262144 cap
# and was hand-served at --max-model-len 131072 — on hardware whose per-rank
# arithmetic supports the full million. When a card knows its KV bytes/token
# and the system card knows the node, the largest context that PROVABLY fits
# is derived and the static cap replaced.

CTX_CARD = (
    '[model]\n'
    'match = "acme/Ctx-Aware-24B*"\n'
    'engine = "vllm"\n'
    'min_vram_gb = 24\n'
    'native_ctx = 1000000\n'
    'kv_bytes_per_token = 131072\n'
    '[model.args]\n'
    'max_model_len = 8192\n'
)


def _ctx_card(tmp_path, monkeypatch, text=CTX_CARD):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    d = tmp_path / "boxy" / "cards" / "models"
    d.mkdir(parents=True, exist_ok=True)
    (d / "ctx-aware.toml").write_text(text)


def test_parse_card_reads_native_ctx_and_kv_bytes(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    card = cards.resolve_model_card("acme/Ctx-Aware-24B-Instruct")
    assert card.native_ctx == 1000000
    assert card.kv_bytes_per_token == 131072.0
    # cards without the fields default to 0 (derivation stays off)
    legacy = cards.resolve_model_card("meta-llama/Llama-3.1-8B-Instruct")
    assert legacy.native_ctx == 0 and legacy.kv_bytes_per_token == 0.0


def test_derive_max_model_len_discrete_arithmetic():
    # hand-checkable: 80GB part at vLLM's 0.9 default -> 72GB claim; minus a
    # 24GB shard and the 8GB reserve -> 40GB KV budget; at 131072 B/token
    # (Llama-8B-class GQA) that is 305175 tokens, floored to a 1024 multiple.
    tokens, why = cards.derive_max_model_len(131072, 1000000, 24, 1, 1, 80)
    assert tokens == 334848
    assert "44GB" in why
    # PP=2 splits the layers -> half the per-rank cost -> double the tokens
    tokens2, _ = cards.derive_max_model_len(131072, 1000000, 24, 1, 2, 80)
    assert tokens2 == 670720
    # native window is the hard ceiling — and the derivation says so
    capped, why3 = cards.derive_max_model_len(131072, 65536, 24, 1, 1, 80)
    assert capped == 65536
    # an explicit utilization changes the claim (0.5 * 80 = 40GB -> 12GB budget)
    lower, _ = cards.derive_max_model_len(131072, 1000000, 24, 1, 1, 80, util=0.5)
    assert lower == 91136
    # fp8 kv cache halves the per-token cost -> double the tokens
    fp8, _ = cards.derive_max_model_len(131072, 1000000, 24, 1, 1, 80, kv_dtype_factor=0.5)
    assert fp8 == 670720


def test_derive_max_model_len_declines():
    # unknown inputs
    assert cards.derive_max_model_len(0, 1000000, 24, 1, 1, 80)[0] is None
    assert cards.derive_max_model_len(131072, 0, 24, 1, 1, 80)[0] is None
    assert cards.derive_max_model_len(131072, 1000000, 24, 1, 1, 0)[0] is None
    # weights + reserve eat the whole claim -> no budget
    tokens, why = cards.derive_max_model_len(131072, 1000000, 70, 1, 1, 80)
    assert tokens is None and "no KV budget" in why
    # a budget that fits fewer than the floor is not worth serving
    tokens, why = cards.derive_max_model_len(13_000_000, 1000000, 24, 1, 1, 80)
    assert tokens is None and "fewer than" in why


def test_derive_max_model_len_unified_agrees_with_util_derivation():
    # the util derivation's calibration point: 140GB over 4 ranks on a 128GB
    # pool claims 0.7 -> the ctx budget must be built on that SAME claim
    tokens, why = cards.derive_max_model_len(131072, 1000000, 140, 4, 1, 128, unified=True)
    assert tokens == 385024
    # ...and where the util derivation refuses (2x70GB shards), the ctx
    # derivation must refuse too — deriving a context on top of a claim that
    # gets the rank OOM-killed would rebuild the field failure
    tokens, why = cards.derive_max_model_len(131072, 1000000, 140, 2, 1, 128, unified=True)
    assert tokens is None and "does not fit" in why


def test_apply_derives_context_and_replaces_static_cap(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 80, "x"))
    # exactly ONE --max-model-len: the derived value, the static 8192 stripped
    assert a.args.count("--max-model-len") == 1
    assert a.args[a.args.index("--max-model-len") + 1] == "334848"
    assert "8192" not in a.args
    assert any(ln.startswith("max-model-len: 334848 (derived:") for ln in lines)
    # the card's only static arg was the cap, so no engine-args line remains
    assert not any(ln.startswith("engine args:") for ln in lines)


def test_apply_context_pair_precedes_derived_gpu_mem_pair(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    cards.apply_to_args(a, shape=(4, 128, "x"), unified=True)
    # the long-standing contract: the derived gpu-mem pair stays LAST...
    assert a.args[-2:] == ["--gpu-memory-utilization", "0.79"]
    # ...and the derived ctx pair sits immediately before it
    assert a.args[-4:-2] == ["--max-model-len", "561152"]


def test_apply_context_user_max_model_len_still_wins(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = ["--max-model-len", "4096"]
    cards.apply_to_args(a, shape=(4, 80, "x"))
    # derived pair emitted, user pair appended after -> engine last-wins
    assert a.args[-2:] == ["--max-model-len", "4096"]
    assert a.args[-4:-2] == ["--max-model-len", "334848"]


def test_apply_context_honors_fp8_kv_cache_and_user_override(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch, CTX_CARD + 'kv_cache_dtype = "fp8"\n')
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    cards.apply_to_args(a, shape=(4, 80, "x"))
    assert a.args[a.args.index("--max-model-len") + 1] == "670720"  # 2x: fp8 kv
    # the user flipping the card's fp8 BACK to auto must restore the bf16 cost
    # (missing this would derive a context twice what the cache can hold)
    b = _args("acme/Ctx-Aware-24B-Instruct")
    b.args = ["--kv-cache-dtype", "auto"]
    cards.apply_to_args(b, shape=(4, 80, "x"))
    assert b.args[b.args.index("--max-model-len") + 1] == "334848"


def test_apply_context_honors_user_pipeline_parallelism(tmp_path, monkeypatch):
    # FIELD: the K3 serve ran user-pinned TP8xPP4 on 8 nodes where the
    # geometric default assumes PP=nodes — the KV-per-rank math must follow
    # the pipeline size the engine will actually use
    _ctx_card(tmp_path, monkeypatch)
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = ["--pipeline-parallel-size", "2"]
    cards.apply_to_args(a, shape=(4, 80, "x"))
    assert a.args[a.args.index("--max-model-len") + 1] == "670720"  # PP=2 halves cost


def test_apply_context_declines_without_real_shape(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    # no shape at all -> static cap stands, remedy names the system card
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=None)
    assert a.args == ["--max-model-len", "8192"]
    assert any("node VRAM unknown" in ln for ln in lines)
    # VRAM assumed from the GPU-type table is NOT real hardware: an a100 could
    # be the 40GB variant and a 2x context overshoot OOMs the KV profile
    b = _args("acme/Ctx-Aware-24B-Instruct")
    b.args = None
    lines = cards.apply_to_args(
        b, shape=(4, 80, "cluster probe inventory (VRAM assumed from the GPU type)"))
    assert b.args == ["--max-model-len", "8192"]
    assert any("node VRAM unknown" in ln for ln in lines)


def test_apply_context_names_missing_kv_field(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch, (
        '[model]\nmatch = "acme/Ctx-Aware-24B*"\nengine = "vllm"\n'
        'min_vram_gb = 24\nnative_ctx = 1000000\n'
        '[model.args]\nmax_model_len = 8192\n'))
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 80, "x"))
    assert a.args == ["--max-model-len", "8192"]
    assert any("regenerate the card" in ln for ln in lines)


def test_apply_context_silent_for_cards_without_the_fields():
    # every pre-existing card: no ctx fields -> not one word about max-model-len
    # beyond the engine-args line; zero decision-line churn
    a = _args("meta-llama/Llama-3.3-70B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 140, "x"))
    assert not any(ln.startswith("max-model-len:") for ln in lines)


def test_apply_context_skips_non_vllm(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch)
    a = _args("acme/Ctx-Aware-24B-Instruct", engine="llama.cpp")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 80, "x"))
    assert not any(ln.startswith("max-model-len:") for ln in lines)


def test_apply_context_full_native_window_says_so(tmp_path, monkeypatch):
    _ctx_card(tmp_path, monkeypatch, CTX_CARD.replace(
        "native_ctx = 1000000", "native_ctx = 65536"))
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args = None
    lines = cards.apply_to_args(a, shape=(4, 80, "x"))
    assert a.args[a.args.index("--max-model-len") + 1] == "65536"
    assert any("FULL native window fits" in ln for ln in lines)


def test_kimi_k3_card_carries_kv_fields():
    # the load_cards silent-drop tripwire: a malformed packaged card VANISHES
    # (load_cards skips it), so the new fields must be pinned here or a typo
    # would quietly remove Kimi-K3 from the registry
    card = cards.find_card("hf://moonshotai/Kimi-K3")
    assert card is not None
    assert card.native_ctx == 1048576
    assert card.kv_bytes_per_token == 27648.0   # (512+64) x 2B x 24 MLA layers


def test_kimi_k3_derives_past_static_cap_on_fat_metal():
    # the recipe's own MI325X shape (1 node x 8 x 256GB): the static 262144
    # fallback is REPLACED by the derived window — 230.4GB claim − 195GB shard
    # − 4GB reserve = 31.4GB of KV at ~27.6KB/token = ~1.14M tokens, clamped
    # at the model's FULL 1M native window. Zero flags.
    a = _args("moonshotai/Kimi-K3")
    a.args = None
    lines = cards.apply_to_args(a, shape=(8, 256, "x"))
    assert a.args.count("--max-model-len") == 1
    derived = int(a.args[a.args.index("--max-model-len") + 1])
    assert derived == 1048576
    assert "262144" not in a.args
    assert any(ln.startswith("max-model-len: 1048576 (derived:") for ln in lines)
    assert any("FULL native window fits" in ln for ln in lines)


# vLLM's own measured ground truth from the fielded 8-node MI300A serve
# (TP8xPP4, 128GB unified pools, 2026-08-22): 'Available KV cache memory:
# 32.97 GiB' per rank, 'GPU KV cache size: 4,588,273 tokens'. This is the
# calibration that pins _CTX_ACT_RESERVE_GB (tune the constant, not the test).
MEASURED_K3_KV_TOKENS = 4_588_273


@pytest.mark.skipif(MEASURED_K3_KV_TOKENS is None,
                    reason="awaiting the field-measured K3 'GPU KV cache size' line")
def test_kimi_k3_context_calibration():
    """The estimator is PROVEN when the derivation's implied per-deployment KV
    capacity lands within 20% of what vLLM itself measured at the exact fielded
    settings (8 nodes x 4 MI300A, TP8xPP4, unified 128GB pools)."""
    card = cards.find_card("hf://moonshotai/Kimi-K3")
    # native_ctx sentinel far above capacity so the clamp cannot mask the error
    tokens, _ = cards.derive_max_model_len(
        card.kv_bytes_per_token, 10**9, card.min_vram_gb, 32, 4, 128, unified=True)
    assert tokens is not None
    assert abs(tokens - MEASURED_K3_KV_TOKENS) / MEASURED_K3_KV_TOKENS < 0.20


# ---- --context: the window as a DEMAND the geometry must meet (Mode B) ---------------


def test_fit_geometry_ctx_kwargs_default_inert():
    # without the kwargs every path is byte-identical (the parity + calibration
    # contracts above already sweep this; here the equality is explicit)
    assert cards.fit_geometry(140, 4, 80) == cards.fit_geometry(
        140, 4, 80, ctx_tokens=0, ctx_kv_bytes_per_token=0.0)
    assert cards.fit_geometry(140, 4, 128, unified=True) == cards.fit_geometry(
        140, 4, 128, unified=True, ctx_tokens=0, ctx_kv_bytes_per_token=0.0)


def test_parse_context_request():
    assert cards.parse_context_request("4096", 0) == 4096
    assert cards.parse_context_request("256k", 0) == 262144
    assert cards.parse_context_request("1m", 0) == 1048576
    assert cards.parse_context_request("full", 131072) == 131072
    with pytest.raises(ValueError, match="native window"):
        cards.parse_context_request("full", 0)
    with pytest.raises(ValueError, match="token count"):
        cards.parse_context_request("lots", 131072)
    with pytest.raises(ValueError, match="positive"):
        cards.parse_context_request("-5", 131072)


def test_fit_geometry_grows_nodes_for_ctx_demand():
    # 140GB of weights on 4x80 nodes fits ONE node by weight — but a 1M-token
    # window of GQA KV (131072 B/token) needs PP to split the layers: at PP=3
    # each rank holds ~11.7GB of shard + ~43.7GB of KV inside a 72GB claim
    nodes, gpus, why = cards.fit_geometry(
        140, 4, 80, ctx_tokens=1_000_000, ctx_kv_bytes_per_token=131072.0)
    assert (nodes, gpus) == (3, 4)
    assert "1000000-token context" in why
    # the same call without the demand stays single-node (the inert contract)
    assert cards.fit_geometry(140, 4, 80)[:2] == (1, 4)


def test_fit_geometry_ctx_widens_within_a_node_before_spilling():
    # a 24GB model asking 400K tokens: 1 GPU gives a 44GB budget (~335K), but
    # widening to 2 GPUs shrinks the shard and frees enough — no second node
    nodes, gpus, why = cards.fit_geometry(
        24, 4, 80, ctx_tokens=400_000, ctx_kv_bytes_per_token=131072.0)
    assert (nodes, gpus) == (1, 2)
    assert "widened" in why
    assert cards.fit_geometry(24, 4, 80)[:2] == (1, 1)


def test_ctx_solver_and_derivation_never_disagree():
    """The Mode B analog of the unified sweep above: whatever geometry the
    solver picks FOR a context demand, the runtime derivation at that exact
    geometry must honor the demand — otherwise the serve would print a
    'sized to fit' line and then emit a smaller window."""
    for card in cards.load_cards():
        if card.source != "packaged" or not card.min_vram_gb or not card.kv_bytes_per_token:
            continue
        demand = card.native_ctx
        for pool in (64, 96, 128, 192, 256):
            for width in (1, 2, 4, 8):
                for uni in (False, True):
                    nodes, gpus, why = cards.fit_geometry(
                        card.min_vram_gb, width, pool, unified=uni,
                        ctx_tokens=demand, ctx_kv_bytes_per_token=card.kv_bytes_per_token)
                    if "does not fit" in why:
                        continue
                    tokens, dwhy = cards.derive_max_model_len(
                        card.kv_bytes_per_token, demand, card.min_vram_gb,
                        nodes * gpus, nodes, pool, unified=uni)
                    assert tokens is not None and tokens >= demand, (
                        f"{card.label}: solver chose {nodes}x{gpus} on a {pool}GB "
                        f"{'unified ' if uni else ''}pool for {demand} tokens, but the "
                        f"derivation only honors {tokens} ({dwhy})")


def test_serve_context_full_grows_geometry_end_to_end(tmp_path, monkeypatch, capsys):
    _ctx_card(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOXY_GPUS_PER_NODE", "4")
    monkeypatch.setenv("BOXY_GPU_VRAM_GB", "80")
    rc = main(["serve", "acme/Ctx-Aware-24B-Instruct",
               "--scheduler", "slurm", "--context", "full", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: context: 1000000 tokens (--context full)" in out
    assert "auto: nodes: 3" in out                       # grown for the window
    assert "--max-model-len 1000000" in out.replace("=", " ")
    assert "auto: max-model-len: 1000000 (--context full: verified" in out


def test_serve_context_infeasible_is_a_clean_error(tmp_path, monkeypatch, capsys):
    # absurd per-token cost: no node count holds the window — refuse UP FRONT
    # (a submitted job would burn its queue wait and OOM the KV profile)
    _ctx_card(tmp_path, monkeypatch, CTX_CARD.replace(
        "kv_bytes_per_token = 131072", "kv_bytes_per_token = 1310720000"))
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOXY_GPUS_PER_NODE", "4")
    monkeypatch.setenv("BOXY_GPU_VRAM_GB", "80")
    rc = main(["serve", "acme/Ctx-Aware-24B-Instruct",
               "--scheduler", "slurm", "--context", "full", "--dryrun"])
    cap = capsys.readouterr()
    assert rc == 1
    assert "boxy: error:" in cap.err and "does not fit" in cap.err


def test_context_with_pinned_geometry_verifies_only(tmp_path, monkeypatch, capsys):
    # explicit --gpus pins the world; boxy must not silently under-serve the
    # request — it warns, emits NO pair, and the static cap stands
    _ctx_card(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOXY_GPUS_PER_NODE", "4")
    monkeypatch.setenv("BOXY_GPU_VRAM_GB", "80")
    rc = main(["serve", "acme/Ctx-Aware-24B-Instruct", "--gpus", "1",
               "--scheduler", "slurm", "--context", "400k", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "context: NOT honored" in out
    assert "--max-model-len 409600" not in out.replace("=", " ")
    assert "--max-model-len 8192" in out.replace("=", " ")   # static cap kept


def test_context_requires_kv_and_a_real_shape(tmp_path, monkeypatch):
    # no kv_bytes_per_token -> the remedy names the card fix
    _ctx_card(tmp_path, monkeypatch, (
        '[model]\nmatch = "acme/Ctx-Aware-24B*"\nengine = "vllm"\n'
        'min_vram_gb = 24\nnative_ctx = 1000000\n'
        '[model.args]\nmax_model_len = 8192\n'))
    a = _args("acme/Ctx-Aware-24B-Instruct")
    a.args, a.context = None, "full"
    with pytest.raises(ValueError, match="KV bytes/token"):
        cards.apply_to_args(a, shape=(4, 80, "x"))
    # kv known but the shape is missing or assumed -> the system-card remedy
    _ctx_card(tmp_path, monkeypatch)
    b = _args("acme/Ctx-Aware-24B-Instruct")
    b.args, b.context = None, "full"
    with pytest.raises(ValueError, match="VRAM is unknown"):
        cards.apply_to_args(b, shape=None)
    c = _args("acme/Ctx-Aware-24B-Instruct")
    c.args, c.context = None, "full"
    with pytest.raises(ValueError, match="VRAM is unknown"):
        cards.apply_to_args(c, shape=(4, 80, "probe (VRAM assumed from the GPU type)"))


def test_context_refuses_replicas(tmp_path, monkeypatch, capsys):
    _ctx_card(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "acme/Ctx-Aware-24B-Instruct", "--scheduler", "slurm",
               "--context", "full", "--replicas", "2", "--dryrun"])
    assert rc == 2
    assert "--context sizes ONE" in capsys.readouterr().err


def test_context_local_path_is_rejected(capsys):
    rc = main(["serve", "hf://meta-llama/Llama-3.1-8B-Instruct",
               "--context", "full", "--here", "--dryrun"])
    assert rc == 2
    assert "--context" in capsys.readouterr().err


def test_context_overshoot_clamps_to_native(tmp_path, monkeypatch, capsys):
    _ctx_card(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("BOXY_GPUS_PER_NODE", "4")
    monkeypatch.setenv("BOXY_GPU_VRAM_GB", "80")
    rc = main(["serve", "acme/Ctx-Aware-24B-Instruct",
               "--scheduler", "slurm", "--context", "9m", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "clamped" in out
    assert "--max-model-len 1000000" in out.replace("=", " ")


def test_a_boxy_pulled_model_served_BY_PATH_still_finds_its_card():
    """FIELD: `boxy pull` stages a model into a directory named by its SLUG —
    hf://moonshotai/Kimi-K3 -> .../models/moonshotai-kimi-k3 — and serving that
    path by hand matched NO card, silently dropping the pinned image, the engine
    args, the geometry and the derived context. Exactly the failure the org/name
    path handling was added for, missed for boxy's OWN layout."""
    for path in ("/tscratch/u/boxy/models/moonshotai-kimi-k3",
                 "/pscratch/u/boxy/models/moonshotai-kimi-k3",
                 "~/boxy/models/moonshotai-kimi-k3"):
        card = cards.find_card(path)
        assert card is not None and card.card_name == "kimi-k3", path
        assert card.images.get("cuda")          # the pin really rides along
        assert card.native_ctx == 1048576       # ...and so does the context data
    # an org whose NAME contains dashes resolves too (the split is ambiguous,
    # so every dash is offered and the card's glob decides)
    llama = cards.find_card("/scratch/u/boxy/models/meta-llama-llama-3.1-8b-instruct")
    assert llama is not None and "llama-3.1-8b" in llama.card_name
    # and an unrelated directory still matches nothing
    assert cards.find_card("/data/some-unrelated-dir") is None


def test_by_path_serve_gets_the_cards_geometry_and_context():
    a = _args("/tscratch/u/boxy/models/moonshotai-kimi-k3")
    a.args = None
    lines = cards.apply_to_args(a, shape=(8, 256, "x"))
    assert a.gpus == 8                                   # solved, not guessed
    assert "--max-model-len" in a.args                   # context derived
    assert any("image:" in ln and "kimi-k3" in ln for ln in lines)
