"""One regression test per gap found in the feature-by-feature audit."""

import pytest

from boxy import deploy, envs, sky_export
from boxy.backends import get_backend
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location, Resources
from tests.conftest import EXAMPLES


def test_gap1_run_does_not_leak_dash_dash_separator(capsys):
    # `boxy run ... -- serve m1` must not put a literal "--" in the command.
    rc = main(
        [
            "run",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"),
            "--dryrun",
            "--",
            "serve", "m1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert " -- " not in out.split("### Running Command:")[1]


def test_gap3_sky_export_transport_uri_becomes_repo_id(cloud_gpu):
    box = Box(name="v", image="i", model="hf://Qwen/Qwen2.5-0.5B-Instruct")
    yaml_text = sky_export.to_sky_task(box, cloud_gpu)
    assert "vllm serve Qwen/Qwen2.5-0.5B-Instruct" in yaml_text
    assert "/models" not in yaml_text.split("run: |")[1]
    assert "fetched by the engine at task start" in yaml_text


def test_gap4_apptainer_honors_ro_mount_option(vllm_box, eldorado):
    mounts = [("/host/models", "/models", "ro")]
    cmd = get_backend("apptainer").build_command(vllm_box, eldorado, ["x"], {}, mounts, "rocm")
    i = cmd.index("--bind")
    assert cmd[i + 1] == "/host/models:/models:ro"


def test_gap5_llamacpp_engine_gets_no_vllm_env():
    env = envs.build_env({}, "rocm", offline=True, engine="llama.cpp")
    # vLLM *engine hygiene* vars must not leak into llama.cpp boxes
    # (VLLM_NO_USAGE_STATS stays: it belongs to the offline/telemetry set).
    for key in ("VLLM_DISABLE_COMPILE_CACHE", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_V1"):
        assert key not in env
    assert env["HF_HUB_OFFLINE"] == "1"          # offline set still applies
    assert env["OMP_NUM_THREADS"] == "1"         # common hygiene still applies
    vllm_env = envs.build_env({}, "rocm", offline=True, engine="vllm")
    assert vllm_env["VLLM_USE_V1"] == "1"        # vllm keeps its quirks


def test_gap6_generate_sky_warns_on_hpc_scheduler(capsys):
    rc = main(
        [
            "generate", "sky",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"),
        ]
    )
    assert rc == 0
    assert "boxy serves Slurm/Flux natively" in capsys.readouterr().err


def test_gap7_default_image_resolved_from_engine_and_accelerator(eldorado):
    box = Box(name="noimg", engine="vllm", model="m")  # no image
    d = deploy.plan_serve(box, eldorado, dryrun=True)
    assert d.box.image  # resolved
    assert "vllm" in d.box.image.lower() or "rocm" in d.box.image.lower()
    # GGUF on a ROCm location must get a GPU-capable llama.cpp image (the
    # upstream ghcr server image is CPU-only) with llama-server named
    # explicitly — it is on $PATH there, not the image ENTRYPOINT.
    llbox = Box(name="noimg2", engine="llama.cpp", model="m.gguf")
    d2 = deploy.plan_serve(llbox, eldorado, dryrun=True)
    assert "rocm" in d2.box.image
    assert d2.box.entrypoint == "llama-server"


def test_gap7b_missing_required_key_is_clean_error(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('[box]\nimage = "i"\n')  # no name
    with pytest.raises(ValueError, match="invalid \\[box\\] section"):
        Box.from_toml(bad)


def test_gap8_stop_and_list_commands(capsys):
    rc = main(["stop", "--box", str(EXAMPLES / "boxes" / "vllm.toml"), "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "stop vllm" in out
    rc = main(["list", "--dryrun"])
    assert rc == 0
    assert "label=boxy.box" in capsys.readouterr().out


def test_gap8b_container_label_present(vllm_box, hops):
    cmd = get_backend("podman").build_command(vllm_box, hops, ["x"], {}, [], "cuda")
    assert "--label=boxy.box=vllm" in cmd


@pytest.fixture
def cloud_gpu() -> Location:
    return Location(
        name="cloud-gpu",
        scheduler="none",
        accelerator="cuda",
        runtime="docker",
        resources=Resources(nodes=1, gpus_per_node=4, accelerator_type="H100"),
    )
