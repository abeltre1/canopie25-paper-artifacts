"""CharlieCloud backend (experimental) — golden argv for the no-daemon ch-run
flow: pull+convert(+fromhost) prepare steps and the ch-run exec line. Live
container execution is out of scope (no ch-run on CI)."""

import pytest

from boxy.backends import BACKENDS, get_backend
from boxy.box import Box
from boxy.location import Location


@pytest.fixture
def box():
    return Box(name="vllm", image="vllm/vllm-openai:latest", engine="vllm",
               model="hf://meta-llama/Llama-3.1-8B-Instruct", ports=[8000])


@pytest.fixture
def loc():
    return Location(name="cc", scheduler="slurm", accelerator="cuda", runtime="charliecloud")


def test_registered_in_backends():
    assert "charliecloud" in BACKENDS
    assert get_backend("charliecloud").name == "charliecloud"


def test_location_accepts_charliecloud_runtime():
    Location(name="x", runtime="charliecloud")   # must not raise (RUNTIMES updated)


def test_prepare_pull_convert_fromhost_for_cuda(box, loc):
    cmds = get_backend("charliecloud").prepare(box, loc, accelerator="cuda")
    assert cmds[0] == ["ch-image", "pull", "docker://vllm/vllm-openai:latest"]
    assert cmds[1] == ["ch-convert", "vllm/vllm-openai:latest", "vllm-cuda.dir"]
    assert cmds[2] == ["ch-fromhost", "--nvidia", "vllm-cuda.dir"]   # NVIDIA lib injection


def test_prepare_skips_fromhost_for_rocm(box):
    loc = Location(name="cc", accelerator="rocm", runtime="charliecloud")
    cmds = get_backend("charliecloud").prepare(box, loc, accelerator="rocm")
    assert all("fromhost" not in c[0] for c in cmds)               # no NVIDIA step
    assert cmds[0][:2] == ["ch-image", "pull"]


def test_build_command_is_ch_run_exec(box, loc):
    cmd = get_backend("charliecloud").build_command(
        box, loc, inner_cmd=["vllm", "serve", "m"], env={"K": "V"},
        mounts=[("/models", "/models", "")], accelerator="cuda")
    assert cmd[0] == "ch-run" and "--write" in cmd
    assert "-b" in cmd and "/models:/models" in cmd
    assert "--set-env=K=V" in cmd
    # image dir then the `--` separator then the inner command
    i = cmd.index("vllm-cuda.dir")
    assert cmd[i + 1] == "--" and cmd[i + 2:] == ["vllm", "serve", "m"]


def test_rocm_binds_device_nodes(box):
    loc = Location(name="cc", accelerator="rocm", runtime="charliecloud")
    cmd = get_backend("charliecloud").build_command(
        box, loc, inner_cmd=["vllm"], env={}, mounts=[], accelerator="rocm")
    assert "/dev/kfd" in cmd and "/dev/dri" in cmd


def test_serve_system_card_charliecloud_dryrun(monkeypatch, tmp_path, capsys):
    from boxy.cli import main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.1-8B-Instruct",
               "--system", "slurm-charliecloud", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0 and "#SBATCH" in out                            # slurm batch emitted
