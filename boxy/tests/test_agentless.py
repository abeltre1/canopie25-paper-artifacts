"""Agentless (zero-install) execution: boxy emits a self-contained batch script
(podman + a shared-FS endpoint write) so the compute node running the workload
needs NO boxy/Python/RamaLama — only a scheduler + container runtime + shared FS
(SPEC §8c). Two boundaries: the model must be pre-staged, and the accelerator/
image must be pinned (hardware can't be detected off-node)."""

import pytest

from boxy import deploy
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location


@pytest.fixture
def staged_gguf(tmp_path):
    m = tmp_path / "llama.q4.gguf"
    m.write_bytes(b"GGUF")
    return m


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    return tmp_path / "jobs"


def _box(model, image="", engine="llama.cpp"):
    return Box(name="boxy-al", model=str(model), engine=engine, image=image)


def _loc(accelerator="cuda", runtime="podman"):
    return Location(name="hops", scheduler="slurm", accelerator=accelerator, runtime=runtime)


# ---- the rendered script is self-contained (no boxy on the node) --------------


def test_render_script_is_boxy_free(staged_gguf, tmp_path):
    script = deploy.render_agentless_script(
        _box(staged_gguf), _loc(), "slurm", "boxy-al",
        str(tmp_path / "boxy-al.endpoint.json"), str(tmp_path / "boxy-al-%j.log"),
        site_args=["--account=fy260064"], port=8090)
    assert "boxy serve" not in script and "python -m boxy" not in script   # NO boxy anywhere
    assert "podman run" in script and "ghcr.io/ggml-org/llama.cpp:server-cuda" in script
    assert "#SBATCH --job-name=boxy-al" in script and "#SBATCH --account=fy260064" in script
    assert "--device nvidia.com/gpu=all" in script                          # cuda pinned -> GPU args
    assert 'cat > "${_EP}.tmp"' in script and "EOF_BOXY_EP" in script        # bash endpoint write
    assert '"host": "${_H}"' in script and '"port": 8090' in script          # $(hostname):port


def test_render_rejects_transport_uri(tmp_path):
    with pytest.raises(deploy.AgentlessError, match="PRE-STAGED"):
        deploy.render_agentless_script(_box("hf://org/model.gguf"), _loc(), "slurm", "x",
                                       str(tmp_path / "x.json"), str(tmp_path / "x.log"), [])


def test_render_rejects_unpinned_accelerator(staged_gguf, tmp_path):
    with pytest.raises(deploy.AgentlessError, match="pin --accelerator"):
        deploy.render_agentless_script(_box(staged_gguf), _loc(accelerator=""), "slurm", "x",
                                       str(tmp_path / "x.json"), str(tmp_path / "x.log"), [])


def test_render_flux_uses_flux_directives(staged_gguf, tmp_path):
    script = deploy.render_agentless_script(
        _box(staged_gguf), Location(name="e", scheduler="flux", accelerator="rocm", runtime="podman"),
        "flux", "boxy-al", str(tmp_path / "e.json"), str(tmp_path / "e.log"), [], port=8090)
    assert "# flux:" in script and "podman run" in script and "boxy serve" not in script


# ---- generate + serve wiring --------------------------------------------------


def test_generate_slurm_emits_agentless_script(staged_gguf, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    box = tmp_path / "box.toml"
    box.write_text(f'[box]\nname = "boxy-al"\nmodel = "{staged_gguf}"\nengine = "llama.cpp"\n')
    loc = tmp_path / "loc.toml"
    loc.write_text('[location]\nname = "hops"\nscheduler = "slurm"\nruntime = "podman"\n')
    rc = main(["generate", "slurm", "--box", str(box), "--location", str(loc), "--accelerator", "cuda"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "podman run" in out and "boxy serve" not in out and "#SBATCH" in out


def test_serve_agentless_dryrun_is_boxy_free(staged_gguf, jobs_dir, capsys):
    rc = main(["serve", str(staged_gguf), "--scheduler", "slurm", "--gpus", "1",
               "--agentless", "--accelerator", "cuda", "--account", "fy260064", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Agentless (no boxy on the compute node)" in out
    assert "podman run" in out and "boxy serve --foreground" not in out


def test_serve_agentless_refuses_transport_uri(jobs_dir, capsys):
    rc = main(["serve", "hf://org/model.gguf", "--scheduler", "slurm", "--gpus", "1",
               "--agentless", "--accelerator", "cuda", "--dryrun"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "PRE-STAGED" in err and "RamaLama" in err
