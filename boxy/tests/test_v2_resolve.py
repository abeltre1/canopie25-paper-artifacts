"""v2 model-first auto-resolution tests (the RamaLama-parity front door),
including the design-panel fixes: hip->rocm normalization, syntax-based model
classification, login-node guard, gpus-require-scheduler, free-port scan."""

import socket

import pytest

from boxy import resolve

pytestmark = []


@pytest.fixture
def cpu_host(monkeypatch):
    """Pin detection to a clean no-GPU, no-scheduler host."""
    monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "none")
    monkeypatch.setattr(resolve.shutil, "which", lambda name: None)


def test_gguf_infers_llamacpp_anywhere():
    engine, why = resolve.infer_engine("hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/q4.gguf", "none")
    assert engine == "llama.cpp" and "GGUF" in why
    engine, _ = resolve.infer_engine("/shared/models/llama.Q4_K_M.gguf", "cuda")
    assert engine == "llama.cpp"


def test_ollama_models_are_gguf():
    engine, why = resolve.infer_engine("ollama://granite3-moe", "none")
    assert engine == "llama.cpp" and "ollama" in why.lower()


def test_safetensors_needs_gpu_for_vllm():
    engine, why = resolve.infer_engine("hf://Qwen/Qwen2.5-0.5B-Instruct", "cuda")
    assert engine == "vllm" and "cuda" in why
    with pytest.raises(RuntimeError, match="GGUF"):
        resolve.infer_engine("hf://Qwen/Qwen2.5-0.5B-Instruct", "none")
    # a scheduler job request (--gpus N) counts as GPU-present
    engine, why = resolve.infer_engine("hf://Qwen/Qwen2.5-0.5B-Instruct", "none", gpus=4)
    assert engine == "vllm" and "--gpus" in why


def test_hip_and_cann_normalized_at_the_seam(monkeypatch):
    """ramalama's get_accel() says 'hip'/'cann'; boxy speaks 'rocm'/'ascend'.
    Without normalization every v2 command dies on a ROCm node."""
    import ramalama.common

    from boxy import ramalama_shim

    monkeypatch.setattr(ramalama.common, "get_accel", lambda: "hip")
    assert ramalama_shim.detect_accel() == "rocm"
    monkeypatch.setattr(ramalama.common, "get_accel", lambda: "cann")
    assert ramalama_shim.detect_accel() == "ascend"


def test_inside_allocation_runs_direct(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    sched, why = resolve.detect_scheduler_context()
    assert sched == "none" and "inside Slurm allocation" in why


def test_login_node_guard_refuses_without_here(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("FLUX_ENCLOSING_ID", raising=False)
    monkeypatch.delenv("FLUX_JOB_ID", raising=False)
    monkeypatch.setattr(resolve.shutil, "which", lambda name: "/usr/bin/srun" if name == "srun" else None)
    with pytest.raises(RuntimeError, match="login node"):
        resolve.detect_scheduler_context()
    sched, why = resolve.detect_scheduler_context(here=True)
    assert sched == "none" and "--here" in why


def test_bare_names_are_never_guessed(cpu_host):
    with pytest.raises(RuntimeError, match="ollama://granite3-moe"):
        resolve.resolve("granite3-moe", runtime="docker")


def test_gpus_and_nodes_require_scheduler(cpu_host):
    with pytest.raises(RuntimeError, match="--scheduler"):
        resolve.resolve("m.gguf", runtime="docker", gpus=4, require_exists=False)
    with pytest.raises(RuntimeError, match="--scheduler"):
        resolve.resolve("m.gguf", runtime="docker", nodes=2, require_exists=False)


def test_gpu_job_from_cpu_login_node_needs_explicit_accelerator(cpu_host):
    with pytest.raises(RuntimeError, match="--accelerator"):
        resolve.resolve("hf://org/repo", runtime="podman", scheduler="slurm", gpus=4)


def test_explicit_accelerator_pins_the_submission(cpu_host):
    r = resolve.resolve("hf://org/repo", runtime="podman", scheduler="slurm", gpus=4,
                        accelerator="rocm")
    assert r.location.accelerator == "rocm"
    assert r.box.engine == "vllm"
    assert r.location.resources.gpus_per_node == 4


def test_resolution_end_to_end_no_files(cpu_host, tmp_path):
    model = tmp_path / "tiny.q4.gguf"
    model.write_bytes(b"GGUF")
    r = resolve.resolve(str(model), runtime="docker")
    assert r.box.engine == "llama.cpp"
    assert r.box.name.startswith("boxy-tiny")
    assert 8090 <= r.box.ports[0] < 8154  # engine default, advanced past busy ports
    assert r.location.runtime == "docker"
    assert r.location.scheduler == "none"
    assert any("accelerator:" in d for d in r.decisions)
    assert any("image:" in d for d in r.decisions)  # every choice is explained
    assert any("model:" in d for d in r.decisions)


def test_resolution_overrides_win(cpu_host):
    r = resolve.resolve("m.gguf", engine="llama.cpp", runtime="podman",
                        scheduler="slurm", image="custom:1", port=9999, gpus=4, nodes=2,
                        name="myname", accelerator="cuda", require_exists=False)
    assert r.box.image == "custom:1" and r.box.ports == [9999] and r.box.name == "myname"
    assert r.location.scheduler == "slurm"
    assert r.location.resources.nodes == 2 and r.location.resources.gpus_per_node == 4


def test_free_port_advances_past_busy():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        busy = blocker.getsockname()[1]
        port, note = resolve._free_port(busy)
        assert port != busy and "busy" in note


def test_runtime_probe_skips_broken_runtimes(monkeypatch):
    """podman on PATH but not working (no subuids / dead daemon) must fall
    through to the next runtime instead of dead-ending the user."""
    monkeypatch.setattr(
        resolve.shutil, "which",
        lambda name: f"/usr/bin/{name}" if name in ("podman", "docker") else None,
    )
    monkeypatch.setattr(resolve, "_runtime_works", lambda c, timeout_s=10.0: c == "docker")
    runtime, why = resolve.detect_runtime()
    assert runtime == "docker" and "podman" in why  # decision explains the skip


def test_slug_generation():
    assert resolve._slug("hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf") \
        == "boxy-qwen2.5-0.5b-instruct-q4_k_m"
    assert resolve._slug("weird name!!.gguf").startswith("boxy-weird-name")
