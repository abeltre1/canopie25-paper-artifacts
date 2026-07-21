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


def test_safetensors_no_gpu_error_gives_concrete_alternatives():
    """The refusal must hand the user runnable next steps for THEIR model.
    It must NOT invent repo names (two guessed-repo 404s in the field) —
    point at search + ollama instead."""
    with pytest.raises(RuntimeError) as e:
        resolve.infer_engine("hf://meta-llama/Meta-Llama-3-8B-Instruct", "none")
    msg = str(e.value)
    assert "'Meta-Llama-3-8B-Instruct GGUF'" in msg      # search phrase for THEIR model
    assert "ollama://" in msg
    assert "--scheduler slurm|flux" in msg and "hf://meta-llama/Meta-Llama-3-8B-Instruct" in msg


def test_port_probe_sees_wildcard_binds():
    """macOS gvproxy binds 0.0.0.0; a 127.0.0.1+SO_REUSEADDR test claims that
    port is free and the launch dies with 'proxy already running' (field
    finding 18). The probe must conflict with wildcard listeners."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as wild:
        wild.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        wild.bind(("", 0))
        wild.listen(1)
        port = wild.getsockname()[1]
        assert resolve._port_taken(port) is True
    assert resolve._port_taken(port) is False  # released after close


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


def test_gpu_job_from_cpu_login_node_defaults_accelerator(cpu_host):
    # Turnkey: a GPU-less login node no longer hard-errors; it assumes the
    # compute node's accelerator (site.default_accelerator, default cuda) and
    # says so — the compute node re-detects the real device when the job runs.
    res = resolve.resolve("hf://org/repo", runtime="podman", scheduler="slurm",
                          gpus=4, require_exists=False)
    assert res.location.accelerator == "cuda"
    assert any("no GPU on this login node" in d for d in res.decisions)


def test_gpu_job_login_node_accelerator_default_is_configurable(cpu_host, monkeypatch):
    monkeypatch.setenv("BOXY_DEFAULT_ACCELERATOR", "rocm")
    res = resolve.resolve("hf://org/repo", runtime="podman", scheduler="slurm",
                          gpus=4, require_exists=False)
    assert res.location.accelerator == "rocm"


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


def test_absolute_path_verified_on_the_remote_target(monkeypatch):
    """Field: `boxy serve /shared-fs/model --ssh cluster` failed the LAPTOP
    exists() check for a path that only exists on the cluster. With a remote
    target, an absolute path is verified over the ssh master instead."""
    from boxy import remote, resolve

    monkeypatch.setattr(remote, "ensure_master", lambda t: 0)
    monkeypatch.setattr(remote, "ssh_capture",
                        lambda t, cmd, timeout=15: (0, "") if "test -e" in cmd else (1, ""))
    model, note = resolve._classify_model("/shared/models/llama-x", True,
                                          remote_target="user1@clustera-login")
    assert model == "/shared/models/llama-x" and "verified over ssh" in note

    monkeypatch.setattr(remote, "ssh_capture", lambda t, cmd, timeout=15: (1, ""))
    with pytest.raises(RuntimeError, match="no such model path on user1@clustera-login"):
        resolve._classify_model("/shared/models/llama-x", True,
                                remote_target="user1@clustera-login")


def test_local_path_check_unchanged_without_remote_target():
    from boxy import resolve

    with pytest.raises(RuntimeError, match="no such model file"):
        resolve._classify_model("/definitely/not/here", True)


def test_remote_verified_path_infers_real_engine(monkeypatch):
    """Field: a shared-FS HF DIRECTORY served over --ssh got 'engine:
    llama.cpp (assumed — model file not present)' because the assumption
    checked the LAPTOP filesystem — the job died on GGUF 'failed to read
    magic'. A remote-verified path infers like any existing path: dir ->
    vLLM, *.gguf -> llama.cpp."""
    from boxy import remote, resolve

    monkeypatch.setattr(remote, "ensure_master", lambda t: 0)
    monkeypatch.setattr(remote, "ssh_capture", lambda t, cmd, timeout=15: (0, ""))
    r = resolve.resolve("/shared/models/Llama-4-Maverick", scheduler="slurm",
                        gpus=4, accelerator="rocm", runtime="podman",
                        remote_target="user1@clustera-login")
    assert r.box.engine == "vllm"
    assert not any("assumed" in d for d in r.decisions)
    r2 = resolve.resolve("/shared/models/llama.gguf", scheduler="slurm",
                         gpus=1, accelerator="rocm", runtime="podman",
                         remote_target="user1@clustera-login")
    assert r2.box.engine == "llama.cpp"
