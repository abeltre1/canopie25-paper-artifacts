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
    return Location(name="clustera", scheduler="slurm", accelerator=accelerator, runtime=runtime)


# ---- the rendered script is self-contained (no boxy on the node) --------------


def test_render_script_is_boxy_free(staged_gguf, tmp_path):
    script = deploy.render_agentless_script(
        _box(staged_gguf), _loc(), "slurm", "boxy-al",
        str(tmp_path / "boxy-al.endpoint.json"), str(tmp_path / "boxy-al-%j.log"),
        site_args=["--account=ab110003"], port=8090)
    assert "boxy serve" not in script and "python -m boxy" not in script   # NO boxy anywhere
    assert "podman run" in script and "ghcr.io/ggml-org/llama.cpp:server-cuda" in script
    assert "#SBATCH --job-name=boxy-al" in script and "#SBATCH --account=ab110003" in script
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
    loc.write_text('[location]\nname = "clustera"\nscheduler = "slurm"\nruntime = "podman"\n')
    rc = main(["generate", "slurm", "--box", str(box), "--location", str(loc), "--accelerator", "cuda"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "podman run" in out and "boxy serve" not in out and "#SBATCH" in out


def test_serve_agentless_dryrun_is_boxy_free(staged_gguf, jobs_dir, capsys):
    rc = main(["serve", str(staged_gguf), "--scheduler", "slurm", "--gpus", "1",
               "--agentless", "--accelerator", "cuda", "--account", "ab110003", "--dryrun"])
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


def test_serve_agentless_live_submit_reaches_ready(staged_gguf, jobs_dir, monkeypatch, capsys):
    """End-to-end (no real cluster): submit the self-contained script via a fake
    sbatch, simulate the compute node publishing the endpoint on RUNNING, and
    confirm boxy follows it to READY. Guards the live agentless path."""
    import boxy.cli as cli
    from boxy import jobs, readiness

    states = ["PENDING\n", "RUNNING\n"]
    seen = {}

    def fake_run(cmd, **kw):
        out, rc = "", 0
        if cmd[0] == "sbatch":
            out = "4242\n"
            seen["script"] = jobs.script_path("boxy-al").read_text()  # what was submitted
        elif cmd[0] == "squeue":
            out = states.pop(0) if len(states) > 1 else states[0]
            if "RUNNING" in out and not jobs.read_endpoint("boxy-al"):
                jobs.write_endpoint("boxy-al", 8090, job_id="4242")  # the bash script's job
        return type("R", (), {"returncode": rc, "stdout": out, "stderr": ""})()

    real_which = cli.shutil.which
    monkeypatch.setattr(cli.shutil, "which",
                        lambda b: "/usr/bin/x" if b in ("sbatch", "squeue", "scancel", "podman")
                        else real_which(b))
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(readiness, "wait_ready",
                        lambda url, **kw: "llama" if jobs.read_endpoint("boxy-al") else None)

    rc = main(["serve", str(staged_gguf), "--scheduler", "slurm", "--gpus", "1",
               "--agentless", "--accelerator", "cuda", "--name", "boxy-al"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "### Submitted slurm job 4242" in out and "### READY" in out and ":8090/v1" in out
    # the submitted script was boxy-free and named the container after the job
    assert "boxy serve" not in seen["script"] and "--name=boxy-al" in seen["script"]
    assert jobs.read_record("boxy-al")["job"] == "4242"


# ---- multi-node (Ray) agentless: one vLLM instance across the allocation ------


def _vllm_loc(nodes, gpus, scheduler="slurm"):
    from boxy.location import Resources
    return Location(name="clustera", scheduler=scheduler, accelerator="cuda", runtime="podman",
                    resources=Resources(nodes=nodes, gpus_per_node=gpus))


def test_render_multinode_fans_ray_workers_from_the_script(tmp_path):
    # --nodes 2 over the agentless path: the head node runs `ray start --head` +
    # vllm (TP=gpus x PP=nodes), and the batch script itself sruns ONE worker
    # container onto the other node — still zero boxy on the cluster.
    script = deploy.render_agentless_script(
        _box("meta-llama/Llama-3.1-70B-Instruct", engine="vllm"), _vllm_loc(2, 4),
        "slurm", "boxy-70b", str(tmp_path / "ep.json"), str(tmp_path / "%j.log"),
        [], port=8000, engine_pulls_model=True)
    assert "#SBATCH --nodes=2" in script and "#SBATCH --ntasks-per-node=1" in script
    # head: ray head + wait-for-8-GPUs + vllm with the derived geometry
    assert "ray start --head" in script
    assert "--tensor-parallel-size=4" in script and "--pipeline-parallel-size=2" in script
    assert "--distributed-executor-backend=ray" in script
    # worker fan-out: srun places it on the non-head node, head IP found at runtime
    assert 'srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --exclude "$(hostname -s)"' in script
    assert '_HEAD_IP="$(hostname -I' in script
    assert 'BOXY_RAY_HEAD="${_HEAD_IP}"' in script
    # the WORKER container gets the same site CA the head does — its ray
    # self-heal pip install rides the interceptor and dies TLS otherwise
    # (field: head installed ray fine, every worker failed CERTIFICATE_VERIFY)
    worker_line = [ln for ln in script.splitlines() if "srun --nodes=1" in ln][0]
    assert "${_CAARGS}" in worker_line
    assert "ray start --address=${BOXY_RAY_HEAD}" in script
    assert "boxy serve" not in script                     # still zero-install
    # workers launch in the BACKGROUND; the head podman stays the foreground exec
    head_line = [line for line in script.splitlines() if line.startswith("exec ")][-1]
    assert "ray start --head" in head_line


def test_render_multinode_flux_uses_flux_run(tmp_path):
    script = deploy.render_agentless_script(
        _box("meta-llama/Llama-3.1-70B-Instruct", engine="vllm"), _vllm_loc(3, 4, "flux"),
        "flux", "boxy-70b", str(tmp_path / "ep.json"), str(tmp_path / "%j.log"),
        [], port=8000, engine_pulls_model=True)
    # `flux exec -r`, never `flux run`: the scheduler can't see the head's
    # plain podman process and co-locates the worker on the head's node
    # (audit-confirmed); broker ranks 1..N-1 are exactly the non-head nodes
    assert "flux exec -r 1-2" in script
    assert "flux run" not in script
    assert "--pipeline-parallel-size=3" in script


def test_render_multinode_opt_out_and_single_node_unchanged(staged_gguf, tmp_path):
    # --no-distributed renders the plain single-node script even at nodes=2 …
    script = deploy.render_agentless_script(
        _box("meta-llama/Llama-3.1-70B-Instruct", engine="vllm"), _vllm_loc(2, 4),
        "slurm", "b", str(tmp_path / "ep.json"), str(tmp_path / "l.log"),
        [], port=8000, engine_pulls_model=True, distributed=False)
    assert "ray start" not in script and "srun" not in script
    # … llama.cpp is never distributed …
    script = deploy.render_agentless_script(
        _box(staged_gguf), _vllm_loc(2, 1), "slurm", "b",
        str(tmp_path / "ep2.json"), str(tmp_path / "l2.log"), [], port=8090)
    assert "ray start" not in script
    # … and nodes=1 never grows Ray, even with distributed requested
    script = deploy.render_agentless_script(
        _box("meta-llama/Llama-3.1-8B-Instruct", engine="vllm"), _vllm_loc(1, 4),
        "slurm", "b", str(tmp_path / "ep3.json"), str(tmp_path / "l3.log"),
        [], port=8000, engine_pulls_model=True, distributed=True)
    assert "ray start" not in script


def test_render_multinode_needs_gpu_count(tmp_path):
    with pytest.raises(deploy.AgentlessError, match="gpus"):
        deploy.render_agentless_script(
            _box("meta-llama/Llama-3.1-70B-Instruct", engine="vllm"), _vllm_loc(2, 0),
            "slurm", "b", str(tmp_path / "ep.json"), str(tmp_path / "l.log"),
            [], port=8000, engine_pulls_model=True)


# ---- target platform: the script RUNS on Linux no matter where it is BUILT ----


def test_render_from_mac_still_targets_linux(staged_gguf, tmp_path, monkeypatch):
    # FIELD (clusterb, Nemotron TP=2): rendered on a Mac laptop, the podman
    # command took the darwin branch (-p publishing, NO --ipc=host) — so the
    # container ran on podman's 64MB /dev/shm and RCCL died at ncclCommInitRank
    # ('NCCL error: unhandled system error') as soon as a second GPU joined.
    # The agentless render must pin the TARGET platform (Linux), not follow
    # the laptop's.
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "darwin")
    script = deploy.render_agentless_script(
        _box(staged_gguf), _loc(), "slurm", "boxy-al",
        str(tmp_path / "e.json"), str(tmp_path / "l-%j.log"), [], port=8090)
    assert "--network=host" in script and "--ipc=host" in script
    assert "-p 8090:8090" not in script                      # not the mac port-publish branch
    # and the pin does not leak: a plain local darwin build publishes ports again
    from boxy.backends.podman import PodmanBackend

    args = PodmanBackend().network_args(_box(staged_gguf), ["llama-server", "--port", "8090"])
    assert args == ["-p", "8090:8090"]


def test_render_hf_via_s3_uri_names_the_hf_transport(tmp_path):
    """Field: `s3://huggingface.co/org/name` — HF is not an S3 bucket; the
    error must point at the hf:// transport the user actually wanted."""
    with pytest.raises(deploy.AgentlessError,
                       match=r"hf://meta-llama/Llama-3.1-8B-Instruct"):
        deploy.render_agentless_script(
            _box("s3://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"), _loc(),
            "slurm", "x", str(tmp_path / "x.json"), str(tmp_path / "x.log"), [])
