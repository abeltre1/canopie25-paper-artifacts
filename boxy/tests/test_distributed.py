"""Phase 1: multi-node distributed (Ray) serving.

Golden/dryrun coverage for the two-axis scale-out: geometry -> tensor/pipeline
parallelism, the Ray head/worker inner commands, the launcher dispatch
(slurm srun / flux run / local containers), the `#SBATCH --ntasks-per-node=1`
directive, inner-command flag forwarding, and the deploy-level head+worker plan.
`nodes==1` must stay byte-identical to the single-container path (no Ray).
"""

import shlex
from types import SimpleNamespace

import pytest

from boxy import distributed, engines
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location, Resources, Staging
from boxy.schedulers import get_scheduler
from tests.conftest import EXAMPLES


# ---- geometry -> parallelism -------------------------------------------------


def test_derive_parallelism_geometry():
    # TP = gpus_per_node (intra-node), PP = nodes (inter-node), world = TP*PP.
    assert distributed.derive_parallelism(Resources(nodes=2, gpus_per_node=4)) == (
        4,
        2,
        8,
    )
    assert distributed.derive_parallelism(Resources(nodes=4, gpus_per_node=8)) == (
        8,
        4,
        32,
    )
    assert distributed.derive_parallelism(Resources(nodes=1, gpus_per_node=2)) == (
        2,
        1,
        2,
    )


def test_derive_parallelism_needs_gpu_count():
    # gpus_per_node==0 means "autodetect": TP is unknowable, so fail fast with a
    # message that tells the user to pass --gpus.
    with pytest.raises(RuntimeError, match="tensor-parallel"):
        distributed.derive_parallelism(Resources(nodes=2, gpus_per_node=0))


# ---- when is distributed on? -------------------------------------------------


def test_is_distributed_default_on_for_vllm_multinode():
    assert distributed.is_distributed("vllm", 2, None) is True
    assert distributed.is_distributed("vllm", 8, None) is True


def test_is_distributed_off_for_single_node():
    # --distributed --nodes 1 cleanly degrades to single-node (no Ray).
    assert distributed.is_distributed("vllm", 1, None) is False
    assert distributed.is_distributed("vllm", 1, True) is False


def test_is_distributed_off_for_llamacpp():
    assert distributed.is_distributed("llama.cpp", 4, None) is False
    assert distributed.is_distributed("llama.cpp", 4, True) is False


def test_is_distributed_explicit_flag_wins():
    assert distributed.is_distributed("vllm", 2, False) is False  # --no-distributed
    assert distributed.is_distributed("vllm", 2, True) is True  # --distributed


# ---- Ray inner commands ------------------------------------------------------


def test_ray_head_inner_starts_ray_waits_then_execs_vllm():
    argv = ["vllm", "serve", "/mnt/models/m", "--tensor-parallel-size=4"]
    head = distributed.ray_head_inner(argv, gpus_per_node=4, world=8)
    assert head[:2] == ["bash", "-lc"]
    script = head[2]
    # order: ray fallback shim -> start head -> wait for the whole cluster ->
    # exec vLLM in the same shell (ray.init marks the cluster-wait step)
    assert script.index("command -v ray") < script.index(
        "ray start --head --port=6379 --num-gpus=4"
    )
    assert script.index("ray start --head") < script.index("ray.init(address=")
    assert script.index("ray.init(address=") < script.index(
        "exec vllm serve /mnt/models/m"
    )
    assert (
        "&&" in script
    )  # each step gates the next; a failed ray start never reaches vLLM


def test_ray_fallback_shim_covers_imageless_ray():
    """Field: `bash: ray: command not found` (rc=127) on a ROCm vLLM image —
    the inner scripts shim `ray` over the module entrypoint when the CLI is
    absent, and pip-install ray as the last resort before failing loudly."""
    for inner in (
        distributed.ray_head_inner(["vllm", "serve", "/m"], 4, 8),
        distributed.ray_worker_inner(4),
    ):
        script = inner[2]
        assert "command -v ray" in script
        assert (
            "from ray.scripts.scripts import main" in script
        )  # module-entrypoint shim
        assert (
            "pip install -q --no-cache-dir" in script and " ray" in script
        )  # self-heal rung
        assert (
            'SSL_CERT_FILE:+--cert "$SSL_CERT_FILE"' in script
        )  # pip trusts the site CA
        assert "vllm[ray]" in script  # the loud last-resort error


def test_ray_head_wait_probes_are_time_boxed_subprocesses():
    """Field: a healthy 2-node cluster (disjoint placement, worker joined) still
    went silent — the wait driver's first GCS RPC blocked inside Ray's C++
    client right after 'Connected to Ray cluster', where no in-process guard
    can interrupt it. So the wait is a bash poll loop of FRESH `timeout`-boxed
    python probes: a wedged probe is reaped (with KILL escalation) and simply
    retried on the next tick, and the loop itself is pure bash — it cannot
    hang, and a transient wedge is no longer fatal to the job."""
    script = distributed.ray_head_inner(["vllm", "serve", "/m"], 1, 2)[2]
    assert "timeout -k 10 45" in script  # per-probe hard box + KILL escalation
    assert "timeout 660" not in script  # the old all-or-nothing outer guard is gone
    assert "ray cluster GPUs: $g/2" in script  # heartbeat every 30s
    assert "ray cluster complete" in script  # loud success line before vLLM
    assert "never reached 2 GPUs" in script  # deadline -> node-table dump
    assert "exit 8" in script  # loud abort path retained


def test_ray_head_wait_targets_world_gpu_count():
    head = distributed.ray_head_inner(["vllm", "serve", "/m"], gpus_per_node=4, world=8)
    # the readiness gate blocks until all 8 GPUs register, else exits non-zero
    assert ">=8" in head[2] or "8" in head[2]


def test_ray_worker_inner_joins_head_and_blocks():
    worker = distributed.ray_worker_inner(gpus_per_node=4)
    assert worker[:2] == ["bash", "-lc"]
    # head IP comes from the env var the launcher bakes in; --block holds the node
    assert "ray start --address=${BOXY_RAY_HEAD}:6379 --num-gpus=4 --block" in worker[2]
    # the join RETRIES — workers race the head's `ray start --head` (field: the
    # agentless batch script launches both into the allocation together)
    assert "for _i in" in worker[2] and "sleep 10" in worker[2]


def test_ray_worker_does_not_rejoin_a_dead_cluster():
    """Field: the head died mid-job and the worker's retry loop cycled 'Ray
    runtime started' forever in an orphaned container. A join that held 120s+
    was a formed cluster — when its --block exits, the worker stops instead of
    retrying against a head that is gone."""
    worker = distributed.ray_worker_inner(gpus_per_node=1)[2]
    assert "-ge 120" in worker
    assert "not retrying" in worker
    # a QUICK failure (head not up yet) still retries
    assert "retrying in 10s" in worker


# ---- launcher dispatch -------------------------------------------------------


def test_detect_launcher_from_allocation_env(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("FLUX_JOB_ID", raising=False)
    assert distributed.detect_launcher() == "none"
    monkeypatch.setenv("FLUX_JOB_ID", "f123")
    assert distributed.detect_launcher() == "flux"
    monkeypatch.setenv("SLURM_JOB_ID", "999")  # slurm wins if both present
    assert distributed.detect_launcher() == "slurm"


def test_worker_launch_prefix_slurm():
    # one worker task per non-head node, head excluded so it isn't double-booked
    assert distributed.worker_launch_prefix("slurm", "node0", 2) == [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--exclude",
        "node0",
    ]
    assert distributed.worker_launch_prefix("slurm", "n0", 4)[:3] == [
        "srun",
        "--nodes=3",
        "--ntasks=3",
    ]


def test_worker_launch_prefix_flux():
    """`flux exec -r`, never `flux run`: fluxion can't see the head's plain
    podman process and co-locates the worker on the head's node (audit); broker
    ranks 1..N-1 are exactly the non-head nodes."""
    assert distributed.worker_launch_prefix("flux", "node0", 2) == [
        "flux",
        "exec",
        "-r",
        "1",
    ]
    assert distributed.worker_launch_prefix("flux", "node0", 4) == [
        "flux",
        "exec",
        "-r",
        "1-3",
    ]


def test_worker_launch_prefix_none_is_local():
    # no scheduler: caller launches worker containers directly on the local host
    assert distributed.worker_launch_prefix("none", "whatever", 3) == []


# ---- engines: TP/PP tack-on + override precedence ----------------------------


def _vllm_box(**args) -> Box:
    return Box(
        name="vllm",
        engine="vllm",
        image="vllm/vllm-openai:v0.9.1",
        entrypoint="vllm",
        ports=[8000],
        args=args,
    )


def _loc() -> Location:
    return Location(
        name="t",
        scheduler="slurm",
        accelerator="cuda",
        runtime="podman",
        resources=Resources(nodes=2, gpus_per_node=4),
        staging=Staging(models_dir="./models"),
    )


def test_vllm_serve_cmd_tacks_on_parallelism():
    cmd = engines.build_vllm_serve_cmd(
        _vllm_box(), _loc(), "/mnt/models/m", parallelism=(4, 2)
    )
    assert "--tensor-parallel-size=4" in cmd
    assert "--pipeline-parallel-size=2" in cmd
    assert "--distributed-executor-backend=ray" in cmd


def test_vllm_serve_cmd_no_parallelism_is_unchanged():
    # single-node path: no parallelism arg passed -> no Ray/TP flags appear
    cmd = engines.build_vllm_serve_cmd(_vllm_box(), _loc(), "/mnt/models/m")
    assert not any(a.startswith("--tensor-parallel-size") for a in cmd)
    assert not any(a.startswith("--distributed-executor-backend") for a in cmd)


def test_user_tensor_parallel_size_wins_over_derived():
    # a box/tuning/user value must survive the derived tack-on (skip-if-present rule)
    cmd = engines.build_vllm_serve_cmd(
        _vllm_box(tensor_parallel_size=8), _loc(), "/mnt/models/m", parallelism=(4, 2)
    )
    assert "--tensor-parallel-size=8" in cmd
    assert "--tensor-parallel-size=4" not in cmd
    # pipeline-parallel had no user value, so the derived one still lands
    assert "--pipeline-parallel-size=2" in cmd


def test_llamacpp_ignores_parallelism():
    box = Box(
        name="lcpp",
        engine="llama.cpp",
        image="ghcr.io/ggml-org/llama.cpp:server",
        entrypoint="",
        ports=[8090],
    )
    cmd = engines.build_serve_cmd(box, _loc(), "/mnt/models/m.gguf", parallelism=(4, 2))
    assert not any("tensor-parallel" in a for a in cmd)
    assert not any("distributed-executor" in a for a in cmd)


# ---- slurm resource directives ----------------------------------------------


def test_slurm_ntasks_per_node_only_when_distributed():
    loc = _loc()
    plain = get_scheduler("slurm").resource_directives(loc, distributed=False)
    dist = get_scheduler("slurm").resource_directives(loc, distributed=True)
    assert "#SBATCH --ntasks-per-node=1" not in plain
    assert "#SBATCH --ntasks-per-node=1" in dist
    # the node/gpu directives are identical either way
    assert "#SBATCH --nodes=2" in plain and "#SBATCH --nodes=2" in dist
    assert "#SBATCH --gpus-per-node=4" in plain and "#SBATCH --gpus-per-node=4" in dist


def test_flux_resource_directives_accept_distributed_flag():
    # flux's slot-based directives are unchanged by distributed (param accepted)
    loc = Location(
        name="t",
        scheduler="flux",
        accelerator="rocm",
        runtime="apptainer",
        resources=Resources(nodes=2, gpus_per_node=4),
    )
    assert get_scheduler("flux").resource_directives(
        loc, distributed=True
    ) == get_scheduler("flux").resource_directives(loc, distributed=False)


# ---- inner-command forwarding ------------------------------------------------


def _inner_args(**over):
    base = dict(
        location=None,
        models_dir=None,
        nodes=None,
        gpus=None,
        distributed=None,
        engine=None,
        image=None,
        runtime=None,
        accelerator=None,
        port=None,
        args=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_inner_serve_command_forwards_geometry_and_distributed():
    from boxy.cli import _inner_serve_command

    inner = _inner_serve_command(
        _inner_args(nodes=2, gpus=4, distributed=True), "m", "vllm"
    )
    toks = shlex.split(inner)
    assert "--nodes" in toks and toks[toks.index("--nodes") + 1] == "2"
    assert "--gpus" in toks and toks[toks.index("--gpus") + 1] == "4"
    assert "--distributed" in toks
    # always re-resolved on the compute node
    assert "--foreground" in toks and "--here" in toks


def test_inner_serve_command_no_distributed_forwarded():
    from boxy.cli import _inner_serve_command

    toks = shlex.split(
        _inner_serve_command(
            _inner_args(nodes=2, gpus=4, distributed=False), "m", "vllm"
        )
    )
    assert "--no-distributed" in toks
    assert "--distributed" not in toks


def test_inner_serve_command_single_node_omits_distributed():
    from boxy.cli import _inner_serve_command

    toks = shlex.split(_inner_serve_command(_inner_args(nodes=1), "m", "vllm"))
    assert "--distributed" not in toks
    assert "--no-distributed" not in toks


# ---- deploy: head + worker plan ---------------------------------------------


def test_plan_serve_distributed_head_and_worker():
    box = _vllm_box()
    box = Box(
        name="vllm",
        engine="vllm",
        image="vllm/vllm-openai:v0.9.1",
        entrypoint="vllm",
        model="models/llama",
        ports=[8000],
    )
    from boxy import deploy

    dep = deploy.plan_serve(
        box, _loc(), port=8000, dryrun=True, distributed=True, head_ip="192.0.2.7"
    )
    assert dep.distributed is True
    assert dep.parallelism == (4, 2)
    assert dep.world_size == 8
    assert dep.port == 8000

    head = shlex.join(dep.command)
    # head runs the Ray head + vLLM directly — NOT wrapped in an srun launch prefix
    assert not dep.command[0] == "srun"
    assert "ray start --head --port=6379 --num-gpus=4" in head
    assert "exec vllm serve" in head
    assert "--tensor-parallel-size=4" in head
    assert "--pipeline-parallel-size=2" in head
    assert "--distributed-executor-backend=ray" in head

    assert dep.worker_command is not None
    worker = shlex.join(dep.worker_command)
    assert "ray start --address=${BOXY_RAY_HEAD}:6379 --num-gpus=4 --block" in worker
    # the head IP is baked into the worker container's env
    assert "BOXY_RAY_HEAD=192.0.2.7" in worker
    # worker container gets its own name so it never collides with the head
    assert "--name=vllm-worker" in worker


def test_plan_serve_distributed_needs_gpu_count():
    box = Box(
        name="vllm",
        engine="vllm",
        image="vllm/vllm-openai:v0.9.1",
        entrypoint="vllm",
        model="models/llama",
        ports=[8000],
    )
    loc = Location(
        name="t",
        scheduler="slurm",
        accelerator="cuda",
        runtime="podman",
        resources=Resources(nodes=2, gpus_per_node=0),
    )
    from boxy import deploy

    with pytest.raises(RuntimeError, match="tensor-parallel"):
        deploy.plan_serve(box, loc, dryrun=True, distributed=True, head_ip="192.0.2.7")


# ---- CLI dryrun: auto-distribute + launcher dispatch ------------------------


def test_cli_auto_distributes_vllm_on_slurm_multinode(capsys):
    # clustera = 2-node slurm vLLM: auto-distributed, workers placed with srun.
    rc = main(
        [
            "serve",
            "--box",
            str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location",
            str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"),
            "--dryrun",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Head" in out and "### Worker" in out
    assert "ray start --head" in out and "ray start --address=${BOXY_RAY_HEAD}" in out
    assert "--tensor-parallel-size=4" in out and "--pipeline-parallel-size=2" in out
    # slurm launcher: worker fanned out with srun, one task per non-head node
    assert "srun --nodes=1 --ntasks=1 --ntasks-per-node=1" in out
    assert "podman" in out


def test_cli_no_distribute_flag_forces_single_container(capsys):
    # --no-distributed on the 2-node slurm location: back to the classic srun wrap.
    rc = main(
        [
            "serve",
            "--box",
            str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location",
            str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"),
            "--no-distributed",
            "--dryrun",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Running Command:" in out
    assert "### Head" not in out
    assert "ray start" not in out


def test_cli_distributed_no_scheduler_is_local_container_set(capsys, tmp_path):
    # scheduler=none + nodes>1: "a set of containers" on the local host — no srun,
    # no flux run; each worker container gets its own name so they don't collide.
    loc = tmp_path / "local2.toml"
    loc.write_text(
        '[location]\nname = "local2"\nscheduler = "none"\naccelerator = "none"\n'
        'runtime = "docker"\noffline = true\n'
        "[location.resources]\nnodes = 2\ngpus_per_node = 4\n"
        '[location.staging]\nmodels_dir = "./models"\n'
    )
    rc = main(
        [
            "serve",
            "--box",
            str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location",
            str(loc),
            "--dryrun",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Head" in out and "### Worker" in out
    assert "local containers" in out  # the launcher label
    assert "srun" not in out and "flux run" not in out
    assert "ray start --head" in out and "ray start --address=${BOXY_RAY_HEAD}" in out
    assert "--name=vllm-worker0" in out  # nodes-1 == 1 local worker, uniquely named
    assert "--pipeline-parallel-size=2" in out
