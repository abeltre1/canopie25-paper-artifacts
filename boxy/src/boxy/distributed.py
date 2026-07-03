"""Multi-node distributed vLLM serving via a Ray cluster on the allocation.

One model instance spread across N nodes: tensor-parallel WITHIN a node (fast
intra-node NVLink/xGMI) and pipeline-parallel ACROSS nodes. The head node starts
`ray start --head` and runs `vllm serve`; every other node runs
`ray start --address=<head>:6379 --block` to join. Containers use host
networking so Ray's ports span nodes; the head container both starts Ray (a
daemon that returns) and runs vLLM in the same shell so vLLM joins that Ray.

Only the PURE builders live here (golden-testable). The runtime orchestration —
topology discovery and launching head/workers — is driven from cli.py's
foreground serve branch, which runs on the head node inside the allocation.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess

from boxy.location import Resources

RAY_PORT = 6379
HEAD_ENV = "BOXY_RAY_HEAD"  # a worker reads the head IP from this env var


def is_distributed(engine: str, nodes: int, flag: bool | None) -> bool:
    """Distributed (Ray) serving = one vLLM instance across >1 node. On by default
    for vllm + nodes>1; `--no-distributed` (flag is False) opts out. A single node
    never needs Ray, so `--distributed --nodes 1` cleanly degrades to single-node.
    Never for llama.cpp."""
    if engine != "vllm" or nodes <= 1:
        return False
    return flag is not False


def derive_parallelism(resources: Resources) -> tuple[int, int, int]:
    """(tensor_parallel_size, pipeline_parallel_size, world_size) from geometry:
    TP = gpus_per_node (intra-node), PP = nodes (inter-node), world = TP*PP."""
    n, g = resources.nodes, resources.gpus_per_node
    if g < 1:
        raise RuntimeError(
            "distributed serving needs a GPU count per node so the tensor-parallel size is "
            "known — pass --gpus N (or set [location.resources] gpus_per_node)."
        )
    return g, n, n * g


def _ray_wait(world: int, timeout_s: int = 600) -> str:
    """Inline python (runs INSIDE the head container, where ray is importable):
    block until the Ray cluster registers `world` GPUs, else exit non-zero so the
    container fails loudly instead of vLLM starting against a half-formed cluster."""
    py = (
        "import ray,time,sys;ray.init(address='auto');dl=time.time()+{t};"
        "g=lambda:ray.cluster_resources().get('GPU',0);"
        "[time.sleep(2) for _ in iter(lambda:(g()<{w} and time.time()<dl),False)];"
        "sys.exit(0 if g()>={w} else "
        "(sys.stderr.write('ray cluster never reached {w} GPUs\\n') or 1))"
    ).format(t=timeout_s, w=world)
    return "python3 -c " + shlex.quote(py)


def ray_head_inner(vllm_argv: list[str], gpus_per_node: int, world: int) -> list[str]:
    """Container inner command for the HEAD node: start the Ray head, wait for all
    workers to join (all `world` GPUs), then exec vLLM using the Ray cluster."""
    script = (
        f"ray start --head --port={RAY_PORT} --num-gpus={gpus_per_node} && "
        f"{_ray_wait(world)} && "
        f"exec {shlex.join(vllm_argv)}"
    )
    return ["bash", "-lc", script]


def ray_worker_inner(gpus_per_node: int) -> list[str]:
    """Container inner command for a WORKER node: join the head's Ray cluster and
    --block (keeps the node in the cluster for the job's lifetime). The head IP
    arrives via the BOXY_RAY_HEAD env var set on the srun that launches it."""
    script = f"ray start --address=${{{HEAD_ENV}}}:{RAY_PORT} --num-gpus={gpus_per_node} --block"
    return ["bash", "-lc", script]


def detect_launcher() -> str:
    """How to place worker containers on the other nodes, from the allocation we
    are actually running inside: 'slurm' (srun), 'flux' (flux run), or 'none' (a
    set of containers on the local host — no scheduler)."""
    if os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if os.environ.get("FLUX_JOB_ID"):
        return "flux"
    return "none"


def worker_launch_prefix(launcher: str, head_node: str, nodes: int) -> list[str]:
    """Prefix that fans ONE worker container out onto the N-1 non-head nodes.
    Empty for 'none' — there the caller launches the worker containers directly
    on the local host (a single-host set of containers)."""
    if launcher == "slurm":
        return ["srun", f"--nodes={nodes - 1}", f"--ntasks={nodes - 1}",
                "--ntasks-per-node=1", "--exclude", head_node]
    if launcher == "flux":
        return ["flux", "run", f"-N{nodes - 1}", f"-n{nodes - 1}", "--tasks-per-node=1"]
    return []  # none: launched locally, one Popen per worker


def discover_topology(launcher: str) -> tuple[str, str, int]:
    """(head_node, head_ip, node_count) at runtime on the head node. head_ip is
    this node's primary IP (containers use host networking, so workers reach it)."""
    nodes: list[str] = []
    try:
        if launcher == "slurm":
            nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
            if nodelist:
                out = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                                     capture_output=True, text=True, timeout=15)
                nodes = [n for n in out.stdout.split() if n]
        elif launcher == "flux":
            out = subprocess.run(["flux", "hostlist", "-e", "local"],
                                 capture_output=True, text=True, timeout=15)
            nodes = [n for n in out.stdout.split() if n]
    except (OSError, subprocess.SubprocessError):
        nodes = []
    head = nodes[0] if nodes else socket.gethostname()
    head_ip = "127.0.0.1"
    try:
        ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=15).stdout.split()
        if ips:
            head_ip = ips[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return head, head_ip, len(nodes) or 1
