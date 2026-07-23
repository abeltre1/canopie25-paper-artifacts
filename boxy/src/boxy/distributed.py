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

from boxy import config
from boxy.location import Resources

RAY_PORT = 6379  # built-in default; effective value via config network.ray_port (BOXY_RAY_PORT)
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
    container fails loudly instead of vLLM starting against a half-formed cluster.

    NEVER silent (field: two runs stalled after 'Connected to Ray cluster' with
    nothing to diagnose from): a heartbeat prints the GPU count every 30s, the
    success path announces itself before vLLM takes over, and a timeout dumps
    every node's address/liveness/resources."""
    py = (
        "import json, sys, time\n"
        "import ray\n"
        "ray.init(address='auto')\n"
        f"world = {world}\n"
        f"deadline = time.time() + {timeout_s}\n"
        "def gpus():\n"
        "    return int(ray.cluster_resources().get('GPU', 0))\n"
        "last = 0.0\n"
        "while gpus() < world and time.time() < deadline:\n"
        "    if time.time() - last >= 30:\n"
        "        last = time.time()\n"
        "        print('boxy: ray cluster GPUs: %d/%d — waiting for workers ...'\n"
        "              % (gpus(), world), file=sys.stderr, flush=True)\n"
        "    time.sleep(2)\n"
        "if gpus() >= world:\n"
        "    print('boxy: ray cluster complete (%d/%d GPUs) — starting vLLM'\n"
        "          % (gpus(), world), file=sys.stderr, flush=True)\n"
        "    sys.exit(0)\n"
        "print('boxy: ray cluster never reached %d GPUs (have %d) — nodes:'\n"
        "      % (world, gpus()), file=sys.stderr, flush=True)\n"
        "for n in ray.nodes():\n"
        "    print(json.dumps({'ip': n.get('NodeManagerAddress'), 'alive': n.get('Alive'),\n"
        "                      'resources': n.get('Resources', {})}), file=sys.stderr, flush=True)\n"
        "sys.exit(1)\n"
    )
    return "python3 -c " + shlex.quote(py)


# The serving image may not expose a `ray` CLI on PATH (field: `bash: ray:
# command not found`, rc=127, ROCm vLLM image): when the binary is absent,
# shim a bash function named `ray` over the module entrypoint — and when even
# the module is missing, SELF-HEAL with a pip install at container start
# (rides the job's forwarded proxy + CA, the card-pip-deps pattern).
RAY_FALLBACK = (
    "if ! command -v ray >/dev/null 2>&1; then "
    "if ! python3 -c 'import ray' >/dev/null 2>&1; then "
    "echo 'boxy: this image ships no ray — installing it (multi-node serving needs it) ...' >&2; "
    # the in-container pip rides the site interceptor: use the CA boxy mounts
    # (SSL_CERT_FILE) explicitly, or pip's vendored certifi rejects the cert
    "python3 -m pip install -q --no-cache-dir ${SSL_CERT_FILE:+--cert \"$SSL_CERT_FILE\"} ray || "
    "{ echo 'boxy: could not install ray — multi-node serving needs an image with vllm[ray]: "
    "pin one with --image, or add ray to the model card pip list' >&2; exit 9; }; fi; "
    "ray(){ python3 -c 'import sys; from ray.scripts.scripts import main; sys.exit(main())' \"$@\"; }; "
    "fi"
)


def ray_head_inner(vllm_argv: list[str], gpus_per_node: int, world: int) -> list[str]:
    """Container inner command for the HEAD node: start the Ray head, wait for all
    workers to join (all `world` GPUs), then exec vLLM using the Ray cluster."""
    script = (
        f"{RAY_FALLBACK}; "
        f"ray start --head --port={config.get_int('network.ray_port')} --num-gpus={gpus_per_node} && "
        f"{_ray_wait(world)} && "
        f"exec {shlex.join(vllm_argv)}"
    )
    return ["bash", "-lc", script]


def ray_worker_inner(gpus_per_node: int) -> list[str]:
    """Container inner command for a WORKER node: join the head's Ray cluster and
    --block (keeps the node in the cluster for the job's lifetime). The head IP
    arrives via the BOXY_RAY_HEAD env var set on the srun that launches it.

    The join RETRIES: the worker containers race the head container's `ray start
    --head` (both are launched into the allocation together), and `ray start
    --address` exits non-zero while the head port isn't up yet. On a clean join,
    --block holds until the cluster shuts down and the loop breaks."""
    join = (f"ray start --address=${{{HEAD_ENV}}}:{config.get_int('network.ray_port')} "
            f"--num-gpus={gpus_per_node} --block")
    script = (f"{RAY_FALLBACK}; "
              f"for _i in $(seq 30); do {join} && break; "
              f"echo \"boxy: ray join attempt $_i failed (head still starting?) — retrying in 10s\" >&2; "
              f"sleep 10; done")
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
    return head, _primary_ip(), len(nodes) or 1


def _primary_ip() -> str:
    """The host's primary (default-route) IPv4, portably. `hostname -I` is
    GNU-only (absent on macOS/BSD); the UDP-connect trick transmits nothing —
    connecting a datagram socket just makes the kernel pick the egress interface —
    and works offline. Falls back to hostname resolution, then loopback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.254.254.254", 1))  # unroutable TEST-NET-ish; no packet sent
            return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
