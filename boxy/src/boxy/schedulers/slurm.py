"""Slurm adapter: srun launch prefix sized from the location's resources."""

from __future__ import annotations

from boxy.location import Location
from boxy.schedulers.base import PartitionInfo, Scheduler


# Site GRES convention auto-detected from `sinfo` over --ssh (set by the CLI just
# before rendering; consulted only when site.gpu_directive is 'auto'). Process-
# global for one invocation; reset between tests (conftest).
_AUTO_GRES = {"form": "", "type": ""}


def set_auto_gres(form: str, gtype: str) -> None:
    _AUTO_GRES["form"], _AUTO_GRES["type"] = (form or ""), (gtype or "")


def reset_auto_gres() -> None:
    _AUTO_GRES["form"], _AUTO_GRES["type"] = "", ""


def _gpu_flag(n: int) -> str | None:
    """The GPU request flag for N GPUs/node in the site's GRES convention. None to
    omit. Sites differ: '--gpus-per-node=N' works on most modern Slurm, but many
    reject it with 'Invalid generic resource (gres) specification' and want
    '--gres=gpu:N' (optionally typed, gpu:a100:N).

    config site.gpu_directive: 'auto' (default) uses the form auto-detected from
    the cluster's `sinfo` GRES (set_auto_gres), falling back to --gpus-per-node
    when nothing was detected; or pin 'gres'/'gpus'/'gpus-per-node'/'none'.
    config site.gpu_type pins the GRES type (else the detected one)."""
    if n <= 0:
        return None
    from boxy import config

    form = (config.get_str("site.gpu_directive") or "auto").strip().lower()
    gtype = config.get_str("site.gpu_type").strip()
    if form == "auto":
        form = _AUTO_GRES["form"] or "gpus-per-node"
        gtype = gtype or _AUTO_GRES["type"]
    typed = f"{gtype}:{n}" if gtype else str(n)     # a100:2  /  2
    if form == "none":
        return None
    if form == "gres":
        return f"--gres=gpu:{typed}"                # --gres=gpu:a100:2 / --gres=gpu:2
    if form == "gpus":
        return f"--gpus={typed}"
    return f"--gpus-per-node={typed}"               # default fallback


class SlurmScheduler(Scheduler):
    name = "slurm"
    launcher = "srun"

    def launch_prefix(self, location: Location) -> list[str]:

        prefix = [self.launcher, f"--nodes={location.resources.nodes}"]
        gpu = _gpu_flag(location.resources.gpus_per_node)
        if gpu:
            prefix.append(gpu)
        for arg in location.scheduler_args:
            # split only the single-char "-X value" spelling; everything else
            # is ONE token (shlex.split choked on values with apostrophes)
            if arg.startswith("-") and not arg.startswith("--") and " " in arg:
                prefix += arg.split(" ", 1)
            else:
                prefix.append(arg)
        return prefix

    def host_env_fixups(self) -> list[str]:
        return ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]

    def alloc_command(self, location: Location) -> list[str]:
        """Interactive allocation (paper: 0-alloc-compute-node.sh)."""
        cmd = ["salloc", f"--nodes={location.resources.nodes}"]
        gpu = _gpu_flag(location.resources.gpus_per_node)
        if gpu:
            cmd.append(gpu)
        return cmd

    # ---- batch submission ----

    directive_prefix = "#SBATCH"
    output_token = "%j"  # Slurm substitutes the job id into --output

    def resource_directives(self, location: Location, distributed: bool = False) -> list[str]:
        lines = [f"#SBATCH --nodes={location.resources.nodes}"]
        gpu = _gpu_flag(location.resources.gpus_per_node)
        if gpu:
            lines.append(f"#SBATCH {gpu}")
        if distributed:
            # one Ray launcher (srun task) per node
            lines.append("#SBATCH --ntasks-per-node=1")
        return lines

    def site_directive(self, kind: str, value: str) -> str:
        return {"partition": f"--partition={value}",
                "account": f"--account={value}",
                "time": f"--time={value}"}[kind]

    def submit_command(self, script: str) -> list[str]:
        return ["sbatch", "--parsable", script]

    def parse_job_id(self, submit_stdout: str) -> str:
        # --parsable prints "jobid" or "jobid;cluster"
        last = super().parse_job_id(submit_stdout)
        return last.split(";")[0]

    def cancel_command(self, job_id: str) -> list[str]:
        return ["scancel", job_id]

    def state_command(self, job_id: str) -> list[str]:
        return ["squeue", "-h", "-j", job_id, "-o", "%T"]

    def interpret_state(self, stdout: str) -> str:
        state = stdout.strip().upper()
        if not state:
            return "DONE"  # left the queue
        if state in ("PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED", "RESIZING"):
            return "PENDING"  # alive but not serving yet (r2: these spun as UNKNOWN)
        if state in ("RUNNING", "COMPLETING"):
            return "RUNNING"
        if state in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY"):
            return "DONE"
        return "UNKNOWN"

    def partitions_command(self) -> list[str]:
        # %R = partition name (no default `*` marker), %a = up/down,
        # %F = nodes as allocated/idle/other/total (idle ranks soonest-start),
        # %G = generic resources (gpu:… when the partition has accelerators).
        # Pipe-delimited because %G can be `(null)`/contain colons; `-h` drops
        # the header.
        return ["sinfo", "-h", "-o", "%R|%a|%F|%G"]

    def parse_partitions(self, stdout: str) -> list[PartitionInfo]:
        # sinfo prints a partition on several lines (one per node-state group);
        # %F is the whole-partition A/I/O/T on each, so aggregate by name (max
        # idle seen, up if any line is up, has_gpu if any group advertises gpu).
        agg: dict[str, list] = {}
        for line in stdout.splitlines():
            cols = line.split("|")
            if len(cols) < 3:
                continue
            name, avail, nodes = cols[0].strip(), cols[1].strip(), cols[2].strip()
            if not name:
                continue
            bits = nodes.split("/")
            idle = int(bits[1]) if len(bits) >= 2 and bits[1].isdigit() else 0
            up = avail.lower().startswith("up")
            gres = cols[3].strip().lower() if len(cols) > 3 else ""
            has_gpu = "gpu" in gres  # e.g. "gpu:a100:4"
            if name in agg:
                agg[name][0] = max(agg[name][0], idle)
                agg[name][1] = agg[name][1] or up
                agg[name][2] = agg[name][2] or has_gpu
            else:
                agg[name] = [idle, up, has_gpu]
        return [PartitionInfo(n, v[0], v[1], v[2]) for n, v in agg.items()]
