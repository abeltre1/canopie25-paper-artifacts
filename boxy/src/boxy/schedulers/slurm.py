"""Slurm adapter: srun launch prefix sized from the location's resources."""

from __future__ import annotations

from boxy.location import Location
from boxy.schedulers.base import Scheduler


class SlurmScheduler(Scheduler):
    name = "slurm"
    launcher = "srun"

    def launch_prefix(self, location: Location) -> list[str]:

        prefix = [self.launcher, f"--nodes={location.resources.nodes}"]
        if location.resources.gpus_per_node:
            prefix.append(f"--gpus-per-node={location.resources.gpus_per_node}")
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
        if location.resources.gpus_per_node:
            cmd.append(f"--gpus-per-node={location.resources.gpus_per_node}")
        return cmd

    # ---- batch submission ----

    directive_prefix = "#SBATCH"
    output_token = "%j"  # Slurm substitutes the job id into --output

    def resource_directives(self, location: Location, distributed: bool = False) -> list[str]:
        lines = [f"#SBATCH --nodes={location.resources.nodes}"]
        if location.resources.gpus_per_node:
            lines.append(f"#SBATCH --gpus-per-node={location.resources.gpus_per_node}")
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
