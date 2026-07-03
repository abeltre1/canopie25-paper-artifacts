"""Flux adapter: flux run launch prefix sized from the location's resources."""

from __future__ import annotations

from boxy.location import Location
from boxy.schedulers.base import Scheduler


class FluxScheduler(Scheduler):
    name = "flux"
    launcher = "flux"

    def launch_prefix(self, location: Location) -> list[str]:
        import shlex

        prefix = [self.launcher, "run", f"-N{location.resources.nodes}"]
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
        cmd = ["flux", "alloc", f"-N{location.resources.nodes}"]
        if location.resources.gpus_per_node:
            cmd.append(f"--gpus-per-node={location.resources.gpus_per_node}")
        return cmd

    # ---- batch submission ----

    directive_prefix = "#FLUX:"

    def resource_directives(self, location: Location) -> list[str]:
        lines = [f"#FLUX: -N{location.resources.nodes}"]
        if location.resources.gpus_per_node:
            lines.append(f"#FLUX: --gpus-per-node={location.resources.gpus_per_node}")
        return lines

    def site_directive(self, kind: str, value: str) -> str:
        # Flux spells the site knobs differently: queue not partition, bank
        # (flux-accounting) not account, -t not --time.
        return {"partition": f"--queue={value}",
                "account": f"--bank={value}",
                "time": f"-t {value}"}[kind]

    def submit_command(self, script: str) -> list[str]:
        return ["flux", "batch", script]

    def cancel_command(self, job_id: str) -> list[str]:
        return ["flux", "cancel", job_id]

    def state_command(self, job_id: str) -> list[str]:
        return ["flux", "jobs", "-n", "-o", "{state}", job_id]

    def interpret_state(self, stdout: str) -> str:
        state = stdout.strip().upper()
        if not state:
            return "DONE"
        if state in ("DEPEND", "PRIORITY", "SCHED"):
            return "PENDING"
        if state in ("RUN", "RUNNING", "CLEANUP"):
            return "RUNNING"
        if state == "INACTIVE":
            return "DONE"
        return "UNKNOWN"
