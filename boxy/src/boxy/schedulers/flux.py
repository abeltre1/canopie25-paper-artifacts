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

    # Flux's directive sentinel is the literal `flux:`; the leading `#` is just
    # the script's comment syntax (RFC 36). It is CASE-SENSITIVE lowercase —
    # `#FLUX:` is an ordinary comment flux silently ignores, so directives
    # written that way never take effect (the job lands with default resources
    # and default queue). Field report: `--scheduler flux --gpus 4` produced a
    # job with no GPUs because every directive was dropped.
    directive_prefix = "# flux:"
    output_token = "{{id}}"  # Flux substitutes the job id (mustache) into --output

    def resource_directives(self, location: Location, distributed: bool = False) -> list[str]:
        # `flux batch` does not launch tasks, so it speaks SLOTS, not the
        # per-node GPU spelling that `flux run`/`flux alloc` accept. GPUs are
        # requested with -g/--gpus-per-slot; map "nodes x gpus_per_node" onto
        # one slot per node (-N nodes, -n nodes) each carrying the GPUs
        # (-g gpus_per_node). `--gpus-per-node` is NOT a flux-batch option.
        r = location.resources
        lines = [f"# flux: -N{r.nodes}"]
        if r.gpus_per_node:
            lines.append(f"# flux: -n{r.nodes}")
            lines.append(f"# flux: -g{r.gpus_per_node}")
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
