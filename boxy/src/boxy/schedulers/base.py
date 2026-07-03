"""Scheduler contract: submit to Slurm/Flux, never reimplement scheduling."""

from __future__ import annotations

import shlex
from abc import ABC

from boxy.location import Location


class Scheduler(ABC):
    name: str = ""
    launcher: str = ""  # binary that prefixes the command (srun/flux); "" = none

    def wrap(self, cmd: list[str], location: Location) -> list[str]:
        """Prefix `cmd` with the scheduler launcher for this location."""
        return self.launch_prefix(location) + cmd

    def launch_prefix(self, location: Location) -> list[str]:
        return []

    def host_env_fixups(self) -> list[str]:
        """Env vars to *unset* before launch (prototype: XDG vars break
        rootless podman inside interactive Slurm/Flux jobs)."""
        return []

    def with_modules(self, cmd: list[str], location: Location) -> list[str]:
        """Wrap with `module load ...` when the location requires modules
        (e.g. rocm/6.4.0 before Apptainer --rocm)."""
        if not location.modules:
            return cmd
        loads = " && ".join(f"module load {m}" for m in location.modules)
        return ["bash", "-lc", f"{loads} && exec {shlex.join(cmd)}"]

    # ---- batch submission (the seamless path: submit, detach, rendezvous) ----

    directive_prefix: str = ""  # "#SBATCH" / "#FLUX:"

    def resource_directives(self, location: Location) -> list[str]:
        """Scheduler-flag directive lines for the job request (nodes/gpus)."""
        raise NotImplementedError

    def site_directive(self, kind: str, value: str) -> str:
        """Map a generic site knob (partition/account/time) to this
        scheduler's flag spelling."""
        raise NotImplementedError

    def dynamic_directive(self, key: str, value: str | None) -> str:
        """Translate a pass-through flag (--slurm-KEY=VALUE / --flux-KEY=VALUE)
        into this scheduler's flag spelling. Any flag the scheduler grows is
        immediately usable — boxy never needs a code change for a new one."""
        if len(key) == 1:
            return f"-{key} {value}" if value is not None else f"-{key}"
        return f"--{key}={value}" if value is not None else f"--{key}"

    def batch_script(self, inner_command: str, location: Location, name: str,
                     log_file: str, site_args: list[str]) -> str:
        """A complete batch script: directives + module loads + exec inner.
        `site_args` are raw scheduler flags in this scheduler's spelling
        (e.g. --license=tscratch:1)."""
        lines = ["#!/bin/bash"]
        lines.append(f"{self.directive_prefix} --job-name={name}")
        lines += self.resource_directives(location)
        for arg in site_args:
            # sbatch's directive parser splits on whitespace unless quoted:
            # --comment=hello world  ->  'Invalid directive: world' (r2 audit)
            if "=" in arg and any(c.isspace() for c in arg.split("=", 1)[1]):
                flag, value = arg.split("=", 1)
                arg = f'{flag}="{value.replace(chr(34), chr(92) + chr(34))}"'
            lines.append(f"{self.directive_prefix} {arg}")
        lines.append(f"{self.directive_prefix} --output={log_file}")
        lines.append("")
        for module in location.modules:
            lines.append(f"module load {module}")
        lines.append(f"exec {inner_command}")
        return "\n".join(lines) + "\n"

    def submit_command(self, script: str) -> list[str]:
        raise NotImplementedError

    def parse_job_id(self, submit_stdout: str) -> str:
        return submit_stdout.strip().splitlines()[-1].strip() if submit_stdout.strip() else ""

    def cancel_command(self, job_id: str) -> list[str]:
        raise NotImplementedError

    def state_command(self, job_id: str) -> list[str]:
        raise NotImplementedError

    def interpret_state(self, stdout: str) -> str:
        """Normalize scheduler output to PENDING | RUNNING | DONE | UNKNOWN."""
        raise NotImplementedError
