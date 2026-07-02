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
