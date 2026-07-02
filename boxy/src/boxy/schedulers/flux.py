"""Flux adapter: flux run launch prefix sized from the location's resources."""

from __future__ import annotations

from boxy.location import Location
from boxy.schedulers.base import Scheduler


class FluxScheduler(Scheduler):
    name = "flux"
    launcher = "flux"

    def launch_prefix(self, location: Location) -> list[str]:
        prefix = [self.launcher, "run", f"-N{location.resources.nodes}"]
        if location.resources.gpus_per_node:
            prefix.append(f"--gpus-per-node={location.resources.gpus_per_node}")
        return prefix

    def host_env_fixups(self) -> list[str]:
        return ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]

    def alloc_command(self, location: Location) -> list[str]:
        cmd = ["flux", "alloc", f"-N{location.resources.nodes}"]
        if location.resources.gpus_per_node:
            cmd.append(f"--gpus-per-node={location.resources.gpus_per_node}")
        return cmd
