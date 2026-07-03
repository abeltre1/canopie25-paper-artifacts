"""No scheduler: run directly on the current node (laptop, cloud VM, or an
already-allocated compute node — the paper's interactive salloc/flux-alloc flow)."""

from __future__ import annotations

import os

from boxy.schedulers.base import Scheduler


class NoScheduler(Scheduler):
    name = "none"

    def host_env_fixups(self) -> list[str]:
        # Even with no launcher prefix we may be *inside* an interactive
        # allocation; rootless podman needs the XDG session vars cleared there
        # (prototype: check_podman in common_boxy.sh).
        if os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_ENCLOSING_ID") or os.environ.get("FLUX_JOB_ID"):
            return ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]
        return []
