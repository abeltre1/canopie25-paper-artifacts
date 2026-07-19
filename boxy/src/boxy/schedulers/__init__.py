"""Scheduler adapters: wrap a container command for the site's batch system."""

from __future__ import annotations

from boxy.schedulers.base import Scheduler
from boxy.schedulers.flux import FluxScheduler
from boxy.schedulers.none_ import NoScheduler
from boxy.schedulers.slurm import SlurmScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    "none": NoScheduler,
    "slurm": SlurmScheduler,
    "flux": FluxScheduler,
}


def get_scheduler(name: str) -> Scheduler:
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise ValueError(f"unknown scheduler {name!r} (available: {', '.join(SCHEDULERS)})") from None
