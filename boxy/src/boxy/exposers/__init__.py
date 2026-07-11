"""Pluggable exposers: turn boxy's laptop-loopback tunnel into a reachable URL
(RUNBOOK §0.993). Same registry shape as backends/ and schedulers/."""

from __future__ import annotations

from boxy.exposers.base import ExposeError, Exposer
from boxy.exposers.hosts import HostsExposer
from boxy.exposers.relay import RelayExposer

__all__ = ["EXPOSERS", "ExposeError", "Exposer", "get_exposer"]

EXPOSERS: dict[str, type[Exposer]] = {
    "relay": RelayExposer,
    "hosts": HostsExposer,
}


def get_exposer(name: str) -> Exposer:
    try:
        return EXPOSERS[name]()
    except KeyError:
        raise ValueError(f"unknown exposer {name!r} (available: {', '.join(EXPOSERS)})") from None
