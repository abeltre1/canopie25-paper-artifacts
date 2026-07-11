"""Pluggable exposers: turn boxy's laptop-loopback tunnel into a reachable URL
(RUNBOOK §0.993/§0.994). Same registry shape as backends/ and schedulers/.

Members, widest-trust first:
  gateway — OpenSSH-only pod that dials the login node itself (no third-party
            tunnel binary; the default — cyber-friendly).
  relay   — OpenShift chisel reverse-tunnel relay.
  hosts   — a this-machine /etc/hosts name (local-only; proves the plug point).
"""

from __future__ import annotations

from boxy.exposers.base import ExposeError, Exposer, ShareContext
from boxy.exposers.gateway import GatewayExposer
from boxy.exposers.hosts import HostsExposer
from boxy.exposers.relay import RelayExposer

__all__ = ["EXPOSERS", "ExposeError", "Exposer", "ShareContext", "get_exposer"]

EXPOSERS: dict[str, type[Exposer]] = {
    "gateway": GatewayExposer,
    "relay": RelayExposer,
    "hosts": HostsExposer,
}


def get_exposer(name: str) -> Exposer:
    try:
        return EXPOSERS[name]()
    except KeyError:
        raise ValueError(f"unknown exposer {name!r} (available: {', '.join(EXPOSERS)})") from None
