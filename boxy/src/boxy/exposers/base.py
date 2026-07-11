"""Exposer contract: turn a live LOCAL loopback tunnel (boxy's `ssh -L` end at
127.0.0.1:<lport>) into a URL other people can reach — a separate, pluggable
component layered on top of the SSH machinery, mirroring the RuntimeBackend /
Scheduler registries. Members: `gateway` (an OpenSSH-only pod that dials the
login node itself — no third-party tunnel binary, the default), `relay` (an
OpenShift chisel relay) and `hosts` (a this-machine /etc/hosts name; the trivial
member proving the plug point).

The exposer NEVER owns the tunnel: it attaches after boxy's forward is live and
its failure must never take the tunnel down (callers degrade to the Tier-1
`--route` behavior)."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ShareContext:
    """Everything an exposer might need beyond the local port. The `relay`/`hosts`
    members bridge the LAPTOP loopback and ignore all of this; the `gateway`
    member instead reconstructs the model's real address on the cluster
    (`ssh -L :<remote_port>:<node>:<remote_port>` from a pod to the login node),
    so it needs the login target, compute node, and remote port that boxy only
    learns once the job is READY."""

    ssh_host: str = ""      # the user's --ssh login target, e.g. "ambelt@hops"
    node: str = ""          # the compute node the job landed on, e.g. "hops18"
    remote_port: int = 0    # the model's port on that node, e.g. 8090


class Exposer(ABC):
    name: str = ""
    binary: str = ""  # external tool probed by available(); "" -> no tool needed

    def available(self) -> bool:
        return not self.binary or shutil.which(self.binary) is not None

    @abstractmethod
    def expose(self, alias: str, lport: int, ctx: ShareContext | None = None) -> tuple[str, str]:
        """Make the served model reachable under `alias`; return (url, note).
        `lport` is boxy's laptop-loopback forward; `ctx` carries the cluster-side
        address for exposers that don't route through the laptop. Raise
        ExposeError with a user-actionable message on failure."""

    @abstractmethod
    def unexpose(self, alias: str) -> None:
        """Tear down whatever expose() created for `alias` (idempotent)."""

    def is_live(self, record: dict) -> bool:
        """Is the share created from `record` still up? (For `boxy list`.)
        Default: assume live — members with a checkable liveness override."""
        return True


class ExposeError(RuntimeError):
    """Expose failed — message is printed as a warning; the tunnel lives on."""
