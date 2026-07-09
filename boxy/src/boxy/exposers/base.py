"""Exposer contract: turn a live LOCAL loopback tunnel (boxy's `ssh -L` end at
127.0.0.1:<lport>) into a URL other people can reach — a separate, pluggable
component layered on top of the SSH machinery, mirroring the RuntimeBackend /
Scheduler registries. Members: `relay` (OpenShift chisel relay — a shared
corporate URL under the cluster's wildcard DNS) and `hosts` (a this-machine
/etc/hosts name; the trivial member proving the plug point).

The exposer NEVER owns the tunnel: it attaches after boxy's forward is live and
its failure must never take the tunnel down (callers degrade to the Tier-1
`--route` behavior)."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod


class Exposer(ABC):
    name: str = ""
    binary: str = ""  # external tool probed by available(); "" -> no tool needed

    def available(self) -> bool:
        return not self.binary or shutil.which(self.binary) is not None

    @abstractmethod
    def expose(self, alias: str, lport: int) -> tuple[str, str]:
        """Make 127.0.0.1:<lport> reachable under `alias`; return (url, note).
        Raise ExposeError with a user-actionable message on failure."""

    @abstractmethod
    def unexpose(self, alias: str) -> None:
        """Tear down whatever expose() created for `alias` (idempotent)."""


class ExposeError(RuntimeError):
    """Expose failed — message is printed as a warning; the tunnel lives on."""
