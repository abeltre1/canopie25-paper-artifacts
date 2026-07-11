"""The trivial exposer: a friendly name on THIS machine via /etc/hosts — no
daemon, no network, no cluster. It exists to prove the exposer plug point (and
as the honest floor of the family: everything above it buys wider reach)."""

from __future__ import annotations

from boxy.exposers.base import Exposer, ShareContext


class HostsExposer(Exposer):
    name = "hosts"
    binary = ""  # nothing external — always available

    def expose(self, alias: str, lport: int, ctx: ShareContext | None = None) -> tuple[str, str]:
        return (f"http://{alias}:{lport}/v1",
                f"add '127.0.0.1  {alias}' to /etc/hosts on THIS machine "
                f"(local-only; for a URL teammates can open use --exposer relay)")

    def unexpose(self, alias: str) -> None:
        return  # nothing was created
