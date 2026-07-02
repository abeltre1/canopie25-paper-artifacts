"""Inner-command builders: the command that runs *inside* the container.

boxy builds these itself (pure functions) rather than calling RamaLama's
VllmPlugin._cmd_serve, because that builder is impure — it resolves the model
through the RamaLama store, which the paper's shared-filesystem flow doesn't
use. For store-pulled models the resolved path flows in the same way.
"""

from __future__ import annotations

from boxy.box import Box
from boxy.location import Location


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _tack_on_last(cmd: list[str], extra: dict[str, object]) -> list[str]:
    """Append args unless the user already set them (prototype rule from
    common_boxy.sh: 'If user has already set any of these args, don't tack
    them on (don't override the user).')."""
    present = {a.split("=", 1)[0] for a in cmd if a.startswith("--")}
    for key, value in extra.items():
        flag = _flag(str(key))
        if flag in present:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.append(f"{flag}={value}")
    return cmd


def build_vllm_serve_cmd(
    box: Box,
    location: Location,
    model_path: str,
    host: str = "0.0.0.0",
    port: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """`vllm serve <model> ...` argv, with box.args then location.tuning
    tacked on last (user-supplied args always win)."""
    entrypoint = box.entrypoint or "vllm"
    cmd = [entrypoint, "serve", model_path]
    cmd += list(extra_args or [])
    resolved_port = port or (box.ports[0] if box.ports else 8000)
    cmd = _tack_on_last(cmd, {"host": host, "port": resolved_port})
    cmd = _tack_on_last(cmd, box.args)
    cmd = _tack_on_last(cmd, location.tuning)
    return cmd


def build_raw_cmd(box: Box, user_args: list[str], location: Location) -> list[str]:
    """`boxy run` passthrough: entrypoint + user args + tack-ons, mirroring
    the prototype's boxy-run-vllm.sh \"$@\" behavior."""
    cmd = ([box.entrypoint] if box.entrypoint else []) + list(user_args)
    cmd = _tack_on_last(cmd, box.args)
    cmd = _tack_on_last(cmd, location.tuning)
    return cmd
