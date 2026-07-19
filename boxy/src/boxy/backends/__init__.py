"""Pluggable container runtime backends (SPEC §4b)."""

from __future__ import annotations

from boxy.backends.apptainer import ApptainerBackend
from boxy.backends.base import RuntimeBackend
from boxy.backends.charliecloud import CharlieCloudBackend
from boxy.backends.docker import DockerBackend
from boxy.backends.podman import PodmanBackend

BACKENDS: dict[str, type[RuntimeBackend]] = {
    "podman": PodmanBackend,
    "apptainer": ApptainerBackend,
    "docker": DockerBackend,
    "charliecloud": CharlieCloudBackend,  # experimental (ch-run; golden-tested)
}


def get_backend(name: str) -> RuntimeBackend:
    try:
        return BACKENDS[name]()
    except KeyError:
        raise ValueError(f"unknown runtime backend {name!r} (available: {', '.join(BACKENDS)})") from None
