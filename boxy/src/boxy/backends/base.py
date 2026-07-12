"""RuntimeBackend contract: build the run command, inject env, map GPU
devices, mount volumes — so a `box` stays runtime-agnostic and the
`location` merely selects the backend (SPEC §4b)."""

from __future__ import annotations

import shutil
import sys
from abc import ABC, abstractmethod

from boxy.box import Box
from boxy.location import Location

# Accelerators whose GPU is reachable through the DRI render nodes: Intel oneAPI,
# Vulkan (any GPU via the Vulkan ICD), and Apple Silicon under Linux/Asahi. The
# device path is the same; only the flag spelling differs per runtime
# (podman/docker: --device /dev/dri; apptainer: --bind /dev/dri).
DRI_ACCELERATORS = ("intel", "vulkan", "asahi")


def selinux_enforcing() -> bool:
    """True on an SELinux-enforcing host (RHEL/Fedora HPC nodes). Reading the
    kernel's enforce node is cheaper and more reliable than shelling `getenforce`;
    absent (Ubuntu/macOS) or unreadable => not enforcing."""
    try:
        with open("/sys/fs/selinux/enforce") as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def relabel_option(options: str, mode: str, enforcing: bool) -> str:
    """Return `options` with an SELinux ':z' relabel appended when appropriate, so
    rootless podman bind mounts don't hit 'permission denied' on enforcing hosts.
    mode: auto (only when `enforcing`) | always | never. No-op if the user already
    set a z/Z label. Pure — the caller supplies `enforcing` (testable without a
    real SELinux host)."""
    opts = [o for o in options.split(",") if o]
    if any(o in ("z", "Z") for o in opts):
        return options
    if mode == "always" or (mode == "auto" and enforcing):
        opts.append("z")
    return ",".join(opts)


def warn_cpu_only(accelerator: str, backend: str) -> None:
    """A known accelerator with no container device pass-through boxy can wire
    would otherwise run silently on CPU and burn the allocation — say so."""
    print(f"boxy: warning: accelerator {accelerator!r}: no known container device "
          f"pass-through for {backend}; the container will run CPU-only. Wire devices "
          f"yourself via [[box.volumes]]/box args if your site needs them.",
          file=sys.stderr)


class RuntimeBackend(ABC):
    name: str = ""
    image_format: str = "oci"  # oci | sif

    def available(self) -> bool:
        return shutil.which(self.name) is not None

    def prepare(self, box: Box, location: Location, dryrun: bool = False,
                accelerator: str | None = None) -> list[list[str]]:
        """Commands to run before launch (e.g. OCI->SIF build). `accelerator`
        is the RESOLVED accelerator - the same one build_command receives, so
        artifacts named per-accelerator (SIFs) match between build and run.
        Default: none."""
        return []

    @abstractmethod
    def build_command(
        self,
        box: Box,
        location: Location,
        inner_cmd: list[str],
        env: dict[str, str],
        mounts: list[tuple[str, str, str]],  # (source, target, options)
        accelerator: str,
    ) -> list[str]:
        """Full host argv that runs `inner_cmd` inside the box's container."""

    def image_ref(self, box: Box, location: Location) -> str:
        """Every image reference resolves through registries.py (site mirrors,
        --registry, localhost) — swap registries there, never per-backend."""
        from boxy import registries

        return registries.resolve_image(box.image, location.registry, location.image_mirrors)
