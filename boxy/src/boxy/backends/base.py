"""RuntimeBackend contract: build the run command, inject env, map GPU
devices, mount volumes — so a `box` stays runtime-agnostic and the
`location` merely selects the backend (SPEC §4b)."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod

from boxy.box import Box
from boxy.location import Location


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
