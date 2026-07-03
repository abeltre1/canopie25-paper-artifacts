"""The `location` abstraction: the target site / execution environment (the *where*).

Generalizes the `hops`/`eldorado` $CLUSTER switch from the paper prototype
(hpc-workflow/common_boxy.sh). The location selects the scheduler, container
runtime backend, accelerator, offline mode, staging paths, and site tuning.
"""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCHEDULERS = ("slurm", "flux", "none")
RUNTIMES = ("podman", "apptainer", "docker")
ACCELERATORS = ("cuda", "rocm", "intel", "vulkan", "asahi", "ascend", "musa", "none")


@dataclass
class Resources:
    nodes: int = 1
    gpus_per_node: int = 0
    accelerator_type: str = ""  # e.g. "H100", "MI300"; used by the cloud/sky path


@dataclass
class Staging:
    models_dir: str = "./models"
    s3_endpoint: str = ""


@dataclass
class Location:
    name: str
    scheduler: str = "none"
    accelerator: str = ""  # "" => autodetect via ramalama get_accel()
    runtime: str = ""      # "" => autodetect first available backend
    registry: str = ""
    offline: bool = False
    resources: Resources = field(default_factory=Resources)
    modules: list[str] = field(default_factory=list)
    staging: Staging = field(default_factory=Staging)
    # Raw site flags for batch submissions, in the scheduler's own spelling
    # (e.g. ["--partition=short", "--license=tscratch:1", "--account=fy260064"]).
    scheduler_args: list[str] = field(default_factory=list)
    # Site quirks mapped to engine args, appended last unless the user
    # already set them (e.g. MI300a: gpu_memory_utilization = 0.7).
    tuning: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"location {self.name}: unknown scheduler {self.scheduler!r} (expected {SCHEDULERS})")
        if self.runtime and self.runtime not in RUNTIMES:
            raise ValueError(f"location {self.name}: unknown runtime {self.runtime!r} (expected {RUNTIMES})")
        if self.accelerator and self.accelerator not in ACCELERATORS:
            raise ValueError(
                f"location {self.name}: unknown accelerator {self.accelerator!r} (expected {ACCELERATORS})"
            )

    def resolve_accelerator(self) -> str:
        """Explicit accelerator wins; otherwise autodetect via RamaLama."""
        if self.accelerator:
            return self.accelerator
        from boxy import ramalama_shim

        return ramalama_shim.detect_accel()

    def resolve_runtime(self) -> str:
        """Explicit runtime wins; otherwise first available on this host."""
        if self.runtime:
            return self.runtime
        for candidate in RUNTIMES:
            if shutil.which(candidate):
                return candidate
        raise RuntimeError(
            f"location {self.name}: no container runtime found on host (looked for {', '.join(RUNTIMES)}); "
            "set [location].runtime explicitly"
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "Location":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get("location")
        if section is None:
            raise ValueError(f"{path}: missing [location] section")
        resources = Resources(**section.pop("resources", {}))
        staging = Staging(**section.pop("staging", {}))
        modules = section.pop("modules", {})
        module_list = modules.get("load", []) if isinstance(modules, dict) else list(modules)
        tuning = section.pop("tuning", {})
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"resources", "staging", "modules", "tuning"}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"{path}: unknown [location] keys: {sorted(unknown)}")
        return cls(resources=resources, staging=staging, modules=module_list, tuning=tuning, **section)
