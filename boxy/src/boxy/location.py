"""The `location` abstraction: the target site / execution environment (the *where*).

Generalizes the `clusterB`/`clusterA` $CLUSTER switch from the paper prototype
(hpc-workflow/common_boxy.sh). The location selects the scheduler, container
runtime backend, accelerator, offline mode, staging paths, and site tuning.
"""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# RUNTIMES is the autodetect PREFERENCE order for resolve_runtime(); the set of
# VALID schedulers/runtimes is derived from the plugin registries at validation
# time (see __post_init__) so adding a backend/scheduler there is enough — no
# second list to keep in sync.
RUNTIMES = ("podman", "apptainer", "docker", "charliecloud")
ACCELERATORS = ("cuda", "rocm", "intel", "vulkan", "asahi", "ascend", "musa", "metal", "none")


@dataclass
class Resources:
    nodes: int = 1
    gpus_per_node: int = 0
    accelerator_type: str = ""  # e.g. "H100", "MI300"; used by the cloud/sky path
    distributed: bool | None = None  # multi-node Ray serving; None = auto (on for vllm+nodes>1)
    # per-GPU memory (GB) of THIS system's parts — the supply side of the card
    # geometry solver (a model card's min_vram_gb is the demand side). 0 = unknown;
    # the solver then assumes config cardgen.gpu_class_gb (80, A100/H100-class).
    gpu_vram_gb: int = 0
    # CLUSTER INVENTORY (from `boxy generate system`) — informational supply-side
    # facts, deliberately SEPARATE from `nodes` above, which is a JOB REQUEST.
    # total_nodes/total_gpu_nodes cap what the geometry solver may ever ask for.
    cpus_per_node: int = 0
    mem_gb_per_node: int = 0
    total_nodes: int = 0
    total_gpu_nodes: int = 0


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
    registry: str = ""  # send ALL images to this registry (see registries.py)
    # per-registry rewrite map ("docker.io" -> "registry.site.gov/dockerhub",
    # "*" catch-all) for site mirrors / air-gapped pulls; wins over `registry`.
    image_mirrors: dict[str, str] = field(default_factory=dict)
    offline: bool = False
    # Run boxy against this cluster FROM ANYWHERE: "user@login-node". When set
    # (and not already on the cluster), the CLI re-runs the same command on that
    # host over ONE multiplexed SSH session (OTP/YubiKey prompted once; see remote.py).
    remote: str = ""
    resources: Resources = field(default_factory=Resources)
    modules: list[str] = field(default_factory=list)
    staging: Staging = field(default_factory=Staging)
    # Raw site flags for batch submissions, in the scheduler's own spelling
    # (e.g. ["--partition=short", "--license=sitescratch:1", "--account=fyNNNNNN"]).
    scheduler_args: list[str] = field(default_factory=list)
    # Site quirks mapped to engine args, appended last unless the user
    # already set them (e.g. MI300a: gpu_memory_utilization = 0.7).
    tuning: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Derive the valid sets from the plugin registries (lazy import: this runs
        # at instance creation, when both modules are fully loaded — no cycle).
        from boxy.backends import BACKENDS
        from boxy.schedulers import SCHEDULERS
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"location {self.name}: unknown scheduler {self.scheduler!r} "
                             f"(expected {tuple(SCHEDULERS)})")
        if self.runtime and self.runtime not in BACKENDS:
            raise ValueError(f"location {self.name}: unknown runtime {self.runtime!r} "
                             f"(expected {tuple(BACKENDS)})")
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
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}") from None
        section = data.get("location")
        if section is None:
            raise ValueError(f"{path}: missing [location] section")
        try:
            resources = Resources(**section.pop("resources", {}))
            staging = Staging(**section.pop("staging", {}))
        except TypeError as e:
            raise ValueError(f"{path}: invalid [location.resources]/[location.staging]: {e}") from None
        modules = section.pop("modules", {})
        if isinstance(modules, str):
            # modules = "rocm/6.4.0" iterated as characters: 'module load r &&
            # module load o && ...' (finding 8)
            module_list = [modules]
        elif isinstance(modules, dict):
            module_list = modules.get("load", [])
        elif isinstance(modules, list):
            module_list = list(modules)
        else:
            raise ValueError(f"{path}: modules must be a list of module names")
        tuning = section.pop("tuning", {})
        if not isinstance(tuning, dict):
            raise ValueError(f"{path}: [location.tuning] must be a table of engine flags "
                             f"(optionally nested per engine: [location.tuning.vllm])")
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"resources", "staging", "modules", "tuning"}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"{path}: unknown [location] keys: {sorted(unknown)}")
        try:
            return cls(resources=resources, staging=staging, modules=module_list, tuning=tuning, **section)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: invalid [location] section: {e}") from None
