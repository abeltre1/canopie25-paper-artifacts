"""The `box` abstraction: the containerized application/service (the *what*).

A box is runtime- and accelerator-agnostic: it never names Podman, Apptainer,
CUDA, or ROCm. Those come from the `location` (see location.py).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

TRANSPORT_SCHEMES = ("hf://", "huggingface://", "ollama://", "oci://", "ms://", "modelscope://", "rlcr://")


@dataclass
class Volume:
    source: str
    target: str
    options: str = ""


ENGINES = ("vllm", "llama.cpp")


@dataclass
class Box:
    name: str
    image: str = ""  # "" => default image for engine+accelerator (RamaLama-informed map)
    engine: str = "vllm"  # inference engine inside the box: vllm | llama.cpp
    entrypoint: str = ""
    model: str = ""
    workdir: str = ""
    ports: list[int] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    volumes: list[Volume] = field(default_factory=list)
    # Engine args appended last, without overriding user-supplied args
    # (the prototype's "tack on last" rule from common_boxy.sh).
    args: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"box {self.name}: unknown engine {self.engine!r} (expected {ENGINES})")
        # ports = "8090" (a string) would serve on port 8 via ports[0] (finding 1)
        if (not isinstance(self.ports, list)
                or not all(isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= 65535
                           for p in self.ports)):
            # bool IS an int in Python: ports=[true] emitted a dangling --port (r2)
            raise ValueError(f"box {self.name}: ports must be a list of integers in 1-65535, "
                             f"e.g. ports = [8090]")
        if not isinstance(self.args, dict):
            raise ValueError(f"box {self.name}: [box.args] must be a table of engine flags")

    @property
    def model_is_transport_uri(self) -> bool:
        return self.model.startswith(TRANSPORT_SCHEMES)

    @classmethod
    def from_toml(cls, path: str | Path) -> "Box":
        # every loader error names the file: with --box AND --location the
        # user must know WHICH TOML is broken (findings 6/7)
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}") from None
        section = data.get("box")
        if section is None:
            raise ValueError(f"{path}: missing [box] section")
        try:
            volumes = [Volume(**v) for v in section.pop("volumes", [])]
            env = {k: str(v) for k, v in section.pop("env", {}).items()}
        except (TypeError, AttributeError) as e:
            raise ValueError(f"{path}: invalid [box] sub-table "
                             f"([[box.volumes]] needs source+target; [box.env] is a table): {e}") from None
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"volumes", "env"}
        unknown = set(section) - known
        if unknown:
            raise ValueError(f"{path}: unknown [box] keys: {sorted(unknown)}")
        try:
            return cls(volumes=volumes, env=env, **section)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path}: invalid [box] section: {e}") from None
