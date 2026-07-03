"""v2 auto-resolution: turn a bare MODEL argument into a runnable Box+Location.

This is the RamaLama-style front door: `boxy serve <model>` with zero files.
Every decision made here is (a) printed so the user sees what was chosen,
(b) overridable by a flag, and (c) snapshottable to TOML for reproducibility.

Design rules (from the v2 design review):
  * syntax decides what a MODEL is (scheme => remote, else local path) —
    filesystem state never changes the meaning of a command;
  * a scheduler is NEVER auto-wrapped; on a login node boxy refuses to run
    an LLM server unless --here is given;
  * runtimes are probed for viability (`podman info`), not just PATH presence;
  * default ports are bind-tested and advanced when busy; explicit --port wins.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass

from boxy import ramalama_shim
from boxy.box import TRANSPORT_SCHEMES, Box
from boxy.location import Location, Resources

# Accelerators that can run vLLM (a default image exists for each).
VLLM_ACCELS = ("cuda", "rocm", "intel")


@dataclass
class Resolution:
    box: Box
    location: Location
    decisions: list[str]  # human-readable "chose X because Y" lines


def _slug(model: str) -> str:
    base = model.rsplit("/", 1)[-1]
    base = re.sub(r"\.(gguf|safetensors)$", "", base, flags=re.I)
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-").lower() or "model"
    return f"boxy-{slug}"[:63]


def looks_like_gguf(model: str) -> bool:
    # ollama registries serve GGUF blobs; treat the transport as GGUF.
    return ".gguf" in model.lower() or model.startswith("ollama://")


def infer_engine(model: str, accelerator: str, gpus: int = 0) -> tuple[str, str]:
    """(engine, reason). GGUF/ollama -> llama.cpp. Otherwise vLLM, which needs a GPU
    (detected here, or requested via --gpus for a scheduler submission)."""
    if model.startswith("ollama://"):
        return "llama.cpp", "ollama models are GGUF"
    if looks_like_gguf(model):
        return "llama.cpp", "model is GGUF"
    if accelerator in VLLM_ACCELS:
        return "vllm", f"safetensors/HF repo + {accelerator} GPU"
    if gpus > 0:
        return "vllm", f"safetensors/HF repo + --gpus {gpus} requested for the job"
    base = model.rstrip("/").rsplit("/", 1)[-1]
    raise RuntimeError(
        f"model {model!r} is a safetensors/HF repo: that needs vLLM and a GPU, and none was "
        f"detected on this host.\n"
        f"  CPU serving here:  use a GGUF build with llama.cpp — community quants are usually at\n"
        f"      boxy serve hf://bartowski/{base}-GGUF/{base}-Q4_K_M.gguf\n"
        f"      (verify the exact repo/file on huggingface.co, or use ollama://<name>)\n"
        f"  GPU cluster:       this command works as-is on a GPU node; from a login node submit it:\n"
        f"      boxy serve {model} --scheduler slurm|flux --gpus N --accelerator cuda|rocm\n"
        f"  Or override:       --engine/--accelerator if detection is wrong on this host."
    )


# Presence on PATH is not viability: HPC nodes routinely carry a podman binary
# with no subuid ranges or with containers-storage on NFS, and docker binaries
# whose daemon the user cannot reach. Probe before committing.
_PROBES: dict[str, list[str]] = {
    "podman": ["info", "--format", "{{.Host.Arch}}"],
    "docker": ["info", "--format", "{{.ServerVersion}}"],
    "apptainer": ["version"],
}


def _runtime_works(candidate: str, timeout_s: float = 10.0) -> bool:
    try:
        subprocess.run(
            [candidate] + _PROBES[candidate],
            capture_output=True,
            timeout=timeout_s,
            check=True,
        )
        return True
    except Exception:
        return False


def detect_runtime() -> tuple[str, str]:
    skipped: list[str] = []
    for candidate in ("podman", "docker", "apptainer"):
        if not shutil.which(candidate):
            continue
        if not _runtime_works(candidate):
            skipped.append(f"{candidate} is on PATH but its probe failed")
            continue
        note = f"{candidate} found on PATH and responding"
        if skipped:
            note += f" ({'; '.join(skipped)})"
        return candidate, note
    if skipped:
        raise RuntimeError(
            "no working container runtime: " + "; ".join(skipped) + ". "
            "Rootless podman needs /etc/subuid entries and non-NFS storage; "
            "docker needs a reachable daemon; on HPC try `module load apptainer`."
        )
    raise RuntimeError(
        "no container runtime found (looked for podman, docker, apptainer); "
        "load your site's container module (e.g. `module load apptainer`) or install one"
    )


def detect_scheduler_context(here: bool = False) -> tuple[str, str]:
    """v2 default: NEVER auto-wrap with a scheduler. Inside an allocation, run
    direct (the job step owns the server). On a login node (scheduler CLI on
    PATH, no allocation env), REFUSE unless --here: an LLM server on a shared
    login node is the canonical acceptable-use violation."""
    if os.environ.get("SLURM_JOB_ID"):
        return "none", f"inside Slurm allocation {os.environ['SLURM_JOB_ID']} — running direct on this node"
    if os.environ.get("FLUX_ENCLOSING_ID") or os.environ.get("FLUX_JOB_ID"):
        return "none", "inside Flux allocation — running direct on this node"
    for probe, sched, alloc_hint in (
        ("srun", "slurm", "srun -N1 --gpus-per-node=1 --pty bash"),
        ("flux", "flux", "flux run -N1 -g1"),
    ):
        if shutil.which(probe):
            if here:
                return "none", f"{sched} present but --here given — running direct on THIS node"
            raise RuntimeError(
                f"this looks like a {sched} login node ({probe} on PATH, no allocation env). "
                f"Serving an LLM here is usually against site policy. Either:\n"
                f"  submit as a job:         add --scheduler {sched} --gpus N [--accelerator cuda|rocm]\n"
                f"  serve inside an alloc:   {alloc_hint}   then rerun boxy serve\n"
                f"  force this node anyway:  add --here"
            )
    return "none", "no scheduler on host"


def _free_port(preferred: int) -> tuple[int, str]:
    """Bind-test `preferred` and walk forward to the first free port (shared
    login/compute nodes collide on fixed defaults). Explicit --port skips this."""
    for offset in range(64):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", candidate))
            except OSError:
                continue
            return candidate, "" if offset == 0 else f"; {preferred} was busy"
    raise RuntimeError(f"no free port found in [{preferred}, {preferred + 63}]; pass --port explicitly")


def auto_location(
    runtime: str | None = None,
    scheduler: str | None = None,
    accelerator: str | None = None,
    gpus: int = 0,
    nodes: int = 1,
    here: bool = False,
) -> tuple[Location, list[str]]:
    """Resolve the *where* (used for bare-MODEL serves and for --box without
    --location). Explicit flags win; everything else is detected and explained."""
    decisions: list[str] = []

    if scheduler is None:
        scheduler, why = detect_scheduler_context(here=here)
        decisions.append(f"scheduler: {scheduler} ({why})")
    else:
        decisions.append(f"scheduler: {scheduler} (--scheduler)")

    if scheduler == "none" and (gpus > 0 or nodes > 1):
        raise RuntimeError(
            "--gpus/--nodes describe a scheduler job request and have no effect without "
            "--scheduler slurm|flux (GPU pass-through itself follows the detected accelerator)"
        )

    if accelerator is not None:
        decisions.append(f"accelerator: {accelerator} (--accelerator)")
    else:
        accelerator = ramalama_shim.detect_accel()
        if scheduler in ("slurm", "flux"):
            if accelerator == "none" and gpus > 0:
                raise RuntimeError(
                    "submitting a GPU job from a node with no detectable GPU: boxy cannot guess "
                    "the compute node's accelerator from here. Pass --accelerator cuda|rocm "
                    "(or use a --location site profile, which pins it)."
                )
            decisions.append(
                f"accelerator: {accelerator} (detected on THIS node — the compute node may differ; "
                f"pass --accelerator or a --location profile to pin it)"
            )
        else:
            decisions.append(f"accelerator: {accelerator} (autodetected)")

    if runtime is None:
        runtime, why = detect_runtime()
        decisions.append(f"runtime: {runtime} ({why})")
    else:
        decisions.append(f"runtime: {runtime} (--runtime)")

    location = Location(
        name="auto",
        scheduler=scheduler,
        accelerator=accelerator,
        runtime=runtime,
        resources=Resources(nodes=nodes, gpus_per_node=gpus),
    )
    return location, decisions


def resolve(
    model: str,
    engine: str | None = None,
    runtime: str | None = None,
    scheduler: str | None = None,
    image: str | None = None,
    port: int | None = None,
    gpus: int = 0,
    nodes: int = 1,
    name: str | None = None,
    accelerator: str | None = None,
    here: bool = False,
    require_exists: bool = True,
) -> Resolution:
    decisions: list[str] = []

    # Syntax decides: a transport scheme means remote; anything else is a local
    # path, full stop. Bare names are never guessed into registries — same
    # command, same meaning, on every machine.
    if model.startswith(TRANSPORT_SCHEMES):
        resolved_model = model
        decisions.append(f"model: {model} (transport URI — pulled via RamaLama)")
    else:
        resolved_model = os.path.abspath(model)
        if not os.path.exists(resolved_model):
            if require_exists:
                base = model.rsplit("/", 1)[-1]
                raise RuntimeError(
                    f"no such model file: {model!r}. MODEL is a local path or a transport URI — "
                    f"did you mean ollama://{base} or hf://<org>/{base}?"
                )
            decisions.append(f"model: {resolved_model} (local path; not present — dryrun)")
        else:
            decisions.append(f"model: {resolved_model} (local file)")

    location, loc_decisions = auto_location(
        runtime=runtime, scheduler=scheduler, accelerator=accelerator, gpus=gpus, nodes=nodes, here=here
    )
    decisions += loc_decisions

    if engine is None:
        engine, why = infer_engine(model, location.accelerator, gpus=gpus)
        decisions.append(f"engine: {engine} ({why})")
    else:
        decisions.append(f"engine: {engine} (--engine)")

    resolved_image = image or ""
    if image:
        decisions.append(f"image: {image} (--image)")
    else:
        preview = ramalama_shim.default_image(engine, location.accelerator)
        decisions.append(f"image: {preview} (default for {engine}+{location.accelerator})")

    if port is not None:
        resolved_port = port
        decisions.append(f"port: {resolved_port} (--port)")
    else:
        default_port = 8000 if engine == "vllm" else 8090
        if location.scheduler == "none":
            resolved_port, busy_note = _free_port(default_port)
            decisions.append(f"port: {resolved_port} ({engine} default{busy_note})")
        else:
            # bind-testing the submitting node says nothing about the compute node
            resolved_port = default_port
            decisions.append(f"port: {resolved_port} ({engine} default; on the compute node)")

    box_name = name or _slug(model)
    box = Box(
        name=box_name,
        image=resolved_image,  # "" -> deploy fills in default_image (same map as preview)
        engine=engine,
        model=resolved_model,
        ports=[resolved_port],
    )
    return Resolution(box=box, location=location, decisions=decisions)
