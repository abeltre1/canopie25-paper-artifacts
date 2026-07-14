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
    # Extension of the FINAL path component only: '/data/models.gguf/x' is a
    # directory name, not a GGUF file (sweep finding 20).
    if model.lower().startswith("ollama://"):
        return True
    base = model.rstrip("/").rsplit("/", 1)[-1].lower()
    return base.endswith(".gguf")


def infer_engine(model: str, accelerator: str, gpus: int = 0) -> tuple[str, str]:
    """(engine, reason). GGUF/ollama -> llama.cpp. Otherwise vLLM, which needs a GPU
    (detected here, or requested via --gpus for a scheduler submission)."""
    if model.lower().startswith("ollama://"):
        return "llama.cpp", "ollama models are GGUF"
    if looks_like_gguf(model):
        return "llama.cpp", "model is GGUF"
    if accelerator in VLLM_ACCELS:
        return "vllm", f"safetensors/HF repo + {accelerator} GPU"
    if gpus > 0 and accelerator in VLLM_ACCELS + ("none",):
        # 'none' = submitting from a GPU-less node; a non-vLLM accelerator
        # (vulkan/asahi/...) must not sail into the CUDA-only vLLM image
        # (sweep finding 21).
        return "vllm", f"safetensors/HF repo + --gpus {gpus} requested for the job"
    base = model.rstrip("/").rsplit("/", 1)[-1]
    accel_note = ("no GPU was detected on this host" if accelerator == "none"
                  else f"this host's accelerator is {accelerator!r}, which vLLM's images do not "
                       f"support (supported: {', '.join(VLLM_ACCELS)})")
    raise RuntimeError(
        f"model {model!r} is a safetensors/HF repo: that needs vLLM, and {accel_note}.\n"
        f"  CPU serving here:  use a GGUF build with llama.cpp — search huggingface.co for\n"
        f"      '{base} GGUF' (publishers like TheBloke, bartowski, QuantFactory), then:\n"
        f"      boxy serve hf://<owner>/<repo>/<file>.gguf\n"
        f"      (on a wrong file name, boxy lists the repo's actual GGUF files)\n"
        f"      — or skip repo guessing entirely:  boxy serve ollama://<name>\n"
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


def _port_taken(port: int) -> bool:
    # Something answering on loopback? Catches listeners the bind test can
    # miss (e.g. forwarders bound to other interfaces).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return True
    # Wildcard bind WITHOUT SO_REUSEADDR: on macOS/BSD, SO_REUSEADDR lets a
    # 127.0.0.1 bind succeed while another process (podman-machine's gvproxy)
    # holds 0.0.0.0 on the same port — which is exactly how a "free" port
    # then dies with gvproxy's "proxy already running". (Field finding #18.)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
        except OSError:
            return True
    return False


def _free_port(preferred: int) -> tuple[int, str]:
    """Probe `preferred` and walk forward to the first free port (shared
    login/compute nodes collide on fixed defaults). Explicit --port skips this."""
    for offset in range(64):
        candidate = preferred + offset
        if _port_taken(candidate):
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
    sources: dict | None = None,
    distributed: bool | None = None,
) -> tuple[Location, list[str]]:
    """Resolve the *where* (used for bare-MODEL serves and for --box without
    --location). Explicit flags win; everything else is detected and explained.
    `sources` labels where a pinned value came from ("--flag" vs "location
    profile") so decision lines never claim false provenance (r2 audit)."""
    decisions: list[str] = []
    sources = sources or {}

    if scheduler is None:
        scheduler, why = detect_scheduler_context(here=here)
        decisions.append(f"scheduler: {scheduler} ({why})")
    else:
        decisions.append(f"scheduler: {scheduler} ({sources.get('scheduler', '--scheduler')})")

    if scheduler == "none" and (gpus > 0 or nodes > 1) and not here and distributed is not True:
        # ...unless this is distributed serving (a Ray set of containers) or the
        # inner compute-node serve (--here inside an allocation), where the
        # geometry drives tensor/pipeline parallelism.
        raise RuntimeError(
            "--gpus/--nodes describe a scheduler job request and have no effect without "
            "--scheduler slurm|flux (GPU pass-through itself follows the detected accelerator). "
            "For multi-node serving on this host, add --distributed."
        )
    if nodes > 1 and gpus > 0 and (distributed is True or scheduler == "none"):
        decisions.append(f"resources: {nodes} node(s) x {gpus} GPU(s)")

    if accelerator is not None:
        decisions.append(f"accelerator: {accelerator} ({sources.get('accelerator', '--accelerator')})")
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
        decisions.append(f"runtime: {runtime} ({sources.get('runtime', '--runtime')})")

    location = Location(
        name="auto",
        scheduler=scheduler,
        accelerator=accelerator,
        runtime=runtime,
        resources=Resources(nodes=nodes, gpus_per_node=gpus),
    )
    return location, decisions


def _classify_model(model: str, require_exists: bool) -> tuple[str, str]:
    """Syntax decides: a transport scheme means remote; anything else is a local
    path, full stop. Bare names are never guessed into registries — same
    command, same meaning, on every machine. Returns (resolved, decision)."""
    if model.lower().startswith("file://"):
        # RamaLama supports file:, but boxy's shared-FS flow IS a path —
        # strip the scheme instead of mangling it into a cwd-joined mount
        # (sweep finding 35).
        model = model[len("file://"):]
    scheme_split = model.split("://", 1)
    if model.lower().startswith("s3://"):
        # site-local S3 bucket: staged to the shared FS at serve time, then
        # served by path (not a RamaLama registry — origin policy N/A).
        return model, f"model: {model} (S3 bucket — staged to the shared filesystem)"
    if len(scheme_split) == 2 and scheme_split[0].lower() + "://" in TRANSPORT_SCHEMES:
        scheme, rest = scheme_split[0].lower(), scheme_split[1]
        if not rest.strip("/"):
            raise RuntimeError(
                f"malformed model URI {model!r}: nothing after the scheme "
                f"(an unset shell variable? e.g. hf://$ORG/$FILE)"
            )
        from boxy import policy

        policy.check_transport(f"{scheme}://{rest}")  # registry origin allowlist
        return f"{scheme}://{rest}", f"model: {scheme}://{rest} (transport URI — pulled via RamaLama)"
    if len(scheme_split) == 2:
        raise RuntimeError(
            f"unsupported model scheme {scheme_split[0]!r}:// — supported: "
            f"{', '.join(s[:-3] for s in TRANSPORT_SCHEMES)} (or a local path)"
        )
    resolved = os.path.abspath(model)
    if not os.path.exists(resolved):
        base = model.rsplit("/", 1)[-1]
        hint = (f"no such model file: {model!r}. MODEL is a local path or a transport URI — "
                f"did you mean ollama://{base} or hf://<org>/{base}?")
        if require_exists:
            raise RuntimeError(hint)
        return resolved, f"model: {resolved} (local path; NOT PRESENT — dryrun only. {hint})"
    return resolved, f"model: {resolved} (local file)"


def resolve_submission(
    model: str,
    scheduler: str,
    name: str | None = None,
    require_exists: bool = True,
) -> tuple[str, str, list[str]]:
    """Login-side resolution for a batch submission: classify the model and
    name the job. Hardware truths (accelerator/engine/image/port/runtime) are
    deliberately NOT resolved here — the inner `boxy serve` re-resolves them
    ON the compute node, where they are actually true (the design review's
    'wrong locus' fix). Returns (resolved_model, name, decisions)."""
    resolved_model, model_decision = _classify_model(model, require_exists)
    decisions = [
        model_decision,
        f"scheduler: {scheduler} (submitting a batch job — detaches once READY)",
        "accelerator/engine/image/port: resolved on the compute node at job start",
    ]
    return resolved_model, name or _slug(model), decisions


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
    sources: dict | None = None,
    distributed: bool | None = None,
) -> Resolution:
    decisions: list[str] = []
    resolved_model, model_decision = _classify_model(model, require_exists)
    decisions.append(model_decision)

    location, loc_decisions = auto_location(
        runtime=runtime, scheduler=scheduler, accelerator=accelerator, gpus=gpus, nodes=nodes,
        here=here, sources=sources, distributed=distributed
    )
    decisions += loc_decisions

    if engine is None:
        if (not resolved_model.startswith(TRANSPORT_SCHEMES) and not resolved_model.startswith("s3://")
                and not os.path.exists(resolved_model)):
            # dryrun with a missing path: don't assert facts about a file that
            # doesn't exist (finding 22) — plan as llama.cpp and say so
            engine = "llama.cpp"
            decisions.append("engine: llama.cpp (assumed — model file not present)")
        else:
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

    # Model-card engine args (e.g. max_model_len for the 70B cards) merge in at
    # box.args level — the tack-on-last rule still lets explicit user args win.
    # Only REAL cards contribute args (the size heuristic knows geometry, not
    # flags). This runs wherever the box is actually built — locally AND on the
    # compute node of a batch job (same wheel, same packaged cards) — so card
    # args need no extra flag plumbing through the scheduler.
    from boxy import cards as _cards

    card = _cards.find_card(resolved_model)
    card_args: dict = {}
    if card and card.args:
        card_args = dict(card.args)
        kv = ", ".join(f"{k}={v}" for k, v in card_args.items())
        decisions.append(f"engine args: {kv} ({card.label})")

    box_name = name or _slug(model)
    box = Box(
        name=box_name,
        image=resolved_image,  # "" -> deploy fills in default_image (same map as preview)
        engine=engine,
        model=resolved_model,
        ports=[resolved_port],
        args=card_args,
    )
    return Resolution(box=box, location=location, decisions=decisions)
