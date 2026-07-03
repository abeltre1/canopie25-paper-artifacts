"""Compose a box + location into a runnable host command.

Pipeline:  resolve accel/runtime -> env merge -> model & mounts ->
           inner cmd -> backend command -> module preamble -> scheduler wrap
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath

from boxy import engines, envs, ramalama_shim
from boxy.backends import get_backend
from boxy.backends.base import RuntimeBackend
from boxy.box import Box
from boxy.location import Location
from boxy.schedulers import get_scheduler
from boxy.schedulers.base import Scheduler

STORE_MOUNT = "/mnt/models"


@dataclass
class Deployment:
    box: Box
    location: Location
    accelerator: str
    backend: RuntimeBackend
    scheduler: Scheduler
    command: list[str]
    prepare_commands: list[list[str]]
    env_unset: list[str]
    warnings: list[str] = field(default_factory=list)
    # THE serving port: parsed from the actual inner command (user --port after
    # `--` wins there), so banners/probes/publishing never disagree with the
    # server. 0 for run-mode deployments with no port. (Sweep findings 2/10/25.)
    port: int = 0


def _expand(value: str, location: Location) -> str:
    value = value.replace("${MODELS_DIR}", location.staging.models_dir)
    # Docker rejects relative bind-mount sources, and a bare relative source
    # ('models') silently becomes an empty NAMED volume (sweep finding 60):
    # absolutize every non-absolute source against the CWD, mirroring how
    # the prototype was always run from the workflow directory.
    if not os.path.isabs(value):
        value = os.path.abspath(value)
    return value


def resolve_mounts(box: Box, location: Location) -> list[tuple[str, str, str]]:
    return [(_expand(v.source, location), v.target, v.options) for v in box.volumes]


def resolve_model(box: Box, location: Location, dryrun: bool) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (model path as seen inside the container, extra mounts).

    - transport URI (hf://...): pull via RamaLama into boxy's store, bind-mount
      the resolved blob/snapshot under /mnt/models.
    - path: the paper's shared-FS flow — relative paths resolve against the
      box workdir (which the box mounts from ${MODELS_DIR}); absolute host
      paths get bind-mounted under /mnt/models.
    """
    if not box.model:
        return "", []
    if box.model_is_transport_uri:
        host_path = ramalama_shim.pull_model(box.model, dryrun=dryrun)
    elif os.path.isabs(box.model):
        host_path = box.model
    else:
        return box.model, []  # relative shared-FS path (resolves against the workdir mount)
    _verify_checkpoint(box, host_path, dryrun)
    name = PurePosixPath(host_path).name
    return f"{STORE_MOUNT}/{name}", [(host_path, f"{STORE_MOUNT}/{name}", "ro")]


def _verify_checkpoint(box: Box, host_path: str, dryrun: bool) -> None:
    """Fail fast on an incomplete safetensors checkpoint (vLLM only, real run):
    otherwise vLLM loads for minutes, then dies with 'weights were not
    initialized from checkpoint', burning the allocation. Opt out with
    BOXY_NO_MODEL_VERIFY=1."""
    if dryrun or box.engine != "vllm" or os.environ.get("BOXY_NO_MODEL_VERIFY"):
        return
    problems = ramalama_shim.verify_safetensors_complete(host_path)
    if problems:
        raise RuntimeError(
            "boxy: the model checkpoint on disk is incomplete/corrupt — vLLM would load for "
            "minutes and then crash with 'weights were not initialized from checkpoint':\n  - "
            + "\n  - ".join(problems)
            + f"\n  path: {host_path}\n"
            "  fix: re-pull clean —  boxy pull <model> --force\n"
            "       or download directly and serve by path:\n"
            "         huggingface-cli download <repo> --local-dir DIR --exclude 'original/*'\n"
            "         boxy serve DIR ...\n"
            "  (bypass this check with BOXY_NO_MODEL_VERIFY=1)"
        )


def _apply_defaults(box: Box, accelerator: str) -> Box:
    """RamaLama-informed default image for the box's engine + this location's
    accelerator (SPEC §3c: leverage, don't reinvent). Runs BEFORE the inner
    command is built: some default images need an explicit entrypoint.

    The entrypoint default applies to USER-PINNED images too (sweep finding
    53: a pinned quay.io/ramalama/* image has no ENTRYPOINT, so deferral
    launches nothing)."""
    if not box.image:
        box = replace(box, image=ramalama_shim.default_image(box.engine, accelerator))
    if not box.entrypoint:
        entrypoint = ramalama_shim.default_entrypoint(box.engine, box.image)
        if entrypoint:
            box = replace(box, entrypoint=entrypoint)
    return box


def plan_serve(
    box: Box,
    location: Location,
    port: int | None = None,
    extra_args: list[str] | None = None,
    dryrun: bool = False,
) -> Deployment:
    accelerator = location.resolve_accelerator()
    box = _apply_defaults(box, accelerator)
    model_path, extra_mounts = resolve_model(box, location, dryrun)
    inner = engines.build_serve_cmd(box, location, model_path, port=port, extra_args=extra_args)
    deployment = _plan(box, location, inner, extra_mounts, dryrun, accelerator)
    deployment.port = engines.serving_port(inner, box)
    return deployment


def plan_run(box: Box, location: Location, user_args: list[str], dryrun: bool = False) -> Deployment:
    accelerator = location.resolve_accelerator()
    box = _apply_defaults(box, accelerator)
    inner = engines.build_raw_cmd(box, user_args, location)
    return _plan(box, location, inner, [], dryrun, accelerator)


def _plan(
    box: Box,
    location: Location,
    inner_cmd: list[str],
    extra_mounts: list[tuple[str, str, str]],
    dryrun: bool,
    accelerator: str,
) -> Deployment:
    backend = get_backend(location.resolve_runtime())
    scheduler = get_scheduler(location.scheduler)
    env = envs.build_env(box.env, accelerator, location.offline, engine=box.engine)
    mounts = resolve_mounts(box, location) + extra_mounts
    warnings: list[str] = []
    # Podman (unlike Docker) refuses to start when the workdir doesn't exist
    # in the image; a workdir no volume provides is usually a box bug.
    # (Field finding: Mac run-through, 2026-07.)
    if box.workdir and not any(target == box.workdir for _, target, _ in mounts):
        warnings.append(
            f"box {box.name!r}: workdir {box.workdir!r} is not the target of any volume; "
            "the image must already contain this directory or Podman will refuse to start "
            "(drop `workdir` or add a [[box.volumes]] entry targeting it)"
        )
    # A bind-mount source missing on THIS host fails immediately under
    # podman/docker ('statfs ... no such file or directory'). Only a warning:
    # with a scheduler wrap the path may exist on the compute node instead.
    # (Field finding #10: Mac run-through, 2026-07.)
    for source, _target, _options in mounts:
        if source == "/path/to/model":
            continue  # ramalama's --dryrun placeholder for a not-yet-pulled model
        if ":" in source:
            warnings.append(
                f"volume source {source!r} contains ':' — the --volume/--bind syntax cannot "
                "escape it; rename the path"
            )
        if os.path.isabs(source) and not os.path.exists(source):
            warnings.append(
                f"volume source {source!r} does not exist on this host — podman/docker will fail "
                "with 'statfs: no such file or directory' unless it exists on the target node "
                "(create it, or point [location.staging] models_dir at your model directory)"
            )
    cmd = backend.build_command(box, location, inner_cmd, env, mounts, accelerator)
    cmd = scheduler.with_modules(cmd, location)
    cmd = scheduler.wrap(cmd, location)
    prepare = (backend.prepare(box, location, dryrun, accelerator=accelerator)
               if backend.image_format == "sif" else [])
    return Deployment(
        box=box,
        location=location,
        accelerator=accelerator,
        backend=backend,
        scheduler=scheduler,
        command=cmd,
        prepare_commands=prepare,
        env_unset=scheduler.host_env_fixups(),
        warnings=warnings,
    )


def execute(deployment: Deployment) -> int:
    env = dict(os.environ)
    for var in deployment.env_unset:
        env.pop(var, None)
    for prep in deployment.prepare_commands:
        # SIF build is idempotent-ish but expensive: skip when present.
        sif = next((a for a in prep if a.endswith(".sif")), None)
        if sif and os.path.exists(sif):
            continue
        result = subprocess.run(prep, env=env)
        if result.returncode != 0:
            return result.returncode
    return subprocess.run(deployment.command, env=env).returncode
