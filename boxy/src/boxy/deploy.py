"""Compose a box + location into a runnable host command.

Pipeline:  resolve accel/runtime -> env merge -> model & mounts ->
           inner cmd -> backend command -> module preamble -> scheduler wrap
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
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


def _expand(value: str, location: Location) -> str:
    value = value.replace("${MODELS_DIR}", location.staging.models_dir)
    # Docker rejects relative bind-mount sources (and older Podman does too):
    # absolutize against the CWD, mirroring how the prototype was always run
    # from the workflow directory.
    if value.startswith("."):
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
        name = PurePosixPath(host_path).name
        return f"{STORE_MOUNT}/{name}", [(host_path, f"{STORE_MOUNT}/{name}", "ro")]
    if os.path.isabs(box.model):
        name = PurePosixPath(box.model).name
        return f"{STORE_MOUNT}/{name}", [(box.model, f"{STORE_MOUNT}/{name}", "ro")]
    return box.model, []


def plan_serve(
    box: Box,
    location: Location,
    port: int | None = None,
    extra_args: list[str] | None = None,
    dryrun: bool = False,
) -> Deployment:
    model_path, extra_mounts = resolve_model(box, location, dryrun)
    inner = engines.build_serve_cmd(box, location, model_path, port=port, extra_args=extra_args)
    return _plan(box, location, inner, extra_mounts, dryrun)


def plan_run(box: Box, location: Location, user_args: list[str], dryrun: bool = False) -> Deployment:
    inner = engines.build_raw_cmd(box, user_args, location)
    return _plan(box, location, inner, [], dryrun)


def _plan(
    box: Box,
    location: Location,
    inner_cmd: list[str],
    extra_mounts: list[tuple[str, str, str]],
    dryrun: bool,
) -> Deployment:
    accelerator = location.resolve_accelerator()
    backend = get_backend(location.resolve_runtime())
    scheduler = get_scheduler(location.scheduler)
    env = envs.build_env(box.env, accelerator, location.offline)
    mounts = resolve_mounts(box, location) + extra_mounts
    cmd = backend.build_command(box, location, inner_cmd, env, mounts, accelerator)
    cmd = scheduler.with_modules(cmd, location)
    cmd = scheduler.wrap(cmd, location)
    prepare = backend.prepare(box, location, dryrun) if backend.image_format == "sif" else []
    return Deployment(
        box=box,
        location=location,
        accelerator=accelerator,
        backend=backend,
        scheduler=scheduler,
        command=cmd,
        prepare_commands=prepare,
        env_unset=scheduler.host_env_fixups(),
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
