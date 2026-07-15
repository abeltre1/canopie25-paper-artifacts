"""Compose a box + location into a runnable host command.

Pipeline:  resolve accel/runtime -> env merge -> model & mounts ->
           inner cmd -> backend command -> module preamble -> scheduler wrap
"""

from __future__ import annotations

import os
import shlex
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
    # Multi-node distributed (Ray) serving: `command` runs the HEAD (ray head +
    # vllm serve); `worker_command` is the bare per-node worker container to be
    # placed on the N-1 non-head nodes via srun by the caller. (tp, pp) is the
    # derived parallelism; world_size = tp*pp.
    distributed: bool = False
    worker_command: list[str] | None = None
    parallelism: tuple[int, int] | None = None
    world_size: int = 0


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
    if box.model_is_s3:
        from boxy import s3

        host_path = s3.stage_model(box.model, location.staging.models_dir,
                                   endpoint=location.staging.s3_endpoint or None, dryrun=dryrun,
                                   runtime=location.resolve_runtime())
    elif box.model_is_transport_uri:
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
    distributed: bool = False,
    head_ip: str = "",
) -> Deployment:
    accelerator = location.resolve_accelerator()
    box = _apply_defaults(box, accelerator)
    model_path, extra_mounts = resolve_model(box, location, dryrun)
    if distributed:
        return _plan_distributed(box, location, model_path, extra_mounts, accelerator,
                                 port, extra_args, head_ip, dryrun)
    inner = engines.build_serve_cmd(box, location, model_path, port=port, extra_args=extra_args)
    deployment = _plan(box, location, inner, extra_mounts, dryrun, accelerator)
    deployment.port = engines.serving_port(inner, box)
    return deployment


def _plan_distributed(box, location, model_path, extra_mounts, accelerator,
                      port, extra_args, head_ip, dryrun) -> Deployment:
    """Head + worker plan for multi-node Ray serving. The HEAD command (ray head
    + vllm serve) runs directly on this node — NOT wrapped in srun (that would
    launch N colliding copies); the worker command is placed on the other nodes
    by the caller. TP/PP are derived from the geometry and baked into the vLLM
    argv (a user/box/tuning value still wins)."""
    from boxy import distributed

    tp, pp, world = distributed.derive_parallelism(location.resources)
    gpus = location.resources.gpus_per_node
    vllm_argv = engines.build_serve_cmd(box, location, model_path, port=port,
                                        extra_args=extra_args, parallelism=(tp, pp))
    head_inner = distributed.ray_head_inner(vllm_argv, gpus, world)
    head = _plan(box, location, head_inner, extra_mounts, dryrun, accelerator, wrap=False)

    worker_box = replace(box, name=f"{box.name}-worker")
    worker_inner = distributed.ray_worker_inner(gpus)
    worker_env = {distributed.HEAD_ENV: head_ip} if head_ip else {}
    worker = _plan(worker_box, location, worker_inner, extra_mounts, dryrun, accelerator,
                   wrap=False, extra_env=worker_env)

    head.distributed = True
    head.parallelism = (tp, pp)
    head.world_size = world
    head.port = engines.serving_port(vllm_argv, box)
    head.worker_command = worker.command
    return head


class AgentlessError(RuntimeError):
    """Agentless mode can't proceed: the model isn't staged, or accel/image
    aren't pinned so the podman command can't be resolved off the compute node."""


def render_agentless_script(box: Box, location: Location, scheduler_name: str, name: str,
                            endpoint_file: str, log_file: str, site_args: list[str],
                            proxy_prefix: str = "", port: int | None = None,
                            engine_pulls_model: bool = False) -> str:
    """A FULLY SELF-CONTAINED batch script — no boxy/Python/RamaLama on the
    cluster. The compute node runs only `podman run` (resolved HERE) plus a
    bash endpoint-write to the shared FS; the laptop submits it and polls that
    file. The accelerator/image must be pinned (resolution can't run on the
    compute node).

    The model: normally a PRE-STAGED shared-FS path. With `engine_pulls_model`
    (the caller has rewritten box.model to a bare repo id), the ENGINE itself
    downloads it at container start — vLLM `vllm serve <repo>` pulls from
    HuggingFace over the forwarded proxy — so no RamaLama on the cluster. An s3
    model still needs staging.

    The container command is resolved with a scheduler='none' overlay so it is
    the plain foreground `podman run` (no srun); the REAL scheduler supplies
    only the batch directives."""
    if box.model_is_s3 or (box.model_is_transport_uri and not engine_pulls_model):
        raise AgentlessError(
            f"agentless needs a PRE-STAGED model on the shared filesystem, not {box.model!r} — "
            "a transport-URI/s3 pull requires RamaLama on the cluster. Stage it first "
            "(boxy pull / copy the GGUF over) and pass the shared-FS path.")
    # scheduler='none' -> plain foreground podman (no srun); runtime lives on the
    # cluster, so never probe this host — default to podman when unpinned.
    local = replace(location, scheduler="none", runtime=(location.runtime or "podman"))
    if not local.accelerator:
        raise AgentlessError(
            "agentless can't detect hardware on the compute node — pin --accelerator "
            "(cuda|rocm|…) so the podman command is fully resolved here.")
    deployment = plan_serve(box, local, port=port, dryrun=True)  # dryrun: no pull, no verify
    podman = shlex.join(deployment.command)
    resolved_port = deployment.port or port or 0
    scheduler = get_scheduler(scheduler_name)
    # atomic endpoint publish + foreground container; $(hostname) is the compute node.
    body = (
        'set -e\n'
        '_H="$(hostname)"\n'
        f'_EP={shlex.quote(endpoint_file)}\n'
        'cat > "${_EP}.tmp" <<EOF_BOXY_EP\n'
        f'{{"name": "{name}", "host": "${{_H}}", "port": {resolved_port}, '
        f'"url": "http://${{_H}}:{resolved_port}", "job": "${{SLURM_JOB_ID:-${{FLUX_JOB_ID:-}}}}"}}\n'
        'EOF_BOXY_EP\n'
        'mv -f "${_EP}.tmp" "${_EP}"\n'
        f'exec {proxy_prefix}{podman}'
    )
    return scheduler.batch_script("", location, name, log_file, site_args, body=body)


def plan_run(box: Box, location: Location, user_args: list[str], dryrun: bool = False) -> Deployment:
    accelerator = location.resolve_accelerator()
    box = _apply_defaults(box, accelerator)
    inner = engines.build_raw_cmd(box, user_args, location)
    return _plan(box, location, inner, [], dryrun, accelerator)


CA_CONTAINER_PATH = "/etc/ssl/certs/boxy-ca-merged.pem"


def _propagate_ca_bundle(env: dict, mounts: list) -> None:
    """Mount boxy's merged CA (certifi public CAs + your site CA) into the container
    and point its TLS stacks at it, so IN-CONTAINER HuggingFace/transformers/httpx
    downloads trust the site CA. Without this a model that fetches code or weights
    at load (custom architectures, remote-code deps) dies with CERTIFICATE_VERIFY_
    FAILED even though host-side pulls work.

    Only the MERGED bundle is propagated (it contains the public CAs too, so public
    HTTPS keeps working) — never a bare site CA, which would REPLACE the container's
    trust and break huggingface.co. No-op when the merge is disabled/absent or the
    user set the cert env in [box.env]."""
    ca = os.environ.get("SSL_CERT_FILE")
    if not ca or not ca.endswith("ca-merged.crt") or not os.path.isfile(ca):
        return
    if any(target == CA_CONTAINER_PATH for _, target, _ in mounts):
        return
    mounts.append((ca, CA_CONTAINER_PATH, "ro"))
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        env.setdefault(var, CA_CONTAINER_PATH)  # box.env still wins (already merged)


def _propagate_proxy(env: dict) -> None:
    """Carry the host's proxy vars INTO the container so in-container downloads
    (vLLM fetching remote code/weights at load) reach the corporate proxy — the
    same idea as the CA bundle. The IMAGE pull happens host-side, so it's fed
    separately (boxy prefixes the compute-node command with the proxy env at
    submit). box.env still wins."""
    from boxy import ramalama_shim

    for var, val in ramalama_shim.raw_proxy_env().items():
        env.setdefault(var, val)


def _plan(
    box: Box,
    location: Location,
    inner_cmd: list[str],
    extra_mounts: list[tuple[str, str, str]],
    dryrun: bool,
    accelerator: str,
    wrap: bool = True,
    extra_env: dict[str, str] | None = None,
) -> Deployment:
    backend = get_backend(location.resolve_runtime())
    scheduler = get_scheduler(location.scheduler)
    env = envs.build_env(box.env, accelerator, location.offline, engine=box.engine)
    if extra_env:
        env.update(extra_env)
    mounts = resolve_mounts(box, location) + extra_mounts
    _propagate_ca_bundle(env, mounts)
    _propagate_proxy(env)
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
    if wrap:  # distributed head/worker run directly / via their own srun — no launch-prefix wrap
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
