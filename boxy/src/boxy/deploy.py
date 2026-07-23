"""Compose a box + location into a runnable host command.

Pipeline:  resolve accel/runtime -> env merge -> model & mounts ->
           inner cmd -> backend command -> module preamble -> scheduler wrap
"""

from __future__ import annotations

import os
import re
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
                            engine_pulls_model: bool = False,
                            prelude: list[str] | None = None,
                            distributed: bool | None = None) -> str:
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
        # field: `s3://huggingface.co/org/name` — HF is not an S3 bucket; the
        # user wanted the hf:// transport, which agentless serves directly
        if box.model_is_s3 and "huggingface" in box.model.lower():
            repo = re.sub(r"^s3://[^/]+/", "", box.model)
            raise AgentlessError(
                f"{box.model!r}: huggingface.co is not an S3 bucket — the s3:// transport is "
                f"for SITE buckets fed by `boxy push`. For HuggingFace use "
                f"`boxy serve hf://{repo} ...` — the agentless serve stages it onto the "
                f"cluster's shared FS automatically, no S3 involved.")
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
    # Multi-node = ONE vLLM instance across the allocation via Ray (same policy
    # as the boxy-on-cluster path: auto for vllm + nodes>1, --no-distributed
    # opts out; llama.cpp is never distributed). The head node runs `ray start
    # --head` + vllm; the N-1 workers are fanned out by srun / flux run from
    # inside the batch script — still zero boxy on the cluster.
    eng = _apply_defaults(box, local.resolve_accelerator()).engine
    from boxy import distributed as distmod
    multinode = distmod.is_distributed(eng, location.resources.nodes, distributed)
    if multinode and scheduler_name not in ("slurm", "flux"):
        raise AgentlessError(
            f"multi-node serving needs a scheduler that can place the Ray workers "
            f"(slurm or flux), not {scheduler_name!r}")
    global _AGENTLESS_RENDER
    _AGENTLESS_RENDER = True
    # the command RUNS on a Linux compute node no matter where it is RENDERED:
    # a Mac laptop's darwin branch (-p publishing, no --ipc=host) left the
    # container on podman's 64MB /dev/shm, and RCCL died at ncclCommInitRank
    # the moment TP>1 needed a communicator (field: clusterb, Nemotron TP=2).
    from boxy.backends import podman as podman_backend

    podman_backend.set_target_os("linux")
    try:
        if multinode:
            try:
                deployment = plan_serve(box, local, port=port, dryrun=True,
                                        distributed=True, head_ip=_HEAD_IP_TOKEN)
            except RuntimeError as e:  # e.g. gpus_per_node unknown
                raise AgentlessError(str(e)) from e
        else:
            deployment = plan_serve(box, local, port=port, dryrun=True)  # dryrun: no pull, no verify
    finally:
        _AGENTLESS_RENDER = False
        podman_backend.set_target_os(None)

    # CA into the container, picked ON THE COMPUTE NODE at job runtime: the
    # engine's in-container HuggingFace fetch must trust the site's TLS
    # interceptor or it dies with CERTIFICATE_VERIFY_FAILED (field: clustera). The
    # laptop's CA path is meaningless here; the node's own system bundle provably
    # trusts the interceptor (host-side pulls work). Candidates: the boxy-staged
    # laptop bundle (if any) first, then the node's system stores. $_CAARGS is
    # spliced into the container argv BEFORE the image; empty when nothing found.
    cmd = list(deployment.command)
    image = deployment.box.image or ""
    ca_block = ""
    runtime = local.resolve_runtime()
    if runtime in ("podman", "docker") and image and image in cmd:
        idx = cmd.index(image)
        podman = f"{shlex.join(cmd[:idx])} ${{_CAARGS}} {shlex.join(cmd[idx:])}"
        candidates = ([_AGENTLESS_CA_SOURCE] if _AGENTLESS_CA_SOURCE else []) + list(_HOST_CA_BUNDLES)
        ca_env = " ".join(f"-e {var}={CA_CONTAINER_PATH}"
                          for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"))
        ca_block = (
            '_CAARGS=""\n'
            f'for _c in {" ".join(shlex.quote(c) for c in candidates)}; do\n'
            '  if [ -f "$_c" ]; then\n'
            f'    _CAARGS="-v $_c:{CA_CONTAINER_PATH}:ro {ca_env}"\n'
            '    break\n'
            '  fi\n'
            'done\n'
        )
    else:
        podman = shlex.join(cmd)
    # engine-pull: the model lands in the container's HF cache, which the caller
    # bind-mounts from the shared FS (download once, reuse every run) — create it
    # here so podman never fails on a missing bind source.
    # engine-pull: create the HF-cache bind source (shared FS) at job runtime so
    # podman never fails on a missing mount. The REAL mounted path is read off the
    # planned --volume args (the cache may live on the big scratch FS, not next to
    # the endpoint file — see cli._remote_model_store); the endpoint-adjacent dir
    # is the fallback for a plan without an HF-cache mount.
    mk_cache = ""
    if engine_pulls_model:
        hf_srcs = [a.split("=", 1)[1].split(":")[0] for a in cmd
                   if a.startswith("--volume=") and ":/root/.cache/huggingface" in a]
        mk_cache = "".join(f"mkdir -p {shlex.quote(s)}\n" for s in hf_srcs) \
            or 'mkdir -p "$(dirname "${_EP}")/hfcache"\n'

    resolved_port = deployment.port or port or 0
    scheduler = get_scheduler(scheduler_name)

    # Multi-node: discover the head's IP AT JOB RUNTIME (unknowable laptop-side)
    # and fan ONE worker container out to the N-1 non-head nodes in the
    # background; the head podman (ray head + wait-for-cluster + vllm) then
    # runs in the foreground exactly like the single-node case. Workers retry
    # their Ray join, so the head/worker start race is harmless.
    worker_block = ""
    if multinode:
        nodes = location.resources.nodes
        # the worker container needs the SAME site CA the head gets — its
        # in-container ray self-heal (pip install) rides the interceptor and
        # dies with CERTIFICATE_VERIFY_FAILED without REQUESTS_CA_BUNDLE
        # (field: head installed ray fine, workers all failed TLS). Splice
        # ${_CAARGS} before the image exactly like the head podman command.
        wlist = list(deployment.worker_command or [])
        wimg = deployment.box.image or ""
        if runtime in ("podman", "docker") and wimg and wimg in wlist:
            widx = wlist.index(wimg)
            wcmd = f"{shlex.join(wlist[:widx])} ${{_CAARGS}} {shlex.join(wlist[widx:])}"
        else:
            wcmd = shlex.join(wlist)
        wcmd = wcmd.replace(_HEAD_IP_TOKEN, '"${_HEAD_IP}"')
        if scheduler_name == "slurm":
            fan = (f'srun --nodes={nodes - 1} --ntasks={nodes - 1} --ntasks-per-node=1 '
                   f'--exclude "$(hostname -s)" ')
        else:  # flux (guarded above). NOT `flux run`: the scheduler can't see the
            # head's plain podman process, requests no GPUs for the worker, and has
            # no --exclude — audit-confirmed it will happily place the worker ON the
            # head's node, leaving the other node idle while Ray still reports the
            # full DECLARED GPU count and vLLM wedges on the broken layout. `flux
            # exec -r` runs directly on broker ranks (rank 0 = the head, always),
            # so ranks 1..N-1 are exactly the non-head nodes — deterministic
            # disjoint placement on every flux version, no scheduler involved.
            ranks = "1" if nodes == 2 else f"1-{nodes - 1}"
            fan = f'flux exec -r {ranks} '
        # sweep stale same-name containers off EVERY job node before launching:
        # field — conmon keeps a -worker container alive after the scheduler
        # tears a failed job down, and the orphan then glomms onto the next
        # run's Ray cluster on the shared host network (or hard-fails the new
        # worker with 'name already in use'); a stale co-located worker on the
        # head node puts two raylets on one host — the audit-confirmed
        # driver-registration wedge. Apptainer needs no sweep (its containers
        # are plain processes the scheduler already kills with the job).
        sweep = ""
        if runtime in ("podman", "docker"):
            rmcmd = f"{runtime} rm -f {box.name} {box.name}-worker 2>/dev/null; true"
            if scheduler_name == "slurm":
                sweep = (f"srun --ntasks={nodes} --ntasks-per-node=1 "
                         f"bash -c {shlex.quote(rmcmd)} 2>/dev/null || true\n")
            else:  # flux exec with no -r targets every broker rank, head included
                sweep = f"flux exec bash -c {shlex.quote(rmcmd)} 2>/dev/null || true\n"
        worker_block = (
            sweep
            + '_HEAD_IP="$(hostname -I 2>/dev/null | awk \'{print $1}\')"\n'
            '[ -n "$_HEAD_IP" ] || _HEAD_IP="$(getent hosts "$_H" | awk \'{print $1}\')"\n'
            f'{fan}{proxy_prefix}{wcmd} &\n'
        )

    # atomic endpoint publish + foreground container; $(hostname) is the compute node.
    prelude_block = ("\n".join(prelude) + "\n") if prelude else ""
    body = (
        'set -e\n'
        f'{prelude_block}'
        '_H="$(hostname)"\n'
        f'_EP={shlex.quote(endpoint_file)}\n'
        f'{mk_cache}'
        f'{ca_block}'
        f'{worker_block}'
        'cat > "${_EP}.tmp" <<EOF_BOXY_EP\n'
        f'{{"name": "{name}", "host": "${{_H}}", "port": {resolved_port}, '
        f'"url": "http://${{_H}}:{resolved_port}", "job": "${{SLURM_JOB_ID:-${{FLUX_JOB_ID:-}}}}"}}\n'
        'EOF_BOXY_EP\n'
        'mv -f "${_EP}.tmp" "${_EP}"\n'
        f'exec {proxy_prefix}{podman}'
    )
    return scheduler.batch_script("", location, name, log_file, site_args, body=body,
                                  distributed=multinode)


def plan_run(box: Box, location: Location, user_args: list[str], dryrun: bool = False) -> Deployment:
    accelerator = location.resolve_accelerator()
    box = _apply_defaults(box, accelerator)
    inner = engines.build_raw_cmd(box, user_args, location)
    return _plan(box, location, inner, [], dryrun, accelerator)


CA_CONTAINER_PATH = "/etc/ssl/certs/boxy-ca-merged.pem"

# Multi-node agentless: the head's IP is only knowable at job runtime, but the
# worker plan is rendered laptop-side. This token stands in for it in the planned
# worker command and is swapped for the script's ${_HEAD_IP} after shlex-joining
# (it contains no shell metacharacters, so the join never quotes it).
_HEAD_IP_TOKEN = "__BOXY_HEAD_IP__"

# Agentless CA handling: over --ssh the compute node can't see the LAPTOP's
# SSL_CERT_FILE path, so a static mount of it bind-mounts nothing and the
# container still doesn't trust the site's TLS interceptor (field: clustera,
# CERTIFICATE_VERIFY_FAILED fetching config.json from huggingface.co). The
# agentless batch script instead picks a CA bundle AT JOB RUNTIME on the compute
# node — the boxy-staged laptop bundle if the caller staged one (set via
# set_agentless_ca), else the NODE's OWN system trust store, which provably
# trusts the interceptor (host-side podman pulls succeed). While an agentless
# render is in flight, _propagate_ca_bundle must stay out of the way (a laptop
# path in the mounts would be invalid on the node).
_AGENTLESS_CA_SOURCE: str | None = None
_AGENTLESS_RENDER = False

# RHEL/Fedora first (the site clusters), then Debian/Ubuntu — checked in order on
# the COMPUTE node by the batch script.
_HOST_CA_BUNDLES = (
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
)


def set_agentless_ca(cluster_path: str | None) -> None:
    """Register a CLUSTER-side staged merged-CA path as the batch script's first
    CA candidate (ahead of the node's system bundle). Pass None to clear.
    Process-global; the caller sets it around render_agentless_script and clears
    it after (mirrors schedulers.slurm.set_auto_gres)."""
    global _AGENTLESS_CA_SOURCE
    _AGENTLESS_CA_SOURCE = cluster_path or None


def _propagate_ca_bundle(env: dict, mounts: list) -> None:
    """Mount boxy's merged CA (certifi public CAs + your site CA) into the container
    and point its TLS stacks at it, so IN-CONTAINER HuggingFace/transformers/httpx
    downloads trust the site CA. Without this a model that fetches code or weights
    at load (custom architectures, remote-code deps) dies with CERTIFICATE_VERIFY_
    FAILED even though host-side pulls work.

    Only the MERGED bundle is propagated (it contains the public CAs too, so public
    HTTPS keeps working) — never a bare site CA, which would REPLACE the container's
    trust and break huggingface.co. No-op when the merge is disabled/absent or the
    user set the cert env in [box.env].

    No-op during an agentless render: the laptop's SSL_CERT_FILE path is invalid
    on the compute node, so the batch script mounts a node-side bundle at job
    runtime instead (see render_agentless_script's CA-pick block)."""
    if _AGENTLESS_RENDER:
        return
    ca = os.environ.get("SSL_CERT_FILE")
    if not ca or not ca.endswith("ca-merged.crt") or not os.path.isfile(ca):
        return
    if any(target == CA_CONTAINER_PATH for _, target, _ in mounts):
        return
    mounts.append((ca, CA_CONTAINER_PATH, "ro"))
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        env.setdefault(var, CA_CONTAINER_PATH)  # box.env still wins (already merged)


_AIRGAP_RENDER = False


def set_airgap(on: bool) -> None:
    """Toggle air-gapped rendering (serve --bundle): suppresses proxy/CA
    propagation into the container plan. Reset by the caller / test conftest."""
    global _AIRGAP_RENDER
    _AIRGAP_RENDER = bool(on)


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
    if not _AIRGAP_RENDER:
        # an air-gapped --bundle serve must carry ZERO network configuration —
        # no proxy, no CA (everything resolves from the bundle, offline).
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
