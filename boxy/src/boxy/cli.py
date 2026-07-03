"""boxy CLI: one tool to deploy/serve containerized GenAI across HPC sites.

    boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf

Everything (engine, image, runtime, port) is auto-resolved and printed;
every choice is overridable by a flag or a --box/--location TOML profile.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys

from boxy import __version__, ramalama_shim
from boxy.backends import BACKENDS
from boxy.box import TRANSPORT_SCHEMES, Box
from boxy.location import ACCELERATORS, Location
from boxy.schedulers import SCHEDULERS

NOT_IN_MVP = "not implemented in the MVP — see SPEC.md §8 (roadmap) for the phase that adds it"


class UsageError(ValueError):
    """CLI misuse — exits 2 like argparse's own usage errors (finding 51)."""


def _add_common(parser: argparse.ArgumentParser, location_required: bool = True) -> None:
    parser.add_argument("--box", required=True, help="path to a box TOML definition")
    parser.add_argument("--location", required=location_required, help="path to a location TOML definition")
    parser.add_argument("--dryrun", action="store_true", help="print the command instead of executing it")


def _load(args: argparse.Namespace) -> tuple[Box, Location]:
    return Box.from_toml(args.box), Location.from_toml(args.location)


def _emit(deployment, dryrun: bool) -> int:
    from boxy import deploy

    for warning in deployment.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for prep in deployment.prepare_commands:
        print(f"### Prepare: {shlex.join(prep)}")
    print(f"### Running Command:\n    {shlex.join(deployment.command)}")
    if dryrun:
        return 0
    return deploy.execute(deployment)


def cmd_info(args: argparse.Namespace) -> int:
    print(f"boxy {__version__}")
    print(f"ramalama library: {'available' if ramalama_shim.ramalama_available() else 'not installed'}")
    print(f"accelerator: {ramalama_shim.detect_accel()}")
    runtimes = [name for name in BACKENDS if shutil.which(name)]
    print(f"container runtimes: {', '.join(runtimes) or 'none found'}")
    launchers = [name for name, probe in (("slurm", "srun"), ("flux", "flux")) if shutil.which(probe)]
    print(f"schedulers: {', '.join(launchers) or 'none found'}")
    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        status = "" if os.path.exists(ssl_cert) else "  (MISSING FILE!)"
        print(f"tls: SSL_CERT_FILE={ssl_cert}{status}")
    else:
        os_bundle = ramalama_shim.discover_os_ca_bundle()
        if os_bundle:
            print(f"tls: system default CA store; boxy auto-merges the OS trust store "
                  f"({os_bundle}) with certifi on pull (disable: BOXY_NO_CA_MERGE=1)")
        else:
            print("tls: system default CA store; no OS CA bundle found — if pulls fail with "
                  "CERTIFICATE_VERIFY_FAILED, set SSL_CERT_FILE to your site CA and persist it")
    from boxy import policy

    allowed = policy.allowed_transports()
    blocked = sorted({policy._canonical(s) for s in policy.REGISTRIES} - set(allowed))
    print(f"registries: allowed [{', '.join(allowed)}]  blocked [{', '.join(blocked)}]"
          + ("" if os.environ.get("BOXY_ALLOW_TRANSPORTS") else "  (default policy)"))
    # auth STATUS only — values are never printed. When BOTH sources exist,
    # say which one WINS: RamaLama's precedence is HF_TOKEN env outright; the
    # cache file is ignored while HF_TOKEN is set (verified at its source).
    token, source = ramalama_shim.effective_hf_token()
    hf_sources = ramalama_shim._hf_token_sources()
    if token:
        note = f"present, using {source}"
        if len(hf_sources) > 1:
            note += " — takes precedence; ~/.cache/huggingface/token is IGNORED while HF_TOKEN is set"
        note += "  (validate: boxy info --net)"
    elif source.startswith("HF_TOKEN env var (set but EMPTY"):
        note = source
    else:
        note = "not configured (export HF_TOKEN=... for gated repos)"
    print(f"auth: HuggingFace token: {note}")
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        s3 = "present (AWS_ACCESS_KEY_ID env)"
    elif os.environ.get("AWS_PROFILE"):
        s3 = f"profile '{os.environ['AWS_PROFILE']}' (AWS_PROFILE env)"
    elif os.path.exists(os.path.expanduser("~/.aws/credentials")):
        s3 = "present (~/.aws/credentials)"
    else:
        s3 = "not configured (only needed for [location.staging] s3_endpoint)"
    print(f"auth: S3 credentials: {s3}")
    if getattr(args, "net", False):
        return _probe_registries()
    return 0


def _probe_registries() -> int:
    """`boxy info --net`: try each ALLOWED model registry with the CURRENT
    trust store (after boxy's certifi merge, like a real pull). Registries
    outside the policy allowlist are not even probed. Any HTTP response —
    even 401/404 — proves TLS worked; only transport errors are failures."""
    import urllib.error
    import urllib.request

    from boxy import policy

    ramalama_shim.ensure_trust_bundle()
    failures = 0
    token, source = ramalama_shim.effective_hf_token()
    if token:
        # the definitive "did my token take effect" answer: ask HF who the
        # EFFECTIVE token belongs to (same resolution as the pull itself)
        request = urllib.request.Request("https://huggingface.co/api/whoami-v2",
                                         headers={"Authorization": f"Bearer {token}",
                                                  "User-Agent": "boxy"})
        try:
            import json as _json

            with urllib.request.urlopen(request, timeout=8) as resp:
                who = _json.load(resp)
            print(f"net: hf-auth    whoami OK — token from {source} is VALID "
                  f"(user: {who.get('name', '?')})")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                print(f"net: hf-auth    token from {source} is INVALID (HTTP {e.code}) — "
                      f"HF rejects it; gated/private pulls WILL fail. Generate a fresh token at "
                      f"https://huggingface.co/settings/tokens and re-export HF_TOKEN.")
                failures += 1
            else:
                print(f"net: hf-auth    could not validate (HTTP {e.code})")
        except Exception as e:
            print(f"net: hf-auth    could not validate ({getattr(e, 'reason', e)})")
    for scheme, url in policy.registry_probes():
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                print(f"net: {scheme:10s} {url}  OK (HTTP {resp.status})")
        except urllib.error.HTTPError as e:
            print(f"net: {scheme:10s} {url}  OK (TLS fine; HTTP {e.code})")
        except Exception as e:
            reason = getattr(e, "reason", e)
            print(f"net: {scheme:10s} {url}  FAIL ({reason})")
            failures += 1
    if failures:
        print("net: FAILing registries cannot be pulled from this shell — "
              "if the reason is CERTIFICATE_VERIFY_FAILED, see the SSL_CERT_FILE notes in RUNBOOK §2.1")
    return 1 if failures else 0


def _resolve_or_load(args: argparse.Namespace):
    """v2 front door: positional MODEL with full auto-resolution; --box/--location
    profiles still honored. Explicit flags ALWAYS win over profile values, and
    a profile's pinned scheduler/accelerator/runtime inform resolution (they
    behave like flags — a slurm profile must not trip the login-node guard).
    (Sweep findings 11/12/13/39/46.)"""
    from dataclasses import replace as dc_replace

    from boxy import resolve

    if args.box and args.model:
        raise UsageError(
            "MODEL and --box are mutually exclusive (the profile names its model); "
            "extra engine args go after `--`"
        )
    if not args.model and not args.box:
        raise UsageError(
            f"usage: boxy {args.subcommand} MODEL   "
            f"(or: boxy {args.subcommand} --box box.toml [--location loc.toml])"
        )

    profile = Location.from_toml(args.location) if args.location else None

    if args.box:
        box = Box.from_toml(args.box)
        decisions: list[str] = []
        # explicit flags overlay the box profile
        for field_name, value in (("name", args.name), ("image", args.image), ("engine", args.engine)):
            if value:
                box = dc_replace(box, **{field_name: value})
                decisions.append(f"{field_name}: {value} (flag overrides profile)")
        if profile is not None:
            location = profile
        else:
            location, loc_decisions = resolve.auto_location(
                runtime=args.runtime,
                scheduler=args.scheduler,
                accelerator=args.accelerator,
                gpus=args.gpus or 0,
                nodes=args.nodes or 1,
                here=args.here,
            )
            decisions += loc_decisions
        location = _overlay_location_flags(location, args, decisions)
        return box, location, decisions

    # model mode: profile values act as defaults, explicit flags win.
    # A --port inside the engine extras counts as explicit too (r2 audit:
    # the decision line advertised the scanned default while extras won).
    from boxy.engines import parse_port_flag

    extras_port = parse_port_flag(list(args.args or []))
    if extras_port is not None and args.port is not None and extras_port != args.port:
        print(f"warning: both --port {args.port} and an engine-level --port {extras_port} were "
              f"given; the engine flag wins (serving on {extras_port})", file=sys.stderr)
    # a slurm/flux profile's job geometry counts for engine inference too
    # (r2 audit: profile gpus_per_node=2 still produced a 'no GPU' refusal)
    profile_gpus = (profile.resources.gpus_per_node
                    if profile and profile.scheduler in ("slurm", "flux") else 0)
    profile_nodes = (profile.resources.nodes
                     if profile and profile.scheduler in ("slurm", "flux") else 1)
    sources = {}
    if profile is not None:
        if not args.runtime and profile.runtime:
            sources["runtime"] = "location profile"
        if not args.scheduler:
            sources["scheduler"] = "location profile"
        if not args.accelerator and profile.accelerator:
            sources["accelerator"] = "location profile"
    r = resolve.resolve(
        args.model,
        engine=args.engine,
        runtime=args.runtime or (profile.runtime or None if profile else None),
        scheduler=args.scheduler or (profile.scheduler if profile else None),
        image=args.image,
        port=extras_port if extras_port is not None else args.port,
        gpus=args.gpus or profile_gpus or 0,
        nodes=args.nodes or profile_nodes or 1,
        name=args.name,
        accelerator=args.accelerator or (profile.accelerator or None if profile else None),
        here=args.here,
        require_exists=not args.dryrun,
        sources=sources,
    )
    if profile is not None:
        # keep the profile's site details (modules/tuning/offline/staging/
        # scheduler_args) but overlay explicit flags, and fill fields the
        # profile leaves to autodetection from this resolution
        location = _overlay_location_flags(profile, args, r.decisions)
        fill = {}
        if not location.runtime:
            fill["runtime"] = r.location.runtime
        if not location.accelerator:
            fill["accelerator"] = r.location.accelerator
        if fill:
            location = dc_replace(location, **fill)
        return r.box, location, r.decisions
    return r.box, r.location, r.decisions


def _overlay_location_flags(location: Location, args: argparse.Namespace,
                            decisions: list[str]) -> Location:
    """Explicit --runtime/--accelerator/--scheduler/--gpus/--nodes override a
    loaded location profile (finding 12/40: they were silently ignored)."""
    from dataclasses import replace as dc_replace

    from boxy.location import Resources

    updates = {}
    for field_name, value in (("runtime", args.runtime), ("accelerator", args.accelerator),
                              ("scheduler", args.scheduler)):
        if value and getattr(location, field_name) != value:
            updates[field_name] = value
            if not any(d.startswith(f"{field_name}: {value}") for d in decisions):
                decisions.append(f"{field_name}: {value} (flag overrides profile)")
    resources = location.resources
    if args.gpus is not None and args.gpus != resources.gpus_per_node:
        resources = Resources(nodes=resources.nodes, gpus_per_node=args.gpus,
                              accelerator_type=resources.accelerator_type)
        decisions.append(f"gpus-per-node: {args.gpus} (flag overrides profile)")
    if args.nodes is not None and args.nodes != resources.nodes:
        resources = Resources(nodes=args.nodes, gpus_per_node=resources.gpus_per_node,
                              accelerator_type=resources.accelerator_type)
        decisions.append(f"nodes: {args.nodes} (flag overrides profile)")
    if resources is not location.resources:
        updates["resources"] = resources
    return dc_replace(location, **updates) if updates else location


def _toml_value(value) -> str:
    """TOML-safe scalar: json.dumps escapes quotes/backslashes for strings
    and matches TOML syntax for ints/floats/bools (finding 5/15)."""
    import json

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _save_profile(prefix: str, box, location) -> None:
    """Snapshot the resolved configuration to TOML profiles. Full fidelity:
    env/volumes/args and modules/tuning/staging are serialized too — a
    'reproducibility snapshot' that reloads into a different deployment is
    worse than none (findings 0/9/48)."""
    header = (
        "# written by `boxy --save-profile`: values autodetected on the node where\n"
        "# it ran — review accelerator/runtime before reusing on a different node.\n"
    )
    box_lines = [header + "[box]"]
    for key in ("name", "image", "engine", "entrypoint", "model", "workdir"):
        value = getattr(box, key)
        if value:
            box_lines.append(f"{key} = {_toml_value(value)}")
    if box.ports:
        box_lines.append(f"ports = {[int(p) for p in box.ports]}")
    if box.env:
        box_lines.append("[box.env]")
        box_lines += [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in box.env.items()]
    if box.args:
        box_lines.append("[box.args]")
        box_lines += [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in box.args.items()]
    for volume in box.volumes:
        box_lines.append("[[box.volumes]]")
        box_lines.append(f"source = {_toml_value(volume.source)}")
        box_lines.append(f"target = {_toml_value(volume.target)}")
        if volume.options:
            box_lines.append(f"options = {_toml_value(volume.options)}")

    loc_lines = [header + "[location]"]
    for key in ("name", "scheduler", "accelerator", "runtime", "registry"):
        value = getattr(location, key)
        if value:
            loc_lines.append(f"{key} = {_toml_value(value)}")
    if location.offline:
        loc_lines.append("offline = true")
    if location.modules:
        loc_lines.append("modules = [" + ", ".join(_toml_value(m) for m in location.modules) + "]")
    if location.scheduler_args:
        loc_lines.append("scheduler_args = ["
                         + ", ".join(_toml_value(a) for a in location.scheduler_args) + "]")
    loc_lines += ["[location.resources]", f"nodes = {location.resources.nodes}",
                  f"gpus_per_node = {location.resources.gpus_per_node}"]
    if location.staging.models_dir != "./models" or location.staging.s3_endpoint:
        loc_lines.append("[location.staging]")
        loc_lines.append(f"models_dir = {_toml_value(location.staging.models_dir)}")
        if location.staging.s3_endpoint:
            loc_lines.append(f"s3_endpoint = {_toml_value(location.staging.s3_endpoint)}")
    if location.tuning:
        flat = {k: v for k, v in location.tuning.items() if not isinstance(v, dict)}
        nested = {k: v for k, v in location.tuning.items() if isinstance(v, dict)}
        if flat:
            loc_lines.append("[location.tuning]")
            loc_lines += [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in flat.items()]
        for engine_name, table in nested.items():
            loc_lines.append(f'[location.tuning.{_toml_value(engine_name)}]')
            loc_lines += [f"{_toml_value(k)} = {_toml_value(v)}" for k, v in table.items()]
    with open(f"{prefix}.box.toml", "w") as f:
        f.write("\n".join(box_lines) + "\n")
    with open(f"{prefix}.location.toml", "w") as f:
        f.write("\n".join(loc_lines) + "\n")
    print(f"profiles written: {prefix}.box.toml, {prefix}.location.toml")


def _inside_allocation() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_ENCLOSING_ID")
                or os.environ.get("FLUX_JOB_ID"))


def _detachable(deployment) -> bool:
    """Detach only where a runtime daemon owns the container lifetime: laptop /
    workstation podman|docker with no scheduler in play. Inside an allocation
    the job step must own the server (epilog would reap a daemonized one), and
    under srun/flux wrap `-d` would end the job step immediately."""
    return (deployment.location.scheduler == "none"
            and not _inside_allocation()
            and deployment.command[:2] and deployment.command[1] == "run"
            and deployment.command[0] in ("podman", "docker"))


def _container_exists(runtime: str, name: str) -> bool:
    result = subprocess.run([runtime, "inspect", "--format", "{{.Id}}", name],
                            capture_output=True, text=True)
    return result.returncode == 0


def _container_label(runtime: str, name: str) -> str:
    result = subprocess.run(
        [runtime, "inspect", "--format", '{{index .Config.Labels "boxy.box"}}', name],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _container_port(runtime: str, name: str) -> int | None:
    """The port an existing container actually serves on, parsed from its own
    command line (boxy always injects --port). The FRESH resolution's port is
    wrong here by construction: the running instance makes its own port look
    'busy', so the scan advances past it and the probe would miss."""
    import json

    result = subprocess.run([runtime, "inspect", "--format", "{{json .Config.Cmd}}", name],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        cmd = json.loads(result.stdout) or []
    except ValueError:
        return None
    from boxy.engines import parse_port_flag

    return parse_port_flag(cmd)


def _reclaim_or_report(runtime: str, name: str, url: str) -> tuple[int | None, str]:
    """Name-collision policy (field finding #14, 2026-07):

      * name held by OUR container, answering  -> report the endpoint, exit 0
        (rerunning `boxy serve MODEL` is idempotent — a suffix here would
        silently double-serve the model and its memory);
      * ours, running but not answering        -> say it's still loading;
      * ours, exited                           -> dump its logs, remove, relaunch;
      * NOT created by boxy                    -> auto-suffix (-2, -3, ...) and
        proceed — someone else owns that name, colliding is pure friction.

    Returns (exit code | None to launch, final container name)."""
    from boxy import readiness

    if not _container_exists(runtime, name):
        return None, name
    if _container_label(runtime, name) != name:
        # foreign owner: walk the suffixes — reclaim OUR suffixed instance if
        # one exists (keeps reruns idempotent), else take the first free name.
        for i in range(2, 10):
            candidate = f"{name}-{i}"
            if not _container_exists(runtime, candidate):
                print(f"  auto: name: {candidate} ({name!r} exists but was not created by boxy)")
                return None, candidate
            if _container_label(runtime, candidate) == candidate:
                return _reclaim_or_report(runtime, candidate, url)
        raise RuntimeError(f"container names {name} and {name}-2..-9 are all taken; pass --name")
    if _container_running(runtime, name):
        # ask the CONTAINER for its port: the fresh resolution's port is wrong
        # by construction (the running instance made it look "busy").
        actual_port = _container_port(runtime, name)
        if actual_port:
            url = f"http://127.0.0.1:{actual_port}"
        model_id = readiness.wait_ready(url, timeout_s=2, interval_s=0.5)
        if model_id:
            print(f"### ALREADY SERVING  {url}/v1   (model: {model_id})")
            print(f"###   try:  curl -s {url}/v1/models")
            print(f"###   stop: boxy stop {name}")
            return 0, name
        print(f"boxy: {name} is already running but not answering yet (model still loading?)\n"
              f"  follow: {runtime} logs -f {name}\n"
              f"  stop:   boxy stop {name}", file=sys.stderr)
        return 1, name
    print("boxy: found an exited container from a previous attempt; its last log lines:", file=sys.stderr)
    _print_diagnosis(_dump_logs(runtime, name))
    subprocess.run([runtime, "rm", name], capture_output=True)
    print(f"boxy: removed {name}; relaunching ...", file=sys.stderr)
    return None, name


def _container_running(runtime: str, name: str) -> bool:
    result = subprocess.run([runtime, "inspect", "--format", "{{.State.Running}}", name],
                            capture_output=True, text=True)
    return result.returncode == 0 and "true" in result.stdout


def _dump_logs(runtime: str, name: str, tail: int = 50) -> str:
    """Print the container's last log lines to stderr and return the captured
    text so the caller can run it through the startup diagnostics."""
    result = subprocess.run([runtime, "logs", "--tail", str(tail), name],
                            capture_output=True, text=True)
    captured = []
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(stream.rstrip(), file=sys.stderr)
            captured.append(stream)
    return "\n".join(captured)


def _print_diagnosis(log_text: str) -> None:
    """Scan dumped engine logs for a known failure signature and, if found,
    print a plain-language fix instead of leaving the user with a raw trace."""
    from boxy import diagnostics

    hint = diagnostics.diagnose(log_text)
    if hint:
        print(hint, file=sys.stderr)


def _job_state(scheduler, job_id: str) -> str:
    """PENDING | RUNNING | DONE | UNKNOWN. A scheduler that cannot be REACHED
    (controller down, squeue missing) must be UNKNOWN, never DONE: squeue's
    connect-failure signature (rc!=0, empty stdout) is identical to
    'job left the queue', and misreading it reaped live jobs, resubmitted
    duplicates, and cancelled the wrong job (r2 audit, reproduced live)."""
    try:
        result = subprocess.run(scheduler.state_command(job_id), capture_output=True,
                                text=True, timeout=20)
    except Exception:
        return "UNKNOWN"
    if result.returncode != 0:
        err = ((result.stderr or "") + (result.stdout or "")).lower()
        if "invalid job" in err or "unknown job" in err:
            return "DONE"  # the scheduler answered: no such job
        return "UNKNOWN"   # the scheduler did NOT answer — assume nothing
    return scheduler.interpret_state(result.stdout)


def _dump_file_tail(path, tail: int = 30) -> str:
    """Print the tail of a job log file and return the captured text for
    diagnosis."""
    try:
        lines = open(path, errors="replace").read().splitlines()[-tail:]
        if not lines:
            print(f"    (log at {path} is empty)", file=sys.stderr)
        for line in lines:
            print(f"    {line}", file=sys.stderr)
        return "\n".join(lines)
    except OSError:
        print(f"    (no log at {path} yet)", file=sys.stderr)
        return ""


def _inner_serve_command(args, model: str, name: str) -> str:
    """The command the batch job runs ON the compute node: boxy itself, in
    foreground (the job step owns the server), re-resolving hardware there
    and publishing its endpoint over the shared filesystem."""
    from boxy import jobs

    boxy_bin = shutil.which("boxy")
    base = [boxy_bin] if boxy_bin else [sys.executable, "-m", "boxy.cli"]
    inner = base + ["serve", model, "--foreground", "--here",
                    "--name", name, "--endpoint-file", str(jobs.endpoint_path(name))]
    if args.location:
        # the profile's runtime/offline/tuning/staging must reach the compute
        # node too, not just its batch directives (finding 38); shared FS
        inner += ["--location", os.path.abspath(args.location)]
    for flag, value in (("--engine", args.engine), ("--image", args.image),
                        ("--runtime", args.runtime), ("--accelerator", args.accelerator)):
        if value:
            inner += [flag, value]
    if args.port:
        inner += ["--port", str(args.port)]
    if args.args:
        inner += ["--"] + list(args.args)
    return shlex.join(inner)


def _serve_submission(args, scheduler_name: str, profile) -> int:
    """The seamless scheduler path: generate a batch script, submit it, follow
    the job to READY, print the endpoint — then get out of the way."""
    import time

    from boxy import jobs, readiness, resolve
    from boxy.location import Location, Resources
    from boxy.schedulers import get_scheduler

    model, name, decisions = resolve.resolve_submission(
        args.model, scheduler_name, name=args.name, require_exists=not args.dryrun)
    for line in decisions:
        print(f"  auto: {line}")

    if profile is not None:
        location = profile
        resources = location.resources
        # explicit flags win over the profile's job geometry (finding 40)
        if args.gpus is not None and args.gpus != resources.gpus_per_node:
            resources = Resources(nodes=resources.nodes, gpus_per_node=args.gpus,
                                  accelerator_type=resources.accelerator_type)
        if args.nodes is not None and args.nodes != resources.nodes:
            resources = Resources(nodes=args.nodes, gpus_per_node=resources.gpus_per_node,
                                  accelerator_type=resources.accelerator_type)
        if resources is not location.resources:
            from dataclasses import replace as dc_replace

            location = dc_replace(location, resources=resources)
            print(f"  auto: job geometry: {resources.nodes} node(s) x "
                  f"{resources.gpus_per_node} GPU(s) (flags override profile)")
    else:
        location = Location(
            name="auto", scheduler=scheduler_name,
            resources=Resources(nodes=args.nodes or 1, gpus_per_node=args.gpus or 0))
    scheduler = get_scheduler(scheduler_name)
    site_args = list(location.scheduler_args)
    for kind, value in (("partition", args.partition), ("account", args.account), ("time", args.time)):
        if value:
            site_args.append(scheduler.site_directive(kind, value))
    site_args += list(args.scheduler_args or [])
    dynamic = getattr(args, "dynamic_flags", [])
    site_args += [scheduler.dynamic_directive(k, v) for s, k, v in dynamic if s == scheduler_name]
    ignored = [f"--{s}-{k}" for s, k, v in dynamic if s != scheduler_name]
    if ignored:
        print(f"warning: ignoring {' '.join(ignored)} (active scheduler is {scheduler_name})",
              file=sys.stderr)

    if args.save_profile:
        print("note: --save-profile is not yet supported for batch submissions "
              "(the box resolves on the compute node)", file=sys.stderr)

    # deterministic name = singleton lock, batch edition. Probe the existing
    # record with ITS OWN scheduler, never the one requested now: a slurm job id
    # is meaningless to `flux jobs` (and vice versa), so querying the wrong
    # scheduler always returns UNKNOWN and wedges resubmission. Field report:
    # `boxy serve ... --scheduler flux` reported "slurm job 1786916 unreachable"
    # because a stale slurm record was probed with the flux state command.
    record = jobs.read_record(name)
    if record:
        rec_sched_name = record.get("scheduler") or scheduler_name
        try:
            rec_scheduler = get_scheduler(rec_sched_name)
        except ValueError:
            rec_scheduler = scheduler
        state = _job_state(rec_scheduler, record["job"])
        mismatch = rec_sched_name != scheduler_name
        if state != "DONE":
            endpoint = jobs.read_endpoint(name)
            if endpoint and state in ("PENDING", "RUNNING"):
                model_id = readiness.wait_ready(endpoint["url"], timeout_s=2, interval_s=0.5)
                if model_id:
                    print(f"### ALREADY SERVING  {endpoint['url']}/v1   "
                          f"(model: {model_id}, {rec_sched_name} job {record['job']})")
                    print(f"###   stop: boxy stop {name}")
                    return 0
            if state == "UNKNOWN":
                hint = (f" That job was submitted under '{rec_sched_name}' but you asked for "
                        f"'{scheduler_name}'; if the old scheduler isn't available here, clear it "
                        f"with boxy stop {name} and rerun."
                        if mismatch else
                        f" Retry when it answers, or boxy stop {name}.")
                print(f"boxy: cannot determine the state of {rec_sched_name} job "
                      f"{record['job']} ({name}) — scheduler unreachable? Not resubmitting.{hint}",
                      file=sys.stderr)
            elif mismatch:
                print(f"boxy: {name} is already submitted as a {rec_sched_name} job "
                      f"({record['job']}, {state}), but you requested {scheduler_name}. "
                      f"Stop it first: boxy stop {name}.", file=sys.stderr)
            else:
                print(f"boxy: {name} is already submitted as {rec_sched_name} job {record['job']} "
                      f"({state}) — watch: boxy list; stop: boxy stop {name}", file=sys.stderr)
            return 1
        if not args.dryrun:
            jobs.remove(name)  # stale record from a finished job (S6: dryrun must not mutate)

    inner = _inner_serve_command(args, model, name)
    script_text = scheduler.batch_script(inner, location, name, str(jobs.log_path(name)), site_args)
    submit = scheduler.submit_command(str(jobs.script_path(name)))
    print(f"### Batch script ({jobs.script_path(name)}):")
    for line in script_text.rstrip().splitlines():
        print(f"    {line}")
    print(f"### Submit Command:\n    {shlex.join(submit)}")
    if args.dryrun:
        return 0

    jobs.script_path(name).write_text(script_text)
    jobs.endpoint_path(name).unlink(missing_ok=True)
    result = subprocess.run(submit, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"boxy: submission failed: {result.stderr.strip() or result.stdout.strip()}", file=sys.stderr)
        return result.returncode
    job_id = scheduler.parse_job_id(result.stdout)
    jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                             "model": model, "submitted_from": socket.gethostname()})
    print(f"### Submitted {scheduler_name} job {job_id}  ({name})")
    print("### Waiting for the job to start and the server to become ready ... "
          "(Ctrl-C detaches; the job keeps running)")

    last_state, ready_deadline = None, None
    last_note = time.time()
    unknown_streak = 0
    try:
        while True:
            state = _job_state(scheduler, job_id)
            unknown_streak = unknown_streak + 1 if state == "UNKNOWN" else 0
            if unknown_streak >= 10:
                # scheduler unreachable / unmapped state: never spin silently
                # forever (r2 audit) — detach and leave the job alone
                print(f"boxy: cannot determine job {job_id}'s state (scheduler unreachable?) — "
                      f"detaching; the job (if alive) keeps running.\n"
                      f"  status: boxy list    log: {jobs.log_path(name)}\n"
                      f"  stop:   boxy stop {name}", file=sys.stderr)
                return 1
            if state != last_state:
                print(f"###   job {job_id}: {state}")
                last_state = state
                last_note = time.time()
            elif time.time() - last_note > 30:
                print(f"###   still waiting (job {job_id}: {state}); log: {jobs.log_path(name)}")
                last_note = time.time()
            endpoint = jobs.read_endpoint(name)
            if endpoint:
                url = endpoint["url"]
                if ready_deadline is None:
                    ready_deadline = time.time() + args.ready_timeout
                    print(f"###   server starting on {endpoint['host']} — "
                          f"waiting for readiness at {url}/v1/models")
                model_id = readiness.wait_ready(url, timeout_s=3, interval_s=1)
                if model_id:
                    print(f"### READY  {url}/v1   (model: {model_id}, {scheduler_name} job {job_id})")
                    print(f"###   try:   curl -s {url}/v1/models")
                    print(f"###   tunnel: ssh -L {endpoint['port']}:{endpoint['host']}:{endpoint['port']} <login-node>")
                    print(f"###   stop:  boxy stop {name}")
                    return 0
                if time.time() > ready_deadline:
                    print(f"boxy: server not ready within {args.ready_timeout:.0f}s (job still {state}). "
                          f"Large models load slowly — watch the log:\n  tail -f {jobs.log_path(name)}\n"
                          f"  then: curl -s {url}/v1/models ; stop: boxy stop {name}", file=sys.stderr)
                    return 1
            if state == "DONE":
                print(f"boxy: job {job_id} ended before the server became ready; last log lines:",
                      file=sys.stderr)
                log_text = _dump_file_tail(jobs.log_path(name))
                _print_diagnosis(log_text)
                jobs.remove(name)
                return 1
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n### Detached — {scheduler_name} job {job_id} keeps running.")
        print(f"###   status: boxy list      endpoint file: {jobs.endpoint_path(name)}")
        print(f"###   stop:   boxy stop {name}")
        return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from boxy import deploy, readiness

    # Seamless scheduler path: MODEL + slurm/flux (via flag or --location
    # profile) submits a batch job unless --foreground pins the attached
    # srun/flux-run mode.
    if args.model and not args.foreground and not args.box:
        scheduler_name = args.scheduler
        profile = None
        if args.location:
            profile = Location.from_toml(args.location)
            if scheduler_name is None and profile.scheduler in ("slurm", "flux"):
                scheduler_name = profile.scheduler
        if scheduler_name in ("slurm", "flux"):
            return _serve_submission(args, scheduler_name, profile)

    box, location, decisions = _resolve_or_load(args)
    for line in decisions:
        print(f"  auto: {line}")
    dynamic = getattr(args, "dynamic_flags", [])
    site_flags = [("partition", getattr(args, "partition", None)),
                  ("account", getattr(args, "account", None)),
                  ("time", getattr(args, "time", None))]
    raw_args = list(getattr(args, "scheduler_args", None) or [])
    if location.scheduler in ("slurm", "flux"):
        # attached srun/flux-run mode consumes the SAME scheduler flags as
        # batch mode (r2 audit: --partition/--account/--time/--scheduler-arg
        # silently vanished here — a mis-billed job on a real site)
        from boxy.schedulers import get_scheduler

        sched_obj = get_scheduler(location.scheduler)
        for kind, value in site_flags:
            if value:
                location.scheduler_args.append(sched_obj.site_directive(kind, value))
        location.scheduler_args.extend(raw_args)
        location.scheduler_args.extend(
            sched_obj.dynamic_directive(k, v) for s, k, v in dynamic if s == location.scheduler)
        ignored = [f"--{s}-{k}" for s, k, v in dynamic if s != location.scheduler]
        if ignored:
            print(f"warning: ignoring {' '.join(ignored)} (active scheduler is {location.scheduler})",
                  file=sys.stderr)
    else:
        ignored = [f"--{kind}" for kind, value in site_flags if value] + raw_args
        ignored += [f"--{s}-{k}" for s, k, v in dynamic]
        if ignored:
            print(f"warning: ignoring {' '.join(ignored)} — no scheduler in play "
                  f"(scheduler is 'none'; add --scheduler slurm|flux)", file=sys.stderr)
    deployment = deploy.plan_serve(box, location, port=args.port, extra_args=args.args, dryrun=args.dryrun)
    if getattr(args, "save_profile", None):
        from dataclasses import replace as dc_replace

        snap_box = deployment.box
        if deployment.port and snap_box.ports != [deployment.port]:
            snap_box = dc_replace(snap_box, ports=[deployment.port])  # r2: box-mode --port was lost
        if args.args:
            print("note: engine extras after `--` are not captured by --save-profile; "
                  "add them to [box.args] in the snapshot for full reproducibility", file=sys.stderr)
        _save_profile(args.save_profile, snap_box, deployment.location)

    port = deployment.port  # parsed from the ACTUAL command (findings 2/10/25/47/55)
    url = f"http://127.0.0.1:{port}"
    runtime_bin = deployment.command[0]
    cname = deployment.box.name
    detach = _detachable(deployment) and not args.foreground and not args.dryrun and args.model
    if detach:
        rc_existing, final_name = _reclaim_or_report(runtime_bin, cname, url)
        if rc_existing is not None:
            return rc_existing
        if final_name != cname:
            from dataclasses import replace

            deployment.command[:] = [
                a.replace(f"--name={cname}", f"--name={final_name}")
                 .replace(f"--label=boxy.box={cname}", f"--label=boxy.box={final_name}")
                for a in deployment.command
            ]
            deployment.box = replace(deployment.box, name=final_name)
            cname = final_name
        if "--rm" in deployment.command:
            # keep the container after a crash so its logs stay inspectable;
            # `boxy stop` removes it.
            deployment.command.remove("--rm")
        deployment.command.insert(2, "-d")  # runtime-managed daemon; boxy waits for readiness

    for warning in deployment.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for prep in deployment.prepare_commands:
        print(f"### Prepare: {shlex.join(prep)}")
    print(f"### Running Command:\n    {shlex.join(deployment.command)}")
    if args.dryrun:
        return 0

    if getattr(args, "endpoint_file", None):
        # batch-job rendezvous: publish host+port over the shared FS so the
        # submitting boxy (on the login node) can find and readiness-gate us
        from boxy import jobs

        jobs.write_endpoint_file(
            args.endpoint_file, name=cname, port=port,
            job_id=os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_JOB_ID", ""))

    if not detach:
        if deployment.location.scheduler == "none":
            host = socket.gethostname() if _inside_allocation() else "127.0.0.1"
            print(f"### Endpoint (once the model loads): http://{host}:{port}/v1")
            if _inside_allocation():
                print(f"###   from your workstation: ssh -L {port}:{host}:{port} <login-node>")
        return deploy.execute(deployment)

    rc = deploy.execute(deployment)  # returns immediately (-d)
    if rc == 0 and args.ready_timeout <= 0:
        # 'launch, don't wait' spelling (finding 27)
        print(f"### Launched (not waiting; --ready-timeout {args.ready_timeout:g})")
        print(f"###   endpoint once loaded: {url}/v1     stop: boxy stop {cname}")
        return 0
    if rc != 0:
        if "-p" in deployment.command:
            # macOS podman-machine: gvproxy refuses a port forward another
            # container (running OR exited) still claims. (Field finding #18.)
            print(f"boxy: launch failed. If the error says 'proxy already running' or 'address already\n"
                  f"in use', another container still claims port {port}:\n"
                  f"  find it:  {runtime_bin} ps -a --filter label=boxy.box\n"
                  f"  stop it:  boxy stop <name>    (or rerun with --port {port + 1})\n"
                  f"  stale forward with no container: {runtime_bin} machine stop && {runtime_bin} machine start",
                  file=sys.stderr)
        return rc
    print(f"### Waiting for readiness at {url}/v1/models ...")
    try:
        model_id = readiness.wait_ready(
            url, timeout_s=args.ready_timeout,
            still_alive=lambda: _container_running(runtime_bin, cname),
        )
    except RuntimeError:
        print(f"boxy: server exited during startup; last log lines from {cname}:", file=sys.stderr)
        log_text = _dump_logs(runtime_bin, cname)
        _print_diagnosis(log_text)
        subprocess.run([runtime_bin, "rm", "-f", cname], capture_output=True)
        return 1
    if model_id is None:
        print(f"boxy: endpoint not ready within {args.ready_timeout:.0f}s — the container is still "
              f"running (large models load slowly). Last log lines:", file=sys.stderr)
        _dump_logs(runtime_bin, cname)
        print(f"  follow: {runtime_bin} logs -f {cname}\n  stop:   boxy stop {cname}", file=sys.stderr)
        return 1
    print(f"### READY  {url}/v1   (model: {model_id})")
    print(f"###   try:  curl -s {url}/v1/models")
    print(f"###   stop: boxy stop {cname}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from boxy import deploy

    box, location = _load(args)
    # argparse.REMAINDER keeps the literal "--" separator; it must not reach
    # the container command.
    user_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    deployment = deploy.plan_run(box, location, user_args, dryrun=args.dryrun)
    return _emit(deployment, args.dryrun)


def cmd_pull(args: argparse.Namespace) -> int:
    """Pre-stage a model (login-node flow: pull where the network is, serve
    where the GPUs are — the store defaults to shared $HOME)."""
    model = args.model
    if not model and args.box:
        box = Box.from_toml(args.box)
        if not box.model:
            print(f"box {box.name}: no model set", file=sys.stderr)
            return 1
        model = box.model
    if not model:
        raise UsageError("usage: boxy pull MODEL   (or: boxy pull --box box.toml)")
    if not model.startswith(TRANSPORT_SCHEMES):
        print(f"model is a path ({model}); nothing to pull (shared-FS flow)")
        return 0
    path = ramalama_shim.pull_model(model, dryrun=args.dryrun)
    print(f"model available at: {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from boxy import deploy
    from boxy.backends import get_backend

    box, location = _load(args)
    backend = get_backend(location.resolve_runtime())
    prepare = backend.prepare(box, location, args.dryrun)
    if not prepare:
        print(f"runtime {backend.name}: uses {backend.image_format} images directly; nothing to build")
        return 0
    for prep in prepare:
        print(f"### Build: {shlex.join(prep)}")
    if args.dryrun:
        return 0
    import subprocess

    for prep in prepare:
        result = subprocess.run(prep)
        if result.returncode != 0:
            return result.returncode
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from boxy import sky_export

    if args.format != "sky":
        print(f"boxy generate: unknown format {args.format!r} (available: sky)", file=sys.stderr)
        return 1
    box, location = _load(args)
    from boxy.deploy import _apply_defaults

    box = _apply_defaults(box, location.resolve_accelerator())  # finding 41: empty image_id
    if location.scheduler != "none":
        print(
            f"warning: location {location.name!r} uses scheduler={location.scheduler!r}; the SkyPilot "
            "path targets cloud/K8s — boxy serves Slurm/Flux natively (use `boxy serve`)",
            file=sys.stderr,
        )
    yaml_text = sky_export.to_sky_task(box, location, port=args.port, serve=args.serve)
    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_text)
        print(f"wrote {args.output}  (launch: sky {'serve up' if args.serve else 'launch'} {args.output})")
    else:
        print(yaml_text, end="")
    return 0


def _container_runtime(location: Location | None) -> str:
    from boxy import resolve

    if location is not None and location.runtime:
        if location.runtime == "apptainer":
            raise RuntimeError(
                "apptainer runs are foreground in the MVP: Ctrl-C the process, "
                "or cancel the job (scancel / flux cancel)"
            )
        return location.runtime
    # viability, not PATH presence: serve picks the WORKING runtime, so the
    # `boxy stop` printed in its banner must pick the same one (finding 24)
    for candidate in ("podman", "docker"):
        if shutil.which(candidate) and resolve._runtime_works(candidate):
            return candidate
    for candidate in ("podman", "docker"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("no container runtime found on host (looked for podman, docker)")


def _run_or_print(cmd: list[str], dryrun: bool) -> int:
    print(f"### Running Command:\n    {shlex.join(cmd)}")
    if dryrun:
        return 0
    import subprocess

    return subprocess.run(cmd).returncode


def cmd_stop(args: argparse.Namespace) -> int:
    from boxy import jobs
    from boxy.schedulers import get_scheduler

    if args.name:
        target = args.name
    elif args.box:
        target = Box.from_toml(args.box).name
    else:
        raise UsageError("usage: boxy stop NAME   (names are printed at serve time and by `boxy list`)")

    record = jobs.read_record(target)
    if record:
        # scheduler-submitted serve: cancel the job (the job step owns the
        # server, so the container dies with it)
        scheduler = get_scheduler(record["scheduler"])
        rc = _run_or_print(scheduler.cancel_command(record["job"]), args.dryrun)
        if not args.dryrun:
            jobs.remove(target)
        return rc

    location = Location.from_toml(args.location) if args.location else None
    runtime = args.runtime or _container_runtime(location)
    if not args.dryrun and _container_exists(runtime, target) and _container_label(runtime, target) != target:
        raise RuntimeError(
            f"container {target!r} was not created by boxy (no boxy.box label) — refusing to "
            f"stop it; use `{runtime} stop {target}` directly if you own it"
        )
    rc = _run_or_print([runtime, "stop", target], args.dryrun)
    if rc == 0 and not args.dryrun:
        # detached serves drop --rm so crash logs survive; clean up here
        subprocess.run([runtime, "rm", target], capture_output=True)
    return rc


def cmd_list(args: argparse.Namespace) -> int:
    from boxy import jobs
    from boxy.schedulers import get_scheduler

    records = jobs.list_records()
    if records:
        print("scheduler jobs:")
        for record in records:
            state = _job_state(get_scheduler(record["scheduler"]), record["job"])
            endpoint = jobs.read_endpoint(record["name"])
            url = f"{endpoint['url']}/v1" if endpoint else "-"
            print(f"  {record['name']}  {record['scheduler']} job {record['job']}  {state}  {url}")
            if state == "DONE" and not args.dryrun:
                jobs.remove(record["name"])  # reap finished jobs from the list
    location = Location.from_toml(args.location) if args.location else None
    try:
        runtime = args.runtime or _container_runtime(location)
    except RuntimeError:
        if records:
            return 0  # jobs listed; no container runtime on this host is fine
        raise
    return _run_or_print([runtime, "ps", "--filter", "label=boxy.box"], args.dryrun)


def cmd_bench(args: argparse.Namespace) -> int:
    from boxy import bench

    from boxy import engines

    box = Box.from_toml(args.box)
    # the port precedence bench sees must match what serve binds (r2 audit):
    # [box.args] port > ports[0] > engine default
    args_port = box.args.get("port")
    default = (int(args_port) if isinstance(args_port, int) and not isinstance(args_port, bool)
               else box.ports[0] if box.ports else engines.default_port(box.engine))
    url = args.url or f"http://127.0.0.1:{default}"
    try:
        batch_sizes = ([int(b) for b in args.batch_sizes.split(",")]
                       if args.batch_sizes else bench.DEFAULT_BATCH_SIZES)
    except ValueError:
        raise UsageError(f"--batch-sizes must be a comma-separated list of integers, "
                         f"got {args.batch_sizes!r}") from None
    if args.dryrun:
        print(f"### Bench plan: url={url} batch_sizes={batch_sizes} max_tokens={args.max_tokens} "
              f"dataset={args.dataset or 'synthetic'}")
        return 0
    import urllib.error

    try:
        report = bench.run_bench(url, batch_sizes, max_tokens=args.max_tokens, dataset=args.dataset)
    except (urllib.error.URLError, OSError, ConnectionError) as e:
        raise RuntimeError(
            f"cannot reach {url} ({getattr(e, 'reason', e)}) — is the box serving? "
            f"(boxy list; or point --url at the endpoint from the READY banner)"
        ) from e
    if args.json:
        print(report.to_json())
    else:
        print(f"# model={report.model} url={report.url} max_tokens={report.max_tokens}")
        print(f"{'batch':>6} {'ok':>4} {'err':>4} {'req/s':>8} {'tok/s':>9} {'p50 ms':>9} {'p95 ms':>9}")
        for r in report.results:
            print(f"{r.batch_size:>6} {r.ok:>4} {r.errors:>4} {r.requests_per_s:>8.2f} "
                  f"{r.tokens_per_s:>9.1f} {r.latency_p50_ms:>9.1f} {r.latency_p95_ms:>9.1f}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_csv())
        print(f"wrote {args.output}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    from boxy import cloud

    box, location = _load(args)
    if location.scheduler != "none":
        print(
            f"warning: location {location.name!r} uses scheduler={location.scheduler!r}; "
            "`boxy launch` delegates to SkyPilot (cloud) — use `boxy serve` for Slurm/Flux",
            file=sys.stderr,
        )
    from boxy.deploy import _apply_defaults

    box = _apply_defaults(box, location.resolve_accelerator())
    if args.down:
        cmd = cloud.launch_command(box, "", serve=args.serve, down=True)
    else:
        yaml_path = cloud.write_task_yaml(box, location, args.port, args.serve, output=args.output)
        print(f"### Task YAML: {yaml_path}")
        cmd = cloud.launch_command(box, yaml_path, serve=args.serve)
    print(f"### Running Command:\n    {shlex.join(cmd)}")
    if args.dryrun:
        return 0
    cloud.ensure_sky()
    import subprocess

    return subprocess.run(cmd).returncode


def _stub(name: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"boxy {name}: {NOT_IN_MVP}", file=sys.stderr)
        return 2

    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boxy", description=__doc__)
    parser.add_argument("--version", action="version", version=f"boxy {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("info", help="show detected accelerator, runtimes, schedulers, TLS state")
    p.add_argument("--net", action="store_true",
                   help="also probe each model registry with the current trust store")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser(
        "serve",
        help="serve MODEL as an OpenAI-compatible endpoint (engine/image/runtime/port auto-resolved)",
    )
    p.add_argument("model", nargs="?", default=None,
                   help="local path or transport URI (hf://, ollama://, oci://); alternative: --box")
    p.add_argument("--box", default=None, help="serve from a box TOML profile instead of MODEL")
    p.add_argument("--location", default=None, help="site TOML profile (scheduler/runtime/accelerator/tuning)")
    p.add_argument("--engine", choices=["llama.cpp", "vllm"], default=None,
                   help="inference engine (default: inferred — GGUF/ollama -> llama.cpp, else vLLM)")
    p.add_argument("--runtime", choices=["podman", "docker", "apptainer"], default=None,
                   help="container runtime (default: first WORKING one found)")
    p.add_argument("--scheduler", choices=["none", "slurm", "flux"], default=None,
                   help="submit as a job via this scheduler (never done automatically)")
    p.add_argument("--accelerator", choices=list(ACCELERATORS), default=None,
                   help="pin the accelerator (needed when submitting GPU jobs from GPU-less login nodes)")
    p.add_argument("--image", default=None, help="container image (default: per engine+accelerator)")
    p.add_argument("--port", type=int, default=None, help="serving port (default: engine default, next free)")
    p.add_argument("--gpus", type=int, default=None, help="GPUs per node for the --scheduler job request")
    p.add_argument("--nodes", type=int, default=None, help="node count for the --scheduler job request")
    p.add_argument("--name", default=None, help="container name (default: derived from the model)")
    p.add_argument("--partition", default=None,
                   help="partition/queue for --scheduler jobs (Slurm --partition, Flux --queue)")
    p.add_argument("--account", default=None,
                   help="account/bank for --scheduler jobs (Slurm --account, Flux --bank)")
    p.add_argument("--time", default=None,
                   help="time limit for --scheduler jobs (e.g. 4:00:00)")
    p.add_argument("--scheduler-arg", action="append", default=[], dest="scheduler_args", metavar="FLAG",
                   help="extra raw scheduler flag for the job (repeatable), "
                        "e.g. --scheduler-arg=--license=tscratch:1")
    p.add_argument("--here", action="store_true",
                   help="allow serving directly on a scheduler login node (bypasses the guard)")
    p.add_argument("--foreground", action="store_true",
                   help="stay attached with engine logs; with --scheduler, uses attached srun/flux-run "
                        "instead of submitting a batch job")
    p.add_argument("--ready-timeout", type=float, default=180.0,
                   help="seconds to wait for the endpoint once the server starts (default 180)")
    p.add_argument("--endpoint-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--save-profile", default=None, metavar="PREFIX",
                   help="write the resolved config to PREFIX.box.toml + PREFIX.location.toml")
    p.add_argument("--dryrun", action="store_true", help="print the command instead of executing it")
    p.add_argument("args", nargs="*", help="extra engine args (put them after --)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run", help="run the box with explicit arguments (raw passthrough; profile mode)")
    _add_common(p)
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the box entrypoint")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="pre-stage a model via RamaLama transports (pull on the login node)")
    p.add_argument("model", nargs="?", default=None,
                   help="transport URI (hf://, ollama://, oci://); alternative: --box")
    p.add_argument("--box", default=None, help="pull the model named by a box TOML profile")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("build", help="build/convert the image for the location's runtime (OCI->SIF)")
    _add_common(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("generate", help="transpile box+location to another orchestrator (cloud path)")
    p.add_argument("format", choices=["sky"], help="output format (sky = SkyPilot task YAML)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--serve", action="store_true", help="add a SkyServe service block (sky serve up)")
    p.add_argument("-o", "--output", default=None, help="write YAML to file instead of stdout")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("bench", help="throughput/latency sweep against a served box (paper's step 5)")
    p.add_argument("--box", required=True)
    p.add_argument("--url", default=None, help="endpoint (default: http://127.0.0.1:<box port>)")
    p.add_argument("--batch-sizes", default=None, help="comma list, default 1,2,4,...,1024")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--dataset", default=None, help="JSON list of prompts or ShareGPT JSON")
    p.add_argument("-o", "--output", default=None, help="write plot-ready CSV here")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("launch", help="launch the box on cloud via SkyPilot (delegated)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--serve", action="store_true", help="managed serving via SkyServe (sky serve up)")
    p.add_argument("--down", action="store_true", help="tear down instead of launching")
    p.add_argument("-o", "--output", default=None, help="also keep the task YAML at this path")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("stop", help="stop (and remove) a boxy-served container")
    p.add_argument("name", nargs="?", default=None,
                   help="container name from the READY banner or `boxy list`; alternative: --box")
    p.add_argument("--box", default=None, help="stop the container named by a box TOML profile")
    p.add_argument("--location", default=None)
    p.add_argument("--runtime", choices=["podman", "docker"], default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("list", help="list running boxy-launched containers")
    p.add_argument("--location", default=None)
    p.add_argument("--runtime", choices=["podman", "docker"], default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_list)

    for name, help_text in (
        ("alloc", "request nodes via the location's scheduler"),
        ("stage", "stage models to shared FS / site-local S3"),
    ):
        p = sub.add_parser(name, help=f"{help_text} (post-MVP)")
        p.set_defaults(func=_stub(name))

    return parser


_DYNAMIC_FLAG = re.compile(r"^--(slurm|flux)-([A-Za-z0-9][A-Za-z0-9-]*)(?:=(.*))?$")


def main(argv: list[str] | None = None) -> int:
    # Everything after a standalone `--` is engine args, verbatim. argparse
    # cannot express this next to optional positionals (a `*` positional only
    # matches one contiguous chunk), so split before parsing.
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    args, unknown = build_parser().parse_known_args(argv)
    # Scheduler flag pass-through: any --slurm-FLAG[=VALUE] / --flux-FLAG[=VALUE]
    # flows into the job request untranslated except for spelling — new
    # scheduler flags never require a boxy change. Values need `=`.
    dynamic: list[tuple[str, str, str | None]] = []
    bad: list[str] = []
    for token in unknown:
        match = _DYNAMIC_FLAG.match(token)
        if match and getattr(args, "subcommand", "") == "serve":
            dynamic.append((match[1], match[2], match[3]))
        else:
            bad.append(token)
    if bad:
        print(f"boxy: error: unrecognized arguments: {' '.join(bad)}\n"
              f"  (scheduler flags pass through as --slurm-FLAG[=VALUE] or --flux-FLAG[=VALUE])",
              file=sys.stderr)
        return 2
    args.dynamic_flags = dynamic
    try:
        if extra:
            if not hasattr(args, "args"):
                raise ValueError(f"'boxy {args.subcommand}' takes no engine args after --")
            args.args = list(args.args or []) + extra
        return args.func(args)
    except UsageError as e:
        print(f"boxy: error: {e}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"boxy: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
