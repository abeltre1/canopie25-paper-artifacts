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

from boxy import config, ramalama_shim, version_string
from boxy.backends import BACKENDS
from boxy.box import TRANSPORT_SCHEMES, Box
from boxy.location import ACCELERATORS, Location

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


def _boto3_present() -> bool:
    try:
        import boto3  # noqa: F401

        return True
    except Exception:
        return False


def _info_section(title: str, rows: list[tuple[str, str]]) -> None:
    """Readable info block: a section header + label/value rows in an aligned
    column (labels keep their trailing colon so greps like 'accelerator:' hold)."""
    print(f"\n{title}")
    width = max(len(label) for label, _ in rows) + 1
    for label, value in rows:
        print(f"  {label + ':':<{width}}  {value}")


def cmd_info(args: argparse.Namespace) -> int:
    print(f"boxy {version_string()}")

    runtimes = [name for name in BACKENDS if shutil.which(name)]
    launchers = [name for name, probe in (("slurm", "srun"), ("flux", "flux")) if shutil.which(probe)]
    _info_section("host", [
        ("accelerator", ramalama_shim.detect_accel()),
        ("container runtimes", ", ".join(runtimes) or "none found"),
        ("schedulers", ", ".join(launchers) or "none found"),
        ("ramalama library", "available" if ramalama_shim.ramalama_available() else "not installed"),
    ])

    ssl_cert = os.environ.get("SSL_CERT_FILE")
    if ssl_cert:
        status = "" if os.path.exists(ssl_cert) else "  (MISSING FILE!)"
        tls = f"SSL_CERT_FILE={ssl_cert}{status}"
    else:
        os_bundle = ramalama_shim.discover_os_ca_bundle()
        if os_bundle:
            tls = (f"system default CA store; boxy auto-merges the OS trust store "
                   f"({os_bundle}) with certifi on pull (disable: BOXY_NO_CA_MERGE=1)")
        else:
            tls = ("system default CA store; no OS CA bundle found — if pulls fail with "
                   "CERTIFICATE_VERIFY_FAILED, set SSL_CERT_FILE to your site CA and persist it")
    from boxy import policy

    allowed = policy.allowed_transports()
    blocked = sorted({policy._canonical(s) for s in policy.REGISTRIES} - set(allowed))
    registries = (f"allowed [{', '.join(allowed)}]   blocked [{', '.join(blocked)}]"
                  + ("" if os.environ.get("BOXY_ALLOW_TRANSPORTS") else "   (default policy)"))
    remote = os.environ.get("BOXY_SSH_HOST")
    net_rows = [("tls", tls), ("registries", registries)]
    proxies = ramalama_shim.active_proxies()
    if proxies:
        net_rows.append(("proxy", "  ".join(f"{k}: {v}" for k, v in proxies.items())
                         + "   (all registry traffic follows these)"))
        if "http" in proxies and "https" not in proxies:
            # the classic ordering bug: `export https_proxy="${http_proxy}"` ran
            # BEFORE http_proxy was set, exporting an EMPTY string (which urllib
            # ignores) — so https traffic (ALL registries) silently goes direct.
            net_rows.append(("proxy WARNING", "http_proxy is set but https_proxy is NOT — registries "
                             "are all https and will bypass the proxy. If your profile does "
                             'https_proxy="${http_proxy}", it must come AFTER http_proxy is set.'))
    if remote:
        net_rows.append(("remote", f"{remote} (BOXY_SSH_HOST — commands run there over SSH)"))
    _info_section("network & trust", net_rows)

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
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        s3 = "present (AWS_ACCESS_KEY_ID env)"
    elif os.environ.get("AWS_PROFILE"):
        s3 = f"profile '{os.environ['AWS_PROFILE']}' (AWS_PROFILE env)"
    elif os.path.exists(os.path.expanduser("~/.aws/credentials")):
        s3 = "present (~/.aws/credentials)"
    else:
        s3 = "not configured (needed to stage s3:// models)"
    auth_rows = [("HuggingFace token", note), ("S3 credentials", s3)]
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if endpoint or os.environ.get("S3_BUCKET_NAME"):
        target = endpoint or "AWS S3"
        bucket = os.environ.get("S3_BUCKET_NAME", "")
        path = os.environ.get("S3_PATH", "")
        loc = f"  bucket {bucket}/{path}" if bucket else ""
        backend = ("boto3" if _boto3_present() else "aws CLI" if shutil.which("aws") else
                   "NONE — pip install boto3 or install the aws CLI")
        auth_rows.append(("S3 staging", f"endpoint {target}{loc}  (via {backend})"))
    _info_section("auth", auth_rows)

    if getattr(args, "net", False):
        print()
        return _probe_registries()
    return 0


def _print_doctor(results) -> int:
    from boxy import doctor

    rows = [(r.name, f"[{r.status}] {r.detail}") for r in results]
    _info_section("doctor", rows)
    fails = [r for r in results if r.status == doctor.FAIL]
    warns = [r for r in results if r.status == doctor.WARN]
    if fails or warns:
        print()
        for r in fails + warns:
            print(f"  {r.status} {r.name}: {r.fix}" if r.fix else f"  {r.status} {r.name}")
    print()
    if fails:
        print(f"doctor: {len(fails)} FAIL, {len(warns)} WARN — fix the FAILs above before serving")
    elif warns:
        print(f"doctor: all critical checks OK ({len(warns)} warning(s) to review)")
    else:
        print("doctor: all checks OK")
    return 1 if fails else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Audit the environment for the issues that actually bite in the field
    (SPEC §8b): proxy/CA/token, runtime, scheduler, accelerator, per-cluster
    state, OOM'd containers, and (with --net) image-registry reach. Prints
    OK/WARN/FAIL + a fix per check; exits non-zero on FAIL. `--ssh user@login`
    audits the CLUSTER over SSH with **no boxy needed there** — plain shell
    probes — so you can check readiness before installing boxy on it."""
    from boxy import doctor, remote

    target = remote.resolve_target(args)
    on_target = bool(target) and target.split("@")[-1].split(".")[0] == socket.gethostname().split(".")[0]
    if target and not on_target and not os.environ.get(remote.ENV_ACTIVE):
        if remote.ensure_master(target) != 0:
            print(f"boxy: could not open an SSH session to {target} — check the host, VPN, and "
                  f"that you completed the OTP/YubiKey prompt", file=sys.stderr)
            return 1
        print(f"boxy doctor — remote audit of {target} (no boxy required on the cluster)")
        results = doctor.remote_checks(lambda cmd: remote.ssh_capture(target, cmd))
        return _print_doctor(results)

    print(f"boxy {version_string()}")
    return _print_doctor(doctor.run_checks(net=getattr(args, "net", False)))


def _probe_registries() -> int:
    """`boxy info --net`: try each ALLOWED model registry with the CURRENT
    trust store (after boxy's certifi merge, like a real pull). Registries
    outside the policy allowlist are not even probed. Any HTTP response proves
    TLS worked; 401/404 are fine, but a 403 on an anonymous front-door GET
    means the network/site refuses this client — flagged as BLOCKED."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from boxy import policy

    ramalama_shim.ensure_trust_bundle()
    proxies = ramalama_shim.active_proxies()
    if proxies:
        print("net: proxy      " + "  ".join(f"{k}: {v}" for k, v in proxies.items())
              + "   (probes below go THROUGH this)")
    failures = 0
    kinds_seen: set[str] = set()
    probe_status: dict[str, object] = {}  # scheme -> HTTP code or error string
    for scheme, url in policy.registry_probes():
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                probe_status[scheme] = resp.status
                print(f"net: {scheme:10s} {url}  OK (HTTP {resp.status})")
        except urllib.error.HTTPError as e:
            probe_status[scheme] = e.code
            if e.code == 403:
                # an UNAUTHENTICATED GET of the registry front door never
                # legitimately 403s — the network path (proxy policy, IP block)
                # is refusing this client. Not certificates, not tokens.
                print(f"net: {scheme:10s} {url}  BLOCKED (TLS fine; HTTP 403 — this network or the "
                      f"site refuses the request: proxy/Zscaler policy or an IP-level block. "
                      f"Try on/off VPN or another network.)")
                failures += 1
            else:
                print(f"net: {scheme:10s} {url}  OK (TLS fine; HTTP {e.code})")
        except Exception as e:
            reason = getattr(e, "reason", e)
            probe_status[scheme] = str(reason)
            print(f"net: {scheme:10s} {url}  FAIL ({reason})")
            kinds_seen.add(ramalama_shim.net_failure_kind(reason))
            host = urllib.parse.urlsplit(url).hostname or ""
            if "CERTIFICATE_VERIFY_FAILED" in str(reason) and host:
                issuer = _tls_issuer(host)
                if issuer:
                    print(f"net:            the cert this shell SEES for {host} was issued by: {issuer}")
                    print("net:            that issuer's root CA is NOT in your trust bundle — if it's your"
                          " corporate interceptor (Zscaler etc.), append ITS root to SSL_CERT_FILE."
                          " Interceptors often bypass some hosts, which is why other registries pass.")
            failures += 1
    token, source = ramalama_shim.effective_hf_token()
    if token:
        # the definitive "did my token take effect" answer: ask HF who the
        # EFFECTIVE token belongs to (same resolution as the pull itself).
        # Runs AFTER the anonymous probes so a 403 can be attributed correctly:
        # if the UNAUTHENTICATED probe was also refused, it's the network, not
        # the token (field report: a good token blamed while Zscaler 403'd HF).
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
            if e.code == 401:
                print(f"net: hf-auth    token from {source} is INVALID (HTTP 401) — "
                      f"HF rejects it; gated/private pulls WILL fail. Generate a fresh token at "
                      f"https://huggingface.co/settings/tokens and re-export HF_TOKEN.")
                failures += 1
            elif e.code == 403 and probe_status.get("hf://") == 403:
                print(f"net: hf-auth    whoami got HTTP 403 — but so did the UNAUTHENTICATED hf:// "
                      f"probe above, so huggingface.co is refusing this CLIENT (network/proxy policy), "
                      f"not the token from {source}. The token is probably fine; fix the network first.")
            elif e.code == 403:
                print(f"net: hf-auth    token from {source} is REJECTED (HTTP 403) — HF answers "
                      f"anonymous requests fine, so the token exists but lacks permission. Regenerate "
                      f"it with 'read' scope at https://huggingface.co/settings/tokens and re-export "
                      f"HF_TOKEN.")
                failures += 1
            else:
                print(f"net: hf-auth    could not validate (HTTP {e.code})")
        except Exception as e:
            print(f"net: hf-auth    could not validate ({getattr(e, 'reason', e)})")
    if failures:
        # explain each KIND of failure actually observed — DNS/proxy/conn get
        # network guidance, never the cert remedy (DNS is upstream of TLS).
        for kind in sorted(k for k in kinds_seen if k):
            print(f"net: {kind}: {ramalama_shim.network_remedy(kind)}")
        print("net: FAILing registries cannot be pulled from this shell "
              "(certificate + proxy + offline notes: RUNBOOK §2.1)")
    return 1 if failures else 0


def _tls_issuer(host: str, port: int = 443) -> str:
    """Who signed the certificate this shell actually SEES for `host`? (best
    effort; '' on any failure). Corporate TLS interception (Zscaler) swaps
    chains per-host — some hosts bypassed, some intercepted — so one registry
    can verify while another fails with the SAME bundle. Naming the issuer
    turns a bare CERTIFICATE_VERIFY_FAILED into 'your bundle lacks THIS root'."""
    import ssl

    ctx = ssl._create_unverified_context()  # look, don't trust: report-only handshake
    try:
        with socket.create_connection((host, port), timeout=6) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
        if not der:
            return ""
        result = subprocess.run(["openssl", "x509", "-noout", "-issuer"],
                                input=ssl.DER_cert_to_PEM_cert(der),
                                capture_output=True, text=True, timeout=6)
        if result.returncode != 0:
            return ""
        return result.stdout.strip().removeprefix("issuer=").strip()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return ""


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
                distributed=getattr(args, "distributed", None),
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
        distributed=getattr(args, "distributed", None),
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
    """Name-collision policy for LOCAL container serves.

    The name is deterministic (from the model), so without --unique it is a
    per-model singleton: rerunning `boxy serve MODEL` REDEPLOYS — the existing
    instance of the same name is a duplicate and is replaced with a fresh one
    (user request, 2026-07). To run a SECOND instance alongside, use --unique
    (a timestamped name + auto-incremented port; this function isn't reached
    because that name doesn't exist yet).

      * name held by OUR container, running  -> replace it (stop+rm, relaunch);
      * ours, exited                         -> dump its logs, remove, relaunch;
      * NOT created by boxy                  -> auto-suffix (-2, -3, ...) and
        proceed — someone else owns that name, colliding is pure friction.

    Returns (exit code | None to launch, final container name)."""
    if not _container_exists(runtime, name):
        return None, name
    if _container_label(runtime, name) != name:
        # foreign owner: walk the suffixes — reuse OUR suffixed instance's name
        # if one exists (so we replace it, not a stranger's), else first free.
        for i in range(2, 10):
            candidate = f"{name}-{i}"
            if not _container_exists(runtime, candidate):
                print(f"  auto: name: {candidate} ({name!r} exists but was not created by boxy)")
                return None, candidate
            if _container_label(runtime, candidate) == candidate:
                return _reclaim_or_report(runtime, candidate, url)
        raise RuntimeError(f"container names {name} and {name}-2..-9 are all taken; pass --name")
    if _container_running(runtime, name):
        # a duplicate of a per-model singleton: replace it. `rm -f` kills + removes
        # in one step (no 10s SIGTERM wait). Tell the user how to run a 2nd instead.
        print(f"boxy: replacing the running instance {name} — rerun without --unique redeploys it. "
              f"To run a SECOND instance alongside, use: boxy serve MODEL --unique", file=sys.stderr)
        subprocess.run([runtime, "rm", "-f", name], capture_output=True)
        return None, name
    print("boxy: found an exited container from a previous attempt; its last log lines:", file=sys.stderr)
    _dump_logs(runtime, name)
    _diagnose_container(runtime, name)
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
    """Scan engine-log text for a known failure signature and, if found, print a
    plain-language fix instead of leaving the user with a raw trace."""
    from boxy import diagnostics

    hint = diagnostics.diagnose(log_text)
    if hint:
        print(hint, file=sys.stderr)


def _diagnose_container(runtime: str, name: str, scan_tail: int = 400) -> None:
    """Diagnose over a WIDE log window (not just the ~50 human-printed lines):
    vLLM prints a generic 'Engine core initialization failed. See root cause
    above' wrapper, and the actual exception can be dozens of lines earlier."""
    result = subprocess.run([runtime, "logs", "--tail", str(scan_tail), name],
                            capture_output=True, text=True)
    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    # An OOM-killed container often leaves EMPTY logs (SIGKILL, no final line).
    # Inspect the exit code so the host-OOM diagnosis still fires: 137 = 128+9
    # (SIGKILL), and podman/docker set OOMKilled=true when the cgroup reaped it.
    inspect = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}", name],
        capture_output=True, text=True)
    if inspect.returncode == 0 and inspect.stdout.split():
        code, _, ooms = inspect.stdout.strip().partition(" ")
        if code == "137" or ooms.strip().lower() == "true":
            log_text += "\nOOMKilled exit 137"  # synthetic signal for the host-oom rule
    _print_diagnosis(log_text)


def _diagnose_file(path, scan_tail: int = 400) -> None:
    try:
        lines = open(path, errors="replace").read().splitlines()[-scan_tail:]
    except OSError:
        return
    _print_diagnosis("\n".join(lines))


def _unique_instance_name(base: str) -> str:
    """Append a readable, collision-checked suffix so N launches of the SAME
    model coexist: each instance gets its own record / endpoint / script / log
    (all keyed by this name) and its own scheduler job. Format:
    <base>-<mmdd-HHMMSS>-<hex> — sortable by launch time, unique by the hex."""
    import secrets
    import time as _time

    from boxy import jobs

    stamp = _time.strftime("%m%d-%H%M%S")
    for _ in range(20):
        candidate = f"{base}-{stamp}-{secrets.token_hex(2)}"
        if jobs.read_record(candidate) is None and not jobs.endpoint_path(candidate).exists():
            return candidate
    return f"{base}-{stamp}-{secrets.token_hex(6)}"


def _auto_unique(args) -> bool:
    """Turnkey: when a LIVE instance of this model already holds the name, start
    an independent instance instead of blocking. On by default; opt out with
    --no-auto-unique, BOXY_AUTO_UNIQUE=false, or config serve.auto_unique=false
    to keep the strict singleton (a re-run reports 'already submitted')."""
    if getattr(args, "no_auto_unique", False):
        return False
    from boxy import config

    return config.get_bool("serve.auto_unique")


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


# Minimum time to wait for a scheduler-submitted server to answer once the job
# is RUNNING — a compute-node image pull + LLM weight-load + CUDA-graph capture
# routinely takes 10-20 min (the old 180s default gave up mid-load). A crash
# short-circuits this via the state==DONE check, so the floor only ever applies
# to genuinely-slow loads; --ready-timeout / BOXY_READY_TIMEOUT raise it further.
_SCHED_READY_FLOOR = 1200.0


def _start_local_health_watch(port: int, endpoint_file, model_hint: str = "") -> None:
    """On the node that RUNS the server (a compute node in a batch job), poll
    `http://localhost:port/health` in a background daemon thread and, the moment
    it answers, flip the shared-FS endpoint file to ready (mark_endpoint_ready).

    This is the ideal readiness probe — same host as the server, so it never hits
    the corporate proxy or a compute-node-not-routable-from-login-node wall that
    the submitting boxy's own probe does. The submitting loop just watches the
    file. Best-effort and self-terminating: it stops on first success, and dies
    with the process when the foreground container exits."""
    import threading
    import time

    from boxy import jobs, readiness

    url = f"http://127.0.0.1:{port}"

    def _run() -> None:
        # the job walltime bounds this; a daemon thread also dies with the process.
        for _ in range(200000):
            mid = readiness.probe_once(url, timeout=2)
            if mid:
                jobs.mark_endpoint_ready(endpoint_file, model=(mid if mid != "ready" else model_hint))
                return
            time.sleep(3)

    threading.Thread(target=_run, daemon=True).start()


def _last_log_line(path, maxlen: int = 140) -> str:
    """The last non-empty line of a job log, truncated — shown live during the
    readiness wait so a long load reads as progress, not a silent spinner."""
    try:
        with open(path, errors="replace") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    last = lines[-1]
    return last if len(last) <= maxlen else last[:maxlen - 1] + "…"


# Ordered phase markers scanned in the job log (most-progressed match wins). Each
# maps a regex to (phase-label, optional fraction group spec). The engines
# (vLLM/llama.cpp) and the container runtime print these as a big model loads;
# surfacing them turns the readiness wait into a real progress display instead of
# a silent spinner (field request). group spec: ("pct", i) => group i is 0-100;
# ("ratio", n, d) => groups n/d; None => no fraction (indeterminate phase).
_PROGRESS_MARKERS: tuple[tuple[str, str, object], ...] = (
    (r"Application startup complete|Uvicorn running on|Started server process|"
     r"llama server listening|HTTP server listening", "server starting", None),
    (r"Capturing CUDA graph.*?(\d+)\s*%", "capturing CUDA graphs", ("pct", 1)),
    (r"Loading safetensors checkpoint shards:\s*(\d+)\s*%.*?(\d+)/(\d+)",
     "loading weights", ("ratio", 2, 3)),
    (r"Loading safetensors checkpoint shards:\s*(\d+)\s*%", "loading weights", ("pct", 1)),
    (r"load_tensors:|llama_model_loader:\s*loaded|loading model", "loading weights", None),
    (r"(?:Downloading|Fetching).*?(\d+)\s*%", "downloading model", ("pct", 1)),
    (r"Copying (?:blob|config)|Pulling|Storing signatures|Writing manifest",
     "pulling container image", None),
    (r"Automatically detected platform|Initializing (?:an? )?LLM engine|"
     r"vLLM API server version|Started engine", "engine init", None),
)


def _parse_load_progress(path, scan: int = 200) -> tuple[str, float | None]:
    """Scan the tail of a job log and return (phase-label, fraction|None) for the
    most-progressed recognizable marker — the readiness-wait progress signal. Pure
    (regex over the file text); ('', None) when nothing is recognized."""
    try:
        with open(path, errors="replace") as fh:
            text = "\n".join(fh.read().splitlines()[-scan:])
    except OSError:
        return "", None
    best_rank, best = -1, ("", None)
    for rank, (pat, label, spec) in enumerate(_PROGRESS_MARKERS):
        m = None
        for m in re.finditer(pat, text):  # last occurrence in the tail
            pass
        if not m:
            continue
        frac: float | None = None
        try:
            if spec and spec[0] == "pct":
                frac = max(0.0, min(1.0, int(m.group(spec[1])) / 100.0))
            elif spec and spec[0] == "ratio":
                num, den = int(m.group(spec[1])), int(m.group(spec[2]))
                frac = max(0.0, min(1.0, num / den)) if den else None
        except (ValueError, ZeroDivisionError, IndexError):
            frac = None
        # a later marker in the list = a later phase; prefer it (server starting
        # ranks 0 but is terminal-most — handled by putting it first so it wins).
        if rank == 0:  # 'server starting' is the most-progressed signal
            return label, frac
        if rank > best_rank:
            best_rank, best = rank, (label, frac)
    return best


def _progress_bar(frac: float | None, width: int = 14) -> str:
    """A compact ASCII meter: `[############----]  72%`, or an indeterminate
    `[· · · · · ]` when the fraction is unknown."""
    if frac is None:
        return "[· · · · · ·]"
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    return f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:3.0f}%"


def _fmt_elapsed(secs: float) -> str:
    secs = int(max(0.0, secs))
    return f"{secs // 60:d}:{secs % 60:02d}"


def _phase_for(state: str, endpoint_seen: bool, marker: str) -> str:
    """Human phase for the progress line, from the scheduler job state, whether the
    endpoint file exists yet, and any log marker."""
    if state in ("PENDING", "CONFIGURING"):
        return "QUEUED"
    if marker:
        return marker.upper()
    if endpoint_seen:
        return "LOADING"
    return "STARTING"


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


def _inner_serve_command(args, model: str, name: str, *, port: int | None = None,
                         visible_gpus: str | None = None, gpus: int | None = None,
                         forward_geometry: bool = True,
                         extra_engine_args: list[str] | None = None) -> str:
    """The command the batch job runs ON the compute node: boxy itself, in
    foreground (the job step owns the server), re-resolving hardware there
    and publishing its endpoint over the shared filesystem.

    Overrides (for the co-located --replicas fan-out): `port` pins the serve port,
    `visible_gpus` pins the container to specific GPU ids, `gpus` forwards a
    per-replica GPU count, `forward_geometry=False` suppresses the --nodes/--gpus/
    --distributed pass-through (each replica is a single-node instance), and
    `extra_engine_args` are appended after `--` (e.g. --tensor-parallel-size=R)."""
    from boxy import jobs

    boxy_bin = shutil.which("boxy")
    base = [boxy_bin] if boxy_bin else [sys.executable, "-m", "boxy.cli"]
    inner = base + ["serve", model, "--foreground", "--here",
                    "--name", name, "--endpoint-file", str(jobs.endpoint_path(name))]
    if args.location:
        # the profile's runtime/offline/tuning/staging must reach the compute
        # node too, not just its batch directives (finding 38); shared FS
        inner += ["--location", os.path.abspath(args.location)]
    mdir = getattr(args, "models_dir", None)
    if mdir:
        # absolutize: the compute node must download to the SAME shared-FS path
        inner += ["--models-dir", os.path.abspath(mdir)]
    if forward_geometry:
        # geometry the compute node needs to derive Ray parallelism (currently only
        # the batch directives carried these); the head-node boxy re-decides distributed.
        if args.nodes:
            inner += ["--nodes", str(args.nodes)]
        if args.gpus:
            inner += ["--gpus", str(args.gpus)]
        if getattr(args, "distributed", None) is True:
            inner += ["--distributed"]
        elif getattr(args, "distributed", None) is False:
            inner += ["--no-distributed"]
    if gpus is not None:
        inner += ["--gpus", str(gpus)]
    if visible_gpus is not None:
        inner += ["--visible-gpus", visible_gpus]
    if getattr(args, "trust_remote_code", False):
        inner += ["--trust-remote-code"]  # re-applied engine-aware on the compute node
    for flag, value in (("--engine", args.engine), ("--image", args.image),
                        ("--registry", getattr(args, "registry", None)),
                        ("--runtime", args.runtime), ("--accelerator", args.accelerator)):
        if value:
            inner += [flag, value]
    resolved_port = port if port is not None else args.port
    if resolved_port:
        inner += ["--port", str(resolved_port)]
    tail = list(extra_engine_args or []) + list(args.args or [])
    if tail:
        inner += ["--"] + tail
    return shlex.join(inner)


def _resolve_proxy(args) -> str:
    """The explicit proxy URL to propagate, turnkey-style: `--proxy` wins, else
    config `network.proxy` / $BOXY_PROXY. Empty => fall back to the ambient
    http(s)_proxy env (captured by raw_proxy_env), so a login node that already
    exports a proxy needs no flag and no config at all."""
    flag = getattr(args, "proxy", "") or ""
    if flag:
        return flag
    from boxy import config

    return config.get_str("network.proxy")


def _proxy_prefix(args) -> str:
    """`env VAR=val ...` prefix for the COMPUTE-NODE command, so its host-side
    `podman pull` (and the inner boxy) reach the corporate proxy — the usual fix
    for a ghcr.io 403 on an isolated compute node. `--proxy` / config
    network.proxy wins; otherwise the submitter's ambient proxy env is carried
    over automatically. '' when nothing is configured."""
    from boxy import ramalama_shim

    proxies = ramalama_shim.raw_proxy_env(_resolve_proxy(args))
    if not proxies:
        return ""
    return "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in proxies.items()) + " "


def _apply_proxy_env(args) -> None:
    """Export the resolved proxy (--proxy / config network.proxy) into THIS
    process's env so the LOGIN-NODE model download (RamaLama / huggingface_hub,
    run in-process during resolve_model) reaches the corporate proxy too — and,
    over --ssh, so run_remote's ambient-env capture forwards it to the cluster.
    Idempotent; no-op when nothing beyond the ambient env is configured."""
    proxy = _resolve_proxy(args)
    if not proxy:
        return
    for key, val in ramalama_shim.raw_proxy_env(proxy).items():
        os.environ[key] = val


def _apply_bind_host_env(args) -> None:
    """Realize the flag>env>file>default layering for the bind host: a `--bind-host`
    flag is exported as BOXY_BIND_HOST so every downstream config.get("network.
    bind_host") (engines, router, foreground/agentless re-runs) sees it. No-op
    without the flag."""
    host = getattr(args, "bind_host", "") or ""
    if host:
        os.environ["BOXY_BIND_HOST"] = host


def cmd_cards(args: argparse.Namespace) -> int:
    """List the built-in deployment cards: MODEL cards (per-model geometry so
    `serve MODEL` needs no flags) and SYSTEM cards (per-system-type profiles for
    --system)."""
    from boxy import cards

    print("model cards (auto GPUs/engine/args for a model — `boxy serve <id> --scheduler slurm`):")
    seen = set()
    for c in cards.load_cards():
        if c.card_name in seen:
            continue
        seen.add(c.card_name)
        geo = f"{c.gpus} GPU" + ("s" if c.gpus != 1 else "") if c.gpus else "size-heuristic"
        eng = f", {c.engine}" if c.engine else ""
        print(f"  {c.match:<40} {geo}{eng}  [{c.source}]")
    print("  (any other model -> sized from its name, e.g. '-70B' -> 4 GPUs)")

    print("\nsystem cards (deployment profile per system type — `boxy serve MODEL --system <name>`):")
    by_type: dict[str, list[str]] = {}
    for cname, typ in cards.system_card_names():
        by_type.setdefault(typ, []).append(cname)
    for typ in sorted(by_type):
        print(f"  {typ:<12} {', '.join(sorted(by_type[typ]))}")

    from boxy import appcards

    print("\napp cards (HPC applications/benchmarks — `boxy app <name> --ssh <cluster>`):")
    seen_apps: set[str] = set()
    for a in appcards.load_cards():
        if a.name in seen_apps:
            continue
        seen_apps.add(a.name)
        geo = f"{a.nodes}x{a.tasks_per_node}"
        print(f"  {a.name:<24} {a.kind:<10} {geo:<6} {a.summary}  [{a.source}]")

    print("\nservice cards (persistent cloud services boxy emits — `boxy generate <name>`):")
    for sname, sdesc in (("flux-mcp", "Flux MCP server as a persistent OpenShift service (Helm chart)"),
                         ("relay", "chisel reverse-tunnel relay — the everyone-URL share ingress")):
        print(f"  {sname:<24} {sdesc}")

    print("\ndrop your own in ~/.config/boxy/cards/{models,systems,apps}/ (they win over built-ins).")
    return 0


def cmd_app(args: argparse.Namespace) -> int:
    """Run an HPC application/benchmark from an APP CARD — the deployment-OS
    counterpart of `boxy serve MODEL`: the card carries what to build (a spack
    spec) or pull (a container image), the geometry, and the run lines; boxy
    resolves the site (scheduler/account/partition/time) with the same zero-flag
    machinery the serve path uses and submits a self-contained batch script.
    Agentless over --ssh: the cluster needs only spack (or podman), never boxy."""
    from boxy import appcards

    if not getattr(args, "name", None):
        rows = appcards.load_cards()
        if not rows:
            print("boxy: no app cards found — add one under ~/.config/boxy/cards/apps/", file=sys.stderr)
            return 1
        print("app cards (run one: `boxy app <name> --ssh <cluster>`):")
        seen: set[str] = set()
        for a in rows:
            if a.name in seen:
                continue
            seen.add(a.name)
            print(f"  {a.name:<24} {a.kind:<10} {a.nodes}x{a.tasks_per_node:<4} {a.summary}  [{a.source}]")
        return 0

    card = appcards.find_card(args.name)
    if card is None:
        names = ", ".join(sorted({c.name for c in appcards.load_cards()})) or "(none)"
        print(f"boxy: no app card named {args.name!r} — available: {names}. "
              f"Add your own under ~/.config/boxy/cards/apps/ (see `boxy cards`).", file=sys.stderr)
        return 2

    target = getattr(args, "ssh", None) or os.environ.get("BOXY_SSH_HOST", "")
    if target:
        return _app_agentless_ssh(args, card, target)
    return _app_local(args, card)


def _app_site_args(args, card, scheduler, scheduler_name: str, target: str = "",
                   host: str = "") -> list[str]:
    """Site directives for an app job, resolved the same way the serve path does:
    account (config/env, else mywcid probed on the cluster), partition (explicit
    or auto-ranked from the cluster), time (flag > card > config default), and the
    Slurm license default. Explicit flags always win."""
    from boxy import remote, site

    site_args: list[str] = []
    acct, awhy = site.resolve_account(getattr(args, "account", None))
    if not acct and target:
        rc, out = remote.ssh_capture(target, site.remote_account_probe(), timeout=20)
        rows = site.parse_account_rows(out) if rc == 0 else []
        acct, awhy = _pick_remote_account(args, rows, host, target)
    if acct:
        site_args.append(scheduler.site_directive("account", acct))
        print(f"  auto: account: {acct} (via {awhy})")
    need_gpu = card.gpus_per_node > 0
    part = getattr(args, "partition", None)
    if target and site.partition_mode(part) in ("auto", "all"):
        rc, out = remote.ssh_capture(target, site.remote_partition_probe(scheduler_name), timeout=20)
        value, pwhy = site.rank_remote_partitions(out, scheduler_name, need_gpu) if rc == 0 else ("", "")
        if value:
            names = [p for p in value.split(",") if p]
            pick, pnote = _pick_remote_partition(args, names, host, target, scheduler_name)
            part = pick
            print(f"  auto: partition: {part} (via {pnote or f'{scheduler_name} on {host}: {pwhy}'})")
    if part and site.partition_mode(part) == "set":
        if scheduler_name == "flux" and "," in part:
            part = part.split(",")[0].strip()
        site_args.append(scheduler.site_directive("partition", part))

    explicit_t = getattr(args, "time", None) or card.time
    t, twhy = (explicit_t, "--time / the app card") if explicit_t else site.resolve_time(None)
    if t:
        site_args.append(scheduler.site_directive("time", t))
        print(f"  auto: time: {t} (via {twhy} — the scheduler stops the job at this walltime)")
    if scheduler_name == "slurm":
        lic, lwhy = site.resolve_license(getattr(args, "license", None))
        if lic:
            site_args.append(scheduler.site_directive("license", lic))
            print(f"  auto: license: {lic} (via {lwhy})")
    site_args += list(getattr(args, "scheduler_args", None) or [])
    return site_args


_SRC_ARCHIVE_RE = re.compile(r"\.(tar\.(gz|bz2|xz)|tgz|zip)$")


def _spack_fetch_blocked(text: str) -> bool:
    """The cluster-egress-filter signature for a failed spack source fetch:
    every fetcher died on a 403/block page (field: Zscaler CATEGORY_DENIED for
    mirror.spack.io AND the package's upstream on the cluster)."""
    low = text.lower()
    return "all fetchers failed" in low and ("block-page" in low or "block-message" in low
                                             or "403" in low or "forbidden" in low)


def _extract_spack_sources(text: str) -> tuple[list[str], str]:
    """From a failed job log: (source-archive URLs to try laptop-side, the
    relative mirror path to store the archive at). Egress-filter block-page URLs
    are unwrapped from their percent-encoded url= parameter. The store path
    prefers spack's sha-addressed _source-cache layout (the digest doubles as an
    integrity check); else the per-package <name>/<name>-<version>.<ext> layout
    parsed from the failing stage name."""
    import urllib.parse as _up

    urls: list[str] = []
    for m in re.finditer(r"https?://\S+", text):
        u = m.group(0).rstrip(":,)\"'")
        if "block" in _up.urlparse(u).netloc:
            wrapped = _up.parse_qs(_up.urlparse(u).query).get("url", [""])[0]
            if not wrapped:
                continue
            u = _up.unquote(wrapped)
        if _SRC_ARCHIVE_RE.search(u) and u not in urls:
            urls.append(u)
    # the spack mirror URL first: its path embeds the sha256 we verify against
    urls.sort(key=lambda u: 0 if "_source-cache/archive/" in u else 1)
    rel = ""
    for u in urls:
        m = re.search(r"_source-cache/archive/[0-9a-f]{2}/[0-9a-f]{64}\.[a-z0-9.]+$", u)
        if m:
            rel = m.group(0)
            break
    if not rel and urls:
        m = re.search(r"failed for (?:spack-stage-)?([a-z0-9-]+?)-([0-9][0-9a-z.]*)-[a-z0-9]{20,}",
                      text)
        if m:
            ext = next((c for c in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")
                        if urls[0].endswith(c)), ".tar.gz")
            rel = f"{m.group(1)}/{m.group(1)}-{m.group(2)}{ext}"
    return urls, rel


def _download_bytes(url: str, timeout: int = 600) -> bytes:
    """Fetch a URL on THIS machine through the ambient proxy + CA bundle (the
    same opener `generate card` uses against the Hub)."""
    from boxy import cardgen

    with cardgen._opener().open(url, timeout=timeout) as r:
        return r.read()


def _maybe_spack_source_heal(target: str, host: str, tail: str, mirror_remote: str,
                             downloader=None) -> bool:
    """Egress-filter heal for spack app jobs: the CLUSTER can't download the
    source archive (category-blocked 403), but the LAPTOP usually can — so pull
    it here, sha256-verify when the mirror path names the digest, push it into
    the job's file:// spack mirror on the shared FS, and tell the caller to
    resubmit. False = not stageable (the caller reports the log as-is)."""
    import hashlib

    from boxy import remote

    urls, rel = _extract_spack_sources(tail)
    if not urls or not rel:
        return False
    print("boxy: the cluster's egress filter blocked spack's source download — fetching "
          "the archive on YOUR machine and staging it into a spack mirror on the "
          "cluster's shared FS ...", file=sys.stderr)
    data, src = None, ""
    for u in urls:
        try:
            data = (downloader or _download_bytes)(u)
            src = u
            break
        except Exception as e:  # noqa: BLE001 — every fetcher gets its chance
            print(f"boxy:   {u}: {e}", file=sys.stderr)
    if data is None:
        print(f"boxy: the source is unreachable from this machine too — download the "
              f"archive yourself and place it at {host}:{mirror_remote}/{rel}, then rerun.",
              file=sys.stderr)
        return False
    digest = re.search(r"([0-9a-f]{64})", rel)
    if digest and hashlib.sha256(data).hexdigest() != digest.group(1):
        print("boxy: the downloaded archive does not match spack's sha256 — refusing to "
              "stage it (a filter may have served an HTML block page instead).", file=sys.stderr)
        return False
    dest = f"{mirror_remote}/{rel}"
    if remote.push_file(target, dest, data) != 0:
        return False
    print(f"boxy: staged {src} -> {host}:{dest} ({len(data) // 1024} KiB) — resubmitting.",
          file=sys.stderr)
    return True


_ARCHIVE_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")


def _stage_source_file(args, target: str, host: str, mirror_remote: str) -> bool | None:
    """--stage-source: push a hand-downloaded source archive into the job's
    file:// spack mirror, addressed by ITS OWN sha256 — spack fetches by digest,
    so the right file is found and a wrong one is simply never consulted. The
    escape hatch when the egress filter blocks the cluster AND the laptop's
    batch fetchers: a BROWSER download passes the filter's user auth (the block
    page's 'noauth-useragent' is exactly the batch-client case).
    None = flag not given; True = staged; False = hard error."""
    import hashlib

    from boxy import remote

    path = getattr(args, "stage_source", None)
    if not path:
        return None
    p = os.path.expanduser(path)
    if not os.path.isfile(p):
        print(f"boxy: --stage-source {path}: no such file", file=sys.stderr)
        return False
    base = os.path.basename(p)
    ext = next((c for c in _ARCHIVE_EXTS if base.endswith(c)), "")
    if not ext:
        print(f"boxy: --stage-source {path}: not a source archive "
              f"(expected one of {', '.join(_ARCHIVE_EXTS)})", file=sys.stderr)
        return False
    with open(p, "rb") as f:
        data = f.read()
    sha = hashlib.sha256(data).hexdigest()
    dest = f"{mirror_remote}/_source-cache/archive/{sha[:2]}/{sha}{ext}"
    if remote.push_file(target, dest, data) != 0:
        return False
    print(f"  auto: source: staged {base} -> {host}:{dest} ({len(data) // 1024} KiB, "
          f"digest-addressed — spack uses it iff it matches the spec's sha256)")
    return True


def _app_agentless_ssh(args, card, target: str) -> int:
    """Agentless app run over --ssh: render the card's batch script laptop-side,
    push + submit over the one master, then (unless --detach) poll the job to
    completion and print the log — so `boxy app osu-benchmarks --ssh kahuna`
    IS the benchmark run, results included."""
    import time as _time

    from boxy import appcards, jobs, remote, site
    from boxy.schedulers import get_scheduler

    host = target.split("@")[-1]
    if not args.dryrun and remote.ensure_master(target) != 0:
        print(f"boxy: could not open an SSH session to {target} — check the host, your VPN, "
              f"and that you completed any OTP/YubiKey prompt", file=sys.stderr)
        return 1

    scheduler_name = getattr(args, "scheduler", None)
    if scheduler_name not in ("slurm", "flux"):
        rc, avail = remote.ssh_capture(target, site.remote_scheduler_probe(), timeout=20)
        detected, why = site.pick_scheduler(avail if rc == 0 else "", None)
        if detected not in ("slurm", "flux"):
            print(f"boxy: no live scheduler detected on {host} ({why}) — pass --scheduler slurm|flux",
                  file=sys.stderr)
            return 1
        scheduler_name = detected
        print(f"  auto: scheduler: {detected} (via {why} on {host})")
    scheduler = get_scheduler(scheduler_name)

    site_args = _app_site_args(args, card, scheduler, scheduler_name, target=target, host=host)

    rc, rhome = remote.ssh_capture(target, 'printf %s "$HOME"', timeout=15)
    rhome = rhome.strip() if rc == 0 and rhome.strip() else f"/home/{target.split('@')[0]}"
    rdir = f"{rhome}/{AGENTLESS_REMOTE_SUBDIR}/{host}"
    name = f"app-{card.card_name}"
    script_remote = f"{rdir}/{name}.sh"
    tok = scheduler.output_token or ""
    log_remote = f"{rdir}/{name}{('-' + tok) if tok else ''}.log"

    pfx = _proxy_prefix(args) if card.kind == "container" else ""
    # a boxy-owned file:// source mirror rides in every spack script: normally
    # empty and inert, but the landing spot when the egress filter blocks the
    # cluster's fetch and the archive has to be staged from the laptop.
    mirror_remote = f"{rdir}/spack-mirror"
    script_text = appcards.render_app_script(
        card, scheduler_name, name, log_remote, site_args,
        nodes=getattr(args, "nodes", 0) or 0,
        tasks_per_node=getattr(args, "tasks_per_node", 0) or 0,
        proxy_prefix=pfx, spack_mirror_dir=(mirror_remote if card.kind == "spack" else ""))

    print(f"### Agentless app run from {card.label} (no boxy on the cluster).")
    print(f"### Batch script ({script_remote}):")
    for line in script_text.rstrip().splitlines():
        print(f"    {line}")
    submit_cmd = shlex.join(scheduler.submit_command(script_remote))
    print(f"### Submit Command (on {host}):\n    {submit_cmd}")
    if args.dryrun:
        return 0

    remote.ssh_capture(target, f"mkdir -p {shlex.quote(rdir)}", timeout=15)

    if card.kind == "spack":
        # a hand-downloaded archive (--stage-source): push it into the mirror at
        # its OWN sha256 — if it's the right file spack finds it by digest, if
        # not spack ignores it. The escape hatch when even the laptop's egress is
        # filtered: download in a BROWSER (which passes the filter's auth), then
        #   boxy app <name> --ssh <host> --stage-source ~/Downloads/<archive>
        staged = _stage_source_file(args, target, host, mirror_remote)
        if staged is False:
            return 1
        # rerun-after-failure: if the PREVIOUS run of this job died on a blocked
        # source fetch (the follow loop wasn't there to see it — detached or
        # Ctrl-C'd), stage the archive NOW, before this submission.
        if not staged:
            prev = _remote_log_tail(target, log_remote, n=200)
            if prev.strip() and _spack_fetch_blocked(prev):
                print("boxy: the previous run died on a blocked source fetch — staging the "
                      "archive before this submission ...", file=sys.stderr)
                _maybe_spack_source_heal(target, host, prev, mirror_remote)

    if remote.push_file(target, script_remote, script_text) != 0:
        return 1
    rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}", timeout=60)

    # a site-default license other clusters don't define is dropped and resubmitted,
    # same auto-recovery as the serve path (field: kahuna).
    if rc != 0 and scheduler_name == "slurm" and _is_license_error(out):
        lic_args = [a for a in site_args if "--license" in a]
        if lic_args:
            site_args = [a for a in site_args if "--license" not in a]
            print(f"boxy: the site rejected the license request ({lic_args[0]!r}); retrying "
                  f"WITHOUT it ...", file=sys.stderr)
            script_text = appcards.render_app_script(
                card, scheduler_name, name, log_remote, site_args,
                nodes=getattr(args, "nodes", 0) or 0,
                tasks_per_node=getattr(args, "tasks_per_node", 0) or 0,
                proxy_prefix=pfx, spack_mirror_dir=(mirror_remote if card.kind == "spack" else ""))
            if remote.push_file(target, script_remote, script_text) == 0:
                rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}",
                                             timeout=60)

    if rc != 0:
        print(f"boxy: submit failed on {host}:\n{out.strip()}", file=sys.stderr)
        return 1
    job_id = scheduler.parse_job_id(out)
    print(f"### Submitted {scheduler_name} job {job_id}  ({name})")
    jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                             "app": card.name, "submitted_from": "app-agentless-ssh",
                             "target": target, "log": log_remote})
    if getattr(args, "detach", False):
        print(f"### Detached. Log (on {host}): {log_remote}")
        return 0

    # follow to completion: an app job is FINITE — the results ARE the log.
    print("### Waiting for the job to finish (Ctrl-C detaches; the job keeps running) ...")
    last_state = ""
    healed_sources = False
    try:
        while True:
            src, sout = remote.ssh_capture(target, shlex.join(scheduler.state_command(job_id)),
                                           timeout=20)
            state = scheduler.interpret_state(sout) if src == 0 else "UNKNOWN"
            if state != last_state and state in ("PENDING", "RUNNING"):
                print(f"### Job {job_id}: {state}")
                last_state = state
            if state == "DONE":
                # a spack job that died because the CLUSTER's egress filter blocked
                # the source download is healed from the laptop: stage the archive
                # into the job's file:// mirror and resubmit ONCE.
                tail = _remote_log_tail(target, log_remote, n=200)
                if (card.kind == "spack" and not healed_sources and _spack_fetch_blocked(tail)
                        and _maybe_spack_source_heal(target, host, tail, mirror_remote)):
                    healed_sources = True
                    rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}",
                                                 timeout=60)
                    if rc == 0:
                        job_id = scheduler.parse_job_id(out)
                        print(f"### Resubmitted as {scheduler_name} job {job_id} "
                              f"(sources staged from your machine).")
                        jobs.write_record(name, {"name": name, "scheduler": scheduler_name,
                                                 "job": job_id, "app": card.name,
                                                 "submitted_from": "app-agentless-ssh",
                                                 "target": target, "log": log_remote})
                        last_state = ""
                        _time.sleep(5)
                        continue
                    print(f"boxy: resubmit failed on {host}:\n{out.strip()}", file=sys.stderr)
                break
            _time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n### Detached. Log (on {host}): {log_remote}")
        return 0
    tail = _remote_log_tail(target, log_remote, n=200)
    print(f"### Job {job_id} finished. Log ({host}:{log_remote}):")
    print(tail if tail.strip() else "    (empty log)")
    return 0


def _app_local(args, card) -> int:
    """Run an app card on THIS machine (a cluster login node with boxy installed):
    render the batch script, write it under the jobs dir, and submit it with the
    local scheduler. --dryrun prints the script without touching the scheduler."""
    import subprocess

    from boxy import appcards, jobs
    from boxy.schedulers import get_scheduler

    scheduler_name = getattr(args, "scheduler", None)
    if scheduler_name not in ("slurm", "flux"):
        detected = "slurm" if shutil.which("sbatch") else ("flux" if shutil.which("flux") else "")
        if not detected and not args.dryrun:
            print("boxy: no scheduler on this machine (sbatch/flux not on PATH) — run with "
                  "--ssh <cluster> from your laptop, or --dryrun to inspect the script",
                  file=sys.stderr)
            return 1
        scheduler_name = detected or "slurm"
    scheduler = get_scheduler(scheduler_name)

    site_args = _app_site_args(args, card, scheduler, scheduler_name)
    name = f"app-{card.card_name}"
    script_path = str(jobs.script_path(name))
    tok = scheduler.output_token or ""
    log_path = str(jobs.log_path(name, tok) if tok else jobs.log_path(name))

    script_text = appcards.render_app_script(
        card, scheduler_name, name, log_path, site_args,
        nodes=getattr(args, "nodes", 0) or 0,
        tasks_per_node=getattr(args, "tasks_per_node", 0) or 0)

    print(f"### App run from {card.label}.")
    print(f"### Batch script ({script_path}):")
    for line in script_text.rstrip().splitlines():
        print(f"    {line}")
    if args.dryrun:
        return 0
    with open(script_path, "w") as f:
        f.write(script_text)
    result = subprocess.run(scheduler.submit_command(script_path), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"boxy: submit failed:\n{(result.stderr or result.stdout).strip()}", file=sys.stderr)
        return 1
    job_id = scheduler.parse_job_id(result.stdout)
    print(f"### Submitted {scheduler_name} job {job_id}  ({name}) — log: {log_path}")
    jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                             "app": card.name, "submitted_from": "app-local", "log": log_path})
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    """List / show / export the packaged example box & location profiles. Uses
    importlib.resources so it works from an installed wheel (not just a checkout)."""
    from importlib.resources import files
    from pathlib import Path

    root = files("boxy.data") / "examples"

    def _iter():
        for kind in ("boxes", "locations"):
            d = root / kind
            for entry in sorted(d.iterdir(), key=lambda p: p.name):
                if entry.name.endswith(".toml"):
                    yield kind, entry

    if args.action == "show":
        for _, entry in _iter():
            if entry.name in (args.name, f"{args.name}.toml"):
                print(entry.read_text(), end="")
                return 0
        print(f"boxy examples: no example named {args.name!r} (see `boxy examples`)", file=sys.stderr)
        return 1

    if args.action == "export":
        import shutil as _shutil
        dest = Path(args.dest)
        for kind, entry in _iter():
            target = dest / kind / entry.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with entry.open("rb") as src, open(target, "wb") as out:
                _shutil.copyfileobj(src, out)
        print(f"exported packaged examples to {dest}/  (boxes/ + locations/)")
        return 0

    # default: list
    print("packaged examples (boxy examples show NAME | export DIR):")
    for kind, entry in _iter():
        print(f"  {kind:<10} {entry.name}")
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show the effective config (value + provenance for each setting), or emit a
    starter config.toml with --init. The debugging tool for 'why did my layered
    value not take effect' — provenance is 'env <NAME>', 'file', or 'default'."""
    if args.init:
        print(config.render_template(), end="")
        return 0
    width = max(len(k) for k in config.SETTINGS)
    for key in config.SETTINGS:
        value, prov = config.source(key)
        print(f"{key:<{width}}  = {value!r:<40}  ({prov})")
    return 0


def _require_scheduler_binary(binary: str, scheduler_name: str) -> None:
    """Fail early and CLEARLY when the scheduler's submit tool isn't on this
    host — otherwise subprocess raises a bare '[Errno 2] ... sbatch'. The usual
    cause: `--scheduler slurm` run on a laptop without --ssh (field report)."""
    if shutil.which(binary):
        return
    raise UsageError(
        f"'{binary}' is not on this host — it has no {scheduler_name} scheduler. Either:\n"
        f"  • submit to a remote cluster FROM here:  add --ssh user@login  (boxy runs it there\n"
        f"    and tunnels the endpoint back), or\n"
        f"  • run locally with no scheduler:  drop --scheduler (and --gpus/--time/--account).")


def _submission_hint(stderr: str) -> str:
    """Plain-language next step for the sbatch/flux-batch rejections users
    actually hit. Empty string when the error isn't recognized."""
    low = stderr.lower()
    if "flux batch" in low or "flux-batch" in low:
        return ("boxy hint: you asked for --scheduler slurm, but this cluster's `sbatch` is a\n"
                "  FLUX compatibility wrapper — this is a Flux system (clusterA-class). Rerun with\n"
                "  --scheduler flux (same portable flags: --partition/--account/--time translate).\n"
                "  Tip: keep a --location profile per cluster so the scheduler is pinned correctly.")
    if "invalid account or account/partition combination" in low:
        return ("boxy hint: the scheduler rejected the ACCOUNT+PARTITION pairing, not the job.\n"
                "  - check which accounts you may use on which partitions:\n"
                "      sacctmgr show assoc user=$USER format=account%20,partition%20,qos%30\n"
                "  - list this cluster's partitions:  sinfo -s\n"
                "  - if you passed several partitions (--partition=a,b), every one must accept\n"
                "    the account — retry with the single partition you know works.")
    if "invalid partition" in low:
        return "boxy hint: that partition doesn't exist here — list them with: sinfo -s"
    if "invalid qos" in low:
        return ("boxy hint: that QOS isn't available to this account — see yours with:\n"
                "      sacctmgr show assoc user=$USER format=account%20,qos%40")
    return ""


AGENTLESS_REMOTE_SUBDIR = ".local/share/boxy/agentless"


def _is_gres_error(text: str) -> bool:
    """True when a scheduler's rejection is about the GPU request — the signal to
    auto-retry with a different GPU form/type. Covers both the wrong-FORM error
    ('Invalid generic resource (gres) specification') and the wrong-TYPE/count
    error ('Requested node configuration is not available' — e.g. a pinned
    BOXY_GPU_TYPE that the site doesn't have)."""
    low = (text or "").lower()
    return ("gres" in low or "generic resource" in low
            or "requested node configuration is not available" in low)


def _hf_model_facts(repo: str) -> dict:
    """Best-effort LAPTOP-side probe of a HF repo's config.json — used before
    burning a GPU allocation. Returns a dict:
      refusal          str|None — set when the arch is plainly NOT servable by any
                                  engine boxy drives (ASR/speech/embedding — field:
                                  an ASR nemotron burned a 400s job); None => proceed.
      trust_remote_code bool    — the repo ships custom modeling code (config has
                                  `auto_map`); vLLM needs --trust-remote-code or it
                                  dies at config validation (field: Nemotron-Parse).
      vlm              bool     — a vision/multimodal model (config has vision_config
                                  or an image token) — needs --limit-mm-per-prompt.
      arch             str      — the first declared architecture (for messages).
    Empty facts (all falsey) when uncheckable (gated/offline) or BOXY_NO_PREFLIGHT."""
    empty = {"refusal": None, "trust_remote_code": False, "vlm": False, "arch": ""}
    if os.environ.get("BOXY_NO_PREFLIGHT"):
        return empty
    from boxy import cardgen

    try:
        cfg = cardgen.hf_get_json(repo, "config.json", cardgen.resolve_token(None), timeout=8) or {}
    except Exception:
        return empty                     # can't check (gated/offline) — let the engine try
    arch = (cfg.get("architectures") or [""])[0] or ""
    low = arch.lower()
    trust = cardgen.needs_trust_remote_code(cfg)   # custom code => --trust-remote-code
    vlm = cardgen.is_vision_model(cfg)             # vision => --limit-mm-per-prompt
    refusal = None
    # a VLM/image-text-to-text model IS servable by vLLM — never refuse it; only the
    # plainly-non-generative families are hopeless on vLLM/llama.cpp.
    if arch and not vlm and any(tok in low for tok in (
            "asr", "rnnt", "speech", "audio", "wav2vec",
            "embedding", "reranker", "sequenceclassification")):
        refusal = (f"{repo} declares architecture {arch!r} — not a text-generation model. "
                   f"No engine boxy drives (vLLM, llama.cpp) can serve it, on any cluster: "
                   f"ASR/speech models need their own runtime (e.g. NVIDIA NeMo/Riva), "
                   f"embedding/reranker models need an embedding server. "
                   f"Pass --no-preflight to try anyway.")
    return {"refusal": refusal, "trust_remote_code": trust, "vlm": vlm, "arch": arch}


def _apply_model_facts(box, facts: dict, args):
    """Fold the auto-detected serve knobs into box.args (rendered as vLLM flags on
    the compute node). --trust-remote-code for a custom-code repo (config auto_map)
    or an explicit --trust-remote-code; --limit-mm-per-prompt for a vision model so
    it accepts one image (matches the model cards' recommended serve line). vLLM
    only — llama.cpp's server rejects these. box.args wins if already set."""
    from dataclasses import replace as _replace

    if box.engine != "vllm":
        if facts.get("trust_remote_code") or getattr(args, "trust_remote_code", False):
            print(f"  auto: trust-remote-code: skipped (engine is {box.engine}, not vllm)",
                  file=sys.stderr)
        return box
    new_args = dict(box.args)
    want_trust = facts.get("trust_remote_code") or getattr(args, "trust_remote_code", False)
    if want_trust and "trust_remote_code" not in new_args and "trust-remote-code" not in new_args:
        new_args["trust_remote_code"] = True
        why = ("the repo ships custom code (config auto_map)"
               if facts.get("trust_remote_code") else "requested via --trust-remote-code")
        print(f"  auto: trust-remote-code: enabled — {why}; vLLM will run the model's custom code")
    if facts.get("vlm") and not any(k.replace("_", "-") == "limit-mm-per-prompt" for k in new_args):
        new_args["limit-mm-per-prompt"] = '{"image": 1}'
        print(f"  auto: multimodal: {facts.get('arch') or 'vision model'} — "
              "--limit-mm-per-prompt '{\"image\": 1}' (send an image + prompt to /v1/chat/completions)")
    return _replace(box, args=new_args)


def _is_license_error(text: str) -> bool:
    """True when a submit failed on the LICENSE line ('Invalid license
    specification') — the shipped site.license default (tscratch:1, a hops-ism)
    doesn't exist on every cluster (field: kahuna). The self-heal drops the
    directive and resubmits."""
    return "invalid license" in (text or "").lower()


def _looks_like_pull_block(text: str) -> bool:
    """True when a job log shows a blocked/failed container-image pull (compute
    node can't reach the registry — Docker Hub/ghcr 403, Zscaler block, air-gap)."""
    low = (text or "").lower()
    return ("pinging container registry" in low or "initializing source docker://" in low
            or "initializing source" in low and "403" in low
            or ("registry" in low and ("403" in low or "denied" in low)))


def _looks_like_proxy_failure(text: str) -> bool:
    """True when a pull failed trying to reach the FORWARDED proxy itself
    (`proxyconnect ... i/o timeout` / `dial tcp <proxy>: timeout`) — the compute
    node can't reach the proxy boxy forwarded. A cluster that doesn't use the
    site proxy (field: eldorado vs the proxy hops needs) should retry WITHOUT it."""
    low = (text or "").lower()
    return "proxyconnect" in low or ("proxy" in low and ("i/o timeout" in low or "timeout" in low))


def _looks_like_trust_remote_code_error(text: str) -> bool:
    """True when vLLM refused to load the model's CUSTOM code — 'Please pass the
    argument `trust_remote_code=True`' at config validation (field: nvidia
    Nemotron-Parse). The self-heal resubmits with the flag folded in, so the fix
    works even when the laptop can't reach the Hub to pre-detect it."""
    low = (text or "").lower()
    return "trust_remote_code=true" in low or (
        "trust_remote_code" in low and "custom code" in low)


def _local_gpu_types() -> list[str]:
    """Best-effort GPU TYPE tokens from LOCAL sinfo (login node), so a local
    auto-recovery tries typed --gres=gpu:<type>:N forms before the untyped one.
    [] when sinfo is missing/errors or reports no types."""
    from boxy import site
    from boxy.schedulers import get_scheduler

    try:
        p = subprocess.run(get_scheduler("slurm").partitions_command(),
                           capture_output=True, text=True, timeout=15)
        out = p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return []
    return site.gpu_types_from_gres(out)


def _gpu_form_label(gform: str, gtype: str) -> str:
    """The human GPU-request label for a recovery attempt (matches what's submitted
    so the 'retrying with …' message never lies): e.g. --gpus-per-node=N,
    --gres=gpu:a100:N, --gpus=N."""
    if gform == "none":
        return "NO GPU directive (this site has no GRES config — the partition implies the GPUs)"
    t = f"{gtype}:" if gtype else ""
    return {"gres": f"--gres=gpu:{t}N", "gpus": f"--gpus={t}N",
            "gpus-per-node": f"--gpus-per-node={t}N"}.get(gform, f"--{gform}={t}N")


def _gres_fallback_forms(gtypes: list[str]) -> list[tuple[str, str]]:
    """The GPU-request forms to try, in order, when a site REJECTS the GPU line
    ('Invalid generic resource (gres) specification', or 'Requested node
    configuration is not available' from a wrong pinned type — field: kahuna/hops).

    Order: an untyped --gpus-per-node=N FIRST but only when a pinned
    BOXY_GPU_TYPE poisoned the initial request (dropping the wrong type is the
    fix there); then a typed --gres=gpu:<type>:N for EVERY type sinfo reported —
    kahuna REQUIRES the type and rejects all untyped spellings, and a cluster
    mixing types across partitions never yielded the single 'spanning' type the
    old probe wanted; then untyped --gres=gpu:N; then --gpus=N. Recovers without
    the user setting BOXY_GPU_DIRECTIVE / BOXY_GPU_TYPE by hand."""
    forms: list[tuple[str, str]] = []
    if config.get_str("site.gpu_type").strip():
        forms.append(("gpus-per-node", ""))    # shed the poisoned pinned type first
    forms += [("gres", t) for t in gtypes]
    forms += [("gres", ""), ("gpus", "")]
    # LAST rung: no GPU directive at all — a site with NO GRES configured (every
    # partition reports (null); field: kahuna) rejects every gres/gpus spelling,
    # and GPUs there are implied by the PARTITION (hopper/grace/blackwell).
    forms.append(("none", ""))
    seen: set = set()
    ordered: list[tuple[str, str]] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered


def _pick_mode(args) -> str:
    """The interactive-picker mode for this serve: --pick-account/--no-pick-account
    (a tri-state store_true/false, default None) override config site.pick_account
    ('auto' | 'always' | 'never')."""
    from boxy import config

    v = getattr(args, "pick_account", None)
    if v is True:
        return "always"
    if v is False:
        return "never"
    return (config.get_str("site.pick_account") or "auto").strip().lower()


def _pick_account(args, where: str = "", rows=None) -> None:
    """Resolve the charge account interactively when several are discovered and
    none was named, setting args.account to the pick (so the rest of the pipeline
    is unchanged) and printing the decision line. A no-op when the picker is off
    ('never'), an account is already set, or a pin ($WCID / site.account) applies
    — those keep boxy's existing silent resolution. `rows` may be pre-probed
    (over --ssh); otherwise they're discovered locally from the account command."""
    from boxy import config, picker, site

    if getattr(args, "account", None) or _pick_mode(args) == "never":
        return
    explicit = os.environ.get("WCID", "").strip() or config.get_str("site.account").strip()
    if rows is None:
        rows = site.discover_account_rows()
    pick, note = picker.choose_account(rows, explicit=explicit or None,
                                       remembered=picker.recall(where), mode=_pick_mode(args),
                                       where=where, source="mywcid")
    if pick:
        args.account = pick
        print(f"  auto: account: {pick} ({note})")


def _pick_partition_mode(args) -> str:
    from boxy import config

    v = getattr(args, "pick_partition", None)
    if v is True:
        return "always"
    if v is False:
        return "never"
    return (config.get_str("site.pick_partition") or "auto").strip().lower()


def _pick_partition(args, scheduler_name: str, need_gpu: bool, where: str = "") -> None:
    """Interactive partition pick when 2+ are available and none was named: sets
    args.partition to the choice (so resolve_site sees it as explicit). A no-op
    under 'never' (suite/CI default), when --partition/site.partition is set, or
    when discovery finds <2 partitions. Local path — the list is discovered here."""
    from boxy import config, picker, site

    if getattr(args, "partition", None) or _pick_partition_mode(args) == "never":
        return
    if config.get_str("site.partition").strip():
        return
    part, _why = site.resolve_partition(None, scheduler_name, need_gpu)
    names = [p for p in (part or "").split(",") if p]
    if len(names) < 2:
        return
    pick, note = picker.choose_partition(
        names, remembered=picker.recall(where, kind="partition"), mode=_pick_partition_mode(args),
        where=where, source=scheduler_name, allow_all=(scheduler_name != "flux"))
    if pick:
        args.partition = pick
        print(f"  auto: partition: {pick} ({note})")


def _pick_remote_partition(args, names, host: str, target: str, scheduler_name: str) -> tuple[str, str]:
    """Choose a partition from a cluster-probed list for an --ssh serve. Returns
    (value, note). Interactive menu only when the picker is enabled AND 2+ exist;
    otherwise the soonest-start comma-list (Slurm) / first (Flux) — legacy."""
    from boxy import picker

    if len(names) > 1 and picker.is_interactive(_pick_partition_mode(args)):
        pick, note = picker.choose_partition(
            names, remembered=picker.recall(target, kind="partition"),
            mode=_pick_partition_mode(args), where=target, source=f"{scheduler_name} on {host}",
            allow_all=(scheduler_name != "flux"))
        if pick:
            return pick, note
    return ",".join(names), ""


def _pick_remote_account(args, rows, host: str, target: str) -> tuple[str | None, str]:
    """Choose the account from a cluster-probed (wcid, label) list for an --ssh
    serve. Returns (account_or_None, note). Shows the interactive menu only when
    the picker is enabled AND several accounts exist; otherwise the silent
    first-pick with the legacy `mywcid on <host>; also: …` note, so batch/CI and
    the delegation tests are unchanged."""
    from boxy import picker

    accounts = [w for w, _ in rows]
    if not accounts:
        return None, ""
    if len(accounts) > 1 and picker.is_interactive(_pick_mode(args)):
        pick, note = picker.choose_account(rows, remembered=picker.recall(target),
                                           mode=_pick_mode(args), where=target,
                                           source=f"mywcid on {host}")
        if pick:
            return pick, note
    extra = f"; also: {', '.join(accounts[1:3])}" if len(accounts) > 1 else ""
    return accounts[0], f"mywcid on {host}{extra}"


def _stage_agentless_ca(target: str, host: str, rdir: str, dryrun: bool = False) -> str | None:
    """Stage the laptop's MERGED CA bundle (public CAs + your site interceptor CA)
    onto the cluster's shared FS and return its ABSOLUTE cluster path, so the
    agentless container mounts a path that actually exists on the compute node.

    Why: over --ssh the compute node can't see the laptop's SSL_CERT_FILE path, so
    the old mount bind-mounted an empty file and the in-container HuggingFace fetch
    died with CERTIFICATE_VERIFY_FAILED behind the site's TLS interceptor (field:
    hops). We push the bundle here and hand the path to deploy.set_agentless_ca().
    No-op when there's no merged bundle or BOXY_NO_CA_PROPAGATE is set (tests).

    In `dryrun` the path is still returned (so the printed script shows the real
    mount) but nothing is copied — --dryrun must not touch the cluster."""
    from boxy import remote

    if os.environ.get("BOXY_NO_CA_PROPAGATE"):
        return None
    ca = os.environ.get("SSL_CERT_FILE", "")
    if not (ca.endswith("ca-merged.crt") and os.path.isfile(ca)):
        return None  # no boxy-merged bundle to carry (the cluster uses its own store)
    remote_path = f"{rdir}/boxy-ca-merged.pem"
    if dryrun:
        print(f"  auto: CA: would stage your merged site CA -> {host}:{remote_path} "
              "(mounted into the container so its HuggingFace/TLS trusts the site CA)")
        return remote_path
    try:
        data = open(ca, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    if remote.push_file(target, remote_path, data) != 0:
        print(f"boxy: warning: could not stage the site CA to {host}; an in-container "
              "HuggingFace pull may fail TLS (CERTIFICATE_VERIFY_FAILED)", file=sys.stderr)
        return None
    print(f"  auto: CA: staged your merged site CA -> {host}:{remote_path} "
          "(mounted into the container so its HuggingFace/TLS trusts the site CA)")
    return remote_path


def _prestage_mode(args) -> str:
    """Resolve the agentless pre-stage policy: --prestage/--no-prestage win, else
    config serve.agentless_prestage (auto|always|never)."""
    flag = getattr(args, "prestage", None)
    if flag in ("always", "never"):
        return flag
    mode = (config.get_str("serve.agentless_prestage") or "auto").strip().lower()
    return mode if mode in ("auto", "always", "never") else "auto"


def _prestage_agentless_model(args, target: str, host: str, box, image: str,
                              pfx: str, rdir: str, ca_remote: str | None):
    """PRE-STAGE the container image + hf:// model on the LOGIN node (which has the
    SSH session's network + the forwarded proxy), onto the cluster's shared FS, so a
    fully ISOLATED compute node needs no runtime network. Returns the staged
    shared-FS model path on success, or None to leave engine-pull in place.

    The model download runs INSIDE the just-pulled image (it already ships
    huggingface_hub), so nothing extra is installed on the cluster. Only vLLM
    (safetensors) is auto-staged this way; a GGUF/llama.cpp repo is left to the
    engine (its image may lack python). The image pull alone still runs so a
    networked compute node reuses $HOME's podman store with no re-download."""
    from boxy import deploy, remote

    repo = box.model  # already the bare repo id (scheme stripped by the caller)
    # 1) pull the image on the login node over the forwarded proxy (reused by every
    #    compute node sharing $HOME's podman store — no re-download on the node).
    print(f"  auto: prestage: pulling image {image} on {host} (login-node network) ...")
    rc, out = remote.ssh_capture(target, f"{pfx}podman pull {shlex.quote(image)}", timeout=3600)
    if rc != 0:
        print(f"boxy: warning: login-node image pull failed on {host}:\n{out.strip()[-800:]}\n"
              "boxy: continuing — the compute node will try to pull it itself.", file=sys.stderr)
    if box.engine != "vllm":
        # llama.cpp/GGUF: skip the in-container model download (image may have no
        # python). Image is pre-pulled; the engine fetches the GGUF at start.
        print(f"  auto: prestage: {box.engine} model left to the engine (only the image is "
              "pre-pulled); pass a shared-FS GGUF path for a fully offline node.")
        return None
    stage_dir = f"{rdir}/models/{_model_slug(repo)}"
    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    # proxy vars for INSIDE the download container (the pull reaches HF via the proxy)
    penv = remote.remote_proxy_env()
    run = ["podman", "run", "--rm", "--network", "host", "-v", f"{stage_dir}:/stage"]
    cenv = {"HF_HUB_ENABLE_HF_TRANSFER": "0"}
    if hf_tok:
        cenv["HF_TOKEN"] = hf_tok
        cenv["HUGGING_FACE_HUB_TOKEN"] = hf_tok
    if ca_remote:
        run += ["-v", f"{ca_remote}:{deploy.CA_CONTAINER_PATH}:ro"]
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            cenv[var] = deploy.CA_CONTAINER_PATH
    cenv.update(penv)
    for k, v in cenv.items():
        run += ["-e", f"{k}={v}"]
    # skip the redundant PyTorch 'original/' checkpoint (Llama ships both) and any
    # GGUF — vLLM serves the safetensors; this halves the download.
    py = ("from huggingface_hub import snapshot_download; "
          "snapshot_download(repo_id=%r, local_dir='/stage', "
          "ignore_patterns=['original/*','*.pth','consolidated*','*.gguf'])"
          % repo)
    run += ["--entrypoint", "python3", image, "-c", py]
    dl = pfx + shlex.join(run)
    print(f"  auto: prestage: downloading {repo} on {host} -> {stage_dir} "
          "(in-container huggingface_hub; may take several minutes) ...")
    rc, out = remote.ssh_capture(target, f"mkdir -p {shlex.quote(stage_dir)} && {dl}", timeout=7200)
    if rc != 0:
        print(f"boxy: warning: login-node model download failed on {host}:\n{out.strip()[-1200:]}\n"
              "boxy: continuing agentless with engine-pull (works only if the COMPUTE node has "
              "network); pass --no-prestage to silence, or stage the model yourself and serve by "
              "path.", file=sys.stderr)
        return None
    print(f"  auto: prestage: model staged at {host}:{stage_dir} — serving BY PATH "
          "(the compute node needs no network).")
    return stage_dir


def _model_slug(repo: str) -> str:
    """Filesystem-safe slug for a HF repo id (org/name -> org-name)."""
    return repo.strip().strip("/").replace("/", "-").replace(":", "-").lower()


def _serve_agentless_ssh(args, target: str) -> int:
    """Fully AGENTLESS serve over --ssh: NOTHING is installed on the HPC system —
    no boxy, no Python, no RamaLama. The laptop does everything over the one ssh
    master: detect the scheduler + resolve the site (account/partition/time),
    render a self-contained `podman run` batch script, write + submit it, then
    poll the shared-FS endpoint file and confirm readiness via the tunnel
    (localhost/health). The compute node runs only podman + a bash endpoint-write.

    Model: an hf:// (transport-URI) model is pulled by the ENGINE at container
    start (`vllm serve <repo>` over the forwarded proxy); a shared-FS path is
    mounted as-is; s3:// needs staging first."""
    import json
    import time
    from dataclasses import replace as dc_replace

    from boxy import cards, deploy, jobs, ramalama_shim, remote, resolve, site
    from boxy.schedulers import get_scheduler

    host = target.split("@")[-1]
    if remote.ensure_master(target) != 0:
        print(f"boxy: could not open an SSH session to {target} — check the host, your VPN, "
              f"and that you completed any OTP/YubiKey prompt", file=sys.stderr)
        return 1

    # 1) scheduler — liveness-detect over the master unless pinned
    scheduler_name = getattr(args, "scheduler", None)
    if scheduler_name not in ("slurm", "flux"):
        rc, avail = remote.ssh_capture(target, site.remote_scheduler_probe(), timeout=20)
        detected, why = site.pick_scheduler(avail if rc == 0 else "", None)
        if detected not in ("slurm", "flux"):
            print(f"boxy: no live scheduler detected on {host} ({why}) — pass --scheduler slurm|flux",
                  file=sys.stderr)
            return 1
        scheduler_name = detected
        print(f"  auto: scheduler: {detected} (via {why} on {host})")
    args.scheduler = scheduler_name  # pin it so _resolve_or_load doesn't trip the login-node guard
    scheduler = get_scheduler(scheduler_name)

    # 1b) accelerator: with no explicit flag, probe the TARGET's login-node GPU
    #     stack over the master (nvidia-smi vs rocm-smi//opt/rocm) BEFORE
    #     resolution — the local resolver can only see THIS laptop, so it would
    #     fall back to the cuda config default and hand an AMD cluster the CUDA
    #     image (field: AMD system). Probed value rides in as the explicit flag.
    if not getattr(args, "accelerator", None):
        arc, aout = remote.ssh_capture(target, site.remote_accel_probe(), timeout=20)
        probed = site.parse_remote_accel(aout) if arc == 0 else ""
        if probed:
            args.accelerator = probed
            print(f"  auto: accelerator: {probed} (via the GPU stack on {host}'s login node)")

    # 2) model card -> gpus/engine/args (all laptop-side; nothing needed on the cluster)
    for line in cards.apply_to_args(args):
        print(f"  auto: {line}")

    # 3) box + location, accelerator/runtime PINNED so no local hardware probe runs
    args.runtime = args.runtime or "podman"
    try:
        box, location, _dec = _resolve_or_load(args)
    except UsageError as e:
        print(f"boxy: {e}", file=sys.stderr)
        return 2
    accel = args.accelerator or location.accelerator or config.get_str("site.default_accelerator")
    location = dc_replace(location, scheduler=scheduler_name, accelerator=accel,
                          runtime=(args.runtime or "podman"))

    _, name, _ = resolve.resolve_submission(args.model, scheduler_name,
                                            name=getattr(args, "name", None), require_exists=False)

    # 4) model handling: engine-pull for a transport URI, mount for a path
    engine_pull = box.model_is_transport_uri and not box.model_is_s3
    if engine_pull:
        bare = box.model.split("://", 1)[1]
        box = dc_replace(box, model=bare)
        # laptop-side HF probe: refuse a plainly unservable model in seconds instead
        # of burning a GPU allocation (field: ASR nemotron), AND auto-detect the
        # serve knobs a custom-code / vision model needs so `serve hf://…` stays
        # turnkey (field: Nemotron-Parse died for want of --trust-remote-code).
        facts = {} if getattr(args, "no_preflight", False) else _hf_model_facts(bare)
        if facts.get("refusal"):
            print(f"boxy: {facts['refusal']}", file=sys.stderr)
            return 2
        box = _apply_model_facts(box, facts, args)
        print(f"  auto: model: {args.model} — the engine downloads it at container start "
              f"(no RamaLama on the cluster)")
    box = dc_replace(box, name=name, image=(args.image or box.image))

    # carry the corporate proxy (for the in-container model/image pull) and, if the
    # laptop has it, the HF token for gated repos — INTO the container env. The
    # token lands in the batch script on the shared FS, so the script is written
    # mode 600 (below); prefer exporting HF_TOKEN on the cluster if that worries you.
    extra_env = dict(remote.remote_proxy_env())
    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_tok and engine_pull:
        extra_env["HF_TOKEN"] = hf_tok
        print("  auto: HF token: forwarded from your laptop env into the container "
              "(gated-repo downloads authenticate)")
    elif engine_pull:
        print("boxy: note: no HF_TOKEN in your laptop env — if the model is a GATED repo "
              "(meta-llama/* is), the download will 401; export HF_TOKEN and rerun.",
              file=sys.stderr)
    if extra_env:
        box = dc_replace(box, env={**box.env, **extra_env})

    # 5) site over ssh: account (config/env, else mywcid on the cluster), partition
    #    (sinfo/flux queue on the cluster), time (config default). Explicit flags win.
    site_args = list(location.scheduler_args)
    acct, awhy = site.resolve_account(getattr(args, "account", None))
    if not acct:
        # probe mywcid on the cluster and pick laptop-side — works against an
        # OLD/absent cluster boxy. A TTY-driven menu when several accounts and the
        # picker is enabled; otherwise the silent first-pick (legacy behavior).
        rc, out = remote.ssh_capture(target, site.remote_account_probe(), timeout=20)
        rows = site.parse_account_rows(out) if rc == 0 else []
        acct, awhy = _pick_remote_account(args, rows, host, target)
    if acct:
        site_args.append(scheduler.site_directive("account", acct))
        print(f"  auto: account: {acct} (via {awhy})")
    need_gpu = _job_wants_gpu(args)
    part = getattr(args, "partition", None)
    if site.partition_mode(part) in ("auto", "all"):
        rc, out = remote.ssh_capture(target, site.remote_partition_probe(scheduler_name), timeout=20)
        value, pwhy = site.rank_remote_partitions(out, scheduler_name, need_gpu) if rc == 0 else ("", "")
        if value:
            names = [p for p in value.split(",") if p]
            pick, pnote = _pick_remote_partition(args, names, host, target, scheduler_name)
            part = pick
            why = pnote or f"{scheduler_name} on {host}: {pwhy}"
            print(f"  auto: partition: {part} (via {why})")
    if part and site.partition_mode(part) not in ("off",):
        if scheduler_name == "flux" and "," in part:
            part = part.split(",")[0].strip()
        site_args.append(scheduler.site_directive("partition", part))
    # Slurm license(s): --license / config, e.g. #SBATCH --license=tscratch:1
    if scheduler_name == "slurm":
        lic, lwhy = site.resolve_license(getattr(args, "license", None))
        if lic:
            site_args.append(scheduler.site_directive("license", lic))
            print(f"  auto: license: {lic} (via {lwhy})")

    # GPU request form: keep the proven default (--gpus-per-node, what works on
    # hops/eldorado/…). We DON'T proactively change a working cluster's directive;
    # if a site rejects it at submit (field: kahuna), the submit block below
    # auto-recovers to the portable --gres form. See _gpu_flag / _gres_fallback_forms.

    t, twhy = site.resolve_time(getattr(args, "time", None))
    if t:
        site_args.append(scheduler.site_directive("time", t))
        print(f"  auto: time: {t} (via {twhy} — the scheduler stops the job at this walltime)")
    site_args += list(args.scheduler_args or [])

    # 6) remote paths on the CLUSTER's shared $HOME (discover it — shlex.quote in the
    #    renderer would freeze a literal $HOME, so we need the real absolute path).
    rc, rhome = remote.ssh_capture(target, 'printf %s "$HOME"', timeout=15)
    rhome = rhome.strip() if rc == 0 and rhome.strip() else f"/home/{target.split('@')[0]}"
    rdir = f"{rhome}/{AGENTLESS_REMOTE_SUBDIR}/{host}"
    script_remote = f"{rdir}/{name}.sh"
    ep_remote = f"{rdir}/{name}.endpoint.json"
    tok = scheduler.output_token or ""
    log_remote = f"{rdir}/{name}{('-' + tok) if tok else ''}.log"

    # engine-pull: persist the engine's HF cache on the CLUSTER's shared FS so the
    # model downloads ONCE and every rerun (and every compute node) reuses it —
    # no per-run 16GB re-download, no prestage wait. The batch script mkdir -p's
    # the dir at job runtime (shared FS, so it exists wherever the job lands).
    if engine_pull:
        from boxy.box import Volume

        hf_cache = f"{rdir}/hfcache"
        box = dc_replace(
            box,
            volumes=[*box.volumes, Volume(source=hf_cache, target="/root/.cache/huggingface")],
            env={**box.env, "HF_HOME": "/root/.cache/huggingface"})
        print(f"  auto: model cache: {host}:{hf_cache} (shared FS — downloaded once, "
              f"reused by every rerun)")

    # 6b) CA: stage the laptop's MERGED bundle onto the shared FS and mount THAT
    #     compute-node-valid path into the container, so an in-container HuggingFace
    #     pull trusts the site's TLS interceptor (field: hops CERTIFICATE_VERIFY_
    #     FAILED). The laptop's SSL_CERT_FILE path is invalid on the node.
    ca_remote = _stage_agentless_ca(target, host, rdir, dryrun=args.dryrun)

    # 7) render the self-contained script LAPTOP-side. Forward the corporate proxy
    #    to the COMPUTE-NODE `podman pull` (Docker Hub/ghcr 403 behind Zscaler): the
    #    submitter's ambient proxy env (or --proxy) is carried over — same as every
    #    other agentless render. Without this the isolated node can't reach the registry.
    pfx = _proxy_prefix(args)
    if pfx:
        print(f"  auto: proxy: forwarding {pfx.strip()}to the compute-node image pull "
              f"(reach the registry behind the site proxy)")

    # 6c) PRE-STAGE (agentless on an isolated compute node): pull the image + download
    #     the hf:// model on the LOGIN node (network via your SSH session), land both
    #     on the shared FS, then serve the model BY PATH — the compute node needs no
    #     runtime network. 'auto' stages a transport-URI model; 'always' also for a
    #     path model's image; 'never'/--no-prestage skips it. Best-effort: a failure
    #     falls back to engine-pull (only works on a networked node). Skipped under
    #     --dryrun (a real network/disk op); the plan line prints instead.
    pmode = _prestage_mode(args)
    if pmode != "never" and (engine_pull or pmode == "always"):
        image = args.image or box.image or ramalama_shim.default_image(box.engine, accel)
        if args.dryrun:
            what = f"model {box.model} + image {image}" if engine_pull else f"image {image}"
            tail = ("then serve BY PATH — the compute node needs no network"
                    if engine_pull else "so an isolated compute node reuses $HOME's podman store")
            print(f"  auto: prestage: would stage {what} on {host} (login node), {tail} "
                  "(--no-prestage to skip; not run under --dryrun)")
        elif engine_pull:
            staged = _prestage_agentless_model(args, target, host, box, image, pfx, rdir, ca_remote)
            if staged:
                box = dc_replace(box, model=staged)
                engine_pull = False
        else:
            # path model: just warm the image on the login node so an isolated node
            # reuses $HOME's podman store (no model download needed — it's already staged).
            print(f"  auto: prestage: pulling image {image} on {host} (login-node network) ...")
            prc, pout = remote.ssh_capture(target, f"{pfx}podman pull {shlex.quote(image)}", timeout=3600)
            if prc != 0:
                print(f"boxy: warning: login-node image pull failed on {host}:\n{pout.strip()[-800:]}",
                      file=sys.stderr)

    deploy.set_agentless_ca(ca_remote)
    try:
        script_text = deploy.render_agentless_script(
            box, location, scheduler_name, name, ep_remote, log_remote, site_args,
            proxy_prefix=pfx, port=args.port, engine_pulls_model=engine_pull)
    except deploy.AgentlessError as e:
        deploy.set_agentless_ca(None)
        print(f"boxy: agentless: {e}", file=sys.stderr)
        return 2

    print("### Agentless (no boxy on the cluster): a self-contained podman batch script, "
          "submitted + polled from your laptop over SSH.")
    print(f"### Batch script ({script_remote}):")
    for line in script_text.rstrip().splitlines():
        print(f"    {line}")
    submit_cmd = shlex.join(scheduler.submit_command(script_remote))
    print(f"### Submit Command (on {host}):\n    {submit_cmd}")
    if args.dryrun:
        deploy.set_agentless_ca(None)
        return 0

    # 8) write the script (mode 600 — it may carry HF_TOKEN) + submit, over the master
    if remote.push_file(target, script_remote, script_text) != 0:
        return 1
    remote.ssh_capture(target, f"chmod 600 {shlex.quote(script_remote)}", timeout=10)
    rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}", timeout=60)

    # AUTO-RECOVER from a rejected GPU request line: some sites reject
    # --gpus-per-node ('Invalid generic resource (gres) specification' — field:
    # kahuna) and want --gres=gpu:[type:]N. Instead of telling the user to set an
    # env var and rerun, re-render with the portable forms and RESUBMIT until one
    # is accepted. Only when the form is 'auto' (else the user pinned it on purpose).
    if (rc != 0 and scheduler_name == "slurm" and need_gpu and _is_gres_error(out)
            and config.get_str("site.gpu_directive").strip().lower() == "auto"):
        from boxy.schedulers import slurm as _slurm

        # probe the site's GPU type NOW (best-effort) so a site that REQUIRES a
        # typed --gres=gpu:<type>:N is tried before the untyped form.
        grc, gout = remote.ssh_capture(target, site.remote_partition_probe("slurm"), timeout=20)
        sel = ({p.strip() for p in part.split(",")}
               if part and site.partition_mode(part) == "set" else None)
        gtypes = site.gpu_types_from_gres(gout, sel) if grc == 0 else []
        if not gtypes:
            # some sites only surface GRES types per-NODE (%G at partition level
            # shows (null)); ask node-wise before giving up on a typed form.
            nrc, nout = remote.ssh_capture(target, "sinfo -h -N -o %G 2>/dev/null || true",
                                           timeout=20)
            if nrc == 0:
                gtypes = site.gpu_types_from_gres(nout)
        for gform, gtype in _gres_fallback_forms(gtypes):
            _slurm.set_auto_gres(gform, gtype)
            try:
                script_text = deploy.render_agentless_script(
                    box, location, scheduler_name, name, ep_remote, log_remote, site_args,
                    proxy_prefix=_proxy_prefix(args), port=args.port, engine_pulls_model=engine_pull)
            except deploy.AgentlessError:
                break
            shown = _gpu_form_label(gform, gtype)
            print(f"boxy: the site rejected the GPU request; retrying with {shown} ...",
                  file=sys.stderr)
            if remote.push_file(target, script_remote, script_text) != 0:
                break
            remote.ssh_capture(target, f"chmod 600 {shlex.quote(script_remote)}", timeout=10)
            rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}", timeout=60)
            if rc == 0:
                print(f"### GPU request accepted as {shown} (auto-recovered).")
                break

    # AUTO-RECOVER from a rejected LICENSE line: site.license ships a site default
    # (tscratch:1 — hops gates filesystems behind it) that other clusters don't
    # define ('Invalid license specification' — field: kahuna, surfaced after the
    # GPU ladder cleared). Drop the directive and resubmit; the accepted GPU form
    # from the ladder above is still active via set_auto_gres.
    if rc != 0 and scheduler_name == "slurm" and _is_license_error(out):
        lic_args = [a for a in site_args if "--license" in a]
        if lic_args:
            site_args = [a for a in site_args if "--license" not in a]
            print(f"boxy: the site rejected the license request ({lic_args[0]!r} — a "
                  f"site-specific default); retrying WITHOUT it ... (set BOXY_LICENSE= to "
                  f"skip it on this cluster permanently)", file=sys.stderr)
            try:
                script_text = deploy.render_agentless_script(
                    box, location, scheduler_name, name, ep_remote, log_remote, site_args,
                    proxy_prefix=_proxy_prefix(args), port=args.port, engine_pulls_model=engine_pull)
            except deploy.AgentlessError:
                script_text = ""
            if script_text and remote.push_file(target, script_remote, script_text) == 0:
                remote.ssh_capture(target, f"chmod 600 {shlex.quote(script_remote)}", timeout=10)
                rc, out = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}",
                                             timeout=60)
                if rc == 0:
                    print("### license request dropped (auto-recovered).")

    deploy.set_agentless_ca(None)  # done rendering; don't leak the override
    if rc != 0:
        print(f"boxy: submit failed on {host}:\n{out.strip()}", file=sys.stderr)
        low = out.lower()
        if "gres" in low or "generic resource" in low:
            print("boxy: hint: this site's Slurm rejected every GPU request form boxy tried "
                  "(--gpus-per-node, --gres=gpu:[type:]N, --gpus). Pin one explicitly if you "
                  "know the site's spelling:\n"
                  "        export BOXY_GPU_DIRECTIVE=gres        # --gres=gpu:N\n"
                  "  and, if it needs a GPU TYPE (see `sinfo -o %G` on the cluster):\n"
                  "        export BOXY_GPU_TYPE=<a100|h100|...>  # gpu:<type>:N\n"
                  "  (Other forms: BOXY_GPU_DIRECTIVE=gpus | none.)", file=sys.stderr)
        elif "account" in low or "invalid account" in low:
            print("boxy: hint: the account was rejected — pass --account <wcid> or export "
                  "BOXY_ACCOUNT.", file=sys.stderr)
        elif "partition" in low:
            print("boxy: hint: the partition was rejected — try --partition <name> "
                  "(list them: `sinfo -s` on the cluster) or --partition off.", file=sys.stderr)
        return 1
    job_id = scheduler.parse_job_id(out)
    print(f"### Submitted {scheduler_name} job {job_id}  ({name})")
    jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                             "model": args.model, "submitted_from": "agentless-ssh",
                             "target": target, "endpoint_remote": ep_remote, "log": log_remote})

    # 9) poll the shared-FS endpoint file over the master; the moment it names the
    #    compute node, hand off to the tunnel + localhost/health readiness.
    print("### Waiting for the job to start and the server to become ready ... "
          "(Ctrl-C detaches; the job keeps running)")
    # readiness window: never below the scheduler floor — the flag's config DEFAULT
    # (timeouts.readiness, 180s) must not masquerade as an explicit short window;
    # a cold vLLM start takes 5-15 min and detaching at 3 min lost the tunnel/READY
    # (field: hops). An explicit LONGER --ready-timeout still wins.
    ready_window = max(getattr(args, "ready_timeout", 0) or 0, _SCHED_READY_FLOOR)
    deadline = time.time() + 24 * 3600
    last_state = None
    proxy_healed = False
    trust_healed = False

    def _resubmit_current() -> bool:
        """Re-render with the CURRENT box/pfx, push, and resubmit — the shared
        tail of every death-path self-heal. Updates job_id/last_state and the job
        record; the poll loop then continues on the new job."""
        nonlocal job_id, last_state
        deploy.set_agentless_ca(ca_remote)
        try:
            heal_text = deploy.render_agentless_script(
                box, location, scheduler_name, name, ep_remote, log_remote, site_args,
                proxy_prefix=pfx, port=args.port, engine_pulls_model=engine_pull)
        except deploy.AgentlessError:
            return False
        finally:
            deploy.set_agentless_ca(None)
        if remote.push_file(target, script_remote, heal_text) != 0:
            return False
        remote.ssh_capture(target, f"chmod 600 {shlex.quote(script_remote)}", timeout=10)
        remote.ssh_capture(target, f"rm -f {shlex.quote(ep_remote)}", timeout=10)   # drop stale endpoint
        hrc, hout = remote.ssh_capture(target, f"cd {shlex.quote(rhome)} && {submit_cmd}", timeout=60)
        if hrc != 0:
            return False
        job_id = scheduler.parse_job_id(hout)
        last_state = None
        jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                                 "model": args.model, "submitted_from": "agentless-ssh",
                                 "target": target, "endpoint_remote": ep_remote, "log": log_remote})
        return True

    def _maybe_proxy_heal(tail: str) -> bool:
        """The job died reaching the FORWARDED proxy (field: eldorado can't reach
        the proxy hops needs). Resubmit ONCE without the proxy — the node may hit
        the registry directly / via its own proxy."""
        nonlocal pfx, proxy_healed
        if proxy_healed or not pfx or not _looks_like_proxy_failure(tail):
            return False
        proxy_healed = True
        print(f"boxy: the compute node couldn't reach the forwarded proxy "
              f"({pfx.strip()}); resubmitting WITHOUT it — this cluster may reach the "
              f"registry directly or via its own proxy (set BOXY_PROXY= to skip it here)...",
              file=sys.stderr)
        pfx = ""
        if not _resubmit_current():
            return False
        print(f"### Resubmitted {scheduler_name} job {job_id} without the proxy (auto-recovered).")
        return True

    def _maybe_trust_heal(tail: str) -> bool:
        """vLLM refused the model's CUSTOM code ('pass trust_remote_code=True' —
        field: Nemotron-Parse). Resubmit ONCE with the flag folded into the serve
        args — works even when the laptop-side HF probe couldn't pre-detect it
        (gated repo, laptop offline, or an older install)."""
        nonlocal box, trust_healed
        if trust_healed or not _looks_like_trust_remote_code_error(tail):
            return False
        if box.engine != "vllm" or any(
                k.replace("-", "_") == "trust_remote_code" for k in box.args):
            return False                          # already on (or not vLLM) — a real failure
        trust_healed = True
        print("boxy: the model ships custom loader code vLLM refused to run; resubmitting "
              "WITH --trust-remote-code (this executes code from the model repo — only for "
              "models you trust)...", file=sys.stderr)
        box = dc_replace(box, args={**box.args, "trust_remote_code": True})
        if not _resubmit_current():
            return False
        print(f"### Resubmitted {scheduler_name} job {job_id} with --trust-remote-code "
              f"(auto-recovered).")
        return True

    try:
        while time.time() < deadline:
            rc, st = remote.ssh_capture(target, shlex.join(scheduler.state_command(job_id)), timeout=20)
            state = scheduler.interpret_state(st) if rc == 0 else "UNKNOWN"
            if state != last_state:
                print(f"###   job {job_id}: {state}", flush=True)
                last_state = state
            if state == "DONE":
                tail = _remote_log_tail(target, log_remote)
                if _maybe_proxy_heal(tail) or _maybe_trust_heal(tail):
                    time.sleep(5)
                    continue
                print(f"boxy: job {job_id} ended before the server became ready; last log lines:",
                      file=sys.stderr)
                print(tail, file=sys.stderr)
                # turn a raw compute-node failure (blocked image pull, OOM, bad
                # image tag, …) into a plain-language fix instead of a bare trace.
                _print_diagnosis(tail)
                if _looks_like_pull_block(tail):
                    try:
                        from boxy import ramalama_shim
                        img = box.image or ramalama_shim.default_image(box.engine, location.accelerator)
                    except Exception:
                        img = box.image or ""
                    if img:
                        print(f"###   fastest fix on {host}: pre-pull the image where the network "
                              f"works, then rerun (compute nodes share $HOME's podman store, so the "
                              f"pull is reused with NO re-download):\n"
                              f"###     ssh {target} podman pull {img}\n"
                              f"###   or point boxy at a reachable mirror: --registry <site-mirror> "
                              f"(or --proxy <url> if your proxy allows the registry).", file=sys.stderr)
                return 1
            rc, epj = remote.ssh_capture(target, f"cat {shlex.quote(ep_remote)} 2>/dev/null || true", timeout=15)
            ep = None
            if rc == 0 and epj.strip():
                try:
                    ep = json.loads(epj)
                except ValueError:
                    ep = None
            if ep and ep.get("host") and ep.get("port"):
                print(f"###   server starting on {ep['host']} — confirming readiness from your laptop "
                      f"(localhost/health through the tunnel; boxy stays attached while the job is alive)")

                def _job_alive() -> bool:
                    arc, ast = remote.ssh_capture(target, shlex.join(scheduler.state_command(job_id)),
                                                  timeout=20)
                    return arc == 0 and scheduler.interpret_state(ast) in ("RUNNING", "PENDING")

                ok = remote.await_ready_and_tunnel(
                    target, ep["host"], int(ep["port"]), log_remote,
                    getattr(args, "local_port", None), getattr(args, "route", "") or "",
                    getattr(args, "share", "") or "", getattr(args, "exposer", None) or "relay",
                    getattr(args, "share_auto", False), timeout_s=ready_window,
                    still_alive=_job_alive)
                if ok:
                    print(f"###   stop: boxy stop {name} --ssh {target}")
                    return 0
                # the wait extends while the job is alive, so reaching here means
                # the job ENDED before the server answered — diagnose, don't shrug.
                tail = _remote_log_tail(target, log_remote)
                if _maybe_proxy_heal(tail) or _maybe_trust_heal(tail):
                    time.sleep(5)
                    continue
                print(f"boxy: {scheduler_name} job {job_id} ended before the server became ready; "
                      f"last log lines:", file=sys.stderr)
                print(tail, file=sys.stderr)
                _print_diagnosis(tail)
                print(f"###   full log: boxy logs {name}", file=sys.stderr)
                return 1
            time.sleep(5)
    except KeyboardInterrupt:
        print(f"\n### Detached — {scheduler_name} job {job_id} keeps running on {host}.")
        print(f"###   reattach: boxy attach {name}    stop: boxy stop {name}")
        return 0
    return 1


def _serve_submission(args, scheduler_name: str, profile, name_override: str | None = None,
                      follow: bool = True) -> int:
    """The seamless scheduler path: generate a batch script, submit it, follow
    the job to READY, print the endpoint — then get out of the way.

    `name_override` pins the job name (used by the --replicas fan-out, which owns
    the name and prints the shared auto: lines once). `follow=False` submits and
    returns immediately without the readiness wait (each replica is followed via
    `boxy list`, not a blocking loop per replica)."""
    import time

    from boxy import jobs, readiness, resolve
    from boxy.location import Location, Resources
    from boxy.schedulers import get_scheduler

    model, name, decisions = resolve.resolve_submission(
        args.model, scheduler_name, name=args.name, require_exists=not args.dryrun)
    if name_override is not None:
        name = name_override
    else:
        for line in decisions:
            print(f"  auto: {line}")
        if getattr(args, "unique", False):
            name = _unique_instance_name(name)
            print(f"  auto: name: {name} (--unique — independent instance; "
                  f"log/endpoint/job are its own)")

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
        # A job under a DIFFERENT scheduler that we cannot confirm is alive (state
        # UNKNOWN) is NOT ours to protect: it lives on another cluster (labs share
        # $HOME across sites, so an clusterA flux record shows up on a clusterB slurm
        # login node) or on a scheduler instance we can't reach from here. Blocking
        # the local submission would strand the user. A different-scheduler job we
        # CAN see resolves to PENDING/RUNNING (handled below), never UNKNOWN, so
        # this only fires for genuinely unreachable foreign jobs. (Same-scheduler
        # UNKNOWN is a controller flap — that still blocks, to never double-submit.)
        if mismatch and state == "UNKNOWN":
            print(f"warning: ignoring a stale {rec_sched_name} record for {name} "
                  f"(job {record['job']}) — it can't be reached from this host, so it belongs to "
                  f"another cluster/scheduler. Submitting a fresh {scheduler_name} job and taking "
                  f"over the name here. (If the {rec_sched_name} job is still running, stop it from "
                  f"its own cluster.)", file=sys.stderr)
            if not args.dryrun:
                jobs.remove(name)
        elif state != "DONE":
            endpoint = jobs.read_endpoint(name)
            if endpoint and state in ("PENDING", "RUNNING"):
                model_id = readiness.wait_ready(endpoint["url"], timeout_s=2, interval_s=0.5)
                if model_id:
                    print(f"### ALREADY SERVING  {endpoint['url']}/v1   "
                          f"(model: {model_id}, {rec_sched_name} job {record['job']})")
                    print(f"###   stop: boxy stop {name}")
                    return 0
            if state == "UNKNOWN":
                # same scheduler, controller unreachable: never reap a maybe-live job
                print(f"boxy: cannot determine the state of {rec_sched_name} job "
                      f"{record['job']} ({name}) — scheduler unreachable? Not resubmitting. "
                      f"Retry when it answers, or boxy stop {name}.", file=sys.stderr)
                return 1
            if mismatch:
                print(f"boxy: {name} is already submitted as a {rec_sched_name} job "
                      f"({record['job']}, {state}), but you requested {scheduler_name}. "
                      f"Stop it first: boxy stop {name}.", file=sys.stderr)
                return 1
            # A PUBLISHED endpoint means a server already came up at this name.
            # Never fork a duplicate GPU job just because the 2s readiness probe
            # above was slow (busy server mid-generation, login->compute latency)
            # — point the user at the live endpoint instead (adversarial-review
            # finding: transient probe failure must not spawn a second allocation).
            if endpoint:
                print(f"### ALREADY SERVING  {endpoint['url']}/v1   "
                      f"({rec_sched_name} job {record['job']}, {state}; readiness probe was slow — "
                      f"the server may still be loading its weights)")
                print(f"###   check: curl -s {endpoint['url']}/v1/models    stop: boxy stop {name}")
                print("###   (--unique starts an independent second instance)")
                return 0
            # No endpoint published yet (queued / still starting, nothing serving).
            # Turnkey default: don't block — fork a fresh independent instance.
            # Explicit --unique already renamed above; opt out with --no-auto-unique
            # / BOXY_AUTO_UNIQUE=false to keep the strict singleton.
            if name_override is None and _auto_unique(args):
                forked = _unique_instance_name(name)
                print(f"  auto: name: {forked} ({name} is already {rec_sched_name} job "
                      f"{record['job']} / {state} — starting an independent instance; "
                      f"stop either with `boxy stop <name>`)")
                name = forked
                record = None  # fall through and submit under the fresh name
            else:
                print(f"boxy: {name} is already submitted as {rec_sched_name} job {record['job']} "
                      f"({state}) — watch: boxy list; stop: boxy stop {name} "
                      f"(--unique starts a second instance)", file=sys.stderr)
                return 1
        elif not args.dryrun:
            jobs.remove(name)  # stale record from a finished job (S6: dryrun must not mutate)

    # We are committed to submitting (the record logic above already returned for
    # already-serving / unreachable / mismatched jobs). Now: fail early and
    # clearly if the scheduler isn't on THIS host (no --ssh) — before any site
    # probing — then fill account/partition/time from cards/mywcid/env/sacctmgr.
    if not args.dryrun:
        _require_scheduler_binary(scheduler.submit_command("x")[0], scheduler_name)
    site_args = list(location.scheduler_args)
    # Explicit flags win; each filled value prints an auto: line; the Flux
    # single-queue guard trims a Slurm-style comma partition. (Skip the prints
    # for the --replicas fan-out, which owns the shared auto: block once.)
    from boxy import site

    # a GPU job (the card/flags asked for GPUs per node) must only be offered
    # partitions that actually have accelerators — else it parks in a CPU
    # partition and never starts (field failure).
    need_gpu = location.resources.gpus_per_node > 0
    # interactive WCID picker (once, on the primary submission): several accounts +
    # no --account/pin + a TTY -> menu; sets args.account so resolve_site sees it as
    # explicit and stays silent. A no-op under 'never' (the suite/CI default).
    if name_override is None:
        _pick_account(args, where="")
        _pick_partition(args, scheduler_name, need_gpu, where="")
    site_map, site_decisions = site.resolve_site(args, scheduler_name, need_gpu=need_gpu)
    if name_override is None:
        for line in site_decisions:
            print(f"  auto: {line}")
    for kind in ("partition", "account", "time", "license"):
        if site_map.get(kind):
            site_args.append(scheduler.site_directive(kind, site_map[kind]))
    site_args += list(args.scheduler_args or [])
    dynamic = getattr(args, "dynamic_flags", [])
    site_args += [scheduler.dynamic_directive(k, v) for k, v in _dynamic_for(dynamic, scheduler_name)]
    ignored = _dynamic_ignored(dynamic, scheduler_name)
    if ignored:
        print(f"warning: ignoring {' '.join(ignored)} (active scheduler is {scheduler_name})",
              file=sys.stderr)

    inner = _proxy_prefix(args) + _inner_serve_command(args, model, name)
    # request one task per node when the job will serve distributed (Ray needs a
    # launcher per node). Engine isn't resolved login-side, but a multi-node
    # vllm-shaped request means distributed unless --no-distributed; the harmless
    # case (llama.cpp) just gets one task per node, which is what we want anyway.
    want_distributed = getattr(args, "distributed", None) is not False and location.resources.nodes > 1
    # unique per-job output log: the scheduler substitutes its job-id token
    # (%j / {{id}}) so repeated submissions never overwrite each other's logs.
    output_log = str(jobs.log_path(name, scheduler.output_token) if scheduler.output_token
                     else jobs.log_path(name))
    if getattr(args, "agentless", False):
        # zero-install: the compute node runs pure `podman run` + a bash
        # endpoint-write; NO boxy on the node running the workload (SPEC §8c).
        from dataclasses import replace as dc_replace

        from boxy import deploy

        # the runtime lives on the CLUSTER, not this host — never probe locally;
        # pin podman as the HPC default (override with --runtime / a --location).
        args.runtime = args.runtime or "podman"
        box, _aloc, _dec = _resolve_or_load(args)
        aloc = dc_replace(location, accelerator=(args.accelerator or location.accelerator),
                          runtime=(args.runtime or location.runtime))
        # the container --name/label must match the job/record/endpoint name so
        # boxy list/stop find it (the box's model-slug name would diverge).
        box = dc_replace(box, name=name, image=args.image or box.image)
        try:
            script_text = deploy.render_agentless_script(
                box, aloc, scheduler_name, name, str(jobs.endpoint_path(name)),
                output_log, site_args, proxy_prefix=_proxy_prefix(args), port=args.port)
        except deploy.AgentlessError as e:
            print(f"boxy: agentless: {e}", file=sys.stderr)
            return 2
        print("### Agentless (no boxy on the compute node): the batch script below runs only "
              "podman + a shared-FS endpoint write.")
    else:
        script_text = scheduler.batch_script(inner, location, name, output_log, site_args,
                                             distributed=want_distributed)
    submit = scheduler.submit_command(str(jobs.script_path(name)))
    print(f"### Batch script ({jobs.script_path(name)}):")
    for line in script_text.rstrip().splitlines():
        print(f"    {line}")
    print(f"### Submit Command:\n    {shlex.join(submit)}")
    if args.dryrun:
        return 0
    _require_scheduler_binary(submit[0], scheduler_name)

    jobs.script_path(name).write_text(script_text)
    jobs.endpoint_path(name).unlink(missing_ok=True)
    result = subprocess.run(submit, capture_output=True, text=True)

    # AUTO-RECOVER from a rejected GPU request line (a site that wants
    # --gres=gpu:[type:]N, not --gpus-per-node — field: kahuna): re-render with the
    # portable forms and RESUBMIT until one is accepted, rather than making the
    # user set BOXY_GPU_DIRECTIVE and rerun. Only when the form is auto (else pinned).
    if (result.returncode != 0 and scheduler_name == "slurm" and need_gpu
            and _is_gres_error(result.stderr + result.stdout)
            and config.get_str("site.gpu_directive").strip().lower() == "auto"):
        from boxy import deploy
        from boxy.schedulers import slurm as _slurm

        for gform, gtype in _gres_fallback_forms(_local_gpu_types()):
            _slurm.set_auto_gres(gform, gtype)
            try:
                if getattr(args, "agentless", False):
                    script_text = deploy.render_agentless_script(
                        box, aloc, scheduler_name, name, str(jobs.endpoint_path(name)),
                        output_log, site_args, proxy_prefix=_proxy_prefix(args), port=args.port)
                else:
                    script_text = scheduler.batch_script(inner, location, name, output_log,
                                                         site_args, distributed=want_distributed)
            except deploy.AgentlessError:
                break
            jobs.script_path(name).write_text(script_text)
            shown = _gpu_form_label(gform, gtype)
            print(f"boxy: the site rejected the GPU request; retrying with {shown} ...", file=sys.stderr)
            result = subprocess.run(submit, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"### GPU request accepted as {shown} (auto-recovered).")
                break

    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        print(f"boxy: submission failed: {err}", file=sys.stderr)
        hint = _submission_hint(err)
        if hint:
            print(hint, file=sys.stderr)
        return result.returncode
    job_id = scheduler.parse_job_id(result.stdout)
    expected_log = jobs.log_path(name, job_id)  # where the scheduler should write it
    jobs.write_record(name, {"name": name, "scheduler": scheduler_name, "job": job_id,
                             "model": model, "submitted_from": socket.gethostname(),
                             "log": str(expected_log)})
    print(f"### Submitted {scheduler_name} job {job_id}  ({name})")
    if not follow:
        # --replicas fan-out: don't block on this one; it's tracked via boxy list.
        print(f"###   endpoint (when ready): {jobs.endpoint_path(name)}")
        return 0
    print("### Waiting for the job to start and the server to become ready ... "
          "(Ctrl-C detaches; the job keeps running)")

    t_start = time.time()
    last_state, ready_deadline, ready_window = None, None, 0.0
    last_note = time.time()
    unknown_streak = 0
    endpoint_seen = False
    try:
        while True:
            state = _job_state(scheduler, job_id)
            unknown_streak = unknown_streak + 1 if state == "UNKNOWN" else 0
            if unknown_streak >= 10:
                # scheduler unreachable / unmapped state: never spin silently
                # forever (r2 audit) — detach and leave the job alone
                print(f"boxy: cannot determine job {job_id}'s state (scheduler unreachable?) — "
                      f"detaching; the job (if alive) keeps running.\n"
                      f"  status: boxy list    log: {expected_log}\n"
                      f"  stop:   boxy stop {name}", file=sys.stderr)
                return 1
            endpoint = jobs.read_endpoint(name)
            endpoint_seen = endpoint_seen or bool(endpoint)
            if state != last_state:
                print(f"###   [{_fmt_elapsed(time.time() - t_start)}] job {job_id}: {state}", flush=True)
                last_state = state
                last_note = time.time()
            elif time.time() - last_note >= 10:
                # a live progress line: elapsed clock + phase + a bar parsed from
                # the engine/container log (weight-load %, CUDA-graph capture, image
                # pull), so a long load reads as forward motion, not a silent spinner.
                marker, frac = _parse_load_progress(jobs.resolve_log(name, job_id))
                phase = _phase_for(state, endpoint_seen, marker)
                line = f"###   [{_fmt_elapsed(time.time() - t_start)}] {phase}"
                if frac is not None:
                    line += f"  {_progress_bar(frac)}"
                else:
                    tail = _last_log_line(jobs.resolve_log(name, job_id), maxlen=80)
                    if tail:
                        line += f"  ›  {tail}"
                print(f"{line}   (job {job_id})", flush=True)
                last_note = time.time()
            if endpoint:
                url = endpoint["url"]
                if ready_deadline is None:
                    # LLM weight-load + CUDA-graph capture legitimately takes many
                    # minutes; never give up on a RUNNING job before this window
                    # (a CRASH is caught immediately by the state==DONE check, so a
                    # generous floor only ever waits on genuinely-slow loads).
                    # --ready-timeout 0 (submit-and-detach) is respected as-is.
                    ready_window = (args.ready_timeout if args.ready_timeout <= 0
                                    else max(args.ready_timeout, _SCHED_READY_FLOOR))
                    ready_deadline = time.time() + max(ready_window, 0.0)
                    print(f"###   server starting on {endpoint['host']} — waiting up to "
                          f"{ready_window / 60:.0f} min for readiness ({url}/health, checked on "
                          f"the compute node) (Ctrl-C detaches; the job keeps loading)")
                # readiness, in order of authority:
                #  1) the endpoint file's `ready` flag — set by the COMPUTE node's
                #     own localhost:port/health probe (no proxy, no cross-node
                #     routing). This is the ideal signal and needs no probe here.
                #  2) our own /health|/v1/models probe (proxy-bypassed).
                #  3) the engine's "server is up" line in the shared-FS log.
                # (2)+(3) cover a compute node running an OLDER boxy that doesn't
                # write the ready flag. Field report: vLLM logged "Application
                # startup complete." on cronus5 but http://cronus5:8000 was
                # unreachable from the login node, so boxy looped forever.
                if endpoint.get("ready"):
                    model_id = endpoint.get("model") or "ready"
                else:
                    model_id = readiness.wait_ready(url, timeout_s=3, interval_s=1,
                                                    log_path=jobs.resolve_log(name, job_id))
                if model_id:
                    print(f"### READY  {url}/v1   (model: {model_id}, {scheduler_name} job {job_id})")
                    print(f"###   try:   curl -s {url}/v1/models")
                    print(f"###   tunnel: ssh -L {endpoint['port']}:{endpoint['host']}:{endpoint['port']} <login-node>")
                    print(f"###   stop:  boxy stop {name}")
                    return 0
                if time.time() > ready_deadline:
                    print(f"### DETACHED — {scheduler_name} job {job_id} is still RUNNING and loading "
                          f"(waited {ready_window / 60:.0f} min); boxy stopped watching but the SERVER IS NOT DEAD.\n"
                          f"###   reconnect: boxy open {name}   (add --ssh <host> from your laptop)\n"
                          f"###   watch:     tail -f {expected_log}\n"
                          f"###   wait longer next time: --ready-timeout 1800  (or export BOXY_READY_TIMEOUT=1800)\n"
                          f"###   stop:      boxy stop {name}", file=sys.stderr)
                    return 1
            if state == "DONE":
                print(f"boxy: job {job_id} ended before the server became ready; last log lines:",
                      file=sys.stderr)
                actual_log = jobs.resolve_log(name, job_id)  # the file the scheduler really wrote
                _dump_file_tail(actual_log)
                _diagnose_file(actual_log)
                jobs.remove(name)
                return 1
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"\n### Detached — {scheduler_name} job {job_id} keeps running.")
        print(f"###   status: boxy list      endpoint file: {jobs.endpoint_path(name)}")
        print(f"###   stop:   boxy stop {name}")
        return 0


def _serve_replicas_multinode(args, scheduler_name, profile, replicas, base_name, router_port,
                              nodes_per_replica) -> int:
    """--replicas K --nodes-per-replica M: K replicas, each a MULTI-NODE distributed
    instance (a full Ray job spanning M nodes). One distributed job per replica."""
    import copy

    from boxy import jobs

    # each replica submission is a distributed job of M nodes (distributed auto-on
    # for nodes>1); --nodes-per-replica is the per-replica span, so set the geometry.
    per_replica = copy.copy(args)
    per_replica.nodes = nodes_per_replica
    names = [f"{base_name}-r{i}" for i in range(replicas)]
    print(f"  auto: replicas: {replicas} distributed instances of {base_name} — each "
          f"{nodes_per_replica} nodes x {args.gpus or '?'} GPU (data-parallel of model-parallel)")
    rcs = []
    for nm in names:
        print(f"\n### Replica {nm}")
        rcs.append(_serve_submission(per_replica, scheduler_name, profile, name_override=nm, follow=False))
    if args.dryrun:
        print(f"\n### dryrun: {replicas} distributed replica job(s) shown above; nothing submitted")
        return 0
    submitted = [nm for nm, rc in zip(names, rcs) if rc == 0]
    print(f"\n### replicas: {len(submitted)}/{replicas} submitted")
    for nm in submitted:
        rec = jobs.read_record(nm)
        if rec:
            print(f"###   {nm}: {scheduler_name} job {rec['job']}")
    if submitted:
        print("###   watch: boxy list      stop: boxy stop <name>")
    if router_port and submitted:
        _run_router(base_name, router_port, args.ready_timeout, submitted)
    return 0 if len(submitted) == replicas else 1


def _serve_replicas(args, scheduler_name: str, profile, replicas: int, router_port: int | None = None) -> int:
    """Data-parallel fan-out. By default (single-node replicas) K replicas BIN-PACK
    onto a node's GPUs — rpn = --gpus // --gpus-per-replica per node, each replica
    pinned to its own GPU(s) on its own port — so K replicas take ceil(K/rpn) node
    jobs, not one whole node each. With --nodes>1 each replica is instead a
    multi-node distributed instance (see _serve_replicas_multinode). With
    `router_port`, once the replicas are READY a login-node router fronts them."""
    import math
    from dataclasses import replace as dc_replace

    from boxy import jobs, resolve
    from boxy.location import Location, Resources
    from boxy.schedulers import get_scheduler

    _model, base_name, decisions = resolve.resolve_submission(
        args.model, scheduler_name, name=args.name, require_exists=not args.dryrun)
    for line in decisions:
        print(f"  auto: {line}")
    if getattr(args, "unique", False):
        base_name = _unique_instance_name(base_name)

    npr = max(1, getattr(args, "nodes_per_replica", 1) or 1)
    if npr > 1:
        # each replica is itself a MULTI-NODE distributed (Ray) instance spanning M
        # nodes: data-parallel OF model-parallel (explicit opt-in).
        return _serve_replicas_multinode(args, scheduler_name, profile, replicas,
                                         base_name, router_port, nodes_per_replica=npr)

    # Bin-pack across a node pool. A node fits rpn_cap = --gpus // R replicas (each R
    # GPUs). --nodes N spreads the K replicas across N nodes (ceil(K/N) per node);
    # without --nodes, pack tight (rpn_cap per node). Each replica is pinned to its
    # own GPU(s) on its own port. --nodes is the POOL size here, never per-replica.
    r = max(1, getattr(args, "gpus_per_replica", 1) or 1)
    gpus_per_node = args.gpus if args.gpus else r
    if gpus_per_node < r:
        print(f"boxy: --gpus {gpus_per_node} < --gpus-per-replica {r}: a replica needs {r} GPU(s) "
              f"but the per-node budget is {gpus_per_node}. Raise --gpus or lower "
              f"--gpus-per-replica.", file=sys.stderr)
        return 2
    rpn_cap = max(1, gpus_per_node // r)
    if args.nodes and args.nodes > 1:
        per_node = math.ceil(replicas / args.nodes)
        if per_node > rpn_cap:
            print(f"boxy: {replicas} replicas across {args.nodes} node(s) needs {per_node} per node, "
                  f"but a node only fits {rpn_cap} at {r} GPU(s) each (--gpus {gpus_per_node}). "
                  f"Raise --gpus or --nodes, or lower --replicas.", file=sys.stderr)
            return 2
    else:
        per_node = rpn_cap
    nodes = math.ceil(replicas / per_node)
    replica_names = [f"{base_name}-r{i}" for i in range(replicas)]
    tp_args = ["--tensor-parallel-size", str(r)] if r > 1 else None
    print(f"  auto: replicas: {replicas} x {r} GPU (tensor-parallel={r}), {per_node}/node "
          f"across {nodes} node job(s){' + router' if router_port else ''}")

    base_loc = (profile if profile is not None
                else Location(name="auto", scheduler=scheduler_name, resources=Resources()))
    scheduler = get_scheduler(scheduler_name)
    # Route through the SAME site resolver as the single-serve path so
    # --partition auto/all/off (and the auto default) work for replicas too —
    # else the magic keywords reach sbatch verbatim ('invalid partition') and a
    # GPU replicas job can park in the site-default CPU partition (review find).
    from boxy import site

    site_map, site_decisions = site.resolve_site(args, scheduler_name, need_gpu=gpus_per_node > 0)
    for line in site_decisions:
        print(f"  auto: {line}")
    site_args = list(base_loc.scheduler_args)
    for kind in ("partition", "account", "time"):
        if site_map.get(kind):
            site_args.append(scheduler.site_directive(kind, site_map[kind]))
    site_args += list(getattr(args, "scheduler_args", None) or [])
    dynamic = getattr(args, "dynamic_flags", [])
    site_args += [scheduler.dynamic_directive(k, v) for k, v in _dynamic_for(dynamic, scheduler_name)]

    submitted_jobs: list[tuple[str, str]] = []
    failed = 0
    for n in range(nodes):
        members = list(range(n * per_node, min(replicas, (n + 1) * per_node)))
        m = len(members)
        job_name = base_name if nodes == 1 else f"{base_name}-n{n}"
        loc = dc_replace(base_loc, resources=Resources(
            nodes=1, gpus_per_node=m * r, accelerator_type=base_loc.resources.accelerator_type))
        proxy_prefix = _proxy_prefix(args)
        inner_cmds = []
        for slot, i in enumerate(members):
            ids = ",".join(str(g) for g in range(slot * r, slot * r + r))
            inner_cmds.append(proxy_prefix + _inner_serve_command(
                args, args.model, replica_names[i],
                port=config.get_int("network.replica_port_base") + slot, visible_gpus=ids,
                gpus=r, forward_geometry=False, extra_engine_args=tp_args))
        output_log = str(jobs.log_path(job_name, scheduler.output_token) if scheduler.output_token
                         else jobs.log_path(job_name))
        script_text = scheduler.group_batch_script(inner_cmds, loc, job_name, output_log, site_args)
        submit = scheduler.submit_command(str(jobs.script_path(job_name)))
        print(f"\n### Node job {job_name}: {m} replica(s) x {r} GPU = {m * r} GPU on 1 node "
              f"({', '.join(replica_names[i] for i in members)})")
        print(f"### Batch script ({jobs.script_path(job_name)}):")
        for line in script_text.rstrip().splitlines():
            print(f"    {line}")
        print(f"### Submit Command:\n    {shlex.join(submit)}")
        if args.dryrun:
            continue
        _require_scheduler_binary(submit[0], scheduler_name)
        jobs.script_path(job_name).write_text(script_text)
        for i in members:
            jobs.endpoint_path(replica_names[i]).unlink(missing_ok=True)
        result = subprocess.run(submit, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"boxy: submission failed for {job_name}: "
                  f"{result.stderr.strip() or result.stdout.strip()}", file=sys.stderr)
            failed += 1
            continue
        job_id = scheduler.parse_job_id(result.stdout)
        jobs.write_record(job_name, {
            "name": job_name, "scheduler": scheduler_name, "job": job_id, "model": args.model,
            "submitted_from": socket.gethostname(), "replicas": [replica_names[i] for i in members],
            "log": str(jobs.log_path(job_name, job_id))})
        submitted_jobs.append((job_name, job_id))
        print(f"### Submitted {scheduler_name} job {job_id}  ({job_name})")

    if args.dryrun:
        print(f"\n### dryrun: {nodes} node job(s) for {replicas} replicas ({per_node}/node); "
              f"nothing submitted")
        if router_port:
            print(f"### dryrun: would then start the login-node router on :{router_port} "
                  f"fronting {base_name}-r*")
        return 0
    print(f"\n### replicas: {len(submitted_jobs)}/{nodes} node job(s) submitted for {replicas} replicas")
    for jn, jid in submitted_jobs:
        print(f"###   {jn}: {scheduler_name} job {jid}")
    if submitted_jobs:
        stop_hint = base_name if nodes == 1 else f"{base_name}-n0 .. -n{nodes - 1}"
        print(f"###   watch: boxy list      stop: boxy stop {stop_hint}")
    if router_port and submitted_jobs:
        _run_router(base_name, router_port, args.ready_timeout, replica_names)
    return 0 if failed == 0 else 1


def _run_router(base_name: str, port: int, ready_timeout: float, names: list[str]) -> None:
    """Wait for the replicas to become READY, then run the built-in login-node
    router in the foreground on `port`, fronting them with one OpenAI URL."""
    from boxy import router

    urls = _sweep_wait_endpoints(names, ready_timeout)
    if not urls:
        print(f"### router: no replica became ready within {ready_timeout:.0f}s; not starting router",
              file=sys.stderr)
        return
    pool = router.Pool()
    disc = router.DiscoveryThread(base_name, pool)
    disc.scan_once()  # populate before accepting requests (avoid first-hit 503)
    disc.start()
    host = socket.gethostname()
    srv = router.make_server(pool, port)
    print(f"### Router  http://{host}:{port}/v1  -> {base_name}-r* "
          f"({len(pool.snapshot())} replica(s), least-outstanding)")
    print(f"###   from your workstation: ssh -L {port}:{host}:{port} <login-node>")
    print("###   Ctrl-C stops the router; the replicas keep running (boxy list / boxy stop <name>)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n### Router stopped; replicas still running.")
    finally:
        srv.shutdown()
        disc.stop()


def _rename_container(cmd: list[str], old: str, new: str) -> list[str]:
    """Clone a container run command with a fresh --name/label (for local worker
    replicas that would otherwise collide on the container name)."""
    return [a.replace(f"--name={old}", f"--name={new}").replace(f"boxy.box={old}", f"boxy.box={new}")
            for a in cmd]


def _serve_distributed(args, box, location) -> int:
    """Serve ONE vLLM instance across the allocation via a Ray cluster. This
    (head) node runs `ray start --head` + `vllm serve`; the other nodes join. The
    worker placement adapts to the allocation we're in: srun (Slurm), flux run
    (Flux), or a set of containers on the local host (no scheduler)."""
    import subprocess

    from boxy import deploy, distributed, jobs

    # How workers are placed across the allocation: the location's scheduler when
    # it names one (so a login-node dryrun for a slurm/flux profile previews the
    # right srun/flux-run fan-out), else the live allocation env — SLURM_JOB_ID/
    # FLUX_JOB_ID — for the submitted-into-allocation case where the re-invoked
    # boxy carries no --location, else a local set of containers.
    launcher = (location.scheduler
                if location.scheduler in ("slurm", "flux")
                else distributed.detect_launcher())
    head_node, head_ip, _ = distributed.discover_topology(launcher)
    nodes, gpus = location.resources.nodes, location.resources.gpus_per_node
    try:
        dep = deploy.plan_serve(box, location, port=args.port, extra_args=args.args,
                                dryrun=args.dryrun, distributed=True, head_ip=head_ip)
    except RuntimeError as e:  # e.g. gpus_per_node unknown
        print(f"boxy: {e}", file=sys.stderr)
        return 2
    tp, pp = dep.parallelism
    worker = dep.worker_command or []
    prefix = distributed.worker_launch_prefix(launcher, head_node, nodes)
    # slurm/flux fan ONE worker command out to the N-1 nodes; 'none' runs N-1
    # worker containers locally, each with its own name.
    worker_cmds = ([prefix + worker] if prefix
                   else [_rename_container(worker, f"{box.name}-worker", f"{box.name}-worker{i}")
                         for i in range(nodes - 1)])
    print(f"  auto: distributed vLLM: {nodes} nodes x {gpus} GPU -> tensor-parallel={tp}, "
          f"pipeline-parallel={pp} (world {dep.world_size}) via Ray "
          f"({'local containers' if launcher == 'none' else launcher} launcher)")
    for w in dep.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"### Head ({head_node}):\n    {shlex.join(dep.command)}")
    for wc in worker_cmds:
        print(f"### Worker:\n    {shlex.join(wc)}")
    if args.dryrun:
        return 0
    if getattr(args, "endpoint_file", None):
        jobs.write_endpoint_file(
            args.endpoint_file, name=dep.box.name, port=dep.port,
            job_id=os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_JOB_ID", ""))
    print(f"### Endpoint (once the cluster forms and the model loads): "
          f"http://{head_node}:{dep.port}/v1")
    procs = [subprocess.Popen(wc) for wc in worker_cmds]
    try:
        return deploy.execute(dep)  # head: ray head + wait-for-cluster + vllm serve (foreground)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


def _delegate_remote(args, tunnel_ready: bool = False) -> int | None:
    """From-anywhere hook: when a remote target is configured (--ssh flag,
    BOXY_SSH_HOST, or `remote=` in the --location profile) and we are NOT already
    on the cluster, re-run this exact command there over one multiplexed SSH
    session and (for serve) tunnel the READY endpoint back. Returns None to run
    locally. Fully modular: only this hook knows remote exists."""
    from boxy import remote

    if os.environ.get(remote.ENV_ACTIVE):
        return None  # we ARE the remote side
    target = remote.resolve_target(args)
    if not target:
        return None
    # already ON the target (profile with remote= used on the login node itself)?
    target_short = target.split("@")[-1].split(".")[0]
    if target_short and target_short == socket.gethostname().split(".")[0]:
        return None

    # Fully-agentless serve is the DEFAULT over --ssh — NOTHING is installed on the
    # HPC. A batch-eligible model serve (not --foreground/--box/--here, single
    # instance) is rendered + submitted + polled from the laptop. Opt out with
    # --delegate / BOXY_SSH_DELEGATE=1 to run the cluster's own boxy (needed for
    # --replicas / --distributed / --box, which the agentless path doesn't cover yet).
    if (getattr(args, "subcommand", None) == "serve"
            and bool(getattr(args, "model", None))
            and not getattr(args, "foreground", False)
            and not getattr(args, "box", None)
            and not getattr(args, "here", False)
            and (getattr(args, "replicas", 1) or 1) == 1
            and getattr(args, "distributed", None) is not True
            and not getattr(args, "delegate", False)
            and not os.environ.get("BOXY_SSH_DELEGATE")
            and config.get_bool("serve.agentless_ssh")):
        share_name, share_auto = _auto_share_name(args)
        args.share, args.share_auto = share_name, share_auto
        if share_auto and share_name:
            print(f"  auto: share: {share_name} (publishing a team URL once ready — no relay "
                  f"degrades to the local tunnel; BOXY_AUTO_SHARE=false to skip)")
        return _serve_agentless_ssh(args, target)

    raw_argv = _inject_remote_site(args, target, getattr(args, "_raw_argv", []))
    share_name, share_auto = _auto_share_name(args)
    if share_auto and share_name:
        print(f"  auto: share: {share_name} (publishing a team URL — no relay degrades to the "
              f"local tunnel; set BOXY_AUTO_SHARE=false to skip)")
    return remote.run_remote(target, raw_argv, tunnel_ready=tunnel_ready,
                             local_port=getattr(args, "local_port", None),
                             local_route=getattr(args, "route", "") or "",
                             share=share_name, share_auto=share_auto,
                             exposer_name=getattr(args, "exposer", None) or "relay")


def _auto_share_name(args) -> tuple[str, bool]:
    """(share_alias, is_auto). Turnkey: over --ssh a served model auto-publishes a
    team URL so the user need not type --share (config serve.auto_share, default
    on). Explicit --share always wins. Skipped for a direct serve (--foreground/
    --box) or when sharing is disabled. The alias is derived from the model's
    instance name (sanitized for the relay hostname). Best-effort downstream: a
    missing relay degrades to the local tunnel (run_remote catches it)."""
    explicit = getattr(args, "share", None)
    if explicit:
        return explicit, False
    if not getattr(args, "model", None) or getattr(args, "foreground", False) or getattr(args, "box", None):
        return "", False
    if not config.get_bool("share.enabled") or not config.get_bool("serve.auto_share"):
        return "", False
    base = ""
    try:
        from boxy import resolve

        _, base, _ = resolve.resolve_submission(
            args.model, getattr(args, "scheduler", None) or "slurm",
            name=getattr(args, "name", None), require_exists=False)
    except Exception:  # noqa: BLE001 — never block the serve on name resolution
        base = ""
    # relay hostnames are DNS labels: lowercase alnum + dashes, <= 40 chars.
    alias = re.sub(r"[^a-z0-9-]+", "-", (base or "").removeprefix("boxy-").lower()).strip("-")[:40].strip("-")
    return (alias, True) if alias else ("", False)


def _argv_set_flag(argv: list[str], flag: str, value: str | None) -> list[str]:
    """Return argv with `flag`'s value set to `value` in place (handles both
    `--flag V` and `--flag=V`), or the flag+value REMOVED when value is None.
    Used to rewrite a delegated `--partition auto` into a concrete list (or drop
    it) so an older cluster boxy never receives the literal word 'auto'."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == flag:
            if value is not None:
                out += [flag, value]
            i += 2  # skip the flag AND its separate value token
            continue
        if tok.startswith(flag + "="):
            if value is not None:
                out.append(f"{flag}={value}")
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _job_wants_gpu(args) -> bool:
    """Does this scheduler job need a GPU? --gpus wins; else the model card's GPU
    count (resolved laptop-side, since the card isn't applied to args until after
    delegation); else default True — HPC serving is GPU-bound, and the GPU
    partition filter falls back to all partitions when none are identifiable."""
    g = getattr(args, "gpus", None)
    if g is not None:
        return g > 0
    from boxy import cards

    card = cards.resolve_model_card(getattr(args, "model", "") or "")
    if card is not None and card.gpus:
        return card.gpus > 0
    return True


def _inject_remote_site(args, target: str, raw_argv: list[str]) -> list[str]:
    """Turnkey over --ssh (field failures, 2026-07): the delegated command runs
    the CLUSTER's boxy — which may predate turnkey — so site resolution never
    happens there. Resolve it HERE, laptop-side, against the live ssh master and
    rewrite the delegated argv:

      * account — append `--account <val>` (from config/env, else a `mywcid`
        probe ON the cluster) so the batch script carries it.
      * partition — auto is the DEFAULT: probe the cluster's partitions and put a
        concrete soonest-start (GPU-aware) list on the delegated command, so the
        job starts wherever a GPU frees first instead of parking in one queue.
        The literal 'auto'/'all' is resolved HERE — an older cluster boxy would
        otherwise pass it to sbatch ('invalid partition').

    Explicit values win untouched. Opt out: BOXY_NO_REMOTE_ACCOUNT=1. Best-effort
    throughout: any probe failure leaves the site default in place."""
    if os.environ.get("BOXY_NO_REMOTE_ACCOUNT"):
        return raw_argv
    if not raw_argv or raw_argv[0] != "serve":
        return raw_argv
    # Auto-detection only applies to the seamless model-submission path. A direct
    # serve (--foreground / --box / --here) or one with no model is left untouched,
    # so a remote host that merely HAS a scheduler isn't turned into a batch job.
    scheduler = getattr(args, "scheduler", None)
    eligible = bool(getattr(args, "model", None)) and not getattr(args, "foreground", False) \
        and not getattr(args, "box", None) and not getattr(args, "here", False)
    if scheduler not in ("slurm", "flux") and not eligible:
        return raw_argv
    from boxy import cards, remote, resolve, site

    # one master for every probe below (idempotent; run_remote reuses it)
    master_ok = remote.ensure_master(target) == 0
    host = target.split("@")[-1]

    # --- scheduler: auto is the DEFAULT (no --scheduler). Detect it over the ssh
    # master (flux/sbatch on the cluster) so a novice serving over --ssh types
    # only the model name; config site.scheduler or an explicit --scheduler win.
    # The detected value is INJECTED as --scheduler so an older cluster boxy —
    # which can't auto-detect — still submits a batch job. None detectable (or
    # config=none) => leave the delegated command a direct/local serve.
    inject_scheduler = None
    if scheduler not in ("slurm", "flux"):
        avail = ""
        if master_ok:
            rc, avail = remote.ssh_capture(target, site.remote_scheduler_probe(), timeout=20)
            avail = avail if rc == 0 else ""
        detected, why = site.pick_scheduler(avail, None)
        if detected in ("slurm", "flux"):
            scheduler = inject_scheduler = detected
            print(f"  auto: scheduler: {detected} (via {why} on {host})")
        else:
            return raw_argv

    # Split any user engine args (after `--`) off the boxy side. boxy-flag
    # injections (account/partition/unique) go on `head`; card engine args go on
    # the engine side — else an appended --account would land AFTER `--` and be
    # passed to vLLM.
    if "--" in raw_argv:
        sep = raw_argv.index("--")
        head, user_tail, had_sep = list(raw_argv[:sep]), list(raw_argv[sep + 1:]), True
    else:
        head, user_tail, had_sep = list(raw_argv), [], False

    # inject the detected scheduler so the (possibly old) cluster boxy submits a
    # batch job rather than a direct serve.
    if inject_scheduler:
        head = [*head, "--scheduler", inject_scheduler]

    # --- partition: resolve auto/all to a concrete list over the cluster ---
    # auto is the default (no flag); `all` offers every partition; `off`/a
    # concrete name are left for the cluster boxy. A flag present on the argv is
    # REPLACED in place; the default (no flag) is APPENDED.
    part = getattr(args, "partition", None)
    mode = site.partition_mode(part)
    if mode in ("auto", "all"):
        prefer_gpu = mode == "auto" and _job_wants_gpu(args)
        value, why = "", "no ssh master to probe partitions"
        if master_ok:
            rc, out = remote.ssh_capture(target, site.remote_partition_probe(scheduler), timeout=20)
            value, why = site.rank_remote_partitions(out, scheduler, prefer_gpu) if rc == 0 else ("", why)
        flagged = bool(part)
        if value:
            names = [p for p in value.split(",") if p]
            value, pnote = _pick_remote_partition(args, names, host, target, scheduler)
            head = (_argv_set_flag(head, "--partition", value) if flagged
                    else [*head, "--partition", value])
            print(f"  auto: partition: {value} (via {pnote or f'sinfo on {host}: {why}'})")
        else:
            if flagged:
                head = _argv_set_flag(head, "--partition", None)  # drop unresolved auto/all
            # default (no flag): nothing to drop; the cluster's site default applies
            print(f"  auto: partition: could not pick on {target} ({why}) — "
                  f"the scheduler's site default applies")

    # --- account: append --account unless one was given explicitly ---
    if not getattr(args, "account", None):
        acct, why = site.resolve_account(None)   # laptop-side config/env first
        if not acct and master_ok:
            rc, out = remote.ssh_capture(target, site.remote_account_probe(), timeout=20)
            rows = site.parse_account_rows(out) if rc == 0 else []
            # pick laptop-side (a TTY menu when several + the picker is on, else the
            # first) over the cluster-probed list; it rides the delegated command.
            acct, why = _pick_remote_account(args, rows, host, target)
            if not acct:
                first = next((ln.strip()[:100] for ln in (out or "").splitlines() if ln.strip()), "")
                detail = f" (probe answered: {first!r})" if first else ""
                print(f"  auto: account: none discovered on {target}{detail} — the scheduler's "
                      f"site default applies; pass --account if it rejects the job")
        if acct:
            print(f"  auto: account: {acct} (via {why} — placed in the batch script)")
            head = [*head, "--account", acct]

    # --- time: inject the config default walltime (30 min) unless --time was
    # given, so the served job carries a walltime even against an OLD cluster
    # boxy whose default lives laptop-side. NOTE: the scheduler KILLS the job at
    # the walltime — raise BOXY_DEFAULT_TIME / pass --time for long sessions.
    if not getattr(args, "time", None) and "--time" not in head:
        t, twhy = site.resolve_time(None)
        if t:
            print(f"  auto: time: {t} (via {twhy} — the scheduler stops the job at this walltime)")
            head = [*head, "--time", t]

    # --- readiness timeout: raise the delegated boxy's readiness wait so an OLD
    # cluster boxy (whose default is only 180s) doesn't give up while a big model
    # is still loading — the field failure "server not ready within 180s (job
    # still RUNNING)". boxy reports READY the instant the endpoint answers, so a
    # large ceiling never over-waits; it only stops a premature give-up. Injected
    # unless the user set --ready-timeout (0 = submit-and-detach) themselves.
    if not any(t == "--ready-timeout" or t.startswith("--ready-timeout=") for t in head) \
            and getattr(args, "ready_timeout", None) != 0:
        floor = int(_SCHED_READY_FLOOR)
        print(f"  auto: ready-timeout: {floor // 60} min (waits for the model to finish loading; "
              f"boxy prints READY the moment the server answers)")
        head = [*head, "--ready-timeout", str(floor)]

    # --- auto-unique: if a job with this model's name is already LIVE on the
    # cluster, inject --unique so the cluster's boxy (even an old one that lacks
    # auto-unique) starts an independent instance instead of blocking. Runs the
    # decision HERE because over --ssh the singleton check runs on the cluster.
    if (master_ok and _auto_unique(args) and "--unique" not in head
            and getattr(args, "model", None)):
        try:
            _, base_name, _ = resolve.resolve_submission(
                args.model, scheduler, name=getattr(args, "name", None), require_exists=False)
        except Exception:  # noqa: BLE001 — never block delegation on name resolution
            base_name = ""
        if base_name:
            rc, out = remote.ssh_capture(
                target, site.remote_jobname_live_probe(scheduler, base_name), timeout=20)
            if rc == 0 and "LIVE" in out:
                print(f"  auto: --unique ({base_name} is already live on {host} — "
                      f"starting an independent instance)")
                head = [*head, "--unique"]

    # --- engine args from the model card (e.g. --max-model-len so vLLM doesn't
    # OOM profiling the full 128K context). Injected AFTER `--` so even an OLD
    # cluster boxy — which won't apply the card itself — passes them to the
    # engine. Card flags first; the user's own post-`--` args win (last-wins).
    engine_tail = list(user_tail)
    if getattr(args, "model", None):
        card = cards.resolve_model_card(args.model)
        flags = cards.engine_flags(card.args) if (card and card.args) else []
        if flags:
            engine_tail = flags + engine_tail
            print(f"  auto: engine args: {' '.join(flags)} ({card.label} — placed after --)")

    if had_sep or engine_tail:
        return [*head, "--", *engine_tail]
    return head


def cmd_serve(args: argparse.Namespace) -> int:
    # A --system card is a built-in deployment profile: materialize it to a TOML
    # and feed it through the SAME --location machinery (explicit flags still win
    # via the overlay). --location wins if both are given.
    if getattr(args, "system", None) and not args.location:
        from boxy import cards

        try:
            args.location = cards.system_card_path(args.system)
        except ValueError as e:
            raise UsageError(str(e))
        print(f"  auto: system: {args.system} (built-in system card)")
    # resolve the proxy into env BEFORE delegating so run_remote forwards it to
    # the cluster over --ssh (a --proxy/config proxy known only on the laptop
    # otherwise never reaches the cluster job).
    _apply_proxy_env(args)  # --proxy / config reaches the login-node model pull AND the delegated cmd
    rc = _delegate_remote(args, tunnel_ready=True)
    if rc is not None:
        return rc
    _apply_bind_host_env(args)  # --bind-host wins over env/file/default for this serve
    if getattr(args, "share", None):
        raise UsageError("--share needs the laptop tunnel (`boxy serve ... --ssh user@login "
                         "--share NAME`) — it publishes the LOCAL end of the forward")
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
        # Turnkey on the login node itself (no --ssh, no --location scheduler):
        # honor an EXPLICIT config BOXY_SCHEDULER=slurm|flux so `boxy serve MODEL`
        # submits a batch job without the flag. We deliberately do NOT probe PATH
        # under the default 'auto' here: locally the login-node guard already
        # gives a clear "add --scheduler / --here" message, and silently
        # submitting on any host that merely has sbatch would surprise a direct
        # serve. Over --ssh (where the target IS a cluster) auto-probing is on.
        if scheduler_name is None and not getattr(args, "here", False):
            cfg_sched = config.get_str("site.scheduler").strip().lower()
            if cfg_sched in ("slurm", "flux"):
                scheduler_name = cfg_sched
                print(f"  auto: scheduler: {cfg_sched} (via config site.scheduler)")
        if scheduler_name in ("slurm", "flux"):
            # Turnkey: fill --gpus/--nodes/--engine from the model's card (or the
            # size heuristic) when the flags are absent — a novice types only the
            # model name; explicit flags always win (cards.apply_to_args).
            from boxy import cards

            for line in cards.apply_to_args(args):
                print(f"  auto: {line}")
            replicas = getattr(args, "replicas", 1) or 1
            if replicas > 1:
                return _serve_replicas(args, scheduler_name, profile, replicas,
                                       router_port=getattr(args, "router", None))
            if getattr(args, "router", None):
                print("boxy: --router load-balances across --replicas; add --replicas K "
                      "(K>1). For a single instance just use its endpoint.", file=sys.stderr)
                return 2
            return _serve_submission(args, scheduler_name, profile)

    if (getattr(args, "replicas", 1) or 1) > 1:
        print("boxy: --replicas is supported on the batch-submission path "
              "(boxy serve MODEL --scheduler slurm|flux). It doesn't apply to --box, "
              "--foreground, or scheduler=none. For multiple LOCAL instances, run "
              "`boxy serve MODEL --unique` once per instance.", file=sys.stderr)
        return 2

    box, location, decisions = _resolve_or_load(args)
    for line in decisions:
        print(f"  auto: {line}")
    vgpus = getattr(args, "visible_gpus", None)
    if vgpus:
        # pin this instance to specific GPU ids inside the container (co-located
        # --replicas: N servers share a node, each on its own GPU). Set every
        # accelerator's app-level selector; box.env still wins if the user set one.
        from dataclasses import replace as _replace

        pins = {k: vgpus for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
        box = _replace(box, env={**pins, **box.env})
        print(f"  auto: GPU pin: visible devices {vgpus} (CUDA/HIP/ROCR_VISIBLE_DEVICES)")
    if getattr(args, "trust_remote_code", False):
        # vLLM-only flag; llama.cpp's server rejects unknown args. Fold it into the
        # engine extras so it reaches plan_serve AND the distributed/replica paths.
        if box.engine == "vllm":
            if "--trust-remote-code" not in (args.args or []):
                args.args = list(args.args or []) + ["--trust-remote-code"]
            print("  auto: trust-remote-code: enabled (vLLM will run the model's custom code)")
        else:
            print(f"  auto: trust-remote-code: ignored (engine is {box.engine}, not vllm)",
                  file=sys.stderr)
    mdir = getattr(args, "models_dir", None) or os.environ.get("BOXY_MODELS_DIR")
    if mdir:
        # where s3://... models are downloaded (and where ${MODELS_DIR} expands)
        from dataclasses import replace as _replace

        location = _replace(location, staging=_replace(location.staging, models_dir=mdir))
        print(f"  auto: download dir: {os.path.abspath(mdir)} (--models-dir)")
    if getattr(args, "registry", None):
        from dataclasses import replace as _replace

        location = _replace(location, registry=args.registry)
        print(f"  auto: image registry: {args.registry} (--registry — images rewritten to it)")
    if getattr(args, "unique", False) and args.model and not args.box:
        # container edition of --unique: a fresh name (and thus container name +
        # label) per launch, so `boxy serve MODEL --unique` x N coexist. Ports
        # already auto-increment for scheduler=none, so no port clash either.
        from dataclasses import replace as _replace

        box = _replace(box, name=_unique_instance_name(box.name))
        print(f"  auto: name: {box.name} (--unique — independent instance)")
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
            sched_obj.dynamic_directive(k, v) for k, v in _dynamic_for(dynamic, location.scheduler))
        ignored = _dynamic_ignored(dynamic, location.scheduler)
        if ignored:
            print(f"warning: ignoring {' '.join(ignored)} (active scheduler is {location.scheduler})",
                  file=sys.stderr)
    else:
        ignored = [f"--{kind}" for kind, value in site_flags if value] + raw_args
        ignored += [f"--{k}" if s == "sched" else f"--{s}-{k}" for s, k, v in dynamic]
        if ignored:
            print(f"warning: ignoring {' '.join(ignored)} — no scheduler in play "
                  f"(scheduler is 'none'; add --scheduler slurm|flux)", file=sys.stderr)
    from boxy import distributed as _dist

    dist_flag = getattr(args, "distributed", None)
    if dist_flag is None:  # fall back to the profile's [location.resources] distributed
        dist_flag = location.resources.distributed
    if _dist.is_distributed(box.engine, location.resources.nodes, dist_flag):
        return _serve_distributed(args, box, location)
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
        # THIS node is where the server runs, so it is the ideal prober: poll
        # localhost:port/health (no proxy, no cross-node routing) and flip the
        # shared-FS endpoint to ready — the submitting boxy then reports READY
        # without ever probing this compute node over the network.
        _start_local_health_watch(port, args.endpoint_file, model_hint=args.model or "")

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
        _dump_logs(runtime_bin, cname)
        _diagnose_container(runtime_bin, cname)
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
    rc = _delegate_remote(args)
    if rc is not None:
        return rc
    _apply_proxy_env(args)  # --proxy: reach HF through the corporate proxy
    _apply_bind_host_env(args)  # --bind-host wins over env/file/default
    model = args.model
    if not model and args.box:
        box = Box.from_toml(args.box)
        if not box.model:
            print(f"box {box.name}: no model set", file=sys.stderr)
            return 1
        model = box.model
    if not model:
        raise UsageError("usage: boxy pull MODEL   (or: boxy pull --box box.toml)")
    if model.startswith("s3://"):
        return cmd_stage(args)  # s3:// is staged, not RamaLama-pulled
    if not model.startswith(TRANSPORT_SCHEMES):
        print(f"model is a path ({model}); nothing to pull (shared-FS flow)")
        return 0
    path = ramalama_shim.pull_model(model, dryrun=args.dryrun, force=getattr(args, "force", False))
    print(f"model available at: {path}")
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    """Stage a model from a site-local S3 bucket to the shared filesystem, then
    serve it by path. Reads the same env a K8s vLLM deployment uses
    (S3_ENDPOINT_URL / S3_BUCKET_NAME / S3_PATH / AWS_*)."""
    from boxy import s3

    model = getattr(args, "model", None)
    models_dir = getattr(args, "models_dir", None) or "./models"
    endpoint = getattr(args, "s3_endpoint", None)
    if not model and getattr(args, "box", None):
        box = Box.from_toml(args.box)
        model = box.model
    if not model:
        # bare `boxy stage`: fall back entirely to the K8s-style env (bucket+path)
        if os.environ.get("S3_BUCKET_NAME"):
            model = "s3://"
        else:
            raise UsageError("usage: boxy stage s3://BUCKET/PREFIX   "
                             "(or set S3_BUCKET_NAME + S3_PATH, or use --box)")
    if not model.startswith("s3://"):
        print(f"stage only handles s3:// models; {model!r} is served directly (see boxy pull/serve)",
              file=sys.stderr)
        return 2
    runtime = getattr(args, "runtime", None) or next(
        (r for r in ("podman", "docker", "apptainer") if shutil.which(r)), "podman")
    no_sign = True if getattr(args, "no_sign_request", False) else None  # None => env decides
    path = s3.stage_model(model, models_dir, endpoint=endpoint, dryrun=getattr(args, "dryrun", False),
                          runtime=runtime, backend=getattr(args, "s3_backend", "") or "", no_sign=no_sign)
    print(f"model staged at: {path}\n  serve it:  boxy serve {path} [--scheduler slurm|flux --gpus N]")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
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


def cmd_generate_card(args: argparse.Namespace) -> int:
    """`boxy generate card <hf-model-id>`: fetch the model from HuggingFace, derive
    a serving card and write it where boxy reads cards from (with an overwrite
    guard). --dry-run prints without writing; --force replaces (keeping a .bak)."""
    import difflib

    from boxy import cardgen, cards

    repo = getattr(args, "model_id", None)
    if not repo:
        print("usage: boxy generate card <hf-model-id>   (e.g. meta-llama/Llama-3.1-8B-Instruct)",
              file=sys.stderr)
        return 2
    try:
        text, engine, warnings = cardgen.generate(
            repo, engine=(args.engine or ""), max_model_len=getattr(args, "max_model_len", None),
            token=getattr(args, "hf_token", None))
    except cardgen.CardGenError as e:
        print(f"boxy generate card: {e}", file=sys.stderr)
        return 1
    repo = cards.model_key(repo.strip().lstrip("/"))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    dest = cards._user_dir() / f"{cardgen.slug(repo)}.toml"
    # a USER card that already serves this model blocks an accidental overwrite —
    # the target file, or a user card under a different filename whose glob covers
    # the id. A PACKAGED match is only shadowed (never overwritten), so it's an
    # informational note, not a prompt.
    match = cards.find_card(repo)
    user_match = match if (match and match.source == "user") else None
    existing_path = dest if dest.exists() else None
    if user_match and existing_path is None:
        cand = cards._user_dir() / f"{user_match.card_name}.toml"
        existing_path = cand if cand.exists() else None
    prior = existing_path.read_text() if existing_path else ""
    overwriting = bool(existing_path or user_match)

    def _show_diff() -> None:
        diff = difflib.unified_diff(prior.splitlines(), text.splitlines(),
                                    fromfile=str(existing_path or "existing"), tofile=str(dest),
                                    lineterm="")
        print("\n".join(diff), file=sys.stderr)

    if args.dryrun:
        print(text, end="")
        if overwriting:
            where = str(existing_path) if existing_path else f"a user card matching {user_match.match!r}"
            print(f"\n# NOTE: would overwrite {where} (rerun without --dry-run to write; "
                  f"--force to skip the prompt).", file=sys.stderr)
            if prior:
                _show_diff()
        elif match:  # a packaged card exists; the new user card takes precedence
            print(f"\n# NOTE: shadows the packaged card matching {match.match!r} "
                  f"(user cards win); would write {dest}.", file=sys.stderr)
        else:
            print(f"\n# would write {dest}", file=sys.stderr)
        return 0

    if overwriting:
        target_desc = str(existing_path) if existing_path else f"user card matching {user_match.match!r}"
        if args.force:
            if existing_path:
                bak = existing_path.with_suffix(".toml.bak")
                existing_path.replace(bak)
                print(f"boxy: replaced {existing_path} (backup: {bak})")
        elif sys.stdin.isatty() and sys.stdout.isatty():
            print(f"A card already serves {repo}: {target_desc}", file=sys.stderr)
            if prior:
                _show_diff()
            try:
                ans = input(f"Overwrite {dest.name}? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans not in ("y", "yes"):
                print("boxy: kept the existing card (no change).", file=sys.stderr)
                return 0
        else:
            print(f"boxy generate card: a card already serves {repo} ({target_desc}); "
                  f"pass --force to replace it.", file=sys.stderr)
            return 1

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"### wrote {dest}  (engine: {engine})")
    print(f"###   serve it:  boxy serve {repo}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from boxy import sky_export

    if args.format == "card":
        return cmd_generate_card(args)
    if args.format == "flux-mcp":
        return _generate_flux_mcp(args)
    if args.format == "relay":
        return _generate_relay(args)
    if not args.box or not args.location:
        print(f"boxy generate {args.format}: --box and --location are required", file=sys.stderr)
        return 2
    if args.format in ("slurm", "flux", "sbatch"):
        return _generate_agentless(args)
    if args.format != "sky":
        print(f"boxy generate: unknown format {args.format!r} (available: sky, slurm, flux, flux-mcp)",
              file=sys.stderr)
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
    # --proxy = "this task runs on-net behind the corporate proxy": carry the proxy
    # env AND the merged trust bundle onto the task (cloud is explicit-opt-in,
    # unlike the auto-propagating on-net scheduler paths — see sky_export).
    ca = ramalama_shim.ensure_trust_bundle() if args.proxy else None
    yaml_text = sky_export.to_sky_task(box, location, port=args.port, serve=args.serve,
                                       proxy=args.proxy, ca_bundle=ca)
    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_text)
        print(f"wrote {args.output}  (launch: sky {'serve up' if args.serve else 'launch'} {args.output})")
    else:
        print(yaml_text, end="")
    return 0


def _generate_relay(args: argparse.Namespace) -> int:
    """Emit the boxy relay (chisel server) for OpenShift — the everyone-URL
    ingress behind `boxy open --share`. Deployed once per cluster; per-share
    Routes are created at share time by the relay exposer."""
    from boxy.exposers import relay

    if not args.host:
        # zero-flag: derive the Route host from the LOGGED-IN cluster —
        # <relay>.<apps domain> where the apps domain comes from the cluster's
        # ingress config (or api.->apps. off `oc whoami --show-server`).
        dom, why = relay.discover_apps_domain()
        if not dom:
            print(f"boxy generate relay: could not discover the cluster's apps domain ({why}) — "
                  "pass --host <name>.apps.<cluster>.<org>.<tld>", file=sys.stderr)
            return 2
        args.host = f"{relay.RELAY_APP}.{dom}"
        # stderr so stdout stays pure YAML for `| oc apply -f -`
        print(f"  auto: relay host: {args.host} (via {why})", file=sys.stderr)
    text = relay.emit_relay_manifest(args.host, args.namespace or relay.DEFAULT_NAMESPACE,
                                     image=args.image or config.get("images.relay"),
                                     auth=args.auth, key_seed=args.key_seed)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def _generate_flux_mcp(args: argparse.Namespace) -> int:
    """Emit the flux-mcp MCP server as a persistent OpenShift service."""
    from boxy import mcp

    if not args.host:
        print("boxy generate flux-mcp: --host is required (the OpenShift Route hostname, "
              "e.g. flux-mcp.apps.<cluster>)", file=sys.stderr)
        return 2
    text = mcp.emit_flux_mcp_manifest(args.host, args.namespace or "flux-mcp",
                                      image=args.image or config.get("images.flux_mcp"),
                                      flux_uri=args.flux_uri, port=config.get_int("mcp.flux_port"))
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def _agentless_script(box: Box, location: Location, scheduler_name: str, name: str,
                      args, endpoint_file: str | None = None) -> str:
    """Render the self-contained (no-boxy-on-cluster) batch script for box+location,
    reusing the account/partition/time site flags and the --proxy prefix. Raises
    deploy.AgentlessError on the agentless boundaries (unstaged model / unpinned
    hardware)."""
    from dataclasses import replace as dc_replace

    from boxy import deploy, jobs
    from boxy.schedulers import get_scheduler

    if getattr(args, "accelerator", None):
        location = dc_replace(location, accelerator=args.accelerator)
    if getattr(args, "image", None):
        box = dc_replace(box, image=args.image)
    scheduler = get_scheduler(scheduler_name)
    site_args = [scheduler.site_directive(k, v) for k, v in
                 (("partition", getattr(args, "partition", None)),
                  ("account", getattr(args, "account", None)),
                  ("time", getattr(args, "time", None))) if v]
    log_file = str(jobs.log_path(name, scheduler.output_token) if scheduler.output_token
                   else jobs.log_path(name))
    ep = endpoint_file or str(jobs.endpoint_path(name))
    return deploy.render_agentless_script(box, location, scheduler_name, name, ep, log_file,
                                          site_args, proxy_prefix=_proxy_prefix(args), port=args.port)


def _generate_agentless(args: argparse.Namespace) -> int:
    from boxy import deploy

    box, location = _load(args)
    scheduler_name = args.format if args.format in ("slurm", "flux") else location.scheduler
    if scheduler_name not in ("slurm", "flux"):
        print(f"boxy generate {args.format}: the location has scheduler={location.scheduler!r}; "
              "pass `generate slurm` or `generate flux` explicitly", file=sys.stderr)
        return 1
    try:
        script = _agentless_script(box, location, scheduler_name, box.name, args)
    except deploy.AgentlessError as e:
        print(f"boxy generate: {e}", file=sys.stderr)
        return 1
    if args.output:
        with open(args.output, "w") as f:
            f.write(script)
        print(f"wrote {args.output}  (submit on the cluster: "
              f"{'sbatch' if scheduler_name == 'slurm' else 'flux batch'} {args.output})")
        print("### zero-install: this script needs only a scheduler + podman + a shared FS — no boxy")
    else:
        print(script, end="")
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


def cmd_unshare(args: argparse.Namespace) -> int:
    """Tear down an everyone-URL share: kill the detached relay client and
    delete the per-alias Route/Service on the cluster (local state always
    cleared, even when oc is unreachable — the manual command is printed)."""
    from boxy import jobs
    from boxy.exposers import get_exposer

    record = jobs.read_share(args.alias)
    if not record:
        print(f"boxy: no share named {args.alias!r} (see `boxy list`)", file=sys.stderr)
        return 1
    get_exposer(getattr(args, "exposer", None) or "relay").unexpose(args.alias)
    print(f"unshared {args.alias}  ({record['url']} is gone)")
    return 0


def _agentless_records() -> list[dict]:
    """This machine's records of AGENTLESS --ssh serves: the job runs on the
    cluster, but the ONLY state lives here (the cluster has no boxy state — that's
    the point). list/logs/stop must answer from these records over the SSH master,
    never by delegating to a cluster boxy that knows nothing (field: `boxy list
    --ssh hops` ran podman ps on the LOGIN node and showed nothing while the
    container ran on the compute node)."""
    from boxy import jobs

    return [r for r in jobs.list_records()
            if r.get("submitted_from") == "agentless-ssh" and r.get("target")]


def _agentless_state(record: dict) -> str:
    from boxy import remote
    from boxy.schedulers import get_scheduler

    scheduler = get_scheduler(record["scheduler"])
    target = record["target"]
    if remote.ensure_master(target) != 0:
        return "UNREACHABLE"
    rc, out = remote.ssh_capture(target, shlex.join(scheduler.state_command(record["job"])), timeout=20)
    return scheduler.interpret_state(out) if rc == 0 else "UNKNOWN"


def _agentless_url(record: dict) -> str:
    import json as _json

    from boxy import remote

    ep = record.get("endpoint_remote", "")
    if not ep:
        return "-"
    rc, out = remote.ssh_capture(record["target"], f"cat {shlex.quote(ep)} 2>/dev/null || true",
                                 timeout=15)
    if rc == 0 and out.strip():
        try:
            d = _json.loads(out)
            if d.get("url"):
                return f"{d['url']}/v1"
        except ValueError:
            pass
    return "-"


def _list_agentless(records: list[dict]) -> None:
    print("agentless jobs (state probed over SSH; nothing installed on the cluster):")
    for r in records:
        state = _agentless_state(r)
        url = _agentless_url(r) if state == "RUNNING" else "-"
        print(f"  {r['name']}  {r['scheduler']} job {r['job']} on {r['target']}  {state}  {url}")
        print(f"      logs: boxy logs {r['name']}      stop: boxy stop {r['name']}")


def _log_token_glob(path: str) -> str:
    """A remote log path with the scheduler's output token (%j / {{id}}) turned
    into a glob, so `ls -t` finds the actual per-job file(s)."""
    for tok in ("%j", "{{id}}"):
        path = path.replace(tok, "*")
    return path


def _agentless_log_glob(record: dict) -> str:
    return _log_token_glob(record.get("log", ""))


def _remote_log_tail(target: str, log_path: str, n: int = 60) -> str:
    """Tail the NEWEST file matching a (token-bearing) remote log path over the
    master. `tail`ing the literal %j path reads nothing (field: the death path
    printed an empty log because Slurm had substituted the job id)."""
    from boxy import remote

    pat = _log_token_glob(log_path)
    _, out = remote.ssh_capture(
        target, f'tail -n {int(n)} "$(ls -t {pat} 2>/dev/null | head -1)" 2>/dev/null || true',
        timeout=20)
    return out


def _agentless_logs(args, record: dict) -> int:
    """Tail an agentless job's log from the CLUSTER's shared FS over the master.
    The newest per-job file matching the record's log pattern wins (reruns write
    <name>-<jobid>.log siblings)."""
    from boxy import diagnostics, remote

    target = record["target"]
    if remote.ensure_master(target) != 0:
        print(f"boxy: could not open an SSH session to {target}", file=sys.stderr)
        return 1
    pat = _agentless_log_glob(record)
    cmd = f'tail -n {int(args.tail)} "$(ls -t {pat} 2>/dev/null | head -1)" 2>/dev/null || true'
    rc, out = remote.ssh_capture(target, cmd, timeout=30)
    if rc != 0 or not out.strip():
        raise UsageError(f"no log yet for {record['name']!r} on {target} (pattern {pat}) — "
                         f"the job may still be pending; check: boxy list")
    lines = out.splitlines()
    print(f"### {target}:{pat}  (last {len(lines)} lines)")
    for line in lines:
        print(f"    {line}")
    hint = diagnostics.diagnose(out)
    if hint:
        print()
        print(hint)
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    """Re-attach to a serve that DETACHED (e.g. the model was still loading when
    the readiness window closed): read the endpoint from the cluster's shared FS,
    open the tunnel, confirm readiness, and print the READY url — exactly what the
    attached serve would have printed. Agentless --ssh serves (records on this
    machine) are supported; the job itself is untouched."""
    import json as _json

    from boxy import jobs, remote

    name = args.name
    rec = jobs.read_record(name) if name else None
    if rec is None:
        ags = _agentless_records()
        if name:
            ags = [r for r in ags if r["name"] == name]
        if len(ags) == 1:
            rec = ags[0]
            name = rec["name"]
        elif len(ags) > 1:
            raise UsageError("several agentless jobs — pick one: boxy attach "
                             + " | ".join(r["name"] for r in ags))
    if not (rec and rec.get("submitted_from") == "agentless-ssh" and rec.get("target")):
        raise UsageError("nothing to attach to — `boxy attach` re-joins an agentless --ssh serve "
                         "recorded on this machine (see boxy list)")
    target = rec["target"]
    if remote.ensure_master(target) != 0:
        print(f"boxy: could not open an SSH session to {target}", file=sys.stderr)
        return 1
    rc, epj = remote.ssh_capture(
        target, f"cat {shlex.quote(rec.get('endpoint_remote', ''))} 2>/dev/null || true", timeout=15)
    ep = None
    if rc == 0 and epj.strip():
        try:
            ep = _json.loads(epj)
        except ValueError:
            ep = None
    if not (ep and ep.get("host") and ep.get("port")):
        raise UsageError(f"{name}: no endpoint on {target} yet — the job may still be "
                         f"pending (state: {_agentless_state(rec)}); see boxy list")
    # resolve the newest per-job log for the readiness grep fallback
    _, lp = remote.ssh_capture(
        target, f"ls -t {_agentless_log_glob(rec)} 2>/dev/null | head -1", timeout=15)
    print(f"###   attaching to {name} on {ep['host']} (job {rec['job']}) — confirming readiness "
          f"through the tunnel (stays attached while the job is alive)")

    def _job_alive() -> bool:
        return _agentless_state(rec) in ("RUNNING", "PENDING")

    ok = remote.await_ready_and_tunnel(
        target, ep["host"], int(ep["port"]), lp.strip(),
        getattr(args, "local_port", None), "", getattr(args, "share", "") or "",
        getattr(args, "exposer", None) or "relay", False,
        timeout_s=(args.ready_timeout if getattr(args, "ready_timeout", 0) > 0 else _SCHED_READY_FLOOR),
        still_alive=_job_alive)
    if ok:
        print(f"###   stop: boxy stop {name}")
        return 0
    print(f"boxy: job {rec['job']} ended before the server became ready; see: boxy logs {name}",
          file=sys.stderr)
    return 1


def _agentless_stop(args, record: dict, name: str) -> int:
    from boxy import jobs, remote
    from boxy.schedulers import get_scheduler

    scheduler = get_scheduler(record["scheduler"])
    target = record["target"]
    cancel = shlex.join(scheduler.cancel_command(record["job"]))
    print(f"### Remote {target}  $ {cancel}")
    if args.dryrun:
        return 0
    if remote.ensure_master(target) != 0:
        print(f"boxy: could not reach {target} to cancel job {record['job']}", file=sys.stderr)
        return 1
    rc, out = remote.ssh_capture(target, cancel, timeout=30)
    if rc == 0:
        jobs.remove(name)
        print(f"### stopped {record['scheduler']} job {record['job']} on {target}")
    else:
        print(f"boxy: cancel failed on {target}:\n{out.strip()}", file=sys.stderr)
    return rc


def cmd_stop(args: argparse.Namespace) -> int:
    from boxy import jobs as _jobs

    # an AGENTLESS record is handled from HERE over the master — even with --ssh
    # (boxy's own output says `boxy stop NAME --ssh <target>`): the cluster's boxy
    # (if any) has no record of this job and would no-op/fail.
    nm = args.name or (Box.from_toml(args.box).name if args.box else None)
    if nm:
        rec = _jobs.read_record(nm)
        if rec and rec.get("submitted_from") == "agentless-ssh" and rec.get("target"):
            return _agentless_stop(args, rec, nm)

    rc = _delegate_remote(args)
    if rc is not None:
        return rc
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
        # server(s), so the container(s) die with it). A replica group job hosts
        # several servers on one node — cancelling reaps them all.
        scheduler = get_scheduler(record["scheduler"])
        rc = _run_or_print(scheduler.cancel_command(record["job"]), args.dryrun)
        if not args.dryrun:
            for replica in record.get("replicas", []):
                jobs.endpoint_path(replica).unlink(missing_ok=True)
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
    from boxy import remote as _remote

    # AGENTLESS records live on THIS machine; with --ssh to their cluster, answer
    # from here over the master instead of delegating to a cluster boxy that has
    # no records (its login-node `podman ps` shows nothing — the container runs
    # on the COMPUTE node).
    agentless = _agentless_records()
    tgt = _remote.resolve_target(args) or ""
    if tgt:
        thost = tgt.split("@")[-1]
        mine = [r for r in agentless if r["target"].split("@")[-1] == thost]
        if mine:
            _list_agentless(mine)
            return 0

    rc = _delegate_remote(args)
    if rc is not None:
        return rc
    from boxy import jobs
    from boxy.schedulers import get_scheduler

    if agentless:
        _list_agentless(agentless)
    records = [r for r in jobs.list_records() if r not in agentless]
    foreign_seen = False
    if records:
        print("scheduler jobs:")
        for record in records:
            scheduler_obj = get_scheduler(record["scheduler"])
            is_foreign, origin = _record_is_foreign(record)
            if not is_foreign:
                state = _job_state(scheduler_obj, record["job"])
            else:
                # Labs share $HOME across clusters, so this jobs dir holds OTHER
                # clusters' records too (field report: an clusterA flux job listed
                # on clusterB as UNKNOWN). No point probing — the job lives elsewhere;
                # say where it IS instead of UNKNOWN.
                state = f"FOREIGN({origin})"
                foreign_seen = True
            replicas = record.get("replicas")
            if replicas:
                # a replica group job: one job, several co-located servers
                print(f"  {record['name']}  {record['scheduler']} job {record['job']}  {state}  "
                      f"({len(replicas)} replicas)")
                for rn in replicas:
                    ep = jobs.read_endpoint(rn)
                    print(f"      {rn}  {ep['url'] + '/v1' if ep else '-'}")
            else:
                endpoint = jobs.read_endpoint(record["name"])
                url = f"{endpoint['url']}/v1" if endpoint else "-"
                print(f"  {record['name']}  {record['scheduler']} job {record['job']}  {state}  {url}")
            if state == "DONE" and not args.dryrun:
                for rn in replicas or []:
                    jobs.endpoint_path(rn).unlink(missing_ok=True)
                jobs.remove(record["name"])  # reap finished jobs from the list
        if foreign_seen:
            print("  (FOREIGN = submitted on another cluster; manage it there, e.g. "
                  "boxy list --ssh <that-login>. boxy separates clusters automatically now "
                  "(<jobs-root>/<cluster>/); FOREIGN only appears when BOXY_JOBS_DIR pins one "
                  "shared dir, or from legacy records.)")
    shares = jobs.list_shares()
    if shares:
        from boxy.exposers import relay as relay_exposer
        print("shares (everyone-URLs via the OpenShift relay):")
        for s in shares:
            state = ("LIVE" if relay_exposer.share_is_live(s)
                     else "DEAD (relay client gone — rerun with --share, or boxy unshare)")
            print(f"  {s['alias']}  {s['url']}/v1  {state}")
    location = Location.from_toml(args.location) if args.location else None
    try:
        runtime = args.runtime or _container_runtime(location)
    except RuntimeError:
        if records or agentless:
            return 0  # jobs listed; no container runtime on this host is fine
        raise
    rc = _list_local_containers(runtime, args.dryrun, have_records=bool(records or agentless))
    if not args.dryrun:
        _report_exited_containers(runtime)
    return rc


def _list_local_containers(runtime: str, dryrun: bool, have_records: bool) -> int:
    """List boxy's LOCAL containers, quietly. On an HPC login node rootless podman
    has no /run/user/$UID (no user systemd session), so `podman ps` fails with
    'Failed to get rootless runtime dir' + 'creating events dirs: permission
    denied' noise — but the real instances run on the compute nodes (already
    listed as scheduler jobs). Capture podman's output so that noise never reaches
    the user: print the table on success; when it fails and jobs were listed,
    skip silently; only surface the raw error when there's nothing else to show."""
    print(f"### Running Command:\n    {shlex.join([runtime, 'ps', '--filter', 'label=boxy.box'])}")
    if dryrun:
        return 0
    proc = subprocess.run([runtime, "ps", "--filter", "label=boxy.box"],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        if proc.stdout:
            print(proc.stdout, end="")
        return 0
    if have_records:
        # login node with no local runtime: the jobs above ARE the answer.
        print(f"  (no local {runtime} containers on this host — instances run on the "
              f"compute nodes listed above)")
        return 0
    sys.stderr.write(proc.stderr)                       # nothing else to show — be honest
    return proc.returncode


def _report_exited_containers(runtime: str) -> None:
    """Surface boxy containers that EXITED — plain `ps` shows only RUNNING ones,
    so a server that died seconds after READY (crash, or an OOM kill by the
    podman/docker VM) silently vanishes from view. List them with exit code and
    a targeted note (field report: `--unique` instances kept 'disappearing')."""
    result = subprocess.run(
        [runtime, "ps", "-a", "--filter", "label=boxy.box", "--filter", "status=exited",
         "--format", "{{.Names}}\t{{.Status}}"],
        capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return
    print("exited boxy containers (crashed or were killed — NOT running):")
    oom_seen = False
    for line in result.stdout.strip().splitlines():
        name, _, status = line.partition("\t")
        fields = subprocess.run(
            [runtime, "inspect", "--format", "{{.State.ExitCode}} {{.State.OOMKilled}}", name],
            capture_output=True, text=True).stdout.split()
        exit_code = fields[0] if fields else "?"
        oom = len(fields) > 1 and fields[1].lower() == "true"
        note = ""
        if exit_code == "137" or oom:
            note = "  <- OOM/SIGKILL: the runtime VM ran out of RAM"
            oom_seen = True
        print(f"  {name}  ({status}, exit {exit_code}){note}")
    if oom_seen:
        print("  fix OOM: podman machine stop && podman machine set --memory 8192 --cpus 4 "
              "&& podman machine start")
    print(f"  why did one die:  {runtime} logs <name>    "
          f"clear them:  {runtime} rm $({runtime} ps -aq --filter label=boxy.box --filter status=exited)")


def cmd_router(args: argparse.Namespace) -> int:
    """Front a replica set (<base>-r*) with one OpenAI URL. Default: run the
    built-in load-balancing proxy on the login node. --emit prints a config for a
    production proxy (nginx/haproxy/litellm) instead of running anything."""
    from boxy import jobs, router

    endpoints = jobs.list_endpoints(args.base)
    if args.emit:
        if not endpoints:
            raise UsageError(f"no endpoint files for {args.base}-r* in the jobs dir — is the replica "
                             f"set up? (check `boxy list`; base is the name before -r0/-r1/…)")
        emitter = {"nginx": router.emit_nginx, "haproxy": router.emit_haproxy,
                   "litellm": router.emit_litellm}[args.emit]
        print(emitter(args.base, endpoints, args.port) if args.emit != "litellm"
              else emitter(args.base, endpoints))
        return 0
    if args.dryrun:
        print(f"### Router plan: base={args.base}  listen=:{args.port}  policy={args.policy}  "
              f"discovered={len(endpoints)} replica(s) "
              f"({', '.join(e['name'] for e in endpoints) or 'none yet'})")
        return 0
    pool = router.Pool(policy=args.policy)
    disc = router.DiscoveryThread(args.base, pool, interval=args.refresh)
    disc.scan_once()  # populate before serving (avoid a first-hit 503)
    disc.start()
    host = socket.gethostname()
    srv = router.make_server(pool, args.port)
    print(f"### Router  http://{host}:{args.port}/v1  -> {args.base}-r* "
          f"({len(pool.snapshot())} replica(s), {args.policy})")
    print(f"###   from your workstation: ssh -L {args.port}:{host}:{args.port} <login-node>")
    print("###   Ctrl-C stops the router; the replicas keep running")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n### Router stopped.")
    finally:
        srv.shutdown()
        disc.stop()
    return 0


def _scheduler_reachable(scheduler_obj) -> bool:
    """Can THIS host speak that scheduler? (its state binary is on PATH). Only a
    FALLBACK for legacy records without an origin — clusters often ship other
    schedulers' binaries too (field report: clusterB has `flux` on PATH, so an
    clusterA flux record passed this check and boxy curl chased eldo1025)."""
    return shutil.which(scheduler_obj.state_command("x")[0]) is not None


def _cluster_id(host: str) -> str:
    """Cluster identity from a hostname (one source of truth: jobs.cluster_id,
    which also drives the per-cluster jobs dir)."""
    from boxy import jobs

    return jobs.cluster_id(host)


def _local_cluster() -> str:
    from boxy import jobs

    return jobs.local_cluster()


def _record_is_foreign(record: dict) -> tuple[bool, str]:
    """Does this record belong to ANOTHER cluster sharing this $HOME? The seam
    for local-vs-FOREIGN classification — one place, test-injectable. Primary
    signal: the record's submit-host cluster identity vs this host's (scheduler-
    binary presence is unreliable — see _scheduler_reachable). Returns
    (foreign, origin-host)."""
    from boxy.schedulers import get_scheduler

    origin = record.get("submitted_from", "")
    if origin:
        return _cluster_id(origin) != _local_cluster(), origin
    return not _scheduler_reachable(get_scheduler(record["scheduler"])), "another cluster"


def cmd_logs(args: argparse.Namespace) -> int:
    """Show a job's log (newest first) + boxy's crash diagnosis. Works after the
    record is reaped (log files outlive DONE jobs) and over --ssh. An AGENTLESS
    job's log lives on the CLUSTER's shared FS — fetched from here over the
    master (the cluster's own boxy, if any, has no record of it)."""
    from boxy import jobs as _jobs

    ag_rec = None
    if args.name:
        r = _jobs.read_record(args.name)
        if r and r.get("submitted_from") == "agentless-ssh" and r.get("target"):
            ag_rec = r
    else:
        ags = _agentless_records()
        if ags:
            ag_rec = ags[-1]  # newest record wins when unnamed
    if ag_rec and ag_rec.get("log"):
        return _agentless_logs(args, ag_rec)

    rc = _delegate_remote(args)
    if rc is not None:
        return rc
    from boxy import diagnostics, jobs

    d = jobs._dir()
    pattern = f"{args.name}*.log" if args.name else "*.log"
    cands = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        have = sorted({p.name for p in d.glob('*.log')})
        # this cluster's dir is now separate (d = <root>/<cluster>/); if the OLD
        # flat root still holds unattributable logs, point at them rather than
        # silently mixing another cluster's in.
        legacy = ""
        if d.parent != d and not os.environ.get("BOXY_JOBS_DIR"):
            orphans = sorted(p.name for p in d.parent.glob("*.log"))
            if orphans:
                legacy = (f"\n  {len(orphans)} pre-separation log(s) remain in the shared root "
                          f"{d.parent} (not attributed to a cluster); inspect directly if needed.")
        raise UsageError(f"no logs matching {pattern!r} in {d}"
                         + (f" — available: {', '.join(have)}" if have else " (no job logs yet)")
                         + legacy)
    path = cands[0]
    if not args.name and len(cands) > 1:
        print(f"### newest of {len(cands)} logs (pass a NAME for a specific job):")
    text = path.read_text(errors="replace")
    lines = text.splitlines()[-args.tail:]
    print(f"### {path}  (last {len(lines)} lines)")
    for line in lines:
        print(f"    {line}")
    hint = diagnostics.diagnose(text)
    if hint:
        print()
        print(hint)
    return 0


def _select_endpoint(name: str | None, verb: str = "curl") -> tuple[str, dict]:
    """Foreign-aware endpoint selection shared by curl/open. Returns (name, ep).
    Labs share $HOME across clusters, so the jobs dir holds OTHER clusters'
    endpoints too (their node hostnames don't resolve here); those are excluded
    and, if named, pointed at their own cluster. Raises UsageError on
    ambiguity / missing / foreign."""
    from boxy import jobs

    endpoints: dict[str, dict] = {}
    foreign: dict[str, str] = {}  # name -> submitted_from
    for r in jobs.list_records():
        is_foreign, origin = _record_is_foreign(r)
        for n in [r["name"], *r.get("replicas", [])]:
            ep = jobs.read_endpoint(n)
            if ep and is_foreign:
                foreign[n] = origin
            elif ep:
                endpoints[n] = ep
    if name:
        if name in foreign:
            raise UsageError(f"{name} runs on another cluster (submitted from {foreign[name]}) — "
                             f"use it from there: boxy {verb} {name} --ssh <that login node>")
        ep = endpoints.get(name)
        if not ep:
            raise UsageError(f"no endpoint for {name!r} — running here: "
                             f"{', '.join(sorted(endpoints)) or 'none'} (see boxy list)")
        return name, ep
    if len(endpoints) == 1:
        return next(iter(endpoints.items()))
    if not endpoints:
        hint = (f" ({len(foreign)} foreign: {', '.join(sorted(foreign))} — use --ssh on their own "
                f"cluster)" if foreign else "")
        raise UsageError(f"nothing is serving on THIS cluster{hint} — see boxy list")
    raise UsageError(f"several models are serving — pick one: boxy {verb} {' | '.join(sorted(endpoints))}")


def cmd_curl(args: argparse.Namespace) -> int:
    """Query a boxy-served model by NAME from wherever you are: resolve its
    endpoint from the job records, send one chat completion, print the reply.
    With --ssh (or BOXY_SSH_HOST) it runs ON the cluster, where the compute-node
    hostname resolves — so `boxy curl --ssh user@login` works from a laptop."""
    rc = _delegate_remote(args)
    if rc is not None:
        return rc
    import urllib.error

    from boxy import bench

    if args.url:
        url = args.url.rstrip("/").removesuffix("/v1")
    else:
        _name, ep = _select_endpoint(args.name, "curl")
        url = ep["url"]
    try:
        model = bench.discover_model(url)
        body = bench._http_json(f"{url}/v1/chat/completions", {
            "model": model, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": args.prompt}]})
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"cannot reach {url} ({getattr(e, 'reason', e)}) — is the job READY? "
                           f"(boxy list). From a laptop, add --ssh user@login to query "
                           f"from the cluster side.") from e
    if args.json:
        import json as _json

        print(_json.dumps(body, indent=1))
        return 0
    reply = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"[{model} @ {url}]")
    print(reply.strip() or "(empty reply)")
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    """Open a served model in your browser. With --ssh it forwards the compute
    node's endpoint back to a FREE local port over the one SSH session (no
    re-auth) and prints the local URL — llama.cpp serves a web chat UI at the
    root path. Run ON the cluster (no --ssh) it prints the endpoint + the exact
    `ssh -L` a workstation needs. One step for the browser access that used to
    take a manual tunnel (field report, 2026-07)."""
    rc = _delegate_remote(args, tunnel_ready=True)
    if rc is not None:
        return rc
    if getattr(args, "share", None):
        raise UsageError("--share needs the laptop tunnel (`boxy open NAME --ssh user@login "
                         "--share NAME`) — it publishes the LOCAL end of the forward; see "
                         "RUNBOOK §0.993 for the login-node bridge")
    name, ep = _select_endpoint(args.name, "open")
    host, port = ep["host"], ep["port"]
    # The '### READY http://host:port' banner is what the laptop side watches
    # for: when delegated via --ssh, run_remote(tunnel_ready=True) forwards this
    # endpoint to a local port and prints the browsable LOCAL url. Run here on the
    # cluster it is informational — a login shell has no browser, so hand the user
    # the ssh -L their workstation needs (honoring --port for a stable local URL).
    lport = getattr(args, "local_port", None) or port
    print(f"### READY  http://{host}:{port}/v1   (model: {name})")
    print(f"###   browser (llama.cpp web UI):  http://{host}:{port}/")
    route = getattr(args, "route", "") or ""
    if route:
        from boxy import remote
        rurl, rnote = remote.route_url(route, lport)
        print(f"###   from a workstation:  ssh -L {lport}:{host}:{port} <this login node>  "
              f"then open {rurl[:-2]}  (API base: {rurl})")
        if rnote:
            print(f"###   {rnote}")
    else:
        print(f"###   from a workstation:  ssh -L {lport}:{host}:{port} <this login node>  "
              f"then open http://127.0.0.1:{lport}/")
    return 0


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


def _sweep_axis(args) -> tuple[str, list[int]]:
    """Exactly one of --sweep-nodes / --sweep-replicas, parsed to a list of ints
    (powers of two by convention, but any positive ints are allowed)."""
    sn = getattr(args, "sweep_nodes", None)
    sr = getattr(args, "sweep_replicas", None)
    if bool(sn) == bool(sr):
        raise UsageError("boxy sweep needs exactly one of --sweep-nodes or --sweep-replicas "
                         "(a comma list, e.g. 1,2,4,8)")
    axis, raw = ("nodes", sn) if sn else ("replicas", sr)
    try:
        values = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise UsageError(f"--sweep-{axis} must be a comma list of integers, got {raw!r}") from None
    if not values or any(v < 1 for v in values):
        raise UsageError(f"--sweep-{axis} must be positive integers, got {raw!r}")
    return axis, values


def _parse_batch_sizes(args) -> list[int]:
    from boxy import bench

    if not getattr(args, "batch_sizes", None):
        return bench.DEFAULT_BATCH_SIZES
    try:
        return [int(b) for b in args.batch_sizes.split(",")]
    except ValueError:
        raise UsageError(f"--batch-sizes must be a comma list of integers, "
                         f"got {args.batch_sizes!r}") from None


def _rung_serve_args(args, nodes: int, replicas: int, name: str) -> argparse.Namespace:
    """A serve-shaped args namespace for one sweep rung, carrying everything
    _serve_submission/_serve_replicas read (geometry set for this rung)."""
    return argparse.Namespace(
        model=args.model, name=name, dryrun=args.dryrun, unique=False,
        replicas=replicas, gpus=args.gpus, nodes=nodes,
        scheduler_args=list(getattr(args, "scheduler_args", []) or []),
        partition=getattr(args, "partition", None), account=getattr(args, "account", None),
        time=getattr(args, "time", None), save_profile=None, distributed=None,
        ready_timeout=args.ready_timeout, engine=getattr(args, "engine", None),
        image=getattr(args, "image", None), runtime=getattr(args, "runtime", None),
        accelerator=getattr(args, "accelerator", None), port=None,
        location=getattr(args, "location", None), models_dir=getattr(args, "models_dir", None),
        args=[], dynamic_flags=[], foreground=False,
    )


def _sweep_wait_endpoints(names: list[str], timeout_s: float) -> list[str]:
    """Poll the shared-FS endpoint files until every rung server is READY (or
    timeout); return the ready URLs."""
    import time

    from boxy import jobs, readiness

    deadline = time.time() + timeout_s
    ready: dict[str, str] = {}
    print(f"###   waiting up to {timeout_s:.0f}s for {len(names)} endpoint(s) to become ready ...")
    while len(ready) < len(names) and time.time() < deadline:
        for n in names:
            if n in ready:
                continue
            ep = jobs.read_endpoint(n)
            if ep and readiness.wait_ready(ep["url"], timeout_s=3, interval_s=1):
                ready[n] = ep["url"]
                print(f"###   ready: {n} -> {ep['url']}/v1")
        if len(ready) < len(names):
            time.sleep(3)
    return [ready[n] for n in names if n in ready]


def _sweep_teardown(name: str) -> None:
    """Cancel a rung's scheduler job and drop its record."""
    from boxy import jobs
    from boxy.schedulers import get_scheduler

    rec = jobs.read_record(name)
    if not rec:
        return
    try:
        subprocess.run(get_scheduler(rec["scheduler"]).cancel_command(rec["job"]), capture_output=True)
    finally:
        jobs.remove(name)


def cmd_sweep(args: argparse.Namespace) -> int:
    """Scaling study: for each rung (a node or replica count), submit the config,
    wait until it's READY, benchmark it, tear it down, and finally print a scaling
    comparison table. This is the paper's scaling deliverable."""
    from boxy import bench, resolve

    axis, values = _sweep_axis(args)
    scheduler_name = args.scheduler
    profile = Location.from_toml(args.location) if args.location else None
    if scheduler_name is None and profile and profile.scheduler in ("slurm", "flux"):
        scheduler_name = profile.scheduler
    if scheduler_name not in ("slurm", "flux"):
        raise UsageError("boxy sweep needs --scheduler slurm|flux (each rung is a cluster job "
                         "that boxy submits, benchmarks, then tears down)")
    batch_sizes = _parse_batch_sizes(args)
    _model, base_name, _ = resolve.resolve_submission(
        args.model, scheduler_name, name=args.name, require_exists=not args.dryrun)

    print(f"### Scaling sweep: {axis} = {', '.join(map(str, values))}   "
          f"(batch sizes {batch_sizes}, max_tokens {args.max_tokens}, "
          f"{'keep' if args.keep else 'tear down'} each rung)")
    report = bench.ScalingReport(axis=axis, model="", max_tokens=args.max_tokens)
    for v in values:
        # rung tag: n<v> for nodes, x<v> for replica-count (x avoids clashing with
        # the per-replica -r0..-r{K-1} suffix the fan-out appends).
        rung_base = f"{base_name}-{'n' if axis == 'nodes' else 'x'}{v}"
        nodes = v if axis == "nodes" else (args.nodes or 1)
        reps = v if axis == "replicas" else 1
        print(f"\n## Rung {axis}={v}  ({nodes} node(s) x {args.gpus if args.gpus else '?'} GPU, "
              f"{reps} replica(s))")
        ra = _rung_serve_args(args, nodes=nodes, replicas=reps, name=rung_base)
        if reps > 1:
            _serve_replicas(ra, scheduler_name, profile, reps)
            names = [f"{rung_base}-r{i}" for i in range(reps)]
        else:
            _serve_submission(ra, scheduler_name, profile, name_override=rung_base, follow=False)
            names = [rung_base]
        if args.dryrun:
            print(f"##   then: bench {batch_sizes} across the rung endpoint(s), "
                  f"{'keep' if args.keep else 'tear down'}")
            continue
        urls = _sweep_wait_endpoints(names, args.ready_timeout)
        if not urls:
            print(f"warning: rung {axis}={v}: no endpoint ready within {args.ready_timeout:.0f}s; "
                  f"skipping", file=sys.stderr)
            if not args.keep:
                for n in names:
                    _sweep_teardown(n)
            continue
        rep = bench.run_scaling_point(urls, batch_sizes, max_tokens=args.max_tokens, dataset=args.dataset)
        report.model = report.model or rep.model
        pt = bench.summarize_point(f"{axis}={v}", axis, v, len(urls), rep)
        report.points.append(pt)
        print(f"##   {axis}={v}: peak {pt.tokens_per_s:.1f} tok/s @ batch {pt.peak_batch} "
              f"(p50 {pt.latency_p50_ms:.0f} ms) across {len(urls)} endpoint(s)")
        if not args.keep:
            for n in names:
                _sweep_teardown(n)
            print(f"##   torn down {axis}={v}")

    if args.dryrun:
        print(f"\n### dryrun: {len(values)} rungs planned; nothing submitted")
        return 0
    print("\n### Scaling results")
    print(report.to_table())
    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_csv())
        print(f"wrote {args.output}")
    if args.json:
        print(report.to_json())
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
        ca = ramalama_shim.ensure_trust_bundle() if args.proxy else None
        yaml_path = cloud.write_task_yaml(box, location, args.port, args.serve, output=args.output,
                                          proxy=args.proxy, ca_bundle=ca)
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
    parser.add_argument("--version", action="version", version=f"boxy {version_string()}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("info", help="show detected accelerator, runtimes, schedulers, TLS state")
    p.add_argument("--net", action="store_true",
                   help="also probe each model registry with the current trust store")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("config", help="show the effective configuration and where each "
                                      "value comes from (flag > env > file > default)")
    p.add_argument("--init", action="store_true",
                   help="print a starter config.toml (all keys, commented) to stdout")
    p.set_defaults(func=cmd_config, location=None)

    p = sub.add_parser("examples", help="list, show, or export the packaged example "
                                        "box & location profiles")
    ex = p.add_subparsers(dest="action")
    ex.add_parser("list", help="list packaged examples (default)")
    sh = ex.add_parser("show", help="print one example profile to stdout")
    sh.add_argument("name", help="example filename, e.g. vllm.toml or local-podman.toml")
    xp = ex.add_parser("export", help="copy the packaged examples into a directory")
    xp.add_argument("dest", nargs="?", default="./examples", help="destination dir (default ./examples)")
    p.set_defaults(func=cmd_examples, location=None, action=None)

    p = sub.add_parser("cards", help="list the built-in model & system deployment cards "
                                     "(turnkey: `boxy serve MODEL --scheduler slurm`)")
    p.set_defaults(func=cmd_cards, location=None)

    p = sub.add_parser("app", help="run an HPC application/benchmark from an APP CARD "
                                   "(spack build or container) — agentless over --ssh: "
                                   "`boxy app osu-benchmarks --ssh cluster`")
    p.add_argument("name", nargs="?", default=None,
                   help="app card name (omit to list the available cards)")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run on that cluster over one multiplexed SSH session (agentless: "
                        "the cluster needs only spack or podman, never boxy)")
    p.add_argument("--scheduler", choices=("slurm", "flux"), default=None,
                   help="pin the scheduler (default: liveness-detect)")
    p.add_argument("--nodes", type=int, default=0, help="override the card's node count")
    p.add_argument("--tasks-per-node", dest="tasks_per_node", type=int, default=0,
                   help="override the card's tasks per node")
    p.add_argument("--account", default=None, help="charge account/WCID (default: config/env, "
                                                   "else probed on the cluster)")
    p.add_argument("--partition", default=None,
                   help="partition/queue (default: auto — soonest-start pick; 'off' = scheduler default)")
    p.add_argument("--time", default=None, help="walltime (default: the card's, else config)")
    p.add_argument("--license", default=None, help="Slurm license string (default: config site.license)")
    p.add_argument("--proxy", default=None, metavar="URL",
                   help="corporate proxy forwarded to a container app's image pull")
    p.add_argument("--stage-source", default=None, metavar="ARCHIVE", dest="stage_source",
                   help="push a hand-downloaded source archive (e.g. from your browser, which "
                        "passes the egress filter's auth) into the job's spack mirror on the "
                        "cluster before submitting — for sites that block spack's own fetch")
    p.add_argument("--detach", action="store_true",
                   help="submit and return immediately (default: wait for the job and print its log)")
    p.add_argument("--dryrun", action="store_true", help="print the batch script; submit nothing")
    p.set_defaults(func=cmd_app, location=None, box=None, scheduler_args=[])

    p = sub.add_parser("doctor", help="audit the environment for known field issues "
                                      "(proxy/CA/runtime/scheduler/OOM/…); OK/WARN/FAIL + a fix each")
    p.add_argument("--net", action="store_true",
                   help="also probe outbound image-registry reachability (ghcr.io/docker.io 403 check)")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run the audit ON that cluster's login node over SSH")
    p.set_defaults(func=cmd_doctor, location=None)

    p = sub.add_parser(
        "serve",
        help="serve MODEL as an OpenAI-compatible endpoint (engine/image/runtime/port auto-resolved)",
    )
    p.add_argument("model", nargs="?", default=None,
                   help="local path or transport URI: hf://, ollama:// (pulled via RamaLama), "
                        "s3:// (staged from a bucket). oci://, docker:// are recognized but their "
                        "pull is not implemented yet — pull with podman/docker and serve by path. "
                        "Alternative: --box")
    p.add_argument("--box", default=None, help="serve from a box TOML profile instead of MODEL")
    p.add_argument("--location", default=None, help="site TOML profile (scheduler/runtime/accelerator/tuning)")
    p.add_argument("--system", default=None, metavar="NAME",
                   help="a built-in SYSTEM CARD (deployment profile per system type: e.g. "
                        "slurm-cuda, flux-rocm, laptop-podman, cloud-aws-gpu, openshift-gpu). "
                        "List them with `boxy cards`. Sugar over --location; explicit flags win")
    p.add_argument("--engine", choices=["llama.cpp", "vllm"], default=None,
                   help="inference engine (default: inferred — GGUF/ollama -> llama.cpp, else vLLM)")
    p.add_argument("--runtime", choices=["podman", "docker", "apptainer", "charliecloud"], default=None,
                   help="container runtime (default: first WORKING one found; charliecloud is experimental)")
    p.add_argument("--scheduler", choices=["none", "slurm", "flux"], default=None,
                   help="submit as a job via this scheduler (never done automatically)")
    p.add_argument("--accelerator", choices=list(ACCELERATORS), default=None,
                   help="pin the accelerator (needed when submitting GPU jobs from GPU-less login nodes)")
    p.add_argument("--image", default=None, help="container image (default: per engine+accelerator)")
    p.add_argument("--registry", default=None, metavar="HOST[/PATH]",
                   help="pull images from this registry instead (site mirror / local registry): "
                        "replaces the image's registry component. Per-registry rewrites go in "
                        "[location.image_mirrors]")
    p.add_argument("--proxy", default=None, metavar="URL",
                   help="corporate proxy (e.g. http://proxy.example.com:80) applied to BOTH the "
                        "login-node model download (Hugging Face) AND the compute node's image pull + "
                        "in-container downloads. USUALLY UNNEEDED: boxy auto-uses your "
                        "http_proxy/https_proxy env (or config network.proxy / BOXY_PROXY) and, over "
                        "--ssh, forwards it to the cluster. Fixes ghcr.io/huggingface.co 403 on nodes "
                        "that must egress through a proxy")
    p.add_argument("--agentless", action="store_true",
                   help="emit a SELF-CONTAINED batch script (podman + a shared-FS endpoint write, no boxy "
                        "on the compute node). Requires --accelerator/--image (hardware can't be detected "
                        "off-node) and a pre-staged shared-FS model (transport-URI pulls need RamaLama)")
    p.add_argument("--port", type=int, default=None, help="serving port (default: engine default, next free)")
    p.add_argument("--gpus", type=int, default=None, help="GPUs per node for the --scheduler job request")
    p.add_argument("--nodes", type=int, default=None,
                   help="node count for the job. For one instance: nodes to distribute across "
                        "(Ray). With --replicas: the POOL size to spread the replicas across "
                        "(NOT per-replica; see --nodes-per-replica for that)")
    p.add_argument("--name", default=None, help="container name (default: derived from the model)")
    p.add_argument("--models-dir", default=None,
                   help="where to download an s3:// model (default: ./models, or "
                        "[location.staging] models_dir, or $BOXY_MODELS_DIR)")
    p.add_argument("--distributed", dest="distributed", action="store_true", default=None,
                   help="serve one vLLM instance across the allocated nodes via Ray "
                        "(tensor-parallel per node x pipeline-parallel across nodes; auto-on for "
                        "vllm + --nodes>1)")
    p.add_argument("--no-distributed", dest="distributed", action="store_false",
                   help="force single-node serving even with --nodes>1 (no Ray)")
    p.add_argument("--unique", action="store_true",
                   help="append a unique suffix to the name so you can launch MULTIPLE instances of "
                        "the same model at once (each gets its own job, log, and endpoint) instead of "
                        "reusing/blocking on the single deterministic name. NOTE: by default boxy "
                        "already does this automatically when a live instance exists — this just forces "
                        "a fresh instance every time")
    p.add_argument("--no-auto-unique", dest="no_auto_unique", action="store_true",
                   help="restore the strict singleton: if a live instance of the model exists, refuse "
                        "instead of auto-starting an independent one (also BOXY_AUTO_UNIQUE=false)")
    p.add_argument("--replicas", type=int, default=1, metavar="K",
                   help="data-parallel: submit K independent instances of the model, each its own "
                        "batch job named <base>-r0..r{K-1} with its own endpoint/log/port. Requires "
                        "--scheduler slurm|flux; composes with --nodes>1 (each replica is itself a "
                        "distributed instance)")
    p.add_argument("--gpus-per-replica", type=int, default=1, metavar="R", dest="gpus_per_replica",
                   help="GPUs each --replicas instance uses (default 1). Replicas bin-pack onto a "
                        "node: (--gpus // R) replicas per node, each pinned to its own GPU(s). R>1 "
                        "gives each replica tensor-parallel=R")
    p.add_argument("--nodes-per-replica", type=int, default=1, metavar="M", dest="nodes_per_replica",
                   help="make each --replicas instance a MULTI-NODE distributed (Ray) instance "
                        "spanning M nodes (default 1 = single-node replicas). With M>1, --nodes is "
                        "ignored; total nodes = replicas x M")
    p.add_argument("--visible-gpus", default=None, dest="visible_gpus", help=argparse.SUPPRESS)
    p.add_argument("--trust-remote-code", action="store_true", dest="trust_remote_code",
                   help="let vLLM run the model repo's custom loader code (needed by some new/"
                        "custom architectures, e.g. Nemotron-Parse). Only for models you trust")
    p.add_argument("--router", nargs="?", type=int, const=8000, default=None,
                   metavar="PORT",
                   help="with --replicas K, after the replicas are READY start the built-in login-node "
                        "router on PORT (default 8000) presenting ONE OpenAI URL load-balanced across "
                        "them (least-outstanding). For production scale use `boxy router --emit`")
    p.add_argument("--partition", default=None, metavar="NAME|LIST|auto|all|off",
                   help="partition/queue for --scheduler jobs (Slurm --partition, Flux --queue). "
                        "DEFAULT is automatic: boxy picks the soonest-start partitions from sinfo, "
                        "restricted to those with GPUs when the job needs one, so it starts wherever "
                        "a GPU frees first (no flag needed). Override with a name/comma-list, `all` "
                        "(every partition), or `off` (the scheduler's own default). Resolved on the "
                        "cluster over --ssh; opt a fixed default via BOXY_PARTITION.")
    p.add_argument("--account", default=None,
                   help="account/bank for --scheduler jobs (Slurm --account, Flux --bank). "
                        "Omit to auto-discover (mywcid/sacctmgr); with several accounts and a "
                        "TTY, boxy shows an interactive picker. $WCID also bypasses the menu.")
    pick = p.add_mutually_exclusive_group()
    pick.add_argument("--pick-account", dest="pick_account", action="store_true", default=None,
                      help="force the interactive WCID picker even when auto-discovery would "
                           "otherwise take the first account (overrides site.pick_account).")
    pick.add_argument("--no-pick-account", dest="pick_account", action="store_false",
                      help="never prompt; silently use the first / remembered discovered account.")
    pickp = p.add_mutually_exclusive_group()
    pickp.add_argument("--pick-partition", dest="pick_partition", action="store_true", default=None,
                       help="force the interactive partition picker when 2+ are available "
                            "(overrides site.pick_partition).")
    pickp.add_argument("--no-pick-partition", dest="pick_partition", action="store_false",
                       help="never prompt; keep the soonest-start comma-list of partitions.")
    p.add_argument("--license", default=None,
                   help="Slurm license(s) as #SBATCH --license=VAL (e.g. tscratch:1 or "
                        "tscratch:1,pscratch:1), for sites that gate filesystems behind "
                        "licenses. Default from site.license / BOXY_LICENSE.")
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
    p.add_argument("--ready-timeout", type=float, default=config.get_float("timeouts.readiness"),
                   help="seconds to wait for the endpoint once the server starts "
                        "(default from config timeouts.readiness / BOXY_READY_TIMEOUT)")
    p.add_argument("--bind-host", default=None, metavar="ADDR",
                   help="address the engine binds inside the container (default "
                        "network.bind_host / BOXY_BIND_HOST, i.e. 0.0.0.0). Use 127.0.0.1 "
                        "only for a purely local single-host serve.")
    p.add_argument("--endpoint-file", default=None, help=argparse.SUPPRESS)
    p.add_argument("--save-profile", default=None, metavar="PREFIX",
                   help="write the resolved config to PREFIX.box.toml + PREFIX.location.toml")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run this command ON that cluster's login node over SSH (from anywhere; "
                        "OTP/YubiKey prompted once, session reused) and tunnel the endpoint back "
                        "to localhost. Also: BOXY_SSH_HOST env, or `remote=` in a --location profile")
    p.add_argument("--route", default=None, metavar="NAME",
                   help="with --ssh: print a friendly http://NAME.localhost:PORT/ tunnel URL (no DNS)")
    p.add_argument("--share", default=None, metavar="NAME",
                   help="with --ssh: publish the tunnel as https://NAME-boxy.apps.<cluster>/ via the "
                        "OpenShift relay (everyone-URL, zero teammate setup); stop: boxy unshare NAME")
    p.add_argument("--exposer", choices=["relay", "hosts"], default="relay",
                   help="which pluggable exposer --share uses (default relay)")
    p.add_argument("--delegate", action="store_true",
                   help="with --ssh: run the CLUSTER's own boxy instead of the default fully-"
                        "agentless flow (needs boxy installed there; also BOXY_SSH_DELEGATE=1). "
                        "Use for --replicas/--distributed/--box, which agentless doesn't cover yet.")
    p.add_argument("--prestage", dest="prestage", action="store_const", const="always", default=None,
                   help="agentless --ssh: force PRE-STAGING the image + model on the login node (over "
                        "your SSH session's network) so an ISOLATED compute node needs no network. "
                        "Default 'auto' stages an hf:// model automatically; --prestage also pre-pulls "
                        "for a by-path model. Opposite: --no-prestage / BOXY_AGENTLESS_PRESTAGE=never.")
    p.add_argument("--no-prestage", dest="prestage", action="store_const", const="never",
                   help="agentless --ssh: do NOT pre-stage; let the compute node pull the image/model "
                        "itself (only works on a NETWORKED compute node).")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the laptop-side HF architecture sanity check that refuses plainly "
                        "unservable models (ASR/audio/embedding) before a GPU allocation is burned")
    p.add_argument("--dryrun", action="store_true", help="print the command instead of executing it")
    p.add_argument("args", nargs="*", help="extra engine args (put them after --)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run", help="run the box with explicit arguments (raw passthrough; profile mode)")
    _add_common(p)
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the box entrypoint")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="pre-stage a model via RamaLama transports (pull on the login node)")
    p.add_argument("model", nargs="?", default=None,
                   help="transport URI: hf://, ollama:// (pulled via RamaLama). oci://, docker:// "
                        "are recognized but their pull is not implemented yet (pull with "
                        "podman/docker, serve by path). Alternative: --box")
    p.add_argument("--box", default=None, help="pull the model named by a box TOML profile")
    p.add_argument("--force", action="store_true",
                   help="remove any cached copy and re-pull clean (fixes a partial/corrupt "
                        "download from an interrupted pull)")
    p.add_argument("--proxy", default=None, metavar="URL",
                   help="reach Hugging Face through this corporate proxy (e.g. http://proxy.site:80)")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run the pull ON that cluster's login node over SSH (where the network is)")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_pull, location=None)

    p = sub.add_parser("build", help="build/convert the image for the location's runtime (OCI->SIF)")
    _add_common(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("generate", help="transpile box+location to another orchestrator: "
                                        "sky (SkyPilot YAML), slurm|flux (boxy-free job script), "
                                        "flux-mcp (persistent OpenShift MCP service), or "
                                        "relay (the everyone-URL share ingress on OpenShift)")
    p.add_argument("format", choices=["sky", "slurm", "flux", "sbatch", "flux-mcp", "relay", "card"],
                   help="sky = SkyPilot task YAML; slurm|flux = agentless batch script (no boxy on the "
                        "cluster); flux-mcp = the Flux MCP server as a persistent OpenShift service; "
                        "relay = the chisel relay behind `boxy open --share` (deploy once); "
                        "card = a model card from a HuggingFace id (see `card <hf-model-id>`)")
    p.add_argument("model_id", nargs="?", default=None,
                   help="card: the HuggingFace model id (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p.add_argument("--box", default=None)
    p.add_argument("--location", default=None)
    p.add_argument("--port", type=int, default=None)
    # card: generate a boxy model card from a HuggingFace model id
    p.add_argument("--engine", default=None, help="card: force the engine (vllm|llama.cpp) instead "
                                                  "of auto-detecting from the model's architecture")
    p.add_argument("--max-model-len", type=int, default=None, dest="max_model_len",
                   help="card: context length to cap at (default: min(model's native, 8192))")
    p.add_argument("--hf-token", default=None, dest="hf_token",
                   help="card: HuggingFace token for gated repos (else $HF_TOKEN / the HF cache token)")
    p.add_argument("--force", action="store_true",
                   help="card: overwrite an existing card without prompting (keeps a .bak)")
    p.add_argument("--dry-run", "--dryrun", action="store_true", dest="dryrun",
                   help="card: print the generated card without writing it")
    # flux-mcp / relay: persistent OpenShift services (no box/location needed)
    p.add_argument("--namespace", default=None, help="flux-mcp/relay: OpenShift namespace")
    p.add_argument("--host", default=None,
                   help="flux-mcp/relay: the OpenShift Route hostname (e.g. relay-boxy.apps.<cluster>)")
    p.add_argument("--flux-uri", default="", help="flux-mcp: FLUX_URI for reaching a remote Flux instance")
    p.add_argument("--auth", default="", help="relay: user:pass tunnel credential (else a REPLACE_ME "
                                              "placeholder + `oc create secret` hint)")
    p.add_argument("--key-seed", default="", help="relay: seed keeping the chisel host key stable across "
                                                  "pod restarts (else REPLACE_ME)")
    p.add_argument("--serve", action="store_true", help="add a SkyServe service block (sky serve up)")
    # agentless (slurm|flux) pins: hardware can't be detected off the compute node
    p.add_argument("--accelerator", default=None, help="agentless: pin the compute node's accelerator (cuda|rocm|…)")
    p.add_argument("--image", default=None,
                   help="agentless: pin the container image (else the engine+accel default); "
                        "relay: the chisel image (point at a mirror if Docker Hub is blocked)")
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--proxy", default=None, help="corporate proxy carried into the job's podman pull")
    p.add_argument("-o", "--output", default=None, help="write to file instead of stdout")
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

    p = sub.add_parser("logs", help="show a job's log + boxy's crash diagnosis (newest first)")
    p.add_argument("name", nargs="?", default=None,
                   help="job/instance name (prefix ok); default: the newest log")
    p.add_argument("--tail", type=int, default=60, help="lines from the end (default 60)")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="read the logs on that cluster over SSH (reuses the boxy SSH session)")
    p.set_defaults(func=cmd_logs, location=None)

    p = sub.add_parser("attach", help="re-join a detached serve: open the tunnel, wait for READY, "
                                      "print the url (agentless --ssh jobs)")
    p.add_argument("name", nargs="?", default=None,
                   help="job name from `boxy list` (optional when only one is running)")
    p.add_argument("--local-port", type=int, default=None,
                   help="laptop port for the tunnel (default: the serving port when free)")
    p.add_argument("--ready-timeout", type=float, default=0.0,
                   help="max seconds to wait (default: keeps waiting while the job is alive)")
    p.add_argument("--share", default="",
                   help="publish as https://NAME-boxy.apps.<cluster>/ via the OpenShift relay once ready")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="accepted for symmetry; the record already knows its cluster")
    p.set_defaults(func=cmd_attach, location=None)

    p = sub.add_parser("curl", help="query a served model: boxy curl [NAME] --prompt '...' "
                                    "(finds the endpoint from boxy's records; --ssh runs it cluster-side)")
    p.add_argument("name", nargs="?", default=None,
                   help="instance name from the READY banner / boxy list (optional if only one is up)")
    p.add_argument("--prompt", default="Reply with exactly: boxy endpoint OK",
                   help="the user message to send (default: a one-line liveness probe)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--url", default=None, help="query this endpoint directly instead of a NAME")
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run on that cluster's login node over SSH (compute-node hostnames "
                        "resolve there; reuses the boxy SSH session)")
    p.set_defaults(func=cmd_curl, location=None)

    p = sub.add_parser("open", help="open a served model in your browser: boxy open [NAME] --ssh "
                                    "user@login (tunnels the endpoint to a local port)")
    p.add_argument("name", nargs="?", default=None,
                   help="instance name from boxy list (optional if only one is up)")
    p.add_argument("--port", dest="local_port", type=int, default=None, metavar="N",
                   help="pin the LOCAL port for a stable URL (http://127.0.0.1:N/); "
                        "default reuses the remote port when free, else picks a free one")
    p.add_argument("--route", default=None, metavar="NAME",
                   help="print a friendly http://NAME.localhost:PORT/ URL for the tunnel — "
                        "*.localhost resolves to 127.0.0.1 in every browser on macOS+Linux with "
                        "zero DNS setup (RFC 6761); a bare NAME gets '.localhost' appended")
    p.add_argument("--share", default=None, metavar="NAME",
                   help="publish the tunnel as https://NAME-boxy.apps.<cluster>/ via the OpenShift "
                        "relay — reachable by ANYONE on the corporate network, teammates install "
                        "nothing (deploy the relay once: boxy generate relay). Stop: boxy unshare NAME")
    p.add_argument("--exposer", choices=["relay", "hosts"], default="relay",
                   help="which pluggable exposer --share uses (default relay)")
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="tunnel from that cluster's login node back to this machine over SSH")
    p.set_defaults(func=cmd_open, location=None)

    p = sub.add_parser("sweep", help="scaling study: submit each rung (nodes or replicas in "
                                     "powers of 2), benchmark it, tear it down, print a comparison table")
    p.add_argument("model", help="model to serve on each rung (same MODEL as boxy serve)")
    p.add_argument("--sweep-nodes", default=None, metavar="LIST",
                   help="node counts to sweep, comma list (e.g. 1,2,4,8) — one distributed instance per rung")
    p.add_argument("--sweep-replicas", default=None, metavar="LIST",
                   help="replica counts to sweep, comma list (e.g. 1,2,4,8) — K data-parallel instances per rung")
    p.add_argument("--scheduler", choices=["slurm", "flux"], default=None,
                   help="scheduler to submit rungs to (or take it from --location)")
    p.add_argument("--location", default=None, help="site TOML profile")
    p.add_argument("--gpus", type=int, default=None, help="GPUs per node for each rung")
    p.add_argument("--nodes", type=int, default=None,
                   help="nodes per replica when sweeping --sweep-replicas (default 1)")
    p.add_argument("--engine", choices=["llama.cpp", "vllm"], default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--runtime", choices=["podman", "docker", "apptainer"], default=None)
    p.add_argument("--accelerator", choices=list(ACCELERATORS), default=None)
    p.add_argument("--partition", default=None)
    p.add_argument("--account", default=None)
    p.add_argument("--time", default=None)
    p.add_argument("--scheduler-arg", action="append", default=[], dest="scheduler_args", metavar="FLAG")
    p.add_argument("--batch-sizes", default=None, help="comma list, default 1,2,4,...,1024")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--dataset", default=None, help="JSON list of prompts or ShareGPT JSON")
    p.add_argument("--ready-timeout", type=float, default=1800.0,
                   help="seconds to wait for each rung to become ready (default 1800)")
    p.add_argument("--keep", action="store_true", help="leave each rung running instead of tearing it down")
    p.add_argument("-o", "--output", default=None, help="write the scaling table as CSV here")
    p.add_argument("--json", action="store_true", help="print the scaling report as JSON too")
    p.add_argument("--dryrun", action="store_true", help="print the sweep plan without submitting")
    p.set_defaults(func=cmd_sweep, name=None, models_dir=None)

    p = sub.add_parser("launch", help="launch the box on cloud via SkyPilot (delegated)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--serve", action="store_true", help="managed serving via SkyServe (sky serve up)")
    p.add_argument("--proxy", default=None,
                   help="the task runs ON-NET behind this corporate proxy: carry the proxy env "
                        "AND the merged CA bundle onto the task (omit for off-net cloud VMs)")
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
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run on that cluster's login node over SSH (reuses the boxy SSH session)")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("list", help="list running boxy-launched containers")
    p.add_argument("--location", default=None)
    p.add_argument("--runtime", choices=["podman", "docker"], default=None)
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run on that cluster's login node over SSH (reuses the boxy SSH session)")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("unshare", help="tear down an everyone-URL share: kill the relay client "
                                       "and delete its Route/Service (see `boxy list`)")
    p.add_argument("alias", help="the --share name to tear down")
    p.add_argument("--exposer", choices=["relay", "hosts"], default="relay")
    p.set_defaults(func=cmd_unshare, location=None)

    p = sub.add_parser("router",
                       help="front a --replicas set (<base>-r*) with ONE OpenAI URL (load-balanced)")
    p.add_argument("base", help="replica base name — the <base> of <base>-r0..r{K-1} (see `boxy list`)")
    p.add_argument("--port", type=int, default=8000, help="listen port (default 8000)")
    p.add_argument("--policy", choices=["least", "round-robin"], default="least",
                   help="load-balancing policy (default: least-outstanding-requests — best for LLMs)")
    p.add_argument("--emit", choices=["nginx", "haproxy", "litellm"], default=None,
                   help="instead of running the built-in proxy, PRINT a config for a production proxy "
                        "(nginx/haproxy/litellm) built from the live replica endpoints")
    p.add_argument("--refresh", type=float, default=10.0,
                   help="seconds between replica re-scans (join/leave discovery; default 10)")
    p.add_argument("--dryrun", action="store_true", help="print the router plan without serving")
    p.set_defaults(func=cmd_router)

    p = sub.add_parser("stage", help="stage a model from a site-local S3 bucket to the shared FS")
    p.add_argument("model", nargs="?", default=None,
                   help="s3://BUCKET/PREFIX (bucket/prefix default to S3_BUCKET_NAME/S3_PATH)")
    p.add_argument("--box", default=None, help="stage the model named by a box TOML profile")
    p.add_argument("--models-dir", default=None,
                   help="destination on the shared FS (default: ./models)")
    p.add_argument("--s3-endpoint", default=None,
                   help="S3 endpoint URL (default: $S3_ENDPOINT_URL; empty = real AWS)")
    p.add_argument("--s3-backend", choices=["auto", "boto3", "awscli", "container"], default="auto",
                   help="how to fetch: boto3 lib, host aws CLI, or aws-cli container (paper-style); "
                        "default auto (boto3 -> aws -> container)")
    p.add_argument("--runtime", choices=["podman", "docker", "apptainer"], default=None,
                   help="container engine for --s3-backend=container")
    p.add_argument("--no-sign-request", action="store_true",
                   help="anonymous access for a public bucket (no credentials; "
                        "also via S3_NO_SIGN_REQUEST=1)")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser("alloc", help="request nodes via the location's scheduler (post-MVP)")
    p.set_defaults(func=_stub("alloc"))

    return parser


# Scheduler flag pass-through. ANY flag boxy doesn't own is handed to the
# ACTIVE scheduler verbatim (boxy translates the portable trio internally), so
#   boxy serve M --scheduler slurm --account=acct --license=tscratch:1
# just works — and the same command under --scheduler flux renders in flux's
# spelling. Prefixed forms remain: --sched-* (explicitly neutral) and
# --slurm-*/--flux-* (pinned to one scheduler; warned when it isn't active).
_DYNAMIC_FLAG = re.compile(r"^--(sched|slurm|flux)-([A-Za-z0-9][A-Za-z0-9-]*)(?:=(.*))?$")
_BARE_FLAG = re.compile(r"^--([A-Za-z0-9][A-Za-z0-9-]*)(?:=(.*))?$")


def _dynamic_for(dynamic: list, active: str) -> list:
    """The pass-through (key, value) pairs that apply under the ACTIVE scheduler:
    all --sched-* flags, plus --<active>-* pinned ones."""
    return [(k, v) for s, k, v in dynamic if s in ("sched", active)]


def _dynamic_ignored(dynamic: list, active: str) -> list[str]:
    """Pinned flags for a DIFFERENT scheduler (never --sched-*: those always apply)."""
    return [f"--{s}-{k}" for s, k, v in dynamic if s not in ("sched", active)]


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        # boxy leans on POSIX process control (os.killpg, start_new_session, `ps`)
        # and OpenSSH connection multiplexing (`ssh -O`, ControlMaster), none of
        # which exist on native Windows. Fail clearly rather than half-work.
        print("boxy: Windows is not supported (it needs POSIX process control and "
              "OpenSSH multiplexing). Run it under WSL2: "
              "https://learn.microsoft.com/windows/wsl/", file=sys.stderr)
        return 2
    # Everything after a standalone `--` is engine args, verbatim. argparse
    # cannot express this next to optional positionals (a `*` positional only
    # matches one contiguous chunk), so split before parsing.
    argv = list(sys.argv[1:] if argv is None else argv)
    raw_argv = list(argv)  # verbatim command, for --ssh remote re-invocation
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    args, unknown = build_parser().parse_known_args(argv)
    args._raw_argv = raw_argv
    # Scheduler flag pass-through: --sched-FLAG[=VALUE] (neutral; the active
    # scheduler applies it) or the pinned --slurm-*/--flux-* spellings
    # flows into the job request untranslated except for spelling — new
    # scheduler flags never require a boxy change. Values need `=`.
    dynamic: list[tuple[str, str, str | None]] = []
    bad: list[str] = []
    is_serve = getattr(args, "subcommand", "") == "serve"
    # boxy's own flags for this subcommand — a near-miss typo of one of these
    # must ERROR with a suggestion, never silently become a scheduler flag.
    own_flags = {"--" + k.replace("_", "-") for k in vars(args)}
    for token in unknown:
        match = _DYNAMIC_FLAG.match(token)
        if match and is_serve:
            dynamic.append((match[1], match[2], match[3]))
            continue
        bare = _BARE_FLAG.match(token)
        if bare and is_serve:
            import difflib

            close = difflib.get_close_matches(f"--{bare[1]}", sorted(own_flags), n=1, cutoff=0.85)
            if close:
                print(f"boxy: error: unrecognized argument {token} — did you mean {close[0]}?",
                      file=sys.stderr)
                return 2
            dynamic.append(("sched", bare[1], bare[2]))  # → the active scheduler, verbatim
            continue
        bad.append(token)
    if bad:
        print(f"boxy: error: unrecognized arguments: {' '.join(bad)}\n"
              f"  (with --scheduler, any --FLAG[=VALUE] boxy doesn't own passes through to the "
              f"scheduler; values need the = form)",
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
