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

from boxy import ramalama_shim, version_string
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


def _proxy_prefix(args) -> str:
    """`env VAR=val ...` prefix for the COMPUTE-NODE command, so its host-side
    `podman pull` (and the inner boxy) reach the corporate proxy — the usual fix
    for a ghcr.io 403 on an isolated compute node. `--proxy URL` wins; otherwise
    the submitter's proxy env is carried over. '' when nothing is configured."""
    from boxy import ramalama_shim

    proxies = ramalama_shim.raw_proxy_env(getattr(args, "proxy", "") or "")
    if not proxies:
        return ""
    return "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in proxies.items()) + " "


def _apply_proxy_env(args) -> None:
    """Export `--proxy` into THIS process's env so the LOGIN-NODE model download
    (RamaLama / huggingface_hub, run in-process during resolve_model) reaches the
    corporate proxy too — not just the compute-node command (_proxy_prefix).
    Field gap: `serve --proxy` proxied the ghcr image pull on the compute node
    but NOT the HF model pull on the login node. Idempotent; no-op without --proxy."""
    proxy = getattr(args, "proxy", "") or ""
    if not proxy:
        return
    for key, val in ramalama_shim.raw_proxy_env(proxy).items():
        os.environ[key] = val


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
                "  FLUX compatibility wrapper — this is a Flux system (eldorado-class). Rerun with\n"
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
    site_args = list(location.scheduler_args)
    for kind, value in (("partition", args.partition), ("account", args.account), ("time", args.time)):
        if value:
            site_args.append(scheduler.site_directive(kind, value))
    site_args += list(args.scheduler_args or [])
    dynamic = getattr(args, "dynamic_flags", [])
    site_args += [scheduler.dynamic_directive(k, v) for k, v in _dynamic_for(dynamic, scheduler_name)]
    ignored = _dynamic_ignored(dynamic, scheduler_name)
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
        # A job under a DIFFERENT scheduler that we cannot confirm is alive (state
        # UNKNOWN) is NOT ours to protect: it lives on another cluster (labs share
        # $HOME across sites, so an eldorado flux record shows up on a hops slurm
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
            elif mismatch:
                print(f"boxy: {name} is already submitted as a {rec_sched_name} job "
                      f"({record['job']}, {state}), but you requested {scheduler_name}. "
                      f"Stop it first: boxy stop {name}.", file=sys.stderr)
            else:
                print(f"boxy: {name} is already submitted as {rec_sched_name} job {record['job']} "
                      f"({state}) — watch: boxy list; stop: boxy stop {name}", file=sys.stderr)
            return 1
        elif not args.dryrun:
            jobs.remove(name)  # stale record from a finished job (S6: dryrun must not mutate)

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
                      f"  status: boxy list    log: {expected_log}\n"
                      f"  stop:   boxy stop {name}", file=sys.stderr)
                return 1
            if state != last_state:
                print(f"###   job {job_id}: {state}")
                last_state = state
                last_note = time.time()
            elif time.time() - last_note > 30:
                print(f"###   still waiting (job {job_id}: {state}); log: {expected_log}")
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
                          f"Large models load slowly — watch the log:\n  tail -f {expected_log}\n"
                          f"  then: curl -s {url}/v1/models ; stop: boxy stop {name}", file=sys.stderr)
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
    site_args = list(base_loc.scheduler_args)
    for kind, value in (("partition", args.partition), ("account", args.account), ("time", args.time)):
        if value:
            site_args.append(scheduler.site_directive(kind, value))
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
                args, args.model, replica_names[i], port=8000 + slot, visible_gpus=ids,
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
    return remote.run_remote(target, getattr(args, "_raw_argv", []), tunnel_ready=tunnel_ready,
                             local_port=getattr(args, "local_port", None),
                             local_route=getattr(args, "route", "") or "",
                             share=getattr(args, "share", "") or "",
                             exposer_name=getattr(args, "exposer", None) or "relay")


def cmd_serve(args: argparse.Namespace) -> int:
    rc = _delegate_remote(args, tunnel_ready=True)
    if rc is not None:
        return rc
    _apply_proxy_env(args)  # --proxy reaches the login-node model pull, not just the job
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
        if scheduler_name in ("slurm", "flux"):
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


def cmd_generate(args: argparse.Namespace) -> int:
    from boxy import sky_export

    if args.format == "flux-mcp":
        return _generate_flux_mcp(args)
    if args.format == "relay":
        return _generate_relay(args)
    if args.format == "gateway":
        return _generate_gateway(args)
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
        print("boxy generate relay: --host is required (the relay's Route host, "
              "e.g. relay-boxy.apps.<cluster>)", file=sys.stderr)
        return 2
    text = relay.emit_relay_manifest(args.host, args.namespace or relay.DEFAULT_NAMESPACE,
                                     image=args.image or relay.RELAY_IMAGE,
                                     auth=args.auth, key_seed=args.key_seed)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"wrote {args.output}")
    else:
        print(text, end="")
    return 0


def _generate_gateway(args: argparse.Namespace) -> int:
    """Emit the ONE-TIME OpenSSH gateway prerequisites for OpenShift (the key
    Secret placeholder + egress NetworkPolicy + the image-build / authorized_keys
    steps in the header). Per-share Deployment/Service/Route are created at share
    time by the gateway exposer — no persistent server to deploy."""
    from boxy.exposers import gateway

    if not args.login:
        print("boxy generate gateway: --login is required (the pod's ssh target, "
              "e.g. boxy-svc@hops.sandia.gov)", file=sys.stderr)
        return 2
    text = gateway.emit_setup_manifest(args.login, namespace=args.namespace or gateway.DEFAULT_NAMESPACE,
                                       secret=args.secret or gateway.DEFAULT_SECRET,
                                       image=args.image or gateway.DEFAULT_IMAGE)
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
                                      image=args.image or mcp.FLUX_MCP_IMAGE,
                                      flux_uri=args.flux_uri)
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
    # the record knows which exposer created it; an explicit --exposer overrides
    # (legacy records predate the field and default to relay).
    exposer = getattr(args, "exposer", None) or record.get("exposer") or "relay"
    get_exposer(exposer).unexpose(args.alias)
    print(f"unshared {args.alias}  ({record['url']} is gone)")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
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
    rc = _delegate_remote(args)
    if rc is not None:
        return rc
    from boxy import jobs
    from boxy.schedulers import get_scheduler

    records = jobs.list_records()
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
                # clusters' records too (field report: an eldorado flux job listed
                # on hops as UNKNOWN). No point probing — the job lives elsewhere;
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
        from boxy.exposers import get_exposer
        print("shares (everyone-URLs on OpenShift):")
        for s in shares:
            exposer = s.get("exposer", "relay")
            live = get_exposer(exposer).is_live(s)
            state = "LIVE" if live else "DEAD (rerun with --share, or boxy unshare)"
            print(f"  {s['alias']}  {s['url']}/v1  [{exposer}]  {state}")
    location = Location.from_toml(args.location) if args.location else None
    try:
        runtime = args.runtime or _container_runtime(location)
    except RuntimeError:
        if records:
            return 0  # jobs listed; no container runtime on this host is fine
        raise
    rc = _list_local_containers(runtime, args.dryrun, have_records=bool(records))
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
    schedulers' binaries too (field report: hops has `flux` on PATH, so an
    eldorado flux record passed this check and boxy curl chased eldo1025)."""
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
    record is reaped (log files outlive DONE jobs) and over --ssh."""
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
    p.add_argument("--registry", default=None, metavar="HOST[/PATH]",
                   help="pull images from this registry instead (site mirror / local registry): "
                        "replaces the image's registry component. Per-registry rewrites go in "
                        "[location.image_mirrors]")
    p.add_argument("--proxy", default=None, metavar="URL",
                   help="corporate proxy (e.g. http://proxy.mysite.gov:80) applied to BOTH the "
                        "login-node model download (Hugging Face) AND the compute node's image pull + "
                        "in-container downloads. Omit to auto-use your http_proxy/https_proxy env. "
                        "Fixes ghcr.io/huggingface.co 403 on nodes that must egress through a proxy")
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
                        "reusing/blocking on the single deterministic name")
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
    p.add_argument("--ssh", default=None, metavar="USER@HOST",
                   help="run this command ON that cluster's login node over SSH (from anywhere; "
                        "OTP/YubiKey prompted once, session reused) and tunnel the endpoint back "
                        "to localhost. Also: BOXY_SSH_HOST env, or `remote=` in a --location profile")
    p.add_argument("--route", default=None, metavar="NAME",
                   help="with --ssh: print a friendly http://NAME.localhost:PORT/ tunnel URL (no DNS)")
    p.add_argument("--share", default=None, metavar="NAME",
                   help="with --ssh: publish the tunnel as https://NAME-boxy.apps.<cluster>/ on OpenShift "
                        "(everyone-URL, zero teammate setup; --exposer picks how); stop: boxy unshare NAME")
    p.add_argument("--exposer", choices=["gateway", "relay", "hosts"], default="gateway",
                   help="which pluggable exposer --share uses (default gateway: OpenSSH-only pod, "
                        "no third-party tunnel binary; relay: chisel; hosts: local /etc/hosts)")
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
                                        "flux-mcp (persistent OpenShift MCP service), gateway "
                                        "(OpenSSH everyone-URL setup), or relay (chisel share ingress)")
    p.add_argument("format", choices=["sky", "slurm", "flux", "sbatch", "flux-mcp", "gateway", "relay"],
                   help="sky = SkyPilot task YAML; slurm|flux = agentless batch script (no boxy on the "
                        "cluster); flux-mcp = the Flux MCP server as a persistent OpenShift service; "
                        "gateway = one-time OpenSSH share prerequisites (Secret + NetworkPolicy, no "
                        "tunnel binary); relay = the chisel relay behind `boxy open --share` (deploy once)")
    p.add_argument("--box", default=None)
    p.add_argument("--location", default=None)
    p.add_argument("--port", type=int, default=None)
    # flux-mcp / relay: persistent OpenShift services (no box/location needed)
    p.add_argument("--namespace", default=None, help="flux-mcp/relay: OpenShift namespace")
    p.add_argument("--host", default=None,
                   help="flux-mcp/relay: the OpenShift Route hostname (e.g. relay-boxy.apps.<cluster>)")
    p.add_argument("--flux-uri", default="", help="flux-mcp: FLUX_URI for reaching a remote Flux instance")
    p.add_argument("--login", default=None,
                   help="gateway: the pod's ssh target for the login node (e.g. boxy-svc@hops.sandia.gov)")
    p.add_argument("--secret", default=None, help="gateway: name of the login-node key Secret "
                                                  "(default boxy-gw-ssh)")
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
    p.add_argument("--exposer", choices=["gateway", "relay", "hosts"], default="gateway",
                   help="which pluggable exposer --share uses (default gateway: OpenSSH-only pod, "
                        "no third-party tunnel binary; relay: chisel; hosts: local /etc/hosts)")
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
    p.add_argument("--exposer", choices=["gateway", "relay", "hosts"], default=None,
                   help="override the exposer used to tear down (default: whatever created the share)")
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
