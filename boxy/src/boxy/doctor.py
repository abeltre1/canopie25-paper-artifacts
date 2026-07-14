"""`boxy doctor` — a read-only environment audit.

The executable half of the SPEC §8b known-issues registry: instead of waiting
for a job to fail on the compute node, `boxy doctor` checks the things that
actually bit users in the field (proxy/CA/token, container runtime, scheduler,
accelerator, per-cluster jobs dir, OOM'd containers, image-registry reach) and
reports each as OK / WARN / FAIL with a one-line fix. `boxy doctor --ssh
user@login` runs the same audit ON the cluster (like list/curl/logs).

Every check reuses an existing boxy helper — this module only classifies and
phrases; it never re-implements detection. Read-only: it probes and reports,
it never changes anything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Result:
    name: str
    status: str  # OK | WARN | FAIL
    detail: str
    fix: str = ""


# ---- individual checks (each returns one Result) ------------------------------


def _check_runtime() -> Result:
    from boxy import resolve
    from boxy.backends import BACKENDS

    present = [n for n in BACKENDS if shutil.which(n)]
    if not present:
        return Result("container runtime", FAIL, "none of podman/docker/apptainer on PATH",
                      "install one, or on a login node `module load` the site's container runtime")
    working = [n for n in ("podman", "docker") if n in present and resolve._runtime_works(n)]
    if working:
        return Result("container runtime", OK, f"{', '.join(present)} (working: {', '.join(working)})")
    if "apptainer" in present:
        return Result("container runtime", OK, "apptainer (foreground-only in the MVP)")
    return Result("container runtime", WARN,
                  f"{', '.join(present)} on PATH but none responded to a probe",
                  "rootless podman may be broken (no subuid ranges / storage on NFS); "
                  "pin a working one with --runtime, or fix per `podman info`")


def _check_scheduler() -> Result:
    subs = [s for s in ("sbatch", "flux") if shutil.which(s)]
    launchers = [ln for ln in ("srun", "flux") if shutil.which(ln)]
    if subs:
        return Result("scheduler", OK, f"submit: {', '.join(subs)}; launch: {', '.join(launchers) or 'none'}")
    if launchers:
        return Result("scheduler", WARN, f"{', '.join(launchers)} present but no batch submitter (sbatch/flux)",
                      "attached mode only here; for detached serving submit from a full login node")
    return Result("scheduler", OK, "none (local/laptop) — --scheduler needs --ssh to a cluster")


def _check_accelerator() -> Result:
    from boxy import ramalama_shim

    accel = ramalama_shim.detect_accel()
    if accel and accel != "none":
        return Result("accelerator", OK, accel)
    lib = "" if ramalama_shim.ramalama_available() else " (ramalama not installed — detection limited)"
    return Result("accelerator", OK, f"none detected{lib} — pin --accelerator on a GPU job from a GPU-less host")


def _check_proxy() -> Result:
    from boxy import ramalama_shim

    proxies = ramalama_shim.active_proxies()
    if not proxies:
        return Result("proxy", OK, "none set (direct)")
    shown = "  ".join(f"{k}: {v}" for k, v in proxies.items())
    if "http" in proxies and "https" not in proxies:
        return Result("proxy", WARN, f"{shown} — http_proxy set but https_proxy is NOT",
                      "registries are all https and will BYPASS the proxy; if your profile does "
                      'https_proxy="${http_proxy}", it must come AFTER http_proxy is set')
    return Result("proxy", OK, f"{shown}  (registry traffic follows these)")


def _check_tls() -> Result:
    from boxy import ramalama_shim

    cert = os.environ.get("SSL_CERT_FILE")
    if cert and not os.path.exists(cert):
        return Result("tls / CA bundle", FAIL, f"SSL_CERT_FILE={cert} does not exist",
                      "OpenSSL SILENTLY ignores a missing path — every pull then fails verification; "
                      "fix the path or unset it")
    if cert:
        merged = " (boxy merges it with certifi's public CAs on pull)" if not os.environ.get(
            "BOXY_NO_CA_MERGE") else " (BOXY_NO_CA_MERGE set — no merge; non-intercepted hosts may fail)"
        return Result("tls / CA bundle", OK, f"SSL_CERT_FILE={cert}{merged}")
    if ramalama_shim.discover_os_ca_bundle():
        return Result("tls / CA bundle", OK, "system CA store (boxy auto-merges OS CAs + certifi on pull)")
    return Result("tls / CA bundle", WARN, "no SSL_CERT_FILE and no OS CA bundle found",
                  "if pulls fail CERTIFICATE_VERIFY_FAILED: pip install certifi, or set SSL_CERT_FILE "
                  "to your site CA and persist it")


def _check_hf_token() -> Result:
    from boxy import ramalama_shim

    token, source = ramalama_shim.effective_hf_token()
    if token:
        return Result("HuggingFace token", OK, f"present ({source}) — validate with `boxy info --net`")
    if source.startswith("HF_TOKEN env var (set but EMPTY"):
        return Result("HuggingFace token", WARN, source, "export a non-empty HF_TOKEN for gated repos")
    return Result("HuggingFace token", OK, "not set (fine for public models; export HF_TOKEN for gated repos)")


def _check_cluster_state() -> Result:
    from boxy import jobs

    cluster = jobs.local_cluster()
    d = jobs._dir()
    detail = f"cluster '{cluster}' — state in {d}"
    if not os.environ.get("BOXY_JOBS_DIR") and d.parent != d:
        legacy = list(d.parent.glob("*.log")) + list(d.parent.glob("*.json"))
        legacy = [p for p in legacy if p.parent == d.parent]
        if legacy:
            return Result("cluster state", WARN,
                          f"{detail}; {len(legacy)} pre-separation file(s) remain in the shared root {d.parent}",
                          "boxy now partitions per cluster; old flat files stay put — inspect/clear them manually")
    return Result("cluster state", OK, detail)


def _check_exited_containers(runtime: str | None = None) -> Result:
    runtime = runtime or next((n for n in ("podman", "docker") if shutil.which(n)), None)
    if not runtime:
        return Result("exited containers", OK, "no container runtime to inspect")
    result = subprocess.run(
        [runtime, "ps", "-a", "--filter", "label=boxy.box", "--filter", "status=exited",
         "--format", "{{.Names}} {{.Status}}"],
        capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return Result("exited containers", OK, "none")
    lines = result.stdout.strip().splitlines()
    oom = []
    for line in lines:
        name = line.split(" ", 1)[0]
        code = subprocess.run([runtime, "inspect", "--format",
                               "{{.State.ExitCode}} {{.State.OOMKilled}}", name],
                              capture_output=True, text=True).stdout.split()
        if code and (code[0] == "137" or (len(code) > 1 and code[1].lower() == "true")):
            oom.append(name)
    if oom:
        return Result("exited containers", WARN,
                      f"{len(lines)} exited; {len(oom)} OOM-killed (exit 137): {', '.join(oom)}",
                      "the runtime VM ran out of RAM — raise it: "
                      "podman machine stop && podman machine set --memory 8192 --cpus 4 && podman machine start")
    return Result("exited containers", WARN, f"{len(lines)} exited boxy container(s) (crashed/stopped)",
                  f"inspect: {runtime} logs <name>; clear: "
                  f"{runtime} rm $({runtime} ps -aq --filter label=boxy.box --filter status=exited)")


def _check_relay() -> Result:
    """Zero-install `--share` readiness (RUNBOOK §0.993 'deploy once, share
    forever'): (1) can THIS host run the chisel client — a host binary OR a
    container runtime — and (2) when oc is reachable, is the shared OpenShift relay
    Route admitted? Both sides need nothing else installed."""
    from boxy import config
    from boxy.exposers import relay

    mode = config.get_str("relay.client_mode")
    have_chisel = shutil.which(relay.chisel_bin()) is not None
    runtime = relay._first_runtime()
    if runtime and mode != "host":
        client = f"containerized chisel via {runtime} (zero install)"
    elif have_chisel:
        client = "host chisel binary"
    elif runtime:  # mode == host but no binary — still, a runtime exists
        client = f"containerized chisel via {runtime} (set relay.client_mode=container/auto)"
    else:
        return Result("share relay", WARN,
                      "no chisel binary and no container runtime — `--share` can't run the client here",
                      "install podman/docker (boxy runs chisel in a container) or `brew install chisel-tunnel`")

    status, detail = relay.relay_admission()
    if status == "ok":
        return Result("share relay", OK, f"client: {client}; relay Route {detail}")
    if status == "rejected":
        return Result("share relay", FAIL, f"client: {client}; {detail}",
                      "the relay host is taken or ingress rejected it — pick a free host and redeploy the relay")
    if status == "missing":
        return Result("share relay", WARN, f"client: {client}; {detail}",
                      "deploy it ONCE per cluster: "
                      "boxy generate relay --host relay-boxy.apps.<cluster> --auth boxy:<pw> | oc apply -f -")
    return Result("share relay", OK, f"client: {client}; {detail}")


def _check_podman_machine() -> Result | None:
    """macOS/Windows only: the podman VM's RAM is the usual cause of the
    'second instance killed the first' (exit 137). None on Linux (no VM)."""
    import sys

    if sys.platform not in ("darwin", "win32") or not shutil.which("podman"):
        return None
    result = subprocess.run(["podman", "machine", "inspect", "--format",
                             "{{.Resources.Memory}}"], capture_output=True, text=True)
    mem = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    try:
        mb = int(mem)
    except ValueError:
        return Result("podman machine", OK, "no default machine / unknown memory")
    if mb < 4096:
        return Result("podman machine", WARN, f"VM memory {mb} MB — small for LLM serving",
                      "one server + overhead can exceed this; a 2nd instance OOM-kills the 1st. "
                      "Raise: podman machine set --memory 8192 --cpus 4 (stop/start)")
    return Result("podman machine", OK, f"VM memory {mb} MB")


# ---- network checks (opt-in: --net) -------------------------------------------

IMAGE_REGISTRIES = (("ghcr.io", "https://ghcr.io/v2/"),
                    ("docker.io", "https://registry-1.docker.io/v2/"))


def _check_image_registries() -> list[Result]:
    """Can THIS host reach the container-image registries? A 403 is the Zscaler/
    proxy POLICY block that dooms a `podman pull` (SPEC §8b / RUNBOOK §0.965).
    Any other HTTP response proves reachability."""
    import urllib.error
    import urllib.request

    from boxy import ramalama_shim

    out: list[Result] = []
    for name, url in IMAGE_REGISTRIES:
        try:
            with urllib.request.urlopen(url, timeout=8) as resp:
                out.append(Result(f"image registry {name}", OK, f"reachable (HTTP {resp.status})"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):  # normal unauthenticated front-door answers
                out.append(Result(f"image registry {name}", OK, f"reachable (HTTP {e.code})"))
            elif e.code == 403:
                out.append(Result(f"image registry {name}", FAIL, "HTTP 403 — refused by a proxy/policy (Zscaler?)",
                                  "a compute-node `podman pull` will fail. Pass --proxy, pre-pull on a host that "
                                  "can reach it (shared $HOME store), or --registry a site mirror (RUNBOOK §0.965/§0.97)"))
            else:
                out.append(Result(f"image registry {name}", WARN, f"HTTP {e.code}"))
        except Exception as e:  # noqa: BLE001 — report-only probe
            reason = getattr(e, "reason", e)
            kind = ramalama_shim.net_failure_kind(reason)
            out.append(Result(f"image registry {name}", FAIL, f"unreachable ({reason})",
                              ramalama_shim.network_remedy(kind) if kind else "check network/VPN/proxy"))
    return out


# ---- driver -------------------------------------------------------------------


def remote_checks(run) -> list[Result]:
    """Audit a cluster over SSH with NO boxy installed there — `run(cmd)` returns
    (rc, combined_output) from a plain shell command on the login node (a
    `remote.ssh_capture` closure). This is the agentless auditor: it answers
    'is this cluster ready to serve?' before you ever install boxy on it, and
    catches the exact things that bite (no scheduler, no runtime, ghcr 403)."""
    out: list[Result] = []

    _, runtimes = run("for c in podman docker apptainer; do command -v $c >/dev/null && echo $c; done")
    present = runtimes.split()
    if not present:
        out.append(Result("container runtime", FAIL, "none of podman/docker/apptainer on the cluster PATH",
                          "`module load` the site's container runtime, or ask the admins"))
    else:
        out.append(Result("container runtime", OK, ", ".join(present)))

    _, subs = run("for c in sbatch flux srun; do command -v $c >/dev/null && echo $c; done")
    sub = subs.split()
    if any(s in sub for s in ("sbatch", "flux")):
        out.append(Result("scheduler", OK, ", ".join(sub)))
    else:
        out.append(Result("scheduler", WARN, "no sbatch/flux found on the login node",
                          "confirm you're on a real login node; some sites gate the scheduler behind a module"))

    # THE turnkey promise: can boxy discover the charge account here? Run the
    # SAME probe (`mywcid || sacctmgr`) the --ssh serve path uses and parse it,
    # so the user verifies account resolution BEFORE serving — the field failure
    # was a silent no-account batch script the scheduler then rejected.
    from boxy import site

    rc_acct, acct_out = run(site.remote_account_probe())
    accounts = site.parse_accounts(acct_out) if rc_acct == 0 else []
    if accounts:
        extra = f" (also: {', '.join(accounts[1:3])})" if len(accounts) > 1 else ""
        out.append(Result("account discovery", OK,
                          f"{accounts[0]}{extra} — turnkey will place `--account={accounts[0]}` in the "
                          "batch script (override with --account / BOXY_ACCOUNT)"))
    else:
        first = next((ln.strip()[:80] for ln in (acct_out or "").splitlines() if ln.strip()), "")
        detail = f"; the probe answered: {first!r}" if first else ""
        out.append(Result("account discovery", WARN,
                          f"no account parsed from mywcid/sacctmgr on the login node{detail}",
                          "pass --account <wcid> (or export BOXY_ACCOUNT); the scheduler's site default "
                          "applies otherwise and may reject the job"))

    # Which partitions can `--partition auto` choose from? Run the SAME sinfo
    # probe and rank it, so the user sees the soonest-start set before serving.
    if "sbatch" in sub or not any(s in sub for s in ("sbatch", "flux")):
        rc_part, part_out = run(site.remote_partition_probe("slurm"))
        value, _why = site.rank_remote_partitions(part_out, "slurm") if rc_part == 0 else ("", "")
        if value:
            out.append(Result("partitions", OK,
                              f"--partition auto → {value} (soonest-start; Slurm starts in whichever frees first)"))
        else:
            out.append(Result("partitions", OK,
                              "sinfo listed none (single-partition site?) — omit --partition to use the default"))

    _, accel = run("if command -v nvidia-smi >/dev/null; then echo cuda; "
                   "elif command -v rocminfo >/dev/null; then echo rocm; else echo none; fi")
    a = accel.strip() or "none"
    note = "" if a != "none" else " (login nodes often have no GPU — pin --accelerator for the job)"
    out.append(Result("accelerator (login node)", OK, f"{a}{note}"))

    _, prox = run('echo "${https_proxy:-}|${http_proxy:-}|${no_proxy:-}"')
    https, http, _no = (prox.strip().split("|") + ["", "", ""])[:3]
    if https or http:
        w = " (https_proxy empty — registries are https and will bypass it)" if http and not https else ""
        out.append(Result("proxy (login node)", OK if not w else WARN, f"https={https or '-'} http={http or '-'}{w}",
                          "" if not w else 'set https_proxy too (after http_proxy)'))
    else:
        out.append(Result("proxy (login node)", OK, "none set"))

    # THE one that bit the user: can the cluster pull the container image?
    # -L follows redirects (ghcr.io/v2/ 307 -> 401 is normal) so we report the
    # TRUE final status; a 403 is the Zscaler/proxy block that dooms podman pull.
    _, code = run("curl -sL --max-redirs 3 -o /dev/null -w '%{http_code}' --max-time 12 "
                  "https://ghcr.io/v2/ 2>/dev/null || echo curlfail")
    code = code.strip()
    if code in ("200", "401", "404") or (len(code) == 3 and code[0] == "3"):
        note = "reachable" if not code.startswith("3") else "reachable (redirect)"
        out.append(Result("image registry ghcr.io", OK,
                          f"{note} from the login node (HTTP {code}) — `podman pull` here should work; "
                          "pre-pull so compute nodes reuse the shared $HOME store"))
    elif code == "403":
        out.append(Result("image registry ghcr.io", FAIL, "HTTP 403 — refused by a proxy/policy (Zscaler?)",
                          "a `podman pull` will fail here AND on the compute node. Pass --proxy, "
                          "or --registry a site mirror (RUNBOOK §0.965/§0.97)"))
    else:
        out.append(Result("image registry ghcr.io", WARN, f"could not probe ({code or 'no curl'})",
                          "install curl or check the login node's egress; a pull may still work via the proxy"))

    # Is boxy even installed on the cluster, and is it turnkey-aware? The --ssh
    # serve path injects `--account` laptop-side precisely so an OLDER cluster
    # boxy still gets the account — report the version so the user knows whether
    # the injection is load-bearing or the cluster resolves it on its own.
    rc_ver, ver = run("boxy --version 2>/dev/null || echo absent")
    ver = (ver.strip().splitlines() or [""])[0]
    if rc_ver != 0 or not ver or ver == "absent":
        out.append(Result("cluster boxy", OK,
                          "not installed — fine: `--ssh` delegation resolves the account laptop-side and "
                          "passes it as --account (no boxy needed on the cluster to serve turnkey)"))
    else:
        out.append(Result("cluster boxy", OK,
                          f"{ver} — a --ssh serve still injects --account laptop-side, so an older "
                          "cluster boxy without turnkey is covered"))

    _, jobs_ls = run("ls -d ~/.local/share/boxy/jobs/*/ 2>/dev/null | tr '\\n' ' '")
    if jobs_ls.strip():
        out.append(Result("boxy state", OK, f"per-cluster jobs dir(s): {jobs_ls.strip()}"))
    else:
        out.append(Result("boxy state", OK, "no jobs dir yet (nothing served here)"))

    return out


def run_checks(net: bool = False) -> list[Result]:
    """Every check, in a sensible reading order. `net` adds the (slower)
    outbound registry probes."""
    results = [
        _check_runtime(),
        _check_scheduler(),
        _check_accelerator(),
        _check_proxy(),
        _check_tls(),
        _check_hf_token(),
        _check_cluster_state(),
        _check_exited_containers(),
        _check_relay(),
    ]
    machine = _check_podman_machine()
    if machine is not None:
        results.append(machine)
    if net:
        results.extend(_check_image_registries())
    return results
