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
    ]
    machine = _check_podman_machine()
    if machine is not None:
        results.append(machine)
    if net:
        results.extend(_check_image_registries())
    return results
