"""Submit from ANYWHERE: run boxy against a remote cluster over SSH.

    laptop$ boxy serve MODEL --scheduler slurm --gpus 4 --ssh user@clusterB-login1

boxy re-runs the SAME command on the cluster's login node over SSH, streams the
output back, and when the remote serve prints its READY endpoint, opens a port
forward so the model answers at http://127.0.0.1:<port>/v1 ON THE LAPTOP.

OTP + YubiKey (the design constraint that shapes everything here):
  HPC sites authenticate with one-time passwords and hardware tokens, so every
  NEW ssh connection costs an interactive prompt (and a token touch). boxy
  therefore authenticates ONCE into an OpenSSH **ControlMaster** socket
  (ControlPersist keeps it alive in the background) and multiplexes every
  subsequent operation — remote command, output stream, port forward — over
  that one authenticated connection with ZERO re-prompts. Port forwards are
  added dynamically on the live master (`ssh -O forward`), again without
  re-authenticating. The master-establishing ssh runs with the user's TTY
  attached (never captured) so the OTP prompt and YubiKey touch reach them.

Why the system `ssh` binary (not a Python SSH library): OpenSSH natively speaks
keyboard-interactive (OTP) and FIDO2/sk keys (YubiKey); Python SSH libraries do
not handle those reliably. Shelling out also inherits the user's ~/.ssh/config
(ProxyJump, bastions, canonical hostnames) for free — the same approach SkyPilot
takes for cloud nodes. Override the binary with BOXY_SSH (tests use a local shim).

This module is deliberately self-contained: nothing in deploy/schedulers/engines
knows about SSH. The CLI delegates the verbatim command here and boxy-on-the-
cluster does everything else exactly as if the user had typed it there.
"""

from __future__ import annotations

import os
import re
import shlex
import socket
import subprocess
import sys

from boxy import config

ENV_HOST = "BOXY_SSH_HOST"          # set once in your shell profile -> "from anywhere"
ENV_ACTIVE = "BOXY_REMOTE_ACTIVE"   # recursion guard: set on the remote side
ENV_SSH_BIN = "BOXY_SSH"            # override the ssh binary (tests: a local shim)
ENV_REMOTE_CMD = "BOXY_REMOTE_COMMAND"  # remote boxy spelling (default: "boxy")
ENV_PERSIST = "BOXY_SSH_PERSIST"    # how long the master + its tunnels live idle
ENV_NO_CA_PROP = "BOXY_NO_CA_PROPAGATE"  # opt out of copying the laptop CA to the cluster
ENV_NO_PROXY_PROP = "BOXY_NO_PROXY_PROPAGATE"  # opt out of forwarding the laptop proxy env

DEFAULT_PERSIST = "12h"  # one OTP+touch buys this much multiplexed access

# Where the propagated laptop CA lands on the cluster. $HOME (not ~) so it expands
# inside the `bash -lc` wrapper AND so it's on the SHARED filesystem every HPC site
# guarantees — the compute node running the job sees the same file.
REMOTE_CA_DIR = "$HOME/.local/share/boxy/store"
REMOTE_CA_PATH = "$HOME/.local/share/boxy/store/laptop-ca.crt"


def control_persist() -> str:
    """Idle lifetime of the multiplexed SSH master (and every tunnel riding it),
    so one OTP+YubiKey keeps working for a full workday. Override per your site's
    session cap: BOXY_SSH_PERSIST=8h (accepts OpenSSH time formats: 30m, 12h, ...)."""
    return config.get("ssh.control_persist")

# tunnel-worthy endpoint banners from the remote serve: a fresh READY, or an
# ALREADY SERVING reconnect (rerunning the same model finds the live job).
READY_RE = re.compile(r"###\s+(?:READY|ALREADY SERVING)\s+http://([^:/\s]+):(\d+)")

# The remote boxy prints this the moment the compute node starts serving, BEFORE
# it has confirmed readiness — e.g. "server starting on clusterc5 — waiting ... at
# http://clusterc5:8000/...". The laptop latches onto the host:port here and takes
# readiness over ITSELF (tunnel + localhost/health), so an old cluster boxy that
# loops "still waiting" forever (its own probe blocked by the proxy / not routable
# to the compute node) can't stop us from reporting READY + the chisel URL.
WAITING_RE = re.compile(r"(?:server starting on|waiting).*?http://([^:/\s]+):(\d+)")
# The job log path the remote boxy prints — a shared-FS fallback readiness signal.
LOG_RE = re.compile(r"log:\s*(/\S+\.log)")
# the remote boxy's repetitive "still waiting" spam — suppressed once WE own readiness.
STILLWAIT_RE = re.compile(r"###\s+still waiting|waiting for the job to start and the server")

# the remote boxy rejecting a subcommand/flag the local one just sent means the
# CLUSTER's install is older than the laptop's (field report: `boxy logs --ssh
# clusterA` -> "invalid choice: 'logs'") — say so instead of a bare usage error.
STALE_RE = re.compile(r"invalid choice: '[^']+'|unrecognized arguments:")

# rootless podman on an HPC login node has no /run/user/$UID (no user systemd
# session), so `podman ps` spews 'Failed to get rootless runtime dir' + 'creating
# events dirs: ... permission denied' before failing. That noise is meaningless to
# the user (the real instances are the scheduler jobs) — filter it from the SSH
# stream on the LAPTOP side so it's gone regardless of the cluster's boxy version
# (field report: boxy list --ssh clusterA). Kept tight so real errors still show.
NOISE_RE = re.compile(
    r"Failed to get rootless runtime dir"
    r"|creating events dirs:.*(?:/run/user/|permission denied)"
    r"|rootless.*(?:/run/user/\d+).*(?:no such file|permission denied)")


def ssh_bin() -> str:
    return config.get("binaries.ssh")


def control_path() -> str:
    # %C = hash(local host, remote host, port, user): short + collision-free
    # (macOS caps unix-socket paths at ~104 chars, so never build long literals).
    return config.get("ssh.control_path")


def _base_opts() -> list[str]:
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={control_path()}",
        "-o", f"ControlPersist={control_persist()}",
        "-o", f"ServerAliveInterval={config.get_int('ssh.server_alive_interval')}",
    ]


def resolve_target(args) -> str:
    """Where to run: --ssh flag > BOXY_SSH_HOST env > [location] remote. Empty
    string = run here. The env spelling is what makes the SAME command work
    from anywhere: export BOXY_SSH_HOST=user@login once in your shell profile."""
    flag = getattr(args, "ssh", None)
    if flag:
        return flag
    env = os.environ.get(ENV_HOST, "")
    if env:
        return env
    location = getattr(args, "location", None)
    if location:
        try:
            from boxy.location import Location

            return Location.from_toml(location).remote
        except (ValueError, OSError):
            return ""  # a broken profile fails later, locally, with its real error
    return ""


def ensure_master(host: str) -> int:
    """Authenticate ONCE (OTP prompt + YubiKey touch happen HERE, on the user's
    TTY) and leave a persistent multiplexing socket behind. Idempotent and
    instant when the master is already alive."""
    check = subprocess.run([ssh_bin(), "-O", "check", "-o", f"ControlPath={control_path()}", host],
                           capture_output=True, text=True)
    if check.returncode == 0:
        return 0
    print(f"### Connecting to {host} (one-time login — OTP/YubiKey prompts appear below; "
          f"the session is then reused for {control_persist()} with no re-prompts)")
    # NO capture: the OTP prompt and the YubiKey touch notification need the TTY.
    return subprocess.run([ssh_bin(), *_base_opts(), host, "true"]).returncode


def ssh_capture(host: str, remote_command: str, timeout: int = 20) -> tuple[int, str]:
    """Run a shell command on `host` over the live multiplexed master, capturing
    combined output. For the AGENTLESS auditors (e.g. `boxy doctor --ssh`) that
    probe a cluster with NO boxy installed there — plain `command -v`/`curl`/`ls`.
    ensure_master() must have succeeded first."""
    try:
        proc = subprocess.run(
            [ssh_bin(), "-o", f"ControlPath={control_path()}", host, remote_command],
            capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, ""
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def remote_argv(raw_argv: list[str]) -> list[str]:
    """The exact command the user typed, minus the LAPTOP-SIDE-ONLY flags (the
    remote side must run it LOCALLY: recursion is also belt-and-suspenders
    blocked by BOXY_REMOTE_ACTIVE). `--ssh` targets the remote; `--route` names
    the local tunnel URL; `--share`/`--exposer` publish it via the OpenShift
    relay — all consumed here on the laptop, and the cluster's boxy never sees
    them, so a bare `boxy open NAME` runs there even when the CLUSTER's install
    is older and doesn't know these flags (field report)."""
    out: list[str] = []
    skip = False
    for tok in raw_argv:
        if skip:
            skip = False
            continue
        if tok in ("--ssh", "--route", "--share", "--exposer"):
            skip = True
            continue
        if tok == "--delegate":
            continue  # laptop-side only (agentless-vs-delegate switch); old cluster boxy doesn't know it
        if tok.startswith(("--ssh=", "--route=", "--share=", "--exposer=")):
            continue
        out.append(tok)
    return out


def _local_site_ca() -> str | None:
    """The user's ORIGINAL site/interceptor CA to carry to the cluster, or None.
    Prefer a user-set SSL_CERT_FILE that is a real file and NOT boxy's own merged
    bundle; else the site CA boxy recorded when it merged one this process. An
    AUTO-merged OS store is never propagated — the cluster has its own OS store;
    what a compute node is missing is the site's TLS-interceptor CA, which only the
    user's SSL_CERT_FILE carries (field case: a site compute node, 2026-07)."""
    cert = os.environ.get("SSL_CERT_FILE", "")
    if cert and not cert.endswith("ca-merged.crt") and os.path.isfile(cert):
        return cert
    try:  # boxy may have already overwritten SSL_CERT_FILE with its merged bundle
        from boxy import ramalama_shim

        if ramalama_shim._ca_merge_kind == "site" and os.path.isfile(ramalama_shim._ca_merge_source):
            return ramalama_shim._ca_merge_source
    except Exception:  # noqa: BLE001 — CA propagation must never break the delegation
        pass
    return None


def propagate_ca(host: str) -> str | None:
    """Copy the laptop's site CA to `host` over the LIVE ssh master (no re-auth),
    landing it on the cluster's shared $HOME, and return the remote path — so the
    remote command (and the sbatch job it submits, via SLURM --export=ALL) can trust
    the same interceptor CA the laptop does. Returns None when there's nothing to
    send or BOXY_NO_CA_PROPAGATE is set. Best-effort: a copy failure warns but never
    aborts the delegation (the remote may already have a working trust store)."""
    if os.environ.get(ENV_NO_CA_PROP):
        return None
    ca = _local_site_ca()
    if not ca:
        return None
    try:
        with open(ca, "rb") as f:
            data = f.read()
    except OSError:
        return None
    # `cat >` over the master beats scp: no second auth, and it reuses the exact
    # multiplexed connection (ProxyJump/bastion included). $HOME expands remote-side.
    proc = subprocess.run(
        [ssh_bin(), "-o", f"ControlPath={control_path()}", host,
         f"mkdir -p {REMOTE_CA_DIR} && cat > {REMOTE_CA_PATH}"],
        input=data, capture_output=True)
    if proc.returncode != 0:
        print(f"warning: could not copy your site CA to {host} ({(proc.stderr or b'').decode(errors='replace')[:200]}) "
              f"— if remote pulls fail TLS, set SSL_CERT_FILE on the cluster manually", file=sys.stderr)
        return None
    return REMOTE_CA_PATH


def remote_proxy_env() -> dict[str, str]:
    """The proxy vars to FORWARD to the cluster over --ssh so its job's image/
    model pulls reach the corporate proxy — even when the cluster's own env
    doesn't have it (the laptop knows the proxy; the login node may not).
    Config `network.proxy` wins, else the laptop's ambient http(s)_proxy env.
    Empty when nothing is set or BOXY_NO_PROXY_PROPAGATE is set (tests opt out)."""
    if os.environ.get(ENV_NO_PROXY_PROP):
        return {}
    from boxy import ramalama_shim

    override = config.get_str("network.proxy")
    env = ramalama_shim.raw_proxy_env(override)
    # Only forward when there's an ACTUAL http/https proxy. Forwarding the
    # laptop's no_proxy ALONE (no proxy set) would silently override the
    # cluster's own no_proxy and route its internal registry/hosts through a
    # proxy that isn't even there (adversarial-review finding).
    if not (env.get("https_proxy") or env.get("http_proxy")):
        return {}
    return env


def _remote_command(argv: list[str], remote_ca: str | None = None,
                    proxy_env: dict[str, str] | None = None) -> str:
    """The command line run on the login node. `bash -lc` gives a LOGIN shell so
    the user's PATH/venv/modules load — the remote `boxy` must be installed there
    (spelling overridable: BOXY_REMOTE_COMMAND='source ~/venv/bin/activate && boxy').
    `remote_ca` (from propagate_ca) is injected as SSL_CERT_FILE so the cluster and
    its job trust the laptop's interceptor CA; `proxy_env` (from remote_proxy_env)
    is injected so the cluster boxy inherits the proxy and bakes it into the job.
    $HOME expands inside the login shell."""
    boxy_cmd = config.get("binaries.remote_command")
    # PYTHONUNBUFFERED: the cluster boxy's stdout is a PIPE over ssh, so Python
    # block-buffers it and the readiness/progress lines don't stream — they arrive
    # in a lump (or only at exit). Forcing unbuffered makes progress appear live,
    # for ANY cluster boxy version (field report: "not showing how it's progressing").
    prefix = f"{ENV_ACTIVE}=1 PYTHONUNBUFFERED=1"
    if remote_ca:
        prefix += f" SSL_CERT_FILE={remote_ca}"
    for key, val in (proxy_env or {}).items():
        prefix += f" {key}={shlex.quote(val)}"
    inner = f"{prefix} {boxy_cmd} {shlex.join(argv)}"
    return f"bash -lc {shlex.quote(inner)}"


def _local_port_free(port: int) -> bool:
    """Can ssh -L bind this local port? (bind-test — matches what ssh does, so a
    leftover forward/gvproxy holding the port is correctly seen as taken)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def route_url(alias: str, port: int) -> tuple[str, str]:
    """A friendly local URL for the tunnel — NO DNS server needed (SPEC §8b
    'Tier 1'). A bare name gets '.localhost', which resolves to 127.0.0.1 in
    every browser on every OS (RFC 6761, handled by the browser itself); a
    dotted name is used verbatim. Returns (url, note). The tunnel still binds
    127.0.0.1:port — only the printed name changes; the browser resolves the
    name back to loopback and hits the tunnel."""
    host = alias.strip().rstrip("/")
    for pre in ("https://", "http://"):
        if host.startswith(pre):
            host = host[len(pre):]
    host = host.split("/", 1)[0]
    note = ""
    if "." not in host:
        host = f"{host}.localhost"
        note = ("*.localhost -> 127.0.0.1 in browsers on macOS+Linux with zero setup (RFC 6761); "
                "for CLI on Linux it needs systemd-resolved (default on Ubuntu/Fedora/Debian)")
    elif not host.endswith(".localhost"):
        note = (f"point {host} at the tunnel: add '127.0.0.1  {host}' to /etc/hosts per machine")
    suffix = "" if port in (80, 443) else f":{port}"
    return f"http://{host}{suffix}/v1", note


def add_forward(host: str, local_port: int, remote_host: str, remote_port: int) -> int:
    """Add a port forward ON THE LIVE MASTER — no new connection, no re-auth.
    The forward persists as long as the master does (ControlPersist), so it
    outlives this boxy process; cancel with `ssh -O cancel -L ...`."""
    spec = f"{local_port}:{remote_host}:{remote_port}"
    return subprocess.run([ssh_bin(), "-O", "forward", "-L", spec,
                           "-o", f"ControlPath={control_path()}", host],
                          capture_output=True, text=True).returncode


def push_file(host: str, remote_path: str, content: str | bytes) -> int:
    """Write `content` (text or raw bytes — e.g. a staged source tarball) to
    `remote_path` on host over the LIVE master (cat >, no re-auth), creating the
    parent dir. `remote_path` is used UNQUOTED so a leading $HOME expands
    remote-side — pass a boxy-controlled path (a job-name slug), never user
    free-text. Returns the ssh rc (0 = written)."""
    proc = subprocess.run(
        [ssh_bin(), "-o", f"ControlPath={control_path()}", host,
         f'mkdir -p "$(dirname {remote_path})" && cat > {remote_path}'],
        input=content if isinstance(content, bytes) else content.encode(), capture_output=True)
    if proc.returncode != 0:
        print(f"warning: could not write {remote_path} on {host}: "
              f"{(proc.stderr or b'').decode(errors='replace')[:200]}", file=sys.stderr)
    return proc.returncode


def await_ready_and_tunnel(host: str, node: str, port: int, log_path: str,
                           local_port: int | None, local_route: str, share: str,
                           exposer_name: str, share_auto: bool, timeout_s: float = 1800.0,
                           still_alive=None) -> bool:
    """Block until the served endpoint is ready, then print READY -> LOCAL -> SHARE.
    Used by the fully-agentless --ssh serve (no boxy on the cluster): the laptop
    opens the tunnel to the compute node's `node:port` and confirms readiness via
    `http://127.0.0.1:<lport>/health` THROUGH it (unauthenticated; checked as
    localhost from the laptop where the forward reaches the serving node), falling
    back to the engine's "server is up" line in the job log grepped over the master.

    `still_alive` (optional callback): consulted when `timeout_s` expires — if the
    scheduler says the job is still alive, the wait EXTENDS instead of detaching.
    A big model legitimately loads for 10-20+ min; detaching while the job runs
    loses the tunnel + READY for no reason (field: clustera, vLLM cold start outlived
    the window). Returns True on ready; False only when the deadline passes AND
    the job is gone (or no callback was given)."""
    import time

    from boxy import readiness

    lport = _pick_local_port(local_port, port)
    if add_forward(host, lport, node, port) != 0:
        lport = 0  # couldn't forward — fall back to log-only, user tunnels manually
    deadline = time.time() + timeout_s
    while True:
        if time.time() >= deadline:
            if still_alive is None or not still_alive():
                return False
            print("###   the job is still alive (model loading) — continuing to wait ...", flush=True)
            deadline = time.time() + 120.0
        ready = bool(lport) and readiness.probe_once(f"http://127.0.0.1:{lport}", timeout=3) is not None
        if not ready and log_path:
            rc, out = ssh_capture(
                host, f"grep -Eq 'Application startup complete|server is listening|Uvicorn running on' "
                      f"{shlex.quote(log_path)} 2>/dev/null && echo BOXY_READY || true", timeout=15)
            ready = rc == 0 and "BOXY_READY" in out
        if ready:
            if lport:
                print(f"### READY   http://127.0.0.1:{lport}/v1   "
                      f"(confirmed via localhost:{lport}/health through the tunnel)")
                _announce_tunnel(lport, node, port, host, local_route, share, exposer_name, share_auto)
            else:
                print(f"### READY   http://{node}:{port}/v1   (server is up per the job log; "
                      f"couldn't forward a tunnel — tunnel manually: ssh -L {port}:{node}:{port} {host})")
            return True
        time.sleep(5)


def _pick_local_port(local_port: int | None, remote_port: int) -> int:
    """The laptop-side port for the forward: a user-pinned --port when free (stable
    URL), else the remote port when free, else any free one (a stale forward on
    that port never blocks the tunnel — field report: podman gvproxy on 8090)."""
    want = local_port or remote_port
    if _local_port_free(want):
        return want
    lport = _free_local_port()
    if local_port:
        print(f"warning: local port {local_port} is in use — using {lport} instead", file=sys.stderr)
    return lport


def _announce_tunnel(lport: int, node: str, port: int, host: str, local_route: str,
                     share: str, exposer_name: str, share_auto: bool) -> None:
    """Print the LOCAL url and, if asked, the ROUTE / SHARE (chisel) url. A share
    failure NEVER takes the tunnel down (degrades to the local route)."""
    print(f"### LOCAL   http://127.0.0.1:{lport}/v1   "
          f"(tunnel over the SSH session; persists ~{control_persist()})")
    if local_route:
        rurl, rnote = route_url(local_route, lport)
        print(f"### ROUTE   {rurl}   (browser UI: {rurl[:-2]})")
        if rnote:
            print(f"###   {rnote}")
    elif not share:
        print(f"###   browser: open http://127.0.0.1:{lport}/   "
              f"(llama.cpp serves a web UI there; vLLM exposes only /v1)")
    if share and not config.get_bool("share.enabled"):
        rurl, _ = route_url(share, lport)
        print(f"### ROUTE   {rurl}   (team sharing disabled; local tunnel only — "
              f"enable with BOXY_SHARE_ENABLED=1)")
    elif share:
        try:
            from boxy.exposers import get_exposer
            surl, snote = get_exposer(exposer_name).expose(share, lport)
            print(f"### SHARE   {surl}   (browser UI: {surl[:-2]})")
            if snote:
                print(f"###   {snote}")
        except Exception as e:  # the tunnel must never die because the share failed
            rurl, _ = route_url(share, lport)
            if share_auto:
                print(f"### ROUTE   {rurl}   (no team relay reachable — local tunnel "
                      f"only; `boxy generate relay` to set one up, or BOXY_AUTO_SHARE=false)")
            else:
                print(f"warning: share failed — {e}", file=sys.stderr)
                print(f"### ROUTE   {rurl}   (local-only fallback)")
    print(f"###   close: ssh -O cancel -L {lport}:{node}:{port} "
          f"-o ControlPath={control_path()} {host}")


def _laptop_readiness_takeover(host, node, port, log_box, local_port, local_route,
                               share, exposer_name, share_auto, ready_evt, proc) -> None:
    """Own readiness FROM THE LAPTOP so an old cluster boxy's blocked probe can't
    stall us. Open the tunnel to the compute node's port immediately, then confirm
    the server via `http://127.0.0.1:<lport>/health` THROUGH that tunnel (the
    unauthenticated readiness endpoint, checked as localhost from the laptop — the
    port it reaches is the serving node's; falls back to /v1/models) or, as a
    shared-FS fallback, the engine's "server is up" line in the job log (grepped
    over the ssh master; `log_box[0]` is filled in by the stream reader as the
    remote prints it). On readiness, print READY + the tunnel/chisel urls and stop
    the remote loop. The JOB keeps running regardless."""
    import time

    from boxy import readiness

    lport = _pick_local_port(local_port, port)
    if add_forward(host, lport, node, port) != 0:
        lport = 0  # couldn't forward; fall back to log-only readiness

    for _ in range(100000):
        if ready_evt.is_set():
            return
        ready = bool(lport) and readiness.probe_once(f"http://127.0.0.1:{lport}", timeout=3) is not None
        if not ready and log_box[0]:
            rc, out = ssh_capture(
                host, f"grep -Eq 'Application startup complete|server is listening|Uvicorn running on' "
                      f"{shlex.quote(log_box[0])} 2>/dev/null && echo BOXY_READY || true", timeout=15)
            ready = rc == 0 and "BOXY_READY" in out
        if ready:
            ready_evt.set()
            if lport:
                print(f"\n### READY   http://127.0.0.1:{lport}/v1   "
                      f"(confirmed via localhost:{lport}/health through the tunnel)")
                _announce_tunnel(lport, node, port, host, local_route, share, exposer_name, share_auto)
            else:
                print(f"\n### READY   http://{node}:{port}/v1   (server is up per the job log; "
                      f"the login node couldn't forward a tunnel — tunnel manually: "
                      f"ssh -L {port}:{node}:{port} {host})")
            try:
                proc.terminate()  # stop the old cluster boxy's "still waiting" loop; the JOB keeps running
            except Exception:  # noqa: BLE001
                pass
            return
        time.sleep(5)


def run_remote(host: str, raw_argv: list[str], tunnel_ready: bool = False,
               local_port: int | None = None, local_route: str = "",
               share: str = "", exposer_name: str = "relay", share_auto: bool = False) -> int:
    """Run the user's boxy command on `host`, streaming output live. With
    `tunnel_ready`, watch for the '### READY http://node:port' banner and
    forward that endpoint back, then print the local URL — the model is reachable
    on the laptop as http://127.0.0.1:<port>/v1. `local_port` pins the LOCAL side
    (a stable URL, e.g. `boxy open --port 8080`); default reuses the remote port
    when free, else picks any free port. `local_route` (e.g. `--route nemotron`)
    also prints a friendly `http://<name>.localhost:<port>/` URL — no DNS.
    `share` (e.g. `--share nemotron`) hands the live tunnel to the pluggable
    exposer (default: the OpenShift relay) and prints an everyone-URL; a share
    failure NEVER takes the tunnel down (degrades to the Tier-1 route print)."""
    rc = ensure_master(host)
    if rc != 0:
        print(f"boxy: could not open an SSH session to {host} (rc {rc}) — check the host, "
              f"your VPN, and that you completed the OTP/YubiKey prompt", file=sys.stderr)
        return rc
    argv = remote_argv(raw_argv)
    # carry the laptop's site/interceptor CA to the cluster over the same master, so
    # the remote pull trusts what the laptop trusts (compute nodes often lack the
    # site CA even when the laptop has it). Best-effort; never blocks the command.
    remote_ca = propagate_ca(host)
    if remote_ca:
        print(f"### CA      copied your site CA -> {host}:{REMOTE_CA_PATH}  (remote SSL_CERT_FILE)")
    # forward the laptop's proxy so the cluster job's image/model pulls reach it
    # even if the login node's own env doesn't have it (turnkey --proxy).
    proxy_env = remote_proxy_env()
    if proxy_env.get("https_proxy") or proxy_env.get("http_proxy"):
        print(f"### Proxy   forwarding {proxy_env.get('https_proxy') or proxy_env.get('http_proxy')} "
              f"-> {host} (job image/model pulls)")
    # label what follows: everything below runs the CLUSTER's boxy install (keep
    # it as current as the local one: git pull + pip install -e on the login node).
    print(f"### Remote  {host}  $ boxy {shlex.join(argv)}")
    cmd = [ssh_bin(), "-o", f"ControlPath={control_path()}", host,
           _remote_command(argv, remote_ca, proxy_env)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tunneled: set[tuple[str, int]] = set()
    stale = False
    import threading

    ready_evt = threading.Event()
    takeover_started = False
    log_box = [""]  # the job-log path, filled in as the remote prints it (shared with the takeover thread)
    assert proc.stdout is not None
    for line in proc.stdout:
        if NOISE_RE.search(line):
            continue  # drop login-node rootless-podman noise (see NOISE_RE)
        stale = stale or bool(STALE_RE.search(line))
        # Laptop-side readiness takeover: the moment the remote serve names its
        # compute-node endpoint (WAITING_RE, printed BEFORE readiness), start
        # confirming readiness OURSELVES over the tunnel (localhost/health) + the
        # shared-FS log — so an old cluster boxy that loops "still waiting" forever
        # can't stop us reporting READY + the chisel url.
        if tunnel_ready:
            lm = LOG_RE.search(line)
            if lm:
                log_box[0] = lm.group(1)  # for the fallback; the remote prints this in its wait lines
        if tunnel_ready and not takeover_started:
            rm = READY_RE.search(line)
            if rm and (rm.group(1), int(rm.group(2))) not in tunneled:
                # the remote ALREADY confirmed readiness (fresh READY / ALREADY
                # SERVING reconnect) — forward + announce synchronously, no poll.
                node, port = rm.group(1), int(rm.group(2))
                tunneled.add((node, port))
                print(line, end="")
                lport = _pick_local_port(local_port, port)
                if add_forward(host, lport, node, port) == 0:
                    _announce_tunnel(lport, node, port, host, local_route, share, exposer_name, share_auto)
                else:
                    print(f"warning: could not forward local port {lport} (in use?) — "
                          f"tunnel manually: ssh -L {lport}:{node}:{port} {host}", file=sys.stderr)
                continue
            wm = WAITING_RE.search(line)
            if wm and (wm.group(1), int(wm.group(2))) not in tunneled:
                # the remote named the endpoint but is NOT ready yet and may loop
                # "still waiting" forever (its own probe blocked) — take readiness
                # over ourselves from the laptop.
                node, port = wm.group(1), int(wm.group(2))
                tunneled.add((node, port))
                takeover_started = True
                print(line, end="")
                threading.Thread(
                    target=_laptop_readiness_takeover,
                    args=(host, node, port, log_box, local_port, local_route,
                          share, exposer_name, share_auto, ready_evt, proc),
                    daemon=True).start()
                continue
        # once WE own readiness, swallow the remote loop's repetitive spam
        if takeover_started and STILLWAIT_RE.search(line):
            continue
        print(line, end="")
    rc = proc.wait()
    if ready_evt.is_set():
        return 0  # we reported READY + the tunnel/chisel; the job keeps running
    if rc != 0 and stale:
        print(f"boxy: hint: {host} rejected a command this boxy knows — the CLUSTER's boxy "
              f"install is older than yours. Update it there (git pull in the boxy checkout, "
              f"then pip install -e .) and rerun.", file=sys.stderr)
    return rc
