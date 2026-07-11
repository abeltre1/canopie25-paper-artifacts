"""Submit from ANYWHERE: run boxy against a remote cluster over SSH.

    laptop$ boxy serve MODEL --scheduler slurm --gpus 4 --ssh user@hops-login1

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

ENV_HOST = "BOXY_SSH_HOST"          # set once in your shell profile -> "from anywhere"
ENV_ACTIVE = "BOXY_REMOTE_ACTIVE"   # recursion guard: set on the remote side
ENV_SSH_BIN = "BOXY_SSH"            # override the ssh binary (tests: a local shim)
ENV_REMOTE_CMD = "BOXY_REMOTE_COMMAND"  # remote boxy spelling (default: "boxy")
ENV_PERSIST = "BOXY_SSH_PERSIST"    # how long the master + its tunnels live idle

DEFAULT_PERSIST = "12h"  # one OTP+touch buys this much multiplexed access


def control_persist() -> str:
    """Idle lifetime of the multiplexed SSH master (and every tunnel riding it),
    so one OTP+YubiKey keeps working for a full workday. Override per your site's
    session cap: BOXY_SSH_PERSIST=8h (accepts OpenSSH time formats: 30m, 12h, ...)."""
    return os.environ.get(ENV_PERSIST, DEFAULT_PERSIST)

# tunnel-worthy endpoint banners from the remote serve: a fresh READY, or an
# ALREADY SERVING reconnect (rerunning the same model finds the live job).
READY_RE = re.compile(r"###\s+(?:READY|ALREADY SERVING)\s+http://([^:/\s]+):(\d+)")

# the remote boxy rejecting a subcommand/flag the local one just sent means the
# CLUSTER's install is older than the laptop's (field report: `boxy logs --ssh
# eldorado` -> "invalid choice: 'logs'") — say so instead of a bare usage error.
STALE_RE = re.compile(r"invalid choice: '[^']+'|unrecognized arguments:")

# rootless podman on an HPC login node has no /run/user/$UID (no user systemd
# session), so `podman ps` spews 'Failed to get rootless runtime dir' + 'creating
# events dirs: ... permission denied' before failing. That noise is meaningless to
# the user (the real instances are the scheduler jobs) — filter it from the SSH
# stream on the LAPTOP side so it's gone regardless of the cluster's boxy version
# (field report: boxy list --ssh eldorado). Kept tight so real errors still show.
NOISE_RE = re.compile(
    r"Failed to get rootless runtime dir"
    r"|creating events dirs:.*(?:/run/user/|permission denied)"
    r"|rootless.*(?:/run/user/\d+).*(?:no such file|permission denied)")


def ssh_bin() -> str:
    return os.environ.get(ENV_SSH_BIN, "ssh")


def control_path() -> str:
    # %C = hash(local host, remote host, port, user): short + collision-free
    # (macOS caps unix-socket paths at ~104 chars, so never build long literals).
    return "~/.ssh/boxy-cm-%C"


def _base_opts() -> list[str]:
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={control_path()}",
        "-o", f"ControlPersist={control_persist()}",
        "-o", "ServerAliveInterval=30",
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
        if tok.startswith(("--ssh=", "--route=", "--share=", "--exposer=")):
            continue
        out.append(tok)
    return out


def _remote_command(argv: list[str]) -> str:
    """The command line run on the login node. `bash -lc` gives a LOGIN shell so
    the user's PATH/venv/modules load — the remote `boxy` must be installed there
    (spelling overridable: BOXY_REMOTE_COMMAND='source ~/venv/bin/activate && boxy')."""
    boxy_cmd = os.environ.get(ENV_REMOTE_CMD, "boxy")
    inner = f"{ENV_ACTIVE}=1 {boxy_cmd} {shlex.join(argv)}"
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


def run_remote(host: str, raw_argv: list[str], tunnel_ready: bool = False,
               local_port: int | None = None, local_route: str = "",
               share: str = "", exposer_name: str = "relay") -> int:
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
    # label what follows: everything below runs the CLUSTER's boxy install (keep
    # it as current as the local one: git pull + pip install -e on the login node).
    print(f"### Remote  {host}  $ boxy {shlex.join(argv)}")
    cmd = [ssh_bin(), "-o", f"ControlPath={control_path()}", host, _remote_command(argv)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tunneled: set[tuple[str, int]] = set()
    stale = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if NOISE_RE.search(line):
            continue  # drop login-node rootless-podman noise (see NOISE_RE)
        print(line, end="")
        stale = stale or bool(STALE_RE.search(line))
        if tunnel_ready:
            m = READY_RE.search(line)
            if m and (m.group(1), int(m.group(2))) not in tunneled:
                node, port = m.group(1), int(m.group(2))
                tunneled.add((node, port))
                # a user-pinned --port wins (stable URL); else reuse the remote
                # port when free, else pick any free one so a leftover forward on
                # that port (field report: stale podman gvproxy on 8090) never
                # blocks the tunnel.
                want = local_port or port
                if _local_port_free(want):
                    lport = want
                else:
                    lport = _free_local_port()
                    if local_port:
                        print(f"warning: local port {local_port} is in use — using {lport} instead",
                              file=sys.stderr)
                if add_forward(host, lport, node, port) == 0:
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
                    if share:
                        try:
                            from boxy.exposers import get_exposer
                            surl, snote = get_exposer(exposer_name).expose(share, lport)
                            print(f"### SHARE   {surl}   (browser UI: {surl[:-2]})")
                            if snote:
                                print(f"###   {snote}")
                        except Exception as e:  # the tunnel must never die because the share failed
                            print(f"warning: share failed — {e}", file=sys.stderr)
                            rurl, _ = route_url(share, lport)
                            print(f"### ROUTE   {rurl}   (local-only fallback)")
                    print(f"###   close: ssh -O cancel -L {lport}:{node}:{port} "
                          f"-o ControlPath={control_path()} {host}")
                else:
                    print(f"warning: could not forward local port {lport} (in use?) — "
                          f"tunnel manually: ssh -L {lport}:{node}:{port} {host}", file=sys.stderr)
    rc = proc.wait()
    if rc != 0 and stale:
        print(f"boxy: hint: {host} rejected a command this boxy knows — the CLUSTER's boxy "
              f"install is older than yours. Update it there (git pull in the boxy checkout, "
              f"then pip install -e .) and rerun.", file=sys.stderr)
    return rc
