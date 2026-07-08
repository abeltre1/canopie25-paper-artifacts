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
import subprocess
import sys

ENV_HOST = "BOXY_SSH_HOST"          # set once in your shell profile -> "from anywhere"
ENV_ACTIVE = "BOXY_REMOTE_ACTIVE"   # recursion guard: set on the remote side
ENV_SSH_BIN = "BOXY_SSH"            # override the ssh binary (tests: a local shim)
ENV_REMOTE_CMD = "BOXY_REMOTE_COMMAND"  # remote boxy spelling (default: "boxy")

CONTROL_PERSIST = "4h"  # one OTP+touch buys this much multiplexed access

# tunnel-worthy endpoint banners from the remote serve: a fresh READY, or an
# ALREADY SERVING reconnect (rerunning the same model finds the live job).
READY_RE = re.compile(r"###\s+(?:READY|ALREADY SERVING)\s+http://([^:/\s]+):(\d+)")

# the remote boxy rejecting a subcommand/flag the local one just sent means the
# CLUSTER's install is older than the laptop's (field report: `boxy logs --ssh
# eldorado` -> "invalid choice: 'logs'") — say so instead of a bare usage error.
STALE_RE = re.compile(r"invalid choice: '[^']+'|unrecognized arguments:")


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
        "-o", f"ControlPersist={CONTROL_PERSIST}",
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
          f"the session is then reused for {CONTROL_PERSIST} with no re-prompts)")
    # NO capture: the OTP prompt and the YubiKey touch notification need the TTY.
    return subprocess.run([ssh_bin(), *_base_opts(), host, "true"]).returncode


def remote_argv(raw_argv: list[str]) -> list[str]:
    """The exact command the user typed, minus the remote-targeting flags (the
    remote side must run it LOCALLY: recursion is also belt-and-suspenders
    blocked by BOXY_REMOTE_ACTIVE)."""
    out: list[str] = []
    skip = False
    for tok in raw_argv:
        if skip:
            skip = False
            continue
        if tok == "--ssh":
            skip = True
            continue
        if tok.startswith("--ssh="):
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


def add_forward(host: str, local_port: int, remote_host: str, remote_port: int) -> int:
    """Add a port forward ON THE LIVE MASTER — no new connection, no re-auth.
    The forward persists as long as the master does (ControlPersist), so it
    outlives this boxy process; cancel with `ssh -O cancel -L ...`."""
    spec = f"{local_port}:{remote_host}:{remote_port}"
    return subprocess.run([ssh_bin(), "-O", "forward", "-L", spec,
                           "-o", f"ControlPath={control_path()}", host],
                          capture_output=True, text=True).returncode


def run_remote(host: str, raw_argv: list[str], tunnel_ready: bool = False) -> int:
    """Run the user's boxy command on `host`, streaming output live. With
    `tunnel_ready`, watch for the '### READY http://node:port' banner and
    auto-forward that endpoint to the same local port, then print the local URL
    — the model is reachable on the laptop as http://127.0.0.1:<port>/v1."""
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
        print(line, end="")
        stale = stale or bool(STALE_RE.search(line))
        if tunnel_ready:
            m = READY_RE.search(line)
            if m and (m.group(1), int(m.group(2))) not in tunneled:
                node, port = m.group(1), int(m.group(2))
                tunneled.add((node, port))
                if add_forward(host, port, node, port) == 0:
                    print(f"### LOCAL   http://127.0.0.1:{port}/v1   "
                          f"(tunnel over the SSH session; persists ~{CONTROL_PERSIST})")
                    print(f"###   close: ssh -O cancel -L {port}:{node}:{port} "
                          f"-o ControlPath={control_path()} {host}")
                else:
                    print(f"warning: could not forward local port {port} (in use?) — "
                          f"tunnel manually: ssh -L {port}:{node}:{port} {host}", file=sys.stderr)
    rc = proc.wait()
    if rc != 0 and stale:
        print(f"boxy: hint: {host} rejected a command this boxy knows — the CLUSTER's boxy "
              f"install is older than yours. Update it there (git pull in the boxy checkout, "
              f"then pip install -e .) and rerun.", file=sys.stderr)
    return rc
