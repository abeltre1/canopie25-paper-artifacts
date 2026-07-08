"""Submit-from-anywhere (--ssh): the CLI re-runs the same command on a cluster
login node over ONE multiplexed SSH session and tunnels the READY endpoint back.

Tested without a cluster via a fake `ssh` shim (BOXY_SSH): it records every ssh
invocation and EXECUTES the remote command locally, so the delegation is
exercised end-to-end against the real CLI.
"""

import sys
from pathlib import Path

import pytest

from boxy import remote
from boxy.cli import main

SRC = str(Path(__file__).parent.parent / "src")

SHIM = r"""#!/bin/bash
# fake ssh: understands just enough of OpenSSH's CLI for boxy's remote layer.
log() { echo "$*" >> "$SHIM_LOG"; }
if [ "$1" = "-O" ]; then                    # control ops on the master socket
  op="$2"; shift 2
  log "CTL $op $*"
  case "$op" in
    check)   [ -f "$SHIM_STATE" ] && exit 0 || exit 1 ;;
    forward) exit 0 ;;
    *)       exit 0 ;;
  esac
fi
host=""; cmd=()
while [ $# -gt 0 ]; do                      # strip -o KEY=VAL option pairs
  case "$1" in
    -o) shift 2 ;;
    *)  if [ -z "$host" ]; then host="$1"; else cmd+=("$1"); fi; shift ;;
  esac
done
log "RUN $host ${cmd[*]}"
if [ "${cmd[*]}" = "true" ]; then touch "$SHIM_STATE"; exit 0; fi   # master auth
exec bash -c "${cmd[*]}"                    # run the "remote" command locally
"""


@pytest.fixture
def shim(tmp_path, monkeypatch):
    """Install the fake ssh + env; returns the invocation-log path."""
    ssh = tmp_path / "ssh"
    ssh.write_text(SHIM)
    ssh.chmod(0o755)
    log = tmp_path / "ssh.log"
    log.write_text("")
    monkeypatch.setenv(remote.ENV_SSH_BIN, str(ssh))
    monkeypatch.setenv("SHIM_LOG", str(log))
    monkeypatch.setenv("SHIM_STATE", str(tmp_path / "master.up"))
    monkeypatch.delenv(remote.ENV_ACTIVE, raising=False)
    monkeypatch.delenv(remote.ENV_HOST, raising=False)
    # "remote" boxy = this checkout's CLI, run through the shim's bash -c
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       f"PYTHONPATH={SRC} {sys.executable} -m boxy.cli")
    return log


# ---- pure helpers -------------------------------------------------------------


def test_remote_argv_strips_ssh_flag():
    raw = ["serve", "M", "--ssh", "u@h", "--gpus", "4", "--ssh=v@x", "--dryrun"]
    assert remote.remote_argv(raw) == ["serve", "M", "--gpus", "4", "--dryrun"]


def test_resolve_target_precedence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    loc = tmp_path / "site.toml"
    loc.write_text('[location]\nname = "site"\nscheduler = "slurm"\nremote = "u@from-profile"\n')
    monkeypatch.delenv(remote.ENV_HOST, raising=False)
    # location profile provides the target
    assert remote.resolve_target(SimpleNamespace(ssh=None, location=str(loc))) == "u@from-profile"
    # env beats profile
    monkeypatch.setenv(remote.ENV_HOST, "u@from-env")
    assert remote.resolve_target(SimpleNamespace(ssh=None, location=str(loc))) == "u@from-env"
    # flag beats both
    assert remote.resolve_target(SimpleNamespace(ssh="u@from-flag", location=str(loc))) == "u@from-flag"
    # nothing configured -> run here
    monkeypatch.delenv(remote.ENV_HOST)
    assert remote.resolve_target(SimpleNamespace(ssh=None, location=None)) == ""


# ---- end-to-end through the shim ----------------------------------------------


def test_serve_delegates_over_ssh_and_streams(shim, capfd):
    # The same command the user would type, plus --ssh: it must run REMOTELY
    # (through the shim) and stream the remote boxy's real output back.
    rc = main(["serve", "SomeModel", "--ssh", "user@login1",
               "--scheduler", "slurm", "--gpus", "1", "--dryrun"])
    assert rc == 0
    out = capfd.readouterr().out
    assert "### Remote  user@login1  $ boxy serve" in out   # remote output is labeled
    assert "### Batch script" in out          # the REMOTE boxy's dryrun output
    assert "#SBATCH --gpus-per-node=1" in out
    log = shim.read_text()
    assert "RUN user@login1" in log           # went over ssh
    assert "--ssh" not in log.split("RUN", 1)[1]   # remote cmd has no --ssh (no recursion)
    assert remote.ENV_ACTIVE in log           # recursion guard env set on the remote side


def test_master_established_once_then_reused(shim, capfd):
    main(["list", "--ssh", "user@login1", "--dryrun"])
    main(["list", "--ssh", "user@login1", "--dryrun"])
    log = shim.read_text()
    # first call: check fails -> master auth ("RUN ... true"); second: check passes
    assert log.count("RUN user@login1 true") == 1
    assert log.count("CTL check") == 2


def test_ready_line_triggers_port_forward(shim, capfd, monkeypatch):
    # When the remote serve prints its READY banner, boxy adds a forward on the
    # live master and prints the LOCAL url.
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://cn042:8090/v1   (model: m)" ; :')
    rc = remote.run_remote("user@login1", ["serve", "m.gguf"], tunnel_ready=True)
    assert rc == 0
    out = capfd.readouterr().out
    assert "### READY" in out                          # streamed through
    assert "### LOCAL   http://127.0.0.1:8090/v1" in out
    assert "CTL forward -L 8090:cn042:8090" in shim.read_text()


def test_already_serving_banner_also_tunnels(shim, capfd, monkeypatch):
    # Rerunning the same model reconnects to the live job: the ALREADY SERVING
    # banner must open the tunnel exactly like a fresh READY.
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### ALREADY SERVING  http://cn042:8090/v1   (model: m, slurm job 7)" ; :')
    rc = remote.run_remote("user@login1", ["serve", "m.gguf"], tunnel_ready=True)
    assert rc == 0
    out = capfd.readouterr().out
    assert "### LOCAL   http://127.0.0.1:8090/v1" in out
    assert "CTL forward -L 8090:cn042:8090" in shim.read_text()


def test_recursion_guard_runs_locally(shim, capfd, monkeypatch):
    # On the remote side (BOXY_REMOTE_ACTIVE set) the --ssh flag is inert.
    monkeypatch.setenv(remote.ENV_ACTIVE, "1")
    rc = main(["list", "--ssh", "user@login1", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "RUN" not in shim.read_text()               # no ssh happened
    assert "docker ps" in capfd.readouterr().out       # the LOCAL list ran


def test_already_on_target_host_runs_locally(shim, capfd, monkeypatch):
    import socket

    me = socket.gethostname().split(".")[0]
    rc = main(["list", "--ssh", f"user@{me}.example.gov", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "RUN" not in shim.read_text()               # no ssh to ourselves
    assert "docker ps" in capfd.readouterr().out


def test_env_var_makes_every_command_remote(shim, capfd, monkeypatch):
    # export BOXY_SSH_HOST once -> the SAME commands work from anywhere.
    monkeypatch.setenv(remote.ENV_HOST, "user@login1")
    rc = main(["list", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "RUN user@login1" in shim.read_text()


def test_remote_failure_rc_propagates(shim, capfd, monkeypatch):
    monkeypatch.setenv(remote.ENV_REMOTE_CMD, "exit 3 ; :")
    rc = remote.run_remote("user@login1", ["list"])
    assert rc == 3


def test_stale_remote_boxy_gets_update_hint(shim, capfd, monkeypatch):
    # A cluster whose boxy predates a subcommand (field report: `boxy logs --ssh
    # eldorado` -> "invalid choice: 'logs'") must be called out as STALE — the
    # bare usage error reads like a boxy bug instead of an outdated install.
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       "echo \"boxy: error: argument subcommand: invalid choice: 'logs'\" ; exit 2 ; :")
    rc = remote.run_remote("user@login1", ["logs"])
    assert rc == 2
    captured = capfd.readouterr()
    assert "invalid choice" in captured.out                # remote error streamed through
    assert "older" in captured.err and "git pull" in captured.err


def test_master_auth_failure_is_explained(tmp_path, capfd, monkeypatch):
    # ssh that always fails (wrong host / user aborted the OTP prompt)
    bad = tmp_path / "ssh"
    bad.write_text("#!/bin/bash\nexit 255\n")
    bad.chmod(0o755)
    monkeypatch.setenv(remote.ENV_SSH_BIN, str(bad))
    rc = remote.run_remote("user@nowhere", ["list"])
    assert rc == 255
    err = capfd.readouterr().err
    assert "OTP/YubiKey" in err and "nowhere" in err


def test_local_commands_untouched_without_target(shim, capfd, monkeypatch):
    # No --ssh, no env, no profile: nothing goes near ssh (zero-regression guard).
    rc = main(["list", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "RUN" not in shim.read_text()
    assert "docker ps" in capfd.readouterr().out


def test_ssh_prompts_reach_the_tty():
    # OTP/YubiKey CONTRACT: the master-establishing ssh must run with the user's
    # TTY attached (no capture), or the OTP prompt would vanish. Guard the code
    # shape: ensure_master's auth call must NOT capture stdout/stderr.
    import inspect

    src = inspect.getsource(remote.ensure_master)
    auth_call = src.split("# NO capture", 1)[1]
    assert "capture_output" not in auth_call and "PIPE" not in auth_call
