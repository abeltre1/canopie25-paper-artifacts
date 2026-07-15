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


# ---- laptop CA -> cluster propagation (--ssh from-anywhere) --------------------


def test_remote_command_injects_ssl_cert_file():
    cmd = remote._remote_command(["serve", "m"], remote.REMOTE_CA_PATH)
    assert f"SSL_CERT_FILE={remote.REMOTE_CA_PATH}" in cmd
    assert "SSL_CERT_FILE" not in remote._remote_command(["serve", "m"], None)


def test_remote_command_is_unbuffered():
    # the cluster boxy's stdout is a pipe over ssh -> Python block-buffers it and
    # the readiness/progress lines don't stream. PYTHONUNBUFFERED forces live output.
    assert "PYTHONUNBUFFERED=1" in remote._remote_command(["serve", "m"], None)


# ---- proxy forwarding (--ssh from-anywhere) -----------------------------------


def test_remote_command_injects_proxy_env():
    cmd = remote._remote_command(["serve", "m"], None, {"https_proxy": "http://p:80"})
    assert "https_proxy=http://p:80" in cmd
    assert "http_proxy" not in remote._remote_command(["serve", "m"], None, {})


def test_remote_proxy_env_prefers_config_then_ambient(monkeypatch):
    monkeypatch.delenv(remote.ENV_NO_PROXY_PROP, raising=False)
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("BOXY_PROXY", "http://cfg:8080")           # config wins
    assert remote.remote_proxy_env()["https_proxy"] == "http://cfg:8080"
    monkeypatch.delenv("BOXY_PROXY", raising=False)
    monkeypatch.setenv("https_proxy", "http://ambient:3128")      # else ambient env
    assert remote.remote_proxy_env()["https_proxy"] == "http://ambient:3128"


def test_remote_proxy_env_opt_out(monkeypatch):
    monkeypatch.setenv(remote.ENV_NO_PROXY_PROP, "1")
    monkeypatch.setenv("BOXY_PROXY", "http://cfg:8080")
    assert remote.remote_proxy_env() == {}


def test_resolve_proxy_flag_beats_config(monkeypatch):
    import argparse

    from boxy import cli

    monkeypatch.setenv("BOXY_PROXY", "http://cfg:8080")
    assert cli._resolve_proxy(argparse.Namespace(proxy="http://flag:80")) == "http://flag:80"
    assert cli._resolve_proxy(argparse.Namespace(proxy=None)) == "http://cfg:8080"


def test_propagate_ca_copies_site_ca_over_the_master(shim, tmp_path, monkeypatch):
    # $HOME points at tmp so the "remote" `cat >` lands in a scratch tree, not the
    # runner's real home; propagation is opt-in (conftest disables it by default).
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(remote.ENV_NO_CA_PROP, raising=False)
    ca = tmp_path / "new-nix.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nSNLROOT\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    path = remote.propagate_ca("user@login")
    assert path == remote.REMOTE_CA_PATH
    dst = tmp_path / ".local/share/boxy/store/laptop-ca.crt"
    assert "SNLROOT" in dst.read_text()                       # landed on the (fake) cluster $HOME
    log = shim.read_text()
    assert "mkdir -p" in log and "laptop-ca.crt" in log       # copied over the master, not scp


def test_propagate_ca_skips_disabled_missing_and_boxy_merged(shim, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("boxy.ramalama_shim._ca_merge_kind", "")   # no leaked provenance
    ca = tmp_path / "site.crt"
    ca.write_text("X")
    # opt-out wins even with a real site CA present
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv(remote.ENV_NO_CA_PROP, "1")
    assert remote.propagate_ca("h") is None
    # nothing to send when SSL_CERT_FILE is unset
    monkeypatch.delenv(remote.ENV_NO_CA_PROP, raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert remote.propagate_ca("h") is None
    # boxy's OWN merged bundle is never propagated (it carries the laptop OS store,
    # useless to the cluster; only a genuine site CA is worth sending)
    merged = tmp_path / "ca-merged.crt"
    merged.write_text("Y")
    monkeypatch.setenv("SSL_CERT_FILE", str(merged))
    assert remote.propagate_ca("h") is None


def test_run_remote_propagates_ca_and_injects_env(shim, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(remote.ENV_NO_CA_PROP, raising=False)
    ca = tmp_path / "new-nix.crt"
    ca.write_text("SNLROOT")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv(remote.ENV_REMOTE_CMD, "true")         # remote boxy no-op
    rc = remote.run_remote("user@login", ["list"])
    out = capfd.readouterr().out
    assert rc == 0 and "### CA" in out and "laptop-ca.crt" in out
    # the delegated boxy command carried SSL_CERT_FILE=$HOME/.../laptop-ca.crt
    lines = shim.read_text().splitlines()
    assert any("SSL_CERT_FILE=" in ln and "laptop-ca.crt" in ln for ln in lines)


def test_run_remote_without_site_ca_does_not_propagate(shim, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(remote.ENV_NO_CA_PROP, raising=False)
    monkeypatch.setattr("boxy.ramalama_shim._ca_merge_kind", "")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD, "true")
    rc = remote.run_remote("user@login", ["list"])
    out = capfd.readouterr().out
    assert rc == 0 and "### CA" not in out
    assert not any("SSL_CERT_FILE=" in ln for ln in shim.read_text().splitlines())


def test_run_remote_filters_login_node_podman_noise(shim, capfd, monkeypatch):
    # `boxy list --ssh` runs the CLUSTER's (possibly OLD) boxy; on a login node
    # rootless podman spews runtime-dir noise. Filter it laptop-side so it's gone
    # regardless of the remote boxy version — but keep the real output.
    monkeypatch.setenv(
        remote.ENV_REMOTE_CMD,
        'printf \'%s\\n\' '
        '\'time="2026-07-08T11:37:40-06:00" level=warning msg="Failed to get rootless '
        'runtime dir for DefaultAPIAddress: lstat /run/user/140425: no such file or directory"\' '
        '\'Error: creating events dirs: mkdir /run/user/140425: permission denied\' '
        '\'scheduler jobs:\' '
        '\'  boxy-x  flux job f2a  RUNNING  http://eldo1290:8090/v1\'')
    rc = remote.run_remote("user@login1", ["list"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "Failed to get rootless runtime dir" not in out      # noise gone
    assert "creating events dirs" not in out
    assert "scheduler jobs:" in out and "RUNNING" in out         # real output kept


def test_remote_argv_strips_route_so_older_cluster_boxy_still_runs():
    # `--route` is a LAPTOP-side tunnel name consumed by run_remote; the cluster's
    # `boxy open NAME` must never receive it (older installs don't know it).
    raw = ["open", "NAME", "--ssh", "u@h", "--route", "llama"]
    assert remote.remote_argv(raw) == ["open", "NAME"]
    assert remote.remote_argv(["open", "NAME", "--route=llama"]) == ["open", "NAME"]


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


def test_open_tunnels_to_a_free_local_port(shim, capfd, monkeypatch):
    # `boxy open` prints a READY banner cluster-side; the laptop forwards it to a
    # FREE local port (so a leftover forward on the model's port never blocks it)
    # and prints the browser URL.
    monkeypatch.setattr(remote, "_local_port_free", lambda p: False)   # 8090 "taken"
    monkeypatch.setattr(remote, "_free_local_port", lambda: 54321)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://eldo1003:8090/v1   (model: m)" ; :')
    rc = remote.run_remote("user@login1", ["open", "m"], tunnel_ready=True)
    assert rc == 0
    out = capfd.readouterr().out
    assert "### LOCAL   http://127.0.0.1:54321/v1" in out        # remapped, not 8090
    assert "browser: open http://127.0.0.1:54321/" in out       # browser hint
    assert "CTL forward -L 54321:eldo1003:8090" in shim.read_text()


def test_open_pins_a_custom_local_port(shim, capfd, monkeypatch):
    # --port gives a STABLE URL: run_remote forwards the user's chosen local port.
    monkeypatch.setattr(remote, "_local_port_free", lambda p: True)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://eldo1003:8090/v1   (model: m)" ; :')
    rc = remote.run_remote("user@login1", ["open", "m"], tunnel_ready=True, local_port=8080)
    assert rc == 0
    out = capfd.readouterr().out
    assert "### LOCAL   http://127.0.0.1:8080/v1" in out
    assert "CTL forward -L 8080:eldo1003:8090" in shim.read_text()


def test_route_url_bare_name_gets_localhost():
    # SPEC §8b Tier 1: a bare name becomes <name>.localhost -> 127.0.0.1 in every
    # browser with zero DNS setup (RFC 6761). The tunnel still binds loopback.
    url, note = remote.route_url("nemotron", 8090)
    assert url == "http://nemotron.localhost:8090/v1"
    assert "localhost" in note and "RFC 6761" in note


def test_route_url_strips_scheme_and_path_and_honors_default_ports():
    assert remote.route_url("https://chat/foo", 443)[0] == "http://chat.localhost/v1"
    assert remote.route_url("svc", 80)[0] == "http://svc.localhost/v1"


def test_route_url_dotted_name_used_verbatim_with_hosts_note():
    url, note = remote.route_url("chat.example.com", 8090)
    assert url == "http://chat.example.com:8090/v1"
    assert "/etc/hosts" in note


def test_open_route_prints_friendly_localhost_url(shim, capfd, monkeypatch):
    # `boxy open --route nemotron --ssh ...` forwards the endpoint AND prints a
    # friendly http://nemotron.localhost:PORT/ URL (no DNS) alongside ### LOCAL.
    monkeypatch.setattr(remote, "_local_port_free", lambda p: True)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://eldo1003:8090/v1   (model: m)" ; :')
    rc = remote.run_remote("user@login1", ["open", "m"], tunnel_ready=True,
                           local_route="nemotron")
    assert rc == 0
    out = capfd.readouterr().out
    assert "### LOCAL   http://127.0.0.1:8090/v1" in out
    assert "### ROUTE   http://nemotron.localhost:8090/v1" in out
    assert "browser UI: http://nemotron.localhost:8090/" in out


def test_control_persist_defaults_to_12h_and_is_overridable(monkeypatch):
    monkeypatch.delenv(remote.ENV_PERSIST, raising=False)
    assert remote.control_persist() == "12h"
    assert "ControlPersist=12h" in remote._base_opts()
    monkeypatch.setenv(remote.ENV_PERSIST, "8h")
    assert remote.control_persist() == "8h"
    assert "ControlPersist=8h" in remote._base_opts()


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
