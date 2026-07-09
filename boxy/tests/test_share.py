"""The share flow end-to-end WITHOUT chisel/oc/a cluster: fake-chisel and
fake-oc shims (the fake-ssh SHIM pattern) drive RelayExposer through expose →
share.json → unshare, and the full `boxy open --ssh --share` path through all
three shims. The strip-list regression comes FIRST — an older cluster boxy must
never see the laptop-only flags (encoded field regression)."""

import json
import os
import time
from pathlib import Path

import pytest

from boxy import remote
from boxy.cli import main
from boxy.exposers import ExposeError
from boxy.exposers import relay as relay_mod
from boxy.exposers.relay import RelayExposer

# ---- the regression that guards everything else --------------------------------


def test_remote_argv_strips_share_flags_so_older_cluster_boxy_still_runs():
    raw = ["open", "NAME", "--ssh", "u@h", "--share", "nemo", "--exposer", "relay"]
    assert remote.remote_argv(raw) == ["open", "NAME"]
    assert remote.remote_argv(["open", "N", "--share=nemo", "--exposer=relay"]) == ["open", "N"]


# ---- shims ----------------------------------------------------------------------

CHISEL_SHIM = r"""#!/bin/bash
# fake chisel: logs argv; `client` stays alive like the real detached client and
# exits promptly on SIGTERM (real chisel handles the signal; a bare `wait` would
# defer it, so trap explicitly).
echo "$*" >> "$CHISEL_LOG"
if [ "$1" = "client" ]; then
  if [ -n "$FAKE_CHISEL_FAIL_ONCE" ] && [ ! -f "$FAKE_CHISEL_FAIL_ONCE" ]; then
    touch "$FAKE_CHISEL_FAIL_ONCE"
    echo "server: Failed to listen: listen tcp 0.0.0.0:31111: bind: address already in use"
    exit 1
  fi
  trap 'exit 0' TERM INT
  echo "Connected (Latency 1ms)"
  sleep 30 &
  wait $!
fi
exit 0
"""

OC_SHIM = r"""#!/bin/bash
# fake oc: logs argv; answers the exact jsonpath queries the exposer makes.
echo "$*" >> "$OC_LOG"
case "$*" in
  *"get route boxy-relay"*)      echo -n "${FAKE_OC_RELAY_HOST:-relay-boxy.apps.x.y}" ;;
  *"get secret boxy-relay"*)     echo -n "$(printf 'boxy:pw' | base64)" ;;
  *"get svc"*)                   echo -n "${FAKE_OC_TAKEN:-}" ;;
  *"get route boxy-share-"*)     echo -n "${FAKE_OC_ADMITTED:-True}" ;;
  apply*)                        cat > "$OC_APPLIED" ;;
  delete*)                       ;;
esac
exit 0
"""


@pytest.fixture
def share_env(tmp_path, monkeypatch):
    """fake chisel + fake oc + isolated jobs dir + fast poll timings."""
    chisel = tmp_path / "chisel"
    chisel.write_text(CHISEL_SHIM)
    chisel.chmod(0o755)
    oc = tmp_path / "oc"
    oc.write_text(OC_SHIM)
    oc.chmod(0o755)
    (tmp_path / "chisel.log").write_text("")
    (tmp_path / "oc.log").write_text("")
    monkeypatch.setenv(relay_mod.ENV_CHISEL, str(chisel))
    monkeypatch.setenv(relay_mod.ENV_OC, str(oc))
    monkeypatch.setenv("CHISEL_LOG", str(tmp_path / "chisel.log"))
    monkeypatch.setenv("OC_LOG", str(tmp_path / "oc.log"))
    monkeypatch.setenv("OC_APPLIED", str(tmp_path / "applied.yaml"))
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv(relay_mod.ENV_RELAY_URL, raising=False)
    monkeypatch.delenv(relay_mod.ENV_RELAY_AUTH, raising=False)
    monkeypatch.setattr(relay_mod, "BIND_GRACE", 0.3)
    monkeypatch.setattr(relay_mod, "ADMIT_TIMEOUT", 1.0)
    monkeypatch.setattr(relay_mod, "ADMIT_POLL", 0.05)
    return tmp_path


def _wait_dead(pid, timeout=3.0):
    """The detached client's parent is this pytest process, so a terminated
    client lingers as a zombie until reaped — reap it (WNOHANG) and treat a
    zombie as dead (production's ps-based liveness does the same)."""
    import subprocess as sp
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        st = sp.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if not st or st.startswith("Z"):
            return True
        time.sleep(0.05)
    return False


# ---- RelayExposer lifecycle -------------------------------------------------------


def test_expose_full_lifecycle_and_unshare(share_env, capsys):
    from boxy import jobs

    url, note = RelayExposer().expose("nemo", 8090)
    assert url == "https://nemo-boxy.apps.x.y/v1"
    assert "boxy unshare nemo" in note
    # the client got the right reverse spec and the credential came via env (not argv)
    chisel_log = (share_env / "chisel.log").read_text()
    assert "client --keepalive 25s --max-retry-count -1 https://relay-boxy.apps.x.y" in chisel_log
    assert "R:0.0.0.0:31" in chisel_log and ":127.0.0.1:8090" in chisel_log
    assert "boxy:pw" not in chisel_log
    # the applied manifest is the golden Service+Route pair
    applied = (share_env / "applied.yaml").read_text()
    assert "boxy-share-nemo" in applied and "nemo-boxy.apps.x.y" in applied
    # durable record, no credential, live pid
    rec = jobs.read_share("nemo")
    assert rec and rec["url"] == "https://nemo-boxy.apps.x.y" and rec["lport"] == 8090
    assert "pw" not in json.dumps(rec)
    os.kill(rec["pid"], 0)                                   # alive (detached)

    # idempotent re-expose: same lport -> reuse, no second client
    url2, note2 = RelayExposer().expose("nemo", 8090)
    assert url2 == url and "already shared" in note2

    rc = main(["unshare", "nemo"])
    out = capsys.readouterr().out
    assert rc == 0 and "unshared nemo" in out
    assert _wait_dead(rec["pid"])                             # client killed
    assert "delete route,svc" in (share_env / "oc.log").read_text()
    assert jobs.read_share("nemo") is None


def test_unshare_unknown_alias_errors(share_env, capsys):
    rc = main(["unshare", "ghost"])
    assert rc == 1
    assert "no share named 'ghost'" in capsys.readouterr().err


def test_expose_retries_on_bind_conflict(share_env, monkeypatch):
    monkeypatch.setenv("FAKE_CHISEL_FAIL_ONCE", str(share_env / "failed-once"))
    url, _ = RelayExposer().expose("retry", 8090)
    assert url.startswith("https://retry-boxy.")
    log = (share_env / "chisel.log").read_text()
    assert log.count("client --keepalive") == 2              # failed bind, re-picked, succeeded
    main(["unshare", "retry"])


def test_expose_rejects_taken_route_host(share_env, monkeypatch):
    monkeypatch.setenv("FAKE_OC_ADMITTED", "False")
    with pytest.raises(ExposeError, match="is taken on this cluster"):
        RelayExposer().expose("stolen", 8090)
    oc_log = (share_env / "oc.log").read_text()
    assert "delete route,svc" in oc_log                       # cleaned up after itself
    from boxy import jobs
    assert jobs.read_share("stolen") is None


def test_expose_without_oc_prints_yaml_and_keeps_client(share_env, monkeypatch, capsys):
    monkeypatch.setenv(relay_mod.ENV_OC, "/nonexistent/oc")
    monkeypatch.setenv(relay_mod.ENV_RELAY_URL, "https://relay-boxy.apps.x.y")
    monkeypatch.setenv(relay_mod.ENV_RELAY_AUTH, "boxy:pw")
    url, note = RelayExposer().expose("noc", 8090)
    out = capsys.readouterr().out
    assert url == "https://noc-boxy.apps.x.y/v1"
    assert "oc unavailable" in note and "apply the YAML above yourself" in note
    assert "kind: Route" in out                               # the manifest, hand-applicable
    main(["unshare", "noc"])


def test_expose_without_chisel_raises_install_hint(share_env, monkeypatch):
    monkeypatch.delenv(relay_mod.ENV_CHISEL, raising=False)
    monkeypatch.setattr(relay_mod.shutil, "which", lambda b: None)
    with pytest.raises(ExposeError, match="brew install chisel"):
        RelayExposer().expose("nochisel", 8090)


# ---- the full CLI path through the fake-ssh shim -----------------------------------

SSH_SHIM = r"""#!/bin/bash
log() { echo "$*" >> "$SHIM_LOG"; }
if [ "$1" = "-O" ]; then op="$2"; shift 2; log "CTL $op $*"
  case "$op" in check) [ -f "$SHIM_STATE" ] && exit 0 || exit 1 ;; *) exit 0 ;; esac
fi
host=""; cmd=()
while [ $# -gt 0 ]; do case "$1" in
  -o) shift 2 ;;
  *) if [ -z "$host" ]; then host="$1"; else cmd+=("$1"); fi; shift ;;
esac; done
log "RUN $host ${cmd[*]}"
if [ "${cmd[*]}" = "true" ]; then touch "$SHIM_STATE"; exit 0; fi
exec bash -c "${cmd[*]}"
"""


def test_open_share_prints_everyone_url(share_env, monkeypatch, capfd):
    ssh = share_env / "ssh"
    ssh.write_text(SSH_SHIM)
    ssh.chmod(0o755)
    monkeypatch.setenv(remote.ENV_SSH_BIN, str(ssh))
    monkeypatch.setenv("SHIM_LOG", str(share_env / "ssh.log"))
    monkeypatch.setenv("SHIM_STATE", str(share_env / "master.up"))
    monkeypatch.delenv(remote.ENV_ACTIVE, raising=False)
    monkeypatch.delenv(remote.ENV_HOST, raising=False)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://eldo1290:8090/v1   (model: m)" ; :')
    monkeypatch.setattr(remote, "_local_port_free", lambda p: True)
    rc = main(["open", "m", "--ssh", "user@login1", "--share", "nemo"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "### LOCAL   http://127.0.0.1:8090/v1" in out
    assert "### SHARE   https://nemo-boxy.apps.x.y/v1" in out
    # the CLUSTER side never saw the laptop-only flags
    run_line = [ln for ln in (share_env / "ssh.log").read_text().splitlines()
                if ln.startswith("RUN")][0]
    assert "--share" not in run_line and "--exposer" not in run_line
    main(["unshare", "nemo"])


def test_open_share_degrades_when_relay_missing(share_env, monkeypatch, capfd):
    ssh = share_env / "ssh"
    ssh.write_text(SSH_SHIM)
    ssh.chmod(0o755)
    monkeypatch.setenv(remote.ENV_SSH_BIN, str(ssh))
    monkeypatch.setenv("SHIM_LOG", str(share_env / "ssh.log"))
    monkeypatch.setenv("SHIM_STATE", str(share_env / "master.up"))
    monkeypatch.delenv(remote.ENV_ACTIVE, raising=False)
    monkeypatch.delenv(remote.ENV_HOST, raising=False)
    monkeypatch.setenv(remote.ENV_REMOTE_CMD,
                       'echo "### READY  http://eldo1290:8090/v1   (model: m)" ; :')
    monkeypatch.setattr(remote, "_local_port_free", lambda p: True)
    monkeypatch.setenv(relay_mod.ENV_CHISEL, "/nonexistent/chisel")
    rc = main(["open", "m", "--ssh", "user@login1", "--share", "nemo"])
    cap = capfd.readouterr()
    assert rc == 0                                            # tunnel survives the failed share
    assert "### LOCAL   http://127.0.0.1:8090/v1" in cap.out
    assert "### ROUTE   http://nemo.localhost:8090/v1" in cap.out
    assert "share failed" in cap.err and "brew install chisel" in cap.err


def test_open_share_without_ssh_is_a_usage_error(share_env, monkeypatch, capsys):
    monkeypatch.delenv(remote.ENV_HOST, raising=False)
    from boxy import cli, jobs
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: True)
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "9"})
    jobs.write_endpoint("boxy-m", 8090)
    rc = main(["open", "boxy-m", "--share", "nemo"])
    assert rc == 2
    assert "--share needs the laptop tunnel" in capsys.readouterr().err


# ---- boxy list shows shares ---------------------------------------------------------


def test_list_shows_share_liveness(share_env, monkeypatch, capsys):
    from boxy import jobs

    RelayExposer().expose("nemo", 8090)
    rec = jobs.read_share("nemo")
    rc = main(["list", "--runtime", "docker", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "shares (everyone-URLs" in out
    assert "nemo  https://nemo-boxy.apps.x.y/v1  LIVE" in out
    # kill the client -> DEAD with a next step
    os.kill(rec["pid"], 15)
    _wait_dead(rec["pid"])
    main(["list", "--runtime", "docker", "--dryrun"])
    assert "DEAD" in capsys.readouterr().out
    main(["unshare", "nemo"])


# ---- hygiene: the detached fake client must never leak past the suite --------------


@pytest.fixture(autouse=True)
def _reap_stray_clients(share_env):
    yield
    for rec_file in Path(share_env / "jobs").glob("*.share.json"):
        try:
            pid = json.loads(rec_file.read_text()).get("pid")
            if pid:
                os.kill(int(pid), 9)
        except (OSError, ValueError):
            pass
