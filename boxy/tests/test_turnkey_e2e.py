"""Turnkey end-to-end: prove the account fetched from `mywcid` actually reaches
the batch script — the field promise that failed (2026-07) — in BOTH invocation
styles, against a FAKE cluster (bash shims on PATH), never a real scheduler.

Two paths are exercised:

  * on the login node — a real submit (`boxy serve MODEL --scheduler slurm`, no
    --dryrun): a fake `sbatch` writes the endpoint and echoes a job id, so the
    submit/follow loop runs to `### READY`, and we assert the script ON DISK
    carries `#SBATCH --account=<from mywcid>`.

  * from the laptop over `--ssh` — the command is delegated to the CLUSTER's
    boxy (which may predate turnkey), so the account is resolved laptop-side and
    injected as `--account`. We assert the delegated argv carries it and the
    (delegated) boxy's batch script shows it — with the account coming either
    from a locally-known value or from a `mywcid` probe run ON the cluster.
"""

import json
import sys
from pathlib import Path

import pytest

from boxy import readiness, remote
from boxy.cli import main

SRC = str(Path(__file__).parent.parent / "src")

# The real `mywcid` table (see tests/test_site.py): header + dashed rule, data
# rows whose DESCRIPTION opens with a bare numeric id, a 'none' row, and a caps
# note. The account column is fy140001 / fy140252 / fy260064.
REAL_MYWCID = """\
      User    Account                              Description     Parent
---------- ---------- ---------------------------------------- --------------------
     jdoe   fy140001        103732 system software and tools                   nd
     jdoe   fy140252      135101 common computing environment                   nd
     jdoe   fy260064         240928 the genesis project obbba                   nd
     jdoe       none       default account, no job privileges
  The Account could be on Caps too: FY140001
"""

# fake sbatch: log argv, simulate an instant compute-node start by writing the
# endpoint file next to the script (name.sh -> name.endpoint.json), print a job
# id like `sbatch --parsable` does. Lets the real submit/follow loop reach READY.
FAKE_SBATCH = r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"                                   # last positional arg = script path
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
"""

# fake ssh (from tests/test_remote.py): records every invocation, runs the
# "remote" command locally so the delegation is exercised against the real CLI.
FAKE_SSH = r"""#!/bin/bash
log() { echo "$*" >> "$SHIM_LOG"; }
if [ "$1" = "-O" ]; then
  op="$2"; shift 2
  log "CTL $op $*"
  case "$op" in
    check)   [ -f "$SHIM_STATE" ] && exit 0 || exit 1 ;;
    forward) exit 0 ;;
    *)       exit 0 ;;
  esac
fi
host=""; cmd=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    *)  if [ -z "$host" ]; then host="$1"; else cmd+=("$1"); fi; shift ;;
  esac
done
log "RUN $host ${cmd[*]}"
if [ "${cmd[*]}" = "true" ]; then touch "$SHIM_STATE"; exit 0; fi
exec bash -c "${cmd[*]}"
"""

MODEL = "hf://meta-llama/Llama-3.1-8B-Instruct"


def _shim(directory: Path, name: str, body: str) -> Path:
    p = directory / name
    p.write_text(body)
    p.chmod(0o755)
    return p


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    """A fake cluster: mywcid + sbatch on PATH, an isolated jobs dir, and a
    non-blocking readiness probe. Account injection over --ssh is off by default
    in the suite (conftest), so tests that need it re-enable it explicitly."""
    binp = tmp_path / "bin"
    binp.mkdir()
    _shim(binp, "mywcid", "#!/bin/bash\ncat <<'EOF'\n" + REAL_MYWCID + "EOF\n")
    sbatch_log = tmp_path / "sbatch.log"
    sbatch_log.write_text("")
    _shim(binp, "sbatch", FAKE_SBATCH)
    monkeypatch.setenv("SBATCH_LOG", str(sbatch_log))
    monkeypatch.setenv("PATH", f"{binp}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    # the compute node never really starts a server here — report ready at once
    monkeypatch.setattr(readiness, "wait_ready", lambda *a, **k: "llama-3.1-8b")
    return {"bin": binp, "jobs": tmp_path / "jobs", "sbatch_log": sbatch_log,
            "tmp": tmp_path, "monkeypatch": monkeypatch}


def _the_script(jobs_dir: Path) -> str:
    scripts = list(jobs_dir.glob("*.sh"))
    assert len(scripts) == 1, f"expected one batch script, found {scripts}"
    return scripts[0].read_text()


# ---- login-node: a REAL submit puts the mywcid account in the script on disk ----


def test_login_node_submit_writes_account_into_the_script(cluster, capfd):
    rc = main(["serve", MODEL, "--scheduler", "slurm"])
    out = capfd.readouterr().out
    assert rc == 0
    # the turnkey promise: mywcid -> the batch script the scheduler actually ran
    script = _the_script(cluster["jobs"])
    assert "#SBATCH --account=fy140001" in script     # first account from the table
    assert "#SBATCH --gpus-per-node=1" in script      # 8B model card -> 1 GPU
    # ...and it was really submitted + followed to READY
    assert "--parsable" in cluster["sbatch_log"].read_text()
    assert "auto: account: fy140001 (via mywcid" in out
    assert "### READY" in out
    # the endpoint record is discoverable (boxy list would show it)
    rec = json.loads((cluster["jobs"] / [p.name for p in cluster["jobs"].glob("*.json")
                                         if not p.name.endswith(".endpoint.json")][0]).read_text())
    assert rec["job"] == "12345" and rec["scheduler"] == "slurm"


def test_login_node_flux_bank_and_single_queue(cluster, capfd):
    # Flux spells account `--bank` and takes exactly ONE queue: a Slurm-style
    # comma partition is trimmed with a warning (the field failure).
    rc = main(["serve", MODEL, "--scheduler", "flux", "--partition", "short,batch",
               "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "# flux: --bank=fy140001" in cap.out        # account -> flux bank
    assert "# flux: --queue=short" in cap.out          # one queue only
    assert "--queue=short,batch" not in cap.out
    assert "ONE queue" in cap.err                       # the guard warned


# ---- from the laptop over --ssh: the account is injected into the delegated cmd --


@pytest.fixture
def ssh(cluster):
    """Layer a fake ssh over the fake cluster and turn remote-account injection
    back on (conftest disables it for the suite at large)."""
    mp = cluster["monkeypatch"]
    sshp = _shim(cluster["bin"], "ssh", FAKE_SSH)
    log = cluster["tmp"] / "ssh.log"
    log.write_text("")
    mp.setenv(remote.ENV_SSH_BIN, str(sshp))
    mp.setenv("SHIM_LOG", str(log))
    mp.setenv("SHIM_STATE", str(cluster["tmp"] / "master.up"))
    mp.delenv(remote.ENV_ACTIVE, raising=False)
    mp.delenv(remote.ENV_HOST, raising=False)
    mp.setenv(remote.ENV_REMOTE_CMD, f"PYTHONPATH={SRC} {sys.executable} -m boxy.cli")
    mp.delenv("BOXY_NO_REMOTE_ACCOUNT", raising=False)   # opt back in
    cluster["ssh_log"] = log
    return cluster


def test_ssh_injects_locally_known_account(ssh, capfd, monkeypatch):
    # The laptop already knows the account (config/env) — no remote probe needed;
    # it rides the delegated command as --account so the CLUSTER's boxy (turnkey
    # or not) builds the script with it.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    ssh_log = ssh["ssh_log"].read_text()
    assert "RUN user@hops" in ssh_log                     # delegated over the master
    assert "--account fy260064" in ssh_log                # ...carrying the account
    assert "#SBATCH --account=fy260064" in cap.out        # remote boxy put it in the script
    assert "placed in the batch script" in cap.out        # the laptop-side decision line


def test_ssh_probes_mywcid_on_the_cluster(ssh, capfd, monkeypatch):
    # The laptop knows NOTHING (local account command disabled, no env), so the
    # account is discovered by running `mywcid` ON the cluster over the ssh
    # master, then injected into the delegated command.
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")        # no laptop-side probe
    monkeypatch.delenv("SBATCH_ACCOUNT", raising=False)
    monkeypatch.delenv("SLURM_ACCOUNT", raising=False)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: account: fy140001 (via mywcid on hops" in cap.out
    ssh_log = ssh["ssh_log"].read_text()
    assert "--account fy140001" in ssh_log                # injected into the delegation
    assert "#SBATCH --account=fy140001" in cap.out        # ...and into the remote script
