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
    # fake sinfo (-o "%R|%a|%F|%G"): name|up-down|A/I/O/T nodes|GRES. GPU
    # partitions gpu(6 idle) and batch(2 idle); short is CPU-only; down-pt is
    # down. GPU-aware auto -> "gpu,batch"; --partition all -> "gpu,short,batch".
    _shim(binp, "sinfo", "#!/bin/bash\ncat <<'EOF'\n"
          "gpu|up|2/6/0/8|gpu:a100:8\n"
          "short|up|3/5/0/8|(null)\n"
          "batch|up|8/2/0/10|gpu:v100:4\n"
          "down-pt|down|0/8/0/8|gpu:a100:8\n"
          "EOF\n")
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
    rc = main(["serve", MODEL, "--scheduler", "slurm"])   # NO --partition flag
    out = capfd.readouterr().out
    assert rc == 0
    # the turnkey promise: mywcid -> the batch script the scheduler actually ran
    script = _the_script(cluster["jobs"])
    assert "#SBATCH --account=fy140001" in script     # first account from the table
    assert "#SBATCH --gpus-per-node=1" in script      # 8B model card -> 1 GPU
    # ...and the partition was auto-picked with NO flag (the "don't set it"
    # default): a GPU job gets only GPU partitions, idle-first.
    assert "#SBATCH --partition=gpu,batch" in script
    # ...and it was really submitted + followed to READY
    assert "--parsable" in cluster["sbatch_log"].read_text()
    assert "auto: account: fy140001 (via mywcid" in out
    assert "### READY" in out
    # the endpoint record is discoverable (boxy list would show it)
    rec = json.loads((cluster["jobs"] / [p.name for p in cluster["jobs"].glob("*.json")
                                         if not p.name.endswith(".endpoint.json")][0]).read_text())
    assert rec["job"] == "12345" and rec["scheduler"] == "slurm"


def test_login_node_default_partition_is_gpu_aware_auto(cluster, capfd):
    # bare serve (no --partition) auto-picks GPU partitions only; a job never
    # parks in a CPU-only partition (the field 'stuck' failure).
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --partition=gpu,batch" in out       # 'short' (CPU) excluded
    assert "with GPUs" in out


def test_login_node_partition_all_includes_cpu(cluster, capfd):
    # power-user override: --partition all offers EVERY up partition.
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "all", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --partition=gpu,short,batch" in out


def test_login_node_partition_off_uses_site_default(cluster, capfd):
    # power-user override: --partition off emits no partition directive.
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "off", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --partition" not in out


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


# ---- auto-unique: never block on a live instance (start an independent one) --------


def test_auto_unique_forks_when_a_live_instance_exists(cluster, capfd, monkeypatch):
    from boxy import jobs, resolve

    _, name, _ = resolve.resolve_submission(MODEL, "slurm", require_exists=False)
    # a live PENDING instance already holds the deterministic name, no endpoint yet
    jobs.write_record(name, {"name": name, "scheduler": "slurm", "job": "111", "model": MODEL})
    _shim(cluster["bin"], "squeue", "#!/bin/bash\necho PENDING\n")   # _job_state -> PENDING
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--dryrun"])   # NO --unique
    out = capfd.readouterr().out
    assert rc == 0
    assert "starting an independent instance" in out
    assert f"--job-name={name}-" in out                 # forked unique name, not the base
    assert f"#SBATCH --job-name={name}\n" not in out    # the base name was NOT reused


def test_no_auto_unique_restores_singleton_block(cluster, capfd):
    from boxy import jobs, resolve

    _, name, _ = resolve.resolve_submission(MODEL, "slurm", require_exists=False)
    jobs.write_record(name, {"name": name, "scheduler": "slurm", "job": "111", "model": MODEL})
    _shim(cluster["bin"], "squeue", "#!/bin/bash\necho RUNNING\n")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--no-auto-unique", "--dryrun"])
    err = capfd.readouterr().err
    assert rc == 1
    assert "already submitted" in err                   # strict singleton preserved


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


def test_ssh_default_partition_auto_no_flag(ssh, capfd, monkeypatch):
    # No --partition at all over --ssh: boxy still auto-picks (GPU-aware) by
    # probing sinfo on the cluster and APPENDING the concrete list.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: partition: gpu,batch (via sinfo on hops" in cap.out
    assert "--partition gpu,batch" in ssh["ssh_log"].read_text()
    assert "#SBATCH --partition=gpu,batch" in cap.out


def test_ssh_resolves_partition_auto_to_concrete_list(ssh, capfd, monkeypatch):
    # `--partition auto` must be resolved to a CONCRETE list laptop-side (via
    # sinfo on the cluster) before delegating — an older cluster boxy would pass
    # the literal 'auto' to sbatch and get 'invalid partition'. GPU-aware.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")            # keep account quiet/deterministic
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "auto",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: partition: gpu,batch (via sinfo on hops" in cap.out
    ssh_log = ssh["ssh_log"].read_text()
    assert "--partition gpu,batch" in ssh_log                 # concrete GPU list delegated
    assert "--partition auto" not in ssh_log                  # literal 'auto' never sent
    assert "#SBATCH --partition=gpu,batch" in cap.out         # remote script built with it


def test_ssh_partition_all_includes_cpu(ssh, capfd, monkeypatch):
    # power-user override survives delegation: --partition all -> every partition.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "all",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "--partition gpu,short,batch" in ssh["ssh_log"].read_text()
    assert "#SBATCH --partition=gpu,short,batch" in cap.out


def test_ssh_partition_auto_from_config_default(ssh, capfd, monkeypatch):
    # BOXY_PARTITION=auto with NO --partition flag must still be resolved over
    # --ssh: there's no flag to rewrite, so the concrete list is appended.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_PARTITION", "auto")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: partition: gpu,batch (via sinfo on hops" in cap.out
    assert "--partition gpu,batch" in ssh["ssh_log"].read_text()
    assert "#SBATCH --partition=gpu,batch" in cap.out


def test_ssh_injects_unique_when_a_live_instance_is_on_the_cluster(ssh, capfd, monkeypatch):
    # A live job of this model's name already runs on the cluster -> boxy injects
    # --unique laptop-side so the (possibly OLD) cluster boxy starts an
    # independent instance instead of blocking. Works without --unique typed.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "squeue", "#!/bin/bash\necho 111\n")   # a job with the name is LIVE
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: --unique" in cap.out and "already live on hops" in cap.out
    assert "--unique" in ssh["ssh_log"].read_text()          # rode the delegated command


def test_ssh_no_unique_when_nothing_live(ssh, capfd, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "squeue", "#!/bin/bash\ntrue\n")       # no job with the name
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: --unique" not in cap.out
    assert "--unique" not in ssh["ssh_log"].read_text()


def test_ssh_power_user_unique_flag_is_respected_not_doubled(ssh, capfd, monkeypatch):
    # explicit --unique still works and isn't duplicated by the auto-injection.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "squeue", "#!/bin/bash\necho 111\n")   # even with a live one
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--unique", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: --unique" not in cap.out                   # not re-injected
    assert ssh["ssh_log"].read_text().count("--unique") == 1


def test_ssh_forwards_proxy_to_the_cluster(ssh, capfd, monkeypatch):
    # BOXY_PROXY (or ambient http(s)_proxy) is forwarded to the cluster job's
    # pulls over --ssh, even if the login node's own env lacks it.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_PROXY", "http://proxy.corp:80")
    monkeypatch.delenv("BOXY_NO_PROXY_PROPAGATE", raising=False)   # opt back in
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "### Proxy   forwarding http://proxy.corp:80" in cap.out
    assert "https_proxy=http://proxy.corp:80" in ssh["ssh_log"].read_text()


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
