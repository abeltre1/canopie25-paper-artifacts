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


def test_ssh_injects_card_max_model_len_after_separator(ssh, capfd, monkeypatch):
    # the model card's engine args (--max-model-len, so vLLM doesn't OOM on the
    # 128K default) are injected AFTER `--`, while boxy flags (account/partition)
    # stay BEFORE it — even against an OLD cluster boxy that won't apply the card.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: engine args: --max-model-len 8192" in cap.out
    log = ssh["ssh_log"].read_text()
    assert "-- --max-model-len 8192" in log                 # engine args after --
    # boxy flags land BEFORE the `--` separator (not passed to vLLM)
    assert log.index("--account") < log.index(" -- ")
    assert log.index("--partition") < log.index(" -- ")


def test_ssh_user_engine_args_win_over_card(ssh, capfd, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun",
               "--", "--max-model-len", "4096"])
    capfd.readouterr()
    assert rc == 0
    log = ssh["ssh_log"].read_text()
    # card 8192 FIRST then the user's 4096 -> vLLM last-wins -> user's 4096
    assert log.index("8192") < log.index("4096")


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


# ---- scheduler is abstracted: no --scheduler needed over --ssh --------------------


def test_ssh_auto_detects_scheduler_when_flag_absent(ssh, capfd, monkeypatch):
    # NO --scheduler at all: boxy probes the cluster (sbatch on PATH) over the ssh
    # master and INJECTS --scheduler slurm, so even an OLD cluster boxy submits a
    # batch job. A novice types only the model + the host.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--ssh", "user@hops", "--dryrun"])   # no --scheduler
    cap = capfd.readouterr()
    assert rc == 0
    # sinfo returns partitions on the fake cluster -> slurm's control plane is LIVE
    sched_line = next(ln for ln in cap.out.splitlines() if "auto: scheduler:" in ln)
    assert "slurm" in sched_line and "Slurm is live" in sched_line
    ssh_log = ssh["ssh_log"].read_text()
    assert "--scheduler slurm" in ssh_log                  # injected into the delegation
    assert "#SBATCH --account=fy260064" in cap.out         # the remote boxy built a batch job


def test_ssh_flux_system_with_slurm_shims_detects_flux(ssh, capfd, monkeypatch):
    # The eldorado field case: a FLUX system whose slurm-compat layer is COMPLETE
    # enough that `sinfo` answers too (both flux-live AND slurm-live fire) and its
    # `sbatch` shim returns Flux job ids squeue can't track. A reachable Flux broker
    # is authoritative: detection must pick FLUX and NOT be defeated by the working
    # slurm shims. This is the misidentification fix.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    # flux present & live: `flux resource list` (and friends) exit 0; queue list
    # feeds the partition probe; `flux jobs` reports nothing live (auto-unique).
    _shim(ssh["bin"], "flux",
          "#!/bin/bash\n"
          'if [ "$1" = "--uri" ]; then shift 2; fi\n'   # a system flux answers on the system socket
          'case "$1" in\n'
          "  getattr) echo 0 ;;\n"                       # instance-level 0 == SYSTEM instance
          "  resource|uptime) : ;;\n"
          '  queue) echo "pbatch|true" ;;\n'
          "  jobs) : ;;\n"
          "  *) : ;;\n"
          "esac\n"
          "exit 0\n")
    # NB: the fixture's `sinfo` STILL returns partitions (slurm-live also fires) —
    # exactly like eldorado; flux must win anyway.
    rc = main(["serve", MODEL, "--ssh", "user@eldorado", "--dryrun"])   # no --scheduler
    cap = capfd.readouterr()
    assert rc == 0
    # the scheduler line names flux and explains the slurm shims are compat proxies.
    sched_line = next(ln for ln in cap.out.splitlines() if "auto: scheduler:" in ln)
    assert "flux" in sched_line and "Flux broker is live" in sched_line
    assert "compat shims" in sched_line                    # the slurm shims are named, not chosen
    ssh_log = ssh["ssh_log"].read_text()
    assert "--scheduler flux" in ssh_log                   # flux injected, NOT slurm
    assert "--scheduler slurm" not in ssh_log
    assert "# flux: --bank=" in cap.out                    # a Flux batch script was built


def test_ssh_scheduler_config_wins_over_detection(ssh, capfd, monkeypatch):
    # config BOXY_SCHEDULER pins the scheduler without probing PATH — the decision
    # line names the provenance so it's auditable.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_SCHEDULER", "slurm")
    rc = main(["serve", MODEL, "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: scheduler: slurm (via config site.scheduler on hops)" in cap.out
    assert "--scheduler slurm" in ssh["ssh_log"].read_text()


def test_ssh_explicit_scheduler_not_re_injected(ssh, capfd, monkeypatch):
    # power user pins --scheduler: no auto line, and it's not duplicated on the
    # delegated argv.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    # no laptop-side injection line (the delegated boxy still prints its own
    # "auto: scheduler: slurm (submitting a batch job …)" — that's not ours)
    assert "auto: scheduler: slurm (via" not in cap.out
    assert ssh["ssh_log"].read_text().count("--scheduler slurm") == 1


def test_ssh_here_stays_a_direct_serve(ssh, capfd, monkeypatch):
    # --here is an explicit "serve directly on that node": scheduler auto-detection
    # must NOT turn it into a batch job even though sbatch is on the host.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    main(["serve", MODEL, "--here", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert "auto: scheduler:" not in cap.out
    assert "--scheduler" not in ssh["ssh_log"].read_text()


# ---- default walltime (30 min) is abstracted too ----------------------------------


def test_ssh_injects_default_walltime(ssh, capfd, monkeypatch):
    # NO --time: the 30-min default rides the delegated command so the served job
    # carries a walltime even against an OLD cluster boxy (whose default is
    # laptop-side). The decision line warns the scheduler stops the job then.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: time: 1:00:00" in cap.out
    assert "stops the job at this walltime" in cap.out
    assert "--time 1:00:00" in ssh["ssh_log"].read_text()
    assert "#SBATCH --time=1:00:00" in cap.out


def test_ssh_explicit_time_wins(ssh, capfd, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--time", "2:00:00",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: time:" not in cap.out                    # not injected over an explicit flag
    ssh_log = ssh["ssh_log"].read_text()
    assert "--time 2:00:00" in ssh_log and "--time 1:00:00" not in ssh_log


# ---- readiness timeout is raised so an OLD cluster boxy doesn't give up early ------


def test_ssh_injects_ready_timeout(ssh, capfd, monkeypatch):
    # NO --ready-timeout: boxy raises the delegated boxy's readiness wait (an old
    # cluster boxy defaults to 180s and gives up on a still-loading model). The
    # generous ceiling rides the delegated command; READY is still reported the
    # instant the endpoint answers, so it never over-waits.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: ready-timeout: 20 min" in cap.out
    assert "--ready-timeout 1200" in ssh["ssh_log"].read_text()


def test_ssh_explicit_ready_timeout_wins(ssh, capfd, monkeypatch):
    # a power user pinning --ready-timeout (incl. 0 = submit-and-detach) is respected.
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ready-timeout", "0",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: ready-timeout:" not in cap.out
    assert "--ready-timeout 1200" not in ssh["ssh_log"].read_text()


# ---- login node itself: scheduler auto-detected on PATH ---------------------------


def test_login_node_scheduler_from_config(cluster, capfd, monkeypatch):
    # On the login node itself, an explicit BOXY_SCHEDULER submits a batch job
    # with NO --scheduler flag (the default 'auto' deliberately does NOT probe
    # PATH locally — the login-node guard handles that with a clear message). The
    # script on disk carries the mywcid account and the 30-min default walltime.
    monkeypatch.setenv("BOXY_SCHEDULER", "slurm")
    rc = main(["serve", MODEL])                            # no --scheduler flag
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: scheduler: slurm (via config site.scheduler)" in cap.out
    script = _the_script(cluster["jobs"])
    assert "#SBATCH --account=fy140001" in script
    assert "#SBATCH --time=1:00:00" in script


# ---- --share is abstracted: a team URL is published automatically over --ssh ------


def test_ssh_auto_share_derives_alias(ssh, capfd, monkeypatch):
    # NO --share: a served model over --ssh auto-publishes a team URL whose alias is
    # derived from the model's instance name (turnkey). The decision line is printed
    # laptop-side; the delegated command is NOT changed (share is a laptop concern).
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_AUTO_SHARE", "true")   # conftest turns it off for the suite
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    # dots are sanitized to dashes so the alias is a valid relay DNS label
    assert "auto: share: llama-3-1-8b-instruct" in cap.out
    assert "--share" not in ssh["ssh_log"].read_text()   # share is handled laptop-side, not delegated


def test_ssh_no_auto_share_when_disabled(ssh, capfd, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_AUTO_SHARE", "false")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: share:" not in cap.out


# ---- fully agentless over --ssh: NOTHING installed on the HPC ----------------------


def test_ssh_agentless_default_renders_self_contained_script(ssh, capfd, monkeypatch):
    # THE turnkey promise: `boxy serve <hf model> --ssh host` with NO boxy on the
    # cluster. The laptop renders a self-contained podman batch script (engine
    # pulls the model), with the site directives resolved over SSH — no delegation.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")   # production default; conftest opts the suite out
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless (no boxy on the cluster)" in cap.out
    # the engine pulls the bare repo id at container start (no RamaLama on the cluster)
    assert "the engine downloads it at container start" in cap.out
    assert "serve meta-llama/Llama-3.1-8B-Instruct" in cap.out
    assert "podman run" in cap.out
    # site directives resolved laptop-side over SSH and baked into the script
    assert "#SBATCH --account=fy260064" in cap.out
    assert "#SBATCH --partition=gpu,batch" in cap.out
    assert "#SBATCH --time=1:00:00" in cap.out
    assert "sbatch --parsable" in cap.out
    # NOT the delegated path — the cluster's boxy is never invoked
    assert "$ boxy serve" not in cap.out
    assert "RUN user@hops" not in ssh["ssh_log"].read_text() or "boxy serve" not in ssh["ssh_log"].read_text()


def test_ssh_delegate_flag_forces_the_cluster_boxy(ssh, capfd, monkeypatch):
    # --delegate opts out of agentless and runs the cluster's own boxy (the old
    # path) — for --replicas/--distributed/--box, or a cluster that has boxy.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--delegate", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless (no boxy on the cluster)" not in cap.out
    assert "$ boxy serve" in cap.out                       # delegated to the cluster boxy
    assert "--delegate" not in ssh["ssh_log"].read_text()  # laptop-only flag, not forwarded


def test_ssh_agentless_flux_bank_and_engine_pull(ssh, capfd, monkeypatch):
    # Flux system: the agentless script uses `# flux:` directives (--bank) and the
    # single queue, still engine-pulling the model.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "flux", "--ssh", "user@eldorado", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless (no boxy on the cluster)" in cap.out
    assert "# flux: --bank=fy260064" in cap.out
    assert "serve meta-llama/Llama-3.1-8B-Instruct" in cap.out


# ---- agentless: the Slurm GPU request FORM is auto-detected from sinfo GRES --------


def test_ssh_agentless_default_gpu_request_is_gpus_per_node(ssh, capfd, monkeypatch):
    # A WORKING cluster is NEVER changed pre-emptively: even though the fixture's
    # sinfo advertises a GPU type, the default 'auto' emits the proven
    # --gpus-per-node (what works on hops/eldorado). The portable --gres form is
    # only used REACTIVELY, if the site rejects this at submit (see the recovery
    # tests). This is the fix for hops flipping to --gres=gpu:h100:1 unexpectedly.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\ncat <<'EOF'\n"
          "gpu|up|2/6/0/8|gpu:h100:8\n"
          "batch|up|8/2/0/10|gpu:h100:4\n"
          "EOF\n")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "#SBATCH --gpus-per-node=1" in cap.out           # unchanged working form
    assert "--gres=gpu" not in cap.out                      # NOT pre-emptively switched
    assert "auto: gpu request:" not in cap.out              # no proactive decision line


def test_ssh_agentless_pinned_gpu_directive_forces_gres(ssh, capfd, monkeypatch):
    # BOXY_GPU_DIRECTIVE pins the form explicitly (escape hatch): 'gres' emits
    # --gres=gpu:N with no probing and no self-heal.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_GPU_DIRECTIVE", "gres")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "#SBATCH --gres=gpu:1" in cap.out
    assert "--gpus-per-node" not in cap.out


# ---- interactive WCID picker: the chosen account lands in the batch script ----------


def test_login_node_picker_selects_account(cluster, capfd, monkeypatch):
    # Several WCIDs from mywcid + a TTY -> the numbered menu. A pick of '2' puts
    # the SECOND account into the batch script the scheduler actually ran.
    monkeypatch.setenv("BOXY_PICK_ACCOUNT", "always")       # suite default is 'never'
    monkeypatch.setattr("builtins.input", lambda *a, **k: "2")
    rc = main(["serve", MODEL, "--scheduler", "slurm"])     # no --account, no --dryrun
    out = capfd.readouterr().out
    assert rc == 0
    script = _the_script(cluster["jobs"])
    assert "#SBATCH --account=fy140252" in script           # the 2nd WCID, not the 1st
    assert "#SBATCH --account=fy140001" not in script
    assert "auto: account: fy140252 (you picked 2 of 3 from mywcid)" in out
    assert "### READY" in out


def test_login_node_no_prompt_when_disabled(cluster, capfd, monkeypatch):
    # Suite default (never): multi-account discovery keeps the silent first-pick
    # and NEVER calls input() — proves batch/CI can't block on the menu.
    def _boom(*a, **k):
        raise AssertionError("input() must not be called when the picker is off")

    monkeypatch.setattr("builtins.input", _boom)
    rc = main(["serve", MODEL, "--scheduler", "slurm"])     # BOXY_PICK_ACCOUNT=never (conftest)
    assert rc == 0
    assert "#SBATCH --account=fy140001" in _the_script(cluster["jobs"])   # first, silently


def test_ssh_agentless_picker_selects_account(ssh, capfd, monkeypatch):
    # Over --ssh with NO boxy on the cluster: mywcid is probed ON the cluster and
    # the menu runs LAPTOP-side. Emptying the LOCAL account command emulates a
    # laptop with no mywcid, so resolution falls through to the remote probe.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_PICK_ACCOUNT", "always")
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")          # laptop has no mywcid
    monkeypatch.delenv("BOXY_ACCOUNT", raising=False)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "2")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless (no boxy on the cluster)" in cap.out
    assert "#SBATCH --account=fy140252" in cap.out
    assert "you picked 2 of 3 from mywcid on hops" in cap.out


def test_ssh_agentless_forwards_proxy_to_the_compute_node_pull(ssh, capfd, monkeypatch):
    # THE regression: the --ssh agentless script must forward the corporate proxy to
    # the compute-node `podman pull`, or an isolated node 403s on Docker Hub (Zscaler).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.delenv("BOXY_NO_PROXY_PROPAGATE", raising=False)     # opt in (suite opts out)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--proxy", "http://proxy.site:80",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    # the podman pull is prefixed with the proxy env, and boxy says so
    assert "exec env " in cap.out and "proxy.site:80" in cap.out
    assert "env " in cap.out.split("podman run")[0].rsplit("exec ", 1)[-1] or "proxy.site:80" in cap.out
    assert "forwarding" in cap.out and "compute-node image pull" in cap.out


def test_ssh_agentless_license_flag_lands_in_the_script(ssh, capfd, monkeypatch):
    # --license adds #SBATCH --license=... in the agentless batch script (Slurm).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--license", "tscratch:1,pscratch:1",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "#SBATCH --license=tscratch:1,pscratch:1" in cap.out
    assert "auto: license: tscratch:1,pscratch:1 (via --license)" in cap.out


def test_ssh_agentless_ca_picked_on_the_node_at_runtime(ssh, capfd, monkeypatch, tmp_path):
    # THE hops CERTIFICATE_VERIFY_FAILED fix: the batch script picks a CA bundle ON
    # THE COMPUTE NODE at job runtime — the boxy-staged laptop bundle first (when one
    # exists), else the node's OWN system trust store, which provably trusts the
    # site's TLS interceptor (host-side pulls work) — and splices the mount + TLS env
    # into the podman argv via $_CAARGS. Never the laptop's SSL_CERT_FILE path
    # (meaningless on the node; the old bug bind-mounted an empty file).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    laptop_ca = tmp_path / "ca-merged.crt"
    laptop_ca.write_text("-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(laptop_ca))
    monkeypatch.delenv("BOXY_NO_CA_PROPAGATE", raising=False)     # opt in (suite opts out)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "would stage your merged site CA" in cap.out
    # staged shared-FS bundle is the FIRST candidate, node system stores follow
    assert "agentless/hops/boxy-ca-merged.pem" in cap.out
    assert "/etc/pki/tls/certs/ca-bundle.crt" in cap.out
    # the pick is spliced into the container argv before the image ...
    assert "${_CAARGS}" in cap.out
    assert "-e SSL_CERT_FILE=/etc/ssl/certs/boxy-ca-merged.pem" in cap.out
    # ... and the laptop path is NEVER mounted (the CERTIFICATE_VERIFY_FAILED bug)
    assert f"{laptop_ca}:" not in cap.out


def test_ssh_agentless_node_system_ca_without_laptop_bundle(ssh, capfd, monkeypatch):
    # No merged bundle on the laptop: nothing is staged, but the script STILL
    # carries the runtime CA pick over the node's own system stores — the fix must
    # not depend on the laptop's TLS setup at all.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "stage your merged site CA" not in cap.out              # nothing staged
    assert "agentless/hops/boxy-ca-merged.pem" not in cap.out      # no staged candidate
    assert "/etc/pki/tls/certs/ca-bundle.crt" in cap.out           # RHEL store
    assert "/etc/ssl/certs/ca-certificates.crt" in cap.out         # Debian store
    assert "${_CAARGS}" in cap.out


def test_ssh_agentless_persistent_hf_cache_on_shared_fs(ssh, capfd, monkeypatch):
    # engine-pull mounts a shared-FS HF cache into the container (download once,
    # reused by every rerun) and the script creates the dir at job runtime.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: model cache:" in cap.out and "hfcache" in cap.out
    assert "hfcache:/root/.cache/huggingface" in cap.out           # the bind mount
    assert "HF_HOME=/root/.cache/huggingface" in cap.out           # engine writes there
    assert 'mkdir -p "$(dirname "${_EP}")/hfcache"' in cap.out     # created at runtime


def test_agentless_list_logs_stop_answer_from_laptop_records(ssh, capfd, monkeypatch, tmp_path):
    # FIELD: `boxy list --ssh hops` delegated to the cluster's boxy, whose
    # login-node `podman ps` showed NOTHING while the container ran on the compute
    # node. Agentless records live on THIS machine; list/logs/stop must answer
    # from them over the master — never delegate.
    from boxy import jobs

    rdir = tmp_path / "agentless"
    rdir.mkdir()
    (rdir / "boxy-m-777.log").write_text("vllm says hello\nApplication startup complete\n")
    ep = rdir / "boxy-m.endpoint.json"
    ep.write_text('{"name": "boxy-m", "host": "hops7", "port": 8000, "url": "http://hops7:8000"}')
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "777",
                                 "model": "hf://x", "submitted_from": "agentless-ssh",
                                 "target": "user@hops", "endpoint_remote": str(ep),
                                 "log": str(rdir / "boxy-m-%j.log")})
    _shim(ssh["bin"], "squeue", "#!/bin/bash\necho RUNNING\n")
    _shim(ssh["bin"], "scancel", "#!/bin/bash\necho ok\n")

    rc = main(["list", "--ssh", "user@hops"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "agentless jobs" in cap.out
    assert "boxy-m  slurm job 777 on user@hops  RUNNING  http://hops7:8000/v1" in cap.out
    assert "### Remote" not in cap.out                     # no delegation to a cluster boxy
    assert "podman ps" not in cap.out                      # no login-node container listing

    rc = main(["logs", "boxy-m"])                          # fetched over the master
    cap = capfd.readouterr()
    assert rc == 0
    assert "vllm says hello" in cap.out

    rc = main(["stop", "boxy-m", "--ssh", "user@hops"])    # canceled over the master
    cap = capfd.readouterr()
    assert rc == 0
    assert "stopped slurm job 777 on user@hops" in cap.out
    assert jobs.read_record("boxy-m") is None              # record reaped


def test_boxy_attach_rejoins_agentless_serve(ssh, capfd, monkeypatch, tmp_path):
    # FIELD: the serve detached while vLLM was still loading and there was NO way
    # back to the tunnel/READY. `boxy attach` re-joins: reads the endpoint from
    # the shared FS over the master, opens the tunnel, waits (extending while the
    # job is alive), prints READY.
    from boxy import jobs, remote

    rdir = tmp_path / "agentless"
    rdir.mkdir()
    (rdir / "boxy-m-777.log").write_text("Application startup complete.\n")
    ep = rdir / "boxy-m.endpoint.json"
    ep.write_text('{"name": "boxy-m", "host": "hops11", "port": 8000, "url": "http://hops11:8000"}')
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "777",
                                 "model": "hf://x", "submitted_from": "agentless-ssh",
                                 "target": "user@hops", "endpoint_remote": str(ep),
                                 "log": str(rdir / "boxy-m-%j.log")})
    _shim(ssh["bin"], "squeue", "#!/bin/bash\necho RUNNING\n")
    calls = {}

    def fake_await(host, node, port, log_path, *a, **kw):
        calls.update(host=host, node=node, port=port, log=log_path,
                     alive=kw.get("still_alive"))
        return True

    monkeypatch.setattr(remote, "await_ready_and_tunnel", fake_await)
    rc = main(["attach", "boxy-m"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "attaching to boxy-m on hops11" in cap.out
    assert calls["host"] == "user@hops" and calls["node"] == "hops11" and calls["port"] == 8000
    assert calls["log"].endswith("boxy-m-777.log")     # newest real per-job file, not the %j token
    assert calls["alive"] is not None and calls["alive"]() is True   # squeue says RUNNING


def test_await_ready_extends_while_job_alive(monkeypatch, tmp_path):
    # THE detach fix: when the window expires but the scheduler says the job is
    # alive, the wait EXTENDS; when the job is gone, it returns False.
    import boxy.readiness as readiness
    from boxy import remote

    monkeypatch.setattr(remote, "_pick_local_port", lambda lp, rp: 0)
    monkeypatch.setattr(remote, "add_forward", lambda *a: 1)   # no tunnel: log-grep only
    monkeypatch.setattr(readiness, "probe_once", lambda *a, **k: None)
    monkeypatch.setattr(remote, "ssh_capture", lambda *a, **k: (0, ""))  # log never ready
    # fake clock: sleep(5) advances a minute so the 120s extensions elapse instantly
    clock = {"t": 0.0}
    monkeypatch.setattr("time.time", lambda: clock["t"])
    monkeypatch.setattr("time.sleep", lambda s: clock.__setitem__("t", clock["t"] + 60.0))

    lives = iter([True, True, False])                          # alive twice, then gone
    rc = remote.await_ready_and_tunnel("u@h", "n1", 8000, "/log", None, "", "", "relay",
                                       False, timeout_s=0.0, still_alive=lambda: next(lives))
    assert rc is False                                          # detached only when the job died


def test_ssh_agentless_autodetects_rocm_from_gres(ssh, capfd, monkeypatch):
    # FIELD: an AMD cluster got the CUDA image and died with 'unresolvable CDI
    # devices nvidia.com/gpu=all'. The SCHEDULER's GPU inventory is authoritative
    # (it describes the COMPUTE nodes): mi300a GRES -> rocm, even when the login
    # node exposes no GPU userland at all.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\ncat <<'EOF'\n"
          "batch|up|2/6/0/8|gpu:mi300a:4\n"
          "EOF\n")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@eldorado", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: accelerator: rocm (via the GPU stack on eldorado's login node)" in cap.out
    assert "nvidia.com/gpu" not in cap.out                          # not the CUDA device flag
    assert "--device /dev/kfd" in cap.out                           # the AMD device nodes
    exec_line = [ln for ln in cap.out.splitlines() if "podman run" in ln][0]
    assert "rocm" in exec_line                                      # the ROCm vLLM image


def test_ssh_agentless_autodetects_rocm_from_userland(ssh, capfd, monkeypatch):
    # No GRES signal (CPU-only sinfo answer, e.g. a Flux system without the
    # slurm shim): the login node's ROCm userland (rocm-smi) still wins.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\necho 'batch|up|2/6/0/8|(null)'\n")
    _shim(ssh["bin"], "rocm-smi", "#!/bin/bash\ntrue\n")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@eldorado", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: accelerator: rocm (via the GPU stack on eldorado's login node)" in cap.out
    assert "--device /dev/kfd" in cap.out


def test_ssh_agentless_explicit_accelerator_skips_probe(ssh, capfd, monkeypatch):
    # --accelerator pins it; the probe never runs (rocm-smi on PATH is ignored).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    _shim(ssh["bin"], "rocm-smi", "#!/bin/bash\ntrue\n")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--accelerator", "cuda",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: accelerator:" not in cap.out
    assert "nvidia.com/gpu" in cap.out


def test_relay_apps_domain_discovery(monkeypatch):
    # <service>.apps.<cluster>.<org>.<tld>: the apps domain comes from the
    # cluster ingress config, else the api.->apps. transform of the oc server URL.
    from boxy.exposers import relay

    calls = {}

    def fake_oc(args, **kw):
        calls.setdefault("seq", []).append(args[0])
        if args[0] == "get" and "ingresses.config.openshift.io" in args:
            return 0, "apps.goodall.sandia.gov"
        return 1, ""

    monkeypatch.setattr(relay, "_oc", fake_oc)
    dom, why = relay.discover_apps_domain()
    assert dom == "apps.goodall.sandia.gov" and why == "cluster ingress config"

    def fake_oc2(args, **kw):
        if args[0] == "get":
            return 1, "forbidden"                                  # no cluster-scope read
        if args[0] == "whoami":
            return 0, "https://api.goodall.sandia.gov:6443"
        return 1, ""

    monkeypatch.setattr(relay, "_oc", fake_oc2)
    dom, why = relay.discover_apps_domain()
    assert dom == "apps.goodall.sandia.gov" and "api. -> apps." in why

    monkeypatch.setattr(relay, "_oc", lambda a, **k: (1, "not logged in"))
    dom, why = relay.discover_apps_domain()
    assert dom == "" and "oc login" in why


def test_generate_relay_autodiscovers_host(capsys, monkeypatch):
    # zero-flag relay: no --host -> the Route host derives from the logged-in
    # cluster (boxy-relay.apps.<cluster>.<org>.<tld>); stdout stays pure YAML.
    from boxy.exposers import relay

    monkeypatch.setattr(relay, "discover_apps_domain",
                        lambda: ("apps.goodall.sandia.gov", "cluster ingress config"))
    rc = main(["generate", "relay", "--auth", "boxy:pw"])
    cap = capsys.readouterr()
    assert rc == 0
    assert "auto: relay host: boxy-relay.apps.goodall.sandia.gov" in cap.err
    assert "host: boxy-relay.apps.goodall.sandia.gov" in cap.out
    assert "kind: Deployment" in cap.out


def test_config_default_proxy_is_sandia(monkeypatch):
    # zero-flag turnkey on-site: network.proxy ships the Sandia proxy so no --proxy
    # is ever needed; BOXY_PROXY= (empty) opts out off-site.
    from boxy import config

    monkeypatch.delenv("BOXY_PROXY", raising=False)
    config.reset()
    assert config.get_str("network.proxy") == "http://proxy.sandia.gov:80"


def test_ssh_agentless_prestage_dryrun_announces_plan(ssh, capfd, monkeypatch):
    # agentless on an ISOLATED node: with prestaging on (auto), --dryrun ANNOUNCES it
    # would pull the image + download the hf:// model on the login node and serve BY
    # PATH — but performs nothing (dryrun must never touch the cluster).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_AGENTLESS_PRESTAGE", "auto")        # suite default is 'never'
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "prestage: would stage model meta-llama/Llama-3.1-8B-Instruct" in cap.out
    assert "then serve BY PATH" in cap.out
    assert "not run under --dryrun" in cap.out


def test_ssh_agentless_no_prestage_keeps_engine_pull(ssh, capfd, monkeypatch):
    # --no-prestage (also the suite default 'never') keeps the engine-pull render and
    # prints no prestage plan.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_AGENTLESS_PRESTAGE", "auto")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--no-prestage",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "prestage:" not in cap.out
    assert "the engine downloads it at container start" in cap.out


def test_prestage_agentless_model_stages_and_rewrites_to_path(monkeypatch, tmp_path):
    # Unit: _prestage_agentless_model pulls the image, runs the in-container
    # snapshot_download, and returns the shared-FS path (which the caller mounts).
    import argparse

    from boxy import cli
    from boxy.box import Box

    calls = []

    def fake_capture(target, cmd, timeout=20):
        calls.append(cmd)
        return 0, ""

    monkeypatch.setattr("boxy.remote.ssh_capture", fake_capture)
    monkeypatch.setattr("boxy.remote.remote_proxy_env", lambda: {})
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    box = Box(name="m", image="", entrypoint="vllm", engine="vllm",
              model="meta-llama/Llama-3.1-8B-Instruct")
    args = argparse.Namespace()
    staged = cli._prestage_agentless_model(
        args, "user@hops", "hops", box, "vllm/vllm-openai:latest", "", "/home/u/.local/share/boxy/agentless/hops", None)
    assert staged == "/home/u/.local/share/boxy/agentless/hops/models/meta-llama-llama-3.1-8b-instruct"
    # the image was pulled on the login node ...
    assert any("podman pull vllm/vllm-openai:latest" in c for c in calls)
    # ... and the model downloaded IN-CONTAINER via huggingface_hub, with the token
    dl = [c for c in calls if "snapshot_download" in c]
    assert dl and "hf_secret" in dl[0] and "python3" in dl[0]
    assert "meta-llama/Llama-3.1-8B-Instruct" in dl[0]


def test_ssh_agentless_partition_picker_selects_one(ssh, capfd, monkeypatch):
    # 2+ partitions + a TTY -> the partition menu; picking one puts it in the script
    # instead of the soonest-start comma-list.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_PICK_PARTITION", "always")     # suite default is 'never'
    monkeypatch.setattr("builtins.input", lambda *a, **k: "2")   # pick the 2nd partition
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    # the fixture's GPU partitions are gpu(6 idle) + batch(2 idle) -> menu [gpu, batch, all]
    assert "#SBATCH --partition=batch" in cap.out
    assert "#SBATCH --partition=gpu,batch" not in cap.out
    assert "you picked batch of 2" in cap.out


def test_ssh_agentless_partition_default_keeps_comma_list(ssh, capfd, monkeypatch):
    # suite default (never): the soonest-start comma-list is kept, no prompt.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("partition menu shown when disabled")))
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "#SBATCH --partition=gpu,batch" in cap.out


def test_wcid_env_bypasses_the_menu(cluster, capfd, monkeypatch):
    # $WCID is a scripted bypass: it wins with no prompt even under 'always'.
    monkeypatch.setenv("BOXY_PICK_ACCOUNT", "always")
    monkeypatch.setenv("WCID", "fy260064")
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("menu shown despite $WCID")))
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --account=fy260064" in out               # $WCID, no menu


# ---- GPU request auto-recovery: a site that rejects --gpus-per-node -----------------

# a picky Slurm: rejects any script asking with --gpus-per-node ('Invalid generic
# resource (gres) specification', field: kahuna), accepts the portable --gres form.
GRES_PICKY_SBATCH = r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"
if grep -q -- '--gpus-per-node' "$script"; then
  echo "sbatch: error: Invalid generic resource (gres) specification" >&2
  exit 1
fi
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
"""


def test_gres_fallback_forms_order(monkeypatch):
    from boxy.cli import _gres_fallback_forms
    # every sinfo-reported TYPE is tried as --gres=gpu:<type>:N (some sites
    # REQUIRE the type — field: kahuna), then untyped --gres, then --gpus. The
    # just-rejected default --gpus-per-node is NOT retried — unless a pinned
    # BOXY_GPU_TYPE poisoned it, where dropping the type is the fix.
    assert _gres_fallback_forms(["h200", "a100"]) == [
        ("gres", "h200"), ("gres", "a100"), ("gres", ""), ("gpus", ""), ("none", "")]
    assert _gres_fallback_forms([]) == [("gres", ""), ("gpus", ""), ("none", "")]
    monkeypatch.setenv("BOXY_GPU_TYPE", "h200")               # the poisoned-pin case
    assert _gres_fallback_forms([]) == [
        ("gpus-per-node", ""), ("gres", ""), ("gpus", ""), ("none", "")]


def test_login_node_auto_recovers_from_gres_rejection(cluster, capfd, monkeypatch):
    # Same recovery when boxy submits ON the login node: the picky sbatch rejects
    # --gpus-per-node; sinfo names a100 first, so the typed form is tried and
    # accepted (some sites require the type; a site that doesn't still takes it).
    _shim(cluster["bin"], "sbatch", GRES_PICKY_SBATCH)
    rc = main(["serve", MODEL, "--scheduler", "slurm"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "retrying with --gres=gpu:a100:N" in cap.err
    assert "GPU request accepted as --gres=gpu:a100:N (auto-recovered)" in cap.out
    script = _the_script(cluster["jobs"])
    assert "--gres=gpu:a100:1" in script and "--gpus-per-node" not in script


def test_ssh_agentless_recovers_on_type_required_site(ssh, capfd, monkeypatch):
    # THE kahuna failure: the site rejects EVERY untyped GPU spelling
    # (--gpus-per-node, --gres=gpu:N, --gpus=N) and accepts only a TYPED
    # --gres=gpu:<type>:N. The recovery must try each sinfo-reported type.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("HOME", str(ssh["tmp"] / "home"))
    (ssh["tmp"] / "home").mkdir(exist_ok=True)
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\ncat <<'EOF'\n"
          "hopper|up|2/6/0/8|gpu:h200:4\n"
          "short|up|3/5/0/8|(null)\n"
          "EOF\n")
    _shim(ssh["bin"], "sbatch", r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"
if ! grep -Eq -- '--gres=gpu:[a-z0-9]+:[0-9]+' "$script"; then
  echo "sbatch: error: Invalid generic resource (gres) specification" >&2
  exit 1
fi
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
""")
    monkeypatch.setattr(remote, "await_ready_and_tunnel", lambda *a, **k: True)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "hopper",
               "--ssh", "user@kahuna"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "retrying with --gres=gpu:h200:N" in cap.err
    assert "GPU request accepted as --gres=gpu:h200:N (auto-recovered)" in cap.out


def test_ssh_agentless_recovers_via_per_node_gres(ssh, capfd, monkeypatch):
    # Some sites report (null) GRES at PARTITION level and only surface the GPU
    # type per-NODE (`sinfo -N -o %G`). The recovery falls back to that probe
    # before abandoning the typed form.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("HOME", str(ssh["tmp"] / "home"))
    (ssh["tmp"] / "home").mkdir(exist_ok=True)
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\n"
          'if [ "$1" = "-h" ] && [ "$2" = "-N" ]; then echo "gpu:h200:4(S:0-1)"; exit 0; fi\n'
          "cat <<'EOF'\nhopper|up|2/6/0/8|(null)\nEOF\n")
    _shim(ssh["bin"], "sbatch", r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"
if ! grep -Eq -- '--gres=gpu:[a-z0-9]+:[0-9]+' "$script"; then
  echo "sbatch: error: Invalid generic resource (gres) specification" >&2
  exit 1
fi
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
""")
    monkeypatch.setattr(remote, "await_ready_and_tunnel", lambda *a, **k: True)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "hopper",
               "--ssh", "user@kahuna"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "GPU request accepted as --gres=gpu:h200:N (auto-recovered)" in cap.out


def test_ssh_agentless_recovers_on_no_gres_site(ssh, capfd, monkeypatch):
    # FIELD (kahuna): EVERY partition reports (null) GRES — the site has no GRES
    # config at all, GPUs are implied by the PARTITION (hopper/grace/blackwell),
    # and ANY --gres/--gpus/--gpus-per-node spelling errors. The last recovery
    # rung drops the GPU directive entirely.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("HOME", str(ssh["tmp"] / "home"))
    (ssh["tmp"] / "home").mkdir(exist_ok=True)
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\n"
          'if [ "$1" = "-h" ] && [ "$2" = "-N" ]; then echo "(null)"; exit 0; fi\n'
          "cat <<'EOF'\nhopper|up|1/1/2/4|(null)\nblackwell|up|2/0/0/2|(null)\nEOF\n")
    _shim(ssh["bin"], "sbatch", r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"
if grep -Eq -- '--(gres|gpus|gpus-per-node)' "$script"; then
  echo "sbatch: error: Invalid generic resource (gres) specification" >&2
  exit 1
fi
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
""")
    monkeypatch.setattr(remote, "await_ready_and_tunnel", lambda *a, **k: True)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "hopper",
               "--ssh", "user@kahuna"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "GPU request accepted as NO GPU directive" in cap.out


def test_ssh_agentless_recovers_gres_then_license(ssh, capfd, monkeypatch):
    # THE full kahuna sequence: no GRES config (every GPU spelling invalid) AND
    # no tscratch license. The GPU ladder must land on 'no directive', then the
    # license self-heal drops --license and the submit succeeds.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("BOXY_LICENSE", "tscratch:1")          # ship-default, opt back in
    monkeypatch.setenv("HOME", str(ssh["tmp"] / "home"))
    (ssh["tmp"] / "home").mkdir(exist_ok=True)
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\n"
          'if [ "$1" = "-h" ] && [ "$2" = "-N" ]; then echo "(null)"; exit 0; fi\n'
          "cat <<'EOF'\nhopper|up|1/1/2/4|(null)\nEOF\n")
    _shim(ssh["bin"], "sbatch", r"""#!/bin/bash
echo "$@" >> "$SBATCH_LOG"
script="${!#}"
if grep -Eq -- '--(gres|gpus|gpus-per-node)' "$script"; then
  echo "sbatch: error: Invalid generic resource (gres) specification" >&2
  exit 1
fi
if grep -q -- '--license' "$script"; then
  echo "sbatch: error: Batch job submission failed: Invalid license specification" >&2
  exit 1
fi
ep="${script%.sh}.endpoint.json"
host="$(hostname)"
printf '{"name":"x","host":"%s","port":8000,"url":"http://%s:8000","job":"12345"}\n' \
       "$host" "$host" > "$ep"
echo "12345"
""")
    monkeypatch.setattr(remote, "await_ready_and_tunnel", lambda *a, **k: True)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "hopper",
               "--ssh", "user@kahuna"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "retrying with NO GPU directive" in cap.err          # the GPU ladder's last rung
    assert "the site rejected the license request" in cap.err   # then the license self-heal
    assert "### license request dropped (auto-recovered)." in cap.out
    assert "### Submitted slurm job 12345" in cap.out


def test_ssh_agentless_auto_recovers_from_gres_rejection(ssh, capfd, monkeypatch):
    # THE kahuna fix: sinfo shows no gpu GRES so the first script uses
    # --gpus-per-node, which this site rejects. boxy must re-render with the
    # portable --gres=gpu:N and RESUBMIT on its own — no env var, no rerun.
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "true")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    monkeypatch.setenv("HOME", str(ssh["tmp"] / "home"))     # keep pushed scripts in tmp
    (ssh["tmp"] / "home").mkdir(exist_ok=True)
    _shim(ssh["bin"], "sinfo", "#!/bin/bash\ncat <<'EOF'\ngpu|up|2/6/0/8|(null)\nEOF\n")
    _shim(ssh["bin"], "sbatch", GRES_PICKY_SBATCH)
    monkeypatch.setattr(remote, "await_ready_and_tunnel", lambda *a, **k: True)
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--partition", "gpu", "--ssh", "user@hops"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "retrying with --gres=gpu:N" in cap.err
    assert "GPU request accepted as --gres=gpu:N (auto-recovered)" in cap.out
    # and the accepted submission really used the portable form
    submits = ssh["sbatch_log"].read_text()
    assert submits.count("--parsable") >= 2                   # first (rejected) + retry
