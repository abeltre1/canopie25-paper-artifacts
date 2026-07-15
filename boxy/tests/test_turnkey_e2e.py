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
    assert "auto: time: 30:00" in cap.out
    assert "stops the job at this walltime" in cap.out
    assert "--time 30:00" in ssh["ssh_log"].read_text()
    assert "#SBATCH --time=30:00" in cap.out


def test_ssh_explicit_time_wins(ssh, capfd, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT", "fy260064")
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--time", "2:00:00",
               "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "auto: time:" not in cap.out                    # not injected over an explicit flag
    ssh_log = ssh["ssh_log"].read_text()
    assert "--time 2:00:00" in ssh_log and "--time 30:00" not in ssh_log


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
    assert "#SBATCH --time=30:00" in script
