"""Site auto-resolution — account discovery (myaccounts / env / sacctmgr, via bash
shims on PATH), the config defaults for partition/time, the Flux single-queue
guard, and the end-to-end dryrun where a bare `--scheduler slurm` picks up the
account from `myaccounts` with no --account flag."""

import argparse
import os

import pytest

from boxy import site
from boxy.cli import main

# ---- account parser ---------------------------------------------------------------


@pytest.mark.parametrize("text,first", [
    ("ab110003\n", "ab110003"),
    ("ACCOUNT_ID: ab110003  (My Project)\n", "ab110003"),
    ("Account      Project\nab110003     Cool Stuff\n", "ab110003"),
    ("no accounts here\n", None),
])
def test_parse_accounts(text, first):
    got = site.parse_accounts(text)
    assert (got[0] if got else None) == first


def test_parse_accounts_multiple_in_order():
    assert site.parse_accounts("ab110003\nab119999\nab110003\n") == ["ab110003", "ab119999"]


# The REAL myaccounts output (field sample, 2026-07): header row, dashed rule, data
# rows where the DESCRIPTION starts with a bare numeric id right after the
# account, a privilege-less 'none' row, and a trailing caps note repeating the
# first account in uppercase.
REAL_MYACCOUNT_ID = """\
      User    Account                              Description     Parent
---------- ---------- ---------------------------------------- --------------------
     jdoe   ab110001        100001 project alpha                   nd
     jdoe   ab110002      100002 project beta                   nd
     jdoe   ab110003         100003 project gamma                   nd
     jdoe       none       default account, no job privileges
  The Account could be on Caps too: AB110001
"""


def test_parse_accounts_real_myaccounts_table():
    got = site.parse_accounts(REAL_MYACCOUNT_ID)
    # all three real accounts, in order; the bare description ids (100001…) are
    # NOT mistaken for accounts; the caps-note duplicate AB110001 is deduped
    # case-insensitively; the header/'none' rows contribute nothing.
    assert got == ["ab110001", "ab110002", "ab110003"]


def test_parse_account_rows_keeps_labels_for_the_menu():
    rows = site.parse_account_rows(REAL_MYACCOUNT_ID)
    # same three ids in order, each paired with the project text myaccounts prints
    # (the leading numeric description-id is dropped from the label).
    ids = [account_id for account_id, _ in rows]
    assert ids == ["ab110001", "ab110002", "ab110003"]
    labels = {account_id: label for account_id, label in rows}
    assert labels["ab110001"].startswith("project alpha")
    assert "100002" not in labels["ab110002"]                 # numeric desc-id stripped
    assert "project beta" in labels["ab110002"]


def test_parse_account_rows_labelled_layout():
    # the `ACCOUNT_ID: fy... (Project)` layout: id + parenthesized label
    rows = site.parse_account_rows("ACCOUNT_ID: ab110003 (Genesis Project)\n")
    assert rows == [("ab110003", "(Genesis Project)")]


def test_discover_account_rows_uses_the_configured_command(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "myaccounts", "cat <<'EOF'\n" + REAL_MYACCOUNT_ID + "EOF\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    rows = site.discover_account_rows()
    assert [account_id for account_id, _ in rows] == ["ab110001", "ab110002", "ab110003"]


def test_real_myaccounts_first_pick_and_alternatives(clean_env, tmp_path, monkeypatch):
    shim = tmp_path / "myaccounts"
    shim.write_text("#!/bin/bash\ncat <<'EOF'\n" + REAL_MYACCOUNT_ID + "EOF\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "ab110001"                      # first listed wins (deterministic)
    assert "ab110002" in why and "ab110003" in why  # alternatives named for an easy override


# ---- probe chain (shims) ----------------------------------------------------------


def _shim(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(0o755)
    return p


@pytest.fixture
def clean_env(monkeypatch):
    for v in ("SBATCH_ACCOUNT", "SLURM_ACCOUNT", "BOXY_ACCOUNT", "BOXY_PARTITION",
              "BOXY_DEFAULT_TIME", "BOXY_SCHEDULER", "ACCOUNT_ID", "BOXY_LICENSE"):
        monkeypatch.delenv(v, raising=False)


def test_account_flag_wins(clean_env):
    assert site.resolve_account("fy111111") == ("fy111111", "--account")


def test_resolve_license_flag_and_config(clean_env, monkeypatch):
    from boxy import config
    assert site.resolve_license("scratchfs:1") == ("scratchfs:1", "--license")
    # NO shipped default (pip-anywhere: a license request only fits sites that
    # gate filesystems behind one) — unset means none.
    config.reset()
    assert site.resolve_license(None) == ("", "")
    monkeypatch.setenv("BOXY_LICENSE", "pscratch:1")
    config.reset()
    assert site.resolve_license(None) == ("pscratch:1", "config site.license")
    monkeypatch.setenv("BOXY_LICENSE", "")                        # explicit empty -> none
    config.reset()
    assert site.resolve_license(None) == ("", "")


def test_account_id_env_bypasses_discovery(clean_env, tmp_path, monkeypatch):
    # $ACCOUNT_ID is a session bypass: it beats myaccounts/config so a scripted run charges a
    # chosen account without the menu.
    _shim(tmp_path, "myaccounts", "echo ab110003\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("ACCOUNT_ID", "fy_session")
    acct, why = site.resolve_account(None)
    assert acct == "fy_session" and why == "$ACCOUNT_ID"


def test_account_from_myaccounts_shim(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "myaccounts", 'echo "ACCOUNT_ID: ab110003  (Project X)"\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "ab110003" and "myaccounts" in why


def test_account_myaccounts_multiple_notes_alternatives(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "myaccounts", 'printf "ab110003\\nab119999\\n"\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "ab110003" and "ab119999" in why      # first picked, rest noted


def test_account_config_beats_command(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "myaccounts", 'echo ab110003\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy_override")
    acct, why = site.resolve_account(None)
    assert acct == "fy_override" and "config" in why


def test_account_env_used_when_no_command(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")        # disable myaccounts
    monkeypatch.setenv("SBATCH_ACCOUNT", "fy_env")
    acct, why = site.resolve_account(None)
    assert acct == "fy_env" and "SBATCH_ACCOUNT" in why


def test_account_sacctmgr_fallback(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")        # skip myaccounts
    _shim(tmp_path, "sacctmgr", 'echo fy_assoc\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("USER", "tester")
    acct, why = site.resolve_account(None)
    assert acct == "fy_assoc" and "sacctmgr" in why


def test_account_none_when_nothing_discovers(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "definitely-not-a-real-binary-xyz")
    acct, why = site.resolve_account(None)
    assert acct is None and "no account discovered" in why


# ---- time defaults ----------------------------------------------------------------


def test_time_defaults(clean_env, monkeypatch):
    # the default walltime is 1 h (turnkey: the job always carries one, so it
    # never blocks on a missing --time); an explicit config value overrides it.
    assert site.resolve_time(None) == ("1:00:00", "config site.default_time")
    monkeypatch.setenv("BOXY_DEFAULT_TIME", "4:00:00")
    assert site.resolve_time(None)[0] == "4:00:00"
    monkeypatch.setenv("BOXY_DEFAULT_TIME", "")   # explicit empty => scheduler's own default
    assert site.resolve_time(None) == (None, "")
    assert site.resolve_time("2:00:00") == ("2:00:00", "--time")   # explicit --time wins


# ---- partition: auto is the DEFAULT (GPU-aware soonest-start) ----------------------

from boxy.schedulers import get_scheduler  # noqa: E402
from boxy.schedulers.base import PartitionInfo  # noqa: E402

# sinfo -o "%R|%a|%F|%G": name | up/down | A/I/O/T nodes | GRES. gpu(6 idle,GPU),
# short(5 idle,CPU-only), batch(2 idle,GPU), down-pt(down).
SINFO = ("gpu|up|2/6/0/8|gpu:a100:8\n"
         "short|up|3/5/0/8|(null)\n"
         "batch|up|8/2/0/10|gpu:v100:4\n"
         "down-pt|down|4/4/0/8|gpu:a100:8\n")


def test_slurm_parse_partitions_reads_idle_up_and_gpu():
    parts = {p.name: p for p in get_scheduler("slurm").parse_partitions(SINFO)}
    assert parts["gpu"].idle_nodes == 6 and parts["gpu"].has_gpu
    assert parts["short"].has_gpu is False                 # (null) GRES -> CPU-only
    assert parts["batch"].has_gpu is True
    assert parts["down-pt"].up is False


def test_rank_gpu_job_offers_only_gpu_partitions():
    parts = get_scheduler("slurm").parse_partitions(SINFO)
    value, why = site.rank_partitions(parts, "slurm", prefer_gpu=True)
    assert value == "gpu,batch"                            # CPU 'short' + DOWN dropped; idle-first
    assert "with GPUs" in why


def test_rank_all_includes_cpu_partitions():
    parts = get_scheduler("slurm").parse_partitions(SINFO)
    value, _ = site.rank_partitions(parts, "slurm", prefer_gpu=False)
    assert value == "gpu,short,batch"                      # every up partition, idle-first


def test_rank_gpu_falls_back_to_all_when_none_identified():
    parts = [PartitionInfo("a", 3, True, False), PartitionInfo("b", 5, True, False)]
    value, why = site.rank_partitions(parts, "slurm", prefer_gpu=True)
    assert value == "b,a" and "no GPU partitions identified" in why


def test_rank_flux_picks_single_best():
    value, _ = site.rank_partitions([PartitionInfo("pdebug", 0, True), PartitionInfo("pbatch", 0, True)], "flux")
    assert value == "pbatch" and "," not in value


def test_flux_parse_partitions_pipe_and_empty_name():
    # pipe-delimited so an anonymous (empty-name) queue doesn't fabricate a
    # phantom queue named after the enabled flag.
    out = get_scheduler("flux").parse_partitions("pbatch|True\n|True\npdebug|False\n")
    names = {p.name: p.up for p in out}
    assert names == {"pbatch": True, "pdebug": False}   # empty-name row skipped


def test_partition_mode_does_not_shadow_real_named_partitions():
    # a site partition literally named 'default'/'site' must be treated as a
    # concrete name, not the off keyword.
    assert site.partition_mode("default") == "set"
    assert site.partition_mode("site") == "set"
    assert site.partition_mode("off") == "off"
    assert site.partition_mode("none") == "off"


def test_rank_none_up_falls_back_to_site_default():
    value, why = site.rank_partitions([PartitionInfo("x", 0, False)], "slurm")
    assert value == "" and "site default" in why


def _sinfo_shim(tmp_path, monkeypatch):
    _shim(tmp_path, "sinfo", "cat <<'EOF'\n" + SINFO + "EOF\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")


def test_default_partition_is_auto_gpu_aware(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    # NO flag, GPU job -> boxy auto-picks GPU partitions (the "don't set it" default)
    value, why = site.resolve_partition(None, "slurm", need_gpu=True)
    assert value == "gpu,batch" and "with GPUs" in why


def test_default_partition_cpu_job_offers_all(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    value, _ = site.resolve_partition(None, "slurm", need_gpu=False)
    assert value == "gpu,short,batch"                      # no GPU filter for a CPU job


def test_partition_all_flag_ignores_gpu_filter(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    value, _ = site.resolve_partition("all", "slurm", need_gpu=True)
    assert value == "gpu,short,batch"


def test_partition_off_uses_site_default(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)                     # sinfo present, but off wins
    assert site.resolve_partition("off", "slurm", need_gpu=True) == (None, "")


def test_partition_explicit_name_wins(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    assert site.resolve_partition("m2000", "slurm", need_gpu=True) == ("m2000", "--partition")


def test_partition_config_concrete_default(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_PARTITION", "gpu")            # pinned concrete default
    assert site.resolve_partition(None, "slurm", need_gpu=True) == ("gpu", "config site.partition")


def test_partition_config_off_disables_auto(clean_env, tmp_path, monkeypatch):
    _sinfo_shim(tmp_path, monkeypatch)
    monkeypatch.setenv("BOXY_PARTITION", "off")
    assert site.resolve_partition(None, "slurm", need_gpu=True) == (None, "")


def test_default_partition_no_sinfo_degrades_quietly(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))              # no sinfo -> site default, no error
    value, why = site.resolve_partition(None, "slurm", need_gpu=True)
    assert value is None and "site default" in why


# ---- Flux single-queue guard ------------------------------------------------------


def test_flux_comma_partition_trimmed_to_first(clean_env, capsys):
    args = argparse.Namespace(account="fy1", partition="short,batch", time=None)
    out, _ = site.resolve_site(args, "flux")
    assert out["partition"] == "short"
    assert "ONE queue" in capsys.readouterr().err


def test_slurm_keeps_comma_partition(clean_env):
    args = argparse.Namespace(account="fy1", partition="short,batch", time=None)
    out, _ = site.resolve_site(args, "slurm")
    assert out["partition"] == "short,batch"


# ---- end-to-end: bare --scheduler slurm auto-fills the account --------------------


def test_zero_account_flag_uses_myaccounts_in_batch_script(clean_env, tmp_path, monkeypatch, capsys):
    _shim(tmp_path, "myaccounts", 'echo ab110003\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.1-8B-Instruct",
               "--scheduler", "slurm", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: account: ab110003 (via myaccounts)" in out
    assert "#SBATCH --account=ab110003" in out            # ...reached the batch script
    assert "#SBATCH --gpus-per-node=1" in out             # 8B card -> 1 GPU (turnkey T1)


# ---- scheduler is abstracted: auto-detect + config precedence ---------------------


def test_pick_scheduler_explicit_flag_wins(clean_env):
    assert site.pick_scheduler("flux\nslurm", "flux") == ("flux", "--scheduler")
    assert site.pick_scheduler("", "slurm") == ("slurm", "--scheduler")


def test_pick_scheduler_config_beats_detection(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_SCHEDULER", "flux")
    # config pins flux even though only slurm is present in `available`
    assert site.pick_scheduler("slurm", None) == ("flux", "config site.scheduler")


def test_pick_scheduler_config_none_disables(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_SCHEDULER", "none")
    assert site.pick_scheduler("slurm", None) == (None, "config site.scheduler=none")


def test_pick_scheduler_auto_detects_sole_present(clean_env):
    # default config is 'auto' -> detect the one that's present
    assert site.pick_scheduler("slurm", None) == ("slurm", "detected")
    assert site.pick_scheduler("flux", None) == ("flux", "detected")


def test_pick_scheduler_both_present_defaults_slurm(clean_env):
    sched, why = site.pick_scheduler("flux\nslurm", None)
    assert sched == "slurm" and "both" in why


def test_pick_scheduler_none_detected(clean_env):
    assert site.pick_scheduler("", None) == (None, "no scheduler detected")


# ---- operational (liveness) detection: the clusterb misidentification fix -----------


def test_pick_scheduler_live_flux_beats_dead_slurm_shims(clean_env):
    # a Flux system with slurm-compat binaries. flux's broker answers (flux-live);
    # slurm's control plane does NOT (sbatch present, no sinfo/scontrol). Flux wins.
    sched, why = site.pick_scheduler("flux-bin\nflux-live\nslurm-bin", None)
    assert sched == "flux" and "Flux broker is live" in why


def test_pick_scheduler_flux_wins_when_slurm_shims_also_answer(clean_env):
    # THE clusterb case: a Flux system whose slurm compat layer is complete enough
    # that sinfo answers too (slurm-live). A live Flux broker is authoritative — a
    # reachable broker means Flux runs the machine and the slurm commands proxy to
    # it (submitting via them yields Flux job ids slurm can't track). Flux wins,
    # and the reason names the compat shims + the override.
    sched, why = site.pick_scheduler("flux-bin\nflux-live\nslurm-bin\nslurm-live", None)
    assert sched == "flux"
    assert "Flux broker is live" in why and "compat shims" in why and "BOXY_SCHEDULER=slurm" in why


def test_pick_scheduler_live_slurm_beats_dead_flux(clean_env):
    # mirror: a Slurm site that also has the flux tool installed (nested jobs) but
    # no running flux instance. slurm partitions are visible -> slurm.
    sched, why = site.pick_scheduler("slurm-bin\nslurm-live\nflux-bin", None)
    assert sched == "slurm" and "Slurm is live" in why


def test_pick_scheduler_real_slurmctld_is_authoritative(clean_env):
    # a real slurmctld answering `scontrol ping` is the strongest Slurm signal.
    sched, why = site.pick_scheduler("slurm-bin\nslurm-ctld", None)
    assert sched == "slurm" and "scontrol ping" in why


def test_pick_scheduler_nested_flux_loses_to_real_slurmctld(clean_env):
    # adversarial-review finding: a REAL Slurm cluster where the user has a personal
    # NESTED flux instance (flux-nested, instance-level >= 1) reachable in a fresh
    # ssh. A nested instance is NOT the machine's scheduler — a live slurmctld wins.
    sched, why = site.pick_scheduler("flux-bin\nflux-nested\nslurm-bin\nslurm-ctld\nslurm-live", None)
    assert sched == "slurm"
    assert "nested Flux" in why and "Slurm runs this machine" in why


def test_pick_scheduler_nested_flux_only_is_used(clean_env):
    # a personal flux instance is the only reachable scheduler (no slurm) -> use it.
    assert site.pick_scheduler("flux-bin\nflux-nested", None) == ("flux", "detected (a Flux instance is reachable)")


def test_pick_scheduler_system_flux_beats_slurmctld_shim(clean_env):
    # adversarial-review finding: a Flux system whose compat layer fakes even
    # `scontrol ping` (slurm-ctld). A live SYSTEM flux broker still wins.
    sched, why = site.pick_scheduler("flux-bin\nflux-live\nslurm-bin\nslurm-ctld\nslurm-live", None)
    assert sched == "flux" and "compat shims" in why


def test_pick_scheduler_sole_live(clean_env):
    assert site.pick_scheduler("flux-bin\nflux-live", None)[0] == "flux"
    assert site.pick_scheduler("slurm-bin\nslurm-live", None)[0] == "slurm"
    assert site.pick_scheduler("slurm-bin\nslurm-ctld", None)[0] == "slurm"


def test_pick_scheduler_both_bin_no_live_defaults_slurm_loudly(clean_env):
    # neither control plane responded -> can't prove which is real; default slurm
    # but say so loudly (the honest fallback, not a silent guess).
    sched, why = site.pick_scheduler("flux-bin\nslurm-bin", None)
    assert sched == "slurm" and "neither control plane responded" in why


def test_pick_scheduler_config_still_overrides_liveness(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_SCHEDULER", "slurm")
    # even with a live Flux broker, explicit config pins slurm (power-user escape)
    assert site.pick_scheduler("flux-bin\nflux-live\nslurm-bin", None) == ("slurm", "config site.scheduler")


def test_remote_scheduler_probe_probes_system_flux_socket_and_level():
    probe = site.remote_scheduler_probe()
    assert "command -v flux" in probe and "command -v sbatch" in probe
    # reaches the SYSTEM flux instance even without FLUX_URI, via the well-known
    # system socket, and distinguishes it from a nested instance by level.
    assert "local:///run/flux/local" in probe
    assert "instance-level" in probe
    assert "flux-live" in probe and "flux-nested" in probe
    assert "scontrol ping" in probe and "slurm-ctld" in probe   # authoritative slurm signal
    assert "sinfo -h" in probe and "slurm-live" in probe
    assert probe.rstrip().endswith("true")   # never non-zero exit -> ssh_capture stays quiet


# ---- GPU GRES convention auto-detected from sinfo (field: clusterd) -------------------


def test_gpu_request_from_gres_single_type():
    sinfo = "gpu|up|2/6/0/8|gpu:a100:8\nbatch|up|8/2/0/10|gpu:a100:4\n"
    assert site.gpu_request_from_gres(sinfo) == ("gres", "a100")


def test_gpu_request_from_gres_untyped():
    sinfo = "gpu|up|2/6/0/8|gpu:8\n"
    assert site.gpu_request_from_gres(sinfo) == ("gres", "")


def test_gpu_request_from_gres_mixed_types_drops_the_type():
    sinfo = "gpu|up|2/6/0/8|gpu:a100:8\nbatch|up|8/2/0/10|gpu:v100:4\n"
    assert site.gpu_request_from_gres(sinfo) == ("gres", "")


def test_gpu_request_from_gres_restricted_to_selected_partitions():
    sinfo = "gpu|up|2/6/0/8|gpu:a100:8\nbatch|up|8/2/0/10|gpu:v100:4\n"
    # only the 'gpu' partition -> its single a100 type
    assert site.gpu_request_from_gres(sinfo, {"gpu"}) == ("gres", "a100")


def test_gpu_request_from_gres_none_when_no_gpu():
    sinfo = "cpu|up|1/9/0/10|(null)\n"
    assert site.gpu_request_from_gres(sinfo) == ("", "")


def test_gpu_request_from_gres_socket_suffix():
    # sinfo may append a socket affinity suffix — still parses the type
    assert site.gpu_request_from_gres("gpu|up|1/1/0/2|gpu:h100:8(S:0-1)\n") == ("gres", "h100")


# ---- model store: scratch-FS pick (never the $HOME quota) ------------------------


def test_pick_model_store_first_with_room():
    from boxy import site

    out = "/sitescratch/users/me 52428800\n/pscratch/me 1073741824\n"   # 50 GB, 1 TB
    path, free, why = site.pick_model_store(out, min_free_gb=100)
    assert path == "/pscratch/me" and free == 1024                   # first to clear 100 GB
    # ladder order (not max free) when the first candidate has room:
    path, free, why = site.pick_model_store(out, min_free_gb=10)
    assert path == "/sitescratch/users/me" and free == 50


def test_pick_model_store_tight_and_empty():
    from boxy import site

    path, free, why = site.pick_model_store("/scratch/me 10485760\n", min_free_gb=100)
    assert path == "/scratch/me" and free == 10 and "only 10 GB" in why   # roomiest, flagged
    path, free, why = site.pick_model_store("", min_free_gb=100)
    assert path == "" and "no shared scratch" in why
    # garbage lines (motd noise, non-absolute paths) never crash the parse
    path, _, _ = site.pick_model_store("Welcome to clustera!\nfoo bar baz\n", min_free_gb=1)
    assert path == ""


def test_model_store_probe_is_posix_and_sticky_first():
    from boxy import site

    cmd = site.model_store_probe(saved="/sitescratch/users/me/boxy")
    assert cmd.index("/sitescratch/users/me/boxy") < cmd.index("$SCRATCH")  # saved pick first
    assert "df -Pk" in cmd and "mkdir -p" in cmd and "boxy" not in cmd.split()[0]


# ---- `boxy generate system`: the cluster's inventory becomes a card --------------


SINFO_NODES = """ampere01|112|512000|gpu:h100:4
ampere01|112|512000|gpu:h100:4
ampere02|112|512000|gpu:h100:4
ampere03|112|512000|gpu:h100:4
cpu001|96|256000|(null)
cpu002|96|256000|
fat01|64|1024000|gpu:a100_40gb:8,craynetwork:1
"""


def test_parse_sinfo_inventory_dedupes_and_finds_modal_gpu_shape():
    from boxy import site

    inv = site.parse_sinfo_inventory(SINFO_NODES)
    assert inv["total_nodes"] == 6                       # ampere01 listed twice -> once
    assert inv["total_gpu_nodes"] == 4
    assert (inv["gpus_per_node"], inv["gpu_type"]) == (4, "h100")   # modal GPU shape
    assert (inv["cpus_per_node"], inv["mem_gb_per_node"]) == (112, 500)
    assert site.parse_sinfo_inventory("") == {}


def test_gpu_vram_table_covers_the_fleet():
    from boxy import site

    assert site.gpu_vram_from_type("h100")[0] == 80
    assert site.gpu_vram_from_type("nvidia_h200")[0] == 141      # clusterc-class parts
    assert site.gpu_vram_from_type("a100_40gb")[0] == 40         # the ambiguous one
    assert site.gpu_vram_from_type("mi300a")[0] == 128           # clusterb-class
    gb, note = site.gpu_vram_from_type("quantum9000")
    assert gb == 0 and "fill in gpu_vram_gb" in note             # unknown -> operator hint


def test_parse_flux_inventory_coarse_totals():
    from boxy import site

    inv = site.parse_flux_inventory("   4 512 16\n")
    assert inv["total_nodes"] == 4 and inv["gpus_per_node"] == 4
    assert site.parse_flux_inventory("garbage\n") == {}


def test_render_system_card_roundtrips_into_the_solver(tmp_path, monkeypatch):
    from boxy import cards, site

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    inv = site.parse_sinfo_inventory(SINFO_NODES)
    text = site.render_system_card("clustera", "slurm", "cuda", "podman", inv,
                                   ["/sitescratch/users/me  214000 GB free"])
    d = cards._user_systems_dir()
    d.mkdir(parents=True)
    (d / "clustera.toml").write_text(text)
    assert cards.system_shape("clustera") == (4, 80, "clustera")         # solver supply side
    assert "total_nodes = 6" in text and "NOT a job request" in text
    assert "cpus_per_node = 112" in text and "mem_gb_per_node = 500" in text
    assert "/sitescratch/users/me" in text                          # storage documented
    assert "a100_40gb" in text                                   # heterogeneity listed


def test_generate_system_cli_end_to_end(tmp_path, monkeypatch, capsys):
    # `boxy generate system --ssh user@clustera`: probes routed to canned outputs,
    # card written where the geometry solver reads it, --force semantics.
    from boxy import cards, remote
    from boxy.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(remote, "ensure_master", lambda host: 0)

    def fake_capture(target, cmd, timeout=30):
        assert target == "user@clustera"
        if "sinfo -h -N" in cmd:
            return 0, SINFO_NODES
        if "instance-level" in cmd or "sbatch" in cmd:
            return 0, "slurm-bin\nslurm-ctld\nslurm-live\n"
        if "rocm-smi" in cmd or "sinfo -h -o %G" in cmd:
            return 0, "cuda\n"
        if "podman" in cmd:
            return 0, "podman\napptainer\n"
        if "df -Pk" in cmd:
            return 0, "/sitescratch/users/me 224395214848\n"
        return 1, ""

    monkeypatch.setattr(remote, "ssh_capture", fake_capture)
    rc = main(["generate", "system", "--ssh", "user@clustera"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: scheduler: slurm" in out
    assert "auto: inventory: 6 nodes (4 with GPUs)" in out
    dest = tmp_path / "boxy" / "cards" / "systems" / "clustera.toml"
    assert dest.exists()
    assert cards.system_shape("clustera") == (4, 80, "clustera")
    assert "sizes against 4x80GB nodes" in out
    # rerun without --force refuses; with --force keeps a .bak
    assert main(["generate", "system", "--ssh", "user@clustera"]) == 1
    assert main(["generate", "system", "--ssh", "user@clustera", "--force"]) == 0
    assert dest.with_suffix(".toml.bak").exists()


def test_gpu_hw_probe_parses_nvidia_rocm_gfx():
    from boxy import site

    t, v, c = site.parse_gpu_hw("NV: NVIDIA H200, 143771 MiB\nNVCOUNT: 4\n")
    assert (t, v, c) == ("NVIDIA H200", 140, 4)              # VRAM measured, no table
    t, v, _ = site.parse_gpu_hw("banner noise\nROCM: Card series: AMD Instinct MI300A\n")
    assert t == "MI300A" and v == 128                        # name -> table
    t, v, _ = site.parse_gpu_hw("GFX: gfx942\n")
    assert t == "mi300a" and v == 128                        # arch -> family -> table
    assert site.parse_gpu_hw("") == ("", 0, 0)


def test_generate_system_typeless_gres_falls_back_to_hw_probe(tmp_path, monkeypatch, capsys):
    # FIELD: 274-node system whose GRES is 'gpu:4' (no type token) — the card
    # came out 'unknown GPU type'. Now the generator loads the rocm module and
    # asks the hardware.
    from boxy import cards, remote
    from boxy.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(remote, "ensure_master", lambda host: 0)

    def fake_capture(target, cmd, timeout=30):
        if "sinfo -h -N" in cmd:
            return 0, "n[001]|96|768000|gpu:4\nn002|96|768000|gpu:4\n"
        if "rocm-smi" in cmd or "nvidia-smi" in cmd:         # the hw probe
            return 0, "ROCM: Card series: AMD Instinct MI300A\nGFX: gfx942\n"
        if "instance-level" in cmd or "sbatch" in cmd:
            return 0, "slurm-bin\nslurm-ctld\nslurm-live\n"
        if "sinfo -h -o %G" in cmd:
            return 0, "rocm\n"
        if "df -Pk" in cmd:
            return 0, ""
        if "podman" in cmd:
            return 0, "podman\n"
        return 1, ""

    monkeypatch.setattr(remote, "ssh_capture", fake_capture)
    rc = main(["generate", "system", "--ssh", "user@clusterb"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpu hardware probe: MI300A (128GB)" in out
    assert cards.system_shape("clusterb") == (4, 128, "clusterb")
    text = (tmp_path / "boxy" / "cards" / "systems" / "clusterb.toml").read_text()
    assert "gpu_vram_gb = 128" in text and "verify the COMPUTE nodes" in text


def test_generate_system_local_machine_without_scheduler(tmp_path, monkeypatch, capsys):
    # `boxy generate system` on a laptop/workstation: no scheduler -> card THIS
    # machine (nproc/meminfo/GPU probe, scheduler 'none', one node).
    from boxy import cards
    from boxy.cli import main

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        import types
        c = cmd[-1]
        if "nproc" in c:
            return types.SimpleNamespace(returncode=0, stdout="32\n", stderr="")
        if "MemTotal" in c:
            return types.SimpleNamespace(returncode=0, stdout="MemTotal: 131072000 kB\n", stderr="")
        if "nvidia-smi" in c or "rocm-smi" in c:
            return types.SimpleNamespace(returncode=0,
                                         stdout="NV: NVIDIA RTX A6000, 49140 MiB\nNVCOUNT: 2\n",
                                         stderr="")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    rc = main(["generate", "system"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "carding the LOCAL machine itself" in out
    assert "32 CPUs, 125GB RAM, 2x NVIDIA RTX A6000 (48GB)" in out
    shape = cards.system_shape(__import__("boxy.jobs", fromlist=["x"]).local_cluster())
    assert shape and shape[0] == 2 and shape[1] == 48


def test_gpu_hw_probe_rejects_nvidia_driver_failure_prose():
    # FIELD: a GPU-less login node has nvidia-smi ON PATH but no driver; its
    # failure message goes to STDOUT and became the card's 'GPU type'. The NV
    # line must be a real csv row ('<name>, NNNN MiB'), its bogus -L count dies
    # with it, and the ROCm answer wins.
    from boxy import site

    field = ("NV: NVIDIA-SMI has failed because it couldn't communicate with the "
             "NVIDIA driver. Make sure that the latest NVIDIA driver is installed "
             "and running.\n"
             "NVCOUNT: 1\n"
             "ROCM: Card series: AMD Instinct MI300A\n")
    assert site.parse_gpu_hw(field) == ("MI300A", 128, 0)
    # same failure with NO rocm answer: cleanly unknown, never the prose
    alone = field.splitlines()[0] + "\nNVCOUNT: 1\n"
    assert site.parse_gpu_hw(alone) == ("", 0, 0)


def test_gpu_hw_probe_prefers_detected_accelerator():
    from boxy import site

    cmd = site.gpu_hw_probe(prefer="rocm")
    assert cmd.index("rocm-smi") < cmd.index("nvidia-smi")   # rocm asked first
    cmd = site.gpu_hw_probe()
    assert cmd.index("nvidia-smi") < cmd.index("rocm-smi")


# ---- local accel ladder (the detect_accel HPC fallback) ---------------------------


def test_accel_from_gpu_type_maps_amd_and_nvidia():
    assert site.accel_from_gpu_type("MI300A") == "rocm"
    assert site.accel_from_gpu_type("gfx942") == "rocm"
    assert site.accel_from_gpu_type("AMD Instinct MI250") == "rocm"
    assert site.accel_from_gpu_type("NVIDIA H100 80GB HBM3") == "cuda"
    assert site.accel_from_gpu_type("") == ""


def test_local_accel_probe_functional_module_probe_first(monkeypatch):
    # FIELD: on the HPC system RamaLama sees nothing (tools behind `module
    # load`, no CDI) — boxy's ladder must answer. The FUNCTIONAL probe (module
    # load rocm; rocm-smi) runs before existence markers, because a stale
    # broken nvidia-smi on PATH would satisfy a marker check on an AMD system.
    monkeypatch.setattr(site, "_local_accel_cache", None)
    monkeypatch.setattr(site, "_hpc_markers", lambda: True)
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        if "module load rocm" in cmd:
            return "ROCM: Card series: AMD Instinct MI300A\n"
        return ""

    monkeypatch.setattr(site, "_run_local_sh", fake_run)
    accel, why = site.local_accel_probe()
    assert accel == "rocm" and "MI300A" in why
    assert len(calls) == 1                      # functional probe answered; markers never ran
    # memoized: a second call must not probe again
    monkeypatch.setattr(site, "_run_local_sh",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-probed")))
    assert site.local_accel_probe() == (accel, why)


def test_local_accel_probe_marker_ladder_on_gpuless_login(monkeypatch):
    # GPU-less login node of a CUDA cluster: no module answers, but the
    # scheduler inventory (stage 2, the same ladder --ssh trusts) names it.
    monkeypatch.setattr(site, "_local_accel_cache", None)
    monkeypatch.setattr(site, "_hpc_markers", lambda: True)
    monkeypatch.setattr(site, "_run_local_sh",
                        lambda cmd, timeout: "cuda\n" if "sinfo -h" in cmd else "")
    accel, why = site.local_accel_probe()
    assert accel == "cuda" and "scheduler inventory" in why


def test_local_accel_probe_laptop_never_module_loads(monkeypatch):
    # No HPC markers (a laptop): the module-loading probe is skipped outright —
    # only the cheap marker snippet runs, and silence stays silence.
    monkeypatch.setattr(site, "_local_accel_cache", None)
    monkeypatch.setattr(site, "_hpc_markers", lambda: False)
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        return ""

    monkeypatch.setattr(site, "_run_local_sh", fake_run)
    assert site.local_accel_probe() == ("", "")
    assert len(calls) == 1 and "module load" not in calls[0]


def test_detect_accel_falls_back_to_hpc_ladder(monkeypatch):
    from boxy import ramalama_shim

    monkeypatch.setattr(ramalama_shim, "_ramalama_accel", lambda: "none")
    monkeypatch.setattr(site, "local_accel_probe",
                        lambda: ("rocm", "hardware probe after module load (MI300A)"))
    assert ramalama_shim.detect_accel() == "rocm"
    assert "MI300A" in ramalama_shim.last_detect_note


def test_detect_accel_ramalama_verdict_wins(monkeypatch):
    from boxy import ramalama_shim

    monkeypatch.setattr(ramalama_shim, "_ramalama_accel", lambda: "cuda")
    monkeypatch.setattr(site, "local_accel_probe",
                        lambda: (_ for _ in ()).throw(AssertionError("ladder must not run")))
    assert ramalama_shim.detect_accel() == "cuda"
    assert ramalama_shim.last_detect_note == ""


# ---- partition-scoped accel probe -------------------------------------------------


def _probe_with_fake_sinfo(tmp_path, monkeypatch, sinfo_body, partition=""):
    b = tmp_path / "probe-bin"
    b.mkdir(exist_ok=True)
    shim = b / "sinfo"
    shim.write_text("#!/bin/bash\n" + sinfo_body)
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    return site._run_local_sh(site.remote_accel_probe(partition=partition), timeout=10).strip()


def test_accel_probe_partition_scopes_the_gres_query(tmp_path, monkeypatch):
    # FIELD ask: pick a partition ('short'), check ITS accelerators before
    # deploying. The -p flag must reach sinfo, and its answer decides.
    body = 'echo "$@" > "$0.args"; case "$*" in *"-p short"*) echo "gpu:mi300a:4";; *) echo "gpu:h100:4";; esac\n'
    out = _probe_with_fake_sinfo(tmp_path, monkeypatch, body, partition="short")
    assert out == "rocm"
    assert "-p short" in (tmp_path / "probe-bin" / "sinfo.args").read_text()
    # same cluster, no partition scope: the cluster-wide answer differs
    assert _probe_with_fake_sinfo(tmp_path, monkeypatch, body) == "cuda"


def test_accel_probe_partition_with_no_gpu_gres_says_nogpu(tmp_path, monkeypatch):
    # A CPU partition answers '(null)' while OTHER partitions carry typed GPU
    # GRES (the site does configure GRES): that is the VERDICT nogpu, never a
    # fall-through to login-node markers — the login node's userland says
    # nothing about what `-p short` can schedule.
    body = 'case "$*" in *"-p short"*) echo "(null)";; *) echo "gpu:h100:4";; esac\n'
    assert _probe_with_fake_sinfo(tmp_path, monkeypatch, body, partition="short") == "nogpu"
    # cluster-wide, an empty answer still falls to the marker ladder
    assert _probe_with_fake_sinfo(tmp_path, monkeypatch, 'echo "(null)"\n') != "nogpu"


def test_accel_probe_partition_no_gres_site_never_says_nogpu(tmp_path, monkeypatch):
    # clusterd-class site: NO GRES config anywhere — '(null)' even on the GPU
    # partition (the partition implies the GPUs). Scoped silence proves
    # nothing there, so no nogpu verdict (a hold would be a false positive).
    body = 'echo "(null)"\n'
    assert _probe_with_fake_sinfo(tmp_path, monkeypatch, body, partition="hopper") != "nogpu"


def test_accel_probe_partition_typeless_gres_falls_to_markers(tmp_path, monkeypatch):
    # 'gpu:4' (typeless GRES, field: clusterd): GPUs exist but the type is
    # unknowable from sinfo — never the nogpu verdict.
    body = 'echo "gpu:4"\n'
    assert _probe_with_fake_sinfo(tmp_path, monkeypatch, body, partition="short") != "nogpu"


def test_parse_remote_accel_accepts_nogpu_after_banner():
    banner = "*** NOTICE: U.S. Government system ***\nauthorized use only\nnogpu\n"
    assert site.parse_remote_accel(banner) == "nogpu"
    assert site.parse_remote_accel("garbage\n") == ""


# ---- one-shot cluster probe (composite report) ------------------------------------


_FULL_REPORT = """*** NOTICE: U.S. Government system — authorized use only ***
===BOXY:SCHED===
flux-bin
flux-live
===BOXY:GRESPART===
(null)
===BOXY:GRESALL===
gpu:4
===BOXY:MARKERS===
rocm-smi
opt-rocm
===BOXY:GPUHW===
ROCM: Card series: AMD Instinct MI300A
===BOXY:INV_SLURM===
===BOXY:INV_FLUX===
274 25000 1096
===BOXY:PARTS_SLURM===
===BOXY:PARTS_FLUX===
===BOXY:ACCOUNT===
ab110001
===BOXY:STORE===
/sitescratch/users/me 900000000
===BOXY:RUNTIME===
podman
===BOXY:IDENT===
cbnode-login1
/users/me
===BOXY:END===
"""


def test_cluster_probe_script_carries_every_section():
    script = site.cluster_probe(partition="short", store_saved="/sitescratch/users/me/boxy")
    for name in ("SCHED", "GRESPART", "GRESALL", "MARKERS", "GPUHW", "INV_SLURM",
                 "INV_FLUX", "PARTS_SLURM", "PARTS_FLUX", "ACCOUNT", "STORE",
                 "RUNTIME", "IDENT", "END"):
        assert f"===BOXY:{name}===" in script
    assert "-p short" in script                     # partition scopes the GRES query
    assert "/sitescratch/users/me/boxy" in script      # sticky store pick rides along
    assert "module load rocm" in script             # the functional hardware probe


def test_split_cluster_report_tolerates_login_banner():
    sections = site.split_cluster_report(_FULL_REPORT)
    assert "Government" not in str(sections.get("SCHED"))   # banner discarded
    assert sections["ACCOUNT"].strip() == "ab110001"
    assert site.split_cluster_report("banner only, no markers") == {}
    assert site.parse_cluster_report("banner only") is None  # -> caller falls back


def test_cluster_facts_accessors_dispatch_to_existing_parsers():
    facts = site.parse_cluster_report(_FULL_REPORT, partition="short")
    assert site.pick_scheduler(facts.sched_avail(), None)[0] == "flux"
    assert facts.hostname() == "cbnode-login1" and facts.home() == "/users/me"
    assert facts.runtimes() == ["podman"]
    path, free, _ = site.pick_model_store(facts.store_out(), 100)
    assert path == "/sitescratch/users/me" and free >= 800
    assert facts.inventory("flux")["total_nodes"] == 274


def test_accel_from_report_ladder():
    # partition-scoped typed GRES wins outright
    s = {"GRESPART": "gpu:mi300a:4", "GRESALL": "gpu:h100:4"}
    assert site.accel_from_report(s, "short") == ("rocm", "partition short GRES")
    # nogpu: scoped silence is a verdict only when the cluster types/shows GPUs
    s = {"GRESPART": "(null)", "GRESALL": "gpu:h100:4"}
    assert site.accel_from_report(s, "short")[0] == "nogpu"
    # clusterd-class: no GRES anywhere -> never nogpu; markers may still answer
    s = {"GRESPART": "(null)", "GRESALL": "(null)", "MARKERS": "nvidia-proc"}
    assert site.accel_from_report(s, "hopper") == ("cuda", "NVIDIA userland/driver markers")
    # the FUNCTIONAL probe outranks markers: broken nvidia-smi prose is
    # rejected, module-loaded rocm-smi answers truthfully (field: clusterb)
    s = {"GRESALL": "gpu:4",
         "GPUHW": "NV: NVIDIA-SMI has failed because it couldn't communicate...\n"
                  "ROCM: Card series: AMD Instinct MI300A",
         "MARKERS": "nvidia-smi"}
    accel, why = site.accel_from_report(s)
    assert accel == "rocm" and "MI300A" in why
    # nothing anywhere -> undecided
    assert site.accel_from_report({}) == ("", "")
