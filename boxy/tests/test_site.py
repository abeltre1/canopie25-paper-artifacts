"""Site auto-resolution — account discovery (mywcid / env / sacctmgr, via bash
shims on PATH), the config defaults for partition/time, the Flux single-queue
guard, and the end-to-end dryrun where a bare `--scheduler slurm` picks up the
account from `mywcid` with no --account flag."""

import argparse
import os

import pytest

from boxy import site
from boxy.cli import main

# ---- account parser ---------------------------------------------------------------


@pytest.mark.parametrize("text,first", [
    ("fy260064\n", "fy260064"),
    ("WCID: fy260064  (My Project)\n", "fy260064"),
    ("Account      Project\nfy260064     Cool Stuff\n", "fy260064"),
    ("no accounts here\n", None),
])
def test_parse_accounts(text, first):
    got = site.parse_accounts(text)
    assert (got[0] if got else None) == first


def test_parse_accounts_multiple_in_order():
    assert site.parse_accounts("fy260064\nfy999999\nfy260064\n") == ["fy260064", "fy999999"]


# The REAL mywcid output (field sample, 2026-07): header row, dashed rule, data
# rows where the DESCRIPTION starts with a bare numeric id right after the
# account, a privilege-less 'none' row, and a trailing caps note repeating the
# first account in uppercase.
REAL_MYWCID = """\
      User    Account                              Description     Parent
---------- ---------- ---------------------------------------- --------------------
     jdoe   fy140001        103732 system software and tools                   nd
     jdoe   fy140252      135101 common computing environment                   nd
     jdoe   fy260064         240928 the genesis project obbba                   nd
     jdoe       none       default account, no job privileges
  The Account could be on Caps too: FY140001
"""


def test_parse_accounts_real_mywcid_table():
    got = site.parse_accounts(REAL_MYWCID)
    # all three real accounts, in order; the bare description ids (103732…) are
    # NOT mistaken for accounts; the caps-note duplicate FY140001 is deduped
    # case-insensitively; the header/'none' rows contribute nothing.
    assert got == ["fy140001", "fy140252", "fy260064"]


def test_parse_account_rows_keeps_labels_for_the_menu():
    rows = site.parse_account_rows(REAL_MYWCID)
    # same three ids in order, each paired with the project text mywcid prints
    # (the leading numeric description-id is dropped from the label).
    ids = [wcid for wcid, _ in rows]
    assert ids == ["fy140001", "fy140252", "fy260064"]
    labels = {wcid: label for wcid, label in rows}
    assert labels["fy140001"].startswith("system software")
    assert "135101" not in labels["fy140252"]                 # numeric desc-id stripped
    assert "common computing environment" in labels["fy140252"]


def test_parse_account_rows_labelled_layout():
    # the `WCID: fy... (Project)` layout: id + parenthesized label
    rows = site.parse_account_rows("WCID: fy260064 (Genesis Project)\n")
    assert rows == [("fy260064", "(Genesis Project)")]


def test_discover_account_rows_uses_the_configured_command(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "mywcid", "cat <<'EOF'\n" + REAL_MYWCID + "EOF\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    rows = site.discover_account_rows()
    assert [wcid for wcid, _ in rows] == ["fy140001", "fy140252", "fy260064"]


def test_real_mywcid_first_pick_and_alternatives(clean_env, tmp_path, monkeypatch):
    shim = tmp_path / "mywcid"
    shim.write_text("#!/bin/bash\ncat <<'EOF'\n" + REAL_MYWCID + "EOF\n")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "fy140001"                      # first listed wins (deterministic)
    assert "fy140252" in why and "fy260064" in why  # alternatives named for an easy override


# ---- probe chain (shims) ----------------------------------------------------------


def _shim(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/bash\n" + body)
    p.chmod(0o755)
    return p


@pytest.fixture
def clean_env(monkeypatch):
    for v in ("SBATCH_ACCOUNT", "SLURM_ACCOUNT", "BOXY_ACCOUNT", "BOXY_PARTITION",
              "BOXY_DEFAULT_TIME", "BOXY_SCHEDULER", "WCID", "BOXY_LICENSE"):
        monkeypatch.delenv(v, raising=False)


def test_account_flag_wins(clean_env):
    assert site.resolve_account("fy111111") == ("fy111111", "--account")


def test_resolve_license_flag_and_config(clean_env, monkeypatch):
    from boxy import config
    assert site.resolve_license("tscratch:1") == ("tscratch:1", "--license")
    # default (BOXY_LICENSE unset by clean_env) -> the tscratch:1 default
    config.reset()
    assert site.resolve_license(None) == ("tscratch:1", "config site.license")
    monkeypatch.setenv("BOXY_LICENSE", "pscratch:1")
    config.reset()
    assert site.resolve_license(None) == ("pscratch:1", "config site.license")
    monkeypatch.setenv("BOXY_LICENSE", "")                        # explicit empty -> none
    config.reset()
    assert site.resolve_license(None) == ("", "")


def test_wcid_env_bypasses_discovery(clean_env, tmp_path, monkeypatch):
    # $WCID is a session bypass: it beats mywcid/config so a scripted run charges a
    # chosen account without the menu.
    _shim(tmp_path, "mywcid", "echo fy260064\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("WCID", "fy_session")
    acct, why = site.resolve_account(None)
    assert acct == "fy_session" and why == "$WCID"


def test_account_from_mywcid_shim(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "mywcid", 'echo "WCID: fy260064  (Project X)"\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "fy260064" and "mywcid" in why


def test_account_mywcid_multiple_notes_alternatives(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "mywcid", 'printf "fy260064\\nfy999999\\n"\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    acct, why = site.resolve_account(None)
    assert acct == "fy260064" and "fy999999" in why      # first picked, rest noted


def test_account_config_beats_command(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "mywcid", 'echo fy260064\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_ACCOUNT", "fy_override")
    acct, why = site.resolve_account(None)
    assert acct == "fy_override" and "config" in why


def test_account_env_used_when_no_command(clean_env, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")        # disable mywcid
    monkeypatch.setenv("SBATCH_ACCOUNT", "fy_env")
    acct, why = site.resolve_account(None)
    assert acct == "fy_env" and "SBATCH_ACCOUNT" in why


def test_account_sacctmgr_fallback(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_ACCOUNT_COMMAND", "")        # skip mywcid
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


def test_zero_account_flag_uses_mywcid_in_batch_script(clean_env, tmp_path, monkeypatch, capsys):
    _shim(tmp_path, "mywcid", 'echo fy260064\n')
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["serve", "hf://meta-llama/Llama-3.1-8B-Instruct",
               "--scheduler", "slurm", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "auto: account: fy260064 (via mywcid)" in out
    assert "#SBATCH --account=fy260064" in out            # ...reached the batch script
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


# ---- operational (liveness) detection: the eldorado misidentification fix -----------


def test_pick_scheduler_live_flux_beats_dead_slurm_shims(clean_env):
    # a Flux system with slurm-compat binaries. flux's broker answers (flux-live);
    # slurm's control plane does NOT (sbatch present, no sinfo/scontrol). Flux wins.
    sched, why = site.pick_scheduler("flux-bin\nflux-live\nslurm-bin", None)
    assert sched == "flux" and "Flux broker is live" in why


def test_pick_scheduler_flux_wins_when_slurm_shims_also_answer(clean_env):
    # THE eldorado case: a Flux system whose slurm compat layer is complete enough
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


# ---- GPU GRES convention auto-detected from sinfo (field: kahuna) -------------------


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
