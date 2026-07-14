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
              "BOXY_DEFAULT_TIME"):
        monkeypatch.delenv(v, raising=False)


def test_account_flag_wins(clean_env):
    assert site.resolve_account("fy111111") == ("fy111111", "--account")


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
    assert site.resolve_time(None) == (None, "")
    monkeypatch.setenv("BOXY_DEFAULT_TIME", "4:00:00")
    assert site.resolve_time(None)[0] == "4:00:00"


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
