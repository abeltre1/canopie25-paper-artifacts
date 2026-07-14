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


# ---- partition / time defaults ----------------------------------------------------


def test_partition_and_time_defaults(clean_env, monkeypatch):
    assert site.resolve_partition(None) == (None, "")
    assert site.resolve_time(None) == (None, "")
    monkeypatch.setenv("BOXY_PARTITION", "gpu")
    monkeypatch.setenv("BOXY_DEFAULT_TIME", "4:00:00")
    assert site.resolve_partition(None)[0] == "gpu"
    assert site.resolve_time(None)[0] == "4:00:00"


def test_explicit_partition_wins_over_auto_machinery(clean_env):
    assert site.resolve_partition("gpu,short", "slurm") == ("gpu,short", "--partition")


# ---- --partition auto: soonest-start discovery ------------------------------------

SINFO = ("short    up   3/5/0/8\n"        # 5 idle
         "batch    up   8/2/0/10\n"       # 2 idle
         "gpu      up   0/0/0/4\n"        # 0 idle
         "down-pt  down 4/4/0/8\n")       # down -> excluded


def test_slurm_parse_partitions_aggregates_idle():
    from boxy.schedulers import get_scheduler

    parts = dict((n, (idle, up)) for n, idle, up in get_scheduler("slurm").parse_partitions(SINFO))
    assert parts["short"] == (5, True)
    assert parts["gpu"] == (0, True)
    assert parts["down-pt"][1] is False


def test_rank_partitions_slurm_is_idle_first_comma_list():
    parts = [("short", 5, True), ("batch", 2, True), ("gpu", 0, True), ("down-pt", 4, False)]
    value, why = site.rank_partitions(parts, "slurm")
    assert value == "short,batch,gpu"          # idle-first; the DOWN partition dropped
    assert "soonest-start" in why and "short" in why


def test_rank_partitions_flux_picks_single_best():
    value, why = site.rank_partitions([("pdebug", 0, True), ("pbatch", 0, True)], "flux")
    assert value == "pbatch"                    # one queue only, deterministic (name order)
    assert "," not in value


def test_rank_partitions_none_up_falls_back():
    value, why = site.rank_partitions([("x", 0, False)], "slurm")
    assert value == "" and "site default" in why


def test_resolve_partition_auto_runs_sinfo(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "sinfo", "cat <<'EOF'\n" + SINFO + "EOF\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    value, why = site.resolve_partition("auto", "slurm")
    assert value == "short,batch,gpu"
    assert "soonest-start" in why


def test_resolve_partition_auto_from_config(clean_env, tmp_path, monkeypatch):
    _shim(tmp_path, "sinfo", "cat <<'EOF'\n" + SINFO + "EOF\n")
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_PARTITION", "auto")           # config default = auto
    assert site.resolve_partition(None, "slurm")[0] == "short,batch,gpu"


def test_resolve_partition_auto_no_sinfo_degrades(clean_env, monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent-dir-xyz")     # no sinfo anywhere
    value, why = site.resolve_partition("auto", "slurm")
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
