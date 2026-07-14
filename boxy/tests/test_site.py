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
