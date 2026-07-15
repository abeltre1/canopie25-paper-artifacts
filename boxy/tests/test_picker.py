"""Unit tests for the interactive WCID/account picker (src/boxy/picker.py)."""

import builtins

import pytest

from boxy import jobs, picker

ROWS = [("fy140001", "system software"), ("fy140252", "advanced computing"),
        ("fy260064", "ml research")]


@pytest.fixture
def jobsdir(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    return tmp_path


def _answer(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr(builtins, "input", lambda *a, **k: next(it))


# ---- the zero / one / explicit shortcuts --------------------------------------------


def test_zero_accounts_returns_none():
    assert picker.choose_account([], mode="never") == (None, "")


def test_single_account_auto_selects():
    pick, note = picker.choose_account([("fy1", "solo")], mode="always")
    assert pick == "fy1" and "only account" in note


def test_explicit_bypasses_and_returns_none():
    # an explicit account is honored by the caller's own resolution; the picker
    # stays out of the way (returns None) and does not prompt.
    assert picker.choose_account(ROWS, explicit="fy140252", mode="always") == (None, "")


def test_explicit_not_in_list_warns_but_proceeds(capsys):
    pick, note = picker.choose_account(ROWS, explicit="fy999", mode="always")
    assert pick is None
    assert "not among your mywcid accounts" in capsys.readouterr().err


# ---- non-interactive: never blocks --------------------------------------------------


def test_non_interactive_takes_first_of_many():
    pick, note = picker.choose_account(ROWS, mode="never")
    assert pick == "fy140001"
    assert "first of 3" in note and "--account or export WCID" in note


def test_non_interactive_prefers_a_valid_remembered_default():
    pick, note = picker.choose_account(ROWS, remembered="fy260064", mode="never")
    assert pick == "fy260064" and "remembered default" in note


def test_non_interactive_ignores_a_stale_remembered_default():
    # a remembered account no longer in the live list is dropped (can't charge an
    # account you've lost) — falls back to the first.
    pick, _ = picker.choose_account(ROWS, remembered="fyGONE", mode="never")
    assert pick == "fy140001"


# ---- interactive numbered menu ------------------------------------------------------


def test_interactive_pick_by_number(jobsdir, monkeypatch, capsys):
    _answer(monkeypatch, "2")
    pick, note = picker.choose_account(ROWS, mode="always", where="hops")
    assert pick == "fy140252"
    assert "you picked 2 of 3" in note
    err = capsys.readouterr().err
    assert "Select a charge account" in err and "2) fy140252" in err
    # the pick is remembered per-target for next time
    assert picker.recall("hops") == "fy140252"


def test_interactive_enter_reuses_remembered_default(jobsdir, monkeypatch):
    _answer(monkeypatch, "")                       # bare Enter
    pick, _ = picker.choose_account(ROWS, remembered="fy260064", mode="always", where="hops")
    assert pick == "fy260064"


def test_interactive_bad_then_good_input_reprompts(jobsdir, monkeypatch, capsys):
    _answer(monkeypatch, "9", "banana", "3")
    pick, _ = picker.choose_account(ROWS, mode="always")
    assert pick == "fy260064"
    assert "is not in 1-3" in capsys.readouterr().err


def test_interactive_eof_falls_back_to_first(jobsdir, monkeypatch):
    def boom(*a, **k):
        raise EOFError

    monkeypatch.setattr(builtins, "input", boom)
    pick, _ = picker.choose_account(ROWS, mode="always")
    assert pick == "fy140001"                       # default index 0, no traceback


# ---- is_interactive gating ----------------------------------------------------------


def test_is_interactive_modes(monkeypatch):
    assert picker.is_interactive("always") is True
    assert picker.is_interactive("never") is False


def test_is_interactive_auto_follows_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert picker.is_interactive("auto") is True
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    assert picker.is_interactive("auto") is False


# ---- remember / recall roundtrip ----------------------------------------------------


def test_remember_recall_roundtrip_is_per_target(jobsdir):
    picker.remember("fy1", where="hops")
    picker.remember("fy2", where="eldorado")
    assert picker.recall("hops") == "fy1"
    assert picker.recall("eldorado") == "fy2"
    assert picker.recall("unknown") == ""           # missing -> empty, no raise


def test_recall_survives_a_missing_jobs_dir(jobsdir):
    assert picker.recall("nope") == ""
    # and the state file lives under the (per-cluster) jobs dir
    picker.remember("fy1", where="hops")
    assert (jobs._dir() / "last_account.hops").exists()
