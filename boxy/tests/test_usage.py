"""A wrong command line must diagnose, not dump.

argparse's default for an unknown subcommand printed all 31 subcommands in the
usage line and all 31 again inside the error, and never said what the input
looked like. These tests pin the replacement.
"""

import pytest

from boxy import usage
from boxy.cli import build_parser, main

KNOWN = sorted(build_parser()._subcommand_names())


# ---- the case that prompted this ------------------------------------------------------


def test_escaped_space_is_named_as_a_quoting_problem():
    """`boxy serve\\ meta-llama/Llama-3.1-8B` sends ONE argv token containing a
    space. argparse reported the whole string as an invalid choice, leaving the
    user to spot a backslash in their own scrollback. boxy knows the token starts
    with a real subcommand and says so."""
    msg = usage.invalid_subcommand_message("serve meta-llama/Llama-3.1-8B-Instruct", KNOWN)
    assert "arrived as ONE argument" in msg
    assert "escaped or quoted" in msg
    assert "boxy serve meta-llama/Llama-3.1-8B-Instruct" in msg


def test_escaped_space_only_fires_for_a_real_subcommand():
    # 'nonsense foo' must NOT be explained as a quoting problem
    problem, _ = usage.diagnose("nonsense foo", KNOWN)
    assert "ONE argument" not in problem


@pytest.mark.parametrize("typo,expected", [
    ("serv", "serve"), ("serve ", "serve"), ("lsit", "list"), ("benchmark", "bench"),
])
def test_near_miss_suggests_the_real_command(typo, expected):
    _, tries = usage.diagnose(typo.strip(), KNOWN)
    assert f"boxy {expected}" in tries


def test_a_model_in_the_command_slot_suggests_the_verb():
    """Forgetting the verb is a different mistake from mistyping one."""
    problem, tries = usage.diagnose("hf://meta-llama/Llama-3.1-8B", KNOWN)
    assert "looks like a model, not a command" in problem
    assert "boxy serve hf://meta-llama/Llama-3.1-8B" in tries
    assert "boxy pull hf://meta-llama/Llama-3.1-8B" in tries


def test_unknown_word_says_so_without_pretending_to_know():
    problem, tries = usage.diagnose("xyzzy", KNOWN)
    assert "not a boxy command" in problem
    assert tries == []


# ---- the list stays scannable, and cannot drift --------------------------------------


def test_every_subcommand_is_filed_under_a_group():
    """The drift guard. GROUPS is data; the parser is truth. A new subcommand
    that nobody files would otherwise vanish from the grouped help."""
    filed = {name for _label, names in usage.GROUPS for name in names}
    missing = set(KNOWN) - filed
    assert not missing, f"add these to usage.GROUPS: {sorted(missing)}"


def test_groups_name_no_command_that_does_not_exist():
    filed = {name for _label, names in usage.GROUPS for name in names}
    assert not filed - set(KNOWN), f"usage.GROUPS lists non-commands: {sorted(filed - set(KNOWN))}"


def test_an_unfiled_command_still_appears():
    """Belt and braces: even if the guard above is somehow bypassed, a command
    the parser knows must never be invisible."""
    text = usage.command_help(KNOWN + ["brand-new-verb"])
    assert "brand-new-verb" in text


def test_help_wraps_instead_of_scrolling_sideways(monkeypatch):
    monkeypatch.setattr(usage.shutil, "get_terminal_size", lambda _d=None: __import__("os").terminal_size((60, 24)))
    for line in usage.command_help(KNOWN).splitlines():
        assert len(line) <= 100


# ---- the parser itself ---------------------------------------------------------------


def test_usage_line_no_longer_dumps_every_subcommand():
    line = build_parser().format_usage()
    assert line.strip() == "usage: boxy [--version] <command> [options]"
    assert "serve" not in line, "31 names before anything actionable is the bug"


def test_invalid_subcommand_exits_2_and_writes_to_stderr(capsys):
    with pytest.raises(SystemExit) as e:
        main(["serve meta-llama/Llama-3.1-8B-Instruct", "--ssh", "host"])
    assert e.value.code == 2
    cap = capsys.readouterr()
    assert "arrived as ONE argument" in cap.err
    assert cap.out == "", "diagnostics belong on stderr"


def test_a_valid_command_is_untouched(capsys):
    assert main(["config"]) == 0
    assert "not a boxy command" not in capsys.readouterr().err
