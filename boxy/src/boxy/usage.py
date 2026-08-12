"""A readable answer when the command line is wrong.

argparse's default for an unknown subcommand prints the full brace-list of every
subcommand in the usage line, then repeats all of them inside the error, and says
nothing about what the user actually did. With 31 subcommands that is a wall of
text in which the one useful fact — what went wrong — is the hardest part to
find:

    usage: boxy [-h] [--version]
                {info,config,examples,cards,push,trust,wheels,bundle,app,doctor,serve,...}
    boxy: error: argument subcommand: invalid choice: 'serve meta-llama/Llama-3.1-8B'
    (choose from info, config, examples, cards, push, trust, wheels, bundle, app, ...)

This module replaces that with a short man-page-shaped answer: what you typed,
what is wrong with it, the fix, and the commands GROUPED so the list can be
scanned instead of read.

The groups are DATA here but the membership is checked against the live parser
by a test, so a new subcommand cannot quietly go missing from the help.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import sys

# Commands in the order someone meets them, not alphabetical. A flat list of 31
# names is a wall; five short groups can be scanned.
GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("serve & operate", ("serve", "run", "list", "stop", "logs", "attach", "curl", "open", "alloc")),
    ("models", ("pull", "build", "stage", "push", "bundle", "cards", "wheels", "trust")),
    ("measure", ("bench", "sweep", "results", "plot")),
    ("this machine", ("info", "doctor", "config", "examples")),
    ("emit & extend", ("generate", "launch", "app", "router", "unshare", "clean")),
]

_MAX_SUGGESTIONS = 3


def _wrap(names: tuple[str, ...], indent: int, width: int) -> str:
    """Fill names across lines, so a narrow terminal does not scroll sideways."""
    lines: list[str] = []
    current = ""
    for name in names:
        candidate = f"{current} {name}" if current else name
        if len(candidate) + indent > width:
            lines.append(current)
            current = name
        else:
            current = candidate
    if current:
        lines.append(current)
    pad = " " * indent
    return f"\n{pad}".join(lines)


def command_help(known: list[str] | None = None) -> str:
    """The grouped command list. `known` (from the live parser) is used to append
    anything GROUPS has not been told about, so a new subcommand still appears
    even if someone forgets to file it."""
    width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 100))
    label_w = max(len(label) for label, _ in GROUPS) + 2
    out: list[str] = []
    filed: set[str] = set()
    for label, names in GROUPS:
        shown = tuple(n for n in names if known is None or n in known)
        filed.update(shown)
        if shown:
            out.append(f"  {label:<{label_w}}{_wrap(shown, label_w + 2, width)}")
    if known:
        rest = tuple(n for n in known if n not in filed)
        if rest:
            out.append(f"  {'other':<{label_w}}{_wrap(rest, label_w + 2, width)}")
    return "\n".join(out)


def diagnose(bad: str, known: list[str]) -> tuple[str, list[str]]:
    """(what is wrong, what to try) for an unrecognised subcommand.

    The interesting case is the one that produced this module. A shell escape
    like `boxy serve\\ meta-llama/Llama-3.1-8B` sends ONE argv token containing a
    space, so argparse reports the whole string as the invalid choice. boxy knows
    that token starts with a real subcommand and can say so, rather than making
    the user spot a backslash in their own scrollback.
    """
    head, _, rest = bad.partition(" ")
    if rest and head in known:
        return (f"{head!r} and {rest.split()[0]!r} arrived as ONE argument — the space between "
                f"them was escaped or quoted, so the shell did not split them.",
                [f"boxy {head} {rest}"])

    close = difflib.get_close_matches(bad, known, n=_MAX_SUGGESTIONS, cutoff=0.6)
    if close:
        return (f"{bad!r} is not a boxy command.",
                [f"boxy {c}" for c in close])

    # A path or URI in the subcommand slot means the verb was forgotten.
    if "/" in bad or "://" in bad:
        return (f"{bad!r} looks like a model, not a command — boxy needs the verb first.",
                [f"boxy serve {bad}", f"boxy pull {bad}"])

    return (f"{bad!r} is not a boxy command.", [])


def invalid_subcommand_message(bad: str, known: list[str]) -> str:
    """The whole man-page-shaped block, as a string (so it is testable)."""
    problem, tries = diagnose(bad, known)
    parts = [f"boxy: {problem}"]
    if tries:
        parts.append("")
        parts.append("  did you mean:")
        parts += [f"    {t}" for t in tries]
    parts += ["", "COMMANDS", command_help(known), "",
              "  boxy <command> --help     what that command takes",
              "  boxy info                 what boxy detects on this machine"]
    return "\n".join(parts)


class BoxyParser(argparse.ArgumentParser):
    """argparse, minus the two behaviours that make a wrong command unreadable.

    format_usage: the default inlines every subcommand into the usage line. That
    is 31 names before the reader reaches anything actionable, and it is printed
    again by the error. One placeholder instead.

    error: argparse's own invalid-choice text re-lists every command and never
    says what the input looked like. Ours diagnoses first and lists second.
    """

    def format_usage(self) -> str:
        if self.prog == "boxy":
            return "usage: boxy [--version] <command> [options]\n"
        return super().format_usage()

    def error(self, message: str) -> None:  # type: ignore[override]
        marker = "invalid choice: "
        if "argument subcommand" in message and marker in message:
            bad = message.split(marker, 1)[1].split(" (choose from", 1)[0].strip().strip("'\"")
            known = self._subcommand_names()
            print(invalid_subcommand_message(bad, known), file=sys.stderr)
            raise SystemExit(2)
        super().error(message)

    def _subcommand_names(self) -> list[str]:
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001 — argparse has no public accessor
                return list(action.choices)
        return []
