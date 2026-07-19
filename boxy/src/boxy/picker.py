"""Interactive ACCOUNT_ID / charge-account picker.

boxy discovers the accounts a user may charge to (`myaccounts`, sacctmgr) in
`site.py`; historically it silently took the FIRST. This module adds the inline
menu: when several accounts are available and none was named, list them (with the
project text `myaccounts` prints), let the user pick, remember the choice per cluster,
and validate it against the live list so a stale default can't charge an account
the user has lost. Pure stdlib — a numbered prompt, no third-party TUI — and it
NEVER blocks without a TTY, so batch scripts and CI stay safe.

Selection precedence (the caller wires this up): an explicit `--account`, then
`$ACCOUNT_ID`, then config `site.account` — any of these BYPASS the menu entirely. Only
when none is set and more than one account was discovered does the menu appear
(and only on a TTY, unless `site.pick_account=always`)."""

from __future__ import annotations

import re
import sys

from boxy import config

Row = tuple[str, str]  # (account_id, label)


def is_interactive(mode: str | None = None) -> bool:
    """Whether to render the menu. config site.pick_account (or the passed mode):
    'always' forces it, 'never' disables it, 'auto' (default) shows it only when
    both stdin and stdout are a TTY."""
    m = (mode or config.get_str("site.pick_account") or "auto").strip().lower()
    if m == "always":
        return True
    if m == "never":
        return False
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (ValueError, AttributeError):
        return False


def _normalize(accounts) -> list[Row]:
    """Accept a list of (account_id, label) rows OR bare account_id strings; return rows."""
    rows: list[Row] = []
    for a in accounts or []:
        if isinstance(a, (tuple, list)):
            rows.append((str(a[0]), str(a[1]) if len(a) > 1 and a[1] else ""))
        else:
            rows.append((str(a), ""))
    return rows


# ---- remembered per-cluster default -------------------------------------------------


def _state_file(where: str = "", kind: str = "account"):
    from boxy import jobs

    tag = re.sub(r"[^A-Za-z0-9._-]", "_", where) if where else ""
    return jobs._dir() / (f"last_{kind}.{tag}" if tag else f"last_{kind}")


def recall(where: str = "", kind: str = "account") -> str:
    """The last value picked for this cluster/target + kind, or '' if none."""
    try:
        return _state_file(where, kind).read_text().strip()
    except OSError:
        return ""


def remember(value: str, where: str = "", kind: str = "account") -> None:
    """Persist the pick as the per-cluster default (best-effort; never raises)."""
    if not value:
        return
    try:
        _state_file(where, kind).write_text(value + "\n")
    except OSError:
        pass


# ---- the menu -----------------------------------------------------------------------


def render_menu(rows: list[Row], default: str | None = None, *, stream=None,
                title: str = "Select a charge account (ACCOUNT_ID):") -> None:
    stream = stream or sys.stderr
    print(title, file=stream)
    for i, (value, label) in enumerate(rows, 1):
        mark = "  [default]" if default and value == default else ""
        lbl = f"  {label}" if label else ""
        print(f"  {i}) {value}{lbl}{mark}", file=stream)


def _read_choice(n: int, default_index: int | None, *, stream=None,
                 noun: str = "account") -> int | None:
    """A 0-based index chosen at the prompt, or None to keep the default. Bad
    input re-prompts; EOF/Ctrl-C falls back to the default (never a traceback)."""
    stream = stream or sys.stderr
    hint = f" (Enter = {default_index + 1})" if default_index is not None else ""
    prompt = f"{noun} [1-{n}]{hint}: "
    for _ in range(5):
        print(prompt, end="", file=stream, flush=True)
        try:
            raw = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("", file=stream)
            return default_index
        if not raw:
            return default_index
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        print(f"  '{raw}' is not in 1-{n}; try again.", file=stream)
    return default_index


# ---- the decision -------------------------------------------------------------------


def choose_account(accounts, *, explicit: str | None = None, remembered: str | None = None,
                   mode: str | None = None, where: str = "", source: str = "myaccounts",
                   ) -> tuple[str | None, str]:
    """Pick a ACCOUNT_ID. Returns (account_or_None, note). A None account means "the
    caller keeps its existing behavior" — either because an explicit account was
    given (bypass) or nothing was discovered (degrade to the scheduler default).
    A non-empty note is a human decision line the caller prints as
    `auto: account: <acct> (<note>)`. Never blocks without a TTY."""
    rows = _normalize(accounts)
    account_ids = [w for w, _ in rows]

    # an explicit --account / $ACCOUNT_ID / config pin bypasses the menu; validate it
    # against the live list and WARN (not fail) if it isn't there, then let the
    # caller's normal resolution use it.
    if explicit:
        if account_ids and explicit not in account_ids:
            print(f"warning: account {explicit!r} is not among your {source} accounts "
                  f"({', '.join(account_ids)}); submitting anyway.", file=sys.stderr)
        return None, ""

    if not rows:
        return None, ""                                   # caller degrades/errors
    if len(rows) == 1:
        return account_ids[0], f"only account from {source}"

    default = remembered if remembered in account_ids else None

    if not is_interactive(mode):
        if default:
            return default, f"remembered default of {len(rows)} from {source}"
        return account_ids[0], (f"first of {len(rows)} from {source}; "
                          f"use --account or export ACCOUNT_ID to choose")

    default_index = account_ids.index(default) if default else None
    render_menu(rows, default=default)
    idx = _read_choice(len(rows), default_index)
    if idx is None:
        idx = default_index if default_index is not None else 0
    pick = account_ids[idx]
    remember(pick, where)
    return pick, f"you picked {idx + 1} of {len(rows)} from {source}"


def choose_partition(partitions, *, explicit: str | None = None, remembered: str | None = None,
                     mode: str | None = None, where: str = "", source: str = "sinfo",
                     allow_all: bool = True) -> tuple[str | None, str]:
    """Pick ONE partition when several are available (parallel to choose_account).
    `partitions` is a list of names (soonest-start order) or (name, label) rows.
    Returns (value, note): a single partition; OR the full comma-list when the user
    keeps 'all' / non-interactively (preserving boxy's soonest-start default); OR
    (None, '') when --partition was given or there's nothing to choose. allow_all
    offers an 'all' entry (Slurm takes a comma-list; Flux takes ONE queue -> False)."""
    rows = _normalize(partitions)
    names = [n for n, _ in rows]
    full = ",".join(names)
    if explicit or not names:
        return None, ""
    if len(names) == 1:
        return names[0], f"only partition ({source})"
    if not is_interactive(mode):
        return ((full, f"all {len(names)} partitions from {source} (soonest-start)")
                if allow_all else (names[0], f"first of {len(names)} from {source}"))

    menu = list(rows)
    if allow_all:
        menu.append(("all", f"any of {full} — Slurm starts wherever a slot frees first"))
    labels = [m for m, _ in menu]
    default = remembered if remembered in labels else ("all" if allow_all else names[0])
    render_menu(menu, default=default,
                title=f"Select a partition ({len(names)} available, soonest-start):")
    idx = _read_choice(len(menu), labels.index(default), noun="partition")
    if idx is None:
        idx = labels.index(default)
    pick = menu[idx][0]
    if pick != "all":
        remember(pick, where=where, kind="partition")
    if pick == "all":
        return full, f"all {len(names)} from {source}"
    return pick, f"you picked {pick} of {len(names)} from {source}"
