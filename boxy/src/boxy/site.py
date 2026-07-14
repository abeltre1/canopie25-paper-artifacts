"""Site auto-resolution — the turnkey account/partition/time discovery.

A novice runs `boxy serve MODEL --scheduler slurm` and the job must actually be
accepted by the scheduler, which usually means a charge account. boxy fills the
site knobs the user didn't pass, each printing an `auto:` decision line, and an
explicit flag always wins.

Account probe chain (first hit wins):
  1. --account flag                          (handled by the caller; passed in)
  2. config site.account                     (BOXY_ACCOUNT / [site].account)
  3. site.account_command                    (default `mywcid`, Sandia's WC-ID lister)
  4. $SBATCH_ACCOUNT / $SLURM_ACCOUNT
  5. `sacctmgr show assoc user=$USER ...`     (single assoc auto-picks; many -> first + note)
  6. none -> omit (the scheduler uses its own site default)

All external commands run on the LOGIN node (where `_serve_submission` runs,
including under --ssh), are best-effort (missing binary / timeout -> skip), and
are shim-testable (a bash `mywcid`/`sacctmgr` on PATH). Partition/time come from
config defaults only (no probing by default; a system card can pin them).
"""

from __future__ import annotations

import os
import re
import subprocess

from boxy import config

# An account/WC-ID token: optional letters then >=4 digits (fy260064, FY260064,
# 12345678). Kept tight so prose lines in `mywcid` output don't match.
_ACCOUNT_RE = re.compile(r"\b([A-Za-z]{0,4}\d{4,})\b")


def _run(argv: list[str], timeout: float = 8) -> str:
    """stdout of `argv`, or "" on any failure (missing binary, nonzero, timeout).
    Never raises — discovery must never break a submission."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    # getattr guard: a test that stubs subprocess.run module-globally may return
    # a bare object; treat anything without a zero returncode as no output.
    return (p.stdout or "") if getattr(p, "returncode", 1) == 0 else ""


def parse_accounts(text: str) -> list[str]:
    """Account-looking tokens from a command's output, in order, de-duplicated.
    Tolerant of labels/columns: `mywcid` and `sacctmgr` layouts both reduce to
    'the first token per line that looks like an account'."""
    out: list[str] = []
    for line in text.splitlines():
        m = _ACCOUNT_RE.search(line)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _account_from_command() -> tuple[str | None, list[str]]:
    cmd = config.get_str("site.account_command").strip()
    if not cmd:
        return None, []
    accounts = parse_accounts(_run(cmd.split()))
    return (accounts[0] if accounts else None), accounts


def _account_from_sacctmgr() -> tuple[str | None, list[str]]:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if not user:
        return None, []
    out = _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}", "format=account"])
    accounts: list[str] = []
    for line in out.splitlines():
        tok = line.strip()
        if tok and tok not in accounts:
            accounts.append(tok)
    return (accounts[0] if accounts else None), accounts


def resolve_account(explicit: str | None) -> tuple[str | None, str]:
    """(account, provenance). None means 'let the scheduler default decide'."""
    if explicit:
        return explicit, "--account"
    cfg = config.get_str("site.account").strip()
    if cfg:
        return cfg, "config site.account"
    acct, alts = _account_from_command()
    if acct:
        cmd = config.get_str("site.account_command").strip()
        extra = f"; also: {', '.join(alts[1:])}" if len(alts) > 1 else ""
        return acct, f"{cmd}{extra}"
    for env in ("SBATCH_ACCOUNT", "SLURM_ACCOUNT"):
        v = os.environ.get(env)
        if v:
            return v, f"${env}"
    acct, alts = _account_from_sacctmgr()
    if acct:
        extra = f"; also: {', '.join(alts[1:])}" if len(alts) > 1 else ""
        return acct, f"sacctmgr assoc{extra}"
    return None, ("no account discovered (mywcid / $SBATCH_ACCOUNT / sacctmgr) — "
                  "the scheduler will use its site default; pass --account if it rejects the job")


def resolve_partition(explicit: str | None) -> tuple[str | None, str]:
    if explicit:
        return explicit, "--partition"
    cfg = config.get_str("site.partition").strip()
    if cfg:
        return cfg, "config site.partition"
    return None, ""


def resolve_time(explicit: str | None) -> tuple[str | None, str]:
    if explicit:
        return explicit, "--time"
    cfg = config.get_str("site.default_time").strip()
    if cfg:
        return cfg, "config site.default_time"
    return None, ""


def resolve_site(args, scheduler_name: str) -> tuple[dict, list[str]]:
    """Fill account/partition/time for a submission. Returns ({kind: value},
    decision_lines). Only non-empty values are returned. Applies the Flux
    single-queue guard: Slurm accepts a comma-list of partitions, Flux's
    --queue takes exactly ONE, so a comma'd partition is trimmed to the first
    with a warning (field failure: `--partition=short,batch` on Flux)."""
    out: dict = {}
    decisions: list[str] = []

    acct, why = resolve_account(getattr(args, "account", None))
    if acct:
        out["account"] = acct
        if why != "--account":
            decisions.append(f"account: {acct} (via {why})")
    else:
        decisions.append(f"account: {why}")

    part, pwhy = resolve_partition(getattr(args, "partition", None))
    if part:
        if scheduler_name == "flux" and "," in part:
            first = part.split(",")[0].strip()
            print(f"warning: Flux --queue takes ONE queue; using {first!r} from {part!r} "
                  f"(Slurm-style comma-lists aren't valid for Flux)", file=__import__("sys").stderr)
            part = first
        out["partition"] = part
        if pwhy != "--partition":
            decisions.append(f"partition: {part} (via {pwhy})")

    t, twhy = resolve_time(getattr(args, "time", None))
    if t:
        out["time"] = t
        if twhy != "--time":
            decisions.append(f"time: {t} (via {twhy})")

    return out, decisions
