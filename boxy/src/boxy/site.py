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

# An account/WC-ID token: letters then >=4 digits (fy260064, FY140001) — the
# preferred shape; a bare 6-8 digit ID is accepted only when no letter-prefixed
# token exists anywhere. Real `mywcid` rows carry BOTH (`... fy140001   103732
# system software ...` — the description starts with a numeric id), and search()
# order picks the account: 'ambelt' has no digits, so fy140001 is the first hit.
_ACCOUNT_RE = re.compile(r"\b([A-Za-z]{1,4}\d{4,})\b")
_BARE_ID_RE = re.compile(r"\b(\d{6,8})\b")

# Header/separator lines from the real mywcid table (field sample, 2026-07):
#       User    Account                              Description     Parent
#   ---------- ---------- ------------------------------------ ----------
# plus WC-ID-style headers and dashed rules: never mine these for tokens.
_HEADER_RE = re.compile(
    r"^\s*[-=+\s]+$"                      # dashed/blank separator rules
    r"|^\s*user\s+account\b"              # the real mywcid header row
    r"|\bdescription\b|\bparent\b|\btitle\b"   # other header vocabulary
    r"|^\s*wc\s*id\s",
    re.IGNORECASE)


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
    Tolerant of the layouts seen in the field: a `mywcid` TABLE (header row +
    data rows), a labelled line (`WCID: fy260064 (Project)`), a bare list, and
    `sacctmgr -nP` single-column output. Header/separator lines are skipped;
    letter-prefixed IDs (fy260064) win; bare 6-8 digit IDs are the fallback
    ONLY when no letter-prefixed token exists anywhere."""
    prefixed: list[str] = []
    bare: list[str] = []
    seen_ci: set[str] = set()   # mywcid's trailing "could be on Caps too: FY140001"
    for line in text.splitlines():
        if _HEADER_RE.search(line):
            continue
        m = _ACCOUNT_RE.search(line)
        if m:
            if m.group(1).lower() not in seen_ci:
                seen_ci.add(m.group(1).lower())
                prefixed.append(m.group(1))
            continue
        b = _BARE_ID_RE.search(line)
        if b and b.group(1) not in bare:
            bare.append(b.group(1))
    return prefixed if prefixed else bare


def _first_output_line(text: str) -> str:
    """The first non-empty line of a probe's output, truncated — shown when
    parsing finds nothing, so the field fix is a glance, not a debug session."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:100]
    return ""


def _account_from_command() -> tuple[str | None, list[str], str]:
    """(first_account, all_accounts, raw_first_line). raw_first_line is non-empty
    only when the command PRODUCED output that parsed to nothing — the case worth
    showing the user verbatim."""
    cmd = config.get_str("site.account_command").strip()
    if not cmd:
        return None, [], ""
    out = _run(cmd.split())
    accounts = parse_accounts(out)
    raw = "" if accounts else _first_output_line(out)
    return (accounts[0] if accounts else None), accounts, raw


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
    acct, alts, raw = _account_from_command()
    if acct:
        cmd = config.get_str("site.account_command").strip()
        extra = f"; also: {', '.join(alts[1:])}" if len(alts) > 1 else ""
        return acct, f"{cmd}{extra}"
    for env in ("SBATCH_ACCOUNT", "SLURM_ACCOUNT"):
        v = os.environ.get(env)
        if v:
            return v, f"${env}"
    sacct, salts = _account_from_sacctmgr()
    if sacct:
        extra = f"; also: {', '.join(salts[1:])}" if len(salts) > 1 else ""
        return sacct, f"sacctmgr assoc{extra}"
    why = ("no account discovered (mywcid / $SBATCH_ACCOUNT / sacctmgr) — "
           "the scheduler will use its site default; pass --account if it rejects the job")
    if raw:
        cmd = config.get_str("site.account_command").strip()
        why = (f"`{cmd}` answered but no account parsed from: {raw!r} — "
               f"pass --account (or export BOXY_ACCOUNT) and report the format")
    return None, why


def remote_account_probe() -> str:
    """The shell one-liner run ON a cluster login node (over the ssh master) to
    discover the account when delegating with --ssh: the configured site command
    (default `mywcid`), falling back to sacctmgr. Shared by the --ssh serve
    injection and `boxy doctor --ssh`."""
    cmd = config.get_str("site.account_command").strip() or "mywcid"
    return (f"{cmd} 2>/dev/null || "
            f"sacctmgr -nP show assoc user=$USER format=account 2>/dev/null || true")


def discover_partitions(scheduler_name: str) -> list[tuple[str, int, bool]]:
    """(name, idle_nodes, is_up) per partition on THIS host — best-effort, []
    if the scheduler can't enumerate them or the tool is missing/errors."""
    from boxy.schedulers import get_scheduler

    try:
        sched = get_scheduler(scheduler_name)
    except (ValueError, KeyError):
        return []
    cmd = sched.partitions_command()
    if not cmd:
        return []
    try:
        return sched.parse_partitions(_run(cmd))
    except Exception:  # noqa: BLE001 — discovery must never break a submission
        return []


def rank_partitions(parts: list[tuple[str, int, bool]], scheduler_name: str) -> tuple[str, str]:
    """Turn discovered partitions into the `--partition auto` value + provenance.
    Slurm gets ALL up partitions as a comma-list (idle-first) so its own
    scheduler starts the job in whichever frees soonest — the native
    soonest-start behavior. Flux's --queue takes ONE, so pick the single best
    (idle-first, then name). ('' , reason) when nothing usable was found."""
    up = [(n, idle) for (n, idle, is_up) in parts if is_up]
    if not up:
        tool = "sinfo" if scheduler_name == "slurm" else "flux queue list"
        return "", f"auto requested but no partitions discovered ({tool}) — using the scheduler's site default"
    up.sort(key=lambda t: (-t[1], t[0]))  # most idle first, then name (deterministic)
    names = [n for n, _ in up]
    if scheduler_name == "flux":
        return names[0], f"auto → {names[0]} (soonest-start queue of {len(names)}; flux queue list)"
    top_idle = up[0][1]
    note = (f"most idle: {names[0]} ({top_idle} nodes)" if top_idle
            else "none idle now — Slurm queues it to whichever frees first")
    return ",".join(names), f"auto → {len(names)} partitions, soonest-start ({note})"


def rank_remote_partitions(stdout: str, scheduler_name: str) -> tuple[str, str]:
    """rank_partitions for output captured on a REMOTE login node (--ssh): parse
    with the scheduler's own parser, then rank. Used to resolve `--partition
    auto` to a concrete list before delegating (an older cluster boxy would pass
    the literal word 'auto' to sbatch and get 'invalid partition')."""
    from boxy.schedulers import get_scheduler

    try:
        parts = get_scheduler(scheduler_name).parse_partitions(stdout)
    except Exception:  # noqa: BLE001
        parts = []
    return rank_partitions(parts, scheduler_name)


def remote_partition_probe(scheduler_name: str) -> str:
    """The shell one-liner run on a cluster login node (over the ssh master) to
    list partitions for `--partition auto`. `true` when the scheduler can't
    enumerate (auto then degrades to the site default)."""
    import shlex

    from boxy.schedulers import get_scheduler

    try:
        cmd = get_scheduler(scheduler_name).partitions_command()
    except (ValueError, KeyError):
        cmd = []
    if not cmd:
        return "true"
    return shlex.join(cmd) + " 2>/dev/null || true"


def _wants_auto_partition(explicit: str | None) -> bool:
    """True when the user asked boxy to PICK the partition: `--partition auto`,
    or config `site.partition = auto` with no flag."""
    if explicit is not None:
        return explicit.strip().lower() == "auto"
    return config.get_str("site.partition").strip().lower() == "auto"


def resolve_partition(explicit: str | None, scheduler_name: str = "slurm") -> tuple[str | None, str]:
    """(value, provenance). `--partition auto` (or config `site.partition=auto`)
    discovers partitions and picks the soonest-start set; an explicit
    partition/queue always wins; otherwise the config default or None."""
    if explicit and explicit.strip().lower() != "auto":
        return explicit, "--partition"
    if _wants_auto_partition(explicit):
        value, why = rank_partitions(discover_partitions(scheduler_name), scheduler_name)
        return (value or None), why
    if not explicit:
        cfg = config.get_str("site.partition").strip()
        if cfg:  # a non-'auto' config default (auto handled above)
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

    part, pwhy = resolve_partition(getattr(args, "partition", None), scheduler_name)
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
