"""Site auto-resolution — the turnkey account/partition/time discovery.

A novice runs `boxy serve MODEL --scheduler slurm` and the job must actually be
accepted by the scheduler, which usually means a charge account. boxy fills the
site knobs the user didn't pass, each printing an `auto:` decision line, and an
explicit flag always wins.

Account probe chain (first hit wins):
  1. --account flag                          (handled by the caller; passed in)
  2. $WCID                                    (a session bypass for the picker)
  3. config site.account                     (BOXY_ACCOUNT / [site].account)
  4. site.account_command                    (default `mywcid`, Sandia's WC-ID lister)
  5. $SBATCH_ACCOUNT / $SLURM_ACCOUNT
  6. `sacctmgr show assoc user=$USER ...`     (single assoc auto-picks; many -> first + note)
  7. none -> omit (the scheduler uses its own site default)

When several accounts are discovered and none was named, the caller may show an
interactive menu (see picker.py) instead of silently taking the first.

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


def _row_label(line: str, m: re.Match) -> str:
    """A short human label for one account row — the descriptive text `mywcid`
    prints (project/title), shown beside the id in the picker menu. Takes the
    text after the matched id, dropping a leading numeric description-id
    (`fy140001  103732 system software` -> `system software`); falls back to
    text before the id (labelled `WCID: fy... (Project)` layouts)."""
    tail = re.sub(r"\s+", " ", line[m.end():]).strip()
    tail = re.sub(r"^\d{4,}\s+", "", tail).strip()
    if not tail:
        tail = re.sub(r"\s+", " ", line[:m.start()]).strip(" :\t")
    return tail[:60]


def parse_account_rows(text: str) -> list[tuple[str, str]]:
    """Like parse_accounts but keeps each account's row LABEL for the interactive
    picker: [(wcid, label), ...], in order, de-duplicated. Same header-skip and
    prefer-letter-prefixed-ids rules; bare numeric ids are the fallback ONLY when
    no letter-prefixed token exists anywhere."""
    prefixed: list[tuple[str, str]] = []
    bare: list[tuple[str, str]] = []
    seen_ci: set[str] = set()
    for line in text.splitlines():
        if _HEADER_RE.search(line):
            continue
        m = _ACCOUNT_RE.search(line)
        if m:
            if m.group(1).lower() not in seen_ci:
                seen_ci.add(m.group(1).lower())
                prefixed.append((m.group(1), _row_label(line, m)))
            continue
        b = _BARE_ID_RE.search(line)
        if b and all(b.group(1) != w for w, _ in bare):
            bare.append((b.group(1), _row_label(line, b)))
    return prefixed if prefixed else bare


def discover_account_rows() -> list[tuple[str, str]]:
    """(wcid, label) rows from the configured site account command (default
    `mywcid`), for the interactive picker. [] if the command is unset, missing,
    or produced nothing account-looking."""
    cmd = config.get_str("site.account_command").strip()
    if not cmd:
        return []
    return parse_account_rows(_run(cmd.split()))


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
    wcid = os.environ.get("WCID", "").strip()
    if wcid:
        return wcid, "$WCID"
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


def rank_partitions(parts, scheduler_name: str, prefer_gpu: bool = False) -> tuple[str, str]:
    """Turn discovered partitions (PartitionInfo rows) into the auto value +
    provenance. Slurm gets ALL eligible up partitions as a comma-list
    (idle-first) so its own scheduler starts the job in whichever frees soonest
    — native soonest-start. Flux's --queue takes ONE, so pick the single best.
    When `prefer_gpu`, restrict to partitions that advertise a GPU (so a GPU job
    is never parked in a CPU-only partition) — but if NONE are identifiable as
    GPU (e.g. Flux, or a site that doesn't publish GRES), fall back to all up
    partitions rather than emit nothing. ('' , reason) when nothing usable."""
    up = [p for p in parts if p.up]
    if not up:
        tool = "sinfo" if scheduler_name == "slurm" else "flux queue list"
        return "", f"no partitions discovered ({tool}) — using the scheduler's site default"
    pool, gpu_note = up, ""
    if prefer_gpu:
        gpu = [p for p in up if p.has_gpu]
        if gpu:
            pool, gpu_note = gpu, " with GPUs"
        else:
            gpu_note = " (no GPU partitions identified — offering all)"
    pool = sorted(pool, key=lambda p: (-p.idle_nodes, p.name))  # most idle first, then name
    names = [p.name for p in pool]
    if scheduler_name == "flux":
        return names[0], f"{names[0]} (soonest-start queue of {len(names)}{gpu_note}; flux queue list)"
    top_idle = pool[0].idle_nodes
    note = (f"most idle: {names[0]} ({top_idle} nodes)" if top_idle
            else "none idle now — Slurm queues it to whichever frees first")
    return ",".join(names), f"{len(names)} partitions{gpu_note}, soonest-start ({note})"


def rank_remote_partitions(stdout: str, scheduler_name: str, prefer_gpu: bool = False) -> tuple[str, str]:
    """rank_partitions for output captured on a REMOTE login node (--ssh): parse
    with the scheduler's own parser, then rank. Used to resolve auto to a
    concrete list before delegating (an older cluster boxy would pass the literal
    word 'auto'/'all' to sbatch and get 'invalid partition')."""
    from boxy.schedulers import get_scheduler

    try:
        parts = get_scheduler(scheduler_name).parse_partitions(stdout)
    except Exception:  # noqa: BLE001
        parts = []
    return rank_partitions(parts, scheduler_name, prefer_gpu)


def remote_partition_probe(scheduler_name: str) -> str:
    """The shell one-liner run on a cluster login node (over the ssh master) to
    list partitions for auto selection. `true` when the scheduler can't
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


def remote_accel_probe() -> str:
    """Shell snippet run on a cluster LOGIN node (over the ssh master) to
    auto-detect the accelerator family for an agentless serve. Prints exactly
    one token: cuda | rocm | none.

    Signal order (field: an AMD system got the CUDA image + 'unresolvable CDI
    devices nvidia.com/gpu=all'):
    1. the SCHEDULER's GPU inventory (`sinfo -o %G` GRES types) — authoritative,
       it describes the COMPUTE nodes: mi250/mi300/amd => rocm; h100/a100/... =>
       cuda. Works even when the login node carries no GPU userland at all.
    2. login-node userland markers — rocm-smi//opt/rocm//dev/kfd before the
       NVIDIA markers, since ROCm userland on an NVIDIA cluster is far rarer
       than a stray CUDA module on an AMD one."""
    return (
        'g=$( (sinfo -h -o %G 2>/dev/null || true) | tr "[:upper:]" "[:lower:]" | tr "\\n" " " ); '
        'case "$g" in '
        '*mi[0-9]*|*amd*|*gfx[0-9]*) echo rocm ;; '
        '*h100*|*h200*|*a100*|*v100*|*gh200*|*b200*|*l4*|*t4*|*p100*|*k80*|*rtx*|*tesla*|*nvidia*) '
        'echo cuda ;; '
        '*) if command -v rocm-smi >/dev/null 2>&1 || [ -d /opt/rocm ] || [ -e /dev/kfd ]; '
        'then echo rocm; '
        'elif command -v nvidia-smi >/dev/null 2>&1 || [ -e /proc/driver/nvidia/version ]; '
        'then echo cuda; '
        'else echo none; fi ;; '
        'esac')


def parse_remote_accel(out: str) -> str:
    """The probe's token, or '' when nothing usable came back (banner noise from
    a login shell is tolerated — only the LAST line is read)."""
    lines = [ln.strip() for ln in (out or "").strip().splitlines() if ln.strip()]
    tok = lines[-1] if lines else ""
    return tok if tok in ("cuda", "rocm") else ""


def remote_scheduler_probe() -> str:
    """Shell one-liner run on a cluster login node (over the ssh master) to
    auto-detect `--scheduler`. It reports which scheduler is ACTUALLY OPERATIONAL,
    not merely installed — a Flux system commonly ships Slurm-compat `sbatch`/
    `sinfo`/`scontrol` wrappers that PROXY to Flux (field report: `eldorado`,
    whose `sbatch` shim returns FLUX job ids — `f2c5JAAU8BR1` — squeue can't
    track), and a Slurm site may run a personal NESTED flux instance. Emitted
    tokens:
      * flux-bin / slurm-bin — the binary is on PATH.
      * flux-live   — the SYSTEM Flux instance is reachable (`instance-level` 0).
                      Probed via the well-known system socket FIRST
                      (`local:///run/flux/local`) so a non-interactive ssh that
                      lacks FLUX_URI (no profile sourced) still finds it, then via
                      the ambient env. System Flux == Flux runs the machine, so it
                      is authoritative over slurm compat shims.
      * flux-nested — only a NON-system flux instance is reachable (instance-level
                      >= 1, e.g. a personal `flux alloc` under Slurm). NOT
                      authoritative: a real Slurm controller outranks it.
      * slurm-ctld  — a REAL slurmctld answers `scontrol ping` ("... is UP").
      * slurm-live  — `sinfo` returns a partition (weaker: a Flux compat layer can
                      answer this too, so flux-live outranks it).
    pick_scheduler() ranks a live SYSTEM Flux broker first. Robust across a mixed
    fleet with no per-cluster config."""
    return (
        'if command -v flux >/dev/null 2>&1; then echo flux-bin; '
        'for U in "local:///run/flux/local" ""; do '
        'if [ -n "$U" ]; then FX="flux --uri $U"; else FX="flux"; fi; '
        'L=$($FX getattr instance-level 2>/dev/null); '
        'if [ -n "$L" ]; then { [ "$L" = 0 ] && echo flux-live || echo flux-nested; }; break; fi; '
        'if $FX resource list >/dev/null 2>&1 || $FX uptime >/dev/null 2>&1; then echo flux-live; break; fi; '
        'done; fi; '
        'if command -v sbatch >/dev/null 2>&1; then echo slurm-bin; '
        "scontrol ping 2>/dev/null | grep -qi 'is up' && echo slurm-ctld; "
        'sinfo -h -o %R 2>/dev/null | grep -q . && echo slurm-live; fi; '
        'true'
    )


def pick_scheduler(available: str, explicit: str | None = None) -> tuple[str | None, str]:
    """(scheduler, why) from an explicit flag, config site.scheduler, else the
    OPERATIONAL evidence in `available` (whitespace tokens from
    remote_scheduler_probe).

    Ranking (explicit flag > config > evidence):
      * a live SYSTEM Flux broker (flux-live) -> FLUX. System Flux runs the machine;
                                             any slurm commands that also answered are
                                             compat shims that proxy to Flux (submitting
                                             via them yields Flux job ids slurm can't
                                             track — the eldorado failure). Override
                                             with --scheduler slurm / BOXY_SCHEDULER=slurm.
      * a real slurmctld / sinfo (slurm-live) -> SLURM. Outranks a merely NESTED flux
                                             instance (a personal flux under Slurm).
      * only a nested flux instance           -> FLUX (it's the reachable scheduler).
      * no liveness, one binary               -> that one.
      * no liveness, both binaries            -> slurm default, loud override note.
      * nothing                               -> None (a direct/local serve).
    A bare `flux`/`slurm` token (no `-bin`/`-live` suffix) counts as binary-only,
    so older callers/tests that pass plain names still work."""
    if explicit in ("slurm", "flux"):
        return explicit, "--scheduler"
    cfg = config.get_str("site.scheduler").strip().lower()
    if cfg in ("slurm", "flux"):
        return cfg, "config site.scheduler"
    if cfg == "none":
        return None, "config site.scheduler=none"

    toks = set((available or "").split())
    flux_live = "flux-live" in toks            # SYSTEM flux == authoritative
    flux_nested = "flux-nested" in toks        # personal/nested flux only
    slurm_live = "slurm-ctld" in toks or "slurm-live" in toks
    bins = [s for s in ("flux", "slurm") if s in toks or f"{s}-bin" in toks]

    if flux_live:
        # a live SYSTEM Flux broker is authoritative — even if slurm commands answered.
        if slurm_live:
            return "flux", ("detected (Flux broker is live — Flux runs this machine; slurm "
                            "commands also answered but on a Flux system those are compat shims "
                            "that proxy to Flux. Pass --scheduler slurm / set BOXY_SCHEDULER=slurm "
                            "if this cluster's primary really is Slurm)")
        return "flux", "detected (Flux broker is live)"
    if slurm_live:
        how = "slurmctld responded to scontrol ping" if "slurm-ctld" in toks else "sinfo listed partitions"
        extra = " (a personal nested Flux instance was also seen, but Slurm runs this machine)" if flux_nested else ""
        return "slurm", f"detected (Slurm is live — {how}{extra})"
    if flux_nested:
        return "flux", "detected (a Flux instance is reachable)"
    if len(bins) == 1:
        return bins[0], "detected"
    if len(bins) == 2:
        return "slurm", ("detected (both flux+slurm binaries present but neither control plane "
                         "responded — defaulting to slurm; set BOXY_SCHEDULER=flux or pass "
                         "--scheduler to override)")
    return None, "no scheduler detected"


def remote_jobname_live_probe(scheduler_name: str, name: str) -> str:
    """Shell one-liner run on a cluster login node (over the ssh master) that
    prints LIVE iff a scheduler job with job-name `name` is currently queued OR
    running for this user. Used to decide auto-unique laptop-side over --ssh: a
    second serve while one is live gets --unique injected, so the user is NEVER
    forced to type --unique even against the cluster's (possibly older,
    pre-auto-unique) boxy singleton. `grep -Fxq --` / `-n` stop a job name
    beginning with '-' from being read as an option."""
    import shlex

    q = shlex.quote(name)
    if scheduler_name == "flux":
        return (f'flux jobs -no "{{name}}" 2>/dev/null | grep -Fxq -- {q} '
                f'&& echo LIVE || true')
    # slurm: -n filters by job name server-side; any active (pending/running) row.
    return (f'squeue -h -u "$USER" -n {q} -o %i 2>/dev/null '
            f'| grep -q . && echo LIVE || true')


# A `gpu` GRES token in sinfo's %G column: `gpu:a100:8`, `gpu:8`, with an optional
# `(S:0-1)` socket suffix. Group 1 = the type (a100) when present.
_GPU_GRES_RE = re.compile(r"\bgpu:(?:([A-Za-z0-9_.+-]+):)?\d+", re.IGNORECASE)


def gpu_types_from_gres(sinfo_text: str, partitions: set[str] | None = None) -> list[str]:
    """ALL candidate GPU TYPES from Slurm's reported GRES, ordered: types on the
    target partitions first, then the rest of the cluster (dedup, first-seen).
    The GRES self-heal tries a typed --gres=gpu:<type>:N for EACH of these — a
    site that REQUIRES a type (field: kahuna) is unrecoverable with the untyped
    forms alone, and restricting to one 'spanning' type found nothing when the
    cluster mixes types across partitions."""
    def scan(pset: set[str] | None) -> list[str]:
        found: list[str] = []
        for line in (sinfo_text or "").splitlines():
            cols = line.split("|")
            if pset is not None and cols and cols[0].strip() not in pset:
                continue
            gres = cols[3] if len(cols) > 3 else line
            for m in _GPU_GRES_RE.finditer(gres):
                t = m.group(1)
                if t and t.lower() != "null" and t not in found:
                    found.append(t)
        return found

    selected = scan(partitions) if partitions else []
    return selected + [t for t in scan(None) if t not in selected]


def gpu_request_from_gres(sinfo_text: str, partitions: set[str] | None = None) -> tuple[str, str]:
    """Auto-detect the site's GPU request convention from Slurm's reported GRES
    (`sinfo -h -o "%R|%a|%F|%G"`). A cluster that lists a `gpu` GRES wants
    `--gres=gpu:[type:]N` — the portable form — because `--gpus-per-node` is
    rejected on some sites ('Invalid generic resource (gres) specification', field
    report: kahuna). Returns:
      ('gres', '<type>') — a single gpu TYPE spans the target partitions (safest:
                           some sites REQUIRE the type),
      ('gres', '')       — gpu GRES present but untyped or types differ (let Slurm
                           pick the type on the assigned node),
      ('', '')           — no gpu GRES reported: keep boxy's default (--gpus-per-node).
    `partitions` restricts the scan to the ones being submitted to (else all)."""
    types: set[str] = set()
    saw_gpu = False
    for line in (sinfo_text or "").splitlines():
        cols = line.split("|")
        if partitions is not None and cols and cols[0].strip() not in partitions:
            continue
        gres = cols[3] if len(cols) > 3 else line   # %R|%a|%F|%G, else scan the whole line
        for m in _GPU_GRES_RE.finditer(gres):
            saw_gpu = True
            if m.group(1) and m.group(1).lower() != "null":
                types.add(m.group(1))
    if not saw_gpu:
        return "", ""
    return ("gres", next(iter(types))) if len(types) == 1 else ("gres", "")


# `--partition off|none` (or the same in config) opts OUT of auto and uses the
# scheduler's own default partition. Kept to two rarely-real partition names so a
# site partition literally named `default`/`site` isn't shadowed (adversarial-
# review finding); a partition genuinely named `off`/`none` is vanishingly rare.
_PARTITION_OFF = {"off", "none"}


def partition_mode(explicit: str | None) -> str:
    """How to choose the partition, from the flag then config:
      'set'  — a concrete partition/comma-list was given (use it verbatim)
      'all'  — every up partition
      'off'  — the scheduler's own default (no partition directive)
      'auto' — boxy picks the soonest-start (GPU-aware) set — THE DEFAULT when
               nothing is specified, so the user never has to pass --partition.
    """
    val = (explicit or "").strip().lower()
    if not val:  # no flag -> consult config, else default to auto
        val = config.get_str("site.partition").strip().lower()
    if val in ("", "auto"):
        return "auto"
    if val == "all":
        return "all"
    if val in _PARTITION_OFF:
        return "off"
    return "set"


def resolve_partition(explicit: str | None, scheduler_name: str = "slurm",
                      need_gpu: bool = False) -> tuple[str | None, str]:
    """(value, provenance). Auto is the DEFAULT (nothing set) — boxy discovers
    partitions and offers the soonest-start set, restricted to GPU partitions
    when the job needs a GPU. `--partition <name>` wins; `all` offers every up
    partition; `off` uses the scheduler's site default. Discovery failure
    degrades quietly to the site default (None)."""
    mode = partition_mode(explicit)
    if mode == "set":
        if explicit and explicit.strip():
            return explicit.strip(), "--partition"
        return config.get_str("site.partition").strip(), "config site.partition"
    if mode == "off":
        return None, ""
    prefer_gpu = need_gpu and mode == "auto"   # `all` never filters by GPU
    value, why = rank_partitions(discover_partitions(scheduler_name), scheduler_name, prefer_gpu)
    return (value or None), why


def resolve_time(explicit: str | None) -> tuple[str | None, str]:
    if explicit:
        return explicit, "--time"
    cfg = config.get_str("site.default_time").strip()
    if cfg:
        return cfg, "config site.default_time"
    return None, ""


def resolve_site(args, scheduler_name: str, need_gpu: bool = False) -> tuple[dict, list[str]]:
    """Fill account/partition/time for a submission. Returns ({kind: value},
    decision_lines). Only non-empty values are returned. Partition defaults to
    auto (boxy picks the soonest-start, GPU-aware set — `need_gpu` restricts to
    GPU partitions). Applies the Flux single-queue guard: Slurm accepts a
    comma-list of partitions, Flux's --queue takes exactly ONE, so a comma'd
    partition is trimmed to the first with a warning."""
    out: dict = {}
    decisions: list[str] = []

    acct, why = resolve_account(getattr(args, "account", None))
    if acct:
        out["account"] = acct
        if why != "--account":
            decisions.append(f"account: {acct} (via {why})")
    else:
        decisions.append(f"account: {why}")

    part, pwhy = resolve_partition(getattr(args, "partition", None), scheduler_name, need_gpu)
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

    # Slurm always honors the license (explicit --license or the config default,
    # e.g. tscratch:1). Other schedulers (Flux) have no license concept, so only an
    # EXPLICIT --license is passed through verbatim — never the Slurm-oriented config
    # default, which would otherwise attach a bogus directive to every Flux job.
    lic, lwhy = resolve_license(getattr(args, "license", None))
    if lic and (scheduler_name == "slurm" or lwhy == "--license"):
        out["license"] = lic
        if lwhy != "--license":
            decisions.append(f"license: {lic} (via {lwhy})")

    return out, decisions


def resolve_license(explicit: str | None) -> tuple[str, str]:
    """(value, provenance) for a Slurm `--license=` request. --license wins, then
    config site.license (BOXY_LICENSE). Empty => none (many sites auto-add
    filesystem licenses; hops prints 'Adding filesystem licenses to job: …')."""
    if explicit:
        return explicit, "--license"
    cfg = config.get_str("site.license").strip()
    return (cfg, "config site.license") if cfg else ("", "")


# ---- model store: where multi-GB HF downloads land on the CLUSTER ----------------
#
# $HOME on lab clusters carries a small quota; a single 70B model (~140 GB) blows
# through it (field: user ran out of space mid-serve). The model cache belongs on
# the big SHARED scratch FS. The ladder is deterministic — same pick every run on
# a given cluster, so the cache is downloaded once and always reused — and every
# candidate must be shared (visible from the compute nodes), so node-local
# /lscratch|/tmp are deliberately absent.

MODEL_STORE_LADDER = (
    '"$SCRATCH"',                    # many sites export the canonical per-user scratch
    '"/tscratch/users/$USER"',       # Sandia big scratch (hops class)
    '"/tscratch/$USER"',
    '"/pscratch/users/$USER"',
    '"/pscratch/$USER"',
    '"/scratch/users/$USER"',
    '"/scratch/$USER"',
    '"/gscratch/$USER"',
)


def model_store_probe(saved: str = "") -> str:
    """One login-node shell probe: for each candidate (a previously PICKED path
    first, so the choice stays sticky across reruns), try to create it, keep it
    when it's a writable directory, and report its free space in KB as
    '<path> <free_kb>' lines. Runs with plain POSIX tools — no boxy on the cluster."""
    cands = ([f'"{saved}"'] if saved else []) + list(MODEL_STORE_LADDER)
    return (
        'for d in ' + " ".join(cands) + '; do '
        '[ -n "$d" ] || continue; '
        'mkdir -p "$d" 2>/dev/null; '
        '[ -d "$d" ] && [ -w "$d" ] || continue; '
        'f=$(df -Pk "$d" 2>/dev/null | awk \'NR==2{print $4}\'); '
        'echo "$d ${f:-0}"; '
        'done'
    )


def pick_model_store(probe_out: str, min_free_gb: int) -> tuple[str, int, str]:
    """(path, free_gb, why) from the probe's '<path> <free_kb>' lines — the FIRST
    candidate with at least min_free_gb free, else the roomiest one (with a why
    that says it's tight), else ('', 0, why-nothing-found). Ladder order (not max
    free) keeps the pick stable run-over-run so the cache is reused."""
    rows: list[tuple[str, int]] = []
    for line in probe_out.splitlines():
        parts = line.strip().rsplit(None, 1)
        if len(parts) != 2 or not parts[0].startswith("/"):
            continue
        try:
            rows.append((parts[0], int(parts[1]) // (1024 * 1024)))
        except ValueError:
            continue
    if not rows:
        return "", 0, "no shared scratch FS found on the login node"
    for path, free_gb in rows:
        if free_gb >= min_free_gb:
            return path, free_gb, f"first scratch FS with >= {min_free_gb} GB free"
    path, free_gb = max(rows, key=lambda r: r[1])
    return path, free_gb, (f"roomiest scratch FS found, but only {free_gb} GB free "
                           f"(< storage.min_free_gb={min_free_gb})")


# ---- system-card generation: the cluster's own inventory becomes a card ----------
#
# `boxy generate system --ssh HOST` asks the SCHEDULER for the node inventory
# (sinfo knows every node's CPUs/memory/GRES — authoritative for the compute
# nodes, unlike anything visible on a login node) plus the existing accel/
# runtime/scratch probes, and writes a per-cluster system card. The geometry
# solver (cards.fit_geometry) then sizes every serve against real hardware.

# GRES/accelerator type -> per-GPU memory (GB). Slurm never exposes VRAM, only
# the type token, so this table maps the common parts; unknown types leave
# gpu_vram_gb=0 (the solver states its 80GB-class assumption) with a card
# comment telling the operator which one number to fill in.
_GPU_VRAM_GB = (
    ("h200", 141), ("gh200", 96), ("h100", 80), ("b200", 192), ("gb200", 192),
    ("a100", 80),                     # 40GB variant handled by the "40" check below
    ("v100", 32), ("p100", 16), ("t4", 16), ("l40", 48), ("l4", 24),
    ("a40", 48), ("a30", 24), ("a6000", 48), ("rtx6000", 48), ("a10", 24),
    ("mi300a", 128), ("mi300x", 192), ("mi300", 128), ("mi250", 128),
    ("mi210", 64), ("mi100", 32), ("k80", 12),
)


def gpu_vram_from_type(gpu_type: str) -> tuple[int, str]:
    """(vram_gb, note) for a GRES type token like 'h100', 'a100_40gb',
    'nvidia_h200'. 0 = unknown type (the card says how to fill it in)."""
    t = (gpu_type or "").lower()
    for token, gb in _GPU_VRAM_GB:
        if token in t:
            if token == "a100" and "40" in t.replace("a100", ""):
                return 40, "a100 40GB variant"
            note = f"from GRES type '{gpu_type}'"
            if token == "a100":
                note += " (assuming the 80GB variant — set 40 if these are 40GB parts)"
            if token == "gh200":
                note += " (96GB HBM3; 144 for the HBM3e variant)"
            return gb, note
    return 0, f"unknown GPU type '{gpu_type}' — fill in gpu_vram_gb from your site docs"


def sinfo_inventory_probe() -> str:
    """Node-wise inventory off the scheduler: name|cpus|memoryMB|gres, one line
    per node (deduped laptop-side — a node appears once per partition)."""
    return 'sinfo -h -N -o "%N|%c|%m|%G" 2>/dev/null'


def flux_inventory_probe() -> str:
    """Coarse Flux inventory: up nodes/cores/gpus totals."""
    return 'flux resource list -s up -no "{nnodes} {ncores} {ngpus}" 2>/dev/null'


def runtime_probe() -> str:
    return ('for r in podman apptainer docker ch-run; do '
            'command -v $r >/dev/null 2>&1 && echo $r; done')


def parse_sinfo_inventory(text: str) -> dict:
    """Aggregate the node-wise sinfo dump into the card's supply facts:
    total nodes, and the MODAL GPU-node shape (gpus/type/cpus/mem) — clusters
    are heterogeneous, so every distinct shape is also returned for the card's
    comment block. CPU-only clusters come back with gpus=0 shapes only."""
    nodes: dict[str, tuple[int, int, str, int]] = {}   # name -> (cpus, mem_gb, gpu_type, gpus)
    for line in (text or "").splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4 or not parts[0]:
            continue
        name, cpus_s, mem_s, gres = parts
        if name in nodes:
            continue
        try:
            cpus = int(re.sub(r"[^0-9]", "", cpus_s) or 0)
            mem_gb = int(int(re.sub(r"[^0-9]", "", mem_s) or 0) / 1024)
        except ValueError:
            continue
        gpus, gpu_type = 0, ""
        for g in gres.split(","):
            m = re.match(r"gpu:(?:([A-Za-z0-9_.-]+):)?(\d+)", g.strip())
            if m:
                gpu_type = m.group(1) or gpu_type
                gpus += int(m.group(2))
        nodes[name] = (cpus, mem_gb, gpu_type, gpus)
    if not nodes:
        return {}
    shapes: dict[tuple, int] = {}
    for cpus, mem_gb, gpu_type, gpus in nodes.values():
        key = (gpus, gpu_type, cpus, mem_gb)
        shapes[key] = shapes.get(key, 0) + 1
    gpu_shapes = {k: v for k, v in shapes.items() if k[0] > 0}
    modal_pool = gpu_shapes or shapes
    modal = max(modal_pool.items(), key=lambda kv: kv[1])[0]
    return {
        "total_nodes": len(nodes),
        "total_gpu_nodes": sum(gpu_shapes.values()),
        "gpus_per_node": modal[0], "gpu_type": modal[1],
        "cpus_per_node": modal[2], "mem_gb_per_node": modal[3],
        "modal_count": modal_pool[modal],
        "shapes": sorted(((v, k) for k, v in shapes.items()), reverse=True),
    }


def parse_flux_inventory(text: str) -> dict:
    """The Flux totals -> the same shape dict (coarse: per-node = totals/nodes)."""
    for line in (text or "").splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if m:
            n, cores, gpus = (int(x) for x in m.groups())
            if n <= 0:
                continue
            g = gpus // n
            return {"total_nodes": n, "total_gpu_nodes": n if g else 0,
                    "gpus_per_node": g, "gpu_type": "",
                    "cpus_per_node": cores // n, "mem_gb_per_node": 0,
                    "modal_count": n, "shapes": []}
    return {}


def render_system_card(cluster: str, scheduler: str, accelerator: str, runtime: str,
                       inv: dict, storage_lines: list[str]) -> str:
    """The generated per-cluster system card: everything the solvers consume
    ([location]/[location.resources]) plus the operator-facing facts (storage,
    other node shapes) as comments. Deterministic for a given probe output."""
    vram, vram_note = (gpu_vram_from_type(inv.get("gpu_type", ""))
                       if inv.get("gpus_per_node") else (0, ""))
    lines = [
        f"# System card for {cluster} — generated by `boxy generate system` from the",
        "# cluster's own scheduler inventory. Regenerate after a hardware change:",
        f"#   boxy generate system --ssh {cluster} --force",
        "[location]",
        f'name = "{cluster}"',
        f'scheduler = "{scheduler}"',
    ]
    if accelerator:
        lines.append(f'accelerator = "{accelerator}"')
    if runtime:
        lines.append(f'runtime = "{runtime}"')
    lines += ["", "[location.resources]"]
    if inv.get("gpus_per_node"):
        shape = f"{inv['gpus_per_node']}x {inv.get('gpu_type') or 'gpu'}"
        lines.append(f"gpus_per_node = {inv['gpus_per_node']}    "
                     f"# modal GPU-node shape: {shape} ({inv.get('modal_count', '?')} nodes)")
        if vram:
            lines.append(f"gpu_vram_gb = {vram}    # {vram_note}")
        else:
            lines.append(f"# gpu_vram_gb = 80    # {vram_note}")
    if inv.get("cpus_per_node"):
        lines.append(f"cpus_per_node = {inv['cpus_per_node']}")
    if inv.get("mem_gb_per_node"):
        lines.append(f"mem_gb_per_node = {inv['mem_gb_per_node']}")
    if inv.get("total_nodes"):
        lines.append(f"total_nodes = {inv['total_nodes']}    # inventory — NOT a job request")
    if inv.get("total_gpu_nodes"):
        lines.append(f"total_gpu_nodes = {inv['total_gpu_nodes']}")
    others = [s for s in inv.get("shapes", [])[:6]
              if s[1] != (inv.get("gpus_per_node"), inv.get("gpu_type"),
                          inv.get("cpus_per_node"), inv.get("mem_gb_per_node"))]
    if others:
        lines.append("# other node shapes (count x [gpus, type, cpus, mem_gb]):")
        for count, key in others:
            lines.append(f"#   {count} x {list(key)}")
    if storage_lines:
        lines.append("")
        lines.append("# storage (login-node probe at generation time; serves pick the first")
        lines.append("# scratch FS with room automatically — BOXY_MODEL_DIR pins it):")
        for s in storage_lines[:8]:
            lines.append(f"#   {s}")
    return "\n".join(lines) + "\n"
