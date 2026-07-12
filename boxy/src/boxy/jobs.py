"""Job records for scheduler-submitted serves (the seamless Slurm/Flux path).

The rendezvous protocol rides the shared filesystem — the one thing every HPC
site guarantees:

  login node                          compute node (inside the job)
  ----------                          -----------------------------
  boxy serve M --scheduler slurm
    writes <name>.sh, submits it
    writes <name>.json (job id)  -->  batch script runs
    polls squeue/flux jobs              boxy serve M --foreground
    polls <name>.endpoint.json  <--      resolves accel/image/port ON the node
    polls http://host:port/v1           writes <name>.endpoint.json
    prints ### READY                    serves until the job ends
"""

from __future__ import annotations

import json
import os
import re
import socket
from pathlib import Path

DEFAULT_ROOT = "~/.local/share/boxy/jobs"


def cluster_id(host: str) -> str:
    """Best-effort cluster identity from a hostname: 'clusterA-login2',
    'clusterA-login1.example.com', 'clusterA' -> 'clusterA'; 'clusterB42',
    'clusterB-login5' -> 'clusterB'. Sites with unusual naming set BOXY_CLUSTER."""
    short = host.split(".", 1)[0].lower()
    trimmed = re.sub(r"[-_]?login$", "", re.sub(r"\d+$", "", short)).rstrip("-_")
    # a laptop asset tag like 's1088597' trims to 's' — a meaningless bucket; keep
    # the full short name when trimming leaves too little to be a real cluster.
    return trimmed if len(trimmed) >= 2 else (short or host)


def local_cluster() -> str:
    return os.environ.get("BOXY_CLUSTER") or cluster_id(socket.gethostname())


def _dir() -> Path:
    """Where job state (records/endpoints/scripts/logs) lives. Labs share $HOME
    across clusters, so BY DEFAULT this is partitioned per cluster —
    <root>/<cluster>/ — so `boxy logs/list/curl` on clusterB never surface an
    clusterA job (field report). BOXY_JOBS_DIR pins an EXACT dir (no
    partitioning: the explicit escape hatch, and what tests use); BOXY_JOBS_ROOT
    overrides only the partitioned base."""
    exact = os.environ.get("BOXY_JOBS_DIR")
    if exact:
        path = Path(os.path.expanduser(exact))
        path.mkdir(parents=True, exist_ok=True)
        return path
    from boxy import config

    root = Path(os.path.expanduser(config.get("paths.jobs_root")))
    path = root / local_cluster()
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(name: str) -> Path:
    return _dir() / f"{name}.json"


def endpoint_path(name: str) -> Path:
    return _dir() / f"{name}.endpoint.json"


def script_path(name: str) -> Path:
    return _dir() / f"{name}.sh"


def log_path(name: str, job_id: str = "") -> Path:
    """The job's output log. With a job_id, the file is per-JOB
    (<name>-<job_id>.log) so repeated submissions of the same name never
    overwrite each other's logs; without one, the plain <name>.log."""
    if job_id:
        return _dir() / f"{name}-{job_id}.log"
    return _dir() / f"{name}.log"


def resolve_log(name: str, job_id: str = "") -> Path:
    """Best path to the job's log for tailing: the exact per-job file if it
    exists, else the newest <name>-*.log (the scheduler may render its job-id
    token differently than the id we parsed), else the plain <name>.log."""
    exact = log_path(name, job_id)
    if exact.exists():
        return exact
    candidates = sorted(_dir().glob(f"{name}-*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else log_path(name)


def write_record(name: str, data: dict) -> Path:
    path = record_path(name)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def read_record(name: str) -> dict | None:
    path = record_path(name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except ValueError:
        return None


def write_endpoint_file(path: str | Path, name: str, port: int, job_id: str = "") -> Path:
    """Atomic (tmp + rename): the login-side poller must never see a torn
    write over NFS-ish filesystems (r2 audit)."""
    host = socket.gethostname()
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "name": name,
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "job": job_id,
    }) + "\n")
    os.replace(tmp, path)
    return path


def write_endpoint(name: str, port: int, job_id: str = "") -> Path:
    return write_endpoint_file(endpoint_path(name), name, port, job_id)


def read_endpoint(name: str) -> dict | None:
    path = endpoint_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None  # partially written; caller retries
    # junk-typed JSON (or a partial dict) must never KeyError three commands
    # downstream (r2 audit) — treat as not-yet-published
    if isinstance(data, dict) and all(k in data for k in ("url", "host", "port")):
        return data
    return None


def share_path(name: str) -> Path:
    return _dir() / f"{name}.share.json"


def share_log_path(name: str) -> Path:
    return _dir() / f"{name}.share.log"


def write_share(name: str, data: dict) -> Path:
    """Atomic like write_endpoint_file — the record is what `boxy unshare` and
    `boxy list` trust, so it must never be seen torn. Contains NO credential."""
    path = share_path(name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def read_share(name: str) -> dict | None:
    path = share_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return None
    # same shape-guard philosophy as read_endpoint: junk must not KeyError later
    if isinstance(data, dict) and all(k in data for k in ("alias", "url", "relay_port")):
        return data
    return None


def remove_share(name: str) -> None:
    share_path(name).unlink(missing_ok=True)


def list_shares() -> list[dict]:
    suffix = ".share.json"
    out = []
    for path in sorted(_dir().glob(f"*{suffix}")):
        share = read_share(path.name[: -len(suffix)])
        if share:
            out.append(share)
    return out


def list_endpoints(base: str) -> list[dict]:
    """Every published endpoint for a replica set: the `<base>-r*` endpoint files
    (as written by `boxy serve --replicas K`), returned as read_endpoint dicts.
    Used by the router to discover the replicas to load-balance across."""
    out = []
    suffix = ".endpoint.json"
    # `-r[0-9]*` requires a digit after -r so base "m" does not swallow a different
    # set "m-rock-r0"; replica indices are always numeric (<base>-r0..r{K-1}).
    for path in sorted(_dir().glob(f"{base}-r[0-9]*{suffix}")):
        ep = read_endpoint(path.name[: -len(suffix)])
        if ep:
            out.append(ep)
    return out


def remove(name: str) -> None:
    for path in (record_path(name), endpoint_path(name), script_path(name)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def list_records() -> list[dict]:
    records = []
    for path in sorted(_dir().glob("*.json")):
        if path.name.endswith(".endpoint.json"):
            continue
        try:
            record = json.loads(path.read_text())
        except ValueError:
            continue
        # shape-guard: a stale/hand-edited record must not take out `boxy list`
        if isinstance(record, dict) and all(k in record for k in ("name", "scheduler", "job")):
            records.append(record)
    return records
