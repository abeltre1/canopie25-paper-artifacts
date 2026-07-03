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
import socket
from pathlib import Path

JOBS_DIR = os.path.expanduser(os.environ.get("BOXY_JOBS_DIR", "~/.local/share/boxy/jobs"))


def _dir() -> Path:
    path = Path(os.path.expanduser(os.environ.get("BOXY_JOBS_DIR", JOBS_DIR)))
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_path(name: str) -> Path:
    return _dir() / f"{name}.json"


def endpoint_path(name: str) -> Path:
    return _dir() / f"{name}.endpoint.json"


def script_path(name: str) -> Path:
    return _dir() / f"{name}.sh"


def log_path(name: str) -> Path:
    return _dir() / f"{name}.log"


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
