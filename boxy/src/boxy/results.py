"""Persistent bench-result store: every `boxy bench` / `boxy sweep` run lands
here by default, so results can be listed (`boxy results`), re-rendered, and
plotted/compared later (`boxy plot`) — replacing the paper's workflow of
hand-transcribing benchmark stdout into plots/*/results.dat.

Layout mirrors the jobs dir (jobs._dir): labs share $HOME across clusters, so
results are partitioned per cluster — <paths.results_root>/<cluster>/ —
and BOXY_RESULTS_DIR pins an EXACT dir (the escape hatch tests use).

One JSON file per bench invocation, schema "boxy-bench/1": an envelope of
provenance (model, cluster, backend, dataset, seed, geometry) plus per-level
`runs` whose keys follow vLLM's `--save-result` names (`max_concurrency`,
`output_throughput`, `mean_ttft_ms`, ...) so synthetic and real backends emit
the SAME shape and plotting is backend-agnostic.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

SCHEMA = "boxy-bench/1"

# The per-level metric keys every backend must emit (vLLM --save-result names;
# p95_e2el_ms is a boxy extra carried by the synthetic backend). Frozen by a
# golden test — extending is fine, renaming is a schema bump.
RUN_KEYS = [
    "max_concurrency", "status",
    "num_prompts", "completed", "failed", "duration",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_e2el_ms", "median_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
]


def _dir() -> Path:
    exact = os.environ.get("BOXY_RESULTS_DIR")
    if exact:
        path = Path(os.path.expanduser(exact))
        path.mkdir(parents=True, exist_ok=True)
        return path
    from boxy import config, jobs

    root = Path(os.path.expanduser(config.get("paths.results_root")))
    path = root / jobs.local_cluster()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()
    return slug or "bench"


def make_envelope(*, url: str, model: str, backend: str, runs: list[dict],
                  endpoints: list[str] | None = None, instance: str = "",
                  label: str = "", dataset: str = "", seed: int = 0,
                  max_tokens: int = 0, backend_detail: str = "",
                  geometry: dict | None = None, accelerator: str = "",
                  gpu_type: str = "") -> dict:
    """The canonical result envelope. The api key is deliberately NOT a
    parameter: secrets never reach the store."""
    from boxy import jobs, version_string

    cluster = jobs.local_cluster()
    if not label:
        base = f"{cluster}/{instance}" if instance else f"{cluster}/{_slug(model)}"
        label = f"{accelerator}: {base}" if accelerator else base
    return {
        "schema": SCHEMA,
        "boxy_version": version_string(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cluster": cluster,
        "instance": instance or None,
        "label": label,
        "url": url,
        "endpoints": endpoints or [url],
        "model": model,
        "bench_backend": backend,
        "backend_detail": backend_detail,
        "dataset": dataset or "synthetic",
        "seed": seed,
        "max_tokens": max_tokens,
        "geometry": geometry or {},
        "accelerator": accelerator,
        "gpu_type": gpu_type,
        "runs": runs,
    }


def write_result(envelope: dict) -> Path:
    """Atomic (tmp + os.replace), same discipline as the endpoint files."""
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}-{_slug(envelope.get('instance') or envelope.get('model', 'bench'))}"
    path = _dir() / f"{stem}.bench.json"
    n = 1
    while path.exists():  # same-second reruns
        path = _dir() / f"{stem}-{n}.bench.json"
        n += 1
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(envelope, indent=1) + "\n")
    os.replace(tmp, path)
    return path


def read_result(path: str | Path) -> dict | None:
    """Shape-guarded: junk files must never KeyError three commands downstream."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if isinstance(data, dict) and all(k in data for k in ("schema", "model", "runs")) \
            and isinstance(data.get("runs"), list):
        return data
    return None


def list_results() -> list[tuple[Path, dict]]:
    """This cluster's results, newest first; unreadable/junk files skipped."""
    out = []
    for path in sorted(_dir().glob("*.bench.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        data = read_result(path)
        if data is not None:
            out.append((path, data))
    return out


def select(tokens: list[str]) -> list[Path]:
    """Resolve `boxy plot` / `boxy results show` arguments to result files:
    no tokens -> the newest result; an integer N -> index into `boxy results`
    (1 = newest); anything else -> a path, or a name/glob matched against this
    cluster's store. Raises ValueError with a helpful message."""
    listing = list_results()
    if not tokens:
        if not listing:
            raise ValueError(f"no bench results in {_dir()} — run `boxy bench` first")
        return [listing[0][0]]
    chosen: list[Path] = []
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok)
            if not 1 <= idx <= len(listing):
                raise ValueError(f"result index {idx} out of range (1..{len(listing)} — see boxy results)")
            chosen.append(listing[idx - 1][0])
            continue
        p = Path(os.path.expanduser(tok))
        if p.exists():
            chosen.append(p)
            continue
        matches = [path for path, _ in listing if tok in path.name]
        if not matches:
            raise ValueError(f"no result matches {tok!r} in {_dir()} (see boxy results)")
        chosen.extend(matches)
    return chosen


def to_csv(envelope: dict) -> str:
    """Plot-ready CSV of the per-level runs (canonical column order)."""
    cols = [k for k in RUN_KEYS if k != "status"]
    lines = [",".join(cols)]
    for run in envelope.get("runs", []):
        if run.get("status") != "ok":
            continue
        row = []
        for c in cols:
            v = run.get(c, "")
            row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def peak_output_throughput(envelope: dict) -> float:
    vals = [r.get("output_throughput", 0.0) for r in envelope.get("runs", []) if r.get("status") == "ok"]
    return max(vals, default=0.0)


def display_label(envelope: dict) -> str:
    """The label as shown in legends/listings: `<accelerator>: cluster/name`.
    Older envelopes predate the accelerator field — enrich at display time
    from the stored field, else infer it from the serving image recorded in
    backend_detail, so existing results get the full legend without a re-run."""
    label = envelope.get("label") or envelope.get("model", "run")
    accel = envelope.get("gpu_type", "") or envelope.get("accelerator", "")
    if not accel:
        from boxy.bench_backends import accel_from_image

        accel = accel_from_image(envelope.get("backend_detail", ""))
    if envelope.get("instance"):
        # the serve record for this instance (still on this machine) may know
        # MORE than the stored envelope: the GPU MODEL (mi300a/h100) outranks
        # the accelerator family, and old envelopes may lack both
        from boxy import jobs

        rec = jobs.read_record(envelope["instance"]) or {}
        better = rec.get("gpu_type", "")
        if better:
            accel = better
        elif not accel:
            accel = rec.get("accelerator", "") or accel_from_image(rec.get("image", ""))
    if not accel:
        return label
    if label.startswith(f"{accel} - "):          # the short-lived dash format
        return f"{accel}: " + label[len(accel) + 3:]
    if label.lower().startswith(f"{accel}:"):
        return label
    return f"{accel}: {label}"
