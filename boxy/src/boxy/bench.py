"""Phase 4: throughput/latency sweep against an OpenAI-compatible endpoint.

Reproduces the paper's benchmark shape (hpc-workflow/5-run-benchmark.sh:
batch sizes 1..1024 against a served model) with a self-contained, stdlib-only
load generator — no network access needed beyond the endpoint itself, so it
works air-gapped. Results export as plot-ready CSV matching the metrics used
in the paper's plots/ directory (throughput + latency per batch size).

The prompt set is a dataset file if given (JSON list of strings, or ShareGPT
JSON — first human turn of each conversation), else a small synthetic set.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

SYNTHETIC_PROMPTS = [
    "Explain what a batch scheduler does on an HPC system.",
    "Summarize the difference between Podman and Apptainer in two sentences.",
    "Write a haiku about tensor parallelism.",
    "What is an air-gapped deployment and why do HPC centers use them?",
    "List three considerations when serving an LLM on shared GPUs.",
    "Describe the purpose of a SIF file in one sentence.",
    "Why would a site pin OMP_NUM_THREADS=1 for inference servers?",
    "Explain rootless containers to a new HPC user.",
]


def percentile_ms(latencies: list[float], p: float) -> float:
    """Nearest-rank percentile in ms. ceil(p*n)-1: int(p*n) sat one rank high,
    so p50 of two samples reported the MAX (sweep finding 49)."""
    import math

    if not latencies:
        return 0.0
    ordered = sorted(latencies)
    rank = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[rank] * 1000


@dataclass
class BenchResult:
    batch_size: int
    requests: int
    ok: int
    errors: int
    elapsed_s: float
    requests_per_s: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_s: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class BenchReport:
    url: str
    model: str
    max_tokens: int
    results: list[BenchResult] = field(default_factory=list)

    def to_csv(self) -> str:
        cols = list(BenchResult.__dataclass_fields__)
        lines = [",".join(cols)]
        for r in self.results:
            d = asdict(r)
            lines.append(",".join(f"{d[c]:.3f}" if isinstance(d[c], float) else str(d[c]) for c in cols))
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps({"url": self.url, "model": self.model, "max_tokens": self.max_tokens,
                           "results": [asdict(r) for r in self.results]}, indent=1)


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(SYNTHETIC_PROMPTS)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: no prompts found (expected JSON list of strings or ShareGPT JSON)")
    if data and isinstance(data[0], str):
        return data
    # ShareGPT format: [{"conversations": [{"from": "human", "value": ...}, ...]}, ...]
    prompts = []
    for item in data:
        for turn in item.get("conversations", []):
            if turn.get("from") == "human":
                prompts.append(turn["value"])
                break
    if not prompts:
        raise ValueError(f"{path}: no prompts found (expected JSON list of strings or ShareGPT JSON)")
    return prompts


def _http_json(url: str, payload: dict | None = None, timeout: float = 120.0) -> dict:
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def discover_model(url: str) -> str:
    """The served model id from GET /v1/models (vLLM requires it in requests)."""
    data = _http_json(f"{url}/v1/models")
    return data["data"][0]["id"]


def _one_request(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, dict | None]:
    start = time.perf_counter()
    try:
        body = _http_json(
            f"{url}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": max_tokens},
        )
        return time.perf_counter() - start, body.get("usage") or {}
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return time.perf_counter() - start, None


def run_level(url: str, model: str, prompts: list[str], batch_size: int, max_tokens: int) -> BenchResult:
    """One sweep level: `batch_size` concurrent requests (one wave, paper-style)."""
    return run_level_endpoints([(url, model)], prompts, batch_size, max_tokens)


def run_level_endpoints(endpoints: list[tuple[str, str]], prompts: list[str],
                        batch_size: int, max_tokens: int) -> BenchResult:
    """One sweep level spread across N (url, model) endpoints: `batch_size`
    concurrent requests round-robined across the endpoints. For a single endpoint
    this is the classic per-batch wave; for K data-parallel replicas it measures
    the AGGREGATE throughput of the whole fleet (raw latencies pooled across all
    endpoints, so p50/p95 are true fleet percentiles)."""
    wave = [(endpoints[i % len(endpoints)], prompts[i % len(prompts)]) for i in range(batch_size)]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, batch_size)) as pool:
        outcomes = list(pool.map(
            lambda ep_p: _one_request(ep_p[0][0], ep_p[0][1], ep_p[1], max_tokens), wave))
    elapsed = time.perf_counter() - start

    latencies = sorted(lat for lat, _ in outcomes)
    usages = [u for _, u in outcomes if u is not None]
    prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usages)
    completion_tokens = sum(u.get("completion_tokens", 0) for u in usages)

    def pct(p: float) -> float:
        return percentile_ms(latencies, p)

    return BenchResult(
        batch_size=batch_size,
        requests=len(wave),
        ok=len(usages),
        errors=len(wave) - len(usages),
        elapsed_s=elapsed,
        requests_per_s=len(usages) / elapsed if elapsed else 0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_s=completion_tokens / elapsed if elapsed else 0.0,
        latency_mean_ms=statistics.mean(latencies) * 1000 if latencies else 0.0,
        latency_p50_ms=pct(0.50),
        latency_p95_ms=pct(0.95),
    )


def run_bench(
    url: str,
    batch_sizes: list[int],
    max_tokens: int = 32,
    dataset: str | None = None,
    model: str | None = None,
) -> BenchReport:
    prompts = load_prompts(dataset)
    resolved_model = model or discover_model(url)
    report = BenchReport(url=url, model=resolved_model, max_tokens=max_tokens)
    for batch_size in batch_sizes:
        report.results.append(run_level(url, resolved_model, prompts, batch_size, max_tokens))
    return report


# ---- scaling sweep: one rung per node/replica count, compared side by side ----


@dataclass
class ScalingPoint:
    """One rung of a scaling sweep, summarized at its peak-throughput batch level."""
    label: str        # "nodes=2" | "replicas=4"
    axis: str         # "nodes" | "replicas"
    value: int        # 2, 4, 8, ...
    endpoints: int    # how many server URLs served this rung
    peak_batch: int   # batch size at which tokens/s peaked
    requests_per_s: float
    tokens_per_s: float
    latency_p50_ms: float
    latency_p95_ms: float


@dataclass
class ScalingReport:
    axis: str
    model: str
    max_tokens: int
    points: list[ScalingPoint] = field(default_factory=list)

    def to_csv(self) -> str:
        cols = list(ScalingPoint.__dataclass_fields__)
        lines = [",".join(cols)]
        for p in self.points:
            d = asdict(p)
            lines.append(",".join(f"{d[c]:.3f}" if isinstance(d[c], float) else str(d[c]) for c in cols))
        return "\n".join(lines) + "\n"

    def to_json(self) -> str:
        return json.dumps({"axis": self.axis, "model": self.model, "max_tokens": self.max_tokens,
                           "points": [asdict(p) for p in self.points]}, indent=1)

    def to_table(self) -> str:
        head = (f"{self.axis:>9} {'servers':>7} {'peakBS':>6} {'req/s':>8} "
                f"{'tok/s':>9} {'p50 ms':>9} {'p95 ms':>9}")
        rows = [head]
        base = self.points[0].tokens_per_s if self.points and self.points[0].tokens_per_s else 0.0
        for p in self.points:
            speedup = f"{p.tokens_per_s / base:.2f}x" if base else "-"
            rows.append(f"{p.value:>9} {p.endpoints:>7} {p.peak_batch:>6} {p.requests_per_s:>8.2f} "
                        f"{p.tokens_per_s:>9.1f} {p.latency_p50_ms:>9.1f} {p.latency_p95_ms:>9.1f}  {speedup}")
        return "\n".join(rows)


def summarize_point(label: str, axis: str, value: int, endpoints: int,
                    report: BenchReport) -> ScalingPoint:
    """Reduce a per-rung batch sweep to a single row at its peak tokens/s."""
    peak = max(report.results, key=lambda r: r.tokens_per_s, default=None)
    if peak is None:
        return ScalingPoint(label, axis, value, endpoints, 0, 0.0, 0.0, 0.0, 0.0)
    return ScalingPoint(label=label, axis=axis, value=value, endpoints=endpoints,
                        peak_batch=peak.batch_size, requests_per_s=peak.requests_per_s,
                        tokens_per_s=peak.tokens_per_s, latency_p50_ms=peak.latency_p50_ms,
                        latency_p95_ms=peak.latency_p95_ms)


def run_scaling_point(urls: list[str], batch_sizes: list[int], max_tokens: int = 32,
                      dataset: str | None = None, models: list[str] | None = None) -> BenchReport:
    """Benchmark ONE rung (its set of endpoints) across the batch sweep, aggregating
    load over all its URLs. Returns a BenchReport whose `url` field lists the fleet."""
    prompts = load_prompts(dataset)
    resolved = models or [discover_model(u) for u in urls]
    endpoints = list(zip(urls, resolved))
    report = BenchReport(url=",".join(urls), model=resolved[0] if resolved else "",
                         max_tokens=max_tokens)
    for batch_size in batch_sizes:
        report.results.append(run_level_endpoints(endpoints, prompts, batch_size, max_tokens))
    return report
