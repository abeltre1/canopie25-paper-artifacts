"""Real-model benchmark backends: run the OFFICIAL vLLM load generator against
whatever boxy serves, with the built-in synthetic generator as the always-works
fallback. One `run_level()` call = one concurrency level (exactly the paper's
hpc-workflow/5-run-benchmark.sh loop and `vllm bench serve --max-concurrency`).

The auto ladder, most- to least-preferred (each pick prints its provenance):

1. vllm-bench   — the standalone Rust binary (vllm-project/vllm-bench): a
                  drop-in `vllm bench serve` replacement, static, benches any
                  OpenAI endpoint with NOTHING else installed. Fetch it once
                  with `boxy bench --fetch-backend` (air-gap: mirror via
                  config urls.vllm_bench, or bundle --bench).
2. vllm-container — the paper's own trick: run the benchmark INSIDE the
                  already-pulled serving image (needs a job record naming the
                  image + a working container runtime). Probes `vllm bench
                  serve`, falls back to benchmark_serving.py, and finally
                  parses the stdout result block — so it works across image
                  generations.
3. vllm-cli     — a locally installed `vllm` (pip install vllm; heavy).
4. synthetic    — bench.py's stdlib generator (air-gapped default, labeled).

Every backend emits the SAME canonical run record (results.RUN_KEYS — vLLM's
--save-result key names), so `boxy plot`/`boxy results` never care which one
produced a number.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from boxy import bench, results


@dataclass
class BenchSpec:
    """Everything one benchmark level needs. api_key rides the process env /
    request header only — never argv (visible in ps) and never the results."""
    url: str
    model: str
    concurrency: int
    num_prompts: int
    max_tokens: int
    dataset_kind: str = "random"        # random | sharegpt | synthetic | file
    dataset_path: str | None = None
    seed: int = 12345
    api_key: str = ""
    image: str = ""                     # vllm-container only
    runtime: str = ""                   # vllm-container only
    endpoints: list[tuple[str, str]] = field(default_factory=list)  # synthetic fleet mode


def _no_proxy_env(url: str) -> dict[str, str]:
    """Child env for subprocess backends: the bench target must be reached
    DIRECTLY (the corporate proxy can't see compute nodes / localhost tunnels)
    — the same `no_proxy=${no_proxy},${SERVER_NODE}` the paper's script set."""
    env = dict(os.environ)
    host = re.sub(r"^https?://", "", url).split("/")[0].split(":")[0]
    for var in ("no_proxy", "NO_PROXY"):
        prev = env.get(var, "")
        env[var] = f"{prev},{host}" if prev else host
    return env


def _normalize_saved(data: dict, concurrency: int) -> dict:
    """A vllm --save-result JSON (Python or Rust writer) -> canonical record.
    Defensive: required keys default to 0 so upstream drift degrades to zeros,
    never a crash; unknown extra keys are dropped."""
    record = {k: data.get(k, 0) for k in results.RUN_KEYS
              if k not in ("max_concurrency", "status")}
    record["max_concurrency"] = data.get("max_concurrency") or concurrency
    record["status"] = "ok" if data.get("completed") else "error"
    # p95 e2e is a boxy extra the real tools don't emit — leave 0 when absent
    return record


# ---- the stdout-block parser: works on EVERY vllm generation ----------------

_BLOCK_KEYMAP = {
    "successful requests": "completed",
    "benchmark duration (s)": "duration",
    "total input tokens": "total_input_tokens",
    "total generated tokens": "total_output_tokens",
    "request throughput (req/s)": "request_throughput",
    "output token throughput (tok/s)": "output_throughput",
    "total token throughput (tok/s)": "total_token_throughput",
    "mean ttft (ms)": "mean_ttft_ms",
    "median ttft (ms)": "median_ttft_ms",
    "p99 ttft (ms)": "p99_ttft_ms",
    "mean tpot (ms)": "mean_tpot_ms",
    "median tpot (ms)": "median_tpot_ms",
    "p99 tpot (ms)": "p99_tpot_ms",
    "mean itl (ms)": "mean_itl_ms",
    "median itl (ms)": "median_itl_ms",
    "p99 itl (ms)": "p99_itl_ms",
    "mean e2el (ms)": "mean_e2el_ms",
    "median e2el (ms)": "median_e2el_ms",
    "p99 e2el (ms)": "p99_e2el_ms",
}


def parse_stdout_block(text: str, concurrency: int) -> dict | None:
    """Parse the `============ Serving Benchmark Result ============` stdout
    block (the exact text the paper hand-transcribed into results.dat) into a
    canonical record. None when no block is present."""
    if "Serving Benchmark Result" not in text:
        return None
    values: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9 ()/.]+?):\s+([0-9.]+)\s*$", line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower()
        if key in _BLOCK_KEYMAP:
            values[_BLOCK_KEYMAP[key]] = float(m.group(2))
    if "output_throughput" not in values:
        return None
    record = {k: values.get(k, 0) for k in results.RUN_KEYS
              if k not in ("max_concurrency", "status", "num_prompts")}
    record["max_concurrency"] = concurrency
    record["num_prompts"] = int(values.get("completed", 0))
    record["completed"] = int(values.get("completed", 0))
    record["failed"] = 0
    record["status"] = "ok" if record["completed"] else "error"
    return record


# ---- dataset resolution -----------------------------------------------------

def dataset_cache_dir() -> Path:
    from boxy import config

    path = Path(os.path.expanduser(config.get("paths.datasets")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_sharegpt() -> Path:
    """The ShareGPT corpus the paper benched with, downloaded once through the
    proxy/CA-aware opener and cached. Air-gapped sites pre-stage the file at
    the cache path (or set datasets.sharegpt_url to a mirror)."""
    from boxy import config
    from boxy.cardgen import _opener

    dest = dataset_cache_dir() / "ShareGPT_V3_unfiltered_cleaned_split.json"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = config.get("datasets.sharegpt_url")
    part = dest.with_suffix(".part")
    try:
        with _opener().open(url, timeout=600) as resp, open(part, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(part, dest)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"cannot download the ShareGPT dataset ({e}) — pre-stage it at {dest} "
            f"(air-gapped sites: `boxy bundle MODEL --bench` carries it), or point "
            f"datasets.sharegpt_url at a mirror") from e
    return dest


def resolve_dataset(arg: str | None, backend_name: str) -> tuple[str, str | None, str]:
    """(kind, path, provenance). Default keeps zero-flag turnkey: `random` for
    real backends (no download), the built-in prompts for synthetic."""
    if arg in (None, ""):
        if backend_name == "synthetic":
            return "synthetic", None, "built-in prompt set"
        return "random", None, "no download needed (--dataset sharegpt for the paper's corpus)"
    if arg == "random":
        return "random", None, "requested"
    if arg == "sharegpt":
        path = ensure_sharegpt()
        return "sharegpt", str(path), f"cached at {path}"
    path = os.path.expanduser(arg)
    if not os.path.exists(path):
        raise RuntimeError(f"--dataset {arg!r}: not a known name (random|sharegpt) and no such file")
    return "file", path, "user file"


# ---- binary fetch -----------------------------------------------------------

def _store_bin_dir() -> Path:
    from boxy import config

    path = Path(os.path.expanduser(config.get("paths.store"))) / "bin"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_vllm_bench() -> str:
    """The vllm-bench binary: config binaries.vllm_bench (PATH or full path),
    else the boxy store copy. '' when absent."""
    from boxy import config

    name = config.get("binaries.vllm_bench")
    found = shutil.which(os.path.expanduser(name))
    if found:
        return found
    store = _store_bin_dir() / "vllm-bench"
    return str(store) if store.exists() else ""


def fetch_vllm_bench() -> Path:
    """Download the static vllm-bench binary into <store>/bin (proxy/CA-aware).
    Air-gapped sites point urls.vllm_bench at an internal mirror."""
    import sys

    from boxy import config
    from boxy.cardgen import _opener

    if sys.platform == "darwin":
        raise RuntimeError(
            "vllm-bench ships static LINUX binaries only — this Mac can't run them. "
            "Either build natively (`cargo install --git "
            "https://github.com/vllm-project/vllm-bench vllm-bench`), or fetch on the "
            "cluster instead: over --ssh the bench runs cluster-side anyway, where the "
            "serving image can also provide the benchmark with no install at all "
            "(the vllm-container backend).")
    arch = {"x86_64": "x86_64", "amd64": "x86_64",
            "aarch64": "aarch64", "arm64": "aarch64"}.get(platform.machine().lower(), "x86_64")
    url = config.get("urls.vllm_bench").format(arch=arch)
    dest = _store_bin_dir() / "vllm-bench"
    part = dest.with_suffix(".part")
    try:
        with _opener().open(url, timeout=600) as resp, open(part, "wb") as f:
            shutil.copyfileobj(resp, f)
    except OSError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"cannot download vllm-bench from {url} ({e}) — if your site needs a proxy "
            f"for outbound HTTPS, set network.proxy (BOXY_PROXY) first; otherwise set "
            f"urls.vllm_bench to a mirror, or drop a binary at {dest}") from e
    os.replace(part, dest)
    dest.chmod(0o755)
    return dest


# ---- backends ---------------------------------------------------------------

class BenchBackend:
    name = "abstract"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def run_level(self, spec: BenchSpec) -> dict:
        """One concurrency level -> one canonical record. Never raises: any
        failure becomes a status:'error' record (the paper's X cells) so the
        series continues to the next level."""
        raise NotImplementedError


class SyntheticBackend(BenchBackend):
    name = "synthetic"

    def available(self) -> tuple[bool, str]:
        return True, "built-in stdlib load generator (bench.py)"

    def run_level(self, spec: BenchSpec) -> dict:
        prompts = bench.load_prompts(spec.dataset_path)
        endpoints = spec.endpoints or [(spec.url, spec.model)]
        try:
            r = bench.run_level_endpoints(endpoints, prompts, spec.concurrency, spec.max_tokens)
        except Exception as e:  # noqa: BLE001 — an errored level must not kill the series
            return {"max_concurrency": spec.concurrency, "status": "error", "error": str(e)}
        return bench.to_canonical(r)


def _run_and_collect(cmd: list[str], spec: BenchSpec, env: dict[str, str],
                     timeout: float = 3600.0) -> dict:
    """Run one save-result-capable benchmark command; prefer its JSON, fall
    back to the stdout block, else an error record with the tail of stderr."""
    with tempfile.TemporaryDirectory(prefix="boxy-bench-") as tmp:
        full = cmd + ["--save-result", "--result-dir", tmp,
                      "--result-filename", "level.json"]
        try:
            proc = subprocess.run(full, capture_output=True, text=True,
                                  timeout=timeout, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"max_concurrency": spec.concurrency, "status": "error", "error": str(e)}
        saved = Path(tmp) / "level.json"
        if saved.exists():
            try:
                return _normalize_saved(json.loads(saved.read_text()), spec.concurrency)
            except ValueError:
                pass
        parsed = parse_stdout_block(proc.stdout, spec.concurrency)
        if parsed:
            return parsed
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"max_concurrency": spec.concurrency, "status": "error",
                "error": f"exit {proc.returncode}: " + " | ".join(tail)}


def _serve_flags(spec: BenchSpec) -> list[str]:
    flags = ["--backend", "openai-chat", "--endpoint", "/v1/chat/completions",
             "--base-url", spec.url, "--model", spec.model,
             "--num-prompts", str(spec.num_prompts),
             "--max-concurrency", str(spec.concurrency),
             "--seed", str(spec.seed)]
    if spec.dataset_kind in ("random", "synthetic"):
        flags += ["--dataset-name", "random",
                  "--random-output-len", str(spec.max_tokens)]
    elif spec.dataset_kind == "sharegpt":
        flags += ["--dataset-name", "sharegpt", "--dataset-path", spec.dataset_path or ""]
    else:
        flags += ["--dataset-name", "custom", "--dataset-path", spec.dataset_path or ""]
    return flags


class VllmBenchBinary(BenchBackend):
    name = "vllm-bench"

    def available(self) -> tuple[bool, str]:
        path = find_vllm_bench()
        if not path:
            return False, ("vllm-bench binary not found — `boxy bench --fetch-backend` "
                           "downloads the static build")
        return True, f"vllm-bench static binary at {path}"

    def run_level(self, spec: BenchSpec) -> dict:
        env = _no_proxy_env(spec.url)
        if spec.api_key:
            env["OPENAI_API_KEY"] = spec.api_key
        return _run_and_collect([find_vllm_bench()] + _serve_flags(spec), spec, env)


class VllmCli(BenchBackend):
    name = "vllm-cli"

    def available(self) -> tuple[bool, str]:
        path = shutil.which("vllm")
        if not path:
            return False, "no `vllm` CLI on PATH (pip install vllm)"
        return True, f"vllm CLI at {path} (vllm bench serve)"

    def run_level(self, spec: BenchSpec) -> dict:
        env = _no_proxy_env(spec.url)
        if spec.api_key:
            env["OPENAI_API_KEY"] = spec.api_key
        return _run_and_collect(["vllm", "bench", "serve"] + _serve_flags(spec), spec, env)


class VllmContainer(BenchBackend):
    """The paper's trick: the serving image already contains the benchmark —
    run it there. Needs the image name (from the job record or --image) and a
    working container runtime. The in-container command probes `vllm bench
    serve` first and falls back to the bundled benchmark_serving.py (0.9.x-era
    images, e.g. the rocm builds the paper used)."""
    name = "vllm-container"

    def __init__(self, image: str = "", runtime: str = ""):
        self.image, self.runtime = image, runtime

    def available(self) -> tuple[bool, str]:
        if not self.image:
            return False, "no serving image known (bench a boxy-served instance, or pass --image)"
        runtime = self.runtime or next(
            (c for c in ("podman", "docker") if shutil.which(c)), "")
        if not runtime:
            return False, "no container runtime (podman/docker) here"
        self.runtime = runtime
        return True, f"benchmark inside the serving image {self.image} via {runtime}"

    def render_command(self, spec: BenchSpec) -> list[str]:
        flags = " ".join(_serve_flags(spec))
        inner = (
            "if vllm bench serve --help >/dev/null 2>&1; then "
            f"vllm bench serve {flags} \"$@\"; "
            "elif [ -f /app/vllm/benchmarks/benchmark_serving.py ]; then "
            f"python3 /app/vllm/benchmarks/benchmark_serving.py {flags} \"$@\"; "
            "elif [ -f /vllm-workspace/benchmarks/benchmark_serving.py ]; then "
            f"python3 /vllm-workspace/benchmarks/benchmark_serving.py {flags} \"$@\"; "
            "else echo 'boxy: no benchmark tool in this image' >&2; exit 9; fi")
        cmd = [self.runtime, "run", "--rm", "--network=host"]
        host = re.sub(r"^https?://", "", spec.url).split("/")[0].split(":")[0]
        cmd += ["--env", f"no_proxy={host}", "--env", f"NO_PROXY={host}"]
        if spec.api_key:
            cmd += ["--env", "OPENAI_API_KEY"]
        if spec.dataset_path:
            d = os.path.dirname(os.path.abspath(spec.dataset_path))
            cmd += ["-v", f"{d}:{d}:ro"]
        cmd += ["--entrypoint", "/bin/bash", self.image, "-c", inner, "bench"]
        return cmd

    def run_level(self, spec: BenchSpec) -> dict:
        env = _no_proxy_env(spec.url)
        if spec.api_key:
            env["OPENAI_API_KEY"] = spec.api_key
        cmd = self.render_command(spec)
        # --save-result lands inside the container; rely on the stdout block,
        # which every image generation prints.
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600.0, env=env)
        except (OSError, subprocess.TimeoutExpired) as e:
            return {"max_concurrency": spec.concurrency, "status": "error", "error": str(e)}
        parsed = parse_stdout_block(proc.stdout, spec.concurrency)
        if parsed:
            return parsed
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        return {"max_concurrency": spec.concurrency, "status": "error",
                "error": f"exit {proc.returncode}: " + " | ".join(tail)}


def pick_backend(requested: str, *, image: str = "", runtime: str = "",
                 fleet: bool = False) -> tuple[BenchBackend, str]:
    """Resolve --backend/config bench.backend to a working backend + the
    provenance for the `auto: bench backend:` line. Explicit names are hard
    requirements (RuntimeError when unusable). A multi-endpoint fleet pins
    synthetic — the only backend that pools true fleet percentiles."""
    candidates: dict[str, BenchBackend] = {
        "synthetic": SyntheticBackend(),
        "vllm-bench": VllmBenchBinary(),
        "vllm-cli": VllmCli(),
        "vllm-container": VllmContainer(image, runtime),
    }
    if fleet:
        ok, why = candidates["synthetic"].available()
        return candidates["synthetic"], ("replica fleet — synthetic pools true fleet "
                                         "percentiles across all endpoints")
    req = requested or "auto"
    if req != "auto":
        if req not in candidates:
            raise RuntimeError(f"unknown bench backend {req!r} — one of "
                               f"auto|{'|'.join(candidates)}")
        backend = candidates[req]
        ok, why = backend.available()
        if not ok:
            raise RuntimeError(f"bench backend {req!r} unavailable: {why}")
        return backend, why
    for name in ("vllm-bench", "vllm-container", "vllm-cli"):
        backend = candidates[name]
        ok, why = backend.available()
        if ok:
            return backend, why
    return candidates["synthetic"], ("built-in load generator — for the official vLLM "
                                     "benchmark run `boxy bench --fetch-backend` once")


def run_series(backend: BenchBackend, base: BenchSpec, concurrencies: list[int],
               progress=print) -> list[dict]:
    """The concurrency sweep: one canonical record per level; an errored level
    is kept (plots show a gap) and the series continues — a 1024-level crash
    must not discard the 1..512 measurements (the paper's 405B run)."""
    records = []
    for conc in concurrencies:
        spec = BenchSpec(**{**base.__dict__, "concurrency": conc,
                            "num_prompts": base.num_prompts or min(max(10 * conc, 32), 1000)})
        rec = backend.run_level(spec)
        records.append(rec)
        if rec.get("status") == "ok":
            progress(f"###   concurrency {conc}: {rec.get('output_throughput', 0.0):.1f} tok/s "
                     f"({rec.get('completed', 0)}/{rec.get('num_prompts', '?')} ok)")
        else:
            progress(f"###   concurrency {conc}: FAILED ({rec.get('error', '?')}) — continuing")
    return records
