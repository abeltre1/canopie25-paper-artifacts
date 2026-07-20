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
    dataset_kind: str = "random"        # random | sharegpt | hf | synthetic | file
    dataset_path: str | None = None     # file path, or the HF repo id for kind "hf"
    random_prefix_len: int = 0          # random only: shared-prefix tokens (cache hits)
    hf_cache_dir: str = ""              # hf only: persistent HF_HOME for the bench container
    seed: int = 12345
    api_key: str = ""
    image: str = ""                     # vllm-container only
    runtime: str = ""                   # vllm-container only
    endpoints: list[tuple[str, str]] = field(default_factory=list)  # synthetic fleet mode


def served_model_id(model: str) -> str:
    """boxy records carry the TRANSPORT URI (hf://org/name); the server serves
    the plain id — and the benchmark's tokenizer/model args must match the
    server (field: vllm-bench 'No tokenizer.json for hf://…', then server-side
    tokenization 404s on the mismatched name)."""
    return re.sub(r"^[a-z0-9+._-]+://", "", model or "")


# GPU model tokens, longest-first so mi300a never matches as mi300 etc.
_GPU_TOKENS = ["mi325x", "mi300a", "mi300x", "mi250x", "mi250", "mi210", "mi100",
               "gh200", "h200", "h100", "b200", "b100", "a100", "l40s", "l40",
               "a40", "a30", "v100", "p100", "a6000", "rtx6000", "l4", "t4"]


def gpu_name_from_text(text: str) -> str:
    """The GPU MODEL (mi300a, h100, ...) from any scheduler/hardware text —
    typed GRES lines, node Features, rocm-smi/nvidia-smi output, a system
    card. '' when no known model token appears (never guess)."""
    low = (text or "").lower()
    for tok in _GPU_TOKENS:
        if re.search(rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])", low):
            return tok
    return ""


def accel_from_image(image: str) -> str:
    """Best-effort accelerator from a serving-image name — the label fallback
    for records that predate accelerator recording ('vllm-openai-rocm' -> rocm,
    plain vllm-openai / cuda images -> cuda)."""
    low = (image or "").lower()
    for accel in ("rocm", "cuda", "intel", "vulkan", "musa", "cann"):
        if accel in low:
            return accel
    if "vllm-openai" in low or "ramalama/ramalama" in low:
        return "cuda"
    return ""


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


def _remote_home(target: str) -> str:
    """The cluster account's ABSOLUTE home dir — remote paths that end up in
    podman argv (mounts, HF_HOME) must be literal: a quoted $HOME never
    expands there."""
    from boxy import remote

    rc, home = remote.ssh_capture(target, "echo $HOME", timeout=15)
    home = home.strip().splitlines()[-1] if rc == 0 and home.strip() else ""
    if not home.startswith("/"):
        raise RuntimeError(f"cannot resolve $HOME on {target} — is the ssh session live?")
    return home


def ensure_sharegpt_remote(target: str, progress=print) -> str:
    """ShareGPT staged ON the cluster for the agentless bench: cached in the
    cluster-side boxy store, downloaded once by the login node itself through
    the site proxy (the --fetch-backend pattern — nothing rides the laptop
    link). Returns the ABSOLUTE remote path: the container backend mounts its
    dirname into podman, where a literal $HOME would never expand."""
    import shlex

    from boxy import config, remote

    home = _remote_home(target)
    dest = f"{home}/.local/share/boxy/store/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
    q = shlex.quote(dest)
    rc, _ = remote.ssh_capture(target, f"test -s {q}", timeout=15)
    if rc == 0:
        return dest
    url = config.get("datasets.sharegpt_url")
    pfx = config.get("network.proxy")
    env = f"https_proxy={shlex.quote(pfx)} http_proxy={shlex.quote(pfx)} " if pfx else ""
    progress(f"### staging the ShareGPT corpus on {target} (~650 MB, cached after this once) ...")
    cmd = (f"mkdir -p $(dirname {q}) && "
           f"({env}curl -fsSL {shlex.quote(url)} -o {q}.part || "
           f"{env}wget -q {shlex.quote(url)} -O {q}.part) && "
           f"mv {q}.part {q} && echo BOXY_STAGED")
    rc, out = remote.ssh_capture(target, cmd, timeout=3600)
    if rc != 0 or "BOXY_STAGED" not in out:
        remote.ssh_capture(target, f"rm -f {q}.part", timeout=15)
        tail = " | ".join(out.strip().splitlines()[-3:]) or "no output"
        # SELF-HEAL (field: the site filter 403s huggingface.co on login nodes,
        # same as the spack-source blocks): download on THIS machine — whose
        # proxy/CA config is known-good — and stream it up the live master.
        progress(f"### the cluster could not fetch the corpus itself ({tail}) — "
                 f"downloading here and uploading over the ssh session instead ...")
        try:
            local = ensure_sharegpt()
        except RuntimeError as e:
            raise RuntimeError(
                f"could not stage ShareGPT on {target} ({tail}), and the laptop-side "
                f"fallback failed too: {e}") from e
        if remote.push_path(target, f"{dest}.part", local) != 0 or \
                remote.ssh_capture(target, f"mv {q}.part {q}", timeout=60)[0] != 0:
            remote.ssh_capture(target, f"rm -f {q}.part", timeout=15)
            raise RuntimeError(
                f"could not upload the ShareGPT corpus to {target} — set network.proxy / "
                f"a datasets.sharegpt_url mirror the cluster can reach, or pre-stage it "
                f"by hand: `scp {local} {target}:{dest}`")
        progress(f"### staged from this machine: {target}:{dest}")
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
    if arg.startswith("hf:"):
        repo = arg[3:].strip("/")
        if not repo:
            raise RuntimeError("--dataset hf:<repo-id> needs a HuggingFace dataset id, "
                               "e.g. hf:lmarena-ai/VisionArena-Chat")
        if backend_name in ("vllm-bench", "synthetic"):
            raise RuntimeError(
                f"HF-hub datasets need the vLLM datasets loader — the {backend_name} "
                f"backend can't provide it; use --backend vllm-container (or vllm-cli)")
        return "hf", repo, "HuggingFace hub — downloaded by the benchmark itself (public datasets)"
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
        if spec.random_prefix_len > 0:
            # shared prefix across all generated prompts: real prefix-cache
            # hits with NO corpus staged — the quick cache test on clusters
            flags += ["--random-prefix-len", str(spec.random_prefix_len)]
    elif spec.dataset_kind == "sharegpt":
        flags += ["--dataset-name", "sharegpt", "--dataset-path", spec.dataset_path or ""]
    elif spec.dataset_kind == "hf":
        # --dataset-path is the HF REPO ID; the benchmark downloads it via the
        # datasets library (VisionArena & friends for multimodal models)
        flags += ["--dataset-name", "hf", "--dataset-path", spec.dataset_path or ""]
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
            # newest images route the CLI through the module entrypoint (the
            # wrapper's own migration hint, field: vllm-openai-rocm), then the
            # classic `vllm` script, then the bundled benchmark_serving.py.
            "if python3 -m vllm.entrypoints.cli.main bench serve --help >/dev/null 2>&1; then "
            f"python3 -m vllm.entrypoints.cli.main bench serve {flags} \"$@\"; "
            "elif vllm bench serve --help >/dev/null 2>&1; then "
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
        if spec.dataset_kind == "hf":
            # the loader downloads the repo INSIDE the container: pass the
            # proxies through (values stay out of argv) and persist the HF
            # cache across levels so only level 1 pays the download
            cmd += ["--env", "https_proxy", "--env", "http_proxy",
                    "--env", "HTTPS_PROXY", "--env", "HTTP_PROXY"]
            if spec.hf_cache_dir:
                cmd += ["--env", f"HF_HOME={spec.hf_cache_dir}",
                        "-v", f"{spec.hf_cache_dir}:{spec.hf_cache_dir}"]
        elif spec.dataset_path:
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


class VllmContainerRemote(VllmContainer):
    """The container backend executed on a REMOTE login node over boxy's live
    ssh master — the fully AGENTLESS bench: nothing installed on the HPC, the
    already-pulled serving image provides the benchmark, and the stdout result
    block is parsed laptop-side. The login node reaches the compute-node
    endpoint directly, so no tunnel is involved."""
    name = "vllm-container"

    def __init__(self, image: str, runtime: str, target: str):
        super().__init__(image, runtime)
        self.target = target

    def run_level(self, spec: BenchSpec) -> dict:
        import shlex

        from boxy import config, remote

        cmd = self.render_command(spec)
        line = shlex.join(cmd)
        if spec.dataset_kind == "hf":
            # the in-container HF download rides the site proxy — export it on
            # the remote command so podman's --env passthrough has a value
            pfx = config.get("network.proxy")
            if pfx:
                line = (f"https_proxy={shlex.quote(pfx)} http_proxy={shlex.quote(pfx)} "
                        f"HTTPS_PROXY={shlex.quote(pfx)} HTTP_PROXY={shlex.quote(pfx)} {line}")
        rc, out = remote.ssh_capture(self.target, line, timeout=3600)
        parsed = parse_stdout_block(out, spec.concurrency)
        if parsed:
            return parsed
        tail = out.strip().splitlines()[-3:]
        return {"max_concurrency": spec.concurrency, "status": "error",
                "error": f"remote exit {rc}: " + " | ".join(tail)}


REMOTE_BIN = "$HOME/.local/share/boxy/store/bin/vllm-bench"


class VllmBenchRemote(BenchBackend):
    """The vllm-bench static binary ON the cluster login node, run over boxy's
    ssh master — the fastest agentless path once `boxy bench --fetch-backend
    --ssh <cluster>` has installed it there. No container, no image pull; the
    save-result JSON is echoed back behind a marker and parsed laptop-side."""
    name = "vllm-bench"

    def __init__(self, target: str):
        self.target = target

    def available(self) -> tuple[bool, str]:
        from boxy import remote

        rc, _ = remote.ssh_capture(self.target, f"test -x {REMOTE_BIN}", timeout=15)
        if rc != 0:
            return False, (f"no vllm-bench on {self.target} — install it once with "
                           f"`boxy bench --fetch-backend --ssh {self.target}`")
        return True, f"vllm-bench binary on {self.target} (installed via --fetch-backend)"

    def run_level(self, spec: BenchSpec) -> dict:
        import shlex

        from boxy import remote

        host = re.sub(r"^https?://", "", spec.url).split("/")[0].split(":")[0]
        flags = shlex.join(_serve_flags(spec))
        cmd = (f"d=$(mktemp -d); no_proxy={shlex.quote(host)} NO_PROXY={shlex.quote(host)} "
               f"{REMOTE_BIN} {flags} --save-result --result-dir \"$d\" "
               f"--result-filename r.json; rc=$?; "
               f"echo BOXY_RESULT_JSON; cat \"$d/r.json\" 2>/dev/null; rm -rf \"$d\"; exit $rc")
        rc, out = remote.ssh_capture(self.target, cmd, timeout=3600)
        head, _, tail = out.partition("BOXY_RESULT_JSON")
        tail = tail.strip()
        if tail.startswith("{"):
            try:
                return _normalize_saved(json.loads(tail), spec.concurrency)
            except ValueError:
                pass
        parsed = parse_stdout_block(head, spec.concurrency)
        if parsed:
            return parsed
        lines = head.strip().splitlines()[-3:]
        return {"max_concurrency": spec.concurrency, "status": "error",
                "error": f"remote exit {rc}: " + " | ".join(lines)}


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


_CACHE_LINE = re.compile(
    r"^vllm:(?:gpu_)?prefix_cache_(hits|queries)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
_CACHE_GAUGE = re.compile(
    r"^vllm:(?:gpu_)?prefix_cache_hit_rate(?:\{[^}]*\})?\s+([0-9.eE+-]+)")


def parse_cache_metrics(text: str) -> dict | None:
    """vLLM's Prometheus /metrics -> {'hits': float, 'queries': float,
    'gauge': float|None}. Counters are summed across label sets; the old
    hit-rate gauge is the fallback for engines that don't export counters.
    None when the text carries no prefix-cache metrics at all."""
    hits = queries = 0.0
    gauge = None
    seen = False
    for line in (text or "").splitlines():
        m = _CACHE_LINE.match(line.strip())
        if m:
            seen = True
            if m.group(1) == "hits":
                hits += float(m.group(2))
            else:
                queries += float(m.group(2))
            continue
        g = _CACHE_GAUGE.match(line.strip())
        if g:
            seen = True
            gauge = float(g.group(1))
    return {"hits": hits, "queries": queries, "gauge": gauge} if seen else None


def cache_rate_delta(before: dict | None, after: dict | None) -> float | None:
    """Per-level prefix-cache hit rate in PERCENT from two /metrics samples;
    None when it can't be known (no metrics, no queries this level)."""
    if not after:
        return None
    if before and after["queries"] > before["queries"]:
        dq = after["queries"] - before["queries"]
        dh = after["hits"] - before["hits"]
        return max(0.0, min(100.0, 100.0 * dh / dq))
    if after.get("gauge") is not None:
        return max(0.0, min(100.0, 100.0 * after["gauge"]))
    return None


def auto_num_prompts(conc: int) -> int:
    """Prompt-pool size for a sweep level: ~10x the concurrency, floored at 32,
    capped at 1000 — but the cap itself must scale past 256, otherwise a
    1024-level "sweep" issues only 1000 prompts and can never actually hold
    1024 requests in flight (field: the top rung looked like a scaling wall
    when it was a drained queue)."""
    return min(max(10 * conc, 32), max(1000, 3 * conc))


def run_series(backend: BenchBackend, base: BenchSpec, concurrencies: list[int],
               progress=print, metrics_sampler=None) -> list[dict]:
    """The concurrency sweep: one canonical record per level; an errored level
    is kept (plots show a gap) and the series continues — a 1024-level crash
    must not discard the 1..512 measurements (the paper's 405B run).
    metrics_sampler (optional, () -> parse_cache_metrics dict) is called
    around each level to attach the server-side prefix-cache hit rate."""
    records = []
    before = metrics_sampler() if metrics_sampler else None
    for conc in concurrencies:
        spec = BenchSpec(**{**base.__dict__, "concurrency": conc,
                            "num_prompts": base.num_prompts or auto_num_prompts(conc)})
        rec = backend.run_level(spec)
        if metrics_sampler:
            after = metrics_sampler()
            rate = cache_rate_delta(before, after)
            if rate is not None:
                rec["prefix_cache_hit_rate"] = rate
            before = after
        records.append(rec)
        if rec.get("status") == "ok":
            progress(f"###   concurrency {conc}: {rec.get('output_throughput', 0.0):.1f} tok/s "
                     f"({rec.get('completed', 0)}/{rec.get('num_prompts', '?')} ok)")
        else:
            progress(f"###   concurrency {conc}: FAILED ({rec.get('error', '?')}) — continuing")
    if metrics_sampler and not any("prefix_cache_hit_rate" in r for r in records):
        progress("###   cache metrics: the server exported no prefix-cache series — "
                 "enable it at serve time (`boxy serve ... -- --enable-prefix-caching`; "
                 "vLLM V1 engines have it on by default)")
    return records
