"""Real-benchmark backends: flag mapping to the official vLLM tools (via PATH
shims that record argv+env), the auto ladder + provenance, the stdout-block
parser (the text the paper hand-transcribed), dataset resolution, and the
container command render."""

import json
import os
import stat
import textwrap

import pytest

from boxy import bench_backends as bb
from boxy import results

# a captured real `Serving Benchmark Result` block (hpc-workflow example-output shape)
STDOUT_BLOCK = textwrap.dedent("""\
    ============ Serving Benchmark Result ============
    Successful requests:                     1000
    Benchmark duration (s):                  187.42
    Total input tokens:                      215196
    Total generated tokens:                  198532
    Request throughput (req/s):              5.34
    Output token throughput (tok/s):         1059.30
    Total token throughput (tok/s):          2207.45
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          412.11
    Median TTFT (ms):                        301.45
    P99 TTFT (ms):                           2210.87
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          88.32
    Median TPOT (ms):                        84.15
    P99 TPOT (ms):                           190.02
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           86.90
    Median ITL (ms):                         79.11
    P99 ITL (ms):                            201.33
    ==================================================
""")


def _shim(tmp_path, monkeypatch, name, script):
    d = tmp_path / "shims"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(f"#!/bin/sh\n{script}\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")
    return p


# ---------- stdout-block parser ----------

def test_parse_stdout_block_golden():
    rec = bb.parse_stdout_block(STDOUT_BLOCK, 64)
    assert rec is not None and rec["status"] == "ok"
    assert rec["max_concurrency"] == 64
    assert rec["completed"] == 1000
    assert rec["output_throughput"] == pytest.approx(1059.30)
    assert rec["median_ttft_ms"] == pytest.approx(301.45)
    assert rec["p99_itl_ms"] == pytest.approx(201.33)
    assert rec["median_e2el_ms"] == 0                       # block has no E2E rows: zeros, not KeyError
    assert set(rec) >= {k for k in results.RUN_KEYS if k != "duration"} - {"num_prompts"}


def test_parse_stdout_block_absent():
    assert bb.parse_stdout_block("no benchmark here", 1) is None


# ---------- vllm-bench binary backend ----------

def test_vllm_bench_flag_mapping_and_no_proxy(tmp_path, monkeypatch):
    """The shim records argv + env, then writes a save-result JSON where the
    backend asked for it — proving flag mapping AND JSON normalization."""
    log = tmp_path / "argv.json"
    _shim(tmp_path, monkeypatch, "vllm-bench", textwrap.dedent(f"""\
        args="$@"
        outdir=""; fname=""
        while [ $# -gt 0 ]; do
          [ "$1" = "--result-dir" ] && outdir="$2"
          [ "$1" = "--result-filename" ] && fname="$2"
          shift
        done
        printf '{{"argv": "%s", "no_proxy": "%s", "key": "%s"}}' "$args" "$no_proxy" "$OPENAI_API_KEY" > {log}
        printf '{{"max_concurrency": 8, "completed": 32, "failed": 0, "duration": 3.0,
                  "request_throughput": 10.7, "output_throughput": 341.9,
                  "mean_ttft_ms": 21.0, "median_ttft_ms": 20.0, "p99_ttft_ms": 30.0}}' > "$outdir/$fname"
    """))
    monkeypatch.setenv("HOME", str(tmp_path))               # keep the store local
    backend = bb.VllmBenchBinary()
    ok, why = backend.available()
    assert ok and "vllm-bench" in why
    spec = bb.BenchSpec(url="http://node7:8000", model="m/x", concurrency=8,
                        num_prompts=32, max_tokens=16, seed=12345, api_key="tok123")
    rec = backend.run_level(spec)
    seen = json.loads(log.read_text())
    assert "--max-concurrency 8" in seen["argv"]
    assert "--seed 12345" in seen["argv"]
    assert "--dataset-name random" in seen["argv"]
    assert "--base-url http://node7:8000" in seen["argv"]
    assert "node7" in seen["no_proxy"]                      # never through the corporate proxy
    assert seen["key"] == "tok123"                          # env, not argv
    assert rec["status"] == "ok" and rec["output_throughput"] == pytest.approx(341.9)
    assert rec["max_concurrency"] == 8
    assert rec["p95_e2el_ms"] == 0                          # absent key -> 0, not KeyError


def test_vllm_bench_absent_reports_fetch_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    ok, why = bb.VllmBenchBinary().available()
    assert not ok and "--fetch-backend" in why


# ---------- ladder ----------

def test_auto_ladder_prefers_binary_then_falls_to_synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    backend, why = bb.pick_backend("auto")
    assert backend.name == "synthetic" and "--fetch-backend" in why
    _shim(tmp_path, monkeypatch, "vllm-bench", "exit 0")
    backend, why = bb.pick_backend("auto")
    assert backend.name == "vllm-bench"


def test_explicit_backend_is_a_hard_requirement(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    with pytest.raises(RuntimeError, match="unavailable"):
        bb.pick_backend("vllm-bench")
    with pytest.raises(RuntimeError, match="unknown bench backend"):
        bb.pick_backend("bogus")


def test_fleet_pins_synthetic(tmp_path, monkeypatch):
    _shim(tmp_path, monkeypatch, "vllm-bench", "exit 0")
    backend, why = bb.pick_backend("auto", fleet=True)
    assert backend.name == "synthetic" and "fleet" in why


# ---------- container backend ----------

def test_container_render_has_hostnet_noproxy_and_probe_chain(tmp_path, monkeypatch):
    _shim(tmp_path, monkeypatch, "podman", "exit 0")
    backend = bb.VllmContainer(image="docker.io/rocm/vllm:x", runtime="")
    ok, why = backend.available()
    assert ok and "rocm/vllm" in why
    spec = bb.BenchSpec(url="http://cbnode1001:8000", model="m/x", concurrency=4,
                        num_prompts=32, max_tokens=16)
    cmd = backend.render_command(spec)
    joined = " ".join(cmd)
    assert "--network=host" in joined
    assert "no_proxy=cbnode1001" in joined
    assert "vllm bench serve" in joined and "benchmark_serving.py" in joined  # probe chain
    assert cmd[0] == "podman"


def test_container_unavailable_without_image():
    ok, why = bb.VllmContainer(image="").available()
    assert not ok and "--image" in why


# ---------- run_series ----------

def test_run_series_continues_past_errors(tmp_path, monkeypatch):
    class Flaky(bb.BenchBackend):
        name = "flaky"

        def run_level(self, spec):
            if spec.concurrency == 2:
                return {"max_concurrency": 2, "status": "error", "error": "boom"}
            return {"max_concurrency": spec.concurrency, "status": "ok",
                    "output_throughput": 1.0, "completed": spec.num_prompts,
                    "num_prompts": spec.num_prompts}

    base = bb.BenchSpec(url="http://x:1", model="m", concurrency=0, num_prompts=0, max_tokens=4)
    lines = []
    recs = bb.run_series(Flaky(), base, [1, 2, 4], progress=lines.append)
    assert [r["status"] for r in recs] == ["ok", "error", "ok"]
    assert any("FAILED" in ln and "continuing" in ln for ln in lines)
    assert recs[0]["num_prompts"] == 32                     # auto clamp(10*B, 32, 1000)
    assert recs[2]["num_prompts"] == 40


# ---------- datasets ----------

def test_resolve_dataset_defaults():
    kind, path, _ = bb.resolve_dataset(None, "synthetic")
    assert kind == "synthetic" and path is None
    kind, path, why = bb.resolve_dataset(None, "vllm-bench")
    assert kind == "random" and "sharegpt" in why


def test_resolve_dataset_file_and_missing(tmp_path):
    f = tmp_path / "prompts.json"
    f.write_text("[]")
    kind, path, _ = bb.resolve_dataset(str(f), "vllm-bench")
    assert kind == "file" and path == str(f)
    with pytest.raises(RuntimeError, match="no such file"):
        bb.resolve_dataset("nope.json", "vllm-bench")


def test_ensure_sharegpt_downloads_once(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_DATASETS_DIR", str(tmp_path / "ds"))
    calls = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            if calls and calls[-1] == "drained":
                return b""
            calls.append("drained")
            return b'[{"conversations": []}]'

    class FakeOpener:
        def open(self, url, timeout=0):
            calls.append(url)
            return FakeResp()

    monkeypatch.setattr("boxy.cardgen._opener", lambda: FakeOpener())
    p1 = bb.ensure_sharegpt()
    assert p1.exists() and p1.read_bytes().startswith(b"[")
    n_urls = len([c for c in calls if str(c).startswith("http")])
    bb.ensure_sharegpt()                                     # cache hit: no second fetch
    assert len([c for c in calls if str(c).startswith("http")]) == n_urls


def test_fetch_vllm_bench_places_executable(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_STORE", str(tmp_path / "store"))

    class FakeResp:
        done = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            if self.done:
                return b""
            self.done = True
            return b"#!/bin/sh\necho fake\n"

    class FakeOpener:
        def open(self, url, timeout=0):
            assert "{arch}" not in url                       # templated before fetch
            return FakeResp()

    monkeypatch.setattr("boxy.cardgen._opener", lambda: FakeOpener())
    dest = bb.fetch_vllm_bench()
    assert dest.exists() and os.access(dest, os.X_OK)
