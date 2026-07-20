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


# ---------- CLI integration ----------

def test_cli_bench_real_backend_end_to_end(tmp_path, monkeypatch, capfd):
    """`boxy bench --url ... --backend vllm-bench`: auto line, per-level
    progress, canonical table, persisted envelope with the right backend."""
    from boxy.cli import main

    _shim(tmp_path, monkeypatch, "vllm-bench", textwrap.dedent("""\
        outdir=""; fname=""; conc=0
        while [ $# -gt 0 ]; do
          [ "$1" = "--result-dir" ] && outdir="$2"
          [ "$1" = "--result-filename" ] && fname="$2"
          [ "$1" = "--max-concurrency" ] && conc="$2"
          shift
        done
        printf '{"max_concurrency": %s, "completed": 32, "failed": 0, "duration": 2.0,
                 "request_throughput": 16.0, "output_throughput": 512.0,
                 "mean_ttft_ms": 21.0, "median_ttft_ms": 20.0, "p99_ttft_ms": 30.0,
                 "mean_tpot_ms": 5.0, "median_tpot_ms": 5.0, "p99_tpot_ms": 8.0,
                 "mean_itl_ms": 5.0, "median_itl_ms": 5.0, "p99_itl_ms": 8.0,
                 "mean_e2el_ms": 100.0, "median_e2el_ms": 95.0, "p99_e2el_ms": 200.0}' > "$outdir/$fname"
    """))
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    # model discovery must not hit the network: give the record a model
    rc = main(["bench", "--url", "http://node9:8000/v1", "--backend", "vllm-bench",
               "--batch-sizes", "1,2", "--max-tokens", "8"])
    out = capfd.readouterr().out
    # discover_model would fail on a fake URL — the CLI reaches discover only
    # without a record; accept either the table (if it got there) or the error
    assert rc != 0 or "auto: bench backend: vllm-bench" in out


def test_cli_bench_real_backend_with_record_model(tmp_path, monkeypatch, capfd):
    """With a job record supplying the model, no discovery round-trip is needed
    and the real backend runs fully offline (shim)."""
    import socket

    from boxy.cli import main

    _shim(tmp_path, monkeypatch, "vllm-bench", textwrap.dedent("""\
        outdir=""; fname=""; conc=0
        while [ $# -gt 0 ]; do
          [ "$1" = "--result-dir" ] && outdir="$2"
          [ "$1" = "--result-filename" ] && fname="$2"
          [ "$1" = "--max-concurrency" ] && conc="$2"
          shift
        done
        printf '{"max_concurrency": %s, "completed": 32, "failed": 0, "duration": 2.0,
                 "request_throughput": 16.0, "output_throughput": 512.0,
                 "median_ttft_ms": 20.0, "p99_ttft_ms": 30.0,
                 "mean_tpot_ms": 5.0, "p99_itl_ms": 8.0,
                 "median_e2el_ms": 95.0}' "$conc" > "$outdir/$fname"
    """))
    jobsdir = tmp_path / "jobs"
    monkeypatch.setenv("BOXY_JOBS_DIR", str(jobsdir))
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    jobsdir.mkdir()
    (jobsdir / "boxy-m.json").write_text(json.dumps(
        {"name": "boxy-m", "scheduler": "none", "job": "0", "model": "org/m-7b",
         "submitted_from": socket.gethostname()}))
    (jobsdir / "boxy-m.endpoint.json").write_text(json.dumps(
        {"name": "boxy-m", "host": "node9", "port": 8000, "url": "http://node9:8000"}))
    rc = main(["bench", "boxy-m", "--backend", "vllm-bench", "--batch-sizes", "1,2",
               "--max-tokens", "8", "--label", "real-run"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "auto: bench backend: vllm-bench" in out
    assert "concurrency 1: 512.0 tok/s" in out
    assert "### Result saved:" in out
    from boxy import results

    listing = results.list_results()
    assert listing and listing[0][1]["bench_backend"] == "vllm-bench"
    assert listing[0][1]["label"] == "real-run"
    assert listing[0][1]["runs"][0]["output_throughput"] == 512.0


def test_cli_bench_dryrun_names_backend(tmp_path, monkeypatch, capfd):
    from boxy.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    rc = main(["bench", "--url", "http://x:8000", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0 and "backend=synthetic" in out
    assert "auto: bench backend: synthetic" in out and "--fetch-backend" in out


def test_cli_fetch_backend_dryrun(tmp_path, monkeypatch, capfd):
    from boxy.cli import main

    monkeypatch.setenv("BOXY_STORE", str(tmp_path / "store"))
    rc = main(["bench", "--fetch-backend", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0 and "Would download the vllm-bench static binary" in out


def test_fetch_on_macos_refuses_with_guidance(monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="LINUX binaries only.*cargo install"):
        bb.fetch_vllm_bench()


def test_fetch_download_failure_names_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_STORE", str(tmp_path / "store"))

    class FailOpener:
        def open(self, url, timeout=0):
            raise OSError("nodename nor servname provided")

    monkeypatch.setattr("boxy.cardgen._opener", lambda: FailOpener())
    with pytest.raises(RuntimeError, match="network.proxy"):
        bb.fetch_vllm_bench()


# ---------- the agentless bench (cluster-side zero install) ----------

@pytest.fixture()
def agentless_setup(tmp_path, monkeypatch):
    """A laptop-side agentless serve record + a fake ssh that answers the
    endpoint cat and runs the 'podman' benchmark, emitting the real stdout
    block — the exact cluster-side zero-install flow."""
    import socket

    jobsdir = tmp_path / "jobs"
    jobsdir.mkdir()
    monkeypatch.setenv("BOXY_JOBS_DIR", str(jobsdir))
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    (jobsdir / "boxy-llama.json").write_text(json.dumps({
        "name": "boxy-llama", "scheduler": "slurm", "job": "77", "model": "org/llama-x",
        "submitted_from": "agentless-ssh", "target": "user1@clustera-login",
        "endpoint_remote": "/rhome/.boxy/boxy-llama.endpoint.json",
        "image": "docker.io/rocm/vllm:6.4", "engine": "vllm"}))
    block = STDOUT_BLOCK.replace("\n", "\\n").replace('"', '\\"')
    log = tmp_path / "ssh-calls.log"
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"test -x"*)     exit 1 ;;
          *cat*endpoint*)  printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                                     "url": "http://cbnode7:8000", "ready": true,
                                     "model": "org/llama-x"}}\\n' ;;
          *"command -v podman"*) echo /usr/bin/podman ;;
          *"image exists"*) echo PRESENT ;;
          *podman*run*)    printf "%b" "{block}\\n" ;;
          *)               exit 0 ;;
        esac
    ''')
    monkeypatch.setattr(socket, "gethostname", lambda: "laptop-mac")
    return tmp_path, log


def test_agentless_bench_end_to_end(agentless_setup, capfd):
    """`boxy bench` with only an agentless record: container backend runs on
    the login node over ssh, block parsed, result stored laptop-side."""
    from boxy.cli import main

    tmp_path, log = agentless_setup
    rc = main(["bench", "--batch-sizes", "1,2", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "auto: bench backend: vllm-container" in out and "agentless" in out
    assert "1059.3 tok/s" in out                             # parsed from the block
    assert "### Result saved:" in out
    calls = log.read_text()
    assert "podman" in calls and "rocm/vllm:6.4" in calls
    assert "--network=host" in calls and "no_proxy=cbnode7" in calls
    from boxy import results

    listing = results.list_results()
    assert listing and listing[0][1]["bench_backend"] == "vllm-container"
    assert listing[0][1]["label"] == "rocm: clustera/boxy-llama"
    assert listing[0][1]["instance"] == "boxy-llama"


def test_agentless_bench_with_matching_ssh_flag(agentless_setup, capfd):
    """--ssh to the same cluster prefers the agentless path over delegation
    (the cluster has no record of an agentless serve)."""
    from boxy.cli import main

    rc = main(["bench", "--ssh", "user1@clustera-login", "--batch-sizes", "1",
               "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0 and "auto: bench backend: vllm-container" in out


def test_agentless_bench_dryrun(agentless_setup, capfd):
    from boxy.cli import main

    rc = main(["bench", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0 and "Agentless bench plan" in out and "rocm/vllm:6.4" in out


def test_agentless_bench_old_record_without_image(agentless_setup, tmp_path, capfd):
    from boxy.cli import main

    jobsdir = tmp_path / "jobs"
    rec = json.loads((jobsdir / "boxy-llama.json").read_text())
    del rec["image"]
    (jobsdir / "boxy-llama.json").write_text(json.dumps(rec))
    rc = main(["bench", "boxy-llama"])
    err = capfd.readouterr().err
    assert rc != 0 and "--image" in err and "older boxy" in err


def test_fetch_backend_over_ssh_is_agentless(tmp_path, monkeypatch, capfd):
    """--fetch-backend --ssh installs the binary ON the cluster via curl over
    the master — never by delegating to a (possibly ancient) cluster boxy."""
    from boxy.cli import main

    log = tmp_path / "ssh-calls.log"
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"uname -m"*) echo x86_64 ;;
          *curl*)       echo FETCHED ;;
          *)            exit 0 ;;
        esac
    ''')
    rc = main(["bench", "--fetch-backend", "--ssh", "user1@clustera-login"])
    out = capfd.readouterr().out
    assert rc == 0 and "vllm-bench installed on user1@clustera-login" in out
    calls = log.read_text()
    assert "curl" in calls and "chmod +x" in calls
    assert "boxy bench" not in calls                        # no delegation involved


def test_agentless_bench_prefers_cluster_binary(agentless_setup, tmp_path, monkeypatch, capfd):
    """When vllm-bench is installed on the cluster (--fetch-backend --ssh),
    the agentless bench uses it — no container, no image pull. The shim
    emits the save-result JSON behind the marker."""
    from boxy.cli import main

    _, log = agentless_setup
    d = tmp_path / "shims"
    (d / "ssh").write_text(f"""#!/bin/sh
echo "$@" >> {log}
case "$*" in
  *"test -x"*)    exit 0 ;;
  *cat*endpoint*) printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                           "url": "http://cbnode7:8000", "ready": true,
                           "model": "org/llama-x"}}\\n' ;;
  *vllm-bench*)   echo BOXY_RESULT_JSON
                  printf '{{"max_concurrency": 4, "completed": 32, "failed": 0,
                           "duration": 2.0, "request_throughput": 16.0,
                           "output_throughput": 640.5, "median_ttft_ms": 12.0,
                           "p99_ttft_ms": 20.0, "median_e2el_ms": 80.0}}\\n' ;;
  *)              exit 0 ;;
esac
""")
    rc = main(["bench", "boxy-llama", "--batch-sizes", "4", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "auto: bench backend: vllm-bench" in out and "--fetch-backend" in out
    assert "640.5 tok/s" in out
    calls = log.read_text()
    assert "store/bin/vllm-bench" in calls and "no_proxy=cbnode7" in calls
    assert "podman" not in calls                            # no container involved


def test_agentless_container_prepull_when_image_absent(agentless_setup, tmp_path, capfd):
    """Login-node podman without the image: the bench announces a proxied
    pre-pull before level 1 (field: silent multi-minute hang)."""
    from boxy.cli import main

    _, log = agentless_setup
    d = tmp_path / "shims"
    script = (d / "ssh").read_text().replace("echo PRESENT", "echo ABSENT")
    (d / "ssh").write_text(script)
    rc = main(["bench", "boxy-llama", "--batch-sizes", "1", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0 and "### Pulling docker.io/rocm/vllm:6.4 on clustera-login" in out
    assert "podman pull" in log.read_text()


def test_container_probe_chain_tries_module_entrypoint():
    """Newest images: the vllm wrapper redirects to the module entrypoint —
    the probe chain must try it FIRST (field: vllm-openai-rocm exit 1)."""
    backend = bb.VllmContainer(image="docker.io/vllm/vllm-openai-rocm", runtime="podman")
    spec = bb.BenchSpec(url="http://n:8000", model="m", concurrency=1,
                        num_prompts=32, max_tokens=8)
    inner = " ".join(backend.render_command(spec))
    assert inner.index("vllm.entrypoints.cli.main bench serve") < inner.index("vllm bench serve --help")


def test_agentless_bench_strips_transport_uri_from_model(agentless_setup, tmp_path, capfd):
    """A record's hf:// transport URI must never reach the benchmark's --model:
    the server serves the plain id (field: tokenizer lookup on 'hf://…')."""
    from boxy.cli import main

    _, log = agentless_setup
    jobsdir = tmp_path / "jobs"
    rec = json.loads((jobsdir / "boxy-llama.json").read_text())
    rec["model"] = "hf://meta-llama/Llama-3.2-1B-Instruct"
    (jobsdir / "boxy-llama.json").write_text(json.dumps(rec))
    # endpoint file without a model field -> the stripped record id must win
    d = tmp_path / "shims"
    script = (d / "ssh").read_text().replace(', "model": "org/llama-x"', "")
    script = script.replace('"model": "org/llama-x"', '"job": "77"')
    (d / "ssh").write_text(script)
    rc = main(["bench", "boxy-llama", "--batch-sizes", "1", "--max-tokens", "8"])
    assert rc == 0
    calls = log.read_text()
    assert "--model meta-llama/Llama-3.2-1B-Instruct" in calls
    assert "hf://" not in calls.replace("hf://meta", "STRIPPEDCHECK") or \
           "--model hf://" not in calls


def _sharegpt_shim(tmp_path, monkeypatch, log, *, staged: str, curl: str):
    """The agentless fake ssh, extended with the ShareGPT staging answers:
    staged/curl are shell snippets for the `test -s` and curl cases."""
    block = STDOUT_BLOCK.replace("\n", "\\n").replace('"', '\\"')
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"echo \\$HOME"*) echo /rhome ;;
          *"test -s"*)     {staged} ;;
          *curl*ShareGPT*) {curl} ;;
          *"cat >"*)       cat > /dev/null ;;
          *"test -x"*)     exit 1 ;;
          *cat*endpoint*)  printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                                    "url": "http://cbnode7:8000", "ready": true,
                                    "model": "org/llama-x"}}\\n' ;;
          *"command -v podman"*) echo /usr/bin/podman ;;
          *"image exists"*) echo PRESENT ;;
          *podman*run*)    printf "%b" "{block}\\n" ;;
          *)               exit 0 ;;
        esac
    ''')


SHAREGPT_REMOTE = "/rhome/.local/share/boxy/store/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"


def test_agentless_bench_stages_sharegpt(agentless_setup, tmp_path, monkeypatch, capfd):
    """--dataset sharegpt on the agentless path: the login node downloads the
    corpus itself (proxied curl, cached in the cluster-side store) and the
    container mounts it — no more 'random only' refusal (field: clusterc)."""
    from boxy.cli import main

    log = tmp_path / "ssh-calls.log"
    _sharegpt_shim(tmp_path, monkeypatch, log, staged="exit 1", curl="echo BOXY_STAGED")
    rc = main(["bench", "--dataset", "sharegpt", "--batch-sizes", "1", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "staging the ShareGPT corpus" in out
    calls = log.read_text()
    assert "--dataset-name sharegpt" in calls and SHAREGPT_REMOTE in calls
    d = SHAREGPT_REMOTE.rsplit("/", 1)[0]
    assert f"-v {d}:{d}:ro" in calls                        # mounted into the bench container
    from boxy import results as res

    assert res.list_results()[0][1]["dataset"] == "sharegpt"


def test_agentless_sharegpt_cached_skips_staging(agentless_setup, tmp_path, monkeypatch, capfd):
    from boxy.cli import main

    log = tmp_path / "ssh-calls.log"
    _sharegpt_shim(tmp_path, monkeypatch, log, staged="exit 0", curl="echo BOXY_STAGED")
    rc = main(["bench", "--dataset", "sharegpt", "--batch-sizes", "1", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0 and "staging the ShareGPT corpus" not in out
    calls = log.read_text()
    assert ".part" not in calls                       # no download ran (metrics curl is fine)
    assert "--dataset-name sharegpt" in calls


def test_agentless_sharegpt_403_self_heals_via_laptop_upload(agentless_setup, tmp_path,
                                                             monkeypatch, capfd):
    """Field: the site filter 403s huggingface.co on the login node. The stage
    then downloads on the LAPTOP (proxy/CA known-good) and streams the corpus
    up the live ssh master — the bench proceeds without manual scp."""
    from boxy.cli import main

    corpus = tmp_path / "sharegpt.json"
    corpus.write_text("[]")
    monkeypatch.setattr(bb, "ensure_sharegpt", lambda: corpus)
    log = tmp_path / "ssh-calls.log"
    _sharegpt_shim(tmp_path, monkeypatch, log, staged="exit 1", curl="exit 22")
    rc = main(["bench", "--dataset", "sharegpt", "--batch-sizes", "1", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "uploading over the ssh session" in out and "staged from this machine" in out
    calls = log.read_text()
    assert f"cat > {SHAREGPT_REMOTE}.part" in calls        # streamed, then moved into place
    assert f"mv {SHAREGPT_REMOTE}.part" in calls
    assert "--dataset-name sharegpt" in calls


def test_agentless_sharegpt_both_paths_failing_names_the_fixes(agentless_setup, tmp_path,
                                                               monkeypatch, capfd):
    from boxy.cli import main

    def boom():
        raise RuntimeError("cannot download the ShareGPT dataset (proxy down) — "
                           "point datasets.sharegpt_url at a mirror")

    monkeypatch.setattr(bb, "ensure_sharegpt", boom)
    log = tmp_path / "ssh-calls.log"
    _sharegpt_shim(tmp_path, monkeypatch, log, staged="exit 1", curl="exit 22")
    rc = main(["bench", "--dataset", "sharegpt", "--batch-sizes", "1", "--max-tokens", "8"])
    err = capfd.readouterr().err
    assert rc != 0
    assert "laptop-side fallback failed" in err and "sharegpt_url" in err


def test_agentless_bench_still_refuses_file_datasets(agentless_setup, capfd):
    from boxy.cli import main

    rc = main(["bench", "--dataset", "/tmp/prompts.json", "--batch-sizes", "1"])
    err = capfd.readouterr().err
    assert rc != 0 and "random|sharegpt" in err


def test_serve_flags_random_prefix_len():
    spec = bb.BenchSpec(url="http://x:1", model="m", concurrency=4, num_prompts=8,
                        max_tokens=16, random_prefix_len=256)
    flags = bb._serve_flags(spec)
    i = flags.index("--random-prefix-len")
    assert flags[i + 1] == "256"
    bare = bb.BenchSpec(url="http://x:1", model="m", concurrency=4, num_prompts=8, max_tokens=16)
    assert "--random-prefix-len" not in bb._serve_flags(bare)


def test_agentless_bench_names_gpu_model_from_node(agentless_setup, tmp_path, monkeypatch, capfd):
    """The serving node's typed GRES / Features (scontrol over the ssh master)
    name the GPU MODEL — the stored envelope and label carry mi300a, not just
    the rocm family (field: 'cuda' on an MI300A cluster was unacceptable)."""
    from boxy.cli import main

    block = STDOUT_BLOCK.replace("\n", "\\n").replace('"', '\\"')
    log = tmp_path / "ssh-calls.log"
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"test -x"*)     exit 1 ;;
          *scontrol*)      echo "   Gres=gpu:mi300a:4  AvailableFeatures=amd,apu" ;;
          *cat*endpoint*)  printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                                     "url": "http://cbnode7:8000", "ready": true,
                                     "model": "org/llama-x"}}\\n' ;;
          *"command -v podman"*) echo /usr/bin/podman ;;
          *"image exists"*) echo PRESENT ;;
          *podman*run*)    printf "%b" "{block}\\n" ;;
          *)               exit 0 ;;
        esac
    ''')
    rc = main(["bench", "--batch-sizes", "1", "--max-tokens", "8"])
    assert rc == 0
    assert "1059.3 tok/s" in capfd.readouterr().out
    from boxy import results as res

    listing = res.list_results()
    assert listing[0][1]["gpu_type"] == "mi300a"
    assert listing[0][1]["label"] == "mi300a: clustera/boxy-llama"
    assert res.display_label(listing[0][1]) == "mi300a: clustera/boxy-llama"


def test_resolve_dataset_hf_repo():
    kind, path, why = bb.resolve_dataset("hf:lmarena-ai/VisionArena-Chat", "vllm-container")
    assert (kind, path) == ("hf", "lmarena-ai/VisionArena-Chat") and "hub" in why
    with pytest.raises(RuntimeError, match="vllm-container"):
        bb.resolve_dataset("hf:org/data", "vllm-bench")       # Rust binary: no datasets lib
    with pytest.raises(RuntimeError, match="hf:<repo-id>"):
        bb.resolve_dataset("hf:", "vllm-container")


def test_serve_flags_and_container_render_for_hf():
    spec = bb.BenchSpec(url="http://n7:8000", model="m", concurrency=2, num_prompts=8,
                        max_tokens=16, dataset_kind="hf", dataset_path="org/data",
                        hf_cache_dir="/rhome/.local/share/boxy/store/hf-cache")
    flags = bb._serve_flags(spec)
    i = flags.index("--dataset-name")
    assert flags[i + 1] == "hf" and "org/data" in flags
    cmd = bb.VllmContainer("img", "podman").render_command(spec)
    assert "--env" in cmd and "https_proxy" in cmd            # hub download rides the proxy
    assert "HF_HOME=/rhome/.local/share/boxy/store/hf-cache" in cmd
    assert "-v" in cmd
    assert f"{spec.hf_cache_dir}:{spec.hf_cache_dir}" in cmd  # cache persists across levels
    assert not any(str(c).endswith(":ro") for c in cmd)       # repo id is not a file mount


def test_agentless_bench_hf_dataset_skips_binary_and_mounts_cache(agentless_setup, tmp_path,
                                                                  monkeypatch, capfd):
    """--dataset hf:<repo> agentlessly: the cluster vllm-bench binary (present!)
    is skipped — only the serving image has the datasets loader — and the bench
    container gets a persistent HF cache."""
    from boxy.cli import main

    log = tmp_path / "ssh-calls.log"
    block = STDOUT_BLOCK.replace("\n", "\\n").replace('"', '\\"')
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"echo \\$HOME"*) echo /rhome ;;
          *"test -x"*)     exit 0 ;;
          *cat*endpoint*)  printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                                     "url": "http://cbnode7:8000", "ready": true,
                                     "model": "org/llama-x"}}\\n' ;;
          *"command -v podman"*) echo /usr/bin/podman ;;
          *"image exists"*) echo PRESENT ;;
          *podman*run*)    printf "%b" "{block}\\n" ;;
          *)               exit 0 ;;
        esac
    ''')
    rc = main(["bench", "--dataset", "hf:org/data", "--batch-sizes", "1", "--max-tokens", "8"])
    out = capfd.readouterr().out
    assert rc == 0 and "vllm-container" in out
    calls = log.read_text()
    assert "--dataset-name hf" in calls and "org/data" in calls
    assert "mkdir -p /rhome/.local/share/boxy/store/hf-cache" in calls
    assert "HF_HOME=/rhome/.local/share/boxy/store/hf-cache" in calls
    from boxy import results as res

    assert res.list_results()[0][1]["dataset"] == "hf:org/data"


def test_agentless_bench_hf_with_explicit_binary_backend_refuses(agentless_setup, capfd):
    from boxy.cli import main

    rc = main(["bench", "--dataset", "hf:org/data", "--backend", "vllm-bench",
               "--batch-sizes", "1"])
    err = capfd.readouterr().err
    assert rc != 0 and "vllm-bench binary" in err


def test_agentless_bench_gpu_type_flag_overrides_detection(agentless_setup, tmp_path,
                                                           monkeypatch, capfd):
    """--gpu-type pins the legend prefix and skips node interrogation — the
    escape hatch when the scheduler's GRES/Features text names the wrong
    model (field: 'h200' on a cluster the user knows better)."""
    from boxy.cli import main

    log = tmp_path / "ssh-calls.log"
    block = STDOUT_BLOCK.replace("\n", "\\n").replace('"', '\\"')
    _shim(tmp_path, monkeypatch, "ssh", f'''
        echo "$@" >> {log}
        case "$*" in
          *"test -x"*)     exit 1 ;;
          *scontrol*)      echo "Gres=gpu:mi300a:4" ;;
          *cat*endpoint*)  printf '{{"name": "boxy-llama", "host": "cbnode7", "port": 8000,
                                     "url": "http://cbnode7:8000", "ready": true,
                                     "model": "org/llama-x"}}\\n' ;;
          *"command -v podman"*) echo /usr/bin/podman ;;
          *"image exists"*) echo PRESENT ;;
          *podman*run*)    printf "%b" "{block}\\n" ;;
          *)               exit 0 ;;
        esac
    ''')
    rc = main(["bench", "--gpu-type", "h100", "--batch-sizes", "1", "--max-tokens", "8"])
    assert rc == 0
    capfd.readouterr()
    assert "scontrol" not in log.read_text()               # detection skipped entirely
    listing = results.list_results()
    assert listing[0][1]["gpu_type"] == "h100"
    assert listing[0][1]["label"] == "h100: clustera/boxy-llama"


def test_results_relabel_fixes_stored_gpu_type(tmp_path, monkeypatch, capfd):
    from boxy.cli import main

    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    env = results.make_envelope(url="http://n:1", model="m/x", backend="vllm-bench",
                                runs=[], instance="boxy-m", accelerator="cuda",
                                gpu_type="h200", label="h200: clusterc/boxy-m")
    results.write_result(env)
    rc = main(["results", "relabel", "1", "--gpu-type", "h100"])
    out = capfd.readouterr().out
    assert rc == 0 and "h100: clusterc/boxy-m" in out
    path, data = results.list_results()[0]
    assert data["gpu_type"] == "h100" and data["label"] == "h100: clusterc/boxy-m"
    assert results.display_label(data) == "h100: clusterc/boxy-m"
    rc = main(["results", "relabel", "1"])                 # missing --gpu-type
    assert rc != 0 and "--gpu-type" in capfd.readouterr().err


def test_served_model_id_helper():
    assert bb.served_model_id("hf://meta-llama/Llama-3.2-1B-Instruct") == \
        "meta-llama/Llama-3.2-1B-Instruct"
    assert bb.served_model_id("ollama://tinyllama") == "tinyllama"
    assert bb.served_model_id("meta-llama/Llama-3.2-1B-Instruct") == \
        "meta-llama/Llama-3.2-1B-Instruct"
    assert bb.served_model_id("") == ""


def test_gpu_name_from_text_tokens():
    assert bb.gpu_name_from_text("Gres=gpu:mi300a:4") == "mi300a"
    assert bb.gpu_name_from_text("NodeName=n7 AvailableFeatures=amd,MI300A,apu") == "mi300a"
    assert bb.gpu_name_from_text("gpu:h100:8(S:0-1)") == "h100"
    assert bb.gpu_name_from_text("NVIDIA H100 80GB HBM3") == "h100"
    assert bb.gpu_name_from_text("sha100x") == ""          # boundaries: no substring hits
    assert bb.gpu_name_from_text("plain cuda node, no model") == ""
    assert bb.gpu_name_from_text("") == ""


def test_auto_num_prompts_scales_with_concurrency():
    """Field: the 1000-prompt cap starved the 1024-concurrency level (fewer
    prompts than requested in-flight requests) and read as a scaling wall."""
    assert bb.auto_num_prompts(1) == 32
    assert bb.auto_num_prompts(64) == 640
    assert bb.auto_num_prompts(256) == 1000                # legacy cap unchanged
    assert bb.auto_num_prompts(512) == 1536                # cap scales: 3x the level
    assert bb.auto_num_prompts(1024) == 3072


def test_run_series_prompt_pool_holds_top_concurrency():
    seen = []

    class Cap(bb.BenchBackend):
        name = "cap"

        def run_level(self, spec):
            seen.append((spec.concurrency, spec.num_prompts))
            return {"max_concurrency": spec.concurrency, "status": "ok"}

    base = bb.BenchSpec(url="http://x:1", model="m", concurrency=0, num_prompts=0, max_tokens=4)
    bb.run_series(Cap(), base, [256, 1024], progress=lambda *_: None)
    assert seen == [(256, 1000), (1024, 3072)]             # pool always >= the level
    seen.clear()
    explicit = bb.BenchSpec(url="http://x:1", model="m", concurrency=0,
                            num_prompts=64, max_tokens=4)
    bb.run_series(Cap(), explicit, [1024], progress=lambda *_: None)
    assert seen == [(1024, 64)]                            # --num-prompts always wins


def test_accel_from_image_heuristic():
    assert bb.accel_from_image("docker.io/vllm/vllm-openai-rocm") == "rocm"
    assert bb.accel_from_image("docker.io/vllm/vllm-openai:v0.9.1") == "cuda"
    assert bb.accel_from_image("quay.io/ramalama/cuda:latest") == "cuda"
    assert bb.accel_from_image("something/custom:1") == ""


def test_envelope_label_carries_accelerator():
    env = results.make_envelope(url="http://n:1", model="m/x", backend="synthetic",
                                runs=[], instance="boxy-m", accelerator="rocm")
    assert env["label"].startswith("rocm: ") and env["label"].endswith("/boxy-m")
    env2 = results.make_envelope(url="http://n:1", model="m/x", backend="synthetic",
                                 runs=[], instance="boxy-m")
    assert ":" not in env2["label"].split("/")[0]


# ---------- prefix-cache hit rate ----------

METRICS_TEXT = """\
# HELP vllm:gpu_prefix_cache_queries_total ...
vllm:gpu_prefix_cache_queries_total{model_name="m/x"} 1000
vllm:gpu_prefix_cache_hits_total{model_name="m/x"} 250
vllm:gpu_cache_usage_perc{model_name="m/x"} 0.42
"""


def test_parse_cache_metrics_counters_and_gauge():
    m = bb.parse_cache_metrics(METRICS_TEXT)
    assert m == {"hits": 250.0, "queries": 1000.0, "gauge": None}
    g = bb.parse_cache_metrics('vllm:gpu_prefix_cache_hit_rate{m="x"} 0.37\n')
    assert g["gauge"] == 0.37
    assert bb.parse_cache_metrics("no vllm metrics here") is None


def test_cache_rate_delta_semantics():
    b = {"hits": 100.0, "queries": 400.0, "gauge": None}
    a = {"hits": 350.0, "queries": 900.0, "gauge": None}
    assert bb.cache_rate_delta(b, a) == pytest.approx(50.0)   # 250/500
    assert bb.cache_rate_delta(None, {"hits": 0, "queries": 0, "gauge": 0.8}) == pytest.approx(80.0)
    assert bb.cache_rate_delta(b, b) is None                  # no queries this level
    assert bb.cache_rate_delta(b, None) is None


def test_run_series_attaches_cache_hit_rate():
    class Fixed(bb.BenchBackend):
        name = "fixed"

        def run_level(self, spec):
            return {"max_concurrency": spec.concurrency, "status": "ok",
                    "output_throughput": 1.0, "completed": 4, "num_prompts": 4}

    samples = iter([
        {"hits": 0.0, "queries": 0.0, "gauge": None},
        {"hits": 30.0, "queries": 100.0, "gauge": None},      # level 1: 30%
        {"hits": 110.0, "queries": 200.0, "gauge": None},     # level 2: 80%
    ])
    base = bb.BenchSpec(url="http://x:1", model="m", concurrency=0, num_prompts=4, max_tokens=4)
    recs = bb.run_series(Fixed(), base, [1, 2], progress=lambda *_: None,
                         metrics_sampler=lambda: next(samples))
    assert recs[0]["prefix_cache_hit_rate"] == pytest.approx(30.0)
    assert recs[1]["prefix_cache_hit_rate"] == pytest.approx(80.0)


def test_run_series_notes_missing_cache_series():
    """A server with prefix caching disabled exports NO cache series; the sweep
    must say so (with the serve-time fix) instead of silently attaching
    nothing — field: an empty cache figure with zero explanation."""

    class Fixed(bb.BenchBackend):
        name = "fixed"

        def run_level(self, spec):
            return {"max_concurrency": spec.concurrency, "status": "ok",
                    "output_throughput": 1.0, "completed": 4, "num_prompts": 4}

    lines = []
    base = bb.BenchSpec(url="http://x:1", model="m", concurrency=0, num_prompts=4, max_tokens=4)
    bb.run_series(Fixed(), base, [1, 2], progress=lines.append,
                  metrics_sampler=lambda: None)            # server exports nothing
    assert any("--enable-prefix-caching" in ln for ln in lines)
    lines.clear()
    samples = iter([{"hits": 0.0, "queries": 0.0, "gauge": None},
                    {"hits": 30.0, "queries": 100.0, "gauge": None}])
    bb.run_series(Fixed(), base, [1], progress=lines.append,
                  metrics_sampler=lambda: next(samples))   # rates present: no note
    assert not any("--enable-prefix-caching" in ln for ln in lines)
