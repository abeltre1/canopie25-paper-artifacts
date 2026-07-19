"""`boxy bench` serving-metric instrumentation (the vLLM bench-serve metric
set: TTFT / ITL / TPOT / E2E percentiles / throughput), measured for real
against an in-process SSE-streaming OpenAI-compatible server with KNOWN delays
— so the numbers are validated, not just present."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from boxy import bench
from boxy.cli import main

TTFT_S = 0.02
ITL_S = 0.005
TOKENS = 8


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake/tiny-1b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        toks = min(int(req.get("max_tokens", TOKENS)), TOKENS)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(TTFT_S)
        for i in range(toks):
            self.wfile.write(f'data: {json.dumps({"choices": [{"text": f"t{i} "}]})}\n\n'.encode())
            self.wfile.flush()
            time.sleep(ITL_S)
        usage = {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": toks}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_streaming_metrics_match_known_delays(server):
    report = bench.run_bench(server, [2], max_tokens=TOKENS)
    r = report.results[0]
    assert r.ok == 2 and r.errors == 0
    assert r.completion_tokens == 2 * TOKENS               # from the usage chunk
    # TTFT ≈ the server's first-token delay; ITL/TPOT ≈ its per-token delay
    assert TTFT_S * 1000 * 0.8 <= r.ttft_p50_ms <= TTFT_S * 1000 * 4
    assert ITL_S * 1000 * 0.6 <= r.itl_p50_ms <= ITL_S * 1000 * 4
    assert ITL_S * 1000 * 0.6 <= r.tpot_mean_ms <= ITL_S * 1000 * 4
    assert r.latency_p50_ms > r.ttft_p50_ms                # E2E includes generation
    assert r.tokens_per_s > 0 and r.requests_per_s > 0


def test_bench_cli_resolves_newest_record(server, tmp_path, monkeypatch, capfd):
    # turnkey: bare `boxy bench` finds the newest live instance from the job
    # records — no --box, no --url (field: bench still demanded the v1 --box).
    import socket

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    (tmp_path / "boxy-fake-tiny.json").write_text(json.dumps(
        {"name": "boxy-fake-tiny", "scheduler": "none", "job": "0",
         "model": "fake/tiny-1b", "submitted_from": socket.gethostname()}))
    port = server.rsplit(":", 1)[1]
    (tmp_path / "boxy-fake-tiny.endpoint.json").write_text(json.dumps(
        {"name": "boxy-fake-tiny", "host": "127.0.0.1", "port": int(port), "url": server}))
    rc = main(["bench", "--batch-sizes", "1", "--max-tokens", "4",
               "-o", str(tmp_path / "bench.csv")])
    out = capfd.readouterr().out
    assert rc == 0
    assert "TTFT p50" in out and "fake/tiny-1b" in out
    csv = (tmp_path / "bench.csv").read_text()
    assert "ttft_p50_ms" in csv.splitlines()[0]            # plot-ready columns


def test_bench_csv_carries_serving_metric_columns(server):
    report = bench.run_bench(server, [1], max_tokens=4)
    header = report.to_csv().splitlines()[0]
    for col in ("ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms", "itl_p99_ms"):
        assert col in header
