"""Readiness: log-based ready detection (survives an unreachable endpoint) and
the proxy-bypass on the HTTP probe (field report: vLLM up on cronus5 but boxy
looped forever because the /v1/models probe was routed through the corporate
proxy, and no log fallback existed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from boxy import readiness

VLLM_LOG = """\
INFO 07-15 05:46:40 [api_server.py:1611] vLLM API server version 0.11.0
INFO 07-15 05:46:56 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
"""

LLAMACPP_LOG = "main: loading model ...\nmain: server is listening on http://0.0.0.0:8090 - starting\n"

LOADING_LOG = "INFO ... Loading safetensors checkpoint shards:  40% Completed\n"


def test_log_is_ready_detects_vllm(tmp_path):
    p = tmp_path / "job.log"
    p.write_text(VLLM_LOG)
    assert readiness.log_is_ready(p) is True


def test_log_is_ready_detects_llamacpp(tmp_path):
    p = tmp_path / "job.log"
    p.write_text(LLAMACPP_LOG)
    assert readiness.log_is_ready(p) is True


def test_log_is_ready_false_while_still_loading(tmp_path):
    p = tmp_path / "job.log"
    p.write_text(LOADING_LOG)
    assert readiness.log_is_ready(p) is False


def test_log_is_ready_handles_missing_file(tmp_path):
    assert readiness.log_is_ready(tmp_path / "nope.log") is False
    assert readiness.log_is_ready(None) is False


def test_log_is_ready_reads_tail_of_a_huge_log(tmp_path):
    # the ready marker near the end of a big log is still found (bounded tail read)
    p = tmp_path / "job.log"
    p.write_text(("x" * 4096 + "\n") * 500 + "(APIServer pid=1) INFO:     Application startup complete.\n")
    assert readiness.log_is_ready(p) is True


def test_model_from_log(tmp_path):
    p = tmp_path / "job.log"
    p.write_text("INFO served model name: meta-llama/Llama-3.1-8B-Instruct is ready\n")
    assert readiness.model_from_log(p) == "meta-llama/Llama-3.1-8B-Instruct"


def test_wait_ready_returns_via_log_when_endpoint_unreachable(tmp_path):
    # the endpoint is a dead port (compute node not routable / behind a proxy),
    # but the log proves the server is up -> readiness resolves from the log.
    p = tmp_path / "job.log"
    p.write_text(VLLM_LOG + "served model name: my-model\n")
    got = readiness.wait_ready("http://127.0.0.1:9", timeout_s=2, interval_s=0.2, log_path=p)
    assert got == "my-model"


def test_wait_ready_times_out_when_neither_probe_nor_log_ready(tmp_path):
    p = tmp_path / "job.log"
    p.write_text(LOADING_LOG)
    got = readiness.wait_ready("http://127.0.0.1:9", timeout_s=0.6, interval_s=0.2, log_path=p)
    assert got is None


def _make_handler(health_status=200, serve_models=True):
    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(health_status)
                self.end_headers()
            elif self.path == "/v1/models" and serve_models:
                body = json.dumps({"data": [{"id": "served-model"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *a):
            pass
    return _H


def _serve(handler_cls):
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def models_server():
    srv = _serve(_make_handler())
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


def test_health_ok_true_on_200(models_server):
    assert readiness.health_ok(models_server, timeout=3) is True


def test_probe_uses_health_and_names_the_model(models_server):
    # /health 200 -> ready, and the model id is still read from /v1/models
    assert readiness.probe_once(models_server, timeout=3) == "served-model"


def test_probe_ready_via_health_when_models_absent():
    # a server that answers /health but not /v1/models is still 'ready'
    srv = _serve(_make_handler(health_status=200, serve_models=False))
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert readiness.health_ok(url, timeout=3) is True
        assert readiness.probe_once(url, timeout=3) == "ready"
    finally:
        srv.shutdown()


def test_probe_not_ready_when_health_503_and_no_models():
    # server up but not ready (vLLM returns 503 on /health while loading)
    srv = _serve(_make_handler(health_status=503, serve_models=False))
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert readiness.health_ok(url, timeout=3) is False
        assert readiness.probe_once(url, timeout=3) is None
    finally:
        srv.shutdown()


def test_health_needs_no_auth_even_when_models_is_gated():
    # vLLM exempts /health from --api-key auth. Model a server where /v1/models
    # returns 401 (auth required) but /health is open: readiness must still fire
    # via /health without any credential.
    class _Gated(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
            else:  # /v1/models etc. require a key we never send
                self.send_response(401)
                self.end_headers()

        def log_message(self, *a):
            pass
    srv = _serve(_Gated)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert readiness.probe_once(url, timeout=3) == "ready"   # ready, no auth
    finally:
        srv.shutdown()


def test_mark_endpoint_ready_flips_the_shared_file(tmp_path):
    # the compute node's localhost/health watch flips the shared-FS endpoint to
    # ready; the login-node loop reads that instead of probing over the network.
    from boxy import jobs

    ep = tmp_path / "x.endpoint.json"
    jobs.write_endpoint_file(ep, name="x", port=8000, job_id="42")
    import json as _json
    assert _json.loads(ep.read_text())["ready"] is False
    jobs.mark_endpoint_ready(ep, model="meta-llama/Llama-3.1-8B")
    data = _json.loads(ep.read_text())
    assert data["ready"] is True and data["model"] == "meta-llama/Llama-3.1-8B"
    # the host/port/url are preserved so the tunnel target is unchanged
    assert data["port"] == 8000 and data["url"].endswith(":8000")


def test_mark_endpoint_ready_missing_file_is_noop(tmp_path):
    from boxy import jobs

    assert jobs.mark_endpoint_ready(tmp_path / "nope.json", model="m") is None


def test_probe_bypasses_proxy_env(models_server, monkeypatch):
    # a corporate proxy in the env (boxy propagates it for pulls) must NOT capture
    # the readiness probe to an internal endpoint — point http_proxy at a dead
    # address and assert the probe still reaches the real server directly.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    assert readiness.probe_once(models_server, timeout=3) == "served-model"
    assert readiness.wait_ready(models_server, timeout_s=3, interval_s=0.2) == "served-model"
