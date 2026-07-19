"""Router: pool/pick (pure), discovery join/leave, config emitters (pure), and a
real end-to-end fan-out + SSE streaming + failover test against local backends.
No cluster needed."""

import http.client
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from boxy import jobs, router

EPS = [
    {"name": "demo-r0", "host": "node01", "port": 8000, "url": "http://node01:8000"},
    {"name": "demo-r1", "host": "node02", "port": 8000, "url": "http://node02:8000"},
]


# ---- emit (pure) -------------------------------------------------------------


def test_emit_nginx_least_conn_and_servers():
    out = router.emit_nginx("Meta-Llama-3.1-8B", EPS, listen_port=9000)
    assert "least_conn;" in out
    assert "server node01:8000" in out and "server node02:8000" in out
    assert "listen 9000;" in out
    assert "proxy_buffering off" in out  # SSE-safe
    assert "upstream boxy_meta_llama_3_1_8b" in out  # sanitized identifier


def test_emit_haproxy_leastconn_and_healthcheck():
    out = router.emit_haproxy("demo", EPS)
    assert "balance leastconn" in out
    assert "option httpchk GET /v1/models" in out
    assert out.count("server r") == 2


def test_emit_litellm_one_entry_per_replica_same_model_name():
    out = router.emit_litellm("demo", EPS)
    assert out.count("model_name: demo") == 2       # both share the name → LB
    assert "api_base: http://node01:8000/v1" in out
    assert "routing_strategy: least-busy" in out


# ---- pool / pick (pure) ------------------------------------------------------


def _pool(*inflight):
    return router.Pool([router.Backend(f"r{i}", f"http://h{i}:8000", f"h{i}", 8000, inflight=n)
                        for i, n in enumerate(inflight)])


def test_pick_least_outstanding_wins():
    p = _pool(3, 0, 5)
    b = p.pick()
    assert b.name == "r1" and b.inflight == 1  # least loaded, reserved


def test_pick_round_robin_tiebreak_and_exclude():
    p = _pool(0, 0, 0)
    first = p.pick()
    second = p.pick(exclude=frozenset({first.name}))
    assert first.name != second.name           # ties rotate + exclude skips
    assert p.pick(exclude=frozenset({"r0", "r1", "r2"})) is None  # all excluded


def test_pick_none_when_all_unhealthy():
    p = _pool(0, 0)
    for b in p.snapshot():
        p.mark(b.name, False)
    assert p.pick() is None


def test_concurrent_picks_reserve_distinct_slots():
    # 3 concurrent picks (no release) on 3 idle backends must reserve 3 distinct
    # backends — the select+reserve is atomic under the lock (no stampede).
    p = _pool(0, 0, 0)
    picked = []
    lock = threading.Lock()

    def grab():
        b = p.pick()
        with lock:
            picked.append(b.name)

    threads = [threading.Thread(target=grab) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(picked) == ["r0", "r1", "r2"]
    assert all(b.inflight == 1 for b in p.snapshot())


def test_release_decrements():
    p = _pool(2)
    b = p.pick()
    assert b.inflight == 3
    p.release(b)
    assert p.snapshot()[0].inflight == 2


def test_release_ignores_stale_backend_after_rejoin():
    # A replica that leaves and rejoins (health-flap) gets a fresh generation; a
    # stale release from its previous incarnation must NOT steal the new one's slot.
    pool = router.Pool()
    pool.replace([router.Backend("r0", "http://h:8000", "h", 8000)])
    picked = pool.pick()                    # request A reserves on r0 (gen g1)
    assert picked.inflight == 1
    pool.replace([])                        # r0 leaves the pool
    pool.replace([router.Backend("r0", "http://h:8000", "h", 8000)])  # rejoins fresh (gen g2)
    assert pool.snapshot()[0].inflight == 0
    b = pool.pick()                         # request B reserves on the fresh r0
    assert b.inflight == 1
    pool.release(picked)                    # A's stale release — different gen → ignored
    assert pool._by_name["r0"].inflight == 1
    pool.release(b)                         # B's real release
    assert pool._by_name["r0"].inflight == 0


# ---- discovery replace: join / leave / preserve inflight ---------------------


def test_discovery_join_leave_preserves_inflight(monkeypatch):
    from boxy import readiness

    state = {"eps": [dict(EPS[0], url="http://h0:8000", host="h0"),
                     dict(EPS[1], url="http://h1:8000", host="h1")]}
    for i, e in enumerate(state["eps"]):
        e["name"] = f"base-r{i}"
    monkeypatch.setattr(jobs, "list_endpoints", lambda base: state["eps"])
    monkeypatch.setattr(readiness, "wait_ready", lambda url, **k: "fake-model")

    pool = router.Pool()
    disc = router.DiscoveryThread("base", pool)
    disc.scan_once()
    assert {b.name for b in pool.snapshot()} == {"base-r0", "base-r1"}

    pool._by_name["base-r0"].inflight = 3          # survivor has load
    # base-r1 leaves, base-r2 joins
    state["eps"] = [state["eps"][0], {"name": "base-r2", "host": "h2", "port": 8000, "url": "http://h2:8000"}]
    disc.scan_once()
    assert {b.name for b in pool.snapshot()} == {"base-r0", "base-r2"}
    assert pool._by_name["base-r0"].inflight == 3  # preserved across the swap


# ---- jobs.list_endpoints -----------------------------------------------------


def test_list_endpoints_globs_replica_files(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    for i in range(3):
        jobs.write_endpoint(f"myset-r{i}", 8000 + i, job_id=str(i))
    jobs.write_endpoint("other-r0", 9000)          # different base — must not match
    jobs.write_endpoint("myset2-r0", 9001)         # base is a prefix — must not match
    jobs.write_endpoint("myset-rockery-r0", 9002)  # -r not followed by a digit — must not match
    found = jobs.list_endpoints("myset")
    assert sorted(e["name"] for e in found) == ["myset-r0", "myset-r1", "myset-r2"]


# ---- end-to-end: real backends, fan-out + SSE + failover ---------------------


class _FakeBackend(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n)) if n else {}
        self.server.hits += 1
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for tok in ("Hel", "lo", "[DONE]"):
                frame = f"data: {tok}\n\n".encode()
                self.wfile.write(b"%X\r\n" % len(frame) + frame + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        else:
            self._json({"choices": [{"text": "ok"}], "server": self.server.tag,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1}})


def _start_backend(tag):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _FakeBackend)
    srv.hits = 0
    srv.tag = tag
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def router_over_two_backends(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    backends = [_start_backend("A"), _start_backend("B")]
    for i, s in enumerate(backends):
        port = s.server_address[1]
        jobs.write_endpoint(f"e2e-r{i}", port, job_id=str(i))
        # jobs writes host=gethostname; rewrite to 127.0.0.1 so the test reaches it
        ep = tmp_path / f"e2e-r{i}.endpoint.json"
        d = json.loads(ep.read_text())
        d.update(host="127.0.0.1", url=f"http://127.0.0.1:{port}")
        ep.write_text(json.dumps(d))
    pool = router.Pool()
    disc = router.DiscoveryThread("e2e", pool)
    disc.scan_once()
    srv = router.make_server(pool, 0, host="127.0.0.1", connect_timeout=1.0, read_timeout=5.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rport = srv.server_address[1]
    yield rport, backends, pool
    srv.shutdown()
    disc.stop()
    for s in backends:
        s.shutdown()


def _post(rport, stream=False):
    body = json.dumps({"model": "fake-model", "prompt": "hi", "max_tokens": 4, "stream": stream}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{rport}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def test_router_discovers_both_backends(router_over_two_backends):
    _, _, pool = router_over_two_backends
    assert len(pool.snapshot()) == 2
    assert all(b.model == "fake-model" for b in pool.snapshot())


def test_router_fans_out_across_replicas(router_over_two_backends):
    rport, backends, _ = router_over_two_backends
    for _ in range(10):
        resp = _post(rport)
        assert resp.status == 200
        json.load(resp)
    assert backends[0].hits > 0 and backends[1].hits > 0        # both used
    assert backends[0].hits + backends[1].hits == 10           # nothing dropped


def test_router_streams_sse(router_over_two_backends):
    rport, _, _ = router_over_two_backends
    resp = _post(rport, stream=True)
    assert resp.status == 200
    body = resp.read().decode()
    assert "data: Hel" in body and "data: lo" in body
    assert "data: [DONE]" in body                              # full SSE relayed intact


class _AbortBackend(BaseHTTPRequestHandler):
    """Answers /v1/models (so it looks healthy) but aborts every POST mid-chunk —
    simulates a replica dying while streaming."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b"64\r\npartial")  # declares 0x64=100 bytes, sends 7, then closes
        self.wfile.flush()
        self.close_connection = True


def test_router_survives_upstream_abort_midstream(tmp_path, monkeypatch):
    # A replica that dies mid-stream (after headers) must NOT crash the router or
    # trigger a double send_response — the router truncates that one and stays up.
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    good = _start_backend("good")
    bad = ThreadingHTTPServer(("127.0.0.1", 0), _AbortBackend)
    threading.Thread(target=bad.serve_forever, daemon=True).start()
    for name, s in (("mix-r0", good), ("mix-r1", bad)):
        port = s.server_address[1]
        (tmp_path / f"{name}.endpoint.json").write_text(json.dumps(
            {"name": name, "host": "127.0.0.1", "port": port,
             "url": f"http://127.0.0.1:{port}", "job": "1"}))
    pool = router.Pool()
    disc = router.DiscoveryThread("mix", pool)
    disc.scan_once()
    srv = router.make_server(pool, 0, host="127.0.0.1", connect_timeout=1.0, read_timeout=3.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rport = srv.server_address[1]
    try:
        for _ in range(4):  # hit the aborting replica among others; must not wedge
            try:
                _post(rport, stream=True).read()
            except Exception:
                pass
        oks = 0
        for _ in range(6):  # the router is still alive; the good replica still serves
            try:
                r = _post(rport)
                r.read()
                oks += r.status == 200
            except Exception:
                pass
        assert oks >= 1
    finally:
        srv.shutdown()
        disc.stop()
        good.shutdown()
        bad.shutdown()


class _TruncatingBackend(BaseHTTPRequestHandler):
    """Answers /v1/models, but its POST declares Content-Length: 100 and sends only
    10 bytes before closing — a truncated non-streamed response."""
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "100")  # lies: only sends 10
        self.end_headers()
        self.wfile.write(b"x" * 10)
        self.wfile.flush()
        self.close_connection = True


def test_router_closes_on_truncated_content_length(tmp_path, monkeypatch):
    # An upstream that under-delivers its Content-Length must make the router CLOSE
    # the client connection (clean truncation) rather than leave it hung waiting for
    # bytes that never come (keep-alive desync). The client then sees IncompleteRead.
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    bad = ThreadingHTTPServer(("127.0.0.1", 0), _TruncatingBackend)
    threading.Thread(target=bad.serve_forever, daemon=True).start()
    port = bad.server_address[1]
    (tmp_path / "trunc-r0.endpoint.json").write_text(json.dumps(
        {"name": "trunc-r0", "host": "127.0.0.1", "port": port,
         "url": f"http://127.0.0.1:{port}", "job": "1"}))
    pool = router.Pool()
    disc = router.DiscoveryThread("trunc", pool)
    disc.scan_once()
    srv = router.make_server(pool, 0, host="127.0.0.1", connect_timeout=1.0, read_timeout=3.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rport = srv.server_address[1]
    try:
        resp = _post(rport)
        with pytest.raises(http.client.IncompleteRead):
            resp.read()  # closed after 10 of the declared 100 bytes
    finally:
        srv.shutdown()
        disc.stop()
        bad.shutdown()


def test_router_fails_over_when_a_replica_dies(router_over_two_backends):
    rport, backends, _ = router_over_two_backends
    backends[0].shutdown()
    backends[0].server_close()                                 # free the port → connect refused
    ok = 0
    for _ in range(6):
        resp = _post(rport)
        if resp.status == 200:
            ok += 1
            json.load(resp)
    assert ok == 6                                             # all served via the survivor
    assert backends[1].hits >= 6


# ---- boxy curl (endpoint query from the records) ------------------------------


class _ChatBackend(_FakeBackend):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(n)) if n else {}
        self._json({"choices": [{"message": {"role": "assistant", "content": "boxy endpoint OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 3}})


def test_boxy_open_reports_endpoint_and_browser_url(tmp_path, monkeypatch, capsys):
    """`boxy open NAME` (cluster-side, no --ssh) prints the endpoint READY banner
    (the laptop watches for it to tunnel) + the llama.cpp web-UI root URL + the
    ssh -L a workstation needs. N instances each address by their own name."""
    from boxy import cli
    from boxy.cli import main as cli_main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: True)
    for nm, host, port in [("boxy-a", "cbnode1003", 8090), ("boxy-b", "cbnode1004", 8090)]:
        jobs.write_record(nm, {"name": nm, "scheduler": "flux", "job": nm})
        (tmp_path / f"{nm}.endpoint.json").write_text(json.dumps(
            {"name": nm, "host": host, "port": port, "url": f"http://{host}:{port}", "job": nm}))
    assert cli_main(["open", "boxy-b"]) == 0
    out = capsys.readouterr().out
    assert "### READY  http://cbnode1004:8090/v1" in out           # tunnel banner
    assert "http://cbnode1004:8090/" in out                        # browser (web UI) URL
    assert "ssh -L 8090:cbnode1004:8090" in out                    # workstation tunnel
    # ambiguous with N up: must name one
    assert cli_main(["open"]) == 2
    assert "several models are serving" in capsys.readouterr().err


def test_boxy_open_route_prints_friendly_name(tmp_path, monkeypatch, capsys):
    """`boxy open NAME --route foo` (cluster-side) folds a friendly
    http://foo.localhost:PORT/ name into the workstation tunnel instructions —
    no DNS needed (SPEC §8b Tier 1)."""
    from boxy import cli
    from boxy.cli import main as cli_main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: True)
    jobs.write_record("boxy-a", {"name": "boxy-a", "scheduler": "flux", "job": "boxy-a"})
    (tmp_path / "boxy-a.endpoint.json").write_text(json.dumps(
        {"name": "boxy-a", "host": "cbnode1003", "port": 8090,
         "url": "http://cbnode1003:8090", "job": "boxy-a"}))
    assert cli_main(["open", "boxy-a", "--route", "nemotron"]) == 0
    out = capsys.readouterr().out
    assert "ssh -L 8090:cbnode1003:8090" in out                 # tunnel still by port
    assert "http://nemotron.localhost:8090/" in out           # friendly browser name
    assert "http://nemotron.localhost:8090/v1" in out         # friendly API base


def test_boxy_curl_by_name_and_single_default(tmp_path, monkeypatch, capsys):
    from boxy import cli
    from boxy.cli import main as cli_main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: True)  # CI runners have no squeue
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChatBackend)
    srv.hits = 0
    srv.tag = "chat"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        jobs.write_record("boxy-tiny", {"name": "boxy-tiny", "scheduler": "slurm", "job": "1"})
        (tmp_path / "boxy-tiny.endpoint.json").write_text(json.dumps(
            {"name": "boxy-tiny", "host": "127.0.0.1", "port": port,
             "url": f"http://127.0.0.1:{port}", "job": "1"}))
        # by name
        assert cli_main(["curl", "boxy-tiny"]) == 0
        out = capsys.readouterr().out
        assert "boxy endpoint OK" in out and "fake-model" in out
        # single instance: name optional
        assert cli_main(["curl"]) == 0
        assert "boxy endpoint OK" in capsys.readouterr().out
        # unknown name: helpful error listing what's up
        assert cli_main(["curl", "nope"]) == 2
        assert "boxy-tiny" in capsys.readouterr().err
    finally:
        srv.shutdown()


def test_boxy_curl_nothing_serving(tmp_path, monkeypatch, capsys):
    from boxy.cli import main as cli_main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    assert cli_main(["curl"]) == 2
    assert "nothing is serving" in capsys.readouterr().err


def test_boxy_curl_skips_foreign_cluster_endpoints(tmp_path, monkeypatch, capsys):
    """Shared $HOME: another cluster's endpoint (a scheduler this host can't
    speak) must never be auto-picked — its node hostname doesn't resolve here.
    Field report: `boxy curl --ssh clustera` grabbed an clusterb endpoint."""
    from boxy.cli import main as cli_main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    # cluster identity decides (deterministic on any host, incl. CI runners):
    # this host "is" clustera, so the clusterb-submitted record is the foreign one
    monkeypatch.setenv("BOXY_CLUSTER", "clustera")
    # a foreign flux job with an endpoint
    jobs.write_record("boxy-cbnode", {"name": "boxy-cbnode", "scheduler": "flux",
                                    "job": "f2ag", "submitted_from": "clusterb-login2"})
    (tmp_path / "boxy-cbnode.endpoint.json").write_text(json.dumps(
        {"name": "boxy-cbnode", "host": "cbnode1027", "port": 8090,
         "url": "http://cbnode1027:8090", "job": "f2ag"}))
    # bare curl: nothing local -> explains where the foreign one lives
    assert cli_main(["curl"]) == 2
    err = capsys.readouterr().err
    assert "nothing is serving on THIS cluster" in err and "boxy-cbnode" in err and "--ssh" in err
    # naming the foreign one: pointed at its own cluster, no DNS attempt
    assert cli_main(["curl", "boxy-cbnode"]) == 2
    err = capsys.readouterr().err
    assert "another cluster" in err and "clusterb-login2" in err
    # a LOCAL (slurm) endpoint alongside: bare curl picks it, not the foreign one
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _ChatBackend)
    srv.hits = 0
    srv.tag = "chat"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        jobs.write_record("boxy-here", {"name": "boxy-here", "scheduler": "slurm", "job": "9",
                                        "submitted_from": "clustera-login1"})
        (tmp_path / "boxy-here.endpoint.json").write_text(json.dumps(
            {"name": "boxy-here", "host": "127.0.0.1", "port": port,
             "url": f"http://127.0.0.1:{port}", "job": "9"}))
        assert cli_main(["curl"]) == 0
        assert "boxy endpoint OK" in capsys.readouterr().out
    finally:
        srv.shutdown()
