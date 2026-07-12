"""A minimal, stdlib-only reverse proxy that fronts a set of data-parallel
replicas (`<base>-r0..-r{K-1}`) with ONE OpenAI-compatible URL.

`boxy serve MODEL --replicas K` launches K independent servers, each on its own
compute node with its own endpoint file (jobs.py). This router discovers those
endpoints from the shared FS, health-checks them, and load-balances requests
across the healthy ones with **least-outstanding-requests** (request costs vary
wildly in LLM serving, so round-robin under-utilizes; least-outstanding keeps
replicas evenly loaded). It streams responses — including SSE for
`{"stream": true}` — and fails over to another replica when one dies.

Scope: this is the login-node, benchmark-scale option (see the scaling note in
the RUNBOOK). For production throughput / TLS / auth, `boxy router --emit
nginx|haproxy|litellm` prints a config for a mature proxy instead — boxy feeds a
real load balancer rather than reimplementing one.

stdlib only (http.server / http.client / threading / urllib via readiness) — no
new dependencies, matching boxy's air-gap-friendly, stdlib-only runtime.
"""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable

DEFAULT_ROUTER_PORT = 8000

# RFC 7230 §6.1 — connection-scoped headers that must NOT be forwarded across a
# proxy hop, in EITHER direction (forwarding them corrupts framing/keep-alive).
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})


@dataclass
class Backend:
    """One replica endpoint. `url` is `http://host:port` (no `/v1`), matching
    jobs.write_endpoint_file. `gen` distinguishes incarnations of the same name: a
    replica that leaves and rejoins (health-flap) gets a fresh generation, so a
    stale release from the previous incarnation can't decrement the new one."""
    name: str
    url: str
    host: str
    port: int
    model: str | None = None
    inflight: int = 0
    healthy: bool = True
    gen: int = 0


class Pool:
    """Thread-safe set of replica backends with a load-balancing policy.

    `pick` selects a backend and reserves a slot (`inflight += 1`) atomically;
    `release` frees it. The select-then-reserve is a compound read-min-then-write,
    so it runs entirely under one lock — the GIL alone is not enough (two threads
    could both observe the same minimum and stampede one replica)."""

    def __init__(self, backends: Iterable[Backend] = (), policy: str = "least"):
        self._lock = threading.Lock()
        self._by_name: dict[str, Backend] = {b.name: b for b in backends}
        self._rr = 0
        self._gen = 0
        self.policy = policy

    def pick(self, exclude: frozenset[str] = frozenset()) -> Backend | None:
        with self._lock:
            cands = [b for b in self._by_name.values() if b.healthy and b.name not in exclude]
            if not cands:
                return None
            if self.policy == "round-robin":
                chosen = cands[self._rr % len(cands)]
            else:  # least-outstanding-requests, round-robin tiebreak
                lo = min(b.inflight for b in cands)
                tied = [b for b in cands if b.inflight == lo]
                chosen = tied[self._rr % len(tied)]
            self._rr += 1
            chosen.inflight += 1
            return chosen

    def release(self, backend: Backend) -> None:
        with self._lock:
            b = self._by_name.get(backend.name)
            # only decrement if this is the SAME incarnation we reserved on — a
            # replica that left and rejoined has a fresh gen, so a stale release
            # must not steal a slot from the new incarnation's live requests.
            if b is not None and b.gen == backend.gen and b.inflight > 0:
                b.inflight -= 1

    def mark(self, name: str, healthy: bool) -> None:
        with self._lock:
            b = self._by_name.get(name)
            if b is not None:
                b.healthy = healthy

    def replace(self, backends: list[Backend]) -> None:
        """Swap the membership (discovery re-scan) while PRESERVING the in-flight
        counts of survivors — a re-scan must not reset load accounting. Dropped
        names (dead jobs) simply vanish; new names join."""
        with self._lock:
            new: dict[str, Backend] = {}
            for b in backends:
                old = self._by_name.get(b.name)
                if old is not None:
                    b.inflight = old.inflight  # survivor: carry the live-request count
                    b.gen = old.gen            # ...and its generation (same incarnation)
                else:
                    self._gen += 1
                    b.gen = self._gen          # (re)join: fresh generation
                new[b.name] = b
            self._by_name = new

    def snapshot(self) -> list[Backend]:
        with self._lock:
            return list(self._by_name.values())


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forwards every request to a pool-chosen replica, streaming the response.

    Class attributes (pool, connect_timeout, read_timeout, max_retries) are
    injected by make_server via a subclass."""

    protocol_version = "HTTP/1.1"  # keep-alive + chunked responses out
    timeout = 65                   # per-connection socket timeout (slow-loris guard)

    # injected by make_server:
    pool: Pool
    connect_timeout: float = 2.0
    read_timeout: float = 600.0
    max_retries: int = 2

    def log_message(self, *args):  # quiet, like the tests' fake server
        pass

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def do_PUT(self):
        self._proxy()

    def do_DELETE(self):
        self._proxy()

    # ---- proxy core ----

    def _proxy(self):
        try:
            body = self._read_body()
        except (ValueError, OSError):
            self._send_error_json(400, "bad request body")
            return
        self._committed = False  # set once we've sent the client a status line
        tried: set[str] = set()
        for _ in range(self.max_retries + 1):
            backend = self.pool.pick(exclude=frozenset(tried))
            if backend is None:
                self._send_error_json(503, "no healthy replica available")
                return
            try:
                self._forward(backend, body)
                return
            except (ConnectionError, socket.timeout, OSError, http.client.HTTPException):
                if self._committed:
                    # the response was already streaming to the client; we can't
                    # fail over onto a half-sent connection — just close it.
                    self.close_connection = True
                    return
                # failed BEFORE we sent the client any status → safe to retry elsewhere
                tried.add(backend.name)
                self.pool.mark(backend.name, False)  # discovery re-confirms later
            finally:
                self.pool.release(backend)
        self._send_error_json(502, "all replicas failed")

    def _forward(self, backend: Backend, body: bytes) -> None:
        conn = http.client.HTTPConnection(backend.host, backend.port, timeout=self.connect_timeout)
        try:
            conn.connect()                          # connect error here is retryable
            conn.sock.settimeout(self.read_timeout)  # generation can take minutes
            conn.request(self.command, self.path, body=body, headers=self._request_headers(backend))
            resp = conn.getresponse()               # still nothing sent to client → retryable
            self._relay(resp)                        # COMMITS: send_response + stream
        finally:
            conn.close()                             # free the replica's slot immediately

    def _relay(self, resp) -> None:
        """Relay the upstream response. This is the point of no return — once
        send_response fires we can no longer fail over."""
        length = resp.getheader("Content-Length")
        streaming = length is None  # SSE / close-delimited: no Content-Length
        self._committed = True  # past here we've written a status line to the client
        self.send_response(resp.status)
        # BaseHTTPRequestHandler emits its own Server/Date; drop the upstream's
        # (and all hop-by-hop headers) to avoid duplicates / framing corruption.
        skip = HOP_BY_HOP | {"date", "server"}
        for key, value in resp.getheaders():
            if key.lower() in skip:
                continue
            self.send_header(key, value)
        if streaming:
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")  # if an nginx ever fronts us
        self.end_headers()
        try:
            if streaming:
                # http.client de-frames inbound chunks, so re-frame outbound.
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()  # MANDATORY or SSE stalls into a buffer
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                remaining = int(length)
                while remaining > 0:
                    chunk = resp.read(min(65536, remaining))
                    if not chunk:
                        # http.client's read(amt) returns b"" on early EOF instead
                        # of raising (CPython keeps this for back-compat), so a
                        # truncated upstream would otherwise leave the client hung
                        # on a keep-alive connection expecting Content-Length bytes
                        # that never come. Turn the short read into a connection
                        # close so the client sees a clean truncation, not a desync.
                        self.close_connection = True
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            # client vanished (BrokenPipe/ConnectionReset) OR the upstream replica
            # died mid-stream: either way the response is already committed, so we
            # can only truncate and drop the keep-alive — never retry (see _proxy).
            self.close_connection = True

    def _request_headers(self, backend: Backend) -> dict[str, str]:
        # strip hop-by-hop + inbound Content-Length (conn.request recomputes it)
        # and rewrite Host to the chosen replica.
        out = {k: v for k, v in self.headers.items()
               if k.lower() not in HOP_BY_HOP and k.lower() not in ("host", "content-length")}
        out["Host"] = f"{backend.host}:{backend.port}"
        return out

    def _read_body(self) -> bytes:
        cl = self.headers.get("Content-Length")
        if cl is not None:
            return self.rfile.read(int(cl))
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            return self._read_chunked_body()
        return b""

    def _read_chunked_body(self) -> bytes:
        # rare from OpenAI SDKs (they send Content-Length); buffered whole.
        out = bytearray()
        while True:
            size_line = self.rfile.readline().strip()
            if not size_line:
                break
            size = int(size_line.split(b";", 1)[0], 16)
            if size == 0:
                self.rfile.readline()  # trailing CRLF
                break
            out += self.rfile.read(size)
            self.rfile.readline()      # CRLF after each chunk
        return bytes(out)

    def _send_error_json(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"message": f"boxy router: {message}", "type": "router_error"}}).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def make_server(pool: Pool, port: int, host: str = "",
                connect_timeout: float = 2.0, read_timeout: float = 600.0,
                max_retries: int = 2) -> ThreadingHTTPServer:
    from boxy import config

    handler = type("_BoxyProxyHandler", (_ProxyHandler,), {
        "pool": pool, "connect_timeout": connect_timeout,
        "read_timeout": read_timeout, "max_retries": max_retries,
    })
    server = ThreadingHTTPServer((host or config.get("network.bind_host"), port), handler)
    server.daemon_threads = True  # don't block shutdown on a live stream
    return server


# ---- replica discovery -------------------------------------------------------


def scan_backends(base: str, health_timeout: float = 3.0) -> list[Backend]:
    """Read the `<base>-r*` endpoint files and return the ones that answer
    /v1/models. Pure-ish (touches the shared FS + probes each URL)."""
    from boxy import jobs, readiness

    found: list[Backend] = []
    for ep in jobs.list_endpoints(base):
        model = readiness.wait_ready(ep["url"], timeout_s=health_timeout, interval_s=1)
        if model:
            found.append(Backend(name=ep["name"], url=ep["url"], host=ep["host"],
                                  port=int(ep["port"]), model=model))
    return found


class DiscoveryThread(threading.Thread):
    """Periodically re-scan the endpoint files so replicas joining (new job
    becomes ready) or leaving (job dies / health fails) update the live pool."""

    def __init__(self, base: str, pool: Pool, interval: float = 10.0, health_timeout: float = 3.0):
        super().__init__(daemon=True)
        self.base = base
        self.pool = pool
        self.interval = interval
        self.health_timeout = health_timeout
        self._stop = threading.Event()

    def scan_once(self) -> list[Backend]:
        backends = scan_backends(self.base, self.health_timeout)
        self.pool.replace(backends)
        return backends

    def run(self) -> None:
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


# ---- config emitters for a production proxy (pure, testable) -----------------


def _ident(base: str) -> str:
    """A safe nginx/haproxy identifier from a base name."""
    return "boxy_" + re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")


def _servers(endpoints: list[dict]) -> list[str]:
    return [f"{ep['host']}:{ep['port']}" for ep in endpoints]


def emit_nginx(base: str, endpoints: list[dict], listen_port: int = DEFAULT_ROUTER_PORT) -> str:
    name = _ident(base)
    upstream = "\n".join(f"    server {s} max_fails=2 fail_timeout=10s;" for s in _servers(endpoints))
    return (
        f"# nginx reverse proxy for boxy replica set {base!r} ({len(endpoints)} replica(s))\n"
        f"upstream {name} {{\n"
        f"    least_conn;\n"
        f"{upstream}\n"
        f"}}\n"
        f"server {{\n"
        f"    listen {listen_port};\n"
        f"    location / {{\n"
        f"        proxy_pass http://{name};\n"
        f"        proxy_http_version 1.1;\n"
        f"        proxy_buffering off;          # SSE: stream tokens, do not buffer\n"
        f"        proxy_read_timeout 600s;\n"
        f"    }}\n"
        f"}}\n"
    )


def emit_haproxy(base: str, endpoints: list[dict], listen_port: int = DEFAULT_ROUTER_PORT) -> str:
    name = _ident(base)
    servers = "\n".join(f"    server r{i} {s} check inter 5s fall 2 rise 1"
                        for i, s in enumerate(_servers(endpoints)))
    return (
        f"# haproxy config for boxy replica set {base!r} ({len(endpoints)} replica(s))\n"
        f"frontend {name}_fe\n"
        f"    bind *:{listen_port}\n"
        f"    default_backend {name}_be\n"
        f"backend {name}_be\n"
        f"    balance leastconn\n"
        f"    option httpchk GET /v1/models\n"
        f"{servers}\n"
    )


def emit_litellm(base: str, endpoints: list[dict], model_name: str | None = None) -> str:
    name = model_name or base
    lines = [f"# LiteLLM proxy config for boxy replica set {base!r} ({len(endpoints)} replica(s))",
             "# NOTE: set `model:` to the id each server reports at /v1/models.",
             "model_list:"]
    for ep in endpoints:
        served = ep.get("model") or name
        lines.append(f"  - model_name: {name}")
        lines.append("    litellm_params:")
        lines.append(f"      model: openai/{served}")
        lines.append(f"      api_base: {ep['url']}/v1")
        lines.append("      api_key: dummy")
    lines.append("router_settings:")
    lines.append("  routing_strategy: least-busy")
    return "\n".join(lines) + "\n"
