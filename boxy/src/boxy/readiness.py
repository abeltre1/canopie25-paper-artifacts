"""Readiness wait: the difference between "a container was launched" and
"you have a working endpoint". Polls /v1/models until the server answers, and —
because the HTTP probe can't always reach a compute node (a corporate proxy in
the env, or the port only bound inside the container / behind the login node) —
also accepts the engine's own "server is up" line from the job LOG on the shared
filesystem as a ready signal."""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable

# The engine's own "the HTTP server is accepting requests" markers, read from the
# job log on the shared FS. This is the AUTHORITATIVE readiness signal when the
# probe can't reach the endpoint (compute node not routable from the login node,
# a proxy in the way, or the port bound only inside the container). Kept broad on
# purpose — a false "ready" only costs one failed curl, a missed one hangs.
_LOG_READY_MARKERS = (
    "Application startup complete",          # vLLM / uvicorn (FastAPI startup done)
    "Uvicorn running on",                    # vLLM / uvicorn bind line
    "Starting vLLM API server",              # vLLM (older builds print this at bind)
    "server is listening",                   # llama.cpp ("main: server is listening on ...")
    "HTTP server listening",                 # llama.cpp (some builds)
    "all slots are idle",                    # llama.cpp (ready to serve)
)

# `INFO ... model id: <id>` / vLLM's served-model line, to name the model when the
# HTTP probe (which returns the id) never succeeded.
_LOG_MODEL_RE = re.compile(r"served model name[s]?[:=]\s*([^\s,'\"]+)", re.IGNORECASE)


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    """An opener that IGNORES http(s)_proxy. boxy propagates a corporate proxy for
    image/model pulls; that same env would otherwise route the readiness GET to an
    INTERNAL compute node (http://node:8000) through the proxy, which can't reach
    it — so the probe fails forever even though the server is up. Internal
    endpoints must always be direct."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _log_tail(log_path: str | os.PathLike | None, window: int = 262144) -> str:
    """Last `window` bytes of the log (default 256 KiB). The startup ready marker
    lands near the end during boot, so a bounded tail keeps polling a verbose vLLM
    log cheap instead of re-reading the whole file every second. '' on any error."""
    if not log_path:
        return ""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > window:
                f.seek(size - window)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def log_is_ready(log_path: str | os.PathLike | None) -> bool:
    """True if the job log contains an engine 'server is up' marker — the ready
    signal that survives an unreachable endpoint (proxy / compute-node topology)."""
    text = _log_tail(log_path)
    return any(m in text for m in _LOG_READY_MARKERS)


def model_from_log(log_path: str | os.PathLike | None) -> str | None:
    """Best-effort served-model id parsed from the log, for the READY banner when
    the HTTP probe (which normally supplies it) never answered."""
    m = _LOG_MODEL_RE.search(_log_tail(log_path))
    return m.group(1) if m else None


def _model_id_from_models(url: str, timeout: float) -> str | None:
    """Served model id from GET {url}/v1/models, or None."""
    try:
        with _no_proxy_opener().open(f"{url}/v1/models", timeout=timeout) as resp:
            data = json.load(resp)
        # a non-OpenAI responder on this port (JSON array, HTML, junk) is 'not
        # ready yet', never a crash (sweep finding 26)
        if isinstance(data, dict):
            models = data.get("data") or []
            if models and isinstance(models[0], dict):
                return models[0].get("id", "unknown")
    except (urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException):
        pass
    return None


def health_ok(url: str, timeout: float = 2.0) -> bool:
    """True if GET {url}/health returns 2xx. `/health` is the canonical readiness
    endpoint for both vLLM and llama.cpp — a cheap 200 the instant the server can
    serve, lighter and less ambiguous than parsing /v1/models. Proxy-bypassed."""
    try:
        with _no_proxy_opener().open(f"{url}/health", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 300  # unlikely, but a 2xx HTTPError still means "up"
    except (urllib.error.URLError, OSError, http.client.HTTPException):
        return False


def probe_once(url: str, timeout: float = 2.0) -> str | None:
    """One direct (proxy-bypassing) readiness probe. Ready = GET {url}/health is
    2xx (the canonical endpoint) OR — for servers without /health — /v1/models
    returns a model list. Returns the served model id (from /v1/models when it
    answers), else 'ready' when only /health confirmed it, else None."""
    if health_ok(url, timeout):
        return _model_id_from_models(url, timeout) or "ready"
    return _model_id_from_models(url, timeout)


def wait_ready(
    url: str,
    timeout_s: float = 180.0,
    interval_s: float = 1.0,
    still_alive: Callable[[], bool] | None = None,
    log_path: str | os.PathLike | None = None,
) -> str | None:
    """Poll until the server is ready; returns the served model id, or None on
    timeout. Ready = the /v1/models probe answers (proxy-bypassed) OR — when
    `log_path` is given and the probe can't reach the endpoint — the engine's
    'server is up' line appears in the log (the model id is then read from the log,
    falling back to 'ready').

    `still_alive` (e.g. `podman inspect .State.Running`) turns a crashed server
    into an immediate RuntimeError instead of a silent full-length timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        model_id = probe_once(url)
        if model_id:
            return model_id
        if log_path is not None and log_is_ready(log_path):
            return model_from_log(log_path) or "ready"
        if still_alive is not None and not still_alive():
            raise RuntimeError("server exited during startup")
        time.sleep(interval_s)
    return None
