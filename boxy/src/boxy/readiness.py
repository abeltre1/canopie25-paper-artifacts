"""Readiness wait: the difference between "a container was launched" and
"you have a working endpoint". Polls /v1/models until the server answers."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from typing import Callable


def wait_ready(
    url: str,
    timeout_s: float = 180.0,
    interval_s: float = 1.0,
    still_alive: Callable[[], bool] | None = None,
) -> str | None:
    """Poll GET {url}/v1/models until it returns a model list; returns the
    served model id, or None on timeout.

    `still_alive` (e.g. `podman inspect .State.Running`) turns a crashed
    server into an immediate RuntimeError instead of a silent full-length
    timeout — the caller can then surface the container logs."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=2) as resp:
                data = json.load(resp)
            # a non-OpenAI responder on this port (JSON array, HTML, junk)
            # is 'not ready yet', never a crash (sweep finding 26)
            if isinstance(data, dict):
                models = data.get("data") or []
                if models and isinstance(models[0], dict):
                    return models[0].get("id", "unknown")
        except (urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException):
            pass
        if still_alive is not None and not still_alive():
            raise RuntimeError("server exited during startup")
        time.sleep(interval_s)
    return None
