"""OpenShift relay exposer — the everyone-URL for a boxy tunnel, zero teammate
setup (RUNBOOK §0.993).

The laptop already holds `ssh -L 127.0.0.1:<lport> -> compute-node:port`. This
exposer starts a `chisel client` (single static Go binary) that dials OUTBOUND
to a chisel server pod on OpenShift (websocket over the edge-TLS Route, so it
traverses Zscaler like any HTTPS), opens reverse port R:<relay_port> -> the
laptop loopback, and creates a per-alias Service+Route so
`https://<alias>-boxy.apps.<cluster>/` reaches the model. Name resolution rides
the cluster's EXISTING wildcard DNS (`*.apps.<cluster>` already resolves on
every corporate machine) — no nameserver, no client software, no enrollment:
the lessons of the reverted Headscale tier, designed out.

Pure string emitters live at module level (the router.emit_nginx / mcp.py
style) so the manifests are golden-testable without a cluster; all subprocess
calls resolve their binary through env overrides (BOXY_CHISEL / BOXY_OC) so the
whole flow is CI-testable with shims, exactly like BOXY_SSH."""

from __future__ import annotations

import base64
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time

from boxy.exposers.base import ExposeError, Exposer, ShareContext

RELAY_IMAGE = "docker.io/jpillora/chisel:1.10"
RELAY_PORT_RANGE = range(31000, 32000)  # per-share reverse ports; ACL-able as R:31[0-9]{3}
RELAY_APP = "boxy-relay"                # Deployment/Service/Route/Secret name + selector
SHARE_LABEL = "boxy.share"              # every per-alias object carries this for teardown

ENV_RELAY_URL = "BOXY_RELAY_URL"            # https://relay-boxy.apps.<cluster> (skip oc discovery)
ENV_RELAY_NAMESPACE = "BOXY_RELAY_NAMESPACE"
ENV_RELAY_AUTH = "BOXY_RELAY_AUTH"          # user:pass (skip oc secret fetch)
ENV_CHISEL = "BOXY_CHISEL"                  # chisel binary override (tests: a shim)
ENV_OC = "BOXY_OC"                          # oc binary override (tests: a shim)
ENV_CHISEL_ARGS = "BOXY_CHISEL_ARGS"        # extra client args, e.g. --tls-ca / --proxy (Zscaler)

DEFAULT_NAMESPACE = "boxy-relay"
BIND_GRACE = 3.0     # seconds to watch a fresh chisel client for a bind failure
ADMIT_TIMEOUT = 10.0  # seconds to wait for the Route to be admitted (host not taken)
ADMIT_POLL = 0.5

_ALIAS_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?$")


def chisel_bin() -> str:
    return os.environ.get(ENV_CHISEL, "chisel")


def oc_bin() -> str:
    return os.environ.get(ENV_OC, "oc")


# ---- pure helpers ---------------------------------------------------------------


def apps_domain_from_url(relay_url: str) -> str:
    """`https://relay-boxy.apps.goodall.sandia.gov` -> `apps.goodall.sandia.gov`.
    Shared names live directly under the cluster's wildcard, which corporate DNS
    already serves — that's the whole reason teammates need zero setup."""
    host = relay_url.strip().rstrip("/")
    for pre in ("https://", "http://"):
        if host.startswith(pre):
            host = host[len(pre):]
    host = host.split("/", 1)[0].split(":", 1)[0]
    if "." not in host:
        raise ExposeError(f"relay URL {relay_url!r} has no domain — expected "
                          f"https://relay-boxy.apps.<cluster>")
    return host.split(".", 1)[1]


def share_host(alias: str, apps_domain: str) -> str:
    """Public hostname for a share: `<alias>-boxy.<apps_domain>`. The `-boxy`
    suffix namespaces the shared wildcard so an alias can never squat another
    team's Route host. Alias must be a DNS-label-safe name."""
    if not _ALIAS_RE.match(alias):
        raise ExposeError(f"share name {alias!r} must be lowercase letters/digits/hyphens "
                          f"(1-40 chars, no leading/trailing hyphen)")
    return f"{alias}-boxy.{apps_domain}"


def pick_relay_port(taken: set[int], rand: random.Random | None = None) -> int:
    """A random free reverse port in RELAY_PORT_RANGE (random so two laptops
    racing rarely collide; a collision is caught by chisel's bind error and
    retried)."""
    free = [p for p in RELAY_PORT_RANGE if p not in taken]
    if not free:
        raise ExposeError(f"all {len(RELAY_PORT_RANGE)} relay ports are taken — "
                          f"`boxy unshare` stale shares or widen RELAY_PORT_RANGE")
    return (rand or random).choice(free)


def emit_relay_manifest(host: str, namespace: str = DEFAULT_NAMESPACE, *,
                        image: str = RELAY_IMAGE, auth: str = "",
                        key_seed: str = "", port: int = 8080) -> str:
    """The relay itself (deploy ONCE per cluster): chisel server behind an
    edge-TLS Route. AUTH and KEY_SEED reach the container as env from the
    Secret (`secretKeyRef`) — never argv, so `oc get deploy -o yaml` leaks
    nothing. `--key $(KEY_SEED)` keeps the server's host key stable across pod
    restarts WITHOUT a PVC. Both Route timeout annotations matter: `timeout`
    governs pre-upgrade, `timeout-tunnel` governs the live websocket."""
    docs = [
        f"""apiVersion: v1
kind: Secret
metadata:
  name: {RELAY_APP}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}}}
type: Opaque
stringData:
  auth: {auth or "REPLACE_ME:REPLACE_ME"}
  key-seed: {key_seed or "REPLACE_ME"}""",
        f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {RELAY_APP}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}}}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: {RELAY_APP}}}
  template:
    metadata:
      labels: {{app: {RELAY_APP}}}
    spec:
      securityContext:
        runAsNonRoot: true
      containers:
        - name: chisel
          image: {image}
          args: ["server", "--reverse", "--port", "{port}", "--key", "$(KEY_SEED)", "--keepalive", "25s"]
          env:
            - name: AUTH
              valueFrom: {{secretKeyRef: {{name: {RELAY_APP}, key: auth}}}}
            - name: KEY_SEED
              valueFrom: {{secretKeyRef: {{name: {RELAY_APP}, key: key-seed}}}}
          ports:
            - {{containerPort: {port}, name: http}}""",
        f"""apiVersion: v1
kind: Service
metadata:
  name: {RELAY_APP}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}}}
spec:
  selector: {{app: {RELAY_APP}}}
  ports:
    - {{name: http, port: {port}, targetPort: {port}}}""",
        f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: {RELAY_APP}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}}}
  annotations:
    haproxy.router.openshift.io/timeout: "3600s"
    haproxy.router.openshift.io/timeout-tunnel: "3600s"
spec:
  host: {host}
  to: {{kind: Service, name: {RELAY_APP}}}
  port: {{targetPort: http}}
  tls: {{termination: edge}}""",
    ]
    hint = ("" if auth else
            f"# set the credential first:  oc -n {namespace} create secret generic {RELAY_APP} "
            f'--from-literal=auth="boxy:$(openssl rand -hex 16)" '
            f'--from-literal=key-seed="$(openssl rand -hex 16)" --dry-run=client -o yaml | oc apply -f -\n')
    header = (f"# boxy relay (chisel server) on OpenShift — the everyone-URL ingress for boxy shares.\n"
              f"# apply:  oc new-project {namespace} 2>/dev/null; "
              f"boxy generate relay --host {host} | oc apply -f -\n" + hint +
              "# then on the laptop:  brew install chisel-tunnel;  boxy open <inst> --ssh <login> --share <name>\n")
    return header + "\n---\n".join(docs) + "\n"


def emit_share_manifest(alias: str, host: str, relay_port: int,
                        namespace: str = DEFAULT_NAMESPACE) -> str:
    """One share = one Service+Route pair pointing the public hostname at the
    relay pod's reverse port. Services double as the cross-laptop port-
    allocation ledger (label-selected), so no extra state store is needed."""
    docs = [
        f"""apiVersion: v1
kind: Service
metadata:
  name: boxy-share-{alias}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}, {SHARE_LABEL}: {alias}}}
spec:
  selector: {{app: {RELAY_APP}}}
  ports:
    - {{name: share, port: {relay_port}, targetPort: {relay_port}}}""",
        f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: boxy-share-{alias}
  namespace: {namespace}
  labels: {{app: {RELAY_APP}, {SHARE_LABEL}: {alias}}}
  annotations:
    haproxy.router.openshift.io/timeout: "3600s"
    haproxy.router.openshift.io/timeout-tunnel: "3600s"
spec:
  host: {host}
  to: {{kind: Service, name: boxy-share-{alias}}}
  port: {{targetPort: share}}
  tls: {{termination: edge}}""",
    ]
    return (f"# boxy share {alias!r}: https://{host}/ -> relay port {relay_port}\n"
            + "\n---\n".join(docs) + "\n")


# ---- runtime orchestration (all binaries via env overrides -> CI shims) ----------


def _oc(args: list[str], *, stdin: str | None = None, timeout: float = 20) -> tuple[int, str]:
    try:
        p = subprocess.run([oc_bin(), *args], input=stdin, capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


class RelayExposer(Exposer):
    name = "relay"
    binary = "chisel"

    def available(self) -> bool:
        # chisel_bin() honors BOXY_CHISEL; shutil.which resolves an absolute shim
        # path (exists+executable) or a bare name on PATH — a non-existent override
        # correctly reports unavailable.
        return shutil.which(chisel_bin()) is not None

    # -- config resolution ---------------------------------------------------

    def _namespace(self) -> str:
        return os.environ.get(ENV_RELAY_NAMESPACE, DEFAULT_NAMESPACE)

    def _relay_url(self) -> str:
        env = os.environ.get(ENV_RELAY_URL, "").strip()
        if env:
            return env if env.startswith(("http://", "https://")) else f"https://{env}"
        rc, out = _oc(["get", "route", RELAY_APP, "-n", self._namespace(),
                       "-o", "jsonpath={.spec.host}"])
        if rc != 0 or not out:
            raise ExposeError(
                f"relay not found (set {ENV_RELAY_URL} or deploy it: "
                f"`boxy generate relay --host relay-boxy.apps.<cluster> | oc apply -f -`)")
        return f"https://{out}"

    def _credential(self) -> str:
        env = os.environ.get(ENV_RELAY_AUTH, "").strip()
        if env:
            return env
        rc, out = _oc(["get", "secret", RELAY_APP, "-n", self._namespace(),
                       "-o", "jsonpath={.data.auth}"])
        if rc != 0 or not out:
            raise ExposeError(f"relay credential unavailable (set {ENV_RELAY_AUTH}=user:pass "
                              f"or log in to oc — the Secret {RELAY_APP!r} is the source of truth)")
        try:
            return base64.b64decode(out).decode()
        except ValueError as e:
            raise ExposeError(f"could not decode the relay Secret: {e}") from None

    def _taken_ports(self) -> set[int]:
        rc, out = _oc(["get", "svc", "-n", self._namespace(), "-l", SHARE_LABEL,
                       "-o", "jsonpath={.items[*].spec.ports[0].targetPort}"])
        if rc != 0:
            return set()  # oc degraded: random pick + bind-retry still protects us
        return {int(tok) for tok in out.split() if tok.isdigit()}

    # -- chisel client lifecycle ----------------------------------------------

    def _client_argv(self, relay_url: str, relay_port: int, lport: int) -> list[str]:
        extra = shlex.split(os.environ.get(ENV_CHISEL_ARGS, ""))
        return [chisel_bin(), "client", "--keepalive", "25s", "--max-retry-count", "-1",
                *extra, relay_url, f"R:0.0.0.0:{relay_port}:127.0.0.1:{lport}"]

    def _start_client(self, relay_url: str, cred: str, relay_port: int, lport: int,
                      log_path) -> int:
        """Start the detached chisel client; return its pid. Raises ExposeError
        on an early bind failure (caller re-picks the port). The credential
        travels as the subprocess env AUTH — never argv, never disk."""
        argv = self._client_argv(relay_url, relay_port, lport)
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(argv, env={**os.environ, "AUTH": cred},
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)  # outlives boxy, like ControlPersist
        deadline = time.monotonic() + BIND_GRACE
        while time.monotonic() < deadline:
            if proc.poll() is not None:  # died this early = config/bind/auth error
                tail = _tail(log_path)
                if "address already in use" in tail:
                    raise _BindConflict()
                raise ExposeError(f"chisel client exited (rc {proc.returncode}): {tail[-300:]}")
            time.sleep(0.1)
        return proc.pid

    # -- the exposer contract --------------------------------------------------

    def is_live(self, record: dict) -> bool:
        return share_is_live(record)

    def expose(self, alias: str, lport: int, ctx: ShareContext | None = None) -> tuple[str, str]:
        from boxy import jobs

        if not self.available():
            raise ExposeError("chisel not installed — `brew install chisel-tunnel` "
                              "(NOT `brew install chisel`, which is Facebook's LLDB tool; "
                              "or: go install github.com/jpillora/chisel@latest)")
        relay_url = self._relay_url()
        host = share_host(alias, apps_domain_from_url(relay_url))
        url = f"https://{host}"
        ns = self._namespace()

        old = jobs.read_share(alias)
        if old and _pid_is_our_chisel(old.get("pid"), old.get("relay_port")):
            if old.get("lport") == lport:
                return f"{url}/v1", "already shared (reusing the live relay client)"
            _terminate(old["pid"])  # tunnel moved ports: re-expose, SAME public URL
        relay_port = old["relay_port"] if old else None

        cred = self._credential()
        log_path = jobs.share_log_path(alias)
        taken = self._taken_ports() if relay_port is None else set()
        pid = None
        for _ in range(3):
            port = relay_port if relay_port is not None else pick_relay_port(taken)
            try:
                pid = self._start_client(relay_url, cred, port, lport, log_path)
                relay_port = port
                break
            except _BindConflict:
                taken.add(port)
                relay_port = None
        if pid is None:
            raise ExposeError("could not bind a relay port after 3 attempts")

        yaml_text = emit_share_manifest(alias, host, relay_port, ns)
        rc, out = _oc(["apply", "-n", ns, "-f", "-"], stdin=yaml_text)
        note = "reachable by ANYONE on the corporate network; stop: boxy unshare " + alias
        if rc != 0:
            # oc degraded: keep the client alive, hand the user the YAML (router --emit philosophy)
            print(yaml_text)
            note = f"oc unavailable ({out.splitlines()[0] if out else 'not found'}) — apply the YAML above yourself"
        else:
            admitted = self._await_admission(alias, ns)
            if admitted is False:
                _oc(["delete", "route,svc", "-n", ns, "-l", f"{SHARE_LABEL}={alias}"])
                _terminate(pid)
                raise ExposeError(f"share name {alias!r} is taken on this cluster — pick another --share name")

        jobs.write_share(alias, {"alias": alias, "exposer": "relay", "url": url, "host": host,
                                 "relay_port": relay_port, "lport": lport,
                                 "namespace": ns, "pid": pid,
                                 "chisel_argv": self._client_argv(relay_url, relay_port, lport),
                                 "created": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return f"{url}/v1", note

    def _await_admission(self, alias: str, ns: str) -> bool | None:
        """True admitted, False rejected (host taken), None unknown (keep going)."""
        deadline = time.monotonic() + ADMIT_TIMEOUT
        while time.monotonic() < deadline:
            rc, out = _oc(["get", "route", f"boxy-share-{alias}", "-n", ns, "-o",
                           'jsonpath={.status.ingress[0].conditions[?(@.type=="Admitted")].status}'])
            if rc != 0:
                return None
            if out == "True":
                return True
            if out == "False":
                return False
            time.sleep(ADMIT_POLL)
        return None

    def unexpose(self, alias: str) -> None:
        from boxy import jobs

        record = jobs.read_share(alias)
        if not record:
            return
        pid = record.get("pid")
        if _pid_is_our_chisel(pid, record.get("relay_port")):
            _terminate(pid)
        ns = record.get("namespace", self._namespace())
        rc, out = _oc(["delete", "route,svc", "-n", ns, "-l", f"{SHARE_LABEL}={alias}"])
        if rc != 0:
            print(f"warning: could not delete the share's Route/Service — run:\n"
                  f"  oc delete route,svc -n {ns} -l {SHARE_LABEL}={alias}", file=sys.stderr)
        jobs.remove_share(alias)


class _BindConflict(Exception):
    """The chosen relay port was taken (raced another laptop) — re-pick."""


def _tail(path, nbytes: int = 2000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def share_is_live(record: dict) -> bool:
    """Is the share's detached chisel client still running? (For `boxy list`.)"""
    return _pid_is_our_chisel(record.get("pid"), record.get("relay_port"))


def _pid_is_our_chisel(pid, relay_port) -> bool:
    """PID-reuse guard before killing anything: the process must still look
    like OUR chisel client (portable to macOS via ps)."""
    if not pid:
        return False
    try:
        # -ww: unlimited width — else ps truncates to ~80 cols (no TTY) and drops
        # the relay_port from the tail of the command line.
        out = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "chisel" in out and (relay_port is None or str(relay_port) in out)


def _terminate(pid) -> None:
    # the client is started with start_new_session=True, so pid leads its own
    # process group — signal the whole group to reap any child (e.g. a wrapper).
    try:
        os.killpg(int(pid), 15)
    except (OSError, ValueError):
        try:
            os.kill(int(pid), 15)
        except (OSError, ValueError):
            pass
