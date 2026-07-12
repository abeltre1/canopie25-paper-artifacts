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

from boxy.exposers.base import ExposeError, Exposer

RELAY_IMAGE = "docker.io/jpillora/chisel:1.10"
RELAY_PORT_RANGE = range(31000, 32000)  # per-share reverse ports; ACL-able as R:31[0-9]{3}
RELAY_APP = "boxy-relay"                # Deployment/Service/Route/Secret name + selector
SHARE_LABEL = "boxy.share"              # every per-alias object carries this for teardown

# Zero-install client mode: run `chisel client` inside a container so nothing is
# installed on the laptop/login node. podman/docker use a DETACHED named container
# (conmon owns its lifecycle, independent of boxy); apptainer — which HPC sites
# ship instead — shares the host network namespace by default, so it rides the
# same detached-process path as a host chisel binary (pid-based liveness).
CLIENT_CONTAINER = "podman", "docker"       # detached, name-tracked runtimes
CLIENT_RUNTIMES = ("podman", "docker", "apptainer")  # autodetect preference order
CLIENT_NAME_PREFIX = "boxy-chisel-"         # per-alias container name

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
    from boxy import config

    return config.get("binaries.chisel")


def oc_bin() -> str:
    from boxy import config

    return config.get("binaries.oc")


def relay_port_range() -> range:
    from boxy import config

    return range(config.get_int("relay.port_min"), config.get_int("relay.port_max"))


# ---- zero-install client-in-a-container helpers ---------------------------------


def _first_runtime() -> str:
    """First container runtime on PATH, in preference order (podman > docker >
    apptainer). "" when none is present. Cheap (no probe) — matches the family's
    available() idiom; a broken runtime surfaces its real error when we run it."""
    for rt in CLIENT_RUNTIMES:
        if shutil.which(rt):
            return rt
    return ""


def host_loopback(runtime: str) -> str:
    """Where the container must dial to reach the laptop/login-node loopback that
    `ssh -L` is listening on. Linux (podman/docker --network=host, and apptainer's
    default shared netns): the container's 127.0.0.1 IS the host's, so 127.0.0.1.
    macOS: the runtime is a Linux VM, so 127.0.0.1 is the VM — reach the real host
    via the runtime's magic DNS name (host.docker.internal / host.containers.internal).
    Same platform fix as backends/podman.network_args()."""
    if sys.platform == "darwin" and runtime in CLIENT_CONTAINER:
        return "host.docker.internal" if runtime == "docker" else "host.containers.internal"
    return "127.0.0.1"


def relay_image() -> str:
    """The chisel image for the client container — the SAME image the OpenShift
    relay server runs (config images.relay), so one override mirrors both. Resolved
    through registries.py: set images.relay (BOXY_RELAY_IMAGE / [images].relay) to a
    site mirror when Docker Hub is blocked, and air-gapped pulls just work."""
    from boxy import config, registries

    return registries.resolve_image(config.get("images.relay"))


def _run_container(argv: list[str], *, timeout: float = 30, env: dict | None = None) -> tuple[int, str]:
    """Run a one-shot container command (run -d / inspect / logs / rm), capturing
    combined output. `env` (for the AUTH-by-name credential) rides the subprocess
    env, never argv. Never raises — a missing/broken runtime returns (127, msg),
    exactly like _oc()."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 127, str(e)


def _container_status(runtime: str, name: str) -> str:
    """The container's State.Status ('running'|'exited'|'created'|...); "" if the
    container is gone or inspect failed."""
    rc, out = _run_container([runtime, "inspect", "--format", "{{.State.Status}}", name], timeout=10)
    return out.strip().lower() if rc == 0 else ""


def _container_running(runtime: str, name: str) -> bool:
    return _container_status(runtime, name) == "running"


def _container_logs(runtime: str, name: str) -> str:
    rc, out = _run_container([runtime, "logs", name], timeout=10)
    return out if rc == 0 else ""


# ---- pure helpers ---------------------------------------------------------------


def apps_domain_from_url(relay_url: str) -> str:
    """`https://relay-boxy.apps.apps.cluster.example.com` -> `apps.apps.cluster.example.com`.
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
    port_range = relay_port_range()
    free = [p for p in port_range if p not in taken]
    if not free:
        raise ExposeError(f"all {len(port_range)} relay ports are taken — "
                          f"`boxy unshare` stale shares or widen relay.port_min/max")
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
              f"# Deploy ONCE per cluster; teammates then reach shares at https://<name>-boxy.{host.split('.', 1)[-1] if '.' in host else 'apps.<cluster>'}/ with NOTHING installed.\n"
              f"# apply:  oc new-project {namespace} 2>/dev/null; "
              f"boxy generate relay --host {host} | oc apply -f -\n" + hint +
              "# then on the laptop/login node (ZERO install — chisel runs in a container):\n"
              "#   boxy open <inst> --ssh <login> --share <name>\n"
              "#   (needs only podman/docker/apptainer; set relay.client_mode=host to use a chisel binary)\n")
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


def relay_admission(namespace: str = "") -> tuple[str, str]:
    """Doctor helper (read-only): is the shared relay Route admitted on this
    cluster? Returns (status, detail) with status in {'ok', 'missing', 'rejected',
    'no-oc', 'unknown'} — so `boxy doctor` can report 'deploy once, share forever'
    readiness without a second oc-query implementation."""
    from boxy import config

    ns = namespace or config.get("relay.namespace")
    if shutil.which(oc_bin()) is None:
        return "no-oc", "oc not on PATH — relay server not checked from here"
    rc, host = _oc(["get", "route", RELAY_APP, "-n", ns, "-o", "jsonpath={.spec.host}"])
    if rc != 0 or not host:
        return "missing", f"no {RELAY_APP!r} Route in namespace {ns!r}"
    rc, admitted = _oc(["get", "route", RELAY_APP, "-n", ns, "-o",
                        'jsonpath={.status.ingress[0].conditions[?(@.type=="Admitted")].status}'])
    if admitted == "True":
        return "ok", f"https://{host} admitted (ns {ns})"
    if admitted == "False":
        return "rejected", f"Route {RELAY_APP!r} host {host} NOT admitted (ns {ns})"
    return "unknown", f"Route {RELAY_APP!r} present (host {host}); admission status unknown"


class RelayExposer(Exposer):
    name = "relay"
    binary = "chisel"

    def available(self) -> bool:
        # Zero-install: `--share` works with EITHER a host chisel binary OR any
        # container runtime (podman/docker/apptainer), since we can run the chisel
        # client in a container. chisel_bin() honors BOXY_CHISEL; shutil.which
        # resolves an absolute shim path or a bare name on PATH.
        if shutil.which(chisel_bin()) is not None:
            return True
        return _first_runtime() != ""

    def _resolve_client_mode(self) -> tuple[str, str]:
        """(mode, launcher). mode: 'host' (a chisel binary, pid-tracked) |
        'container' (podman/docker detached & name-tracked, or apptainer's shared-
        netns process, pid-tracked). launcher is the concrete tool. Driven by
        config relay.client_mode: host | container | auto (default). auto prefers a
        host chisel (no image pull) and falls back to a container runtime."""
        from boxy import config

        want = config.get_str("relay.client_mode").strip().lower()
        have_chisel = shutil.which(chisel_bin()) is not None
        runtime = _first_runtime()
        if want == "host":
            if not have_chisel:
                raise ExposeError(
                    "relay.client_mode=host but no chisel binary on PATH "
                    "(install chisel-tunnel, or use relay.client_mode=container/auto)")
            return "host", "chisel"
        if want == "container":
            if not runtime:
                raise ExposeError(
                    "relay.client_mode=container but no container runtime found "
                    f"(looked for {', '.join(CLIENT_RUNTIMES)})")
            return "container", runtime
        # auto
        if have_chisel:
            return "host", "chisel"
        if runtime:
            return "container", runtime
        raise ExposeError(_no_client_hint())

    # -- config resolution ---------------------------------------------------

    def _namespace(self) -> str:
        from boxy import config

        return config.get("relay.namespace")

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

    def _reverse_spec(self, relay_port: int, loop: str, lport: int) -> str:
        # relay server binds relay_port and forwards back to <loop>:<lport> as the
        # CLIENT sees it — the laptop/login-node loopback where ssh -L listens.
        return f"R:0.0.0.0:{relay_port}:{loop}:{lport}"

    def _client_argv(self, relay_url: str, relay_port: int, lport: int) -> list[str]:
        """Host chisel BINARY argv. The binary runs directly on the host, so its
        loopback IS the ssh -L endpoint (127.0.0.1)."""
        extra = shlex.split(os.environ.get(ENV_CHISEL_ARGS, ""))
        return [chisel_bin(), "client", "--keepalive", "25s", "--max-retry-count", "-1",
                *extra, relay_url, self._reverse_spec(relay_port, "127.0.0.1", lport)]

    def _container_argv(self, runtime: str, image: str, cname: str,
                        relay_url: str, relay_port: int, lport: int) -> list[str]:
        """Detached chisel CLIENT container (podman/docker) — zero host install.
        --network=host on Linux so the container's loopback is the host's (where
        ssh -L listens); on macOS the runtime is a VM, so no --network=host and the
        reverse spec dials host.*.internal instead. AUTH is passed BY NAME
        (`--env AUTH`) so the credential rides the process env, never argv (so it
        never shows in `podman inspect`/`ps`). No --rm: an early-exiting container
        must survive long enough to read its logs (bind-conflict detection)."""
        extra = shlex.split(os.environ.get(ENV_CHISEL_ARGS, ""))
        net: list[str] = [] if sys.platform == "darwin" else ["--network=host"]
        spec = self._reverse_spec(relay_port, host_loopback(runtime), lport)
        return [runtime, "run", "-d", "--name", cname, *net, "--env", "AUTH",
                image, "client", "--keepalive", "25s", "--max-retry-count", "-1",
                *extra, relay_url, spec]

    def _apptainer_argv(self, image: str, relay_url: str, relay_port: int, lport: int) -> list[str]:
        """chisel client via apptainer (what HPC sites ship). apptainer shares the
        host network namespace by default, so 127.0.0.1 reaches ssh -L; it runs the
        OCI image directly (docker://…). AUTH rides APPTAINERENV_AUTH (env, not argv)."""
        extra = shlex.split(os.environ.get(ENV_CHISEL_ARGS, ""))
        ref = image if "://" in image else f"docker://{image}"
        return ["apptainer", "run", ref, "client", "--keepalive", "25s",
                "--max-retry-count", "-1", *extra, relay_url,
                self._reverse_spec(relay_port, host_loopback("apptainer"), lport)]

    def _launch_client(self, launcher: str, relay_url: str, cred: str, relay_port: int,
                       lport: int, log_path, alias: str) -> dict:
        """Start the relay client via `launcher` and return the share-record fields
        that describe it: name-tracked for podman/docker (a detached container),
        pid-tracked for host chisel / apptainer (a detached process). The
        credential rides the child's env (AUTH / APPTAINERENV_AUTH), never argv.
        Raises _BindConflict when the relay reverse port is taken (caller re-picks)."""
        if launcher in CLIENT_CONTAINER:
            image = relay_image()
            cname = CLIENT_NAME_PREFIX + alias
            self._start_client_container(launcher, image, cname, cred, relay_url, relay_port, lport)
            return {"client": launcher, "container": cname, "image": image}
        if launcher == "apptainer":
            argv = self._apptainer_argv(relay_image(), relay_url, relay_port, lport)
            pid = self._start_client_process(argv, {**os.environ, "APPTAINERENV_AUTH": cred}, log_path)
            return {"client": "apptainer", "pid": pid, "chisel_argv": argv}
        argv = self._client_argv(relay_url, relay_port, lport)
        pid = self._start_client_process(argv, {**os.environ, "AUTH": cred}, log_path)
        return {"client": "chisel", "pid": pid, "chisel_argv": argv}

    def _start_client_process(self, argv: list[str], env: dict, log_path) -> int:
        """Start a detached client PROCESS (host chisel binary or apptainer); return
        its pid. Raises _BindConflict / ExposeError on an early exit."""
        with open(log_path, "ab") as log:
            proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
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

    def _start_client_container(self, runtime: str, image: str, cname: str, cred: str,
                                relay_url: str, relay_port: int, lport: int) -> str:
        """Start the detached chisel client CONTAINER; return its name. Pre-cleans a
        stale same-named container (idempotent restart), then watches briefly for an
        early exit — a taken relay reverse port -> _BindConflict (caller re-picks)."""
        _run_container([runtime, "rm", "-f", cname])  # idempotent restart
        argv = self._container_argv(runtime, image, cname, relay_url, relay_port, lport)
        rc, out = _run_container(argv, env={**os.environ, "AUTH": cred})
        if rc != 0:
            raise ExposeError(f"could not start the chisel client container "
                              f"(runtime {runtime}, image {image}): {out[-300:]}")
        deadline = time.monotonic() + BIND_GRACE
        while time.monotonic() < deadline:
            if _container_status(runtime, cname) in ("exited", "dead"):
                logs = _container_logs(runtime, cname)
                _run_container([runtime, "rm", "-f", cname])
                if "address already in use" in logs:
                    raise _BindConflict()
                raise ExposeError(f"chisel client container exited: {(logs or out)[-300:]}")
            time.sleep(0.1)
        return cname

    # -- the exposer contract --------------------------------------------------

    def expose(self, alias: str, lport: int) -> tuple[str, str]:
        from boxy import jobs

        if not self.available():
            raise ExposeError(_no_client_hint())
        mode, launcher = self._resolve_client_mode()
        relay_url = self._relay_url()
        host = share_host(alias, apps_domain_from_url(relay_url))
        url = f"https://{host}"
        ns = self._namespace()

        old = jobs.read_share(alias)
        if old and share_is_live(old):
            if old.get("lport") == lport:
                return f"{url}/v1", "already shared (reusing the live relay client)"
            _stop_client(old)  # tunnel moved ports: re-expose, SAME public URL
        relay_port = old["relay_port"] if old else None

        cred = self._credential()
        log_path = jobs.share_log_path(alias)
        taken = self._taken_ports() if relay_port is None else set()
        client: dict | None = None
        for _ in range(3):
            port = relay_port if relay_port is not None else pick_relay_port(taken)
            try:
                client = self._launch_client(launcher, relay_url, cred, port, lport, log_path, alias)
                relay_port = port
                break
            except _BindConflict:
                taken.add(port)
                relay_port = None
        if client is None:
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
                _stop_client(client)
                raise ExposeError(f"share name {alias!r} is taken on this cluster — pick another --share name")

        record = {"alias": alias, "url": url, "host": host,
                  "relay_port": relay_port, "lport": lport,
                  "namespace": ns, "client_mode": mode,
                  "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **client}
        jobs.write_share(alias, record)
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
        _stop_client(record)  # rm -f the container, or SIGTERM the process group
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
    """Is the share's detached relay client still running? (For `boxy list`.)
    Container-tracked shares (podman/docker) check the container's state; process-
    tracked shares (host chisel binary / apptainer) check the pid."""
    cname = record.get("container")
    if cname:
        return _container_running(record.get("client", "podman"), cname)
    return _pid_is_our_chisel(record.get("pid"), record.get("relay_port"))


def _stop_client(record: dict) -> None:
    """Stop a share's relay client — a detached CONTAINER (`<rt> rm -f <name>`) or a
    detached PROCESS (SIGTERM the group, PID-reuse-guarded). Idempotent; safe on a
    partial record and on either client mode."""
    cname = record.get("container")
    if cname:
        _run_container([record.get("client", "podman"), "rm", "-f", cname])
        return
    pid = record.get("pid")
    if _pid_is_our_chisel(pid, record.get("relay_port")):
        _terminate(pid)


def _no_client_hint() -> str:
    """One message covering BOTH zero-install (container) and binary install paths."""
    return ("no way to run the relay client: no chisel binary on PATH and no container "
            "runtime (podman/docker/apptainer). Easiest is ZERO install — install podman "
            "(or docker) and boxy runs `chisel client` in a container for you "
            "(relay.client_mode=auto). Or install the binary: `brew install chisel-tunnel` "
            "(NOT `brew install chisel`, Facebook's LLDB tool; or "
            "`go install github.com/jpillora/chisel@latest`).")


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
