"""OpenSSH gateway exposer — the everyone-URL for a boxy tunnel with NO
third-party tunnel binary (RUNBOOK §0.994). The replacement for the chisel
`relay` when cybersecurity won't allow an unfamiliar tunneling tool.

Where the `relay` exposer bridges the LAPTOP loopback (a chisel client dialing
out from your Mac), the `gateway` exposer moves boxy's own `ssh -L` forward into
a pod: a pod running only OpenSSH (a Red Hat UBI image) dials OUTBOUND to the
HPC login node — the exact front door the laptop already uses — and forwards the
model port. The laptop drops out of the data path entirely, so the share
survives the laptop sleeping or shutting down.

Data path:  teammate -> HTTPS Route -> Service -> gateway pod -> ssh -L -> login
node -> compute node:port -> the model. Name resolution rides the cluster's
EXISTING `*.apps.<cluster>` wildcard DNS, so teammates need nothing installed.

Per-share objects (Deployment + Service + Route) are created at share time by
oc apply, labelled for one-shot teardown. The one-time cluster prerequisites
(namespace, the ssh-client image, the login-node key Secret, an egress
NetworkPolicy) are emitted by `boxy generate gateway` and applied once by an
admin — see emit_setup_manifest.

All binaries resolve through env overrides (BOXY_OC) so the whole flow is
CI-testable with shims, exactly like the relay exposer."""

from __future__ import annotations

import os
import shlex

from boxy.exposers.base import ExposeError, Exposer, ShareContext
from boxy.exposers.relay import _oc, oc_bin, share_host  # shared OpenShift plumbing

GW_APP = "boxy-gw"                       # Deployment/Service/Route name prefix + selector base
SHARE_LABEL = "boxy.share"               # every per-alias object carries this for teardown
DEFAULT_NAMESPACE = "boxy-gw"
# UBI + openssh-clients, mirrored into the cluster's OWN registry (pulls natively,
# so no Docker Hub / Zscaler-403 like the chisel image hit). Build: see
# emit_setup_manifest's header.
DEFAULT_IMAGE = "image-registry.openshift-image-registry.svc:5000/boxy-gw/boxy-gw:1"
DEFAULT_SECRET = "boxy-gw-ssh"           # holds id_ed25519 (0400) + known_hosts
KEY_MOUNT = "/keys"

ENV_NAMESPACE = "BOXY_GW_NAMESPACE"
ENV_LOGIN = "BOXY_GW_LOGIN"              # the pod's ssh target, e.g. svcacct@hops.sandia.gov
ENV_USER = "BOXY_GW_USER"               # service account (default 'boxy') if LOGIN unset
ENV_APPS_DOMAIN = "BOXY_GW_APPS_DOMAIN"  # e.g. apps.goodall.sandia.gov (wildcard the URL rides)
ENV_IMAGE = "BOXY_GW_IMAGE"
ENV_SECRET = "BOXY_GW_SECRET"

ADMIT_TIMEOUT = 10.0
ADMIT_POLL = 0.5


# ---- pure helpers ---------------------------------------------------------------


def _host_part(ssh_target: str) -> str:
    """`ambelt@hops` -> `hops`; `hops` -> `hops`. The login node's address is the
    same one the laptop reaches; only the user differs (a service account)."""
    return ssh_target.rsplit("@", 1)[-1].strip()


def login_target(env_login: str, service_user: str, ssh_host: str) -> str:
    """Resolve the pod's ssh target. An explicit BOXY_GW_LOGIN wins; otherwise
    reuse the login node from the user's --ssh but swap in the service account
    (a pod can't complete the human's OTP, so it authenticates as its own key-
    based functional account)."""
    if env_login.strip():
        return env_login.strip()
    host = _host_part(ssh_host)
    if not host:
        raise ExposeError(
            f"gateway: cannot resolve the login node — set {ENV_LOGIN}=<svcuser>@<login-host> "
            f"(no --ssh host to fall back on)")
    return f"{service_user}@{host}"


def _ssh_command(login: str, node: str, remote_port: int, listen_port: int) -> str:
    """The pod's container command: copy the mounted key somewhere the (random
    OpenShift uid) process owns so OpenSSH accepts its permissions, then hold a
    reverse-nothing `ssh -L` forward open, self-healing on drop. The private key
    never reaches argv; the login host key is PINNED via the mounted known_hosts
    (StrictHostKeyChecking=yes) so a MITM can't redirect the tunnel."""
    ssh = (
        f"ssh -NT -i /tmp/gwkey "
        f"-o UserKnownHostsFile={KEY_MOUNT}/known_hosts -o StrictHostKeyChecking=yes "
        f"-o ExitOnForwardFailure=yes -o ServerAliveInterval=25 -o ServerAliveCountMax=3 "
        f"-L 0.0.0.0:{listen_port}:{node}:{remote_port} {shlex.quote(login)}"
    )
    return (
        f"cp {KEY_MOUNT}/id_ed25519 /tmp/gwkey && chmod 400 /tmp/gwkey; "
        f"while true; do {ssh}; "
        f"echo 'boxy-gw: tunnel dropped, retrying in 5s' >&2; sleep 5; done"
    )


def emit_share_manifest(alias: str, host: str, login: str, node: str, remote_port: int,
                        *, namespace: str = DEFAULT_NAMESPACE, image: str = DEFAULT_IMAGE,
                        secret: str = DEFAULT_SECRET, listen_port: int = 0) -> str:
    """One share = a Deployment (the ssh pod) + Service + Route. The ssh target
    (login/node/port) is baked in here because boxy only learns the compute node
    once the job is READY. Labelled `boxy.share=<alias>` for one-shot teardown."""
    lport = listen_port or remote_port
    app = f"{GW_APP}-{alias}"
    cmd = _ssh_command(login, node, remote_port, lport)
    docs = [
        f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app}
  namespace: {namespace}
  labels: {{app: {GW_APP}, {SHARE_LABEL}: {alias}}}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: {app}}}
  template:
    metadata:
      labels: {{app: {app}, {SHARE_LABEL}: {alias}}}
    spec:
      securityContext:
        runAsNonRoot: true
      containers:
        - name: ssh
          image: {image}
          command: ["/bin/sh", "-c", {_yaml_dq(cmd)}]
          ports:
            - {{containerPort: {lport}, name: model}}
          volumeMounts:
            - {{name: key, mountPath: {KEY_MOUNT}, readOnly: true}}
      volumes:
        - name: key
          secret:
            secretName: {secret}
            defaultMode: 0400""",
        f"""apiVersion: v1
kind: Service
metadata:
  name: {app}
  namespace: {namespace}
  labels: {{app: {GW_APP}, {SHARE_LABEL}: {alias}}}
spec:
  selector: {{app: {app}}}
  ports:
    - {{name: model, port: {lport}, targetPort: {lport}}}""",
        f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: {app}
  namespace: {namespace}
  labels: {{app: {GW_APP}, {SHARE_LABEL}: {alias}}}
  annotations:
    haproxy.router.openshift.io/timeout: "3600s"
spec:
  host: {host}
  to: {{kind: Service, name: {app}}}
  port: {{targetPort: model}}
  tls: {{termination: edge}}""",
    ]
    return (f"# boxy gateway share {alias!r}: https://{host}/ -> ssh -L {node}:{remote_port} via {login}\n"
            + "\n---\n".join(docs) + "\n")


def emit_setup_manifest(login: str, *, namespace: str = DEFAULT_NAMESPACE,
                        secret: str = DEFAULT_SECRET, image: str = DEFAULT_IMAGE) -> str:
    """The ONE-TIME cluster prerequisites (apply once, admin): the key Secret
    (placeholder) + an egress NetworkPolicy to the login node:22. The ssh-client
    image build and the login-node forced-command authorized_keys line are
    manual steps described in the header (they can't be YAML)."""
    login_host = _host_part(login)
    header = (
        f"# boxy gateway — ONE-TIME cluster setup (OpenSSH-only everyone-URL, no tunnel binary).\n"
        f"# 1) build the ssh-client image (native pull, no Docker Hub) and push to the internal registry:\n"
        f"#      printf 'FROM registry.access.redhat.com/ubi9/ubi\\nRUN dnf -y install openssh-clients "
        f"&& dnf clean all\\n' > Dockerfile.gw\n"
        f"#      oc new-project {namespace} 2>/dev/null; oc registry login\n"
        f"#      podman build -t {image} -f Dockerfile.gw . && podman push {image}\n"
        f"# 2) create the login-node key Secret (the private half the pod uses; pin the host key):\n"
        f"#      ssh-keygen -t ed25519 -N '' -f ./gw_key -C boxy-gw\n"
        f"#      oc -n {namespace} create secret generic {secret} \\\n"
        f"#        --from-file=id_ed25519=./gw_key "
        f"--from-file=known_hosts=<(ssh-keyscan {login_host or '<login-host>'})\n"
        f"# 3) on the login node, authorize the key locked to ONE forward (OTP-exempt functional acct):\n"
        f'#      command="",restrict,permitopen="*:*" ssh-ed25519 <contents of gw_key.pub>\n'
        f"#      (narrow permitopen to your compute partition, e.g. permitopen=\"hops*:8090\")\n"
        f"# 4) apply this file (Secret placeholder is overwritten by step 2; NetworkPolicy allows egress):\n"
    )
    docs = [
        f"""apiVersion: v1
kind: Secret
metadata:
  name: {secret}
  namespace: {namespace}
  labels: {{app: {GW_APP}}}
type: Opaque
stringData:
  id_ed25519: REPLACE_ME_WITH_oc_create_secret_STEP_2
  known_hosts: REPLACE_ME_WITH_ssh-keyscan_OUTPUT""",
        f"""apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: {GW_APP}-egress-login
  namespace: {namespace}
  labels: {{app: {GW_APP}}}
spec:
  podSelector:
    matchLabels: {{app: {GW_APP}}}
  policyTypes: [Egress]
  egress:
    - ports:
        - {{protocol: TCP, port: 22}}""",
    ]
    footer = (f"\n# then share:  boxy serve MODEL --ssh <you>@{login_host or '<login>'} "
              f"--share NAME --exposer gateway\n")
    return header + "\n---\n".join(docs) + "\n" + footer


def _yaml_dq(value: str) -> str:
    """Double-quoted YAML scalar for the shell command (has spaces/colons)."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---- runtime orchestration ------------------------------------------------------


class GatewayExposer(Exposer):
    name = "gateway"
    binary = ""  # nothing on the LAPTOP — the ssh client runs in the pod; oc does the work

    def available(self) -> bool:
        # the only laptop-side dependency is a logged-in `oc`; probe it like the
        # relay probes chisel (honors BOXY_OC -> a CI shim).
        import shutil
        return shutil.which(oc_bin()) is not None

    def _namespace(self) -> str:
        return os.environ.get(ENV_NAMESPACE, DEFAULT_NAMESPACE)

    def _apps_domain(self) -> str:
        dom = os.environ.get(ENV_APPS_DOMAIN, "").strip()
        if not dom:
            raise ExposeError(
                f"gateway: set {ENV_APPS_DOMAIN}=apps.<cluster> (the wildcard your shared URL "
                f"rides, e.g. apps.goodall.sandia.gov) — corporate DNS already resolves it")
        return dom

    def expose(self, alias: str, lport: int, ctx: ShareContext | None = None) -> tuple[str, str]:
        from boxy import jobs

        if not self.available():
            raise ExposeError("gateway: `oc` not found — log in to the cluster (oc login ...) "
                              "or set BOXY_OC to your oc binary")
        ctx = ctx or ShareContext()
        if not ctx.node or not ctx.remote_port:
            raise ExposeError("gateway needs the cluster-side address — use it with `--ssh` on "
                              "serve/open so boxy knows the compute node and port to forward")
        ns = self._namespace()
        host = share_host(alias, self._apps_domain())  # validates the alias, adds -boxy suffix
        url = f"https://{host}"
        login = login_target(os.environ.get(ENV_LOGIN, ""),
                             os.environ.get(ENV_USER, "boxy"), ctx.ssh_host)
        image = os.environ.get(ENV_IMAGE, DEFAULT_IMAGE)
        secret = os.environ.get(ENV_SECRET, DEFAULT_SECRET)

        # replace-on-rerun: drop a previous share of the same name first (idempotent,
        # SAME public URL) so a moved job cleanly re-points.
        _oc(["delete", "deploy,route,svc", "-n", ns, "-l", f"{SHARE_LABEL}={alias}"])

        yaml_text = emit_share_manifest(alias, host, login, ctx.node, ctx.remote_port,
                                        namespace=ns, image=image, secret=secret)
        rc, out = _oc(["apply", "-n", ns, "-f", "-"], stdin=yaml_text)
        if rc != 0:
            # oc degraded: hand the user the manifest (router --emit philosophy)
            print(yaml_text)
            raise ExposeError(f"oc apply failed ({out.splitlines()[0] if out else 'unknown'}) — "
                              f"apply the manifest above once oc works, or check `oc login`")

        note = ("served from a pod (laptop can disconnect); reachable by ANYONE on the corporate "
                "network; stop: boxy unshare " + alias)
        if self._await_admission(alias, ns) is False:
            _oc(["delete", "deploy,route,svc", "-n", ns, "-l", f"{SHARE_LABEL}={alias}"])
            raise ExposeError(f"share name {alias!r} is taken on this cluster — pick another --share name")

        jobs.write_share(alias, {"alias": alias, "exposer": "gateway", "url": url, "host": host,
                                 "namespace": ns, "login": login, "node": ctx.node,
                                 "remote_port": ctx.remote_port, "image": image,
                                 "created": _timestamp()})
        return f"{url}/v1", note

    def _await_admission(self, alias: str, ns: str) -> bool | None:
        import time
        deadline = time.monotonic() + ADMIT_TIMEOUT
        while time.monotonic() < deadline:
            rc, out = _oc(["get", "route", f"{GW_APP}-{alias}", "-n", ns, "-o",
                           'jsonpath={.status.ingress[0].conditions[?(@.type=="Admitted")].status}'])
            if rc != 0:
                return None
            if out == "True":
                return True
            if out == "False":
                return False
            time.sleep(ADMIT_POLL)
        return None

    def is_live(self, record: dict) -> bool:
        ns = record.get("namespace", DEFAULT_NAMESPACE)
        alias = record.get("alias", "")
        rc, out = _oc(["get", "deploy", f"{GW_APP}-{alias}", "-n", ns,
                       "-o", "jsonpath={.status.readyReplicas}"])
        if rc != 0:
            return True  # oc degraded / not logged in: can't disprove liveness
        return out.strip() not in ("", "0")

    def unexpose(self, alias: str) -> None:
        import sys

        from boxy import jobs

        record = jobs.read_share(alias)
        ns = (record or {}).get("namespace", self._namespace())
        rc, out = _oc(["delete", "deploy,route,svc", "-n", ns, "-l", f"{SHARE_LABEL}={alias}"])
        if rc != 0 and record is not None:
            print(f"warning: could not delete the gateway objects — run:\n"
                  f"  oc delete deploy,route,svc -n {ns} -l {SHARE_LABEL}={alias}", file=sys.stderr)
        jobs.remove_share(alias)


def _timestamp() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")
