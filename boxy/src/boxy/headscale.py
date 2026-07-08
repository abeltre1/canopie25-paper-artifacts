"""Emit deployment artifacts for a self-hosted Headscale on OpenShift — the
Tier-2 naming authority behind `boxy open --publish` (SPEC §8d). Headscale is the
open-source Tailscale control server: it issues a tailnet with MagicDNS so a model
published as `nemotron` resolves to `https://nemotron.<base_domain>` for EVERY
enrolled teammate, with no corporate DNS request.

Pure string builders, exactly like `router.emit_nginx`/`sky_export` — no I/O and
no pyyaml dependency (boxy's only runtime dep is certifi). `emit_values` targets
the Helm chart under deploy/openshift/chart-headscale; `emit_manifest` is a
self-contained multi-doc YAML for `oc apply -f -` when Helm isn't available.

OpenShift specifics baked in (see SPEC §8d for the reasoning):
- Route TLS = **reencrypt** (control plane + the /ts2021 Noise handshake are
  HTTP(S)/TCP, so a Route carries them; reencrypt keeps server_url=https honest
  and the stream encrypted on every hop) with a long router timeout (the control
  connection is persistent; the 30s default would kill it).
- Embedded DERP relays over the same :443 Route by default (no UDP ingress, no
  extra cluster privileges); a UDP LoadBalancer for STUN is opt-in (derp_udp).
- runAsNonRoot, no NET_ADMIN — compatible with the restricted-v2 SCC.
"""

from __future__ import annotations

DEFAULT_IMAGE = "docker.io/headscale/headscale:0.23.0"
ROUTER_TIMEOUT = "3600s"  # the tailnet control connection is long-lived


def _config_yaml(server_url: str, base_domain: str, *, derp_udp: bool) -> str:
    """headscale config.yaml — the bit that makes it a MagicDNS authority."""
    return (
        f"server_url: {server_url}\n"
        "listen_addr: 0.0.0.0:8080\n"
        "metrics_listen_addr: 127.0.0.1:9090\n"
        "grpc_listen_addr: 127.0.0.1:50443\n"
        # default is /var/run/headscale, which OpenShift's non-root/arbitrary-UID
        # SCC can't create — put the socket on the writable PVC instead.
        "unix_socket: /var/lib/headscale/headscale.sock\n"
        "noise:\n"
        "  private_key_path: /var/lib/headscale/noise_private.key\n"
        "prefixes:\n"
        "  v4: 100.64.0.0/10\n"
        "database:\n"
        "  type: sqlite\n"
        "  sqlite:\n"
        "    path: /var/lib/headscale/db.sqlite\n"
        "dns:\n"
        "  magic_dns: true\n"
        f"  base_domain: {base_domain}\n"
        "  nameservers:\n"
        "    global:\n"
        "      - 1.1.1.1\n"
        "derp:\n"
        "  server:\n"
        "    enabled: true\n"
        "    region_id: 999\n"
        "    region_code: boxy\n"
        "    region_name: Boxy Embedded DERP\n"
        # REQUIRED when the embedded DERP is enabled: headscale writes this key on
        # first boot. Omitting it is fatal ('failed to save private key to disk at
        # path ""'). The PVC mounts /var/lib/headscale, so it's writable.
        "    private_key_path: /var/lib/headscale/derp_server_private.key\n"
        f"    stun_listen_addr: 0.0.0.0:3478\n"
        f"    # STUN/UDP {'exposed via a LoadBalancer' if derp_udp else 'not exposed; peers relay over the :443 Route'}\n"
        "  urls: []\n"
        "  auto_update_enabled: false\n"
    )


def emit_values(server_url: str, base_domain: str, preauth_key: str = "",
                *, image: str = DEFAULT_IMAGE, derp_udp: bool = False,
                termination: str = "edge") -> str:
    """values.yaml for the chart-headscale Helm chart. Route TLS defaults to
    `edge` — headscale serves plain HTTP on :8080, so edge (router terminates
    TLS, forwards HTTP) works out of the box. `reencrypt` is stronger but needs
    headscale to serve TLS internally (a service-serving cert + tls_* config)."""
    pk = preauth_key or '""'
    return (
        f"# Helm values for chart-headscale (Tier-2 naming authority on OpenShift).\n"
        f"# install:  helm install headscale ./chart-headscale \\\n"
        f"#             --set serverUrl={server_url} --set baseDomain={base_domain} --set preAuthKey=...\n"
        f"image: {image}\n"
        f"serverUrl: {server_url}\n"
        f"baseDomain: {base_domain}\n"
        f"magicDns: true\n"
        f"# reusable pre-auth key (headscale preauthkeys create --reusable --user boxy); "
        f"prefer --set over committing it\n"
        f"preAuthKey: {pk}\n"
        f"persistence:\n"
        f"  enabled: true\n"
        f"  size: 1Gi\n"
        f"  accessMode: ReadWriteOnce\n"
        f"derp:\n"
        f"  udp:\n"
        f"    enabled: {str(derp_udp).lower()}   # true -> LoadBalancer for STUN/3478; false -> relay over :443\n"
        f"route:\n"
        f"  enabled: true\n"
        f"  host: {_host_of(server_url)}\n"
        f"  termination: {termination}   # edge works with headscale's plain-HTTP :8080; reencrypt needs backend TLS\n"
        f"  timeout: {ROUTER_TIMEOUT}\n"
        f"securityContext:\n"
        f"  runAsNonRoot: true\n"
    )


def emit_manifest(server_url: str, base_domain: str, namespace: str = "headscale",
                  preauth_key: str = "", *, image: str = DEFAULT_IMAGE,
                  derp_udp: bool = False, termination: str = "edge") -> str:
    """A self-contained multi-doc manifest for `oc apply -f -` (no Helm). Route TLS
    defaults to `edge` (works with headscale's plain-HTTP :8080); `reencrypt` needs
    headscale serving TLS internally."""
    host = _host_of(server_url)
    cfg = _indent(_config_yaml(server_url, base_domain, derp_udp=derp_udp), 4)
    docs = [
        # ConfigMap: the headscale config.yaml
        f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: headscale-config
  namespace: {namespace}
data:
  config.yaml: |
{cfg}""",
        # Secret: the reusable pre-auth key (empty by default; set it out-of-band)
        f"""apiVersion: v1
kind: Secret
metadata:
  name: headscale-preauth
  namespace: {namespace}
type: Opaque
stringData:
  preauth-key: {preauth_key or '""'}""",
        # PVC: SQLite DB + noise key
        f"""apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: headscale-data
  namespace: {namespace}
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 1Gi""",
        # Deployment: single replica, non-root, no NET_ADMIN
        f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: headscale
  namespace: {namespace}
  labels: {{app: headscale}}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: headscale}}
  template:
    metadata:
      labels: {{app: headscale}}
    spec:
      securityContext:
        runAsNonRoot: true
      containers:
        - name: headscale
          image: {image}
          args: ["serve"]
          ports:
            - {{containerPort: 8080, name: http}}
            - {{containerPort: 3478, name: stun, protocol: UDP}}
          volumeMounts:
            - {{name: config, mountPath: /etc/headscale}}
            - {{name: data, mountPath: /var/lib/headscale}}
      volumes:
        - {{name: config, configMap: {{name: headscale-config}}}}
        - {{name: data, persistentVolumeClaim: {{claimName: headscale-data}}}}""",
        # Service
        f"""apiVersion: v1
kind: Service
metadata:
  name: headscale
  namespace: {namespace}
spec:
  selector: {{app: headscale}}
  ports:
    - {{name: http, port: 8080, targetPort: 8080}}""",
        # Route: reencrypt, long timeout for the persistent control connection
        f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: headscale
  namespace: {namespace}
  annotations:
    haproxy.router.openshift.io/timeout: "{ROUTER_TIMEOUT}"
spec:
  host: {host}
  to: {{kind: Service, name: headscale}}
  port: {{targetPort: http}}
  tls: {{termination: {termination}}}""",
    ]
    if derp_udp:
        docs.append(
            f"""apiVersion: v1
kind: Service
metadata:
  name: headscale-derp-udp
  namespace: {namespace}
spec:
  type: LoadBalancer
  selector: {{app: headscale}}
  ports:
    - {{name: stun, port: 3478, targetPort: 3478, protocol: UDP}}""")
    header = (f"# Headscale on OpenShift — Tier-2 naming authority for `boxy open --publish`.\n"
              f"# apply:  oc new-project {namespace} 2>/dev/null; boxy generate headscale "
              f"--server-url {server_url} --base-domain {base_domain} --emit manifest | oc apply -f -\n"
              f"# then:   oc exec deploy/headscale -- headscale preauthkeys create --reusable --user boxy\n")
    return header + "\n---\n".join(docs) + "\n"


def _host_of(server_url: str) -> str:
    host = server_url.strip()
    for pre in ("https://", "http://"):
        if host.startswith(pre):
            host = host[len(pre):]
    return host.split("/", 1)[0].split(":", 1)[0]


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "".join(f"{pad}{line}" if line.strip() else line
                   for line in text.splitlines(keepends=True))
