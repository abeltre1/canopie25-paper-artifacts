# chart-headscale — Tier-2 naming authority on OpenShift

Self-hosted [Headscale](https://headscale.net) (open-source Tailscale control
server) as the MagicDNS authority behind `boxy open --publish NAME`. A model
served on HPC and published as `nemotron` becomes `https://nemotron.<baseDomain>`,
resolvable by **every teammate enrolled in the tailnet** — no corporate DNS.

## Install

```bash
SERVER_URL=https://headscale.apps.<cluster> BASE_DOMAIN=boxy.ts.net ./install.sh
# mint a reusable pre-auth key:
oc -n headscale exec deploy/headscale -- headscale preauthkeys create --reusable --user boxy
```

Or generate a no-Helm manifest with boxy and apply it directly:

```bash
boxy generate headscale --server-url https://headscale.apps.<cluster> \
  --base-domain boxy.ts.net --emit manifest | oc apply -f -
```

`boxy generate headscale --emit values` prints a ready-to-use `values.yaml`.

## OpenShift design notes

- **Route TLS = `edge`** (default) with `haproxy.router.openshift.io/timeout: 3600s`.
  headscale serves plain HTTP on :8080, so edge (router terminates TLS, forwards
  HTTP) works out of the box; the long timeout keeps the persistent Tailscale
  control connection alive (the 30s default kills it). `--set route.termination=reencrypt`
  is stronger (encrypted on every hop) but needs headscale serving TLS internally
  (a service-serving cert + `tls_*` config) — otherwise the Route health check fails.
- **DERP over the :443 Route by default** (`derp.udp.enabled=false`): peers that
  can't form a direct path relay over HTTPS — no UDP ingress, no extra cluster
  privileges. Set `derp.udp.enabled=true` to also expose STUN/3478 via a
  `LoadBalancer` for direct paths (needs a routable UDP ingress).
- **`runAsNonRoot`, no `NET_ADMIN`** — compatible with the `restricted-v2` SCC.
  Clients that run *inside* OpenShift should use `tailscaled --tun=userspace-networking`.

## Enroll a client (laptop / HPC login host)

```bash
tailscale up --login-server https://headscale.apps.<cluster> --authkey <preauth-key>
```

> No `helm` in the CI image, so templates are not `helm lint`-ed automatically —
> run `helm lint chart-headscale` and `helm template … | oc apply --dry-run` where
> Helm is available. The `boxy generate headscale --emit manifest` output IS
> validated as multi-doc YAML by the test suite.
