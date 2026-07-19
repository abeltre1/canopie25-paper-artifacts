# DEMO — zero-install team sharing with chisel-tunnel (Phase 1)

Turn a boxy-served model's **local** `/v1` endpoint into a **network-accessible**
URL through an OpenShift relay — with **nothing installed** on either end:

- **Teammates** open `https://<name>-boxy.apps.<cluster>/` in a browser or `curl`
  it. No client software, no DNS setup, no enrollment (rides the cluster's
  existing `*.apps` wildcard DNS).
- **You** (the sharer, on a Mac or an HPC login node) run the `chisel client`
  **inside a container** — no `brew install`, only a container runtime.

```
 model on a compute node ──ssh -L──▶ 127.0.0.1:8090 on your laptop
                                          │
                    chisel client CONTAINER (podman/docker/apptainer)
                                          │  outbound wss over the edge Route (traverses Zscaler)
                                          ▼
             chisel server on OpenShift  ──▶  https://<name>-boxy.apps.<cluster>/v1
                                                (anyone on the corporate network)
```

---

## Prerequisites

| Where | Needs |
|---|---|
| Cluster (one-time) | `oc` logged in with rights to create a namespace + Route |
| Sharing machine (Mac / HPC login) | a container runtime: **podman**, docker, or apptainer. **No chisel install.** |
| Teammates | nothing |

---

## Step 0 — deploy the relay ONCE per cluster

One command (generates a credential + host key, applies Secret/Deployment/Service/
Route, waits for rollout + Route admission):

```bash
deploy/openshift/chart-relay/deploy-relay.sh --host relay-boxy.apps.<cluster>
```

Preview without applying anything:

```bash
deploy/openshift/chart-relay/deploy-relay.sh --host relay-boxy.apps.<cluster> --dry-run
```

<details><summary>Equivalent manual commands</summary>

```bash
oc new-project boxy-relay 2>/dev/null
boxy generate relay --host relay-boxy.apps.<cluster> \
     --auth "boxy:$(openssl rand -hex 16)" --key-seed "$(openssl rand -hex 16)" \
  | oc apply -f -
oc rollout status deploy/boxy-relay -n boxy-relay
```
Or with Helm: `helm install boxy-relay deploy/openshift/chart-relay --namespace boxy-relay
--create-namespace --set host=relay-boxy.apps.<cluster> --set auth="boxy:$(openssl rand -hex 16)"`.
</details>

The relay Deployment is **OpenShift `restricted-v2` SCC compliant** out of the box
(non-root with an SCC-assigned UID, all capabilities dropped, no privilege
escalation, RuntimeDefault seccomp, read-only rootfs) — it admits under the
strictest default policy with no waiver.

---

## Step 1 — turn sharing on + confirm readiness (sharing machine)

```bash
export BOXY_SHARE_ENABLED=1                 # team sharing is off by default
boxy doctor | grep "share relay"
#   share relay:  [OK] client: containerized chisel via podman (zero install); relay Route https://relay-boxy.apps.<cluster> admitted
```

`relay.client_mode` defaults to `auto` = run chisel in a container (zero install).
Force it with `export BOXY_RELAY_CLIENT_MODE=container` (or `host` to use a binary).

---

## Step 2 — serve a model and share it

**HPC (submit a job, tunnel it home, share it) — one command:**
```bash
boxy serve meta-llama/Llama-3.2-1B-Instruct \
     --scheduler slurm --gpus 1 --ssh you@login-node \
     --share demo
```
```text
### READY   http://gpu-node-07:8090/v1        (compute node)
### LOCAL   http://127.0.0.1:8090/v1          (tunneled to your machine)
### SHARE   https://demo-boxy.apps.<cluster>/v1   (reachable by ANYONE on the network)
```

**Local baremetal (a container right here, then share):**
```bash
boxy serve <model> --share demo
```

**Two-step (share a job that's already running):**
```bash
boxy open demo --ssh you@login-node --port 8090 --share demo
```

Under the hood the `### SHARE` line runs (no host chisel needed):
```text
podman run -d --name boxy-chisel-demo --network=host --env AUTH \
   docker.io/jpillora/chisel:1.10 client --keepalive 25s --max-retry-count -1 \
   https://relay-boxy.apps.<cluster> R:0.0.0.0:31xxx:127.0.0.1:8090
```
(macOS swaps `127.0.0.1` → `host.containers.internal`; the credential rides
`--env AUTH`, never the argv.)

---

## Step 3 — a teammate, zero setup

```bash
curl https://demo-boxy.apps.<cluster>/v1/models
curl https://demo-boxy.apps.<cluster>/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"<served-id>","messages":[{"role":"user","content":"hello"}]}'
```

---

## Step 4 — manage / tear down

```bash
boxy list                    # shows the share + LIVE/DEAD (LIVE = client container up)
boxy unshare demo            # podman rm -f the client container + delete its Route/Service
```

---

## Air-gapped / Docker Hub blocked, or a custom image

Build an OpenShift-ready image (declares a numeric non-root `USER`) and point ONE
setting at it — it feeds **both** the relay server AND the client container:

```bash
podman build -t <registry>/user/chisel:1.10.1 \
    -f deploy/openshift/chart-relay/Containerfile deploy/openshift/chart-relay
podman push  <registry>/user/chisel:1.10.1

export BOXY_RELAY_IMAGE=<registry>/user/chisel:1.10.1
deploy/openshift/chart-relay/deploy-relay.sh --host relay-boxy.apps.<cluster> \
    --image <registry>/user/chisel:1.10.1
```

Precedence: `--image` flag → `BOXY_RELAY_IMAGE` → `[images].relay` in config →
default `docker.io/jpillora/chisel:1.10`.

---

## Prove it works — checklist

- [ ] `deploy-relay.sh --host relay-boxy.apps.<cluster>` → `### RELAY READY` and
      `oc get route boxy-relay -n boxy-relay` shows **Admitted=True**.
- [ ] `oc get pod -n boxy-relay` → the relay pod is **Running** (admitted under
      restricted-v2 with no SCC error).
- [ ] `boxy doctor | grep 'share relay'` → **[OK] … zero install … admitted**.
- [ ] `boxy serve <model> --ssh … --share demo` prints a `### SHARE https://…` URL.
- [ ] `boxy list` shows the share **LIVE**; `podman ps` shows `boxy-chisel-demo`
      (no chisel binary was installed).
- [ ] From another machine on the network: `curl https://demo-boxy.apps.<cluster>/v1/models`
      returns the model list.
- [ ] `boxy unshare demo` → the share is gone and the client container is removed.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| relay pod `ImagePullBackOff` | Docker Hub blocked — mirror chisel to your registry and set `BOXY_RELAY_IMAGE` (see RUNBOOK §0.993). |
| `share relay: [WARN] … can't run the client here` | install podman/docker on the sharing machine (or `brew install chisel-tunnel` for host mode). |
| Route not `Admitted` | the host is taken or ingress rejected it — pick a free `--host` and redeploy. |
| `### SHARE` missing, only `### ROUTE` | share failed (relay unreachable) — the local tunnel still works; check `boxy doctor` and `BOXY_RELAY_URL`/`BOXY_RELAY_AUTH`. |
| `--share` errors asking for a tunnel | `--share` needs the laptop tunnel, so pass `--ssh` (or serve locally). |

Full reference: **RUNBOOK.md §0.993**.
