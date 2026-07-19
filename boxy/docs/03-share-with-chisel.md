# 03 — Share a served model with your team (chisel relay, end to end)

*Zero-install team sharing: turn a served model into an everyone-URL through an
OpenShift chisel relay. Serve the model first —
[01 — GPU model on a cluster](01-serve-gpu-model.md) or
[02 — remote non-GPU model](02-serve-remote-nongpu-model.md).*

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

## Full progression: one command → a served, team-shared model

Add `--share <name>` to publish an **everyone-URL** through the OpenShift chisel
relay once the model is up (teammates need nothing installed). The whole
deployment, start to finish:

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler flux --ssh user1@clusterb --share llama8b
Enter OTP Token Value: ······
  auto: partition: pbatch (via flux queue list on clusterb)
  auto: account: AB110001 (via myaccounts on clusterb; also: AB110002, AB110003 — placed in the batch script)
  auto: engine args: --max-model-len 8192 (packaged card 'llama-3.1-8b-instruct' — placed after --)
### CA      copied your site CA -> user1@clusterb:$HOME/.local/share/boxy/store/laptop-ca.crt  (remote SSL_CERT_FILE)
### Remote  user1@clusterb  $ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler flux --account AB110001 -- --max-model-len 8192

  auto: model: hf://meta-llama/Llama-3.1-8B-Instruct (transport URI — pulled via RamaLama)
  auto: scheduler: flux (submitting a batch job — detaches once READY)
### Submitted flux job f2c2yFbcbaAK  (boxy-llama-3.1-8b-instruct)
### Waiting for the job to start and the server to become ready ... (Ctrl-C detaches; the job keeps running)
###   job f2c2yFbcbaAK: RUNNING
###   server starting on cbnode1001 — waiting up to 20 min for readiness at http://cbnode1001:8000/v1/models (Ctrl-C detaches; the job keeps loading)
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Pulling vllm/vllm-openai ... 43%
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Loading safetensors checkpoint shards: 2/5
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Capturing CUDA graph shapes: 18/35
### READY  http://cbnode1001:8000/v1   (model: meta-llama/Llama-3.1-8B-Instruct, flux job f2c2yFbcbaAK)
###   try:   curl -s http://cbnode1001:8000/v1/models
###   stop:  boxy stop boxy-llama-3.1-8b-instruct
### LOCAL   http://127.0.0.1:8000/v1   (tunnel over the SSH session; persists ~12h)
### SHARE   https://llama8b-boxy.apps.clusterb.example.gov/v1   (browser UI: https://llama8b-boxy.apps.clusterb.example.gov/)
```

Three URLs, three audiences — all from that one command:

| URL | Who | How |
|---|---|---|
| `http://cbnode1001:8000/v1` | on the cluster | direct compute-node endpoint |
| `http://127.0.0.1:8000/v1` | **you**, on your laptop | auto SSH tunnel (no setup) |
| `https://llama8b-boxy.apps.…/v1` | **your team** | chisel relay everyone-URL (nothing installed) |

```console
# you (laptop):
$ curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# a teammate (anywhere, nothing installed):
$ curl -s https://llama8b-boxy.apps.clusterb.example.gov/v1/models
```

The chisel relay is deployed **once per cluster** (`boxy generate relay … | oc apply`,
see Step 0 above); after that every `--share` just publishes. The progress
lines (`› Loading safetensors …`) are the live tail of the job log, and boxy now
waits up to **20 min** for the weights to load before detaching (raise it with
`--ready-timeout 1800`), so a slow load no longer ends the command early.

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
| relay pod `ImagePullBackOff` | Docker Hub blocked — mirror chisel to your registry and set `BOXY_RELAY_IMAGE` (see [06-runbook.md](06-runbook.md) §0.993). |
| `share relay: [WARN] … can't run the client here` | install podman/docker on the sharing machine (or `brew install chisel-tunnel` for host mode). |
| Route not `Admitted` | the host is taken or ingress rejected it — pick a free `--host` and redeploy. |
| `### SHARE` missing, only `### ROUTE` | share failed (relay unreachable) — the local tunnel still works; check `boxy doctor` and `BOXY_RELAY_URL`/`BOXY_RELAY_AUTH`. |
| `--share` errors asking for a tunnel | `--share` needs the laptop tunnel, so pass `--ssh` (or serve locally). |

Full reference: **[06-runbook.md](06-runbook.md) §0.993**.
