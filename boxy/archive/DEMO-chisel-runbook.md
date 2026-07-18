# boxy × chisel — full demo runbook (deploy → serve → share)

Serve a model on Slurm from your laptop and publish it to the whole corporate
network through an OpenShift chisel relay — **zero install** on both ends.

> Placeholders: `ambelt@hops` = your cluster login; `<api-url>` = the OpenShift
> API URL. Everything else is filled in.

---

## 0. Prerequisites (once)

- **Laptop:** boxy installed; a container runtime (`docker` or `podman`); the `oc`
  CLI; your site CA at `~/.local/certs/new-nix.crt`.
- **Cluster login (`hops`):** boxy installed; Slurm; `HF_TOKEN` exported in your
  shell profile there (Llama-3.2 is gated — the compute node needs it).
- **OpenShift:** `oc login`, with rights to create a namespace + Route.

---

## 1. Set-once variables (laptop)

```bash
export CLUSTER_LOGIN=ambelt@hops
export SSL_CERT_FILE=~/.local/certs/new-nix.crt     # boxy auto-carries this to the cluster
export BOXY_SHARE_ENABLED=1                          # allow team sharing

oc login <api-url>                                   # laptop needs oc to discover the relay + credential
export APPS=$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}')
echo "apps domain: $APPS"                            # e.g. apps.goodall.sandia.gov
```

---

## 2. Deploy the chisel relay on OpenShift (once per cluster)

```bash
deploy/openshift/chart-relay/deploy-relay.sh --host relay-boxy.$APPS
```
Expect: `### RELAY READY  https://relay-boxy.<apps>  (admitted: True)`.

<details><summary>Air-gapped / custom image</summary>

```bash
podman build -t <registry>/user/chisel:1.10.1 \
    -f deploy/openshift/chart-relay/Containerfile deploy/openshift/chart-relay
podman push  <registry>/user/chisel:1.10.1
export BOXY_RELAY_IMAGE=<registry>/user/chisel:1.10.1        # feeds server AND client
deploy/openshift/chart-relay/deploy-relay.sh --host relay-boxy.$APPS --image "$BOXY_RELAY_IMAGE"
```
</details>

---

## 3. Pre-flight checks

```bash
boxy info --net                        # LAPTOP TLS/net (info is local-only — no --ssh)
boxy doctor --ssh $CLUSTER_LOGIN       # CLUSTER readiness (runtime, scheduler, registry egress)
boxy doctor | grep "share relay"       # relay admitted + zero-install client
#   want: [OK] client: containerized chisel via docker (zero install); relay Route https://relay-boxy.<apps> admitted
```

---

## 4. Serve the model AND share it — one command (from the laptop)

```bash
boxy serve hf://meta-llama/Llama-3.2-1B-Instruct \
     --scheduler slurm --gpus 1 --time 30:00 \
     --partition=short,batch --account fy260064 --unique \
     --ssh $CLUSTER_LOGIN \
     --share demo
```
Expected output, top to bottom:
```text
### CA      copied your site CA -> ambelt@hops:$HOME/.local/share/boxy/store/laptop-ca.crt  (remote SSL_CERT_FILE)
### Remote  ambelt@hops  $ boxy serve hf://meta-llama/Llama-3.2-1B-Instruct --scheduler slurm ...
### READY   http://<gpu-node>:8090/v1
### LOCAL   http://127.0.0.1:8090/v1                 (on your laptop, over SSH)
### SHARE   https://demo-boxy.<apps>/v1              (reachable by ANYONE on the network)
```
The `### SHARE` line is chisel: a `chisel client` **container** on your laptop
dials the OpenShift relay, and a per-alias Route publishes `demo-boxy.<apps>`.

---

## 5. Retrieve / confirm the URL

```bash
boxy list --ssh $CLUSTER_LOGIN     # compute-node URL + Slurm job state
boxy list                          # local: the share + LIVE/DEAD
#   shares (everyone-URLs via the OpenShift relay):
#     demo  https://demo-boxy.<apps>/v1  LIVE
```

---

## 6. Teammate access — nothing installed (any machine on the network)

```bash
curl https://demo-boxy.$APPS/v1/models
curl https://demo-boxy.$APPS/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"meta-llama/Llama-3.2-1B-Instruct","messages":[{"role":"user","content":"hello"}]}'
```

---

## 7. Teardown

```bash
boxy unshare demo                        # rm -f the chisel client container + delete the share Route/Service
boxy stop <name> --ssh $CLUSTER_LOGIN    # cancel the Slurm job (name from `boxy list`)
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| No `### READY` | job pending or the pull failed. `boxy logs demo --ssh ambelt@hops`. |
| pull `CERTIFICATE_VERIFY_FAILED` | `SSL_CERT_FILE` not set on the laptop → CA wasn't propagated. `export SSL_CERT_FILE=~/.local/certs/new-nix.crt` and rerun. |
| pull `401`/gated | `HF_TOKEN` not set on the cluster login shell. `echo 'export HF_TOKEN=hf_…' >> ~/.bashrc` on `hops`. |
| No `### SHARE`, only `### ROUTE` + `warning: share failed` | relay not reachable from the laptop — `oc login`, or `export BOXY_RELAY_URL=https://relay-boxy.$APPS BOXY_RELAY_AUTH=user:pass`. Confirm `boxy doctor \| grep 'share relay'`. |
| `share relay: [WARN] … no 'boxy-relay' Route` | relay not deployed — run step 2. |
| relay pod `ImagePullBackOff` | Docker Hub blocked — build/push the custom image (step 2 details) and set `BOXY_RELAY_IMAGE`. |

---

## Demo narration (the "why it's cool")

1. **One command, from my laptop** — submits a Slurm job on `hops`, waits for READY,
   tunnels it home.
2. **My laptop's TLS trust follows me** — the `### CA copied…` line; the compute node
   pulls from HuggingFace through the site interceptor CA automatically.
3. **`--share` publishes it** — chisel runs in a container (no install), dials the
   OpenShift relay, and a Route under the cluster's existing `*.apps` wildcard makes
   `demo-boxy.<apps>` resolve for everyone with nothing installed.
4. **Teammate hits a browser/`curl`** — done. `boxy unshare` + `boxy stop` to clean up.
</content>
