# 14 — Release runbook: CLOUD

Zero to a served model on cloud GPUs. Two supported targets, chosen by who owns
the machines:

| You have | Target | boxy path |
|---|---|---|
| cloud credentials (AWS/GCP/Azure/K8s) | SkyPilot provisions VMs | `boxy generate sky` / `boxy launch` |
| an OpenShift/Kubernetes project with GPUs | you deploy into it | `boxy generate openshift` |

boxy does **not** reimplement provisioning. It transpiles the same box +
location (and the same model cards) into the target's own object, so what runs
in the cloud agrees with what runs on the cluster instead of drifting.

---

## Lane A — SkyPilot (cloud VMs)

### A.1 Install

```bash
pip install 'boxy-hpc[ramalama]' 'skypilot[aws]'   # or [gcp]/[azure]/[kubernetes]
sky check                                          # credentials must be green
boxy --version
```

### A.2 Pick a box and a location

The packaged profiles work verbatim from any directory — boxy resolves them
from inside the package:

```bash
boxy examples                                      # list what ships
boxy examples show cloud-gpu.toml                  # the cloud location
```

A location file is the only thing that changes per platform: it declares the
accelerator, the node/GPU shape, and (for cloud) the instance family.

### A.3 Emit the task and read it before you spend money

```bash
boxy generate sky \
    --box examples/boxes/vllm-hf.toml \
    --location examples/locations/cloud-gpu.toml \
    --serve -o task.yaml
cat task.yaml
```

Check three things in the emitted YAML:

1. `run:` carries the **model card's** engine args — e.g.
   `--max-model-len=8192`. Without it vLLM profiles KV for the model's full
   window at startup and OOMs. boxy prints what it merged:
   `auto: engine args: … (packaged card '…' — merged into the final command)`.
2. `--tensor-parallel-size` matches the GPUs the location declares.
3. `resources.image_id` is the image you expect.

### A.4 Launch

```bash
sky launch -c boxy-demo task.yaml          # a cluster you manage
# or, managed serving with replicas + readiness probes:
sky serve up task.yaml
```

`boxy launch --box … --location …` does the same thing in one step (it writes
the task and shells out to `sky`), and refuses cleanly when `sky` is absent.

### A.5 Reach the endpoint

The endpoint belongs to SkyPilot, so ask SkyPilot for it:

```bash
sky status --endpoint 8000 boxy-demo        # sky launch
sky serve status                            # sky serve up
curl -s http://<endpoint>/v1/models
```

> Known gap: boxy does not yet record cloud endpoints, so `boxy list/attach/
> curl/open` do not know about sky services. Use the `sky` commands above.

### A.6 Tear down (money keeps burning until you do)

```bash
sky down boxy-demo        # or: sky serve down <service>
```

---

## Lane B — OpenShift / Kubernetes

### B.1 Emit a manifest

```bash
boxy generate openshift \
    --model hf://meta-llama/Llama-3.3-70B-Instruct \
    --gpus 4 --namespace my-project \
    --host llama.apps.<cluster> \
    -o llama.yaml
```

Decisions print on **stderr** so stdout stays pure YAML (`| oc apply -f -` is
safe). The generated container command comes from the same engine builders the
HPC path uses, so it carries `--tensor-parallel-size` for the GPUs the
Deployment requests, eager safetensors loading, and the host/port the Service
expects. A `llama.cpp` manifest defers to the image's own ENTRYPOINT (its
binary is off `$PATH`).

### B.2 Serving weights from a PVC instead of the Hub

```bash
boxy generate openshift --model /models/llama-3.3-70b --model-pvc my-weights \
    --gpus 4 -o llama.yaml
```

Pass the **path under the mount**, not a Hub id — a Hub id makes the pod
download the weights and ignore the PVC entirely (boxy says so if you do).

### B.3 Gated models need a secret

```bash
oc create secret generic hf-token --from-literal=token=hf_...
boxy generate openshift --model hf://<gated>/<model> --hf-secret hf-token ...
```

The manifest references the secret by `secretKeyRef` — never a literal — so it
is safe to commit.

### B.4 Apply and reach it

```bash
oc apply -f llama.yaml
oc rollout status deploy/<name> -w        # a big model loads for minutes;
                                          # the startupProbe allows for that
oc port-forward svc/<name> 8000:8000      # lowest-privilege access
curl -s http://127.0.0.1:8000/v1/models
```

With `--host` a Route is emitted and the model is reachable at
`https://<host>/v1` for anyone on the corporate network.

### B.5 Tear down

```bash
oc delete -f llama.yaml
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Pod OOMs immediately on a multi-GPU node | the argv lacked tensor-parallel | fixed — regenerate the manifest with current boxy and confirm `--tensor-parallel-size=<gpus>` appears |
| `llama.cpp` pod CrashLoops, "executable not found" | `command: [llama-server]` — the binary is at `/app/llama-server` | fixed — the manifest now emits `args:` and defers to the image ENTRYPOINT |
| vLLM can't find a directory named `hf:` | the transport URI reached the engine | fixed — the scheme is stripped; regenerate |
| vLLM OOMs profiling KV on a cloud VM | the task ignored the model card | fixed — `generate sky`/`launch` merge the card; confirm `--max-model-len` in `run:` |
| PVC mounted but the pod downloads anyway | `--model` is a Hub id | pass the path under the mount |
| `sky launch` cannot find credentials | provider not configured | `sky check`, then the provider's own setup |

---

## What is verified where

| Step | How it was verified |
|---|---|
| sky task shape + card args | golden test asserting the card's `--max-model-len` lands in `run:` |
| OpenShift argv (TP, eager, host/port) | manifest tests |
| llama.cpp entrypoint handling | manifest test (args without command) |
| staged model served under its canonical id | manifest test |
| a real cloud account / OpenShift cluster | **your systems** — this runbook is the procedure |
