# boxy

Unified, site-portable, offline-first CLI for deploying and serving
containerized GenAI/LLM services on HPC. It grew out of a bash prototype and
its design notes, both in the
[source repository](https://github.com/abeltre1/canopie25-paper-artifacts).

One command, everything auto-resolved and explained:

```bash
$ boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
  auto: model: hf://Qwen/... (transport URI — pulled via RamaLama)
  auto: scheduler: none (no scheduler on host)
  auto: accelerator: cuda (autodetected)
  auto: runtime: podman (podman found on PATH and responding)
  auto: engine: llama.cpp (model is GGUF)
  auto: image: ghcr.io/ggml-org/llama.cpp:server-cuda (default for llama.cpp+cuda)
  auto: port: 8090 (llama.cpp default)
### Running Command:
    podman run -d --name=boxy-qwen2.5-0.5b-instruct-q4_k_m ...
### Waiting for readiness at http://127.0.0.1:8090/v1/models ...
### READY  http://127.0.0.1:8090/v1   (model: ...)
###   try:  curl -s http://127.0.0.1:8090/v1/models
###   stop: boxy stop boxy-qwen2.5-0.5b-instruct-q4_k_m
```

Every `auto:` decision is overridable by a flag (`--engine --runtime
--scheduler --accelerator --image --port --name`) or by TOML profiles
(`--box` = the *what*, `--location` = the *where*). Profiles are how a site's
quirks (modules, tuning, offline mode, GPU counts) are pinned once and reused.

## Documentation map

| Doc | What it's for |
|-----|---------------|
| [README](README.md) | front door: install, turnkey serve, cards, `push`, air-gap intro |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the layered design (cards → resolution → drivers → agentless) |
| [RUNBOOK.md](RUNBOOK.md) | field operations: TLS/proxy/registry failures and their fixes |
| [DEMO-turnkey.md](DEMO-turnkey.md) | the one-command-per-target demo script + prove-it checklist |
| [DEMO-chisel.md](DEMO-chisel.md) | everyone-URL sharing through the OpenShift chisel relay |
| [AIRGAP.md](AIRGAP.md) | air-gap readiness checklist: bundles, baked images, cleanup, kill switches |
| [RELEASING.md](RELEASING.md) | wheel/PyPI (public + local Nexus), repo extraction |
| [VALIDATION.md](VALIDATION.md) | how the test suite maps to the field guarantees |

(Older session runbooks were retired; anything still useful was folded into the
docs above. `archive/` keeps superseded material for reference.)

## Turnkey: one command, from laptop to cluster

A user with **zero SLURM/container knowledge** serves a model with one command —
no `--scheduler`, `--gpus`, `--account`, `--partition`, `--time`, `--accelerator`:

```bash
boxy serve meta-llama/Llama-3.3-70B-Instruct --ssh user@login   # from your laptop
#   auto: scheduler: slurm (via detected on login)     <- flux/sbatch probed on the cluster; no flag
#   auto: gpus: 4 per node (model card 'llama-3.3-70b-instruct', 80GB-class GPUs)
#   auto: engine: vllm (model card)
#   auto: account: fy260064 (via mywcid)
#   auto: partition: gpu,batch (via sinfo, soonest-start)
#   auto: time: 30:00 (default walltime — the scheduler stops the job then)
#   ...submits a 4-GPU vLLM job, waits for READY, prints the endpoint.
```

`--scheduler slurm|flux` still forces it explicitly. Auto-detection of the
scheduler is on for the `--ssh`-to-a-cluster path; **on a login node directly**
(no `--ssh`) set `BOXY_SCHEDULER=slurm|flux` (or pass `--scheduler`) to submit
without the flag — the bare default there keeps the login-node guard, and
`--here` serves directly on the current node.

boxy fills the gaps from **cards** and **site discovery**, and still prints every
choice (nothing is hidden, only the *work*):

- **model cards** (`boxy cards`) map a model → GPUs / engine / engine args; an
  unknown model is sized from its name (`-70B` → 4 GPUs). RamaLama picks the image.
  Don't have a card for a model? Generate one from its HuggingFace id —
  `boxy generate card <hf-model-id>` fetches the config, derives the engine +
  GPU/VRAM sizing + a capped context, and writes a card into
  `~/.config/boxy/cards/models/` so `boxy serve <id>` just works (see below).
- **system cards** are deployment profiles per system type —
  `--system slurm-cuda | flux-rocm | laptop-podman | cloud-aws-gpu | openshift-gpu`
  (3 per type; the cloud ones drive SkyPilot via `boxy generate sky`).
- **app cards** extend the same idea to *classic HPC applications and
  benchmarks* — boxy as an operating system of deployments, where cards are the
  package format and schedulers/runtimes are the drivers. An app card names a
  spack spec (or a container image), the geometry, and the run lines;
  `boxy app osu-benchmarks --ssh <cluster>` then builds it with spack ON the
  cluster (`--reuse`, so reruns are no-ops), submits it with the same zero-flag
  site resolution as `serve`, waits, and prints the results log — agentless, the
  cluster needs only spack or podman. Packaged cards: `osu-benchmarks`, `stream`,
  `miniem` (`boxy app` lists them; drop yours in `~/.config/boxy/cards/apps/`).
  If the cluster's egress filter blocks spack's source download (Zscaler
  CATEGORY_DENIED on `mirror.spack.io` *and* the upstream), boxy fetches the
  archive on **your machine**, sha256-verifies it, stages it into a `file://`
  spack mirror on the cluster's shared FS, and resubmits — automatically.
  Any container image runs as an ad-hoc app with no card at all:
  `boxy app --image quay.io/x/y:tag --ssh <cluster> [--cmd '...']` — pulled on
  the compute node through the forwarded proxy and launched via `srun`/`flux run`
  with `--nodes/--tasks-per-node` geometry (`--container` is an alias).
- **service cards** deploy LONG-RUNNING services — a web server, an MCP
  server, a database, any microservice — with the same agentless pipeline:
  `boxy app --image docker.io/traefik/whoami --port 8080 --ssh <cluster>`
  submits a job that publishes its endpoint to the shared FS and stays up;
  boxy waits for the URL (not the exit), prints the `ssh -L` line to reach it,
  and `boxy stop <name> --ssh <cluster>` is the off switch. `--env K=V`
  (repeatable) configures it. A card makes it one word: the packaged
  `flux-mcp` card serves the Flux MCP server (`boxy app flux-mcp --ssh
  <cluster>`) — this REPLACES the deprecated `boxy generate flux-mcp` for
  clusters (that emitter remains for OpenShift targets). Write your own under
  `~/.config/boxy/cards/apps/` with `service = true`, `port`, `[app.env]`.
- **site discovery** fills `--account` from `mywcid` / `$SBATCH_ACCOUNT` /
  `sacctmgr`, plus partition/time. When `mywcid` lists **several charge accounts
  (WCIDs)** and you didn't name one, boxy shows an **interactive picker** on a
  terminal so you choose which the job charges to (remembered per cluster,
  validated against the live list); bypass it with `--account`, `WCID=fy…`, or
  `BOXY_ACCOUNT`, and it never blocks a non-terminal run (`BOXY_PICK_ACCOUNT`).
  Likewise, when **2+ partitions** are available and none was named, boxy offers
  an **interactive partition picker** (pick one, or `all` to keep the
  soonest-start comma-list); bypass with `--partition`/`BOXY_PICK_PARTITION`.
  Slurm **licenses** can be requested with `--license tscratch:1` (or
  `BOXY_LICENSE`) for sites that gate filesystems behind them.
  **Partition selection is automatic** — with
  no flag, boxy reads `sinfo`/`flux queue list` and submits to every
  **GPU-bearing** partition (idle-first), so Slurm starts the job wherever a GPU
  frees first instead of parking in one queue. Override with `--partition
  <name|list>`, `--partition all` (include CPU partitions), or `--partition off`
  (the scheduler's own default); `BOXY_PARTITION` pins a fixed choice.
- **auto-unique**: if a live instance of the same model already exists,
  re-running `boxy serve` starts an *independent* instance (its own job / log /
  endpoint) instead of blocking — other users never have to remember `--unique`,
  while power users can still pass it explicitly to force a fresh instance every
  time. A ready service still just reports its URL; over `--ssh` it's decided
  laptop-side so it works even against an older cluster boxy; opt back to the
  strict singleton with `--no-auto-unique` / `BOXY_AUTO_UNIQUE=false`.
- **proxy**: the corporate proxy for image/model pulls is picked up
  automatically from your `http(s)_proxy` env (or config `network.proxy` /
  `BOXY_PROXY`) and, over `--ssh`, forwarded to the cluster job — no `--proxy`
  needed. Pass `--proxy URL` to pin one.

Same command deploys anywhere — laptop (Podman/Docker), HPC (Apptainer/CharlieCloud
+ Slurm/Flux), cloud/OpenShift — because the scheduler and runtime are hidden
behind pluggable drivers. See **[`ARCHITECTURE.md`](ARCHITECTURE.md)** for the
layered diagram of where that hiding happens, and drop your own cards in
`~/.config/boxy/cards/{models,systems}/` to override the built-ins.

Power users keep full control: pass any flag and it wins; `--dryrun` prints the
whole plan (batch script included) with zero network.

**[`DEMO-turnkey.md`](DEMO-turnkey.md)** is the one-command-per-target runbook:
the exact `auto: account:` output to expect on laptop / Slurm / Flux / cloud, a
prove-it checklist mapped to automated tests, and how the account from `mywcid`
reaches the batch script even when the cluster's boxy predates turnkey (it is
resolved laptop-side and injected as `--account` over `--ssh`). Verify a real
cluster first with `boxy doctor --ssh <login>` — it reports the discovered
account with no boxy installed there.

## How boxy decides (v2 resolution rules)

| Decision | Rule |
|---|---|
| model | **Syntax decides**: `hf://`, `ollama://` = remote (pulled via RamaLama); `s3://` = staged from a bucket; anything else = local path. Bare names are never guessed. `oci://`/`docker://` are recognized but their **pull is not implemented yet** (roadmap) — pull the OCI artifact with podman/docker and serve the extracted weights by path. |
| engine | GGUF or `ollama://` → llama.cpp; safetensors/HF repo → vLLM (needs a GPU, detected or `--gpus N`) |
| accelerator | RamaLama's `get_accel()` (nvidia-smi, ROCm sysfs, ...), normalized (`hip`→`rocm`) |
| runtime | first of podman > docker > apptainer that is **actually working** (probed, not just on PATH) |
| image | per engine+accelerator, from RamaLama's own plugin maps where possible; `--image` overrides. Every reference then resolves through `registries.py`: `--registry HOST/path` sends all images to one registry, `[location.image_mirrors]` rewrites per-registry (`"docker.io" = "registry.site.gov/dockerhub"`, `"*"` catch-all) — see RUNBOOK §0.97 |
| port | engine default (vLLM 8000, llama.cpp 8090), advanced to the next free port when busy |
| scheduler | **auto-detected by LIVENESS (turnkey), not binary presence.** Inside an allocation: run direct, foreground. Over `--ssh` to a cluster, boxy probes which control plane is LIVE and applies one rule: **a reachable Flux broker wins** (Flux runs the machine; slurm-compat `sbatch`/`sinfo`/`scontrol` shims proxy to it — submitting through them returns *Flux* job ids `squeue` can't track), else a real `slurmctld`/`sinfo` ⇒ slurm. So a Flux system that ships slurm shims is correctly picked as **flux** and tracked with `flux jobs`, not misread as slurm. The live scheduler then **submits a batch job** — no `--scheduler` needed (`BOXY_SCHEDULER`/`site.scheduler` pins the rare exception; `--here` keeps a direct serve; `none` disables). On a login node directly the default keeps the safety guard — set `BOXY_SCHEDULER` to submit flagless. `--scheduler slurm\|flux` always forces it. Either way boxy writes the sbatch/`flux batch` script (any `--slurm-*`/`--flux-*` flag passes through), the job re-runs boxy on the compute node, the endpoint arrives over the shared FS, and boxy prints READY and detaches. `--foreground` = attached srun/flux-run instead. |
| walltime | `--time` (Slurm colon notation, e.g. `4:00:00`); default **30 min** (`site.default_time`/`BOXY_DEFAULT_TIME`), injected into every batch job. **The scheduler kills the served job at the walltime** — raise it for long sessions; set empty for the scheduler's own default. |

**Registry origin policy:** boxy only pulls from an allowlist of registries —
default `hf` (huggingface.co) and `ollama` (registry.ollama.ai). ModelScope
(`ms://`, modelscope.cn — operated by Alibaba from China) and all other
transports are **blocked by default**; the refusal names the registry and its
origin. Opting in is a deliberate, auditable act:
`export BOXY_ALLOW_TRANSPORTS=hf,ollama,ms` — env-only on purpose, so a TOML
profile in a repo can never widen the policy silently.

**Transport support (today):** model *weights* pull via `hf://` and `ollama://`
(live-verified), `s3://` staging, and local paths. `ms://`/`rlcr://` are
allow-list-gated. `oci://` and `docker://` are recognized (parsed + policy-gated)
but their **pull is not yet implemented** — boxy errors with guidance; pull the
artifact with podman/docker and serve the extracted weights by path. (Note: this
is the model *transport* — a container **image** stored as an OCI artifact works
fine via `--image registry/…`.)

`boxy info` shows the
active allow/block lists plus auth status (HuggingFace token, S3 credentials —
status and source only, values are never printed), and `boxy info --net`
probes only allowed registries.

HPC guard rails (from the design review):

- **Login-node guard**: if `srun`/`flux` is on PATH but you're not inside an
  allocation, boxy refuses to start an LLM server (shared login nodes) and
  prints the exact `--scheduler`/allocation alternatives; `--here` overrides.
- **Inside an allocation** boxy stays in the foreground so the job step owns
  the server, and prints `http://<hostname>:PORT/v1` plus an `ssh -L` tunnel
  hint. Detached mode (+ readiness gate + READY banner) is the default only on
  laptops/workstations.
- **`--gpus`/`--nodes` are a job request** — they error without `--scheduler`.
- Submitting a GPU job from a GPU-less login node requires `--accelerator
  cuda|rocm` (or a `--location` profile): boxy won't guess the compute node's
  hardware from the wrong machine.
- On a crash during startup boxy dumps the container's last log lines and
  cleans up; on a slow model load it leaves the container running and tells
  you how to follow the logs.

## Install

Requires **Python 3.11+** on a POSIX host (Linux, or macOS best-effort; Windows
is unsupported — use WSL2). The core has one dependency (`certifi`); everything
else degrades gracefully.

```bash
pip install boxy-hpc            # the distribution is boxy-hpc; the command is `boxy`
pip install "boxy-hpc[ramalama]"   # + model pulls / accelerator autodetect (recommended)
pip install "boxy-hpc[cloud]"      # + SkyPilot cloud launch
pip install "boxy-hpc[s3]"         # + boto3 for S3 model staging
```

| Extra | Adds | For |
|-------|------|-----|
| `ramalama` | RamaLama | `hf://`/`ollama://` pulls, GPU autodetect |
| `cloud` | SkyPilot | `boxy launch` on cloud VMs |
| `s3` | boto3 | S3 model staging |

Developing on boxy? Use an **editable** install so `git pull` takes effect
without reinstalling: `pip install -e './boxy[ramalama,test]'`.

Building a wheel for an internal PyPI mirror? `make wheel` (or
`make publish LOCAL_PYPI=<upload-url>`) from `boxy/` — see
[RELEASING.md](RELEASING.md#publishing-to-a-local-private-pypi).

**uv note:** uv-managed standalone Pythons don't inherit the system CA store,
so HTTPS (model pulls) fails with `CERTIFICATE_VERIFY_FAILED` until you set
`SSL_CERT_FILE`. `boxy pull` prints the remedy if you hit it.

**uv note:** uv-managed standalone Pythons don't inherit the system CA store,
so HTTPS (model pulls) fails with `CERTIFICATE_VERIFY_FAILED` until you set
`SSL_CERT_FILE` (see RUNBOOK §2.1 / troubleshooting). `boxy pull` prints the
remedy if you hit it.

## Quickstart

**Same model, every platform** — a box (model) is portable; a location (platform) is
swappable. Only the location/scheduler changes (smallest Llama, 3.2 1B GGUF, shown):

```bash
M="hf://hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf"
boxy serve $M                                             # 1) local desktop / baremetal (CPU)
boxy serve $M --scheduler slurm --gpus 1                 # 2) Slurm  (submit + wait + detach)
boxy serve $M --scheduler flux  --gpus 1                 # 3) Flux   (same UX)
boxy launch --box examples/boxes/llama-3.2-1b.toml \
            --location examples/locations/cloud-gpu.toml # 4) any other platform (cloud/SkyPilot,
                                                         #    or write a --location <site>.toml)
```

**From anywhere:** add `--ssh user@login-node` (or `export BOXY_SSH_HOST=...` once) and the
same commands run on the cluster over ONE multiplexed SSH session — OTP/YubiKey prompted
once, endpoint auto-tunneled back to `http://127.0.0.1:<port>/v1` (RUNBOOK §0.95).

```bash
# What does this host have?
boxy info

# Serve a local GGUF (laptop/workstation: detaches, waits, prints READY):
boxy serve /path/to/model.q4_k_m.gguf

# Serve straight from HuggingFace / Ollama (pulled via RamaLama transports):
boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
boxy serve ollama://granite3-moe

# Foreground with engine logs (Ctrl-C stops it):
boxy serve model.gguf --foreground

# Pass engine args through after `--` (yours always win):
boxy serve model.gguf -- --ctx-size 4096

# THE SEAMLESS HPC PATH — one command from the login node. boxy generates and
# submits the batch job, re-resolves hardware ON the compute node, waits for
# readiness over the shared FS, prints the endpoint, and detaches:
boxy serve hf://org/model-GGUF/file.gguf --scheduler slurm --gpus 1 \
    --partition=short --account=myacct --license=tscratch:1
#   ### Submitted slurm job 12345  (boxy-file)
#   ###   job 12345: PENDING ... RUNNING
#   ### READY  http://cn042:8090/v1   (model: ..., slurm job 12345)
# Scheduler variables are passed DIRECTLY: --partition/--account/--time are
# translated per scheduler, and any other --FLAG=VALUE goes to the active one
# — new site flags never require a boxy change. Ctrl-C detaches; job keeps going.
boxy serve <model> --scheduler flux --gpus 4 --partition pbatch      # Flux: identical

# Launch MANY of the same model at once — each gets its own job, log, endpoint:
boxy serve <model> --scheduler flux --gpus 4 --unique   # repeat freely, no name clash
# (default name is a stable singleton so a plain rerun reconnects instead of duplicating.)

# CARDS ARE DETERMINISTIC, NOT GUESSED. Serving an UNCARDED model generates its
# card on the spot from the model's own HuggingFace metadata (config.json +
# safetensors index — exact on-disk bytes, dtype, quantization; nvidia/* models
# publish there too) and writes it to ~/.config/boxy/cards/models/ — fetched
# once, loaded as a plain user card forever after (and carried into air-gap
# bundles). The old name-size guess only fires when the Hub is unreachable, and
# says so: "geometry below is a NAME GUESS; run `boxy generate card <id>`".
# BOXY_CARD_AUTOGEN=false disables the lookup (HF_HUB_OFFLINE=1 also skips it).

# GEOMETRY IS SOLVED FROM CARDS — no --gpus/--nodes needed. The model card's
# min_vram_gb (demand) is fit against the cluster's node shape (supply). GENERATE
# the system card from the cluster's own scheduler inventory (one command, once):
#   boxy generate system --ssh cronus        # probes sinfo/flux: nodes, GPUs/node,
#                                            # GPU type->VRAM, CPUs, memory, scratch
#                                            # storage -> ~/.config/boxy/cards/systems/cronus.toml
# (--dry-run previews; --force regenerates after a hardware change; run it without
# --ssh on a login node to card the local cluster. Edit the file freely — e.g. fill
# gpu_vram_gb yourself if the GPU type isn't in boxy's table.)
# Then `boxy serve meta-llama/Llama-3.3-70B-Instruct --ssh cronus` requests
# 2x140GB GPUs (not the 80GB-class default of 4), and a model bigger than one
# node automatically becomes an N-node Ray instance. Without a system card the
# solver assumes 4x80GB and says so. BOXY_GPUS_PER_NODE/BOXY_GPU_VRAM_GB pin the
# shape via env; --gpus/--nodes always override everything (power users).

# SCALE OUT — three orthogonal modes (all --dryrun-able; see RUNBOOK §4.5):
# 1) One instance ACROSS nodes (model-parallel via Ray; auto-on for vLLM+nodes>1,
#    and AUTO-CHOSEN when the model card exceeds one node — see above).
#    Works agentless over --ssh too: the batch script runs the Ray head + vllm on
#    the head node and sruns the worker containers onto the rest — no boxy on the
#    cluster (--no-distributed opts out):
boxy serve <model> --ssh hops --nodes 2 --gpus 4          # explicit TP=GPUs/node, PP=nodes
# 2) N INDEPENDENT replicas (data-parallel), BIN-PACKED onto node GPUs
#    (4 replicas x 1 GPU = 1 node, not 4). --nodes N spreads across N nodes
#    (12 replicas --nodes 4 = 3/node); --gpus-per-replica R gives each TP=R;
#    --nodes-per-replica M makes each replica an M-node distributed instance:
boxy serve <model> --scheduler slurm --gpus 4 --replicas 4
# Needs a package the image lacks? build a thin image yourself and pass it:
#   podman build -t localhost/vllm-extra:latest ...   (FROM vllm image + pip install)
boxy serve <model> --image localhost/vllm-extra:latest
#    ...with ONE load-balanced URL in front (built-in login-node router):
boxy serve <model> --scheduler slurm --gpus 4 --replicas 4 --router   # http://<login>:8000/v1
boxy router <base> --emit nginx                                       # or hand off to a real proxy
# 3) Scaling SWEEP (submit -> READY -> bench -> teardown per rung; prints a comparison table):
boxy sweep <model> --scheduler slurm --gpus 4 --sweep-nodes 1,2,4,8 -o scaling.csv

# Inside a Slurm/Flux allocation: runs direct + foreground automatically.
srun -N1 --gpus-per-node=1 --pty boxy serve /lustre/models/llama-3.1-8b.Q6_K.gguf

# Attached (old-style srun wrap) instead of batch submission:
boxy serve hf://org/model --scheduler slurm --gpus 4 --accelerator cuda --foreground

# Pre-stage a model on the login node (network) for compute nodes (no network):
boxy pull hf://org/repo/file.gguf
boxy pull hf://org/repo --force    # wipe a partial/corrupt cache and re-pull clean

# WHERE models land on the cluster (--ssh serves): boxy probes the login node for
# a big SHARED scratch FS ($SCRATCH, /tscratch, /pscratch, /scratch — first
# writable one with >= storage.min_free_gb free, default 100) and puts the HF
# cache THERE, never on your $HOME quota. The pick is remembered per cluster so
# the cache is downloaded once and always reused. Pin it explicitly with:
#   export BOXY_MODEL_DIR=/tscratch/users/$USER/boxy    # (or config storage.model_dir)

# vLLM note: boxy defaults `--safetensors-load-strategy eager` (vLLM's recommendation
# for the NFS/Lustre stores HPC uses; vLLM>=0.24's NFS auto-prefetch can misload shards).
# Override per-serve with `-- --safetensors-load-strategy prefetch`, or BOXY_NO_VLLM_EAGER=1.

# Freeze what was resolved into reviewable, reusable profiles:
boxy serve model.gguf --save-profile example     # writes example.box.toml + example.location.toml
boxy serve --box example.box.toml --location example.location.toml

# Lifecycle:
boxy list
boxy stop boxy-model-name         # name is printed in the READY banner

# Everything still works profile-first too (the paper's pipeline):
boxy serve --box examples/boxes/vllm.toml --location examples/locations/clusterA.toml --dryrun
boxy build --box examples/boxes/vllm.toml --location examples/locations/clusterA.toml   # OCI -> SIF
boxy bench --box examples/boxes/vllm.toml --batch-sizes 1,2,4,8 -o results.csv

# Cloud: delegate the same box to SkyPilot (pip install 'boxy-hpc[cloud]'):
boxy launch --box examples/boxes/vllm.toml --location examples/locations/cloud-gpu.toml --serve
```

Drop `--dryrun` from any profile command to execute. The ClusterA dry-run
reproduces the prototype's known-good command:

```
flux run -N2 --gpus-per-node=4 bash -lc 'module load rocm/6.4.0 && exec \
  apptainer exec --fakeroot --writable-tmpfs --cleanenv --no-home \
  --cwd /vllm-workspace/models --bind ./models:/vllm-workspace/models \
  --env HF_HOME=/root/.cache/huggingface --rocm \
  --env HF_HUB_OFFLINE=1 ... vllm-rocm.sif \
  vllm serve Llama-4-Scout-17B-16E-Instruct --host=0.0.0.0 --port=8000 \
  --tensor-parallel-size=4 --seed=12345 --gpu-memory-utilization=0.7'
```

## Publish models to site storage — `boxy push`

Download from HuggingFace ONCE and publish where every cluster can pull —
an S3 bucket or a container registry (OCI model artifact):

```bash
boxy push meta-llama/Llama-3.1-8B-Instruct  s3://models/llama31-8b
boxy push nvidia/NVIDIA-Nemotron-Parse-v1.2 oci://registry.site.gov/models/nemotron-parse:v1.2

# then serve from the site copy anywhere:
boxy serve s3://models/llama31-8b --ssh <cluster>
boxy serve oci://registry.site.gov/models/nemotron-parse:v1.2 --ssh <cluster>
```

S3 uploads use boto3 or the aws CLI (`--endpoint` / `S3_ENDPOINT_URL` for
internal object stores); registry pushes package the model as an OCI artifact
via RamaLama (`podman login <registry>` first). On an intercepted network, fix
TLS first with **`boxy trust huggingface.co`** — it captures the interceptor's
CA chain (with your confirmation) into boxy's trust store, which is exactly
what `boxy info --net` diagnoses.

## Air-gapped deployments

Build a **bundle** on a connected machine, carry it across the gap, serve
entirely from it — no proxy, no egress, nothing fetched:

```bash
# connected side: model + aux custom-code repos (HF cache), engine image
# (oci-archive), the card's pip deps (wheels), manifest
boxy bundle nvidia/NVIDIA-Nemotron-Parse-v1.2 -o nemotron-bundle/

# move nemotron-bundle/ to the cluster (scp -r / tar / media), then:
boxy serve nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --bundle /projects/me/nemotron-bundle --ssh <cluster>
```

The batch script `podman load`s the image from the bundle's oci-archive, mounts
its HF cache into the container with `HF_HUB_OFFLINE=1` (model *and* the custom
code it would otherwise fetch dynamically resolve from cache — the card's
`aux_repos` are pre-bundled), and the card's pip deps install `--no-index` from
the bundle's wheels. Card knowledge (trust flags, geometry) applies exactly as
online.

## Smoke test on a real cluster

1. `boxy info` on a login node — confirm runtime + scheduler detection.
2. `boxy serve <model> --dryrun` — eyeball the `auto:` decisions + command.
3. Inside an allocation (`salloc` / `flux alloc`), run without `--dryrun`.
4. `curl http://<node>:PORT/v1/models` — the OpenAI route is up.

## Design rules (from the prototype, the paper, and the v2 design review)

- **Boxes never name a runtime or accelerator** — locations do.
- **Syntax, not filesystem state, classifies a MODEL** — the same command
  means the same thing on every machine.
- **A scheduler is never invoked implicitly** — job submission can't be a
  side effect of serving a model.
- **User args always win**: box args and location tuning are tacked on last
  and skipped if you already set them.
- **Offline by default on HPC locations**: `HF_HUB_OFFLINE=1` and friends are
  injected when `offline = true`.
- **Every automatic choice is printed and overridable** — `auto:` lines are
  the contract; `--save-profile` freezes them for review and reuse.
- **RamaLama is a seam, not a hard dependency**: every `ramalama` import
  lives in `src/boxy/ramalama_shim.py`; without it boxy still works with
  explicit locations and path-based models.

## Configuration

Every built-in default is layered: **CLI flag > environment variable > config
file > default**. The config file is TOML at `$BOXY_CONFIG` or
`~/.config/boxy/config.toml`:

```bash
boxy config              # show every setting + where its value came from
boxy config --init > ~/.config/boxy/config.toml   # a commented starter file
```

```toml
# ~/.config/boxy/config.toml
[network]
bind_host = "0.0.0.0"        # 127.0.0.1 only for a purely local serve
[timeouts]
readiness = 300             # BOXY_READY_TIMEOUT / --ready-timeout override this
[paths]
jobs_root = "/scratch/$USER/boxy/jobs"   # for sites where $HOME isn't on compute nodes
[mounts]
selinux_relabel = "auto"    # add ':z' to bind mounts on SELinux-enforcing hosts
```

Team sharing (the OpenShift relay) is off unless you enable it — set
`BOXY_SHARE_ENABLED=1` (or `[share] enabled = true`). **Zero install on both
ends:** deploy the chisel relay to OpenShift once per cluster and teammates reach
`https://<name>-boxy.apps.<cluster>/` with nothing installed; on the sharing side
boxy runs the `chisel client` in a **container** by default (`relay.client_mode =
auto` → `container` unless a host `chisel` binary is present), so no `brew install`
on your Mac or the HPC login node — only a container runtime. Point `[images].relay`
at a site mirror for air-gapped clusters. Step-by-step walkthrough (deploy once →
serve → share → teammate access → teardown) in
[`DEMO-chisel.md`](https://github.com/abeltre1/canopie25-paper-artifacts/blob/main/boxy/DEMO-chisel.md);
one-shot relay deploy in `deploy/openshift/chart-relay/deploy-relay.sh`; full
reference in `RUNBOOK.md` §0.993.

## Seen in action

The packaged examples ship inside the wheel — `boxy examples` lists them,
`boxy examples export ./examples` drops them into a directory, and
[`examples/MATRIX.md`](https://github.com/abeltre1/canopie25-paper-artifacts/blob/main/boxy/src/boxy/data/examples/MATRIX.md)
shows a machine-generated command for every engine × runtime × scheduler
combination. [`DEMO.md`](https://github.com/abeltre1/canopie25-paper-artifacts/blob/main/boxy/DEMO.md)
records a real end-to-end run.

## Cloud path (SkyPilot delegation)

For cloud sites, boxy doesn't reimplement provisioning — it transpiles the
same box+location into a SkyPilot task:

```bash
boxy generate sky --box examples/boxes/vllm.toml \
     --location examples/locations/cloud-gpu.toml --serve -o task.yaml
sky launch task.yaml        # batch, or:
sky serve up task.yaml      # managed serving (SkyServe replicas + readiness probe)
```

### Add a model in one command — `boxy generate card`

Serving a model boxy doesn't ship a card for? Generate one from its HuggingFace
id. boxy fetches the model's `config.json` + safetensors index, picks the engine
(vLLM, or llama.cpp for GGUF repos), derives `gpus`/`min_vram_gb` from the weight
size vs an 80GB-class GPU, caps the context at 8192, and writes a card into
`~/.config/boxy/cards/models/` (where `boxy serve` reads it):

```bash
# preview first (writes nothing):
boxy generate card meta-llama/Llama-3.1-8B-Instruct --dry-run
# -> # meta-llama/Llama-3.1-8B-Instruct — 8B params, bf16 (~16GB weights): 1 80GB-class GPU.
#    [model]
#    match = "meta-llama/Llama-3.1-8B-Instruct*"
#    engine = "vllm"
#    gpus = 1
#    min_vram_gb = 24
#    [model.args]
#    max_model_len = 8192

boxy generate card meta-llama/Llama-3.1-8B-Instruct   # writes the card
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct       # now sized automatically
```

Gated repos (`meta-llama/*`) need `export HF_TOKEN=…` (accept the license on the
model page first). Flags: `--engine vllm|llama.cpp` forces the backend,
`--max-model-len N` sets the context, `--force` overwrites an existing card
(keeps a `.bak`), `-o FILE` writes elsewhere. Existing cards are never clobbered
silently — a terminal prompts with a diff, a script must pass `--force`.

## Going to production

**[`RUNBOOK.md`](https://github.com/abeltre1/canopie25-paper-artifacts/blob/main/boxy/RUNBOOK.md)** is the step-by-step path from fresh checkout to
serving on your cluster — laptop first, then Slurm+CUDA, then Flux+ROCm — with
expected output at each step, a test-provenance table (what has been *executed*
vs. verified-by-construction), and a troubleshooting table covering every
failure observed in real-user testing (SSL/CA bundles, macOS podman prompts,
amd64-on-ARM, Podman workdir strictness).

**`boxy doctor`** audits the environment for the known field issues (proxy/CA/
token, container runtime, scheduler, accelerator, per-cluster state, OOM'd
containers; `--net` also probes image-registry reachability) and prints
OK/WARN/FAIL + a fix for each — run it before serving, or `boxy doctor --ssh
user@login` to audit a cluster. The full catalog of issues, severities, and
mitigations is `SPEC.md §8b`.

**Agentless (zero-install).** `boxy serve hf://<model> --ssh user@login` is
**fully agentless by default**: from your laptop, boxy renders a self-contained
`podman run` batch script — a plain container + a shared-FS endpoint write, **no
boxy/Python/RamaLama on the cluster** (needs only a scheduler + container runtime
+ shared FS). `boxy generate slurm|flux -o job.sh` emits the same script to a file.

By default the compute node pulls the image + model itself (fast — no up-front
download), which works whenever the node has network. boxy stages your merged
site **CA bundle** onto the cluster and mounts it into the container, so an
in-container HuggingFace fetch through a TLS-interceptor proxy (Zscaler) doesn't
die with `CERTIFICATE_VERIFY_FAILED`. The hardware is pinned (`--accelerator`)
so the `podman run` is fully resolved laptop-side.

*Truly isolated compute nodes* (no external network at run time) can opt into
`--prestage`: boxy pulls the image *and* downloads the model on the **login
node** (which has your SSH session's network + the forwarded proxy) onto the
shared filesystem, then serves the model **by path** — so the compute node never
needs to reach Docker Hub or HuggingFace. `--prestage` / `BOXY_AGENTLESS_PRESTAGE
= auto|always`; off by default. See `SPEC.md §8c`.

## Tests

```bash
pytest          # 291 tests: golden-argv vs the prototype, one regression test
                # per audit gap and per field finding, the v2 resolution rules
                # (login-node guard, hip->rocm, port scan, runtime probes),
                # bench vs a real HTTP server, a degraded-mode suite run
                # WITHOUT ramalama on the path, and a live Docker cycle
                # (serve -> inference -> list -> stop) that skips cleanly
                # where Docker or the demo image is absent
```

## Not yet implemented (see SPEC.md §8 roadmap; known issues in §8b)

`boxy run MODEL` as an interactive chat REPL (RamaLama parity; `run` is
reserved for it), engine choice by artifact sniffing after pull (GGUF magic
bytes instead of URI text), `--pull=never|missing|always`, `boxy alloc`
(interactive allocation), `boxy stage` (S3/shared-FS sync), Enroot/Pyxis +
Slurm `scrun` backends, apptainer detached serving, bash-probe hardware
auto-detection for the agentless path, and laptop-side `list/curl/logs` over the
shared FS without a cluster boxy.
