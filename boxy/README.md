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

## How boxy decides (v2 resolution rules)

| Decision | Rule |
|---|---|
| model | **Syntax decides**: `hf://`, `ollama://`, `oci://`, ... = remote (pulled via RamaLama); anything else = local path. Bare names are never guessed. |
| engine | GGUF or `ollama://` → llama.cpp; safetensors/HF repo → vLLM (needs a GPU, detected or `--gpus N`) |
| accelerator | RamaLama's `get_accel()` (nvidia-smi, ROCm sysfs, ...), normalized (`hip`→`rocm`) |
| runtime | first of podman > docker > apptainer that is **actually working** (probed, not just on PATH) |
| image | per engine+accelerator, from RamaLama's own plugin maps where possible; `--image` overrides. Every reference then resolves through `registries.py`: `--registry HOST/path` sends all images to one registry, `[location.image_mirrors]` rewrites per-registry (`"docker.io" = "registry.site.gov/dockerhub"`, `"*"` catch-all) — see RUNBOOK §0.97 |
| port | engine default (vLLM 8000, llama.cpp 8090), advanced to the next free port when busy |
| scheduler | **never invoked implicitly.** Inside an allocation: run direct, foreground. On a login node: refuse (see below). `--scheduler slurm\|flux` **submits a batch job**: boxy writes the sbatch/`flux batch` script (any `--slurm-*`/`--flux-*` flag passes through), the job re-runs boxy on the compute node, the endpoint arrives over the shared FS, and boxy prints READY and detaches. `--foreground` = attached srun/flux-run instead. |

**Registry origin policy:** boxy only pulls from an allowlist of registries —
default `hf` (huggingface.co) and `ollama` (registry.ollama.ai). ModelScope
(`ms://`, modelscope.cn — operated by Alibaba from China) and all other
transports are **blocked by default**; the refusal names the registry and its
origin. Opting in is a deliberate, auditable act:
`export BOXY_ALLOW_TRANSPORTS=hf,ollama,ms` — env-only on purpose, so a TOML
profile in a repo can never widen the policy silently. `boxy info` shows the
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

# SCALE OUT — three orthogonal modes (all --dryrun-able; see RUNBOOK §4.5):
# 1) One instance ACROSS nodes (model-parallel via Ray; auto-on for vLLM+nodes>1):
boxy serve <model> --scheduler slurm --nodes 2 --gpus 4   # TP=GPUs/node, PP=nodes
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
`BOXY_SHARE_ENABLED=1` (or `[share] enabled = true`) once the relay client is
installed and approved at your site.

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

**Agentless (zero-install).** `boxy generate slurm|flux -o job.sh` and `boxy
serve --agentless --accelerator cuda --image …` emit a **self-contained** batch
script — a plain `podman run` + a shared-FS endpoint write, **no boxy on the
compute node** (needs only a scheduler + container runtime + shared FS). The
model must be pre-staged and the hardware pinned; see `SPEC.md §8c`.

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
