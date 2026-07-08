# boxy End-to-End Runbook

Every step to take boxy from a fresh checkout to serving on your HPC system,
with expected output at each step and a fix for every failure seen so far.

## 0. Test provenance — what has actually been EXECUTED, where

Be precise about what "tested" means. Three tiers:

| Tier | Meaning |
| --- | --- |
| **E** | Executed end-to-end on a real system |
| **G** | Golden-tested: the exact command boxy emits is asserted token-by-token against the paper prototype's known-good commands — but not executed |
| **P** | Pending: needs your cluster (no GPU/scheduler exists anywhere we've run yet) |

| Capability | Status | Where executed |
| --- | --- | --- |
| Unit/regression suite (291 tests) | **E** | CI sandbox |
| Two adversarial audit rounds (7+4 agents, 80+ findings fixed w/ regression tests) | **E** | CI sandbox |
| Scheduler-outage resilience (controller down != job done; no reap/duplicate/mis-stop) | **E** | CI sandbox vs real Slurm |
| serve→inference→list→stop, llama.cpp on **Docker** | **E** | CI sandbox (air-gapped) |
| **v2** `boxy serve MODEL` → auto-decisions → detach → `### READY` → curl → `boxy stop NAME` | **E** | CI sandbox (air-gapped) |
| **v2** crash fast-fail (bad engine flag → log dump → cleanup, rc 1) | **E** | CI sandbox |
| **v2** login-node guard / hip→rocm / port scan / runtime probes | **E** (unit) → **P** on cluster | CI sandbox |
| **v3** `--scheduler slurm` batch submission → PENDING→RUNNING→READY→curl→idempotent rerun→stop | **E** | CI sandbox vs a REAL single-node Slurm 23.11 |
| **v3** Flux batch submission (same code path, flux spellings) | **G** → **P** | needs a Flux cluster (no flux-core package here) |
| `bench` sweep against a live endpoint | **E** | CI sandbox |
| `pull` hf:// full repo through RamaLama | **E** | User's Mac (after SSL fix) |
| `pull`/serve interplay on **Podman** (mount, store path, argv) | **E**\* | User's Mac — \*container start blocked only by amd64-image-on-ARM |
| Degraded mode (no ramalama installed) | **E** | CI sandbox (subprocess harness) |
| `generate sky` / `launch` YAML | **E** (validated by SkyPilot 0.12.3's parser) | CI sandbox |
| Podman **CUDA** serve (vLLM) | **G** → **P** | needs HOPS-class node |
| Apptainer **ROCm** serve + OCI→SIF build | **G** → **P** | needs Eldorado-class node |
| srun / flux-run wrapping, module preamble | **G** → **P** | needs cluster |
| `sky launch` execution | **P** | needs cloud credentials |

Sections 3–5 below are the P-tier steps: run them in order and every
capability moves to **E**.

---

## 0.9 Deployment matrix — one model, every platform

boxy is modular: a **box** (the model) is portable; a **location** (the platform) is
swappable. The deploy command is the SAME `boxy serve <model>`; only the location or
`--scheduler` changes — so a new platform is a new `--location <site>.toml`, never a
code change. Proven on the smallest Llama (3.2 1B, Q4 GGUF → llama.cpp; runs on CPU
*and* GPU):

```bash
M="hf://hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf"
# (or use the shipped box:  --box examples/boxes/llama-3.2-1b.toml)

# 1) LOCAL / BAREMETAL desktop (no scheduler -> a container right here; CPU is auto)
boxy serve $M                                   # add --here if the host looks like a login node

# 2) SLURM  (submits an sbatch job, waits for READY, prints the endpoint, detaches)
boxy serve $M --scheduler slurm --gpus 1 --partition short --account <acct> --time 30:00

# 3) FLUX   (identical UX; # flux: directives + flux batch)
boxy serve $M --scheduler flux  --gpus 1

# 4) ANY OTHER PLATFORM
#    a. cloud (AWS/GCP/Azure/K8s) via SkyPilot:
boxy launch   --box examples/boxes/llama-3.2-1b.toml --location examples/locations/cloud-gpu.toml
boxy generate sky --box examples/boxes/llama-3.2-1b.toml --location <loc> -o task.yaml   # then: sky launch task.yaml
#    b. any on-prem site: write one location file and reuse every command above:
boxy serve $M --location examples/locations/mysite.toml
```
All scale/serve flags (`--replicas`, `--nodes`, `--nodes-per-replica`, `--distributed`,
`--router`, `boxy sweep`, `boxy bench`) compose on top of ANY row unchanged. For the
official gated weights on vLLM instead of the GGUF: `hf://meta-llama/Llama-3.2-1B-Instruct`
(needs an HF token). Verify a row without deploying: append `--dryrun`.

## 0.95 Submit from ANYWHERE (laptop → cluster, OTP/YubiKey-safe)

Type the same command on your laptop; boxy runs it on the cluster over SSH and
tunnels the endpoint back:

```bash
# one-shot spelling:
boxy serve <model> --scheduler slurm --gpus 4 --ssh ambelt@hops-login1.sandia.gov
# set-and-forget spelling (then EVERY boxy command is remote, verbatim):
export BOXY_SSH_HOST=ambelt@hops-login1.sandia.gov
boxy serve <model> --scheduler slurm --gpus 4
boxy list        # runs on the cluster
boxy stop <name> # runs on the cluster
# profile spelling: put `remote = "ambelt@hops-login1.sandia.gov"` in [location].
```

What happens behind the scenes:
- **One login, many commands.** boxy opens an OpenSSH **ControlMaster** session:
  your OTP prompt + YubiKey touch happen ONCE, on your terminal; the session
  persists (~4h) and every boxy command multiplexes over it with no re-prompts.
  (This is why boxy shells out to the system `ssh` — OpenSSH natively handles
  keyboard-interactive OTP and FIDO2/YubiKey; Python SSH libraries don't. Your
  ~/.ssh/config, ProxyJump/bastions included, is honored for free.)
- **The endpoint comes to you.** When the remote serve prints `### READY
  http://node:port/v1`, boxy adds a port forward ON the live session (no re-auth)
  and prints `### LOCAL http://127.0.0.1:port/v1` — point your client there. The
  tunnel lives on the SSH master, so it outlives the boxy command; close it with
  the printed `ssh -O cancel ...` line.
- **Nothing installed on the cluster** except boxy itself (`pip install` once, or
  set BOXY_REMOTE_COMMAND='source ~/venv/bin/activate && boxy' if it lives in a
  venv). No daemon, no agent — unlike VS Code Remote-SSH's server.
- Prereqs: `ssh user@login` works from this machine (VPN up), and boxy is on the
  login node. Everything else (scheduler flags, --replicas, --router, sweep)
  composes unchanged — it simply runs over there.
- **Keep the cluster's boxy current.** `--ssh` runs the CLUSTER's install, not
  your laptop's. If the remote rejects a subcommand your laptop knows
  (`invalid choice: 'logs'`), boxy prints a *stale install* hint — fix it with
  `git pull && pip install -e .` in the checkout on that login node.

### 0.96 Several clusters, one $HOME (hops + eldorado)

Lab clusters often share your home directory. boxy therefore **partitions its
job state per cluster automatically**: records/endpoints/scripts/logs live in
`~/.local/share/boxy/jobs/<cluster>/` (`hops`, `eldorado`, …), so `boxy list`,
`boxy logs`, and `boxy curl` on one cluster never surface another's — no mixing,
no confusion. The cluster name comes from the host (`eldorado-login2` →
`eldorado`, `hops42` → `hops`); if your site's hostnames don't encode it, set
`BOXY_CLUSTER=<name>` per cluster (shell profile).

Knobs (rarely needed):
- `BOXY_JOBS_ROOT` — change the base that gets the `<cluster>/` subdirs.
- `BOXY_JOBS_DIR` — pin an EXACT directory (no per-cluster nesting); the escape
  hatch if you want a single shared view. With it set, cross-cluster records DO
  co-exist, and boxy falls back to `FOREIGN(origin)` labels + foreign-endpoint
  exclusion (by the submit host's cluster identity) to keep them straight.

Note: upgrading to the partitioned layout, any pre-existing logs in the old flat
`~/.local/share/boxy/jobs/` stay there (boxy does not move your files); `boxy
logs` points at them if it finds no cluster-local match.

### 0.965 Compute node behind a corporate proxy (ghcr.io 403 / Zscaler)

A compute node often can't egress to a public registry directly — the pull dies
with `ghcr.io: StatusCode: 403 ...Zs...` (a Zscaler/proxy *policy* block; note it's
a 403, not a cert error — boxy already mounts your merged CA into the container).
Give the node your proxy and boxy carries it into the job's `podman pull` AND the
container's in-container downloads:

```bash
# provide it explicitly:
boxy serve <model> --scheduler slurm --gpus 1 --ssh user@login \
    --proxy http://proxy.mysite.gov:80 --account <acct> --time 30:00
# or just have http_proxy/https_proxy exported on the login node — boxy auto-uses them.
```

boxy prefixes the compute-node command with `env http_proxy=… https_proxy=…
no_proxy=…` (both cases; `no_proxy` preserved so intra-cluster/localhost stays
direct). If the compute nodes can't reach the proxy at all (fully air-gapped),
fall back to pre-pulling on the login node (shared `$HOME` podman store) or a
site mirror — see §0.97.

### 0.97 Pull images from YOUR registry (site mirrors, air-gap, localhost)

Every image reference resolves through one module (`registries.py`) — swap
registries with data, never code:

```bash
# blanket: send EVERY image to one registry (replaces docker.io/ghcr.io/...)
boxy serve <model> --registry registry.mysite.gov/mirror ...
# an image you built yourself, no registry at all:
boxy serve <model> --image localhost/my-vllm:dev ...
```

Per-registry rewrite map in a location profile (mirrors win over `--registry`):

```toml
[location.image_mirrors]
"docker.io" = "registry.mysite.gov/dockerhub"
"ghcr.io"   = "registry.mysite.gov/ghcr"
"*"         = "registry.mysite.gov/mirror"   # catch-all; omit to leave others alone
```

Bare names (`vllm/vllm-openai`) count as docker.io. `localhost/...` images stay
local unless explicitly mirrored. The rewrite applies uniformly: podman/docker
run, apptainer's OCI→SIF build (`docker://<rewritten>`), and the SkyPilot export.

## 1. Any machine — install & self-test (5 min)

```bash
git clone -b claude/boxy-cli-hpc-spec-ojevsl <repo-url> && cd */boxy

# with pip:
python3 -m venv .boxy && source .boxy/bin/activate
pip install -e '.[ramalama,test]'
# or with uv:
uv venv .boxy && source .boxy/bin/activate
uv pip install -e '.[ramalama,test]'

pytest -q            # EXPECT: 113 passed (live Docker test skips if no Docker)
boxy info            # EXPECT: version, ramalama available, your runtimes/schedulers
```

**uv users:** uv's standalone Pythons ship without system CA wiring — step 2.1
(SSL_CERT_FILE) is *required* for you, not optional.

## 2. Laptop end-to-end (macOS/Linux, CPU — ~10 min)

Proven on an Apple-Silicon Mac with podman; these are the exact steps
including the two environment fixes that run surfaced.

**v2 one-liner (recommended):** the whole of 2.2–2.4 below is now a single
command — pull, engine/image/port choice, launch, readiness wait:

```bash
boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
# EXPECT: "auto: ..." decision lines, the container command, then
#   ### READY  http://127.0.0.1:8090/v1   (model: ...)
curl -s http://127.0.0.1:8090/v1/models
boxy stop boxy-qwen2.5-0.5b-instruct-q4_k_m       # name from the READY banner
```

If startup crashes, boxy prints the container's last log lines immediately
(no timeout wait) and removes the container. If a big model is still loading
at the readiness timeout, the container is left running and boxy prints the
`logs -f` command to watch it. The profile-based steps below remain valid and
are what you'll use once a site profile exists (`--save-profile` writes one).

```bash
# 2.1 TLS setup (uv/standalone Pythons, TLS-intercepting networks). Two facts govern this:
#   - SSL_CERT_FILE REPLACES Python's trust store (it does not add to it), so a file
#     holding only your site/proxy CA breaks every registry that is NOT intercepted
#     with that CA — hf:// can work while ollama:// fails in the same shell;
#   - OpenSSL SILENTLY ignores a missing SSL_CERT_FILE path (then everything fails).
# boxy handles the first automatically: when SSL_CERT_FILE is set and certifi is
# installed (it is a boxy dependency), pulls use a merged bundle = public CAs + your
# site CA (disable: BOXY_NO_CA_MERGE=1). So the setup is just:
export SSL_CERT_FILE=/path/to/your/site-ca.crt      # or $(python3 -m certifi) if no interception
# PERSIST IT — an export dies with its shell (new terminal = broken pulls again):
echo "export SSL_CERT_FILE=$SSL_CERT_FILE" >> ~/.zshrc   # or ~/.bashrc / your venv's bin/activate
boxy info --net
# EXPECT: "net: hf:// ... OK", "net: ollama:// ... OK" — any FAIL line names the registry
# you cannot pull from and why. `boxy info` alone shows the TLS state offline
# (and flags a missing cert file).
#
# PROXIES (corporate networks). boxy honors http_proxy/https_proxy (any case) for
# every pull and probe; `boxy info` prints the EFFECTIVE proxy map (credentials
# masked). Facts that bite:
#   - registries are all https, so https_proxy is the variable that matters. An
#     empty export is IGNORED — the classic bug is `export https_proxy="${http_proxy}"`
#     placed BEFORE http_proxy is set; boxy warns when it sees http-but-not-https.
#   - with a proxy set, DNS/connect errors are about the PROXY host (the proxy
#     resolves the target, not your machine): "nodename nor servname" = the proxy
#     hostname didn't resolve (typo, or on-network-only — unset the vars off-network).
#   - proxies commonly TLS-intercept: a registry that verified fine DIRECT can fail
#     CERTIFICATE_VERIFY_FAILED THROUGH the proxy. `boxy info --net` names the issuer
#     it saw — append that root CA to SSL_CERT_FILE (the merge above keeps public CAs).
#   - auth proxies: export https_proxy=http://user:pass@host:port (407/"Tunnel
#     connection failed" means credentials or policy).
# LAPTOP BLOCKED BUT THE LOGIN NODE WORKS? (typical: HPC login nodes carry a
# complete site proxy setup — http_proxy + https_proxy + no_proxy=.yoursite.gov —
# while the laptop's path is policy-blocked.) You don't need laptop-side registry
# access at all: every boxy command takes --ssh user@login and runs THERE, pulls
# included — `boxy serve MODEL --scheduler slurm --gpus 4 --ssh user@login` pulls
# on the cluster and tunnels the endpoint back (§0.95). Only replicate the login
# node's proxy exports on the laptop if you truly need LOCAL pulls (VPN up, so
# the site proxy resolves; keep .yoursite.gov in no_proxy so ssh stays direct).
# NO NETWORK AT ALL? A local file serves with zero network: download the GGUF
# elsewhere, copy it over, then `boxy serve /path/to/model.gguf`.

# 2.2 Pull a real model (single-file GGUF: no HF CLI needed, ~400 MB)
boxy pull hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf
# (or: boxy pull --box examples/boxes/qwen-gguf.toml)
# EXPECT: "Downloading ..." then "model available at: <store path>"
# NOTE: boxy now auto-suppresses podman's macOS "proceed without GPU?" prompt.

# 2.3 Serve it (llama.cpp engine; default image auto-resolves, multi-arch, CPU-OK)
boxy serve --box examples/boxes/qwen-gguf.toml --location examples/locations/local.toml
# EXPECT: the container line, then llama.cpp startup logs.

# 2.4 In another terminal: query, bench, lifecycle
curl -s http://127.0.0.1:8090/v1/models                       # EXPECT: qwen model id
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":24}'
# EXPECT: coherent text + usage counts
boxy bench --box examples/boxes/qwen-gguf.toml --batch-sizes 1,2,4 -o laptop.csv
boxy list                                                     # EXPECT: qwen-gguf row
boxy stop --box examples/boxes/qwen-gguf.toml                 # EXPECT: container gone
```

**HPC note (v2 guard rails):** on a cluster login node, `boxy serve MODEL`
refuses to run (shared-node policy) and prints the `--scheduler`/allocation
alternatives; inside `salloc`/`flux alloc` it runs foreground so the job owns
the server, and prints the compute node's hostname endpoint + an `ssh -L`
tunnel hint. `--here` forces login-node execution if your site allows it.

**Do NOT serve `vllm-hf.toml` on a laptop** — `vllm/vllm-openai` is a
linux/amd64 CUDA image (podman will print exactly that warning). It's for
step 3. `boxy pull --box examples/boxes/vllm-hf.toml` on the laptop IS a
valid test of the full-repo pull path.

## 3. Slurm + Podman + CUDA cluster (HOPS-class)

### 3.0 The seamless path — one command from the login node

`--scheduler` SUBMITS a batch job (sbatch / flux batch): the job re-runs boxy
on the compute node (so accelerator/image/port resolve where they're true),
publishes its endpoint over the shared filesystem, and the login-side boxy
follows it to READY. **Verified end-to-end against a real Slurm cluster**
(single-node, in the CI sandbox: submit → PENDING → RUNNING → READY → curl →
idempotent rerun → `boxy stop` = scancel).

```bash
# pre-stage once (login node has network; store is on shared $HOME):
boxy pull hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# one command; pass scheduler variables DIRECTLY — boxy maps the portable ones
# (--partition/--account/--time) and hands anything else to the active scheduler:
boxy serve hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --scheduler slurm --gpus 1 \
  --partition=short --account=fy260064 --license=tscratch:1
# EXPECT:
#   ### Submitted slurm job N  (boxy-tinyllama-...)
#   ###   job N: PENDING ... RUNNING
#   ### READY  http://<node>:8090/v1   (model: ..., slurm job N)
# Ctrl-C while waiting DETACHES (the job keeps running).

boxy list                          # job state + endpoint, plus containers
boxy stop boxy-tinyllama-...       # scancel; the job step owns the server

# Flux: the SAME flags — boxy renders them in flux's spelling (queue/bank/-t):
boxy serve <model> --scheduler flux --gpus 4 --partition pbatch --account guests

# Portable spellings if you prefer them: --partition/--account/--time translate
# per scheduler (slurm --partition / flux --queue, etc). Site defaults belong
# in a --location profile:  scheduler_args = ["--partition=short", ...]
boxy serve <model> --location hops.toml
```

Notes: `--foreground` keeps the old attached `srun`/`flux run` mode. Batch
logs live at `~/.local/share/boxy/jobs/<name>.log` — boxy dumps their tail
automatically if the job dies before READY.

```bash
# 3.1 Login node
boxy info
# EXPECT: accelerator: cuda | runtimes include podman | schedulers include slurm

# 3.2 Stage the model. Option A (connected login node):
boxy pull --box examples/boxes/vllm-hf.toml
# Option B (air-gapped): copy weights to the shared FS and use a path-based
# box (examples/boxes/vllm.toml + [location.staging] models_dir), per the paper.

# 3.3 Eyeball the exact command before running anything:
boxy serve --box examples/boxes/vllm-hf.toml --location examples/locations/hops.toml --dryrun
# EXPECT: srun --nodes=.. --gpus-per-node=.. podman run ... --device nvidia.com/gpu=all
#         ... vllm/vllm-openai:v0.24.0 serve <model> --host=0.0.0.0 --port=8000 ...
# Adjust examples/locations/hops.toml (nodes/gpus/partition-specific tuning) to your site.

# 3.4 Single-node first (inside an allocation, scheduler wrap not needed):
salloc -N1 --gpus-per-node=4
#   in the job shell: set scheduler="none" in a copy of hops.toml, then
boxy serve --box examples/boxes/vllm-hf.toml --location my-hops-1node.toml
# EXPECT: vLLM v0.24.0 engine startup, CUDA graphs, "Uvicorn running on 0.0.0.0:8000"
curl -s http://localhost:8000/v1/models

# 3.5 Then the scheduler-wrapped form from the login node (scheduler="slurm"):
boxy serve --box examples/boxes/vllm-hf.toml --location examples/locations/hops.toml

# 3.6 Benchmark (paper's step 5; from a node that can reach the serving node):
boxy bench --box examples/boxes/vllm-hf.toml --url http://<node>:8000 \
     --dataset ShareGPT_V3_unfiltered_cleaned_split.json \
     --batch-sizes 1,2,4,8,16,32,64,128,256,512,1024 -o hops-results.csv

# 3.7 Lifecycle
boxy list ; boxy stop --box examples/boxes/vllm-hf.toml
```

## 4. Flux + Apptainer + ROCm cluster (Eldorado-class)

```bash
boxy info                     # EXPECT: accelerator: rocm | apptainer | flux
# 4.1 Pre-build the SIF once (large image; uses APPTAINER cache):
boxy build --box examples/boxes/vllm.toml --location examples/locations/eldorado.toml
# EXPECT: apptainer build --force vllm-rocm.sif docker://vllm/vllm-openai:v0.24.0
# 4.2 Check the emitted command (module load rocm, --rocm, --fakeroot, tuning):
boxy serve --box examples/boxes/vllm.toml --location examples/locations/eldorado.toml --dryrun
# 4.3 Run inside a flux alloc (or let boxy wrap with flux run):
flux alloc -N1
boxy serve --box examples/boxes/vllm.toml --location my-eldorado-1node.toml
curl -s http://localhost:8000/v1/models
# NOTE eldorado.toml carries the MI300a HBM tuning (gpu-memory-utilization=0.7).
```

## 4.5 Scaling out — distributed, replicas, and sweeps

Three orthogonal ways to use more than one node/GPU. All are `--dryrun`-able:
print the exact jobs before spending an allocation.

```bash
# --- A. One instance ACROSS nodes (model-parallel, via Ray) --------------------
# Auto-on for vLLM whenever --nodes>1: tensor-parallel = GPUs/node (intra-node),
# pipeline-parallel = nodes (inter-node). Works on Slurm, Flux, or a bare set of
# containers with no scheduler. --no-distributed forces a single container.
boxy serve <model> --scheduler slurm --nodes 2 --gpus 4 --dryrun
# EXPECT: #SBATCH --ntasks-per-node=1, and the compute-node inner serve reports
#   "distributed vLLM: 2 nodes x 4 GPU -> tensor-parallel=4, pipeline-parallel=2
#    (world 8) via Ray (slurm launcher)"; a Ray head (ray start --head + vllm
#    serve --tensor-parallel-size=4 --pipeline-parallel-size=2
#    --distributed-executor-backend=ray) plus an srun worker fan-out to the other
#    node (ray start --address=$BOXY_RAY_HEAD:6379 --block).
# On the node, sanity-check the cluster formed before/while the model loads:
#   ray status          # EXPECT: 2 nodes, 8 GPUs total
#   nvidia-smi          # (run per node) EXPECT: all GPUs busy once loaded
boxy serve <model> --scheduler slurm --nodes 2 --gpus 4     # for real; READY -> curl -> boxy stop <name>

# --- B. N INDEPENDENT instances (data-parallel replicas) -----------------------
# Replicas BIN-PACK onto a node's GPUs: with --gpus = the node's GPU count and the
# default --gpus-per-replica 1, K replicas share ONE node (K // 1 per node), each
# pinned to its own GPU (CUDA/HIP/ROCR_VISIBLE_DEVICES) on its own port 8000,8001,…
# So 4 replicas on a 4-GPU node = 1 node job, NOT 4 nodes. tensor-parallel per
# replica = --gpus-per-replica (1 by default; raise it for a bigger model).
boxy serve <model> --scheduler slurm --gpus 4 --replicas 4 --dryrun
# EXPECT: "4/node across 1 node job(s)"; one #SBATCH --gpus-per-node=4 job that
#   launches 4 GPU-pinned servers (--visible-gpus 0..3, --port 8000..8003) + wait.
boxy serve <model> --scheduler slurm --gpus 4 --replicas 4      # for real (1 node, 4 GPUs)
boxy list                                                       # the job + its 4 replica endpoints
boxy stop <base>                                                # cancels the job -> all 4 replicas
# Knobs:
#   --gpus-per-replica 2  -> 2 replicas/4-GPU node, each tensor-parallel=2.
#   --nodes N             -> the POOL SIZE: spread K replicas across N nodes
#                           (12 replicas --nodes 4 = 3/node across 4 node jobs).
#                           NOT per-replica. Errors if K needs > gpus//R per node.
#   (no --nodes)          -> tight-pack gpus//R per node -> ceil(K/rpn) node jobs.
#   --nodes-per-replica M -> each replica is itself an M-node distributed (Ray)
#                           instance (data-parallel of model-parallel; total = K x M).
# Note: GPU pinning uses absolute indices, correct for the exclusive full-node
# allocations HPC partitions grant (--gpus = the node's GPU count).

# --- B''. Models that need an extra pip package (build your own --image) -------
# Some models import a package the stock vLLM image lacks (e.g. a custom VLM vision
# tower needs open_clip_torch — note the PyPI name differs from the import name).
# Build a thin derived image once, then serve it with --image:
printf 'FROM docker.io/vllm/vllm-openai:v0.24.0\nRUN pip install open_clip_torch\n' > Dockerfile.boxy
podman build -t localhost/vllm-extra:latest -f Dockerfile.boxy .
boxy serve hf://nvidia/NVIDIA-Nemotron-Parse-v1.2 --scheduler slurm --gpus 4 \
    --trust-remote-code --image localhost/vllm-extra:latest
# NOTE: the image must be visible where the container RUNS. If the compute node
# doesn't share the login node's podman store, build on the compute node (salloc,
# then podman build there) or push the tag to a registry your site provides.
# (boxy's crash diagnosis prints this recipe when it sees the ImportError.)

# --- B'. ONE URL in front of the replicas (built-in router) --------------------
# Present a single OpenAI endpoint load-balanced (least-outstanding) across the K
# replicas. Runs on the login node; discovers replicas from the endpoint files and
# fails over if one dies. Two ways:
boxy serve <model> --scheduler slurm --gpus 4 --replicas 4 --router  # submit + front on :8000
boxy router <base>                                                  # front an existing set (see boxy list)
# then point any OpenAI client at http://<login-node>:8000/v1 (ssh -L 8000:<login>:8000).
# For production scale (TLS/auth/>~hundreds of concurrent streams), emit a real proxy
# config from the live endpoints instead of running the built-in one:
boxy router <base> --emit nginx   > boxy.conf      # or --emit haproxy | litellm
# (See the scaling note below: the built-in router is right for benchmark scale; the
#  GPUs, not the proxy, are the bottleneck. Beyond that, run the emitted nginx/haproxy/litellm.)

# --- B''''. N instances: unique URLs, or ONE route -----------------------------
# Running N servers on a cluster, you address them ONE of two ways:
#   (a) UNIQUE URL per instance — different models, or you want each individually.
#       Every instance has its own name -> its own endpoint (compute-node:port).
#       `boxy list` shows all their URLs; target one by name:
boxy serve <modelA> --scheduler flux --gpus 1 --unique      # -> name-A on nodeX:8090
boxy serve <modelB> --scheduler flux --gpus 1 --unique      # -> name-B on nodeY:8090
boxy list  --ssh user@login                                 # every instance + its URL
boxy curl  name-A --ssh user@login                          # query a specific one
boxy open  name-A --ssh user@login                          # browser: tunnels name-A to a FREE local port
boxy open  name-B --ssh user@login                          # a DIFFERENT free local port -> both in the browser at once
boxy open  name-A --ssh user@login --port 8080              # PIN the local port -> stable URL http://127.0.0.1:8080/
# Custom URL/domain: boxy binds loopback; for a name like http://mymodel.local:8080/
# add `127.0.0.1  mymodel.local` to /etc/hosts, then --port 8080. Tunnel lifetime is
# the SSH master's: default 12h idle, override per your site's session cap:
#   export BOXY_SSH_PERSIST=8h   (OpenSSH formats: 30m, 12h, ...; one OTP+touch covers it)
#   (b) ONE route for N of the SAME model — use --replicas + the router (B'):
#       all K replicas behind a single load-balanced URL; the client sees one endpoint.
boxy serve <model> --scheduler flux --gpus 4 --replicas 4 --router
# `boxy open` picks a fresh local port each call, so opening several instances never
# collides (field report: a stale forward on 8090 blocked the browser).

# --- C. Scaling SWEEP (the paper's study) --------------------------------------
# Step one axis in powers of two; each rung is submitted, waited-to-READY,
# benchmarked, and torn down; a comparison table is printed at the end.
boxy sweep <model> --scheduler slurm --gpus 4 --sweep-nodes 1,2,4,8 --dryrun
boxy sweep <model> --scheduler slurm --gpus 4 --sweep-nodes 1,2,4,8 -o scaling.csv
# EXPECT (### Scaling results): nodes | servers | peakBS | req/s | tok/s | p50 | p95 | speedup
# --sweep-replicas 1,2,4,8 sweeps the data-parallel axis instead (fleet-aggregated
# throughput). --keep leaves rungs up; --max-tokens / --batch-sizes / --dataset as
# in `boxy bench`.
```

Prereqs for A/C: the model must sit on a within-cluster shared FS reachable by all
allocated nodes (the store or `--models-dir`); the checkpoint-completeness guard
catches a partial copy before the long load. Each cluster is self-contained — run
the same command on each login node; state lives in that cluster's own dir
(`~/.local/share/boxy/jobs/<cluster>/`, partitioned automatically even when `$HOME`
is shared across sites — see §0.96).

## 5. Cloud (SkyPilot delegation; optional)

```bash
pip install 'boxy-hpc[cloud]' && sky check      # needs cloud credentials
boxy launch --box examples/boxes/vllm-hf.toml \
     --location examples/locations/cloud-gpu.toml --serve
# EXPECT: task YAML path, then "sky serve up -n vllm-hf ... --yes" output
boxy launch --box ... --location ... --serve --down     # teardown
```

## 6. Troubleshooting (every failure observed so far, with its fix)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `SSL: CERTIFICATE_VERIFY_FAILED` on pull, no `SSL_CERT_FILE` set | Python without CA bundle (uv/standalone) or TLS-intercepting proxy | `pip install certifi; export SSL_CERT_FILE=$(python3 -m certifi)` — or your site CA bundle. **Persist it** (`~/.zshrc`/`~/.bashrc` or the venv's `bin/activate`) — an `export` dies with its shell. |
| `CERTIFICATE_VERIFY_FAILED` **with** `SSL_CERT_FILE` set (e.g. hf:// works, ollama:// fails) | `SSL_CERT_FILE` *replaces* the trust store: a site-CA-only file breaks non-intercepted registries; a missing path is silently ignored (everything breaks) | `ls -l "$SSL_CERT_FILE"` first. boxy auto-merges certifi's public CAs with your site CA at pull time (certifi is a boxy dep; `BOXY_NO_CA_MERGE=1` disables). Diagnose per registry: `boxy info --net`. |
| Interactive *"proceed without GPU?"* prompt (macOS podman) | RamaLama's applehv check | Fixed — boxy suppresses it automatically. Re-enable: `export RAMALAMA_USER__NO_MISSING_GPU_PROMPT=false` |
| `huggingface cli download not available` | RamaLama 0.23's repo-pull fallback is unimplemented; the *direct* download failed first | boxy now shows the root-cause error + remedy instead of this dead-end message |
| `workdir "..." does not exist on container` (Podman) | box sets a workdir no volume provides | Fixed in examples; boxy now warns before launch on any box with this pattern |
| `image platform (linux/amd64) does not match ... (linux/arm64)` | vLLM images are amd64/CUDA | Expected on Apple Silicon — use `qwen-gguf.toml` locally; vLLM boxes belong on the cluster |
| `no container runtime found on host` | login node without podman/docker/apptainer | set `[location].runtime` explicitly / load the site's container module |
| `podman is on PATH but its probe failed` | rootless podman broken (no subuid ranges, storage on NFS) or unreachable docker daemon | boxy auto-falls-through to the next working runtime; pin with `--runtime` or fix per the message |
| `this looks like a slurm login node ... refusing` | `boxy serve MODEL` outside an allocation on a scheduler host | intended guard — submit with `--scheduler slurm --gpus N`, or run inside `salloc`/`flux alloc`, or force with `--here` |
| `--gpus/--nodes ... have no effect without --scheduler` | job-request flags without a job | add `--scheduler slurm\|flux`, or drop the flags (GPU pass-through follows the detected accelerator) |
| Rootless podman fails inside `salloc`/`flux alloc` | stale XDG session vars | boxy unsets `XDG_SESSION_ID`/`XDG_RUNTIME_DIR` automatically at launch |
| Port already in use | previous serve still running | v2 auto-advances to the next free port (printed as an `auto: port:` line); `boxy list` then `boxy stop NAME` to reclaim |
| `endpoint not ready within N s` but container still running | big model still loading | boxy prints the last log lines + `docker/podman logs -f NAME`; raise `--ready-timeout` |
| `server exited during startup` + log dump | engine crashed (bad flag, OOM, bad model) | read the dumped engine logs — boxy removed the crashed container already |
| Launching a 2nd local instance killed the 1st (it vanished from `podman ps`) | the podman/docker VM ran out of RAM and its OOM killer reaped one server — NOT boxy (boxy never touches another instance). Confirm: `podman ps -a` shows the old one as `Exited (137)` | raise the VM memory: `podman machine stop && podman machine set --memory 8192 --cpus 4 && podman machine start` (docker: Desktop→Settings→Resources). boxy now prints this diagnosis when it sees exit 137/OOMKilled |
| Rerunning `boxy serve MODEL` replaced my running instance | intended: without `--unique` the name is a per-model singleton, so a rerun REDEPLOYS (stop+rm+relaunch) | to run a SECOND instance alongside, add `--unique` (fresh timestamped name + auto-incremented port); each `--unique` instance is independent (own record/endpoint/log) |
| `no space left on device` pulling vLLM image (podman machine) | vLLM images are ~20 GB; the podman VM disk is small | `podman system prune -a`; do vLLM pulls on the cluster, or grow the VM: `podman machine stop && podman machine set --disk-size 200` |
