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

# one command; ANY sbatch flag passes through as --slurm-FLAG[=VALUE]:
boxy serve hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --scheduler slurm --gpus 1 \
  --slurm-partition=short --slurm-account=fy260064 --slurm-license=tscratch:1
# EXPECT:
#   ### Submitted slurm job N  (boxy-tinyllama-...)
#   ###   job N: PENDING ... RUNNING
#   ### READY  http://<node>:8090/v1   (model: ..., slurm job N)
# Ctrl-C while waiting DETACHES (the job keeps running).

boxy list                          # job state + endpoint, plus containers
boxy stop boxy-tinyllama-...       # scancel; the job step owns the server

# Flux is identical; flags pass through as --flux-FLAG[=VALUE]:
boxy serve <model> --scheduler flux --gpus 4 --flux-queue=pbatch --flux-bank=guests

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
| `a container named 'boxy-…' already exists` | same model already being served | `boxy stop <name>`, or serve under `--name` |
| `no space left on device` pulling vLLM image (podman machine) | vLLM images are ~20 GB; the podman VM disk is small | `podman system prune -a`; do vLLM pulls on the cluster, or grow the VM: `podman machine stop && podman machine set --disk-size 200` |
