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
| Unit/regression suite (115 tests, 96% cov) | **E** | CI sandbox |
| serve→inference→list→stop, llama.cpp on **Docker** | **E** | CI sandbox (air-gapped) |
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

pytest -q            # EXPECT: 115 passed (live Docker test skips if no Docker)
boxy info            # EXPECT: version, ramalama available, your runtimes/schedulers
```

**uv users:** uv's standalone Pythons ship without system CA wiring — step 2.1
(SSL_CERT_FILE) is *required* for you, not optional.

## 2. Laptop end-to-end (macOS/Linux, CPU — ~10 min)

Proven on an Apple-Silicon Mac with podman; these are the exact steps
including the two environment fixes that run surfaced.

```bash
# 2.1 If on a network with TLS interception, or using a uv/standalone Python:
pip install certifi
export SSL_CERT_FILE=$(python3 -m certifi)          # or your site CA bundle
python3 -c "import urllib.request as u; print(u.urlopen('https://huggingface.co').status)"
# EXPECT: 200.  If CERTIFICATE_VERIFY_FAILED -> point SSL_CERT_FILE at your site CA.

# 2.2 Pull a real model (single-file GGUF: no HF CLI needed, ~400 MB)
boxy pull --box examples/boxes/qwen-gguf.toml
# EXPECT: "Downloading ..." then "model available at: <store path>"
# NOTE: boxy now auto-suppresses podman's macOS "proceed without GPU?" prompt.

# 2.3 Serve it (llama.cpp engine; default image auto-resolves, multi-arch, CPU-OK)
boxy serve --box examples/boxes/qwen-gguf.toml --location examples/locations/local.toml
# EXPECT: the container line (args follow the image directly — boxy defers to
# the image's own ENTRYPOINT), then llama.cpp startup logs.

# 2.4 In another terminal: query, bench, lifecycle
curl -s http://127.0.0.1:8090/v1/models                       # EXPECT: qwen model id
curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":24}'
# EXPECT: coherent text + usage counts
boxy bench --box examples/boxes/qwen-gguf.toml --batch-sizes 1,2,4 -o laptop.csv
boxy list                                                     # EXPECT: qwen-gguf row
boxy stop --box examples/boxes/qwen-gguf.toml                 # EXPECT: container gone
```

**Do NOT pull/serve the vLLM boxes on a laptop** — `vllm/vllm-openai` is a
linux/amd64 CUDA image and ~20 GB (it can fill the podman VM disk). They're
for step 3. A laptop `boxy pull --box examples/boxes/vllm-hf.toml` of the
MODEL (~1 GB) is fine and valid.

## 3. Slurm + Podman + CUDA cluster (HOPS-class)

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
| `SSL: CERTIFICATE_VERIFY_FAILED` on pull | Python without CA bundle (uv/standalone) or TLS-intercepting proxy | `pip install certifi; export SSL_CERT_FILE=$(python3 -m certifi)` — or your site CA bundle. boxy now prints this remedy itself. |
| Interactive *"proceed without GPU?"* prompt (macOS podman) | RamaLama's applehv check | Fixed — boxy patches the prompt away at the seam (hard guarantee, no env var needed). |
| `huggingface cli download not available` | RamaLama 0.23's repo-pull fallback is unimplemented; the *direct* download failed first | boxy now shows the root-cause error + remedy instead of this dead-end message |
| `workdir "..." does not exist on container` (Podman) | box sets a workdir no volume provides | Fixed in examples; boxy now warns before launch on any box with this pattern |
| `executable file `llama-server` not found in $PATH` | upstream llama.cpp image keeps its binary at /app/llama-server, off $PATH | Fixed — boxes without an explicit entrypoint defer to the image's own ENTRYPOINT (podman/docker pass args only; apptainer uses `run`) |
| `image platform (linux/amd64) does not match ... (linux/arm64)` | vLLM images are amd64/CUDA | Expected on Apple Silicon — use `qwen-gguf.toml` locally; vLLM boxes belong on the cluster |
| `no space left on device` pulling vLLM image (podman machine) | vLLM images are ~20 GB; the podman VM disk is small | `podman system prune -a`; do vLLM pulls on the cluster, or grow the VM: `podman machine stop && podman machine set --disk-size 200` |
| `no container runtime found on host` | login node without podman/docker/apptainer | set `[location].runtime` explicitly / load the site's container module |
| Rootless podman fails inside `salloc`/`flux alloc` | stale XDG session vars | boxy unsets `XDG_SESSION_ID`/`XDG_RUNTIME_DIR` automatically at launch |
| Port already in use | previous serve still running | `boxy list` then `boxy stop --box <box>` |
