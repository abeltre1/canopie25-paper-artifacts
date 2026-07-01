# `boxy` — CLI Specification

**A unified, site-portable, offline-first CLI for deploying and serving
containerized GenAI/LLM services on HPC systems.**

Status: Draft design specification (v0.1)
Audience: maintainers scoping the implementation of `boxy`
Baseline repo: `abeltre1/canopie25-paper-artifacts` (this repo)

---

## 1. Overview & Goals

`boxy` is a command-line tool that lets a user define a containerized GenAI
service **once** (a *box*) and deploy it across heterogeneous HPC and cloud
sites (a *location*) without rewriting launch scripts for each site's container
runtime, scheduler, and accelerator.

It generalizes the working bash prototype in this repository
(`hpc-workflow/common_boxy.sh`, `hpc-workflow/boxy-run-vllm.sh`) into a
maintainable tool, and reuses the container/model infrastructure already built
in [RamaLama](https://github.com/abeltre1/ramalama) rather than reinventing it.

### Goals

- **One definition, many sites.** A single `box` (the app/service) runs on many
  `location`s (Slurm, Flux, or laptop; Podman, Apptainer, or Enroot; CUDA or
  ROCm) by changing only the location, not the box.
- **Serve, not just run, on HPC.** Launch an OpenAI-compatible inference
  endpoint as a batch/interactive job on a Slurm or Flux cluster, with a stable
  route to the service.
- **Runtime-agnostic.** Treat the container runtime as a pluggable backend
  (§4b), so the same box runs under Podman on one site and Apptainer/SIF on
  another.
- **Offline / air-gapped by design.** Everything works fully disconnected from
  the internet: model staging from a shared filesystem or a site-local S3, and
  the offline environment variables the paper documents.
- **Reuse, don't reinvent.** Build on RamaLama's model transports, container
  engine abstraction, GPU detection, and layered TOML config (§3, §6b).

### Non-goals

- `boxy` does **not** provision clusters (that is AWS ParallelCluster / Azure
  CycleCloud / Google Cluster Toolkit territory) — it runs *on top of* an
  existing HPC allocation.
- `boxy` does **not** implement a container runtime, a scheduler, or an
  inference engine. It orchestrates existing ones.
- `boxy` is **not** a training/experiment platform (that is Determined AI /
  Ray Train territory). Its focus is deployment, serving, and benchmarking of
  inference services.

---

## 2. Motivation / HPC Context (from the paper)

This spec is grounded in the CANOPIE-HPC / SC25 paper *"Experience Deploying
Containerized GenAI Services at an HPC Center"* (Beltre, Ogden, Pedretti; DOI
[10.1145/3731599.3767356](https://doi.org/10.1145/3731599.3767356); arXiv
2509.20603), whose artifacts live in this repository. The paper is effectively
the **problem statement** for `boxy`: it documents that *"each of these systems
provides a significantly different user interface for launching the exact same
containerized software,"* and shows concrete breakage (e.g. vLLM expects to run
as root, but Apptainer maps the calling user and home directory, crashing the
container).

The HPC realities `boxy` must abstract, all present in this repo's artifacts:

| Concern | What the paper/artifacts show |
| --- | --- |
| **Multiple container runtimes** | Podman **and** Apptainer/SIF for the *same* workload (`common_boxy.sh` builds both command forms). |
| **Rootless / unprivileged** | Apptainer `--fakeroot`, `--writable-tmpfs`, `--cleanenv`, `--no-home` to work around the vLLM-expects-root problem. |
| **GPU pass-through** | CUDA: `--device nvidia.com/gpu=all` (Podman) / `--nv` (Apptainer). ROCm: `/dev/kfd` + `/dev/dri`, `--group-add=video`, `seccomp=unconfined` (Podman) / `--rocm` (Apptainer). |
| **Batch schedulers** | SLURM `salloc` and Flux `flux alloc` (`0-alloc-compute-node.sh`, 2 nodes × 4 GPUs). XDG env vars must be cleared inside interactive jobs. |
| **Multi-node / multi-GPU** | Tensor parallelism for large models (Llama-4-Scout on 4×80GB; Llama-3.1-405B needs 16×80GB via Ray-on-Slurm). |
| **Offline / air-gapped** | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `HF_HUB_ENABLE_HF_TRANSFER=0`. |
| **Shared FS + object storage** | Models staged to a shared path and mounted into the container; sync to a site-local S3 (`2-upload-to-s3.sh`). |
| **Module systems** | `module load rocm/6.4.0` before ROCm Apptainer runs. |
| **Site heterogeneity** | **HOPS** (NVIDIA H100, SLURM) vs **Eldorado** (AMD MI300a, Flux), incl. MI300a-specific `--gpu-memory-utilization=0.7` to leave HBM for the OS. |
| **Determinism** | `--seed=12345`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `OMP_NUM_THREADS=1`. |

---

## 3. What Already Exists (both repos)

### 3.1 canopie25 prototype (this repo)

The `hpc-workflow/` directory contains a **working bash prototype** of `boxy`
plus the surrounding pipeline. This is the reference implementation `boxy`
formalizes.

| Artifact | Role |
| --- | --- |
| `hpc-workflow/common_boxy.sh` | The "boxy library": platform detection (`hops`→CUDA, `eldorado`→ROCm), image selection, offline env-var injection, **Podman and Apptainer command builders**, GPU pass-through args, auto-build of the Apptainer `.sif` from an OCI image. |
| `hpc-workflow/boxy-run-vllm.sh` | Prototype entrypoint: pick runtime (podman/apptainer), sanity-check it, build and `eval` the command to serve vLLM. |
| `hpc-workflow/0-alloc-compute-node.sh` | Allocate nodes via SLURM `salloc` / Flux `flux alloc`. |
| `hpc-workflow/1-download-models.sh` | Download models from HuggingFace (Podman + git container). |
| `hpc-workflow/2-upload-to-s3.sh` | Sync models to a site-local S3. |
| `hpc-workflow/3-start-vllm-llama4-scout.sh` | Serve vLLM via boxy. |
| `hpc-workflow/4-download-benchmark-dataset.sh` | Fetch ShareGPT benchmark data. |
| `hpc-workflow/5-run-benchmark.sh` | Throughput/latency sweep, batch 1–1024. |
| `k8s-workflow/` | Helm-chart alternative (vLLM on Kubernetes/OpenShift) — the "cloud-native" comparison point. |
| `plots/` | Gnuplot scripts + benchmark results (single-/multi-node). |

The prototype's own comment states the intended design and is the north star for
this spec:

> *"the actual tool would be written in Python/Rust/something and use a modular
> 'box' definition scheme to define the containerized application/service and a
> modular 'location' definition to define particulars for the target site /
> container cluster execution environment."*

### 3.2 RamaLama reusable infrastructure

RamaLama is a mature Python CLI for running/serving models in containers. It
supplies most of what `boxy` needs **except any HPC support**. `boxy` reuses:

| RamaLama subsystem | Path | What `boxy` gets |
| --- | --- | --- |
| Container engine abstraction | `ramalama/engine.py` (`BaseEngine`) | Podman/Docker command building; base for a pluggable runtime backend (§4b). |
| GPU detection | `ramalama/common.py` — `get_accel()`, `get_accel_env_vars()`, `accel_image()` | Detect CUDA/ROCm/Intel/…; per-accelerator env vars and image selection. Replaces the prototype's hardcoded `$CLUSTER` switch. |
| Layered config | `ramalama/config.py` (`BaseConfig`, TOML + env overrides) | Foundation for the `box`/`location` config model (§4). |
| Model transports | `ramalama/transports/` | Pull models from HF, Ollama, OCI, ModelScope, RLCR, file/http. Powers `boxy pull`. |
| Model store | `ramalama/model_store/` | Local blob/ref/snapshot store, checksums — local caching + staging. |
| Inference runtimes | `ramalama/plugins/runtimes/inference/` | llama.cpp / vLLM / MLX plugins — the engines inside a box. |
| Manifest generation | `ramalama/quadlet.py`, `kube.py`, `compose.py` | Quadlet/Kube/Compose output — a model for `boxy`'s HPC batch-script generation. |
| CLI dispatch pattern | `ramalama/cli.py` | argparse subcommand structure to mirror. |

**RamaLama HPC gaps `boxy` fills:** no Apptainer/Singularity, no SLURM/Flux, no
multi-node/tensor-parallel orchestration, no batch scheduling, no offline/air-gap
workflow, no S3 staging, no site/"location" abstraction.

---

## 3b. Market Landscape & Prior Art

*Survey current as of mid-2026. Fast-moving facts are dated and sourced (§9).*
The headline: **the exact niche — a unified, site-portable CLI that serves
containerized LLMs on a Slurm/Flux HPC cluster or elsewhere, abstracting runtime
× scheduler × accelerator, and works air-gapped — is essentially unoccupied.**

### HPC container runtimes

| Tool | HPC/Cloud | LLM serving? | Scheduler | Limitation vs `boxy` |
| --- | --- | --- | --- | --- |
| **Apptainer/Singularity** | HPC | No (runtime) | any | Runtime only — no model pull, serving, or scheduler orchestration. **Wrap it.** |
| **Charliecloud** (LANL) | HPC | No | any | Fully-unprivileged runtime only; smaller ecosystem. |
| **Sarus / Shifter** | HPC | No | native hooks | Runtime only; low momentum / legacy. |
| **NVIDIA Enroot + Pyxis** | HPC | No | **Slurm** (SPANK) | Slurm-only, NVIDIA-centric; no serving layer. **Integrate as a location backend.** |
| **Podman-on-HPC** | Both | No (engine) | via Slurm `--container` | No scheduler abstraction by itself. RamaLama already uses it. |
| **Slurm OCI / `scrun`** | HPC | No | Slurm | Low-level plumbing to target, not a serving tool. |

### Build tools (complementary, build-time)

**HPC Container Maker (HPCCM)**, **Spack**, **EasyBuild** — image authoring, not
deployment/serving. Optional *build backends* for site-tuned images.

### AI/GPU orchestrators

| Tool | HPC/Cloud | Serving? | Runtime | Scheduler | Limitation vs `boxy` |
| --- | --- | --- | --- | --- | --- |
| **SkyPilot** | Both (Slurm added 2026) | Yes — **but not on Slurm** | Docker; **Enroot/Pyxis on Slurm** | K8s, Slurm, cloud | **Closest competitor.** SkyServe serving not supported on Slurm; no Apptainer/SIF; weak air-gap story. |
| **dstack** | Cloud + on-prem | task/service | Docker | own | No Slurm/Flux or SIF; cloud-native oriented. |
| **Run:ai** (NVIDIA) | K8s | via K8s | Docker/K8s | **K8s-only** | No Slurm/Flux, no Apptainer, NVIDIA-only. |
| **Ray + Ray Serve / KubeRay** | Both | Yes | Docker/K8s | K8s native; Slurm by hand | Ray-on-Slurm is DIY; heavy Ray dependency. |
| **Modal** | Cloud only | Yes | proprietary | managed | Hosted SaaS; no on-prem/HPC/air-gap. |
| **Determined AI** (HPE) | Both | **training-focused** | **Apptainer/Singularity/Podman/Enroot** | **Slurm & PBS** | Nearest multi-runtime+Slurm design, but training platform, not a serving CLI. Good prior art. |

### LLM serving peers (RamaLama's neighbors)

| Tool | Runtime | Scheduler | Limitation vs `boxy` |
| --- | --- | --- | --- |
| **RamaLama** (base) | Podman/Docker only | **none** | No HPC/Slurm/Apptainer/multi-node/air-gap. **`boxy` = RamaLama + HPC layer.** |
| **vLLM** (+ production-stack / **llm-d**) | Docker/Apptainer | Slurm via manual Ray; llm-d is K8s-native | The engine, not an orchestrator; multi-node-on-Slurm is DIY. **Serve it as a box.** |
| **NVIDIA NIM / NeMo** | Docker | K8s | Air-gap is a known pain; NVIDIA-only, licensed. |
| **KServe** | Docker/K8s | **K8s-only** | No Slurm/Flux, no SIF, no offline CLI. |
| **Ollama / llama.cpp / MLX / TGI** | own / Docker | none / K8s | Engines / single-node runners; `boxy` wraps them. |

### HPC-in-cloud provisioning (orthogonal)

**AWS ParallelCluster**, **Azure CycleCloud**, **Google Cluster Toolkit**,
**NVIDIA Base Command** — they *provision* the cluster `boxy` runs on; different
layer.

### Closest purpose-built prior art

- **GWDG Chat AI / SAIA** (arXiv 2407.00110; `github.com/gwdg/saia-hub`,
  `saia-hpc`) — Slurm-native vLLM serving, but a **single-site web-service
  architecture** (cloud front-end + SSH proxy into one cluster), not a portable
  box/location CLI.
- **This repo's paper** (arXiv 2509.20603) — deploys the same vLLM/Llama
  container across Podman, Apptainer, and K8s and documents exactly the
  cross-runtime UI fragmentation `boxy` removes.

**Strategic context:** NVIDIA's Dec-2025 acquisition of SchedMD/Slurm tightens
the NVIDIA / Enroot / Pyxis / NIM vertical, making `boxy`'s vendor-neutral,
multi-runtime, **CUDA + ROCm** stance a durable differentiator.

---

## 4. Core Concepts: `box` and `location`

`boxy` separates *what* you deploy from *where* you deploy it. Both are declared
declaratively (TOML, layered on RamaLama's config model) and either can be
overridden on the CLI.

### 4.1 `box` — the containerized app/service

A generalization of the vLLM block in `common_boxy.sh`.

```toml
# boxes/vllm-llama4-scout.toml
[box]
name          = "vllm-llama4-scout"
image         = "vllm/vllm-openai:v0.9.1"   # OCI ref; SIF derived per-runtime
entrypoint    = "vllm"
model         = "hf://meta-llama/Llama-4-Scout-17B-16E-Instruct"
workdir       = "/vllm-workspace/models"
ports         = [8000]

[box.env]                 # merged with location's offline/accel env
OMP_NUM_THREADS = "1"
VLLM_DISABLE_COMPILE_CACHE = "1"

[[box.volumes]]           # host paths resolved by the location
source = "${MODELS_DIR}"
target = "/vllm-workspace/models"

[box.args]                # engine args appended last (don't override user)
tensor_parallel_size = 4
seed = 12345
```

The box is **runtime-agnostic and accelerator-agnostic**: it never names Podman,
Apptainer, CUDA, or ROCm. Those come from the location.

### 4.2 `location` — the target site / execution environment

A generalization of the `hops`/`eldorado` `$CLUSTER` switch.

```toml
# locations/eldorado.toml
[location]
name        = "eldorado"
scheduler   = "flux"          # slurm | flux | none
accelerator = "rocm"          # cuda | rocm | intel | ... (autodetect if unset)
runtime     = "apptainer"     # pluggable backend, see §4b (autodetect if unset)
registry    = ""              # optional local registry prefix
offline     = true            # inject HF_HUB_OFFLINE etc.

[location.resources]
nodes           = 2
gpus_per_node   = 4

[location.modules]            # loaded before container launch
load = ["rocm/6.4.0"]

[location.staging]            # where models live / how they're fetched
models_dir = "./models"       # shared-FS path mounted as MODELS_DIR
s3_endpoint = "http://localhost:9000"   # optional site-local object store

[location.tuning]             # site quirks, e.g. MI300a shared HBM
gpu_memory_utilization = 0.7
```

Running a box on a location resolves everything the prototype hardcodes:
image variant, GPU pass-through, offline env, module loads, and site tuning.

```
boxy serve --box vllm-llama4-scout --location eldorado
```

---

## 4b. Container Runtime Backends (pluggable)

The container runtime is a **swappable backend**, not a hardcoded Podman/Apptainer
pair. A common `RuntimeBackend` contract does four things, and each backend
implements them:

1. build the `run`/`exec` command;
2. inject environment variables;
3. map GPU devices for the target accelerator;
4. mount volumes.

The `box` stays runtime-agnostic; the `location` selects the backend; `boxy`
auto-detects the site's available runtime (preferring it, allowing explicit
override), extending RamaLama's `get_default_engine()` pattern.

| Backend | Status | Notes |
| --- | --- | --- |
| **Podman** | first-class | RamaLama-native + prototype `build_podman_command`. CUDA: `--device nvidia.com/gpu=all`. ROCm: `--group-add=video --cap-add=SYS_PTRACE --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined`. |
| **Apptainer/Singularity** | first-class | Prototype `build_apptainer_command` + auto OCI→SIF build. `--nv` (CUDA) / `--rocm` (ROCm), `--fakeroot`, `--cleanenv`, `--writable-tmpfs`, `--no-home`. |
| **Docker** | reuse | RamaLama's existing Docker path — dev / non-HPC sites. |
| **NVIDIA Enroot (+ Pyxis)** | planned | Slurm-native NVIDIA sites via the Pyxis SPANK plugin (`srun --container-image=...`); the common path on many DOE/NVIDIA clusters (and what SkyPilot uses). |
| **Slurm `scrun` / OCI** | planned | Run OCI images as Slurm jobs via `oci.conf` where a site standardizes on native Slurm container support. |
| **Charliecloud, Sarus/Shifter** | future | The interface admits them (fully-unprivileged / Docker-format HPC runtimes); not committed for v1. |

Each backend also declares its **image format** — OCI (Podman/Docker/Enroot),
SIF (Apptainer), squashfs (Enroot) — so `boxy build` (§5) can convert/build as
needed while the box definition stays unchanged.

---

## 5. CLI Surface (full pipeline + serve + bench)

`boxy` mirrors the paper's 0–5 pipeline and RamaLama's verbs. Every command
takes `--box` and `--location` (or uses the active defaults).

| Command | Purpose | Reuses | New HPC code |
| --- | --- | --- | --- |
| `boxy alloc` | Request nodes via SLURM `salloc` / Flux `flux alloc`; clear XDG vars in interactive jobs. | — | scheduler adapter |
| `boxy pull` | Fetch a model from HF/Ollama/OCI/ModelScope/RLCR. | RamaLama transports + model_store | — |
| `boxy stage` | Place model on shared FS / sync to site-local S3; offline-aware. | RamaLama model_store | S3 + shared-FS staging |
| `boxy build` | Build/convert image for the location's runtime (OCI→SIF for Apptainer, etc.). | `accel_image()` | SIF build (auto from OCI) |
| `boxy serve` | Launch the box as a service (vLLM/llama.cpp) via the selected runtime + scheduler; expose OpenAI-compatible endpoint. | RamaLama runtime plugins | runtime backend + scheduler submit + routing |
| `boxy run` | Interactive / one-shot inference against a box. | RamaLama `run` | scheduler-aware launch |
| `boxy bench` | Throughput/latency sweep (ShareGPT, batch 1–1024); emit plot-ready data. | (wrap a proven offline benchmarker, e.g. GuideLLM) | batch sweep + result export |
| `boxy list` | List boxes, locations, and running services. | RamaLama `list` | scheduler job state |
| `boxy stop` | Stop a running service / cancel its job. | RamaLama `stop` | `scancel` / `flux cancel` |
| `boxy info` | Show detected accelerator, available runtimes, scheduler, site config. | `get_accel()`, `get_default_engine()` | scheduler/runtime probe |

A typical HPC session:

```
boxy alloc  --location hops            # 2 nodes x 4 GPUs via Slurm
boxy pull   --box vllm-llama4-scout    # offline-aware model fetch
boxy stage  --box vllm-llama4-scout --location hops
boxy build  --box vllm-llama4-scout --location hops   # OCI->SIF if Apptainer
boxy serve  --box vllm-llama4-scout --location hops   # OpenAI endpoint on :8000
boxy bench  --box vllm-llama4-scout --location hops   # ShareGPT sweep
```

---

## 6. Architecture Options — the "add-on" of each

`boxy` reuses RamaLama, but *how* it reuses it is a real decision. The three
options differ in what each **adds** on top of RamaLama and what it couples.

| # | Option | What it adds | Cost / coupling |
| --- | --- | --- | --- |
| **A** | **Standalone tool, reuse RamaLama as a library** | A new HPC orchestration package: runtime backends (Apptainer/Enroot/scrun), scheduler adapters (Slurm/Flux), and the box/location layer — built on RamaLama's engine, `get_accel()`, transports, and config. | New package to maintain; depends on RamaLama's public API surface. |
| **B** | **Extend RamaLama (HPC plugin/mode)** | An `apptainer` (and Enroot) engine class + a `scheduler` concept + HPC subcommands **inside** RamaLama. | Maximal reuse, but couples `boxy` to RamaLama's release cadence and review process; HPC concerns spread through a general-purpose tool. |
| **C** | **Thin wrapper over the `ramalama` CLI** | Only orchestration/glue: shell out to `ramalama` and wrap SLURM/Apptainer around it. | Least code, least reuse of internals; duplicates runtime logic RamaLama already has behind its CLI; brittle to CLI changes. |

**Comparison**

| Criterion | A. Standalone lib | B. Extend RamaLama | C. Wrapper |
| --- | --- | --- | --- |
| Reuse of RamaLama internals | High | Highest | Low |
| HPC coverage (Apptainer, Slurm/Flux, multi-node) | Full | Full | Partial |
| Coupling to RamaLama release | Medium | High | Low |
| Implementation effort | Medium | Medium–High | Low |
| Offline / air-gap control | Full | Full | Limited |
| Cohesion of HPC concerns | High | Low | Medium |

**Recommendation: Option A.** It matches the paper's box/location vision, keeps
the HPC concerns cohesive in one place, and still reuses RamaLama's primitives
(transports, engine, `get_accel()`, config) rather than reimplementing them. B
is attractive for maximal reuse but entangles HPC-specific logic with a
general-purpose tool and its release process; C reuses the least and is the most
brittle. (If upstreaming into RamaLama later becomes desirable, A's backend/
scheduler abstractions can be contributed as plugins — A does not preclude B.)

---

## 6b. Gap Analysis — What `boxy` Uniquely Fills

Every individual capability below exists somewhere. **The union does not ship in
any single tool** (see §3b):

1. **Serve — not just run — on Slurm/Flux**, with a stable OpenAI-compatible
   route. SkyPilot serves everywhere *except* Slurm; RamaLama serves but has no
   scheduler.
2. **Apptainer/SIF as a first-class serving runtime.** The orchestrators skip it
   (SkyPilot → Enroot; Run:ai/KServe → Docker/K8s). `boxy` also abstracts the
   Apptainer-vs-root breakage the paper documents.
3. **Offline / air-gapped by design.** NIM struggles here; SkyPilot/dstack/Modal
   are cloud-control-plane oriented. `boxy` reuses RamaLama transports + local
   shared-FS/S3 staging, fully disconnected.
4. **Site portability via a `location` object.** One box retargeted across
   scheduler (Slurm/Flux) × runtime (Apptainer/Podman/Enroot) × accelerator
   (CUDA/ROCm).
5. **Multi-node tensor/pipeline parallelism as a turnkey box.** Today this is DIY
   Ray-on-`srun` (per the paper and vLLM docs); `boxy` encapsulates the pattern.

### Do NOT reinvent — wrap / integrate instead

- **Apptainer/Singularity** — SIF build/run/sign. Wrap the CLI.
- **Pyxis + Enroot** and **Slurm `--container`/`scrun`** — the Slurm container
  path; target them, don't reimplement SPANK/OCI plumbing.
- **RamaLama** — transports, engine abstraction, `get_accel()`, TOML config.
- **vLLM (+ Ray), llama.cpp, MLX** — the inference engines inside boxes.
- **HPCCM / Spack** — optional *build backends* for site-tuned images.
- **Slurm / Flux** — schedulers; submit to them, never reimplement scheduling.
- **A proven offline benchmarker (e.g. GuideLLM)** — for `boxy bench`.

### Differentiation / risks

- **SkyPilot** is the rising overlap. Track its Slurm-serving roadmap; consider
  interop over head-on competition. `boxy` differentiates on serving-first on
  Slurm/Flux *today*, Apptainer/SIF as first-class, true air-gap, and DOE/HPC
  ergonomics (modules, shared-FS staging, ROCm).
- **GWDG SAIA** and **Determined AI** are prior art to learn from (single-site
  web service; training-centric multi-runtime + Slurm, respectively).
- **Run:ai / KServe / llm-d / NIM** are the K8s-gravity / NVIDIA-locked
  alternatives decision-makers will compare against; `boxy`'s answer is
  scheduler-native HPC without requiring K8s, plus ROCm and air-gap.

---

## 7. Requirements

### Functional

- Multiple container runtime backends (§4b), auto-detected and overridable.
- Multiple schedulers: Slurm, Flux, and `none` (local/laptop).
- Multiple accelerators: CUDA and ROCm at minimum (extensible via `get_accel()`).
- Offline / air-gapped operation end-to-end (pull, stage, serve, bench).
- Model staging to shared filesystem and site-local S3.
- Serve an OpenAI-compatible endpoint; run one-shot/interactive inference.
- Benchmark sweeps with plot-ready output.
- Multi-node / tensor-parallel serving as a box option.

### Non-functional

- **Reproducibility / determinism:** honor `--seed`,
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `OMP_NUM_THREADS=1` from the box/location.
- **Portability:** one box definition unchanged across sites.
- **Rootless / unprivileged** operation (Apptainer `--fakeroot`, rootless Podman).
- **No override of user-supplied args** (the prototype's "tack on last" rule).

---

## 8. Phased Roadmap

- **Phase 1 — Core.** `box`/`location` model + `boxy serve`/`run`/`info` on a
  single node; Podman + Apptainer backends; CUDA + ROCm. (Formalizes the
  prototype.)
- **Phase 2 — Models & offline.** `boxy pull`/`stage`/`build`; full air-gap;
  shared-FS + S3 staging; OCI→SIF auto-build.
- **Phase 3 — Scale.** `boxy alloc`; Slurm + Flux scheduler adapters; multi-node
  tensor-parallel serving (Ray-on-Slurm pattern); Enroot/Pyxis + `scrun`
  backends.
- **Phase 4 — Benchmark.** `boxy bench` ShareGPT sweeps + plot export
  (reproduce `plots/`).

---

## 9. Open Questions / Decisions

- **Implementation language.** Python (maximal RamaLama reuse) vs Rust (the
  paper's aspiration). Python is the low-friction path given §3.2.
- **Where the code lands.** This repo (baseline, has the paper + prototype, but
  no Python package skeleton) vs a new package vs contributed into RamaLama.
- **Config format finalization.** Confirm TOML schemas for `box`/`location`.
- **Fast-moving competitive facts to re-confirm before citing as absolute**
  (phrased "as of mid-2026" with sources in this spec, not asserted as
  permanent):
  - (a) SkyPilot's SkyServe still not supporting serving on Slurm.
  - (b) That no shipped tool serves LLMs on Flux.

### Sources

- SkyPilot: `github.com/skypilot-org/skypilot`; `docs.skypilot.co`;
  `blog.skypilot.co/slurm-vs-k8s`
- RamaLama: `github.com/containers/ramalama`
- Pyxis/Enroot: `github.com/NVIDIA/pyxis`; AMD Enroot/Pyxis toolkit docs
- Slurm OCI/scrun: `slurm.schedmd.com/containers.html`, `/scrun.html`
- This repo's paper: arXiv 2509.20603 / `doi.org/10.1145/3731599.3767356`
- GWDG Chat AI / SAIA: arXiv 2407.00110; `github.com/gwdg/saia-hub`, `/saia-hpc`
- Determined AI HPC launcher: `docs.determined.ai/latest/setup-cluster/slurm/`
- HPCCM: `github.com/NVIDIA/hpc-container-maker`
- vLLM multi-node: `docs.vllm.ai` parallelism/scaling
- llm-d / KServe: `llm-d.ai`
- NVIDIA acquires SchedMD (Dec 2025): `blogs.nvidia.com`
