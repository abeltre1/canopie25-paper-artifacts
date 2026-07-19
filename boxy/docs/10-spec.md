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
| **Site heterogeneity** | **CLUSTERA** (NVIDIA H100, SLURM) vs **ClusterB** (AMD MI300a, Flux), incl. MI300a-specific `--gpu-memory-utilization=0.7` to leave HBM for the OS. |
| **Determinism** | `--seed=12345`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `OMP_NUM_THREADS=1`. |

---

## 3. What Already Exists (both repos)

### 3.1 canopie25 prototype (this repo)

The `hpc-workflow/` directory contains a **working bash prototype** of `boxy`
plus the surrounding pipeline. This is the reference implementation `boxy`
formalizes.

| Artifact | Role |
| --- | --- |
| `hpc-workflow/common_boxy.sh` | The "boxy library": platform detection (`clustera`→CUDA, `clusterb`→ROCm), image selection, offline env-var injection, **Podman and Apptainer command builders**, GPU pass-through args, auto-build of the Apptainer `.sif` from an OCI image. |
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

## 3c. Engineering Reuse Map (verified at source level)

What `boxy` concretely *gets* from each project, and where the seams are.

### RamaLama — reuse as a pinned library behind one seam

RamaLama v0.23.0: MIT, on PyPI, runtime deps are only
`argcomplete + pyyaml + jinja2` (good for air-gap). **The seam:** everything up
to `assemble_command(args)` → `VllmPlugin._cmd_serve(args) -> list[str]` (the
*inner* engine argv) is backend-agnostic and reusable; everything after
(`execute_command` → `Engine` → `podman run`, transports/base.py:394/674) is
podman/docker-coupled and is exactly what `boxy` replaces with
Apptainer/srun/flux.

```
        REUSE (backend-agnostic)                 │  REPLACE (podman/docker)
Config → get_accel → New/pull → accel_image →    │
  inner engine argv  ◀ SEAM                      │  execute_command → Engine → podman run
```

| RamaLama asset | Location | How boxy uses it | Friction |
| --- | --- | --- | --- |
| `get_accel()` | common.py:576 | GPU autodetect for `location.accelerator` (probes nvidia-smi/ROCm; mutates `os.environ` — by design) | clean |
| `get_accel_env_vars()` / `set_accel_env_vars()` | common.py:637/591 | accelerator visibility env for pass-through | clean |
| `get_gpu_devices()` | common.py:616 | device nodes (`/dev/kfd`, `/dev/dri`) for binds | clean |
| `Config()` / `DefaultConfig()` | config.py:213/343 | programmatic config; **must set engine/container/image explicitly** (defaults probe the host) | clean |
| `transport_factory.New()` + `ensure_model_exists()` | transport_factory.py:180, base.py:585 | `boxy pull` for `hf://`, `ollama://`, `oci://`, … | needs-shim |
| `GlobalModelStore(base_path)` | global_store.py:22 | local store on scratch/shared FS | clean |
| `VllmPlugin.get_container_image()` | plugins/…/vllm.py:94 | GPU→image selection | clean |
| `Engine`/`BaseEngine` command assembly | engine.py | **not reused** — constructor eagerly emits podman argv (fork-worthy); patterns only | — |

Contract notes: args are duck-typed via `getattr` against the Protocols in
`arg_types.py` (`StoreArgType`: `store/engine/container`; pull also reads
`MODEL, pull, quiet, verify, dryrun`) — a `SimpleNamespace` works, no argparse.
There is **no public API promise** (plugin group is literally
`ramalama.runtimes.v1alpha`), and `_cmd_serve` is impure (resolves models via
the store). Mitigations `boxy` implements: pin `ramalama==0.23.*`; confine every
import to one module (`boxy/ramalama_shim.py`, lazy imports, graceful
degradation when ramalama is absent); build the vLLM serve argv in `boxy`
itself for path-based (shared-FS) models.

### SkyPilot — call for cloud, copy patterns for HPC, never vendor wholesale

SkyPilot is Apache-2.0 but architecturally client–server: every command talks
to an **API server** (auto-started locally), which is the opposite of an
air-gapped, serverless HPC CLI. Its Slurm backend (v0.12+, ~Mar 2026) SSHes to
a login node (SSH-style `~/.slurm/config`), generates **sbatch**, and runs
containers **only via Pyxis/Enroot** (`srun --container-image=…
--container-name=…:create/exec`, readiness via a `.sky_sbatch_ready` sentinel
file). **SkyServe serving is not supported on Slurm.**

| boxy need | SkyPilot asset | Mode |
| --- | --- | --- |
| Cloud provisioning + cloud serving | `sky.launch()` / `sky serve up` (`sky/client/sdk.py`) | **CALL** — Phase 5: transpile box+cloud-location → sky task YAML (`image_id`, `run`, `accelerators`, `service.readiness_probe`) and delegate; inherit 20+ clouds and SkyServe for free |
| Slurm sbatch + container job plumbing | `sky/provision/slurm/instance.py` | **COPY PATTERN** (Apache-2.0 permits lifting with attribution) — informs boxy's future Enroot/Pyxis backend and detached `sbatch` serving |
| Serving with a stable route | `sky/serve/` (readiness-probe-gated replica pool, least-load router, replica=one job, re-launch on failure) | **COPY PATTERN** — reimplement against Slurm/Flux jobs for HPC serving (Phase 3+) |
| Whole framework | client–server core, cloud catalogs, optimizer | **DO NOT VENDOR** — drags in the API server and cloud assumptions |

---

## 3d. What Each Tool Gives You — boxy vs SkyPilot vs RamaLama

The three tools are complements, not substitutes. boxy *uses* the other two
where they are strongest and owns only what neither can do.

| Capability | **boxy** | **SkyPilot** | **RamaLama** |
| --- | --- | --- | --- |
| Serve an LLM on Slurm/Flux | ✅ native (srun/`flux run`; sbatch Phase 3) | ❌ Slurm = batch jobs only; SkyServe unsupported there | ❌ no scheduler concept |
| Apptainer/SIF runtime | ✅ first-class + auto OCI→SIF | ❌ (Enroot/Pyxis only on Slurm) | ❌ (Podman/Docker only) |
| Podman / Docker | ✅ builders + `boxy.box` labels | Docker on cloud/K8s | ✅ core strength |
| Enroot / Pyxis | planned (Phase 3, copying SkyPilot's builder) | ✅ its Slurm container path | ❌ |
| Cloud provisioning (20+ clouds, spot, optimizer) | ➡️ delegated: `boxy generate sky` | ✅ core strength | ❌ |
| Managed cloud serving (replicas, LB, autoscale) | ➡️ delegated: `sky serve up` on boxy-generated YAML | ✅ SkyServe (cloud/K8s only) | ❌ |
| Model transports (hf/ollama/oci/modelscope) + store | ✅ **via RamaLama as a library** | ❌ (file_mounts / buckets / in-task downloads) | ✅ core strength |
| GPU autodetect (CUDA/ROCm/Intel/…) | ✅ **via RamaLama `get_accel()`** | cloud-catalog driven | ✅ core strength |
| Default image per engine+accelerator | ✅ via RamaLama's vLLM plugin mapping | `image_id` manual | ✅ core strength |
| Air-gapped / no control plane | ✅ by design (serverless, offline env, local staging) | ❌ API server + catalogs | partial (local once pulled) |
| One definition → many sites | ✅ `box` + `location` | task YAML (cloud-centric resources) | ❌ single-host |
| Multi-node tensor-parallel serving on HPC | Phase 3 (Ray-on-Slurm pattern) | ❌ on Slurm | ❌ |
| Local dev serving UX (`run`/`serve` on a laptop) | ✅ (scheduler=none) | not its job | ✅ core strength |
| Benchmarking workflow | Phase 4 (paper's ShareGPT sweep) | ❌ | ✅ (`ramalama bench`, llama.cpp) |

**Division of labor in one line:** RamaLama supplies the *model layer*
(transports, store, GPU detect, image maps) as a pinned library; SkyPilot
supplies the *cloud layer* behind `boxy generate sky`; boxy owns the *HPC
layer* (Apptainer/Podman × Slurm/Flux × CUDA/ROCm, offline) and the single
`box`/`location` UX over all three.

**Do we still want RamaLama? Yes — as a library, not as the tool.** boxy now
leverages four of its subsystems through one seam file (`ramalama_shim.py`):
accelerator autodetect (`get_accel`), accelerator env/device maps, model
transports + store (`boxy pull`), and the vLLM plugin's accelerator→image
mapping (default images when a box omits `image`). Adopting the *entire* tool
would mean inheriting its podman-hardcoded execution path and CLI/daemon —
the exact single-host coupling boxy exists to escape — so full adoption is
explicitly rejected (§6, option B analysis).

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

A generalization of the `clustera`/`clusterb` `$CLUSTER` switch.

```toml
# locations/clusterb.toml
[location]
name        = "clusterb"
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
boxy serve --box vllm-llama4-scout --location clusterb
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
boxy alloc  --location clustera            # 2 nodes x 4 GPUs via Slurm
boxy pull   --box vllm-llama4-scout    # offline-aware model fetch
boxy stage  --box vllm-llama4-scout --location clustera
boxy build  --box vllm-llama4-scout --location clustera   # OCI->SIF if Apptainer
boxy serve  --box vllm-llama4-scout --location clustera   # OpenAI endpoint on :8000
boxy bench  --box vllm-llama4-scout --location clustera   # ShareGPT sweep
```

---

## 5b. v2 UX — Model-First Automation (implemented)

Field use of the profile-first CLI surfaced a design problem: requiring two
TOML files before anything runs means *no visible automation* — the opposite
of `ramalama run granite3-moe`. v2 inverts the surface: **the model is the
argument; everything else is resolved, printed, and overridable.**

```
boxy serve MODEL [--engine E] [--runtime R] [--scheduler S] [--accelerator A]
                 [--image I] [--port P] [--gpus N] [--nodes N] [--name NAME]
                 [--here] [--foreground] [--save-profile PREFIX] [--dryrun]
                 [-- extra engine args]
boxy pull  MODEL                  # pre-stage on a login node (shared $HOME store)
boxy stop  NAME                   # name printed in the READY banner / boxy list
```

Resolution rules (each choice printed as an `auto:` line):

1. **MODEL is classified by syntax, never filesystem state**: a transport
   scheme (`hf://`, `ollama://`, `oci://`, ...) is remote; anything else is a
   local path. Bare names are never guessed into a registry — a missing path
   errors with `did you mean ollama://X or hf://<org>/X?`.
2. **Engine**: GGUF or `ollama://` → llama.cpp; safetensors/HF-repo → vLLM,
   which requires a GPU (detected, or requested via `--gpus` for a job).
3. **Accelerator**: RamaLama `get_accel()`, normalized at the seam
   (`hip`→`rocm`, `cann`→`ascend` — get_accel speaks GPU-runtime dialect,
   boxy's location/image/backend maps speak platform names).
4. **Runtime**: first *working* runtime of podman > docker > apptainer —
   probed (`podman info`) not just PATH-checked, because HPC nodes routinely
   carry rootless-broken podman binaries.
5. **Image**: engine+accelerator default from RamaLama's plugin maps;
   llama.cpp on non-CUDA GPUs uses `quay.io/ramalama/{rocm,intel-gpu,...}`
   (upstream ghcr server image is CPU-only) with `llama-server` named
   explicitly (on `$PATH` there, not the image ENTRYPOINT).
6. **Port**: engine default (vLLM 8000, llama.cpp 8090) bind-tested and
   advanced past busy ports; explicit `--port` wins and is never scanned.
7. **Scheduler: never auto-wrapped.** Three contexts:
   - *inside an allocation* (`SLURM_JOB_ID`/`FLUX_*`): run direct,
     **foreground** (the job step owns the server lifetime; a daemonized
     container would be reaped by the epilog), endpoint printed as
     `http://<hostname>:PORT/v1` with an `ssh -L` hint;
   - *login node* (scheduler CLI on PATH, no allocation): **refuse** with the
     exact `--scheduler`/allocation alternatives; `--here` overrides;
   - *workstation/laptop*: detach (`-d`), poll `/v1/models` readiness, print
     the `### READY` banner + stop hint; `--foreground` opts out.
8. `--gpus/--nodes` describe a job request → error without `--scheduler`.
   GPU submission from a GPU-less login node requires `--accelerator` (boxy
   refuses to bake the submitting node's wrong autodetection into the job).
9. Detached serves drop `--rm` so crash logs survive; a startup crash is
   detected immediately (container-alive polling), the last log lines are
   dumped, and the container is removed. `boxy stop NAME` = stop + rm.
10. `--save-profile PREFIX` freezes the resolved pair to
    `PREFIX.box.toml`/`PREFIX.location.toml` (with a header warning that
    values were autodetected on that node) — the bridge from v2 automation
    back to reviewable, air-gap-friendly profiles. Profile mode (`--box`,
    now with optional `--location`) remains fully supported.

The v2 contract was adversarially reviewed by a three-perspective design
panel (RamaLama-parity, HPC-operator, minimal-surface); all verdicts
"sound-with-fixes". Fixes folded in: items 1, 3–9 above. Deliberately
declined: foreground-by-default on laptops (the readiness-gated READY banner
is the automation users validated in the field; `--foreground` is one flag
away). Deferred to the roadmap: `boxy run MODEL` as an interactive chat REPL
(the verb stays reserved for it), engine choice by artifact sniffing after
pull (GGUF magic bytes instead of URI text), `--pull=never|missing|always`,
and deferred re-resolution on the compute node for `--scheduler` submissions.

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
any single tool** (see §3b; the SkyPilot facts below are verified from its
source and docs, the RamaLama facts from its v0.23.0 source — see §3c):

1. **Serve — not just run — on Slurm/Flux**, with a stable OpenAI-compatible
   route. SkyPilot serves everywhere *except* Slurm (its docs list SkyServe as
   unsupported there; on Slurm it only runs batch jobs via SSH+sbatch+Pyxis);
   RamaLama serves but has no scheduler concept at all.
2. **Apptainer/SIF as a first-class serving runtime.** The orchestrators skip it
   (SkyPilot → Enroot; Run:ai/KServe → Docker/K8s). `boxy` also abstracts the
   Apptainer-vs-root breakage the paper documents.
3. **Offline / air-gapped by design.** NIM struggles here; SkyPilot/dstack/Modal
   are cloud-control-plane oriented (SkyPilot's client–server model requires an
   API server even locally, and its cloud paths assume reachable catalogs).
   `boxy` is serverless and stdlib-bootstrappable: RamaLama transports + local
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

## 6c. Standalone vs Leverage — the Decision

The question: should `boxy` lean on RamaLama and SkyPilot, or be a fully
standalone tool that absorbs all that functionality into one CLI?

**Decision: `boxy` is a standalone single CLI — one tool, one UX — that
leverages both projects *behind seams*, absorbing neither.**

- **RamaLama → library behind one file.** Pinned dependency, every import
  confined to `boxy/ramalama_shim.py` with lazy loading and graceful
  degradation (no ramalama ⇒ no autodetect/no transport pulls, everything else
  works). If its unstable internals ever break the pin, the worst case is
  vendoring ~300 lines of accelerator probes under MIT attribution.
- **SkyPilot → optional cloud backend behind a subprocess/SDK boundary.**
  Never a hard dependency (its API-server model must not infect the air-gapped
  path). Phase 5 adds a `cloud` location kind that transpiles to a sky task
  YAML and calls `sky launch` / `sky serve up`.
- **Why not fully standalone (absorb everything)?** Neither project's gaps can
  be configured away — RamaLama has no HPC layer, SkyPilot can't serve on
  Slurm or run air-gapped — so `boxy` must own the HPC layer *either way*.
  Rewriting RamaLama's transports/GPU probes or SkyPilot's cloud provisioning
  on top of that buys no capability, only a permanent maintenance bill for
  code that upstream already maintains.
- **Why not merge into either project?** Upstreaming HPC into RamaLama couples
  boxy to a general-purpose tool's release process (§6, option B); building on
  SkyPilot inherits a control-plane architecture that contradicts the offline
  requirement. The single-CLI user experience is preserved regardless: users
  see only `boxy`.

To the user, `boxy` **is** the single CLI that powers all of it; RamaLama and
SkyPilot are implementation details it can swap out.

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

- **Phase 1 — Core. ✅ implemented in this repo (`boxy/`).** `box`/`location`
  TOML model + `boxy serve`/`run`/`info`/`build`/`pull`/`stop`/`list`/
  `generate sky`; Podman + Apptainer + Docker backends; vLLM + llama.cpp
  engines; CUDA + ROCm; srun/`flux run` wrapping; module-load preamble;
  offline env injection; OCI→SIF auto-build; RamaLama-informed default
  images; RamaLama seam (`ramalama_shim.py`) with graceful degradation.
  46 tests: golden-argv vs the prototype's known-good commands plus one
  regression test per gap found in the feature-by-feature audit. Verified
  live end-to-end (see `boxy/DEMO.md`).
- **Phase 2 — Models & offline.** `boxy stage` (shared-FS + site-local S3
  sync); store-pulled models end-to-end on air-gapped sites; SIF caching
  policy.
- **Phase 3 — Scale & serve.** `boxy alloc`; detached `sbatch`/`flux batch`
  serving with readiness sentinel (SkyPilot Slurm pattern); multi-node
  tensor-parallel (Ray-on-Slurm pattern); HPC serve route (readiness-gated
  replica pool + login-node router, SkyServe pattern); Enroot/Pyxis + `scrun`
  backends.
- **Phase 4 — Benchmark.** `boxy bench` ShareGPT sweeps + plot export
  (reproduce `plots/`).
- **Phase 5 — Cloud delegation. ◐ started.** `boxy generate sky` ships
  (box+location → SkyPilot task YAML, validated by SkyPilot 0.12.3's own
  parser; hf:// models map to bare repo ids fetched in-task). Remaining:
  boxy invoking `sky launch`/`sky serve up` directly as an optional extra.
- **Phase 6 — Field hardening & reliability. ◐ current.** After exhaustive
  live testing across the deployment matrix (local desktop, Flux on ClusterB,
  Slurm on CLUSTERA), a durable **known-issues registry** (§8b) plus a `boxy
  doctor` command that audits the environment for those issues *before* a job
  fails on the compute node. Shipped: a data-driven diagnostics engine, CA +
  proxy auto-propagation into jobs/containers, per-cluster state isolation, the
  login-node/transport guardrails, and OOM/image-pull/network classification.
- **Phase 7 — Agentless (zero-install) execution. ◐ started.** Run a workload
  on a cluster that has **no boxy/Python/RamaLama installed** — only a
  scheduler, a container runtime, SSH, and a shared FS (§8c). Shipped: `boxy
  generate slurm|flux` and `boxy serve --agentless` emit a self-contained batch
  script (a verbatim `podman run` + a bash endpoint-write, no boxy on the node).
  Remaining: laptop-side `list/curl/logs` over the shared FS without a cluster
  boxy; optional bash-probe hardware auto-detection.
- **Phase 8 — Turnkey UX (one command, zero SLURM/container knowledge). ✅
  implemented (`claude/boxy-turnkey`).** `boxy serve <model> --scheduler slurm`
  needs no `--gpus/--account/--partition/--time/--accelerator`. **Model cards**
  (`data/cards/models/*.toml` + `cards.py`) map a model → GPUs/engine/args, with
  a size heuristic (`-70B` → 4 GPUs) for unknowns; **system cards**
  (`data/cards/systems/*.toml`, 3 per type: laptop/HPC-slurm/HPC-flux/cloud/
  OpenShift) are deployment profiles selected by `--system`; **site discovery**
  (`site.py`) fills `--account` from `myaccounts`/`$SBATCH_ACCOUNT`/`sacctmgr` and
  partition/time defaults. A GPU-less login node defaults the accelerator (no
  hard error); the Flux single-queue guard fixes Slurm-style comma partitions.
  New **CharlieCloud** RuntimeBackend (experimental) proves the runtime seam.
  `boxy cards` lists the catalog; `09-architecture.md` diagrams where the machinery
  is hidden. Every filled value still prints an `auto:` decision line. RamaLama
  supplies images; SkyPilot backs the cloud cards. **Roadmap status snapshot
  (2026-07):**

  | Track | Status |
  |---|---|
  | Foundation (box/location, config, v0.1.0 packaging) | ✅ done |
  | Runtime drivers: podman/docker/apptainer, registries/mirrors | ✅ done |
  | Runtime drivers: CharlieCloud | ✅ done (experimental) · k8s-as-runtime ○ not started |
  | HPC: Slurm/Flux submit, distributed, --replicas, --ssh, doctor, CA propagation | ✅ done |
  | HPC: Flux live field validation | ◐ in progress (comma-queue guard shipped) |
  | HPC: Enroot/Pyxis backends | ○ not started |
  | AI serving: serve/bench/sweep, cloud (generate sky), zero-install chisel share | ✅ done |
  | **Turnkey (cards + site + CharlieCloud + diagrams)** | ✅ **done (this phase)** |
  | Dev envs: agentic sandboxes/harness, Atlas UI 3 | ○ not started (after turnkey) |
  | Release: CI matrix + Trusted Publishing; v0.2 with turnkey | ✅ CI done · ○ v0.2 tag pending |

---

## 8b. Known Issues & User Awareness (field registry)

The post-testing catalog: what actually breaks in the field, why, and how boxy
mitigates it. The per-symptom fixes live in `06-runbook.md §6`; the runtime
failure catalog is `boxy/src/boxy/diagnostics.py` (each rule below); the
**executable** form is `boxy doctor`, which checks these *before* you serve.

**Mitigation philosophy (the invariants that ARE the mitigation).**
1. **Diagnose, don't guess** — `diagnostics.diagnose()` returns `None` rather
   than a wrong guess; the generic vLLM-wrapper rule fires last so a specific
   signature always wins (e.g. an image-pull 403 is never misread as a bad GGUF).
2. **Print every automatic choice, and make it overridable** — the `auto:` lines
   (accel/engine/image/port/name) plus a flag/env for each.
3. **Auto-propagate the environment into jobs & containers** — the merged CA
   bundle (`deploy._propagate_ca_bundle`) and the corporate proxy
   (`_propagate_proxy` + the submit-time `env …` prefix) travel with the job so
   the compute node inherits them.
4. **Guardrails** — the login-node serve guard; the transport allowlist
   (China/ModelScope blocked by default, env-only opt-in).
5. **Isolate state** — per-cluster jobs dir (`~/.local/share/boxy/jobs/<cluster>/`)
   so a shared `$HOME` never mixes clusters.
6. **Fail with a next step** — every error names the fix (missing `sbatch` →
   "add `--ssh`"; ghcr 403 → pre-pull / `--proxy` / mirror).

**Severity:** ▲ high (blocks a serve) · ◆ medium (degrades / confuses) · ▽ low.
**Status:** ✅ mitigated-in-code · 📖 documented · ○ open (see limitations).

| Class | What the user sees | Root cause | Sev | boxy mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| Network/trust | `ghcr.io 403 …Zs…` on the compute node | Zscaler/proxy **policy** block of the image registry (a 403 — TLS is fine) | ▲ | `image-pull-blocked` diagnosis; `--proxy` carried into the pull; `--registry` mirror; pre-pull on login (shared store) | ✅ |
| Network/trust | every registry `FAIL [Errno 8]` / hangs | DNS/proxy, not TLS — DNS is upstream of TLS; with a proxy the *proxy* host must resolve | ▲ | `net_failure_kind`/`network_remedy` (dns/proxy/conn/tls); `boxy info --net`/`doctor` name the kind + offline escape | ✅ |
| Network/trust | HF `403`/`401` on pull | token invalid (401) vs network-refused (403 both authed+anon) vs token-lacks-scope (403 anon-ok) | ◆ | `boxy info --net` whoami probe correlates the two → the right verdict | ✅ |
| Network/trust | `CERTIFICATE_VERIFY_FAILED`, or hf works but ollama fails | `SSL_CERT_FILE` *replaces* the store; a site-CA-only file breaks non-intercepted hosts; missing path silently ignored | ▲ | boxy merges site CA + certifi (`ensure_trust_bundle`); names the cert issuer it saw; `doctor` flags a missing path | ✅ |
| Network/trust | proxy set but registries bypass it | `export https_proxy="${http_proxy}"` ran before `http_proxy` set → empty (ignored) | ◆ | `boxy info`/`doctor` print the effective proxy map + warn on http-without-https | ✅ |
| Container image | job dies right after start | compute node can't pull the image (air-gapped/proxied) | ▲ | as ghcr-403 above; agentless pre-stage story (§8c) | ✅ |
| Container image | `no space left on device` pulling vLLM (~20 GB) | small podman-machine disk | ◆ | RUNBOOK: `podman system prune`, grow the VM disk, or pull on the cluster | 📖 |
| Container image | `platform linux/amd64 != arm64` / HIP `no kernel image` | image ↔ host/GPU-arch mismatch | ◆ | `rocm-arch-mismatch` diagnosis (gfx check); use GGUF locally on Apple Silicon | ✅ |
| Resource | 2nd local instance kills the 1st (`Exited (137)`) | the podman/docker **VM** OOM-killed one — NOT boxy | ▲ | `host-oom` diagnosis + `boxy list`/`doctor` surface exit-137 + `podman machine set --memory` | ✅ |
| Resource | `CUDA out of memory` at load | model+KV > VRAM | ◆ | `cuda-oom` diagnosis (util/max-len/TP/quant advice) | ✅ |
| Multi-cluster | clustera shows an clusterb job; `boxy logs` returns the wrong cluster's | shared `$HOME` → one flat jobs dir | ▲ | per-cluster jobs dir (`jobs.local_cluster`); `FOREIGN(origin)` labels; foreign-endpoint exclusion in curl | ✅ |
| Multi-cluster | `--ssh` → `invalid choice: 'logs'` | the CLUSTER's boxy is older than the laptop's | ◆ | stale-install hint; **agentless (§8c) removes cluster boxy entirely** | ✅ |
| Scheduler | Flux `-t 30:00` → "invalid standard duration"; `#FLUX` ignored → no GPUs | Flux wants FSD + lowercase `# flux:` | ▲ | `_to_fsd` conversion; correct sentinel casing | ✅ |
| Scheduler | `--scheduler slurm` on clusterb → `flux batch` usage errors | clusterb's `sbatch` is a Flux wrapper | ◆ | `_submission_hint` → "use `--scheduler flux`" | ✅ |
| Scheduler | `[Errno 2] … sbatch` on a laptop | `--scheduler` with no scheduler + no `--ssh` | ◆ | pre-submit guard → "add `--ssh user@login`, or drop `--scheduler`" | ✅ |
| Scheduler | a live job reaped / duplicate submitted | squeue connect-failure misread as DONE | ▲ | scheduler-unreachable ⇒ `UNKNOWN` (never DONE), never resubmit | ✅ |
| Model load | `weights were not initialized` | vLLM ≥0.24 NFS/Lustre prefetch mis-loads shards, or a partial checkpoint | ◆ | eager loader default (`BOXY_NO_VLLM_EAGER=1` off), checkpoint-completeness guard | ✅ |
| Model load | `trust_remote_code=True` / arch unsupported / missing pip pkg | model ships custom code / new arch / extra dep | ◆ | `trust-remote-code`, `unsupported-arch`, `missing-python-package` diagnoses (+ derived-image recipe) | ✅ |
| Lifecycle | rerun replaced my instance; port changed | per-model singleton redeploys on rerun; port auto-advances when busy | ▽ | documented; `--unique` for a 2nd instance; `auto: port:` line | 📖 |
| Security | can't pull `ms://…` | ModelScope (China) blocked by default | — | transport allowlist; env-only opt-in so a repo TOML can't widen it | ✅ |

**Known limitations / still-open (○).**
- Apptainer serving is **foreground-only** in the MVP (`boxy alloc` is a stub).
- Agentless can't do RamaLama-based multi-file/`ollama://`/`oci://` **pulls**
  (needs Python on the cluster) — the model must be pre-staged (§8c).
- `--proxy` only helps if the **compute node can reach the proxy**; a fully
  air-gapped node still needs a pre-pulled image or a site mirror.
- Flux writes its `--output` log under a job-id spelling that differs from the
  F58 id `boxy list` shows — use `boxy logs <name>` (globs), not a hand-built
  path.
- Real-GPU test provenance: CUDA-vLLM and ROCm/Apptainer paths are golden-tested
  but not executed on GPU hardware in-repo (RUNBOOK §0).

**Mitigation roadmap (next hardening).** Bash-probe hardware auto-detection for
agentless; laptop-side `list/curl/logs` over the shared FS (no cluster boxy); a
`boxy doctor --fix` that applies the safe remedies; a curl-based single-file
GGUF fetch so a lone `hf://…gguf` is agentless-stageable.

## 8c. Agentless (zero-install) execution — design

**Can boxy run a workload on a cluster with no boxy installed there? Yes**, with
two boundaries. The cluster needs only a **scheduler** (`sbatch`/`flux`), a
**container runtime** (`podman`/`apptainer`), **SSH**, and a **shared FS** — no
boxy, Python, or RamaLama on the node running the workload.

**Why it already almost works.** boxy's cluster-side machinery is boxy-*agnostic*
by construction — the SSH/ControlMaster tunnel (`remote.py` shells out to system
`ssh`), the batch-script skeleton (`schedulers/base.py:batch_script`), the
container command (`deploy.execute` runs a plain `podman run …` argv), the
endpoint rendezvous (`jobs.write_endpoint_file` is just JSON on the shared FS),
the readiness poll (an HTTP GET + `squeue`), and the router `--emit` (a config
file). Only **two** things force boxy onto the cluster today:
1. **Compute-node hardware re-resolution** (accel→image→port, via
   `ramalama_shim.detect_accel`) — removed by pinning `--accelerator`/`--image`
   laptop-side (an HPC user knows the partition's hardware); the podman argv is
   then fully resolved off-node.
2. **RamaLama for transport-URI pulls** (`ramalama_shim.pull_model`) — removed by
   a **pre-staged shared-FS model path** (`deploy.resolve_model`'s local branch
   bind-mounts, no pull). Agentless refuses a bare `hf://…` with guidance.

**What ships.** `deploy.render_agentless_script()` renders the resolved `podman
run` argv (foreground, `scheduler='none'` overlay so no `srun`) + an atomic bash
endpoint-write (`host=$(hostname)`, pinned port) into the scheduler's directive
skeleton — **no `boxy` token in the script**. Exposed as `boxy generate
slurm|flux -o job.sh` (inspect/submit by hand) and `boxy serve --agentless`
(boxy on the login node still orchestrates submit + poll + tunnel, reusing the
existing path; only the *workload node* is boxy-free). This directly retires the
stale-remote-boxy issue class from §8b.

**Trade-off vs. the default path.** Agentless trades boxy's smart compute-node
resolution for explicit pins (`--accelerator`/`--image`) and a pre-staged model
— so it's opt-in; the default (boxy-on-cluster) keeps the zero-config
autodetection.

---

## 9. Open Questions / Decisions

- ~~**Implementation language.**~~ **Decided: Python** (maximal RamaLama reuse;
  see §6c). Phase 1 is implemented in `boxy/` in this repo.
- ~~**Where the code lands.**~~ **Decided for now: this repo** (`boxy/`
  subdirectory next to the paper and prototype); can graduate to its own repo
  when it outgrows the artifacts.
- **Config format finalization.** TOML schemas for `box`/`location` are
  implemented as in §4; extend as Phase 2+ adds fields (staging S3 auth,
  partitions/QOS, readiness probes).
- **Fast-moving competitive facts** (verified against SkyPilot source/docs as
  of mid-2026; re-confirm periodically):
  - (a) SkyPilot's SkyServe does not support serving on Slurm (docs list it as
    unsupported; Slurm mode is SSH+sbatch+Pyxis batch jobs only).
  - (b) No shipped tool serves LLMs on Flux.

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
