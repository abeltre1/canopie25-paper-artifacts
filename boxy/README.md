# boxy

Unified, site-portable, offline-first CLI for deploying and serving
containerized GenAI/LLM services on HPC. This is the Phase-1 MVP of the
design in [`../SPEC.md`](../SPEC.md) — the Python formalization of the bash
prototype in [`../hpc-workflow/`](../hpc-workflow/) (`common_boxy.sh`,
`boxy-run-vllm.sh`).

Define the app once (a **box**), describe each site once (a **location**),
and `boxy` composes the right command for that site's container runtime
(Podman / Apptainer / Docker), scheduler (Slurm / Flux / none), and
accelerator (CUDA / ROCm) — with air-gapped env, module loads, and site
tuning applied automatically.

## Install

```bash
pip install ./boxy                 # core (stdlib only)
pip install './boxy[ramalama]'     # + RamaLama: GPU autodetect, model pulls
```

## Quickstart (mirrors the paper's pipeline)

```bash
# What does this host have?
boxy info

# Serve Llama-4-Scout on Eldorado (Flux + Apptainer + ROCm MI300a):
boxy serve --box examples/boxes/vllm.toml \
           --location examples/locations/eldorado.toml --dryrun

# Same box on HOPS (Slurm + Podman + CUDA H100) — only the location changes:
boxy serve --box examples/boxes/vllm.toml \
           --location examples/locations/hops.toml --dryrun

# Pre-build the Apptainer SIF from the OCI image (prototype build_apptainer_image):
boxy build --box examples/boxes/vllm.toml --location examples/locations/eldorado.toml

# Prototype-style passthrough (boxy-run-vllm.sh "$@"):
boxy run --box examples/boxes/vllm.toml --location examples/locations/hops.toml \
         -- serve my-model --max-model-len=4096

# Pull a model by transport URI (via RamaLama: hf://, ollama://, oci://):
boxy pull --box examples/boxes/vllm-hf.toml
```

Drop `--dryrun` to execute. The Eldorado dry-run reproduces the prototype's
known-good command:

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
2. `boxy serve ... --dryrun` — eyeball the command against your site docs.
3. Inside an allocation (`salloc` / `flux alloc`), run without `--dryrun`.
4. `curl http://<node>:8000/v1/models` — the OpenAI route is up.

## Design rules (from the prototype and paper)

- **Boxes never name a runtime or accelerator** — locations do.
- **User args always win**: box args and location tuning are tacked on last
  and skipped if you already set them.
- **Offline by default on HPC locations**: `HF_HUB_OFFLINE=1` and friends are
  injected when `offline = true`.
- **RamaLama is a seam, not a hard dependency**: every `ramalama` import
  lives in `src/boxy/ramalama_shim.py`; without it boxy still works with
  explicit locations and path-based models.

## Seen in action

[`DEMO.md`](DEMO.md) records a real end-to-end run: `boxy serve` launching a
live llama.cpp OpenAI endpoint in a container (in a fully air-gapped sandbox)
and answering `/v1/chat/completions`, plus the cloud-path YAML being accepted
by SkyPilot 0.12.3 itself.

## Cloud path (SkyPilot delegation)

For cloud sites, boxy doesn't reimplement provisioning — it transpiles the
same box+location into a SkyPilot task:

```bash
boxy generate sky --box examples/boxes/vllm.toml \
     --location examples/locations/cloud-gpu.toml --serve -o task.yaml
sky launch task.yaml        # batch, or:
sky serve up task.yaml      # managed serving (SkyServe replicas + readiness probe)
```

## Tests

```bash
pytest          # 37 golden-argv tests vs the prototype's known-good commands
```

## Not in the MVP (see SPEC.md §8)

`boxy alloc` (interactive allocation), `boxy stage` (S3/shared-FS sync),
`boxy bench` (ShareGPT sweeps), Enroot/Pyxis + Slurm `scrun` backends,
sbatch/detached serving, and the SkyPilot cloud delegation.
