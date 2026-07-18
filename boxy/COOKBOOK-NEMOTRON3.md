# Nemotron-3 x boxy cookbook — your fleet edition

The boxy counterpart to NVIDIA's
[Ultra vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/vllm_cookbook.ipynb):
the same validated configurations, mapped onto **hops** (4x H100 80GB/node),
**cronus** (4x H200-class 141GB/node) and **eldorado** (4x MI300A 128GB/node),
each as one boxy command. The packaged model cards carry the cookbook flags —
per accelerator — so the commands below are ZERO-FLAG; every choice prints as
an `auto:` line.

Sources: vLLM day-0 blogs (Nano 2025-12, Super 2026-03, Ultra 2026-06), the
NVIDIA-NeMo usage cookbooks, vllm-project/recipes, AMD ROCm blogs (2026-07).

## Configuration reference (what the cards encode)

| Variant | Weights | NVIDIA-validated on | vLLM floor | Best home in this fleet |
|---|---|---|---|---|
| Nano 30B-A3B **FP8** | ~30 GB | H100-class+ | 0.11.2 | any — 1 GPU (start here on eldorado) |
| Nano 30B-A3B BF16 | ~60 GB | H100-class+ | 0.11.2 | any — 1 GPU |
| Nano 30B-A3B NVFP4 | ~18 GB | Blackwell native; Hopper via Marlin | 0.11.2 | cronus/hops (emulated on eldorado) |
| Super 120B-A12B **FP8** | ~120 GB | H100-class+ | 0.17.1 | cronus 1xGPU-node / eldorado 2 GPUs |
| Super 120B-A12B BF16 | ~240 GB | H100-class+ | 0.17.1 | cronus 1 node / eldorado 1 node |
| Ultra 550B-A55B NVFP4 | ~300 GB | 8x H100 (TP1xDP8) / 4-8x Blackwell | **0.22.0 pinned** | cronus 1 node x4 |
| Ultra 550B-A55B BF16 | ~1.1 TB | 16x H100 / 8x H200 / 8x B200 | **0.22.0 pinned** | cronus or eldorado, 3-node Ray |

Notes baked into the cards:
- **The v0.22.0 pin is load-bearing**: `VLLM_USE_FLASHINFER_MOE_FP4` was
  removed in vLLM 0.24 — the reference command silently degrades on `:latest`.
- **cuda overlay = the Hopper/H200 configuration** (this fleet has no
  Blackwell): float16 mamba cache + stochastic rounding, no FlashInfer FP4.
  On a B200 system, add the Blackwell lines after `--`.
- **rocm overlay** = AITER MoE kernels + triton mamba backend (the only
  ROCm-viable one) + float32 SSM cache. FP8 variants run at full hardware
  speed on MI300A (FNUZ FP8); NVFP4 loads via dequant emulation (unvalidated).
- Reasoning parser `nemotron_v3` (built into vLLM >= 0.17.1) for Super/Ultra;
  Nano's `nano_v3` parser is a plugin .py inside the HF repo — add
  `-- --reasoning-parser-plugin nano_v3_reasoning_parser.py --reasoning-parser nano_v3`
  if you need parsed reasoning from Nano.
- MTP speculative decoding is optional and conflicts with Mamba prefix
  caching; add `-- --speculative-config '{"method":"mtp","num_speculative_tokens":5}'`.

## 1. Start a server (pick your rung)

```bash
# CANARY — validate the family on each system first (1 GPU, ~30GB):
boxy serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --ssh eldorado
boxy serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --ssh cronus

# Super — the 120B midpoint:
boxy serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --ssh eldorado   # 2x MI300A
boxy serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --ssh cronus    # 1 node

# Ultra NVFP4 — cronus is its best home here (1 node x4 H200-class):
boxy serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 --ssh cronus
# on hops (4x80GB nodes) the solver spans 2 nodes as one Ray instance (TP4xPP2);
# the cookbook's own H100 shape is DP8 — replicate that with:
#   boxy serve ...NVFP4 --ssh hops --replicas 8 --gpus-per-replica 1

# Ultra BF16 — a 3-node Ray instance, automatically:
boxy serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 --ssh cronus
```

Watch the `auto:` lines: card+variant chosen, per-accelerator flags, pinned
image, geometry arithmetic, model store on scratch, account/partition/time.
Weights land once on the big scratch FS and are reused by every rerun.

## 2. Generate responses

At `### READY` boxy opens the laptop tunnel itself:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
  "messages": [{"role": "user", "content": "Prove there are infinitely many primes, tersely."}],
  "max_tokens": 512}'
```

Reasoning on/off (family convention — thinking is on by default; disable per
request):

```bash
  ..."chat_template_kwargs": {"enable_thinking": false}...
```

Tool calling works out of the box (`qwen3_coder` parser + auto tool choice are
in the cards): pass the standard OpenAI `tools`/`tool_choice` fields.

## 3. Benchmark

```bash
boxy bench --ssh cronus        # TTFT p50/p99, ITL p99, TPOT, E2E, tokens/s
```

## 4. Operate

```bash
boxy list --ssh cronus         # what's live, where
boxy logs <name> --ssh cronus  # engine log (root-cause extraction on failure)
boxy stop <name> --ssh cronus  # kill switch;  boxy stop --all sweeps
boxy clean --ssh cronus        # sweep finished-job records/scripts/logs
```

## 5. When it breaks (expected first-mover territory on eldorado)

Neither NVIDIA nor AMD validates Nemotron-3 on Instinct hardware yet. The
ladder:
1. The failure output includes `extracted ROOT CAUSE: >>> ...` — read that
   line, not the wrapper traceback.
2. Arch-not-supported => the image's vLLM predates the family: try
   `--image docker.io/rocm/vllm-dev:nightly` (ROCm) or a newer
   `vllm/vllm-openai` tag (CUDA >= the variant's floor above).
3. Mamba/triton kernel errors on ROCm (a known hybrid-model risk): same
   nightly-image escalation; if it persists it is a real ROCm vLLM gap worth
   filing — paste the root-cause line back into this session.
4. OOM: serve the smaller variant (FP8 -> the table above) or add
   `-- --max-model-len 65536`.
