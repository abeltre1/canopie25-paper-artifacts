"""Environment-variable sets for boxes, from the paper's prototype (common_boxy.sh).

Merge order (later wins): base -> offline -> accelerator quirks -> box.env.
The user's box definition always has the last word.
"""

from __future__ import annotations

# Tell inference stacks to operate fully disconnected from the internet.
OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    # HF fast-transfer has issues with web proxies / custom SSL certs on HPC.
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
}

# Reproducibility / resource hygiene defaults from the prototype.
BASE_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
}

# vLLM engine hygiene (prototype ENV_VARS) — only injected for vllm boxes.
VLLM_ENV: dict[str, str] = {
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    # NCCL/RCCL print NOTHING on failure by default — a dead ncclCommInitRank
    # surfaces only as 'unhandled system error' with no cause (field: eldorado
    # MI300A TP=2). WARN is silent on healthy runs and names the failing
    # transport ('Error while creating shared memory segment', peer-access
    # denials, ...) when it isn't. box.env / --env can override.
    "NCCL_DEBUG": "WARN",
}

# vLLM-on-ROCm quirks (prototype: clusterA/MI300a).
ROCM_VLLM_ENV: dict[str, str] = {
    "VLLM_USE_V1": "1",
    "VLLM_USE_TRITON_FLASH_ATTN": "0",
}


def build_env(box_env: dict[str, str], accelerator: str, offline: bool, engine: str = "vllm") -> dict[str, str]:
    env: dict[str, str] = dict(BASE_ENV)
    if engine == "vllm":
        env.update(VLLM_ENV)
        if accelerator == "rocm":
            env.update(ROCM_VLLM_ENV)
    if offline:
        env.update(OFFLINE_ENV)
    env.update(box_env)  # box always wins
    return env
