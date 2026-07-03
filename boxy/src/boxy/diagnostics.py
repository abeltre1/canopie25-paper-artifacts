"""Engine-startup log diagnostics.

When an inference server crashes during startup, boxy dumps the last lines of
the container log. Raw stack traces are hard to act on, so this module scans
that log text for known failure signatures and turns each into a plain-language
diagnosis with the concrete next step (pin a tag, drop a flag, switch engines).

The table is deliberately data-driven: adding a new signature is one entry, no
control-flow changes. Each rule owns a regex over the log text and a builder
that returns the hint block. `diagnose()` returns the FIRST matching hint (rules
are ordered most-specific first) or None when nothing is recognised — an
unrecognised crash must never be masked by a wrong guess.
"""

from __future__ import annotations

import re
from typing import Callable, NamedTuple


class Rule(NamedTuple):
    name: str
    pattern: "re.Pattern[str]"
    build: Callable[["re.Match[str]", str], str]


def _fmt(title: str, body: str) -> str:
    """Consistent hint block: a headline plus indented, wrapped guidance."""
    lines = [f"boxy diagnosis: {title}"]
    for line in body.strip("\n").splitlines():
        lines.append(f"    {line}" if line else "")
    return "\n".join(lines)


def _weights_not_initialized(m: "re.Match[str]", log: str) -> str:
    # Pull the offending weight names out of the set literal vLLM prints, so the
    # message can say WHICH weights and confirm the layernorm-only signature.
    names = re.findall(r"model\.layers\.\d+\.[A-Za-z0-9_.]+", log)
    uniq_suffix = sorted({n.split(".", 3)[-1] for n in names})
    layernorm_only = bool(uniq_suffix) and all(
        s.endswith("layernorm.weight") or s.endswith("norm.weight") for s in uniq_suffix
    )
    detail = ""
    if uniq_suffix:
        shown = ", ".join(uniq_suffix[:4]) + (" ..." if len(uniq_suffix) > 4 else "")
        detail = f"Missing weights are all: {shown}\n"
        if layernorm_only:
            detail += (
                "Only the norm weights are missing (the big proj/mlp tensors loaded),\n"
                "so this is NOT a bad/partial download — it is an architecture <-> engine\n"
                "version mismatch: the checkpoint stores these under a name this vLLM\n"
                "build does not map.\n"
            )
    return _fmt(
        "vLLM could not place some checkpoint weights (model <-> engine version mismatch)",
        detail
        + "\n"
        "Fix, in order:\n"
        "  1. Stop using the 'latest' vLLM image; pin the tag the model card\n"
        "     recommends, e.g.  boxy serve <model> --engine vllm \\\n"
        "                          --image vllm/vllm-openai:vX.Y.Z\n"
        "  2. Check the model's config.json 'architectures' is supported by that\n"
        "     vLLM build (vllm --help / supported-models list). A brand-new arch\n"
        "     needs a newer image; a custom one may need --trust-remote-code or\n"
        "     (newer vLLM) --model-impl transformers.\n"
        "  3. If the model is a hybrid / not yet supported on vLLM, serve it on\n"
        "     llama.cpp instead:  boxy serve <gguf-model>   (boxy's default engine).",
    )


def _cuda_oom(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "GPU ran out of memory while loading the model",
        "The model + KV cache did not fit in VRAM. Try, in order:\n"
        "  1. Lower KV-cache headroom:  --gpu-memory-utilization 0.80\n"
        "  2. Cap context length:       --max-model-len <smaller>\n"
        "  3. Shard across GPUs:        --tensor-parallel-size <n-gpus>\n"
        "  4. Use a smaller model or a quantized (GGUF on llama.cpp) build.",
    )


def _unsupported_arch(m: "re.Match[str]", log: str) -> str:
    arch = m.group("arch") if "arch" in m.re.groupindex else ""
    who = f" ({arch})" if arch else ""
    return _fmt(
        f"This model architecture is not supported by the running vLLM build{who}",
        "  1. Pin a newer vLLM image tag that lists this architecture as supported.\n"
        "  2. Or serve it on llama.cpp with a GGUF build:  boxy serve <gguf-model>.\n"
        "  3. Custom architectures may need --trust-remote-code.",
    )


def _rocm_arch_mismatch(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "ROCm/HIP GPU error — likely a GPU-architecture <-> image mismatch",
        "The vLLM ROCm image was not built for this node's GPU arch (gfx...), or the\n"
        "host ROCm/driver version does not match the image.\n"
        "  1. Confirm the node's arch:  rocminfo | grep gfx   (e.g. gfx90a=MI200,\n"
        "     gfx942=MI300). The image must support that gfx target.\n"
        "  2. Pin an image tag built for your ROCm version instead of :latest:\n"
        "     boxy serve <model> --engine vllm --image docker.io/vllm/vllm-openai-rocm:<tag>\n"
        "  3. Check the container sees the GPUs: the podman run must pass\n"
        "     --device /dev/kfd --device /dev/dri (boxy's rocm backend does this;\n"
        "     verify with: boxy serve ... --dryrun).\n"
        "  4. As a fallback, serve a GGUF build on llama.cpp:  boxy serve <gguf-model>.",
    )


def _engine_core_generic(m: "re.Match[str]", log: str) -> str:
    # Last-resort: the outer vLLM wrapper with no signature we recognise. Point
    # the user at where the REAL exception is so they stop pasting the wrapper.
    return _fmt(
        "vLLM engine core failed to start — the actionable error is higher up",
        "'Engine core initialization failed. See root cause above' is only the\n"
        "outer wrapper. The real exception is printed ABOVE this line in the\n"
        "container log — usually a Python Traceback ending in ValueError/RuntimeError,\n"
        "an out-of-memory line, or a HIP/CUDA error.\n"
        "  See it all:  <podman|docker> logs <container>   (or the job --output log)\n"
        "  Then match it to: model<->vLLM version, GPU OOM, or GPU-arch/image mismatch.",
    )


def _gguf_load_fail(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "llama.cpp could not load the GGUF model file",
        "The file is missing, truncated, or an unsupported GGUF version/quant.\n"
        "  1. Re-pull the model (a partial download is the usual cause).\n"
        "  2. Confirm the path points at a .gguf file, not a directory.\n"
        "  3. If the quant is very new, pin a newer llama.cpp image tag.",
    )


# Ordered most-specific first. diagnose() returns the first match.
RULES: list[Rule] = [
    Rule(
        "vllm-weights-not-initialized",
        re.compile(r"weights?\s+were\s+not\s+initialized\s+from\s+checkpoint", re.IGNORECASE),
        _weights_not_initialized,
    ),
    Rule(
        "vllm-unsupported-arch",
        re.compile(
            r"(?:Model architecture[s]?\s+(?P<arch>[\w,]+)?\s*(?:is|are)\s+not\s+supported"
            r"|are not supported for now"
            r"|not\s+supported\s+by\s+(?:the\s+)?vLLM)",
            re.IGNORECASE,
        ),
        _unsupported_arch,
    ),
    Rule(
        "cuda-oom",
        re.compile(r"(?:CUDA out of memory|torch\.(?:cuda\.)?OutOfMemoryError|HIP out of memory|"
                   r"No available memory for the cache blocks)", re.IGNORECASE),
        _cuda_oom,
    ),
    Rule(
        "rocm-arch-mismatch",
        re.compile(r"(?:HIP error|hipError|no kernel image is available for execution|"
                   r"invalid device function|device kernel image is invalid|"
                   r"gfx\d{3,}\w*\s+(?:is\s+)?not\s+supported|rocm.*not\s+compatible)",
                   re.IGNORECASE),
        _rocm_arch_mismatch,
    ),
    Rule(
        "gguf-load-fail",
        re.compile(r"(?:failed to load model|unable to load model|llama_load_model_from_file|"
                   r"invalid magic|GGUF|unknown model architecture)", re.IGNORECASE),
        _gguf_load_fail,
    ),
    # LAST: the generic vLLM wrapper. Only fires when nothing specific matched,
    # so a real signature above always wins.
    Rule(
        "vllm-engine-core-generic",
        re.compile(r"Engine core initialization failed|EngineCore failed to start|"
                   r"Failed core proc", re.IGNORECASE),
        _engine_core_generic,
    ),
]


def diagnose(log_text: str) -> str | None:
    """Return a plain-language hint for the first recognised failure signature in
    `log_text`, or None if nothing matches."""
    if not log_text:
        return None
    for rule in RULES:
        m = rule.pattern.search(log_text)
        if m:
            return rule.build(m, log_text)
    return None
