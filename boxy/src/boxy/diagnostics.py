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
    nfs = bool(re.search(r"Filesystem type for checkpoints:\s*(NFS|Lustre)|Prefetching checkpoint",
                         log, re.IGNORECASE))
    detail = ""
    if uniq_suffix:
        shown = ", ".join(uniq_suffix[:4]) + (" ..." if len(uniq_suffix) > 4 else "")
        detail = f"Missing weights are all: {shown}\n"
    if nfs:
        detail += (
            "The checkpoint is on a NETWORK filesystem (NFS/Lustre) and vLLM >= 0.24\n"
            "auto-enables its 'prefetch' loader there, which has been observed to\n"
            "MISLOAD shards (weights silently skipped). This is the most likely cause.\n"
        )
    return _fmt(
        "vLLM could not place some checkpoint weights — bad load, not (usually) the model",
        detail
        + "\n"
        "Fix, in order (cheapest first):\n"
        "  1. NETWORK-FS load bug (esp. if the log says NFS/Lustre/Prefetching):\n"
        "     force the eager loader —  boxy serve <model> -- --safetensors-load-strategy eager\n"
        "     (boxy now defaults vLLM to eager; disable with BOXY_NO_VLLM_EAGER=1).\n"
        "  2. PARTIAL/CORRUPT checkpoint (a pull interrupted by an earlier TLS/network\n"
        "     error leaves a short shard): re-pull and re-verify —\n"
        "       boxy pull <model> --force      (then rerun serve)\n"
        "  3. Only if 1-2 don't fix it, a genuine model<->vLLM version mismatch: pin the\n"
        "     image tag the model card recommends (--image vllm/vllm-openai:vX.Y.Z), or\n"
        "     serve a GGUF build on llama.cpp:  boxy serve <gguf-model>.",
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


def _unknown_load_strategy(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "Your vLLM predates the --safetensors-load-strategy flag boxy adds by default",
        "boxy defaults vLLM to '--safetensors-load-strategy eager' (correct for the\n"
        "NFS/Lustre stores HPC uses), but vLLM < 0.24 doesn't know that flag.\n"
        "  Disable boxy's default:  BOXY_NO_VLLM_EAGER=1 boxy serve ...\n"
        "  Or pin a vLLM >= 0.24 image:  --image vllm/vllm-openai:v0.24.0 (or newer).",
    )


def _missing_python_package(m: "re.Match[str]", log: str) -> str:
    mm = re.search(r"requires the following packages that were not found[^:]*:\s*([^\n.]+)", log, re.I)
    pkgs = mm.group(1).strip().rstrip(".") if mm else ""
    if not pkgs:
        also = re.findall(r"No module named '([^']+)'", log)
        pkgs = ", ".join(sorted(set(also))) if also else "the package(s) named above"
    pip = " ".join(p.strip() for p in pkgs.replace(",", " ").split() if p.strip()) or pkgs
    return _fmt(
        f"The model needs a Python package the vLLM image doesn't ship: {pkgs}",
        "This model's custom code imports a package not in vllm/vllm-openai (common for\n"
        "VLMs with a custom vision tower — e.g. Nemotron-Parse needs open_clip).\n"
        "Fix: serve a custom image that layers the package onto the vLLM image.\n"
        "  1. Build it once on the login node (which has network):\n"
        f"       printf 'FROM docker.io/vllm/vllm-openai:v0.24.0\\nRUN pip install {pip}\\n' > Dockerfile.boxy\n"
        "       podman build -t localhost/vllm-extra:latest -f Dockerfile.boxy .\n"
        "  2. Serve with it:\n"
        "       boxy serve <model> --image localhost/vllm-extra:latest --trust-remote-code ...\n"
        "  The package is then baked in — no compute-node install needed.\n"
        "  Note: the PyPI name can differ from the import name (e.g. open_clip is the\n"
        "  package open_clip_torch) — if pip can't find it, search for the right name.",
    )


def _cert_verify_failed(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "TLS certificate verification failed INSIDE the container (site CA not trusted)",
        "A download from inside the container (HuggingFace/transformers fetching model\n"
        "code or weights at load) hit your site's TLS-intercepting proxy and the\n"
        "container did not trust the site CA.\n"
        "  boxy now mounts its merged CA bundle into the container automatically when\n"
        "  SSL_CERT_FILE is set on the LOGIN node before you submit. So:\n"
        "   1. export SSL_CERT_FILE=/path/to/your/site-ca.crt   (persist it), then\n"
        "      re-run boxy serve — the compute-node container will trust it too.\n"
        "   2. Make sure BOXY_NO_CA_MERGE is NOT set (it disables the merge+mount).\n"
        "   3. This model also fetches a SEPARATE repo's remote code at load, which\n"
        "      needs network egress from the COMPUTE node. If the node is air-gapped,\n"
        "      pre-download every required repo on the login node and serve offline.",
    )


def _trust_remote_code(m: "re.Match[str]", log: str) -> str:
    return _fmt(
        "This model needs trust_remote_code — it ships custom loader code vLLM must run",
        "The model repo contains custom Python that vLLM won't execute unless you opt in\n"
        "(common for new/custom architectures, e.g. NVIDIA Nemotron-Parse).\n"
        "  Fix:  boxy serve <model> ... --trust-remote-code\n"
        "        (equivalently, forward it yourself:  ... -- --trust-remote-code)\n"
        "  Only enable it for models you trust — it runs code shipped in the repo.\n"
        "  If it still fails after this, the architecture may be too new for this vLLM\n"
        "  build — pin a newer --image vllm/vllm-openai:<tag>.",
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
        "missing-python-package",
        re.compile(r"requires the following packages that were not found|"
                   r"ImportError:.*Run `pip install", re.IGNORECASE),
        _missing_python_package,
    ),
    Rule(
        "tls-cert-verify-failed",
        re.compile(r"CERTIFICATE_VERIFY_FAILED|unable to get local issuer certificate|"
                   r"self.signed certificate in certificate chain", re.IGNORECASE),
        _cert_verify_failed,
    ),
    Rule(
        "vllm-trust-remote-code",
        re.compile(r"trust_remote_code\s*=\s*True|contains custom code which must be executed",
                   re.IGNORECASE),
        _trust_remote_code,
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
        "vllm-unknown-load-strategy",
        re.compile(r"unrecognized arguments:.*safetensors-load-strategy|"
                   r"error:.*--safetensors-load-strategy", re.IGNORECASE),
        _unknown_load_strategy,
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
