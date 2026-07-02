"""The single seam between boxy and RamaLama.

ALL `ramalama` imports live in this module, and every import is lazy, so:
  * boxy works (dry-run, explicit locations) with no ramalama installed;
  * a RamaLama internals change breaks exactly one boxy file;
  * air-gapped bootstrap needs nothing beyond the stdlib.

RamaLama internals carry no API promise (plugins are `v1alpha`), so boxy pins
`ramalama==0.23.*` and treats the Protocols in ramalama/arg_types.py as the
input contract: args objects are duck-typed SimpleNamespaces, not argparse.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

# ramalama configures an INFO-level logger on import (proxy discovery etc.);
# keep boxy's output clean unless the user opts into verbosity.
logging.getLogger("ramalama").setLevel(os.environ.get("BOXY_RAMALAMA_LOGLEVEL", "WARNING"))

# ramalama's config prompts interactively on macOS when the podman machine has
# no GPU (applehv). boxy must never block on a prompt — a pull doesn't need a
# GPU. Users can still opt back in by exporting the variable themselves.
# (Field finding: Mac run-through, 2026-07.)
os.environ.setdefault("RAMALAMA_USER__NO_MISSING_GPU_PROMPT", "true")

DEFAULT_STORE = os.path.expanduser(os.environ.get("BOXY_STORE", "~/.local/share/boxy/store"))


def ramalama_available() -> bool:
    try:
        import ramalama.common  # noqa: F401

        return True
    except Exception:
        return False


def detect_accel() -> str:
    """GPU autodetect via ramalama.common.get_accel() (probes nvidia-smi, ROCm, ...).

    NOTE: get_accel() mutates os.environ (sets CUDA_VISIBLE_DEVICES etc.);
    that is desirable here — the backends read those vars for pass-through.
    Returns "none" when ramalama is unavailable or no accelerator is found.
    """
    try:
        from ramalama.common import get_accel
    except Exception:
        return "none"
    accel = str(get_accel())
    # ramalama returns e.g. "cuda", "rocm", "intel", ... or "none"
    return accel.split(":")[0] if accel else "none"


def accel_env_vars() -> dict[str, str]:
    """Accelerator visibility env vars (CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, ...)."""
    try:
        from ramalama.common import get_accel_env_vars, set_accel_env_vars

        set_accel_env_vars()
        return {k: str(v) for k, v in get_accel_env_vars().items()}
    except Exception:
        return {}


def gpu_device_paths() -> dict[str, str]:
    """Host GPU device nodes ({"dri": "/dev/dri", "kfd": "/dev/kfd", ...})."""
    try:
        from ramalama.common import get_gpu_devices

        return dict(get_gpu_devices())
    except Exception:
        return {}


def _store_args(model: str, dryrun: bool = False, quiet: bool = False) -> SimpleNamespace:
    """Duck-typed args satisfying ramalama's StoreArgType + pull needs.

    Required by transport_factory.New / Transport.ensure_model_exists:
    store, engine, container, MODEL, pull, quiet, verify, dryrun.
    engine=None/container=False: boxy owns the launch; ramalama only pulls.
    """
    return SimpleNamespace(
        store=DEFAULT_STORE,
        engine=None,
        container=False,
        MODEL=model,
        pull="missing",
        quiet=quiet,
        verify=True,
        dryrun=dryrun,
    )


def pull_model(model_uri: str, dryrun: bool = False, quiet: bool = False) -> str:
    """Pull a model via RamaLama transports (hf://, ollama://, oci://, ...).

    Returns the resolved host path of the model inside boxy's store.
    Raises RuntimeError with guidance if ramalama is not installed.
    """
    try:
        from ramalama.transports.transport_factory import New
    except Exception as e:
        raise RuntimeError(
            "pulling transport URIs requires the 'ramalama' package "
            "(pip install 'boxy-hpc[ramalama]'); for air-gapped sites set "
            "box.model to a path on the shared filesystem instead"
        ) from e
    args = _store_args(model_uri, dryrun=dryrun, quiet=quiet)
    transport = New(model_uri, args)
    try:
        transport.ensure_model_exists(args)
    except Exception as e:
        raise RuntimeError(_pull_failure_message(model_uri, e)) from e
    # use_container=False, should_generate=False => host blob/snapshot path.
    return transport._get_entry_model_path(False, False, dryrun)


def _pull_failure_message(model_uri: str, error: Exception) -> str:
    """Actionable message with the ROOT cause. RamaLama's repo-pull fallback
    masks the original URL error behind 'cli download not available'
    (NotImplementedError in its v0.23 HF transport), so surface the chain.
    (Field findings: Mac run-through, 2026-07.)"""
    chain: list[str] = []
    seen = 0
    cursor: BaseException | None = error
    while cursor is not None and seen < 6:
        chain.append(str(cursor))
        cursor = cursor.__cause__ or cursor.__context__
        seen += 1
    combined = " | ".join(chain)
    msg = f"failed to pull {model_uri}: {chain[0]}"
    if len(chain) > 1:
        msg += f"\n  root cause: {chain[-1]}"
    if "CERTIFICATE_VERIFY_FAILED" in combined:
        msg += (
            "\n  remedy: your Python has no usable CA bundle (common with uv/standalone builds"
            " and TLS-intercepting proxies). Run:\n"
            "    pip install certifi && export SSL_CERT_FILE=$(python3 -m certifi)\n"
            "  or point SSL_CERT_FILE at your site's CA bundle."
        )
    if "cli download not available" in combined:
        msg += (
            "\n  note: RamaLama 0.23's HuggingFace full-repo CLI fallback is unimplemented;"
            " the direct download above is the path that must succeed."
        )
    return msg


def vllm_image_for(accelerator: str) -> str:
    """Default vLLM serving image per accelerator (mirrors VllmPlugin.get_container_image)."""
    return {
        "cuda": "vllm/vllm-openai:latest",
        "rocm": "rocm/vllm:latest",
        "intel": "intel/vllm:latest",
    }.get(accelerator, "vllm/vllm-openai:latest")


def _ramalama_vllm_image(accelerator: str) -> str | None:
    """Ask RamaLama's vLLM plugin for its accelerator->image mapping (deeper
    leverage than the static map); returns None when unavailable."""
    try:
        from ramalama.config import DefaultConfig
        from ramalama.plugins.loader import get_runtime

        gpu_type = {"cuda": "CUDA", "rocm": "HIP", "intel": "INTEL"}.get(accelerator)
        if gpu_type is None:
            return None
        return get_runtime("vllm").get_container_image(DefaultConfig(), gpu_type)
    except Exception:
        return None


def default_image(engine: str, accelerator: str) -> str:
    """Default container image when a box omits `image`, per engine+accelerator.

    vLLM defaults come from RamaLama's own plugin mapping when importable
    (falling back to the static map); llama.cpp uses the upstream server image.
    """
    if engine == "llama.cpp":
        if accelerator == "cuda":
            return "ghcr.io/ggml-org/llama.cpp:server-cuda"
        return "ghcr.io/ggml-org/llama.cpp:server"
    return _ramalama_vllm_image(accelerator) or vllm_image_for(accelerator)
