"""`boxy generate card <hf-model-id>` — write a boxy model card from HuggingFace.

Looks the model up on the HF Hub (config.json + the safetensors index), derives
the engine, GPU/VRAM sizing and a capped context, and renders a TOML card in the
exact format `cards.py` consumes (validated by round-tripping through
`cards._parse_card`). Network is the HF Hub API only; auth via `$HF_TOKEN`.
Sizing is a built-in estimate — weight bytes (from the safetensors index, which
already reflects any quantization) + KV/overhead headroom vs an 80GB-class GPU,
with GPUs made tensor-parallel-friendly.
"""

from __future__ import annotations

import json
import math
import os
import ssl
import urllib.error
import urllib.request

from boxy import config

HF_BASE = "https://huggingface.co"


class CardGenError(Exception):
    """A user-actionable failure (bad id, gated repo, unreachable Hub, …)."""


# Architectures vLLM is known to serve. Not exhaustive — an unknown arch WARNs but
# still writes a vllm card (extend freely; --engine always overrides).
_VLLM_KNOWN_ARCHES = {
    "LlamaForCausalLM", "Llama4ForConditionalGeneration", "MllamaForConditionalGeneration",
    "MistralForCausalLM", "Mistral3ForConditionalGeneration", "MixtralForCausalLM",
    "Qwen2ForCausalLM", "Qwen2MoeForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
    "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForConditionalGeneration",
    "Phi3ForCausalLM", "PhiMoEForCausalLM", "GptOssForCausalLM", "FalconForCausalLM",
    "GPTBigCodeForCausalLM", "DbrxForCausalLM", "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM", "CohereForCausalLM", "Cohere2ForCausalLM",
    "InternLM2ForCausalLM", "StableLmForCausalLM", "Starcoder2ForCausalLM",
    "NemotronForCausalLM", "OlmoForCausalLM", "Olmo2ForCausalLM",
}

_CONTEXT_CAP = 8192  # the turnkey KV-cache safety cap (matches the shipped cards)


# ---- HF Hub fetch (stdlib only) -----------------------------------------------------


def resolve_token(explicit: str | None = None) -> str:
    """The HF token: --hf-token, then $HF_TOKEN/$HUGGING_FACE_HUB_TOKEN, then the
    HF cache token file. '' if none (fine for public models)."""
    if explicit:
        return explicit.strip()
    for env in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(env)
        if v and v.strip():
            return v.strip()
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    for p in (os.environ.get("HF_TOKEN_PATH"), os.path.join(home, "token"),
              os.path.expanduser("~/.huggingface/token")):
        try:
            if p and os.path.exists(p):
                t = open(p, encoding="utf-8").read().strip()
                if t:
                    return t
        except OSError:
            pass
    return ""


def _opener() -> urllib.request.OpenerDirector:
    """An opener that trusts the site/proxy CA bundle and honors the env proxy."""
    ctx = ssl.create_default_context()
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    if not ca:
        try:
            import certifi
            ca = certifi.where()
        except Exception:
            ca = ""
    for extra in (ca, "/root/.ccr/ca-bundle.crt"):
        try:
            if extra and os.path.exists(extra):
                ctx.load_verify_locations(extra)
        except (OSError, ssl.SSLError):
            pass
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def hf_get_json(repo: str, path: str, token: str = "", *, required: bool = True,
                opener: urllib.request.OpenerDirector | None = None, timeout: float = 30):
    """GET https://huggingface.co/<repo>/resolve/main/<path> as JSON. On a 404 for
    a NON-required file, returns None (best-effort extras). Raises CardGenError with
    an actionable message for a bad id / gated repo / unreachable Hub."""
    url = f"{HF_BASE}/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "boxy-cardgen"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    op = opener or _opener()
    try:
        with op.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            if not required:
                return None
            raise CardGenError(
                f"{repo!r} was not found on HuggingFace (HTTP 404). Check the id for a typo; "
                f"if it's private, make sure your token can see it.") from None
        if e.code in (401, 403):
            if token:
                raise CardGenError(
                    f"{repo!r}: your HF token was rejected (HTTP {e.code}). It's likely a GATED "
                    f"repo — accept the license at {HF_BASE}/{repo} with the account that owns "
                    f"this token.") from None
            raise CardGenError(
                f"{repo!r}: access denied (HTTP {e.code}) — this model looks GATED. Set HF_TOKEN "
                f"to a token that has accepted the license at {HF_BASE}/{repo} (or pass "
                f"--hf-token). If your network blocks huggingface.co, that also shows as 403.") from None
        raise CardGenError(f"{repo!r}: HuggingFace returned HTTP {e.code} for {path}.") from None
    except urllib.error.URLError as e:
        raise CardGenError(
            f"{repo!r}: cannot reach HuggingFace ({e.reason}). Check your network/proxy "
            f"(HTTPS_PROXY) and the CA bundle.") from None
    except (ValueError, TimeoutError) as e:
        raise CardGenError(f"{repo!r}: bad/again response fetching {path}: {e}") from None


def fetch_model(repo: str, token: str = "", *, opener=None) -> dict:
    """Fetch the config + best-effort safetensors index for a model. Returns
    {'config':…, 'index':… or None, 'generation':… or None}."""
    cfg = hf_get_json(repo, "config.json", token, required=True, opener=opener)
    if not isinstance(cfg, dict):
        raise CardGenError(f"{repo!r}: config.json was not a JSON object.")
    idx = hf_get_json(repo, "model.safetensors.index.json", token, required=False, opener=opener)
    gen = hf_get_json(repo, "generation_config.json", token, required=False, opener=opener)
    return {"config": cfg, "index": idx if isinstance(idx, dict) else None,
            "generation": gen if isinstance(gen, dict) else None}


# ---- engine detection ---------------------------------------------------------------


def _is_gguf_repo(repo: str) -> bool:
    low = repo.lower()
    return low.endswith("-gguf") or "gguf" in low.rsplit("/", 1)[-1]


def pick_engine(repo: str, cfg: dict, explicit: str = "") -> tuple[str, str, str]:
    """(engine, reason, warn). GGUF repo -> llama.cpp; else vllm, with a warn when
    the architecture isn't in the known-vLLM set. --engine (explicit) overrides."""
    if explicit:
        return explicit, "via --engine", ""
    if _is_gguf_repo(repo):
        return "llama.cpp", "GGUF repository", ""
    arches = cfg.get("architectures") or []
    arch = arches[0] if arches else ""
    warn = ""
    if arch and arch not in _VLLM_KNOWN_ARCHES:
        warn = (f"architecture {arch!r} isn't in boxy's known-vLLM list — defaulting to vllm; "
                f"verify vLLM supports it, or pass --engine.")
    return "vllm", (f"architecture {arch}" if arch else "safetensors/HF repo"), warn


# ---- sizing -------------------------------------------------------------------------


def _bytes_per_param(cfg: dict) -> tuple[float, str]:
    """(bytes/param, label) from quantization_config / torch_dtype."""
    q = cfg.get("quantization_config") or {}
    method = str(q.get("quant_method", "")).lower()
    fmt = str(q.get("fmt", q.get("format", ""))).lower()
    bits = q.get("bits", q.get("w_bit"))
    if "mxfp4" in method or "mxfp4" in fmt:
        return 0.5, "MXFP4 (4-bit)"
    if "fp8" in method or "fp8" in fmt:
        return 1.0, "FP8"
    if method in ("awq", "gptq", "gptq_marlin", "compressed-tensors") or bits in (4, "4"):
        return 0.5, f"{(method or '4-bit').upper()} (4-bit)"
    if bits in (8, "8"):
        return 1.0, "8-bit"
    dtype = str(cfg.get("torch_dtype", "")).lower()
    if dtype in ("float32", "float"):
        return 4.0, "fp32"
    return 2.0, (dtype.replace("bfloat16", "bf16").replace("float16", "fp16") or "bf16")


def _moe_info(cfg: dict) -> tuple[bool, int]:
    """(is_moe, num_experts)."""
    for key in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
        n = cfg.get(key)
        # some configs nest this under text_config (multimodal wrappers)
        if n is None and isinstance(cfg.get("text_config"), dict):
            n = cfg["text_config"].get(key)
        if isinstance(n, int) and n > 1:
            return True, n
    return False, 0


def _params_from_name(repo: str) -> float:
    """Billions of params parsed from the id ('-8B', '8x7B'), else 0."""
    from boxy import cards

    m = cards._SIZE_RE.search(repo.rsplit("/", 1)[-1])
    if not m:
        return 0.0
    experts, size = m.groups()
    return float(size) * (int(experts) if experts else 1)


def size_model(repo: str, cfg: dict, index: dict | None, *, max_model_len: int | None = None,
               gpu_class_gb: int | None = None) -> dict:
    """Derive weight VRAM, gpus, min_vram_gb and the capped context. Prefers the
    safetensors index total bytes (already reflects quantization); falls back to a
    name-based param estimate × bytes/param when no index is available."""
    bpp, dtype_label = _bytes_per_param(cfg)
    gpu_gb = gpu_class_gb or config.get_int("cardgen.gpu_class_gb") or 80

    total_bytes = 0
    if index:
        meta = index.get("metadata") or {}
        total_bytes = int(meta.get("total_size") or 0)
    if total_bytes > 0:
        weight_gb = total_bytes / 1e9
        # TRUE param count implied by the on-disk bytes and the stored dtype — for
        # MoE this is the TOTAL (all experts), which the name understates.
        billions = total_bytes / bpp / 1e9
        param_src = "safetensors index"
    else:
        billions = _params_from_name(repo)
        weight_gb = billions * bpp
        param_src = "model name" if billions else "unknown"

    is_moe, experts = _moe_info(cfg)

    # GPUs so the weights fit an 80GB-class device at ~85% util, tensor-parallel
    # friendly (next power of two).
    usable = gpu_gb * 0.85
    raw = max(1, math.ceil(weight_gb / usable)) if weight_gb > 0 else 1
    gpus = 1 << (raw - 1).bit_length() if raw > 1 else 1

    # advisory VRAM: weights + KV/activation headroom (diminishing margin).
    overhead = max(4.0, min(0.4 * weight_gb, 10.0))
    min_vram_gb = int(math.ceil((weight_gb + overhead) / 2.0) * 2) if weight_gb > 0 else 0

    native_ctx = _native_context(cfg)
    cap = max_model_len if max_model_len else (
        min(native_ctx, _CONTEXT_CAP) if native_ctx else _CONTEXT_CAP)

    return {
        "weight_gb": round(weight_gb, 1), "gpus": gpus, "min_vram_gb": min_vram_gb,
        "billions": round(billions, 1), "param_src": param_src, "dtype_label": dtype_label,
        "bpp": bpp, "is_moe": is_moe, "experts": experts,
        "native_ctx": native_ctx, "max_model_len": cap, "capped": bool(native_ctx and native_ctx > cap),
        "gpu_class_gb": gpu_gb,
    }


def _native_context(cfg: dict) -> int:
    for key in ("max_position_embeddings", "max_sequence_length", "max_seq_len", "n_positions"):
        v = cfg.get(key)
        if v is None and isinstance(cfg.get("text_config"), dict):
            v = cfg["text_config"].get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 0


# ---- render + validate --------------------------------------------------------------


def slug(repo: str) -> str:
    """Card filename stem: lowercased, '/'→'-' (matches the packaged cards)."""
    return repo.strip().lower().replace("/", "-")


def _human_b(billions: float) -> str:
    if billions <= 0:
        return "size unknown"
    return f"{billions:g}B params"


def render_card(repo: str, engine: str, sizing: dict) -> str:
    """The card TOML, hand-formatted to match the packaged style (header comment +
    [model] + [model.args] with inline notes)."""
    b = sizing
    bits: list[str] = [_human_b(b["billions"])]
    if b["is_moe"]:
        bits.append(f"MoE ({b['experts']} experts)")
    if b["dtype_label"] not in ("bf16", "fp16"):
        bits.append(b["dtype_label"])
    weight_note = f"~{b['weight_gb']:g}GB weights" if b["weight_gb"] else "weight size unknown"
    tp = "; tensor-parallel from the GPU count" if b["gpus"] > 1 else ""
    gpu_word = "GPU" if b["gpus"] == 1 else "GPUs"
    header = (f"# {repo} — {', '.join(bits)} ({weight_note}): "
              f"{b['gpus']} {b['gpu_class_gb']}GB-class {gpu_word}{tp}.")

    lines = [header, "[model]", f'match = "{repo}*"', f'engine = "{engine}"',
             f"gpus = {b['gpus']}"]
    if b["min_vram_gb"]:
        lines.append(f"min_vram_gb = {b['min_vram_gb']}")
    lines.append("[model.args]")
    if b["capped"]:
        lines.append(f"# {repo.rsplit('/', 1)[-1]} has a {b['native_ctx']}-token native context; vLLM")
        lines.append("# profiles KV cache for the FULL window at startup and can OOM on a small GPU.")
        lines.append("# Cap it so the turnkey serve fits; raise with `-- --max-model-len N`.")
    elif b["is_moe"]:
        lines.append("# cap KV-cache profiling so the MoE fits its GPUs (override for more).")
    else:
        lines.append("# capped context for a turnkey fit; raise with `-- --max-model-len N`.")
    lines.append(f"max_model_len = {b['max_model_len']}")
    return "\n".join(lines) + "\n"


def validate(text: str, repo: str) -> None:
    """Round-trip the rendered card through boxy's OWN loader so a card boxy would
    reject is never written."""
    from boxy import cards

    try:
        cards._parse_card(text, slug(repo), "user", f"<generated {repo}>")
    except ValueError as e:
        raise CardGenError(f"internal: generated card failed boxy's own validation: {e}") from None


def generate(repo: str, *, engine: str = "", max_model_len: int | None = None,
             token: str | None = None, opener=None) -> tuple[str, str, list[str]]:
    """The end-to-end pipeline: fetch → detect → size → render → validate.
    Returns (card_text, engine, warnings). Raises CardGenError on any user error."""
    repo = repo.strip().lstrip("/")
    from boxy import cards

    repo = cards.model_key(repo)  # strip a hf:// scheme if the user pasted one
    tok = resolve_token(token)
    data = fetch_model(repo, tok, opener=opener)
    eng, _why, warn = pick_engine(repo, data["config"], engine)
    warnings = [warn] if warn else []
    sizing = size_model(repo, data["config"], data["index"], max_model_len=max_model_len)
    if sizing["param_src"] == "unknown":
        warnings.append("could not determine size from the safetensors index or the name; "
                        "the geometry is a guess — check gpus/min_vram_gb and override if needed.")
    text = render_card(repo, eng, sizing)
    validate(text, repo)
    return text, eng, warnings
