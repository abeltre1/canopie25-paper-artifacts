"""Model cards — per-model deployment knowledge for the turnkey UX.

A card carries the deployment details a novice shouldn't have to know (GPU
count, node count, engine, engine args) keyed by a Hugging Face id pattern, so

    boxy serve meta-llama/Llama-3.3-70B-Instruct --scheduler slurm

requests the right geometry with zero extra flags. Cards are DATA:

    packaged  src/boxy/data/cards/models/*.toml   (ships in the wheel)
    user      ~/.config/boxy/cards/models/*.toml  (wins over packaged)

Card format (TOML):

    [model]
    match = "meta-llama/Llama-3.3-70B-Instruct*"   # exact id or glob
    engine = "vllm"          # optional; image still comes from the RamaLama map
    gpus = 4                 # job geometry; tensor-parallel derives from this
    nodes = 1                # optional
    min_vram_gb = 140        # weight footprint — drives the geometry SOLVER (fit_geometry):
                             # solved against the target's node shape (a system card's
                             # gpus_per_node x gpu_vram_gb), it picks the fewest GPUs that
                             # fit, spilling to N-node Ray when the model exceeds one node
    [model.args]             # engine args, merged tack-on-last (user args win)
    max_model_len = 8192

Unknown models fall back to a SIZE HEURISTIC parsed from the name (`-8B`,
`-70B`, `8x7B`), tiered for 80GB-class GPUs. Resolution order everywhere:
flags > user card > packaged card > heuristic > old defaults — and every value
a card fills prints an `auto:` decision line naming the card, keeping the
existing every-choice-is-printed contract.

Import-light on purpose (stdlib only): the compute-node inner serve consults
cards too (same wheel), so geometry resolves login-side and engine args resolve
node-side with no extra flag plumbing.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# transport scheme prefixes stripped before matching (cards match the bare id)
_SCHEMES = ("hf://", "huggingface://", "ollama://", "ms://", "modelscope://",
            "rlcr://", "oci://", "docker://")

# size -> GPUs tiering, assuming 80GB-class devices (the decision line says so).
# (max_billions, gpus)
_SIZE_TIERS = ((13.0, 1), (34.0, 2), (80.0, 4), (float("inf"), 8))

_SIZE_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])")


@dataclass(frozen=True)
class ModelCard:
    match: str
    card_name: str                 # file stem — provenance for decision lines
    source: str                    # "user" | "packaged" | "heuristic"
    engine: str = ""               # "" -> inferred as today
    gpus: int = 0                  # 0 -> no opinion
    nodes: int = 0                 # 0 -> no opinion
    min_vram_gb: int = 0           # weight footprint; 0 -> geometry solver stays off
    # Model's NATIVE context window in tokens (config.json max_position_embeddings);
    # 0 = unknown. The ceiling for the derived --max-model-len.
    native_ctx: int = 0
    # Whole-model KV-cache bytes per ONE token at bf16: summed over the layers
    # that actually cache per-token KV (GQA/MLA); linear-attention/KDA layers
    # cost 0 per token (constant state, counted into the activation reserve).
    # 0 = unknown -> the runtime context derivation stays off and the card's
    # static max_model_len (if any) stands.
    kv_bytes_per_token: float = 0.0
    args: dict = field(default_factory=dict)
    # extra pip packages the model's custom code imports that the engine image
    # doesn't ship (installed at container start; field: Nemotron-Parse/open_clip)
    pip: list = field(default_factory=list)
    # auxiliary HF repos the model's custom code fetches DYNAMICALLY (e.g. its
    # vision encoder) — `boxy bundle` must pre-cache them or an air-gapped serve
    # dies mid-import (field: Nemotron-Parse pulls nvidia/C-RADIOv2-H)
    aux_repos: list = field(default_factory=list)
    # [model.images]: pin the ENGINE IMAGE per accelerator (keys: cuda, rocm,
    # default). For models NEWER than the default images' vLLM — a brand-new
    # architecture dies with 'Engine core initialization failed' in an old
    # image (field: Nemotron-3-Nano on clusterb). --image always wins.
    images: dict = field(default_factory=dict)
    # [model.env]: environment the ENGINE needs — NVIDIA reference commands set
    # kernel selectors as env vars (VLLM_USE_FLASHINFER_MOE_FP4=1), which flags
    # cannot express. Merged into the container env; user box.env wins.
    env: dict = field(default_factory=dict)
    # Which accelerators this CHECKPOINT runs on at all: a quant format can be
    # hardware-bound (NVFP4 = Blackwell/CUDA-only) and fails deep in kernel
    # init elsewhere — the card refuses UP FRONT instead, and unsupported_hint
    # names the variant to serve there. Empty = runs anywhere.
    accelerators: list = field(default_factory=list)
    unsupported_hint: str = ""

    @property
    def label(self) -> str:
        return f"{self.source} card '{self.card_name}'"


def model_key(model: str) -> str:
    """The bare model id a card matches against: transport scheme stripped,
    nothing else touched ('hf://meta-llama/X' and 'meta-llama/X' hit the same
    card)."""
    m = model.strip()
    low = m.lower()
    for scheme in _SCHEMES:
        if low.startswith(scheme):
            return m[len(scheme):]
    return m


def match_keys(model: str) -> list[str]:
    """Every key a card pattern may match for `model`. A plain id yields just
    itself; a filesystem PATH also yields its trailing `org/name` pair and
    basename, so a shared-FS checkout of meta-llama/X hits the same card as
    hf://meta-llama/X (field: a by-path Maverick serve missed its card's
    geometry + context cap and OOMed)."""
    key = model_key(model)
    keys = [key]
    trimmed = key.rstrip("/")
    if "/" in trimmed and (os.path.isabs(os.path.expanduser(trimmed))
                           or trimmed.startswith((".", "~"))):
        parts = [p for p in trimmed.split("/") if p]
        if len(parts) >= 2:
            keys.append(f"{parts[-2]}/{parts[-1]}")
        if parts:
            keys.append(parts[-1])
    return keys


def _hit(keys: list[str], pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(k, pattern) or k == pattern for k in keys)


def _user_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(xdg) / "boxy" / "cards" / "models"


def _parse_card(text: str, card_name: str, source: str, path: str) -> ModelCard:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None
    section = data.get("model")
    if not isinstance(section, dict) or not section.get("match"):
        raise ValueError(f"{path}: a model card needs a [model] section with a 'match' pattern")
    args = section.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"{path}: [model.args] must be a table of engine flags")
    return ModelCard(
        match=str(section["match"]),
        card_name=card_name,
        source=source,
        engine=str(section.get("engine", "")),
        gpus=int(section.get("gpus", 0)),
        nodes=int(section.get("nodes", 0)),
        min_vram_gb=int(section.get("min_vram_gb", 0)),
        native_ctx=int(section.get("native_ctx", 0)),
        kv_bytes_per_token=float(section.get("kv_bytes_per_token", 0.0)),
        args=dict(args),
        pip=[str(x) for x in section.get("pip", [])],
        aux_repos=[str(x) for x in section.get("aux_repos", [])],
        images={str(k): str(v) for k, v in (section.get("images") or {}).items()},
        env={str(k): (dict(v) if isinstance(v, dict) else str(v))
             for k, v in (section.get("env") or {}).items()},
        accelerators=[str(a) for a in section.get("accelerators", [])],
        unsupported_hint=str(section.get("unsupported_hint", "")),
    )


def load_cards() -> list[ModelCard]:
    """User cards first (they win), then packaged. A malformed USER card raises
    with its path (the user wrote it and must know); a malformed PACKAGED card
    is a boxy bug but must never take down `serve` — it is skipped."""
    cards: list[ModelCard] = []
    user_dir = _user_dir()
    if user_dir.is_dir():
        for p in sorted(user_dir.glob("*.toml")):
            cards.append(_parse_card(p.read_text(), p.stem, "user", str(p)))
    from importlib import resources

    try:
        root = resources.files("boxy").joinpath("data/cards/models")
        for entry in sorted(root.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".toml"):
                try:
                    cards.append(_parse_card(entry.read_text(), entry.name[:-5],
                                             "packaged", entry.name))
                except ValueError:
                    continue
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    return cards


def find_card(model: str) -> ModelCard | None:
    """Best card for `model`: user beats packaged; within a source the LONGEST
    match pattern wins (most specific — 'Qwen2.5-7B-Instruct-GGUF*' beats
    'Qwen2.5-7B-Instruct*')."""
    keys = match_keys(model)
    best: ModelCard | None = None
    for card in load_cards():
        if not _hit(keys, card.match):
            continue
        if best is None:
            best = card
        elif best.source == "packaged" and card.source == "user":
            best = card
        elif card.source == best.source and len(card.match) > len(best.match):
            best = card
    return best


def layered_args(model: str) -> tuple[dict, str]:
    """[model.args] with CONFIG-STYLE LAYERING: the best-matching PACKAGED card
    is the base, the best-matching USER card overlays it key-by-key. A user card
    still wins every key it SETS — but keys it doesn't mention fall through to
    the packaged card instead of being erased.

    Field failure this exists for: a stale `generate card` user card (written
    before cardgen knew about trust_remote_code) shadowed the packaged
    Nemotron-Parse card entirely, silently dropping --trust-remote-code and
    killing every serve at vLLM config validation. Returns (args, provenance)."""
    keys = match_keys(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not _hit(keys, card.match):
            continue
        cur = best.get(card.source)
        if cur is None or len(card.match) > len(cur.match):
            best[card.source] = card
    user, packaged = best.get("user"), best.get("packaged")
    if user is None and packaged is None:
        return {}, ""
    if user is None:
        return dict(packaged.args), packaged.label
    if packaged is None or not packaged.args:
        return dict(user.args), user.label
    merged = {**packaged.args, **user.args}
    inherited = [k for k in packaged.args if k not in user.args]
    label = user.label
    if inherited:
        label += f" + {', '.join(inherited)} inherited from the {packaged.label}"
    return merged, label


def layered_pip(model: str) -> list:
    """Extra pip packages for the model, UNION of the best-matching packaged and
    user cards (same layering rationale as layered_args: a user card must never
    silently drop a packaged card's required runtime deps)."""
    keys = match_keys(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not _hit(keys, card.match):
            continue
        cur = best.get(card.source)
        if cur is None or len(card.match) > len(cur.match):
            best[card.source] = card
    out: list = []
    for c in (best.get("packaged"), best.get("user")):
        for p in (c.pip if c else []):
            if p not in out:
                out.append(p)
    return out


def layered_env(model: str) -> dict:
    """[model.env] with the same packaged-base / user-overlay layering as
    layered_args: engine env vars the model needs (kernel selectors like
    VLLM_USE_FLASHINFER_MOE_FP4). User box.env still wins at merge time."""
    keys = match_keys(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not _hit(keys, card.match):
            continue
        cur = best.get(card.source)
        if cur is None or len(card.match) > len(cur.match):
            best[card.source] = card
    out: dict = {}
    for c in (best.get("packaged"), best.get("user")):
        if c:
            out.update(c.env)
    return out


def layered_aux_repos(model: str) -> list:
    """Auxiliary HF repos (dynamically fetched by the model's custom code) from
    the best packaged + user cards — `boxy bundle` pre-caches every one so an
    air-gapped serve never reaches for the network mid-import."""
    keys = match_keys(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not _hit(keys, card.match):
            continue
        cur = best.get(card.source)
        if cur is None or len(card.match) > len(cur.match):
            best[card.source] = card
    out: list = []
    for c in (best.get("packaged"), best.get("user")):
        for r in (c.aux_repos if c else []):
            if r not in out:
                out.append(r)
    return out


def size_heuristic(model: str) -> ModelCard | None:
    """Geometry guess for a model with no card, from the size token in its name:
    '-8B' -> 8, '8x7B' (MoE) -> 56 effective. Tiered for 80GB-class GPUs. None
    when the name carries no size."""
    key = model_key(model).rsplit("/", 1)[-1]
    m = _SIZE_RE.search(key)
    if not m:
        return None
    experts, size = m.groups()
    billions = float(size) * (int(experts) if experts else 1)
    for cap, gpus in _SIZE_TIERS:
        if billions <= cap:
            return ModelCard(match=key, card_name=f"~{billions:g}B", source="heuristic",
                             gpus=gpus)
    return None


# provenance of the LAST resolve_model_card autogen attempt, for the decision
# lines: "note" = path a generated card was written to, "fail" = why generation
# fell back to the name heuristic. apply_to_args consumes (and clears) these.
_last_autogen = {"note": "", "fail": ""}


def _autogen_model_id(model: str) -> str:
    """The bare HF id `model` names, or '' when autogen must not fire: only a
    plain 'org/name' (bare or hf://-prefixed) can be looked up on the Hub —
    never local paths, GGUF file refs, oci/ollama/modelscope URIs."""
    low = model.strip().lower()
    if low.startswith(("oci://", "docker://", "ollama://", "ms://", "modelscope://", "rlcr://", "s3://")):
        return ""
    key = model_key(model)
    if key.count("/") != 1 or key.startswith(("/", ".", "~")) or os.path.exists(key):
        return ""
    if key.lower().endswith((".gguf", ".safetensors", ".bin")):
        return ""
    return key


def _autogen_enabled() -> bool:
    if os.environ.get("HF_HUB_OFFLINE") == "1":  # air-gapped: never call the Hub
        return False
    from boxy import config

    return config.get_bool("cards.autogen")


def resolve_model_card(model: str) -> ModelCard | None:
    """Card if one matches; else GENERATE one deterministically from the model's
    HuggingFace metadata (written to the user cards dir — fetched once, loaded
    as a plain user card forever after); else the name-size heuristic (loudly
    labeled a guess by apply_to_args). The guess is now the last resort, not
    the default for unknown models."""
    card = find_card(model)
    if card:
        return card
    _last_autogen["note"] = _last_autogen["fail"] = ""
    hf_id = _autogen_model_id(model)
    if hf_id and _autogen_enabled():
        from boxy import cardgen

        generated, msg = cardgen.auto_card(hf_id)
        if generated is not None:
            _last_autogen["note"] = msg
            return generated
        _last_autogen["fail"] = msg
    return size_heuristic(model)


# ---- system cards (per-system-type deployment profiles) ---------------------------


def _user_systems_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(xdg) / "boxy" / "cards" / "systems"


def system_card_entries() -> list[tuple[str, str, object]]:
    """(stem, type, read_text) for every system card — user dir first (wins on a
    name clash), then the packaged catalog grouped by type subdir (laptop,
    hpc-slurm, hpc-flux, cloud, openshift). read_text() returns the TOML text;
    nothing is parsed until called."""
    from importlib import resources

    out: list[tuple[str, str, object]] = []
    ud = _user_systems_dir()
    if ud.is_dir():
        for p in sorted(ud.rglob("*.toml")):
            out.append((p.stem, "user", (lambda p=p: p.read_text())))
    try:
        root = resources.files("boxy").joinpath("data/cards/systems")
        for typ in sorted(root.iterdir(), key=lambda e: e.name):
            if "." in typ.name:
                continue
            for entry in sorted(typ.iterdir(), key=lambda e: e.name):
                if entry.name.endswith(".toml"):
                    out.append((entry.name[:-5], typ.name, (lambda e=entry: e.read_text())))
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        pass
    return out


def _match_system_card(name: str) -> tuple[str, str] | None:
    """(text, stem) of the system card matching `name`. The canonical id is the
    card's [location].name (unique: slurm-cuda, flux-cuda, …); the file stem is a
    convenience fallback but can collide across type dirs (cuda-cluster exists
    under both hpc-slurm and hpc-flux), so an exact location-name match wins."""
    parsed: list[tuple[str, str, str]] = []  # (text, stem, loc_name)
    for stem, _typ, read_text in system_card_entries():
        text = read_text()
        try:
            loc_name = (tomllib.loads(text).get("location") or {}).get("name") or ""
        except tomllib.TOMLDecodeError:
            continue
        parsed.append((text, stem, loc_name))
    for text, stem, loc_name in parsed:      # pass 1: canonical location.name
        if name == loc_name:
            return text, stem
    for text, stem, loc_name in parsed:      # pass 2: file-stem fallback
        if name == stem:
            return text, stem
    return None


def system_card_names() -> list[tuple[str, str]]:
    """(canonical_name, type) for every system card — canonical = [location].name.
    Used by `boxy cards` so the listed handle is the one --system matches first."""
    out: list[tuple[str, str]] = []
    for stem, typ, read_text in system_card_entries():
        try:
            loc_name = (tomllib.loads(read_text()).get("location") or {}).get("name")
        except tomllib.TOMLDecodeError:
            loc_name = None
        out.append((loc_name or stem, typ))
    return out


def system_card_path(name: str) -> str:
    """Materialize the system card `name` to a temp TOML file and return the
    path, so `--system` is pure sugar over `--location` (all the existing profile
    machinery — Location.from_toml, flag overlay, batch directives — is reused
    unchanged). Raises ValueError listing choices when unknown."""
    import tempfile

    hit = _match_system_card(name)
    if hit is None:
        known = sorted({stem for stem, _t, _r in system_card_entries()})
        raise ValueError(f"unknown system card {name!r}. Known: {', '.join(known)} "
                         f"(list: `boxy cards`; or drop a TOML in {_user_systems_dir()})")
    text, stem = hit
    f = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, prefix=f"boxy-system-{stem}-")
    f.write(text)
    f.close()
    return f.name


def apply_to_args(args, shape: tuple[int, int, str] | None = None, unified: bool = False) -> list[str]:
    """Turnkey fill for a SCHEDULER submission: when --gpus/--nodes/--engine are
    absent, take them from the model's card (or the size heuristic), returning
    the decision lines to print. Explicit flags always win; local (no-scheduler)
    serves are untouched — there the detected accelerator already drives GPU
    use, and injecting --gpus would change behavior.

    `shape` = (gpus_per_node, gpu_vram_gb, provenance) — the target SYSTEM's
    node hardware (from a user system card / config). With it and a card that
    declares min_vram_gb, the geometry is SOLVED (fit_geometry) instead of
    copied: fewer GPUs on fat-VRAM parts, and models bigger than one node
    automatically become N-node Ray instances. Power users' --gpus/--nodes
    (and a card's own explicit nodes) always bypass the solver.

    `unified` = the target's memory is ONE CPU+GPU pool (APU parts: MI300A).
    The solver then sizes against the claimable fraction of the pool, and
    --gpu-memory-utilization is DERIVED from the model's weight footprint
    (derive_gpu_memory_utilization) instead of trusting vLLM's 0.9 default —
    which starves the host during the weight load and gets an engine rank
    OOM-killed with no traceback."""
    decisions: list[str] = []
    model = getattr(args, "model", None)
    if not model:
        return decisions
    card = resolve_model_card(model)
    if card is None:
        return decisions
    accel = getattr(args, "accelerator", None) or "cuda"
    check_accelerator(card, getattr(args, "accelerator", None) or "")
    if card.source == "generated":
        wrote = f" -> {_last_autogen['note']}" if _last_autogen["note"] else ""
        decisions.append(
            f"card: generated deterministically from HuggingFace metadata "
            f"(~{card.min_vram_gb}GB weights, engine {card.engine or 'vllm'}){wrote}")
    elif card.source == "heuristic" and _last_autogen["fail"]:
        decisions.append(
            f"card: {_last_autogen['fail']} — geometry below is a NAME GUESS; run "
            f"`boxy generate card {model_key(model)}` from a connected machine for the real numbers")
    # --context turns the window into a DEMAND the geometry must meet (Mode B).
    # Hard preconditions raise ValueError — main() prints them as clean
    # `boxy: error:` lines — because silently under-serving an EXPLICIT context
    # request is the one thing this feature must never do.
    card_args_flat = effective_args(card.args, accel)
    user_args = getattr(args, "args", None)
    raw_ctx = getattr(args, "context", None)
    ctx_demand = 0
    ctx_kv_bytes = 0.0
    if raw_ctx is not None:
        if (getattr(args, "engine", None) or card.engine or "vllm") != "vllm":
            raise ValueError("--context: only the vllm engine has KV-cache geometry to size")
        ctx_demand = parse_context_request(raw_ctx, card.native_ctx)
        if card.native_ctx and ctx_demand > card.native_ctx:
            decisions.append(
                f"context: {ctx_demand} requested > the model's {card.native_ctx}-token "
                f"native window — clamped (the engine refuses past max_position_embeddings)")
            ctx_demand = card.native_ctx
        if not card.kv_bytes_per_token:
            raise ValueError(
                f"--context {raw_ctx}: {card.label} doesn't know the model's KV bytes/token — "
                f"regenerate it with `boxy generate card {model_key(model)}` or set "
                f"kv_bytes_per_token in the card")
        _sw, _sv, _ssrc = shape or (0, 0, "")
        if not _sv or "assumed" in (_ssrc or "").lower():
            raise ValueError(
                f"--context {raw_ctx}: the target's node VRAM is unknown, so the KV fit "
                f"can't be proven — write a system card with [location.resources] "
                f"gpu_vram_gb (or set BOXY_GPU_VRAM_GB)")
        ctx_kv_bytes = card.kv_bytes_per_token * _kv_dtype_factor(card_args_flat, user_args)
    gpus_free = getattr(args, "gpus", None) is None
    nodes_free = getattr(args, "nodes", None) is None
    if gpus_free and nodes_free and card.min_vram_gb and not card.nodes:
        w, v, src = shape or (0, 0, "")
        # On a unified pool the solver sizes against what a rank may CLAIM, not
        # the whole part — the host keeps a reserve to stream the load. That is
        # what spreads a 70B over 4 MI300A ranks (35GB shards that load) instead
        # of packing it onto 2 (70GB shards that get the host OOM-killed).
        nodes, gpus, why = fit_geometry(card.min_vram_gb, w, v, unified=bool(unified and v),
                                        ctx_tokens=ctx_demand,
                                        ctx_kv_bytes_per_token=ctx_kv_bytes)
        if ctx_demand and "does not fit" in why:
            # refuse UP FRONT: a submitted job would burn its queue wait and
            # then OOM the KV profile; the message names the largest that fits
            raise ValueError(f"--context {raw_ctx}: {why}")
        args.gpus = gpus
        src_note = f"; {src}" if src else ""
        decisions.append(f"gpus: {gpus} per node ({card.label}: {why}{src_note})")
        if nodes > 1:
            args.nodes = nodes
            decisions.append(f"nodes: {nodes} ({card.label}: the model exceeds one node -> "
                             f"one Ray instance across {nodes} nodes)")
        if ctx_demand:
            decisions.append(f"context: {ctx_demand} tokens (--context {raw_ctx}) — "
                             f"geometry sized so the window provably fits")
    elif gpus_free and card.gpus:
        args.gpus = card.gpus
        note = f" (~{card.min_vram_gb}GB VRAM)" if card.min_vram_gb else ""
        decisions.append(f"gpus: {card.gpus} per node ({card.label}, sized for 80GB-class GPUs{note})")
    if getattr(args, "nodes", None) is None and card.nodes:
        args.nodes = card.nodes
        decisions.append(f"nodes: {card.nodes} ({card.label})")
    if getattr(args, "engine", None) is None and card.engine:
        args.engine = card.engine
        decisions.append(f"engine: {card.engine} ({card.label})")
    if getattr(args, "image", None) is None and card.images:
        pinned = card.images.get(accel) or card.images.get("default", "")
        if pinned:
            args.image = pinned
            decisions.append(f"image: {pinned} ({card.label} pins a {accel} image — this "
                             f"model needs it; --image overrides)")
    # engine args from the card (e.g. max_model_len so vLLM doesn't profile KV
    # cache for the model's full 128K context and OOM). Card flags go FIRST so
    # the user's own post-`--` engine args, appended after, win (last-wins in the
    # engine's argparse). Field failure: bare 8B serve OOM'd because this table
    # was never applied.
    flags = engine_flags(card_args_flat)
    # Derived max_model_len (the card's static cap is a workaround sized for the
    # smallest machine — field: Kimi-K3's 1M native window hand-capped to 131072
    # on hardware whose arithmetic supports the full window). Fires only when
    # the card carries the KV fields AND the system card declares REAL node
    # VRAM (an assumed shape must never change a deployment); otherwise the
    # static cap stands and the decision line names the missing piece.
    ctx_pair: list[str] = []
    ctx_lines: list[str] = []
    if card.min_vram_gb and (getattr(args, "engine", None) or "vllm") == "vllm":
        _w, vram, _src = shape or (0, 0, "")
        # A shape whose VRAM was ASSUMED from the GPU-type table (a100 -> 80GB,
        # but 40GB variants exist) must not feed the context arithmetic: a 2x
        # overshoot OOMs the KV profile at startup. _facts_shape marks that
        # case in the provenance; treat it as unknown here.
        if "assumed" in (_src or "").lower():
            vram = 0
        if card.kv_bytes_per_token and (card.native_ctx or ctx_demand) and vram:
            world = int(getattr(args, "nodes", None) or 1) * int(getattr(args, "gpus", None) or 1)
            # PP divides the layers (and so the per-rank KV cost): honor an
            # explicit pipeline size (card or post-`--`, last-wins) over the
            # geometric default of PP=nodes — the field K3 serve ran TP8xPP4
            # on 8 nodes where the default assumed PP8.
            pp_val = _flag_value(user_args, "--pipeline-parallel-size") \
                or card_args_flat.get("pipeline_parallel_size")
            try:
                pp = int(pp_val) if pp_val is not None else int(getattr(args, "nodes", None) or 1)
            except (TypeError, ValueError):
                pp = int(getattr(args, "nodes", None) or 1)
            tokens, why = derive_max_model_len(
                card.kv_bytes_per_token, card.native_ctx or ctx_demand, card.min_vram_gb,
                world, pp, vram, unified=bool(unified and vram),
                util=None if unified else _explicit_util(card_args_flat, user_args),
                kv_dtype_factor=_kv_dtype_factor(card_args_flat, user_args))
            if ctx_demand:
                # Mode B: the pair carries EXACTLY the demand — the geometry
                # was sized (or is being verified) for it. tokens < demand is
                # only reachable with user-pinned --gpus/--nodes (the solver
                # path refused up front): warn, emit nothing, let the static
                # cap stand — never silently under-serve an explicit request.
                if tokens is not None and tokens >= ctx_demand:
                    ctx_pair = ["--max-model-len", str(ctx_demand)]
                    ctx_lines.append(
                        f"max-model-len: {ctx_demand} (--context {raw_ctx}: verified — {why})")
                else:
                    largest = f"~{tokens} tokens" if tokens else "no context at all"
                    ctx_lines.append(
                        f"context: NOT honored — {ctx_demand} tokens do not fit the pinned "
                        f"{world} rank(s) at PP={pp} ({why}); largest that fits is {largest}; "
                        f"free --gpus/--nodes to let boxy grow the geometry, or set "
                        f"`-- --max-model-len` yourself")
            elif tokens is not None:
                ctx_pair = ["--max-model-len", str(tokens)]
                full = (" — the FULL native window fits" if tokens >= card.native_ctx
                        else "")
                ctx_lines.append(
                    f"max-model-len: {tokens} (derived: {why}{full}; native "
                    f"{card.native_ctx}; `-- --max-model-len N` overrides)")
            else:
                ctx_lines.append(
                    f"max-model-len: NOT derived — {why}; the card's static cap stands; "
                    f"spread wider (--gpus/--nodes) or set it after `--`")
        elif card.native_ctx and not card.kv_bytes_per_token:
            ctx_lines.append(
                "max-model-len: card cap kept (KV bytes/token unknown — regenerate the "
                "card with `boxy generate card`, or set kv_bytes_per_token in it by "
                "hand, to derive the largest context that fits)")
        elif card.kv_bytes_per_token and not vram:
            ctx_lines.append(
                "max-model-len: card cap kept (node VRAM unknown — a system card with the "
                "real gpu_vram_gb lets boxy derive the largest context that fits)")
    if ctx_pair:
        # the derived value REPLACES the card's static pair (a dead duplicate in
        # the argv reads like a bug); the engine-args line is emitted AFTER the
        # removal so it never shows a flag that is not actually passed.
        flags = _strip_flag_pair(flags, "--max-model-len")
    if flags:
        decisions.append(f"engine args: {' '.join(flags)} ({card.label})")
    decisions.extend(ctx_lines)
    # ctx pair goes BEFORE the derived gpu-mem pair (appended below), keeping
    # the long-standing contract that the gpu-mem pair is LAST among boxy's
    # flags; the user's post-`--` args still land after everything and win.
    flags = flags + ctx_pair
    if unified and card.min_vram_gb and (getattr(args, "engine", None) or "vllm") == "vllm":
        _w, pool, _src = shape or (0, 0, "")
        world = int(getattr(args, "nodes", None) or 1) * int(getattr(args, "gpus", None) or 1)
        util = derive_gpu_memory_utilization(card.min_vram_gb, world, pool)
        if util is not None:
            # AFTER the card's own flags (a card's blanket fallback loses to the
            # footprint-derived value) and BEFORE the user's post-`--` args (an
            # explicit --gpu-memory-utilization still wins — engine argparse is
            # last-wins).
            flags = flags + ["--gpu-memory-utilization", f"{util:g}"]
            per = card.min_vram_gb / world
            decisions.append(
                f"gpu-memory-utilization: {util:g} (derived: ~{card.min_vram_gb}GB weights / "
                f"{world} rank(s) = ~{per:.0f}GB per rank on a {pool}GB unified CPU+GPU pool "
                f"— the host keeps ~{unified_host_reserve_gb(per):.0f}GB to stream the load)")
        elif pool:
            # Say what WOULD work. "spread wider" without a number leaves the user
            # to re-derive the arithmetic boxy just did.
            fits = unified_ranks_needed(card.min_vram_gb, pool)
            remedy = (f"spread wider — {fits} ranks fit (--gpus/--nodes)" if fits
                      else "no rank count fits this pool; use a smaller or quantized variant")
            decisions.append(
                f"gpu-memory-utilization: NOT derived — ~{card.min_vram_gb / world:.0f}GB per "
                f"rank leaves a {pool}GB unified pool no room for both the weight load and "
                f"the KV cache; {remedy}, or set it by hand after `--`")
    if flags:
        args.args = flags + list(getattr(args, "args", None) or [])
    return decisions


_ACCEL_OVERLAY_KEYS = ("cuda", "rocm")


def effective_args(card_args: dict, accel: str) -> dict:
    """Flatten a card's [model.args] for ONE accelerator: scalar keys are the
    portable base; a nested [model.args.cuda]/[model.args.rocm] table overlays
    it for that accelerator (overlay wins key-by-key). NVIDIA's reference
    commands are full of CUDA-only knobs (FlashInfer autotune, FP4 MoE) that
    would crash a ROCm vLLM — per-accelerator overlays keep ONE card honest on
    both kinds of metal."""
    base = {k: v for k, v in (card_args or {}).items()
            if not (k in _ACCEL_OVERLAY_KEYS and isinstance(v, dict))}
    overlay = (card_args or {}).get(accel)
    if isinstance(overlay, dict):
        base.update(overlay)
    return base


def check_accelerator(card: ModelCard, accel: str) -> None:
    """Refuse UP FRONT when the checkpoint cannot run on this hardware (e.g. an
    NVFP4 quant on a ROCm system) — the alternative is a kernel-init death an
    hour into the queue. Raises ValueError with the card's redirect hint."""
    if card.accelerators and accel and accel not in card.accelerators:
        hint = f" {card.unsupported_hint}" if card.unsupported_hint else ""
        raise ValueError(
            f"{model_hint_name(card)}: this checkpoint runs on "
            f"{'/'.join(card.accelerators)} only, not {accel}.{hint}")


def model_hint_name(card: ModelCard) -> str:
    return f"{card.label} (match {card.match!r})"


def _flag_value(user_args, flag: str) -> str | None:
    """LAST value of `--flag V` / `--flag=V` in a raw engine-args list — the
    engines' argparse honors the last occurrence, so must we."""
    val = None
    args_list = list(user_args or [])
    for i, a in enumerate(args_list):
        if a == flag and i + 1 < len(args_list):
            val = args_list[i + 1]
        elif isinstance(a, str) and a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


_FP8_KV = ("fp8", "fp8_e4m3", "fp8_e5m2")


def _kv_dtype_factor(card_args_flat: dict, user_args) -> float:
    """0.5 when the EFFECTIVE kv-cache dtype is an fp8 variant — the card's
    flattened args first, then the user's post-`--` list (last-wins, so a user
    flipping a card's fp8 back to auto restores the bf16 cost). Scanning the
    user list matters in the DANGEROUS direction: missing that override would
    derive a context twice as large as the cache can hold."""
    val = card_args_flat.get("kv_cache_dtype")
    uval = _flag_value(user_args, "--kv-cache-dtype")
    if uval is not None:
        val = uval
    return 0.5 if str(val).lower() in _FP8_KV else 1.0


def _explicit_util(card_args_flat: dict, user_args) -> float | None:
    """An explicit gpu-memory-utilization from the card or the user's post-`--`
    args (last-wins), for the DISCRETE-part context budget; None -> vLLM's 0.9."""
    val = card_args_flat.get("gpu_memory_utilization")
    uval = _flag_value(user_args, "--gpu-memory-utilization")
    if uval is not None:
        val = uval
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _strip_flag_pair(flags: list[str], name: str) -> list[str]:
    """Remove every `name value` pair (and `name=value`) from a flag list —
    used when a DERIVED value replaces a card's static one, so the argv never
    carries a dead duplicate."""
    out: list[str] = []
    skip = False
    for f in flags:
        if skip:
            skip = False
            continue
        if f == name:
            skip = True
            continue
        if isinstance(f, str) and f.startswith(name + "="):
            continue
        out.append(f)
    return out


def engine_flags(card_args: dict) -> list[str]:
    """Turn a card's [model.args] table into engine CLI flags:
    {max_model_len: 8192} -> ['--max-model-len', '8192']; a True bool -> a bare
    '--flag' (store_true), False -> omitted. Underscores become dashes. Nested
    tables (per-accelerator overlays) never leak — flatten with effective_args
    first; anything dict-valued here is skipped defensively."""
    out: list[str] = []
    for key, val in (card_args or {}).items():
        if isinstance(val, dict):
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                out.append(flag)
        else:
            out += [flag, str(val)]
    return out


# KV cache + activations + allocator fragmentation on top of a card's advisory
# weight footprint (min_vram_gb). 1.25 is CALIBRATED: on the assumed 4x80GB
# shape it reproduces every packaged card's hand-sized gpus exactly (see
# tests), so geometry only changes when a system card declares real hardware.
_VRAM_HEADROOM = 1.25

# ---- unified-memory (APU) pools -----------------------------------------------------
#
# On MI300A-class APUs the CPU and GPU share ONE physical pool per socket, so the
# engine's claim and the host's working set come out of the same 128GB. Claim too
# much and the kernel OOM-killer reaps a rank with NO traceback (field: a 70B died
# silently after an 18-minute load at vLLM's 0.9 default; 0.7 served).
#
# The host's reserve is modelled as ~a rank's weight shard plus streaming buffers,
# but BOUNDED at both ends. The cap is what makes the model usable: the loader
# streams, so the host's transient need does not keep growing with the shard, and
# an unbounded reserve made big models look impossible — it refused to produce any
# number for Llama-4-Scout on 4 ranks, and sent it to two nodes, a configuration
# the field ran on ONE.
#
# The three constants are pinned by the two configurations observed on real
# MI300A hardware, and reproduce both exactly (see the calibration test):
#   * Llama-3.3-70B, 140GB over 4 ranks: 0.9 was OOM-killed, 0.7 served -> the
#     reserve at a 35GB shard must be ~38GB, and 2x70GB shards must stay
#     INFEASIBLE so the solver spreads to 4.
#   * Llama-4-Scout-17B-16E, 228GB over 4 ranks: ran on one node -> a 57GB shard
#     must remain FEASIBLE.
# No shard-proportional reserve fits both points monotonically, which is the
# honest limit of this model: 0.9 killed the smaller model and served the larger
# one on the same hardware, so shard size cannot be the whole story. The cap keeps
# the envelope that explains the observed FAILURE, which makes the derived value
# conservative for large models (Scout derives ~0.62 where 0.9 was seen to work).
# Conservative-but-serving beats a silent OOM kill; `-- --gpu-memory-utilization`
# raises it for anyone who has measured their own model.
_UNIFIED_LOAD_FACTOR = 1.1     # shard + streaming/allocator buffers
_UNIFIED_HOST_FLOOR_GB = 16    # OS + container + tokenizer, even for tiny models
_UNIFIED_HOST_CAP_GB = 48      # the loader streams: the host's need stops growing

# Ceiling on the unified solver's node search. A model that needs more than this
# is not a turnkey serve; say so rather than return an absurd allocation.
_MAX_SPILL_NODES = 64


def unified_host_reserve_gb(shard_gb: float) -> float:
    """GB of a unified pool the HOST keeps while a rank of `shard_gb` loads."""
    return min(_UNIFIED_HOST_CAP_GB, max(_UNIFIED_HOST_FLOOR_GB, shard_gb * _UNIFIED_LOAD_FACTOR))


def unified_claimable_gb(shard_gb: float, pool_gb: float) -> float:
    """GB of a unified pool the ENGINE may claim — the pool minus the host's reserve."""
    return pool_gb - unified_host_reserve_gb(shard_gb)


def unified_rank_fits(shard_gb: float, pool_gb: float) -> bool:
    """Can one rank hold `shard_gb` of weights plus KV/activation headroom inside
    what it is allowed to claim? This is the single feasibility test — the geometry
    solver and the utilization derivation both use it, so they cannot disagree
    (they used to: the solver would pick a rank count the derivation then refused)."""
    return unified_claimable_gb(shard_gb, pool_gb) >= shard_gb * _VRAM_HEADROOM


def unified_ranks_needed(min_vram_gb: float, pool_gb: float, limit: int = 1024) -> int | None:
    """Smallest power-of-two rank count whose shard FITS a `pool_gb` unified pool
    — the GPU need, derived from the model's footprint. None when even `limit`
    ranks cannot hold it."""
    if min_vram_gb <= 0 or pool_gb <= 0:
        return None
    ranks = 1
    while ranks <= limit:
        if unified_rank_fits(min_vram_gb / ranks, pool_gb):
            return ranks
        ranks *= 2
    return None


def derive_gpu_memory_utilization(min_vram_gb: float, world_size: int, pool_gb: int) -> float | None:
    """The vLLM --gpu-memory-utilization for ONE rank of a unified-memory pool:
    the whole pool minus what the host keeps to stream the load, capped at vLLM's
    own 0.9. None when an input is unknown, or when this rank count genuinely
    cannot hold the model — in which case the fix is more ranks, and
    unified_ranks_needed() says how many.

    Field calibration: 140GB of weights over 4 MI300A ranks on a 128GB pool
    -> 0.7, exactly the hand-tuned value that ended the silent OOM kills."""
    if min_vram_gb <= 0 or world_size <= 0 or pool_gb <= 0:
        return None
    shard = min_vram_gb / world_size
    if not unified_rank_fits(shard, pool_gb):
        return None
    return min(0.9, round(unified_claimable_gb(shard, pool_gb) / pool_gb, 2))


# ---- derived context window (max_model_len) -----------------------------------------
#
# A card's static max_model_len is a workaround sized for the SMALLEST machine
# the model might land on; on real hardware it silently wastes the KV budget
# (field: Kimi-K3's 1M native window hand-capped to 131072 on a deployment
# whose per-rank arithmetic supports the full million). When a card knows its
# KV cost per token and the system card knows the node, boxy derives the
# largest context that PROVABLY fits and serves that instead.

# Per-rank reserve (GB) for activations, CUDA graphs, sampler and allocator
# slack when translating a VRAM budget into KV-cache tokens. PLACEHOLDER
# calibration: to be pinned by the field-measured Kimi-K3 'GPU KV cache size'
# line (see the golden calibration test) — tune it there, not here.
_CTX_ACT_RESERVE_GB = 8.0
# A derived context below this is not worth serving; decline (the card's
# static cap stands) and say why instead.
_CTX_FLOOR = 4096


def derive_max_model_len(kv_bytes_per_token: float, native_ctx: int, min_vram_gb: float,
                         world: int, pp_stages: int, gpu_vram_gb: int,
                         unified: bool = False, util: float | None = None,
                         kv_dtype_factor: float = 1.0) -> tuple[int | None, str]:
    """(tokens, why): the largest context whose KV cache provably fits ONE rank,
    or (None, why-not). Deliberately CONSERVATIVE v1: the per-token cost is NOT
    divided by TP — exact for MLA models (the compressed cache is replicated
    across TP ranks), an under-estimate for GQA models (vLLM shards KV heads
    across TP) — so a derived value never OOMs where a hand-raised one might.
    PP divides layers across stages, so the per-rank cost divides by pp_stages.

    `util` = an explicit gpu-memory-utilization (card/user) on DISCRETE parts;
    None -> vLLM's 0.9 default. On unified pools the claim mirrors what
    derive_gpu_memory_utilization actually grants (min of the claimable pool
    and 0.9), so the two derivations cannot disagree about the budget.
    `kv_dtype_factor` = 0.5 when the effective kv-cache dtype is fp8."""
    if (kv_bytes_per_token <= 0 or native_ctx <= 0 or min_vram_gb <= 0
            or world <= 0 or gpu_vram_gb <= 0):
        return None, "model KV shape or node VRAM unknown"
    shard = min_vram_gb / world
    if unified:
        # same single feasibility test the solver and the util derivation use —
        # deriving a context on top of a claim the util derivation refuses to
        # grant would rebuild the exact silent-OOM the unified model exists to
        # prevent (2x70GB shards on a 128GB pool have a positive "budget" on
        # paper and a dead rank in the field).
        if not unified_rank_fits(shard, gpu_vram_gb):
            return None, (f"a ~{shard:.0f}GB weight shard does not fit a {gpu_vram_gb}GB "
                          f"unified pool (no gpu-memory-utilization derivable either)")
        claim = min(unified_claimable_gb(shard, gpu_vram_gb), 0.9 * gpu_vram_gb)
    else:
        claim = gpu_vram_gb * (util if util is not None else 0.9)
    budget = claim - shard - _CTX_ACT_RESERVE_GB
    per_tok_gb = kv_bytes_per_token * kv_dtype_factor / 1e9 / max(1, pp_stages)
    if budget <= 0:
        return None, (f"~{shard:.0f}GB weight shard + ~{_CTX_ACT_RESERVE_GB:.0f}GB reserve "
                      f"leave no KV budget in the ~{claim:.0f}GB a rank may claim")
    tokens = int(budget / per_tok_gb)
    tokens -= tokens % 1024
    if tokens < _CTX_FLOOR:
        return None, (f"~{budget:.0f}GB KV budget per rank fits fewer than "
                      f"{_CTX_FLOOR} tokens")
    tokens = min(tokens, native_ctx)
    why = (f"~{claim:.0f}GB claimable − ~{shard:.0f}GB weight shard − "
           f"~{_CTX_ACT_RESERVE_GB:.0f}GB reserve = ~{budget:.0f}GB for KV at "
           f"~{kv_bytes_per_token * kv_dtype_factor / max(1, pp_stages) / 1024:.1f}KB/token/rank"
           f" (PP={max(1, pp_stages)})")
    return tokens, why


def parse_context_request(raw, native_ctx: int) -> int:
    """`--context N|full` -> tokens. 'full' = the model's native window; k/m
    suffixes are binary (256k = 262144) because context windows are. Raises
    ValueError with the remedy — main() turns that into a clean exit-1 line."""
    s = str(raw).strip().lower()
    if s == "full":
        if not native_ctx:
            raise ValueError(
                "--context full: the model card doesn't know the native window — regenerate "
                "it with `boxy generate card <id>` or set native_ctx in the card")
        return native_ctx
    mult = 1
    if s.endswith("k"):
        mult, s = 1024, s[:-1]
    elif s.endswith("m"):
        mult, s = 1024 * 1024, s[:-1]
    try:
        n = int(s) * mult
    except ValueError:
        raise ValueError(f"--context {raw!r}: expected a token count (4096, 256k, 1m) or 'full'") from None
    if n <= 0:
        raise ValueError(f"--context {raw!r}: must be positive")
    return n


def fit_geometry(min_vram_gb: float, gpus_per_node: int, gpu_vram_gb: int,
                 unified: bool = False, *, ctx_tokens: int = 0,
                 ctx_kv_bytes_per_token: float = 0.0) -> tuple[int, int, str]:
    """(nodes, gpus_per_node, why): the smallest geometry that FITS a model card's
    min_vram_gb (the demand, plus KV/overhead headroom) on this system's nodes
    (the supply: gpus_per_node x gpu_vram_gb from the location/system card).
    Single node preferred, fewest power-of-two GPUs (TP-friendly); only when the
    model exceeds a FULL node does it spill to N full nodes — which the serve
    path then runs as one Ray instance (TP=gpus/node x PP=nodes). Unknown supply
    degrades to the same 80GB-class / 4-wide assumptions the card tiers use,
    stated in `why`.

    `unified=True` sizes against a shared CPU+GPU pool: a rank can never use the
    whole part, because the host keeps a reserve to stream the load, and that
    reserve depends on the shard — which depends on the rank count being solved
    for. So the unified path tries rank counts against unified_rank_fits()
    directly rather than discounting the pool by a fixed fraction. The old fixed
    discount double-counted headroom (it shrank the supply AND inflated the
    demand) and sent Llama-4-Scout to two nodes; the field ran it on one."""
    from boxy import config

    assumed = []
    vram = int(gpu_vram_gb) if gpu_vram_gb else 0
    if not vram:
        vram = config.get_int("cardgen.gpu_class_gb") or 80
        assumed.append(f"assuming {vram}GB-class GPUs")
    width = int(gpus_per_node) if gpus_per_node else 0
    if not width:
        width = 4
        assumed.append("assuming 4 GPUs/node")
    note = f"; {'; '.join(assumed)}" if assumed else ""

    budget = min_vram_gb * _VRAM_HEADROOM
    need = f"~{min_vram_gb:g}GB weights + KV/overhead headroom = {budget:g}GB"
    node_capacity = width * vram

    # --- context as DEMAND (--context): a candidate geometry must also leave
    # each rank room for the requested window's KV. The fit test IS the runtime
    # derivation at that geometry, so the solver can never pick a shape the
    # derivation then refuses — the same by-construction agreement
    # unified_rank_fits gives the utilization side. ctx kwargs default to 0:
    # every default-path code line below is byte-identical without them (the
    # calibration/parity contracts stand).
    want_ctx = bool(ctx_tokens and ctx_kv_bytes_per_token)

    def _ctx_ok(nodes_: int, gpus_: int) -> bool:
        if not want_ctx:
            return True
        t, _ = derive_max_model_len(ctx_kv_bytes_per_token, ctx_tokens, min_vram_gb,
                                    nodes_ * gpus_, nodes_, vram, unified=unified)
        return t is not None and t >= ctx_tokens

    ctx_note = f"; holds a {ctx_tokens}-token context" if want_ctx else ""

    def _ctx_infeasible() -> str:
        best, _ = derive_max_model_len(ctx_kv_bytes_per_token, 10 ** 12, min_vram_gb,
                                       width * _MAX_SPILL_NODES, _MAX_SPILL_NODES,
                                       vram, unified=unified)
        remedy = (f"the largest context that fits there is ~{best} tokens"
                  if best else "no context fits at all")
        return (f"KV for a {ctx_tokens}-token context does not fit even "
                f"{_MAX_SPILL_NODES} nodes ({width}x{vram}GB each); {remedy}")

    if unified:
        def _pool_note(ranks: int) -> str:
            shard = min_vram_gb / ranks
            return (f"; the {vram}GB pool is shared with the host, leaving each rank "
                    f"~{unified_claimable_gb(shard, vram):.0f}GB to claim for a "
                    f"~{shard:.0f}GB shard")

        gpus = 1
        while gpus <= width:
            if unified_rank_fits(min_vram_gb / gpus, vram) and _ctx_ok(1, gpus):
                return 1, gpus, (f"{need}; a node offers {width}x{vram}GB"
                                 f"{_pool_note(gpus)}{ctx_note}{note}")
            gpus *= 2
        nodes = 2
        while nodes <= _MAX_SPILL_NODES:
            if unified_rank_fits(min_vram_gb / (width * nodes), vram) and _ctx_ok(nodes, width):
                return nodes, width, (f"{need} exceeds one node ({width}x{vram}GB)"
                                      f"{_pool_note(width * nodes)}{ctx_note}{note}")
            # a context demand scans every node count (PP need not be a power
            # of two); the default weight-only spill keeps its doubling steps
            nodes = nodes + 1 if want_ctx else nodes * 2
        if want_ctx:
            return _MAX_SPILL_NODES, width, f"{need}; {_ctx_infeasible()}{note}"
        return _MAX_SPILL_NODES, width, (f"{need} does not fit {_MAX_SPILL_NODES} unified "
                                         f"nodes ({width}x{vram}GB each); pin --gpus/--nodes "
                                         f"and --gpu-memory-utilization by hand{note}")

    if budget <= node_capacity:
        gpus = 1
        while gpus * vram < budget:
            gpus *= 2
        gpus = min(gpus, width)
        if _ctx_ok(1, gpus):
            return 1, gpus, f"{need}; a node offers {width}x{vram}GB{ctx_note}{note}"
        # widen within the node first: smaller shards free per-rank KV budget
        while gpus < width:
            gpus = min(gpus * 2, width)
            if _ctx_ok(1, gpus):
                return 1, gpus, (f"{need}; widened to {gpus} GPUs so the KV for a "
                                 f"{ctx_tokens}-token context fits{note}")
        nodes = 2
    else:
        nodes = -(-int(budget) // node_capacity)               # ceil
        if not want_ctx:
            return nodes, width, (f"{need} exceeds one node ({width}x{vram}GB = "
                                  f"{node_capacity}GB){note}")
    while nodes <= _MAX_SPILL_NODES:
        if budget <= nodes * node_capacity and _ctx_ok(nodes, width):
            return nodes, width, (f"{need}; KV for a {ctx_tokens}-token context fits at "
                                  f"{nodes} nodes / PP={nodes} ({width}x{vram}GB each){note}")
        nodes += 1
    return _MAX_SPILL_NODES, width, f"{need}; {_ctx_infeasible()}{note}"


def system_shape(cluster: str) -> tuple[int, int, str] | None:
    """(gpus_per_node, gpu_vram_gb, card_stem) — the node HARDWARE a system card
    declares for `cluster`, resolved through the normal system-card matching
    (user dir wins; canonical [location].name first, file stem fallback). Write
    ~/.config/boxy/cards/systems/clusterc.toml once with
        [location.resources]
        gpus_per_node = 4
        gpu_vram_gb = 140
    and every serve against that cluster derives its geometry from cards alone.
    None when no card names the cluster or the card carries no shape."""
    hit = _match_system_card(cluster)
    if hit is None:
        return None
    text, stem = hit
    try:
        res = (tomllib.loads(text).get("location") or {}).get("resources") or {}
        shape = (int(res.get("gpus_per_node", 0)), int(res.get("gpu_vram_gb", 0)))
    except (tomllib.TOMLDecodeError, TypeError, ValueError):
        return None
    return (shape[0], shape[1], stem) if any(shape) else None


def system_unified_memory(cluster: str) -> bool:
    """True when the cluster's system card declares
        [location.resources]
        unified_memory = true
    — APU parts (MI300A-class) where CPU and GPU share one physical pool, so
    apply_to_args derives --gpu-memory-utilization from the model's footprint
    instead of letting vLLM's 0.9 default starve the host mid-load. False on
    no card / no flag / a card that doesn't parse."""
    hit = _match_system_card(cluster)
    if hit is None:
        return False
    try:
        res = (tomllib.loads(hit[0]).get("location") or {}).get("resources") or {}
        return bool(res.get("unified_memory", False))
    except tomllib.TOMLDecodeError:
        return False
