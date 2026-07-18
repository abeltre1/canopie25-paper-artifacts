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
    args: dict = field(default_factory=dict)
    # extra pip packages the model's custom code imports that the engine image
    # doesn't ship (installed at container start; field: Nemotron-Parse/open_clip)
    pip: list = field(default_factory=list)
    # auxiliary HF repos the model's custom code fetches DYNAMICALLY (e.g. its
    # vision encoder) — `boxy bundle` must pre-cache them or an air-gapped serve
    # dies mid-import (field: Nemotron-Parse pulls nvidia/C-RADIOv2-H)
    aux_repos: list = field(default_factory=list)

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
        args=dict(args),
        pip=[str(x) for x in section.get("pip", [])],
        aux_repos=[str(x) for x in section.get("aux_repos", [])],
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
    key = model_key(model)
    best: ModelCard | None = None
    for card in load_cards():
        if not (fnmatch.fnmatchcase(key, card.match) or key == card.match):
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
    key = model_key(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not (fnmatch.fnmatchcase(key, card.match) or key == card.match):
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
    key = model_key(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not (fnmatch.fnmatchcase(key, card.match) or key == card.match):
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


def layered_aux_repos(model: str) -> list:
    """Auxiliary HF repos (dynamically fetched by the model's custom code) from
    the best packaged + user cards — `boxy bundle` pre-caches every one so an
    air-gapped serve never reaches for the network mid-import."""
    key = model_key(model)
    best: dict[str, ModelCard] = {}
    for card in load_cards():
        if not (fnmatch.fnmatchcase(key, card.match) or key == card.match):
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


def apply_to_args(args, shape: tuple[int, int, str] | None = None) -> list[str]:
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
    (and a card's own explicit nodes) always bypass the solver."""
    decisions: list[str] = []
    model = getattr(args, "model", None)
    if not model:
        return decisions
    card = resolve_model_card(model)
    if card is None:
        return decisions
    if card.source == "generated":
        wrote = f" -> {_last_autogen['note']}" if _last_autogen["note"] else ""
        decisions.append(
            f"card: generated deterministically from HuggingFace metadata "
            f"(~{card.min_vram_gb}GB weights, engine {card.engine or 'vllm'}){wrote}")
    elif card.source == "heuristic" and _last_autogen["fail"]:
        decisions.append(
            f"card: {_last_autogen['fail']} — geometry below is a NAME GUESS; run "
            f"`boxy generate card {model_key(model)}` from a connected machine for the real numbers")
    gpus_free = getattr(args, "gpus", None) is None
    nodes_free = getattr(args, "nodes", None) is None
    if gpus_free and nodes_free and card.min_vram_gb and not card.nodes:
        w, v, src = shape or (0, 0, "")
        nodes, gpus, why = fit_geometry(card.min_vram_gb, w, v)
        args.gpus = gpus
        src_note = f"; {src}" if src else ""
        decisions.append(f"gpus: {gpus} per node ({card.label}: {why}{src_note})")
        if nodes > 1:
            args.nodes = nodes
            decisions.append(f"nodes: {nodes} ({card.label}: the model exceeds one node -> "
                             f"one Ray instance across {nodes} nodes)")
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
    # engine args from the card (e.g. max_model_len so vLLM doesn't profile KV
    # cache for the model's full 128K context and OOM). Card flags go FIRST so
    # the user's own post-`--` engine args, appended after, win (last-wins in the
    # engine's argparse). Field failure: bare 8B serve OOM'd because this table
    # was never applied.
    flags = engine_flags(card.args)
    if flags:
        args.args = flags + list(getattr(args, "args", None) or [])
        decisions.append(f"engine args: {' '.join(flags)} ({card.label})")
    return decisions


def engine_flags(card_args: dict) -> list[str]:
    """Turn a card's [model.args] table into engine CLI flags:
    {max_model_len: 8192} -> ['--max-model-len', '8192']; a True bool -> a bare
    '--flag' (store_true), False -> omitted. Underscores become dashes."""
    out: list[str] = []
    for key, val in (card_args or {}).items():
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


def fit_geometry(min_vram_gb: float, gpus_per_node: int, gpu_vram_gb: int) -> tuple[int, int, str]:
    """(nodes, gpus_per_node, why): the smallest geometry that FITS a model card's
    min_vram_gb (the demand, plus KV/overhead headroom) on this system's nodes
    (the supply: gpus_per_node x gpu_vram_gb from the location/system card).
    Single node preferred, fewest power-of-two GPUs (TP-friendly); only when the
    model exceeds a FULL node does it spill to N full nodes — which the serve
    path then runs as one Ray instance (TP=gpus/node x PP=nodes). Unknown supply
    degrades to the same 80GB-class / 4-wide assumptions the card tiers use,
    stated in `why`."""
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
    if budget <= node_capacity:
        gpus = 1
        while gpus * vram < budget:
            gpus *= 2
        gpus = min(gpus, width)
        return 1, gpus, f"{need}; a node offers {width}x{vram}GB{note}"
    nodes = -(-int(budget) // node_capacity)               # ceil
    return nodes, width, (f"{need} exceeds one node ({width}x{vram}GB = "
                          f"{node_capacity}GB){note}")


def system_shape(cluster: str) -> tuple[int, int, str] | None:
    """(gpus_per_node, gpu_vram_gb, card_stem) — the node HARDWARE a system card
    declares for `cluster`, resolved through the normal system-card matching
    (user dir wins; canonical [location].name first, file stem fallback). Write
    ~/.config/boxy/cards/systems/cronus.toml once with
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
