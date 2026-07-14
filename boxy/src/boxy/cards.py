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
    min_vram_gb = 140        # advisory, printed in the decision line
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
    min_vram_gb: int = 0           # advisory only
    args: dict = field(default_factory=dict)

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


def resolve_model_card(model: str) -> ModelCard | None:
    """Card if one matches, else the size heuristic, else None."""
    return find_card(model) or size_heuristic(model)


def apply_to_args(args) -> list[str]:
    """Turnkey fill for a SCHEDULER submission: when --gpus/--nodes/--engine are
    absent, take them from the model's card (or the size heuristic), returning
    the decision lines to print. Explicit flags always win; local (no-scheduler)
    serves are untouched — there the detected accelerator already drives GPU
    use, and injecting --gpus would change behavior."""
    decisions: list[str] = []
    model = getattr(args, "model", None)
    if not model:
        return decisions
    card = resolve_model_card(model)
    if card is None:
        return decisions
    if getattr(args, "gpus", None) is None and card.gpus:
        args.gpus = card.gpus
        note = f" (~{card.min_vram_gb}GB VRAM)" if card.min_vram_gb else ""
        decisions.append(f"gpus: {card.gpus} per node ({card.label}, sized for 80GB-class GPUs{note})")
    if getattr(args, "nodes", None) is None and card.nodes:
        args.nodes = card.nodes
        decisions.append(f"nodes: {card.nodes} ({card.label})")
    if getattr(args, "engine", None) is None and card.engine:
        args.engine = card.engine
        decisions.append(f"engine: {card.engine} ({card.label})")
    return decisions
