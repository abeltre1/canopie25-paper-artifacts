# Any model from Hugging Face, systematically

**Status: design — no implementation in this PR.**

## The problem, stated honestly

Deploying Kimi-K3 (PR #19) required a human (or an AI assistant) to read the
vLLM recipe page and hand-author a card. That worked, and the card is good —
but it is not a pipeline. If every frontier model needs a bespoke reading
session, boxy's "one command serves it" promise only covers models someone
already sat down for. The ask: a **systematic, programmatic path from an
arbitrary `hf://org/repo` to a running endpoint**, with human judgment needed
only where it genuinely cannot be automated.

## What is already systematic (and underappreciated)

Boxy ships this today, at serve time, with zero user action. When
`boxy serve hf://<org>/<repo>` matches no card, `cardgen.auto_card()` runs:

```
config.json ─┐
             ├─→ engine pick (gguf→llama.cpp | architectures→vllm)
safetensors  ├─→ exact weight bytes → min_vram_gb   (index reflects quantization)
   index ────┤   MoE detection (experts, activated)
             ├─→ dtype/quant label (fp8/int4/mxfp4 → bytes-per-param)
generation_  ├─→ native context → capped max_model_len (KV-profile OOM guard)
   config ───┤   trust_remote_code (config auto_map — the Nemotron-Parse lesson)
             └─→ vision detection → limit-mm-per-prompt
                      ↓
        rendered card → validated by boxy's own loader → written to user cards
                      ↓
        fit_geometry(min_vram_gb, node shape) → gpus/nodes/TP×PP/Ray
                      ↓
        gpu-memory-utilization derived (unified pools) → serve
```

A *plain* HF model — standard architecture, no exotic parsers — already
deploys end-to-end through this path on any cluster with a system card. That
is most of the catalog.

## The gap: exactly four things HF metadata cannot carry

The Kimi-K3 card needed a human for precisely these, and only these:

| Gap | Example (Kimi-K3) | Where it actually lives |
|---|---|---|
| 1. Parser names | `--tool-call-parser kimi_k3`, `--reasoning-parser kimi_k3` | vLLM's parser registry + the model's recipe |
| 2. Serving image | `vllm/vllm-openai_rocm:kimi-k3` (day-0 ROCm tag) | the recipe; nowhere in HF metadata |
| 3. Vendor env | `VLLM_ROCM_USE_AITER=1` | the recipe's per-hardware sections |
| 4. Judgment caps | 256K not 1M context; no `fastsafetensors` on unknown images | boxy policy (already encoded, per-card today) |

Items 1–3 have a **single canonical, machine-readable-enough source**:
[`vllm-project/recipes`](https://github.com/vllm-project/recipes) — the markdown
behind recipes.vllm.ai. Every recipe is a doc with fenced ```bash blocks
containing literal `vllm serve …`, `export …`, and `docker run …` lines, organized
under per-hardware headings. Item 4 is policy boxy already owns; it just needs to
be applied by machinery instead of by hand per card.

## Design: three tiers, one merge policy

### Tier 0 — HF metadata (exists; the default; never blocks)

Unchanged. `auto_card()` remains the floor: any model, no network beyond the
Hub, best-effort card, serve proceeds. All higher tiers only *add* keys.

### Tier 1 — recipe ingestion (`generate card --recipe`, and auto-probe)

New module `recipegen.py`:

1. **Fetch** `raw.githubusercontent.com/vllm-project/recipes/main/{Org}/{Model}.md`
   (org/model matching is case-normalized against the repo's directory listing;
   404 → tier 0 silently, one `auto:` note).
2. **Parse**, not scrape: walk fenced code blocks; keep lines matching
   `vllm serve`, `export KEY=VALUE`, and image references
   (`docker.io/…`, `rocm/…`, `vllm/…`, `nvcr.io/…`). Shell-split with `shlex` —
   no regex-over-prose. Attribute blocks to the nearest hardware heading
   (`## MI325X`, `## H100`, `## B200`…) to fill per-accelerator overlays
   (`[model.args.cuda]` / `.rocm`, `[model.env.cuda]` / `.rocm` — the card
   schema already supports these).
3. **Translate** flags → `[model.args]` keys (the existing `engine_flags()`
   mapping, inverted), env → `[model.env]`, images → `[model.images]`.
4. **Record provenance** in the card header: recipe path + git SHA of the
   recipes repo at ingestion time. A card must say where its flags came from.

What tier 1 explicitly does NOT do: trust the recipe over measurements.
Geometry keys (`--tensor-parallel-size`, `--data-parallel-size`, GPU counts)
are **dropped with a note** — boxy derives geometry from `min_vram_gb` × the
system card, which is how the same card serves 1×MI325X node and 4×H100 nodes.
A recipe's TP number is a fact about the recipe author's hardware, not the model.

### Tier 2 — vetting (`boxy vet <model>`): verify before burning an allocation

The Kimi-K3 failure mode worth engineering against: a card names a parser or
flag the *image* doesn't have, and the error surfaces 20 minutes into a batch
job. `boxy vet` checks the card **against its pinned image, hardware-free**:

1. `<runtime> run <image> vllm serve --help` → every `[model.args]` key must
   appear as a recognized flag; `--tool-call-parser` / `--reasoning-parser`
   values must be in the help text's listed choices (vLLM prints its parser
   registry there). Catches "kimi_k3 doesn't exist in this old image" on the
   laptop, in seconds.
2. Optional, needs one GPU (`--load`): start the engine with
   `--load-format dummy` — vLLM validates config, parsers, and memory profile
   with random weights, no download. This is the full-fidelity check for a new
   card before the real 1.5TB run.
3. Report in `boxy doctor` style: OK / WARN / FAIL per check, exit nonzero on
   FAIL. `serve` may grow `--vet` to run check 1 inline before submission.

### Merge policy (one rule, printed as `auto:` lines like everything else)

```
user flags / user card  >  recipe-derived  >  HF-metadata-derived  >  boxy policy caps
```

with two standing exceptions, both surfaced as notes rather than applied
silently: geometry is always derived (recipe TP dropped, see above), and
policy caps clamp rather than yield (context cap, no `fastsafetensors`
without image proof — a slow first load beats a crashed one). Every clamped
or dropped key is one `auto:` line naming what and why.

## Air-gapped / filtered sites

The recipes repo is one more thing a filtered laptop cannot fetch. Two-part
answer, mirroring how packaged cards already work:

- **`boxy recipes sync`**: clone/refresh a local snapshot of
  `vllm-project/recipes` (a plain git repo; works through `--proxy`, or from a
  site mirror path). Ingestion reads the snapshot when present — no network.
- **Release-time enrichment**: at each boxy release, run tier 1 over the
  models that already have packaged cards and refresh them (CI job, diff
  reviewed like any PR). Frontier models get packaged cards by machinery;
  hand-authoring becomes the review step, not the authoring step.

## What stays manual, and is said out loud

- A model whose architecture vLLM does not support at all — no pipeline fixes
  that; `vet` reports it (arch not in the image's registry) instead of boxy
  discovering it in a job log.
- A model with no recipe and exotic serving needs — tier 0 card serves or
  fails visibly; the fix is a user card, now with `vet` to check it.
- Trust: recipes execute nothing here — ingestion is parsing text into TOML the
  user can read, with provenance, validated by boxy's own loader before write.

## Rollout

| PR | Contents | Size |
|---|---|---|
| A | `recipegen.py` fetch+parse+translate, `generate card --recipe`, provenance headers, golden tests against vendored recipe fixtures | medium |
| B | `boxy vet` check 1 (help-text vetting) + doctor-style report; `serve --vet` | small |
| C | `boxy recipes sync` + snapshot-first resolution | small |
| D | release CI: recipe-refresh of packaged cards; `vet --load` (GPU dummy-load) | small |

Acceptance test for the whole design: **regenerate the Kimi-K3 card from
machinery alone** (tier 0 sizing + tier 1 recipe ingestion + policy caps) and
diff it against the hand-authored PR #19 card. The diff should be empty except
comments — that hand-authored card is the golden fixture.
