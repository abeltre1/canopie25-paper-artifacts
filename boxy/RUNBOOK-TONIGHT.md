# boxy — tonight's runbook

Hands-on validation of this session's work: the **WCID picker**, **GRES
self-heal**, **blocked-pull diagnosis**, **scheduler auto-detect**, and
**`generate card`**. Everything runs from your **laptop** over `--ssh` — nothing
is installed on the HPC systems (agentless).

## 0. One-time setup (laptop)

```bash
# update boxy on your laptop to this branch
cd <path-to>/canopie25-paper-artifacts
git fetch origin && git checkout claude/boxy-turnkey && git pull
python -m pip install -e boxy            # or: pipx reinstall, per your setup

# gated meta-llama repos need a token that has accepted the license
export HF_TOKEN=hf_xxxxxxxx
boxy --version
```

Model used below (swap freely): `hf://meta-llama/Llama-3.1-8B-Instruct`.

---

## 1. hops — the full turnkey path (Slurm + CUDA)

```bash
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops
```

**Expect, in order:**
- `auto: scheduler: slurm …` (detected live, not guessed)
- **WCID menu** if `mywcid` lists several accounts:
  ```
  Select a charge account (WCID):
    1) fy140001  system software and tools
    2) fy140252  common computing environment
    3) fy260064  the genesis project
  account [1-3] (Enter = …): 2
    auto: account: fy140252 (you picked 2 of 3 from mywcid on hops)
  ```
  (Your pick is remembered per cluster — Enter reuses it next time.)
- `#SBATCH --gpus-per-node=1` in the batch script (the **proven default** — hops
  is no longer flipped to `--gres=gpu:h100`).
- `### Agentless (no boxy on the cluster)` → `### Submitted slurm job …`
- Progress → `### READY  http://…:8000/v1`

**If the image pull 403s** (`registry-1.docker.io: 403`, the Docker Hub block you
hit), boxy now prints the fix. Do this once, then rerun the same command:

```bash
ssh ambelt@hops podman pull docker.io/vllm/vllm-openai:latest   # login node has the network
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops
```

**Bypass the WCID menu** (scripts/CI): pick up front —
```bash
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops --account fy140252
#   or:  WCID=fy140252 boxy serve …          (env bypass)
```

**Preview without submitting** (see the exact batch script first):
```bash
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops --dryrun
```

---

## 2. kahuna — the GRES self-heal (Slurm that rejects `--gpus-per-node`)

```bash
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@kahuna
```

**Expect:** the first submit is rejected (`Invalid generic resource (gres)
specification`), then boxy **recovers on its own** — no env var, no rerun:
```
boxy: the site rejected the GPU request; retrying with --gres=gpu:N ...
### GPU request accepted as --gres=gpu:N (auto-recovered).
```
It cycles typed → untyped → `--gpus` until one is accepted. (Pin it if you ever
want to skip the probe: `export BOXY_GPU_DIRECTIVE=gres`.)

---

## 3. eldorado — Flux auto-detect (Flux + ROCm)

```bash
boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@eldorado
```

**Expect:** `auto: scheduler: flux (Flux broker is live …)` — a live Flux broker
wins over slurm-compat shims; the script uses `# flux: --bank=<wcid>` and a single
`--queue`. Force it if you prefer: add `--scheduler flux`.

Repeat on **cronus** the same way — the scheduler is detected per system, so the
identical command works across all of them.

---

## 4. `generate card` — add any model in one command (laptop, needs HF reachable)

```bash
# preview (writes nothing):
boxy generate card meta-llama/Llama-3.1-8B-Instruct --dry-run
boxy generate card openai/gpt-oss-20b --dry-run
boxy generate card meta-llama/Llama-3.3-70B-Instruct --dry-run
boxy generate card meta-llama/Llama-4-Scout-17B-16E-Instruct --dry-run

# write one, then serve it (now sized automatically):
boxy generate card openai/gpt-oss-20b
boxy serve hf://openai/gpt-oss-20b --ssh ambelt@hops
```

**Sanity-check the sizing** it derives (`gpus`, `min_vram_gb`, `max_model_len`,
engine). Overrides: `--engine vllm|llama.cpp`, `--max-model-len N`, `--force`
(overwrites, keeps a `.bak`).

---

## 5. Quick checks & teardown

```bash
boxy doctor --ssh ambelt@hops     # account/partition/scheduler/runtime probe
boxy list --ssh ambelt@hops       # running instances
curl -s http://localhost:8000/v1/models   # once READY prints the tunnel
boxy stop boxy-llama-3.1-8b-instruct --ssh ambelt@hops
```

---

## What to send back

For each system (hops / kahuna / eldorado / cronus): the last ~20 lines of
`boxy serve …` output — especially any line that isn't `auto:`/`###`/`READY`.
The three things I most want to confirm live:
1. hops shows `#SBATCH --gpus-per-node=1` again (not `--gres=gpu:h100`).
2. kahuna prints the `auto-recovered` GRES line and reaches READY.
3. the WCID menu appears and your pick lands in the batch script.

Anything that stalls or errors — paste it; the failure text now carries the fix,
and I'll turn any gap into a patch.
