# 13 — Release runbook: AIR-GAPPED

Zero to a served model with **no egress at any point inside the gap**. Two
machines: a **connected build machine** (has internet + a container runtime)
and the **inside cluster** (has neither).

Everything that will ever be needed inside must cross once, deliberately. This
runbook is the order to do it in, and the checks that catch a bad transfer
*before* it costs you a trip.

> Companion checklist (certificates, spack, registries, transfer paperwork):
> [04-airgap.md](04-airgap.md). This document is the executable path.

---

## Phase 0 — On the connected build machine (30 min + download time)

### 0.1 Install boxy from a checkout

`boxy wheels` builds boxy's own wheel from your working tree, so it needs an
**editable** install — a frozen copy is refused with that exact advice.

```bash
git clone <repo-url> && cd */boxy
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[ramalama]' huggingface_hub
boxy --version           # EXPECT: a git sha + branch, NOT "(installed copy)"
```

`huggingface_hub` is what fills the bundle's model cache. Without it (and
without `huggingface-cli`/`hf` on PATH) `boxy bundle` stops immediately and
says so.

### 0.2 Build the wheel set that installs boxy INSIDE

```bash
boxy wheels -o boxy-wheels/                     # linux/amd64 + python 3.12
# ARM/Grace target:      --platform linux/arm64
# older inside python:   --python 3.11
```

This builds **inside a `python:<ver>` container for the TARGET platform** and
then verifies the set with a `--network=none` install before you carry it. The
pinning is the point: a laptop-native `pip download` silently produces aarch64
wheels that no x86_64 cluster can install.

### 0.3 Build the model bundle

```bash
export HF_TOKEN=hf_...            # only for gated repos; do NOT carry it across
boxy bundle hf://<org>/<model> -o <model>-bundle/ --accelerator cuda --bake
```

- `--accelerator` picks the engine image family for the **target** (`cuda` or
  `rocm`) — one bundle per accelerator family.
- `--bake` pre-installs the card's pip deps into a derived image, so the
  container starts inside the gap with no pip step at all. Recommended.
- The image comes from the model's **card pin** when it has one, so the archive
  holds exactly the image the serve will name.

Result:

```
<model>-bundle/
  hfcache/          model + the aux repos its custom code fetches dynamically
  wheels/           the card's pip deps (belt-and-suspenders for un-baked)
  image.oci.tar     engine image (with --bake: deps already installed)
  manifest.toml     model, image, aux_repos, pip, created
```

### 0.4 VERIFY BEFORE CROSSING (do not skip)

A bad bundle is only discoverable on the far side, where nothing can be
re-downloaded.

```bash
# the model really landed IN the bundle (not in ~/.cache/huggingface)
ls <model>-bundle/hfcache/hub/ | grep models--          # EXPECT: your repo(s)
du -sh <model>-bundle                                    # EXPECT: ~model size

# the archive is real and matches the manifest
cat <model>-bundle/manifest.toml
tar tf <model>-bundle/image.oci.tar >/dev/null && echo "archive OK"

# nothing secret rides along
grep -rl 'hf_' <model>-bundle/manifest.toml && echo "LEAK — remove it" || echo "clean"

# record checksums for the transfer paperwork
sha256sum <model>-bundle/image.oci.tar <model>-bundle/manifest.toml
```

**Dress rehearsal (strongly recommended):** serve the bundle on a *connected*
cluster first. It exercises the identical offline path, and a `### READY` there
means the same directory works inside.

```bash
boxy serve hf://<org>/<model> --bundle /path/<model>-bundle --ssh <connected-cluster>
```

### 0.5 Cross the gap

```bash
tar cf - <model>-bundle boxy-wheels | <your approved transfer>
```

Extract so the **bundle lives on the shared filesystem** the compute nodes can
read (`/projects/...`, not `$HOME` on a login-only volume).

---

## Phase 1 — Inside the gap

### 1.1 Install boxy with no index

```bash
python3.12 -m venv boxy-env && . boxy-env/bin/activate
pip install --no-index --find-links boxy-wheels/ boxy-hpc
boxy --version
```

### 1.2 Silence the outside world

Two laptop-side conveniences still reach for the Hub; inside they only waste
your time.

```bash
export HF_HUB_OFFLINE=1        # also disables card auto-generation
export BOXY_NO_PREFLIGHT=1     # skips the HF config.json sanity probe
export BOXY_PROXY=             # the shipped default proxy is meaningless here
```

### 1.3 Audit the cluster before spending an allocation

```bash
boxy doctor --ssh <inside-login>
```

Expect OK for container runtime, scheduler, and shared FS. TLS/registry checks
may WARN — inside the gap that is correct, because nothing is fetched.

### 1.4 Serve

```bash
boxy serve hf://<org>/<model> \
    --bundle /projects/me/<model>-bundle \
    --ssh <inside-login> --time 4:00:00
```

What the rendered script does — verify with `--dryrun` first if you like:

- `podman load -i <bundle>/image.oci.tar` **on every node of the allocation**
  (a multi-node Ray serve runs a worker container on each node, and inside the
  gap a node that never loaded the archive cannot pull it). A failed load
  **aborts the job** naming the archive, instead of degrading into an
  impossible registry pull discovered after the queue wait.
- mounts `<bundle>/hfcache` at `/root/.cache/huggingface` with
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- installs the card's pip deps `--no-index --find-links` from `<bundle>/wheels`
- runs with `--pull=never`, so a missing image fails in seconds naming the real
  problem
- carries **zero** network configuration: no proxy, no CA, no token

### 1.5 Reach the endpoint

```bash
boxy attach --ssh <inside-login>          # tunnel + READY + the URL
curl -s http://127.0.0.1:8000/v1/models   # the id to use in requests
boxy curl --ssh <inside-login> --prompt "hello"   # or ask from the login node
```

### 1.6 Tear down

```bash
boxy stop <name>                    # from the READY banner / boxy list
boxy clean                          # finished-job records, scripts, logs
```

Never `boxy clean --deep --hfcache` on a bundle host: that deletes the staged
model you carried across.

---

## Multi-node inside the gap

Multi-node works, with one image caveat: vLLM's ROCm `:latest` image ships
**no ray**, and boxy's self-heal (`pip install ray` at container start) cannot
work air-gapped. Either:

- bundle an image that already contains ray (`--image <one with vllm[ray]>`), or
- add `ray` to the model card's `pip` list before `boxy bundle`, so it rides in
  `wheels/` and installs `--no-index` inside.

Verify before crossing: `podman run --rm <image> python3 -c 'import ray'`.

---

## Troubleshooting

| Symptom (inside) | Cause | Fix |
|---|---|---|
| `podman load failed … cannot be pulled inside the gap` | archive missing/corrupt on some node, or the bundle path isn't readable there | `sha256sum` the archive against Phase 0.4; confirm the path is on the shared FS |
| Engine exits instantly, image "not known" | the run named an image the archive doesn't hold | `cat manifest.toml`; re-bundle with `--image <the one you want>` |
| `CERTIFICATE_VERIFY_FAILED` anywhere | something is still trying to reach the outside | you are not in `--bundle` mode, or `HF_HUB_OFFLINE` is unset |
| Model downloads instead of loading | the serve isn't using the bundle | pass `--bundle`; note `--bundle` needs `--ssh` (it is an agentless-path flag) |
| `pip install` attempts during startup | deps weren't baked and wheels/ is missing | rebuild with `--bake`, or check `<bundle>/wheels` crossed |
| Ray workers never join | image without ray, self-heal can't reach PyPI | see *Multi-node inside the gap* |

---

## What is verified where

| Step | How it was verified |
|---|---|
| bundle contents land in the bundle | unit test pins the download destination against a hostile `HF_HOME` |
| card-pinned image is what gets bundled | unit test (and `--image` still wins) |
| offline serve script shape | rendered-script test: load, mounts, offline env, `--no-index`, no proxy |
| load fans out to every node | rendered-script test for a 4-node allocation |
| `--pull=never` under air-gap | backend test |
| a real air-gapped cluster | **your systems** — this runbook is the procedure |
