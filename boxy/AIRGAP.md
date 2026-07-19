# Air-gap readiness — checklist and runbook

Moving boxy deployments behind an air gap means **everything** must cross once,
deliberately. boxy's `bundle` / `push` / `--bundle` machinery carries the model
side; this checklist covers the rest — the items that bite only after you're
inside.

## What crosses the gap (per model)

One `boxy bundle` directory carries it all:

```bash
boxy bundle nvidia/NVIDIA-Nemotron-Parse-v1.2 -o nemotron-bundle/ --bake
#   hfcache/         model + aux custom-code repos (C-RADIOv2-H pre-cached)
#   image.oci.tar    engine image — with --bake, pip deps already INSTALLED
#   wheels/          the card's pip deps (belt-and-suspenders for the un-baked case)
#   manifest.toml    what's inside
```

- **`--bake` is recommended for the gap**: the derived image has `open_clip_torch`
  etc. installed, so the container starts with no pip step at all.
- **Build the bundle for the TARGET's accelerator**: `--accelerator rocm` picks
  the ROCm vLLM image for an MI300 system (clusterc/clusterb class), `cuda` for
  H100/H200 (clustera class). One bundle per accelerator family.
- **Verify BEFORE crossing** — serve from the bundle on a connected cluster
  first; it exercises the identical offline path:
  `boxy serve MODEL --bundle /path/nemotron-bundle --ssh <connected-cluster>`.
  If it reaches `### READY` there, the same directory works inside.

## The checklist nobody remembers

**Software that must live inside:**
- [ ] **boxy itself**: `make wheel` → carry `dist/boxy_hpc-*.whl` + a `pip download
      -d wheels/ boxy-hpc[ramalama,s3]`-style wheel set, or publish to the inside
      Nexus (`make publish LOCAL_PYPI=...`). uv/pip inside must point at the
      inside index (RELEASING.md).
- [ ] **The chisel relay image** (`quay.example.gov/user1/chisel:1.10.1`) if the
      everyone-URL share is needed inside — mirror it to the inside registry.
- [ ] **App-card toolchains**: spack sources for `boxy app` cards. Pre-populate a
      spack source mirror (the same `_source-cache/archive/<sha>` layout boxy's
      spack heal uses) or a spack build cache on the inside shared FS.
- [ ] **awscli/boto3 wheels** if `boxy push s3://` will run inside against an
      inside object store.

**Certificates & identity (different world inside):**
- [ ] The inside has its OWN CA — run `boxy trust <inside-host>` against inside
      services (registry, object store) after crossing; the outside interceptor
      CA is irrelevant there.
- [ ] No HF tokens are needed inside (everything HF is pre-bundled) — do NOT
      carry `HF_TOKEN` across.
- [ ] `boxy doctor` inside, first thing: it audits runtime/scheduler/TLS and
      names what's missing.

**Config that assumes the outside:**
- [ ] `network.proxy` default (`proxy.example.gov`) is meaningless inside —
      `export BOXY_PROXY=` (empty) or set the inside proxy in config. `--bundle`
      serves already strip it automatically.
- [ ] Registry mirrors: point `location.image_mirrors` / `--registry` at the
      inside registry for anything not in a bundle.
- [ ] The OpenShift `apps_domain` (relay URLs) differs inside — rediscovery is
      automatic when `oc` is logged into the inside cluster; pin
      `BOXY_APPS_DOMAIN` otherwise.

**Data hygiene for the transfer:**
- [ ] Checksums: bundles are content-addressed where it matters (HF blobs,
      wheel hashes), but record `sha256sum image.oci.tar manifest.toml` for the
      transfer-approval paperwork.
- [ ] Size budget: a 70B-class bundle is ~150 GB (weights) + ~20 GB (image).
      Plan media/transfer-window accordingly; `du -sh <bundle>` before asking.
- [ ] Strip secrets: `grep -r hf_ <bundle>` should find nothing — tokens never
      belong in a bundle.

**Operations inside:**
- [ ] Kill switch: `boxy stop NAME` cancels the job; `boxy stop --all` sweeps
      every live boxy job this machine has records for.
- [ ] Cleanup: `boxy clean` removes finished-job records/scripts/logs (laptop +
      cluster shared FS) and exited containers; `boxy clean --deep --ssh <host>`
      clears the cluster's agentless dir (model cache kept unless `--hfcache`).
- [ ] Updates cadence: decide how new models/images cross (periodic bundle
      drops vs. an inside registry/Nexus that a data diode feeds).

## Serving inside

```bash
boxy serve nvidia/NVIDIA-Nemotron-Parse-v1.2 \
    --bundle /projects/me/nemotron-bundle --ssh <inside-cluster>
```

The batch script `podman load`s the image from the bundle, mounts its HF cache
with `HF_HUB_OFFLINE=1`, and carries zero network configuration. Benchmark it
the same way you would outside: `boxy bench --ssh <inside-cluster>` (TTFT / ITL
/ TPOT / throughput — the vLLM bench-serve metric set).
