# boxy validation report

What was exercised this pass — every command, every deployment system, and the
network scenarios — and the result. Run from `boxy/` on `claude/boxy-turnkey`.

## Test suites

| Run | Command | Result |
|---|---|---|
| CI-equivalent (what GitHub Actions runs) | `ruff check src tests && pytest -q --ignore=tests/test_degraded_and_live.py` | **832 passed, 1 skipped, 0 failed** |
| Full incl. air-gapped/live harness | `pytest -q` (all files) | **832 passed, 8 skipped, 0 failed** |
| GitHub Actions `boxy-ci` (py 3.11 / 3.12 / 3.13 + package) | on every push to the branch | **green** (latest: `2fb3393`) |

The 8 skips: the no-ramalama degradation harness needs ramalama *truly absent*
(it's pip-installed here, so PYTHONPATH isolation can't hide it — CI excludes the
file; the harness now SKIPS with a reason instead of failing), and the
live-Docker end-to-end skips because this sandbox's Docker daemon isn't running.

## Commands (`boxy <cmd> --help`, all 22)

`info config examples cards doctor run pull build generate bench logs curl open
sweep launch stop list unshare router stage alloc serve` — **all exit 0.**

## Deployment systems (dryrun / golden)

| System | How exercised | Result |
|---|---|---|
| Laptop (podman/docker/apptainer) | golden dryrun tests; live serve needs a running runtime (absent here) | covered by suite |
| Slurm + CUDA (`--system cuda-cluster`, `--scheduler slurm`) | serve dryrun + agentless + delegated e2e (fake cluster) | ✅ |
| Slurm multinode / distributed / replicas | serve dryrun | ✅ |
| Flux + ROCm (`--system rocm-cluster`, `--scheduler flux`) | serve dryrun; `--bank` + single-queue guard | ✅ |
| CharlieCloud (`--system charliecloud-cuda`) | serve dryrun | ✅ |
| Cloud AWS/GCP (`generate sky`) | SkyPilot YAML emit | ✅ |
| OpenShift (relay / flux-mcp) | `generate relay`, `generate flux-mcp` manifests | ✅ |

`--system aws-gpu|gcp-gpu|gpu-cluster` serve *locally* correctly errors "no
working container runtime" in this sandbox (no daemon) — expected; those targets
are driven via `generate sky` / manifests, and the suite covers them with fake
runtimes.

## Networks

| Scenario | Exercised | Result |
|---|---|---|
| Corporate proxy into the job (`--proxy`) | serve dryrun carries proxy env | ✅ |
| Registry mirror (`--registry`) | serve dryrun rewrites the image | ✅ |
| Site CA propagation over `--ssh` | `test_last_mile` / `test_remote` | ✅ (suite) |
| Blocked image pull (Docker Hub / ghcr 403, Zscaler) | diagnosis fires on the agentless job-died path | ✅ |
| DNS / 403 (network block vs bad HF token) | `test_flux_and_diagnostics` | ✅ (suite) |
| GPU-request rejection (clusterd GRES) | auto-recover retry, both submit paths | ✅ |

## Turnkey commands added this session

`boxy serve` interactive ACCOUNT_ID picker; `boxy generate card <hf-id>`; GRES
self-heal; blocked-pull diagnosis — each with unit + e2e coverage in the suite
above.

## Not runnable in this environment (needs your systems)

- Live serve against a real GPU + running container runtime (no GPU / no daemon here).
- The HuggingFace Hub path for `generate card` (this sandbox's egress proxy blocks
  huggingface.co) — tested against captured config fixtures; run the live
  `--dry-run` where the Hub is reachable.
- Real HPC submission on clustera/clusterb/clusterc/clusterd — exercised via fake-cluster
  e2e; the live runbook is `DEMO-turnkey.md`.
