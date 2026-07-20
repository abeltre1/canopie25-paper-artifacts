# boxy documentation

Numbered in the order you'd test boxy end to end; reference material follows.

| # | Doc | What it walks through |
|---|-----|----------------------|
| 01 | [Serve a GPU model on an HPC cluster](01-serve-gpu-model.md) | the headline: laptop → Slurm/Flux GPU serve, agentless over `--ssh`, auto account/partition/walltime, readiness, troubleshooting |
| 02 | [Serve a non-GPU model](02-serve-remote-nongpu-model.md) | laptop CPU serve; remote CPU-partition serve with `--gpus 0`; the pre-deploy no-GPU hold |
| 03 | [Share with your team (chisel)](03-share-with-chisel.md) | zero-install everyone-URL through the OpenShift chisel relay, end to end |
| 04 | [Air-gapped deployments](04-airgap.md) | `boxy bundle` / `serve --bundle`: what crosses the gap and the checklist nobody remembers |
| 05 | [Nemotron-3 cookbook](05-nemotron3-cookbook.md) | the Nemotron family (Nano/Super/Ultra) as zero-flag commands across a mixed fleet |
| 06 | [Field runbook](06-runbook.md) | the exhaustive ops reference: every deployment mode, proxy/registry/TLS, doctor, per-symptom troubleshooting |
| 07 | [Validation report](07-validation.md) | which suites, commands, systems, and network scenarios were exercised, and their results |
| 08 | [Releasing](08-releasing.md) | cutting a `boxy-v*` release: GitHub Release, GHCR containers, PyPI Trusted Publishing |
| 09 | [Architecture](09-architecture.md) | the layered design: cards → resolver → scheduler/runtime ABCs, with diagrams |
| 10 | [Specification](10-spec.md) | the full design spec: goals, landscape, reuse map, CLI surface, roadmap (§8), known issues (§8b), agentless design (§8c) |
| 11 | [Colocation (design draft)](11-colocation-design.md) | NOT implemented — bin-packing services onto shared nodes; open questions |
| 12 | [Benchmark + plot](12-benchmark-and-plot.md) | real vLLM benchmarking of any served model (vllm-bench/container/CLI ladder), the results store, and `boxy plot` figures + gnuplot pipeline |

Start at [01](01-serve-gpu-model.md). The package front door is
[../README.md](../README.md).
