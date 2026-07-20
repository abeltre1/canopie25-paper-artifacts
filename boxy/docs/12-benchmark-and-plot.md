# 12 — Benchmark a served model, then plot it (end to end)

*Measure ANY model boxy serves — with the official vLLM load generator when
available, the built-in synthetic one when not — persist every result, and
turn results into figures (or the paper's gnuplot pipeline) with one command.
Serve something first: [01](01-serve-gpu-model.md) / [02](02-serve-remote-nongpu-model.md).*

---

## 0. The one-command version

```console
$ boxy bench --plot
  auto: bench backend: vllm-bench (vllm-bench static binary at ~/.local/share/boxy/store/bin/vllm-bench)
  auto: dataset: random (no download needed (--dataset sharegpt for the paper's corpus))
### Benchmarking meta-llama/Llama-3.1-8B-Instruct at http://node07:8000 — 11 concurrency levels
###   concurrency 1: 76.5 tok/s (32/32 ok)
###   concurrency 2: 145.2 tok/s (32/32 ok)
    …
### Result saved: ~/.local/share/boxy/results/clustera/20260720-141002-llama31.bench.json
### Plot: ~/.local/share/boxy/results/clustera/20260720-141002-llama31.png
```

No flags: the newest live instance's endpoint (from `boxy list`), the default
1→1024 concurrency sweep, the best available load generator, auto-persisted,
plotted. Every choice is printed and overridable.

---

## 1. Bench backends: real vs synthetic

| Backend | What it is | Needs |
|---|---|---|
| `vllm-bench` | the official standalone Rust load generator (vllm-project/vllm-bench), drop-in `vllm bench serve` replacement | one static binary — `boxy bench --fetch-backend` |
| `vllm-container` | `vllm bench serve` (or the bundled `benchmark_serving.py`) run **inside the serving image** — the paper's own method | the image already pulled + podman/docker |
| `vllm-cli` | a locally installed `vllm` CLI | `pip install vllm` (heavy) |
| `synthetic` | boxy's stdlib streaming load generator (same TTFT/TPOT/ITL/E2E metric set) | nothing — works air-gapped |

`--backend auto` (the default) walks that ladder top-down and tells you what
it picked:

```
  auto: bench backend: synthetic (built-in load generator — for the official
        vLLM benchmark run `boxy bench --fetch-backend` once)
```

`boxy doctor` has a `bench backend` line showing what a bench would use here.

One-time setup for the real thing (any machine with outbound HTTPS or a
mirror):

```console
$ boxy bench --fetch-backend
### vllm-bench installed: ~/.local/share/boxy/store/bin/vllm-bench  (auto backend now prefers it)
```

Air-gapped sites: point `urls.vllm_bench` at an internal mirror, drop the
binary at `<store>/bin/vllm-bench` by hand, or carry it with
`boxy bundle MODEL --bench`.

---

## 2. Datasets

- **`random`** (real-backend default) — synthetic token streams, no download,
  reproducible via `--seed` (default 12345, the paper's).
- **`--dataset sharegpt`** — the ShareGPT corpus the paper benched with;
  downloaded once through your proxy/CA config and cached under
  `~/.local/share/boxy/datasets/`. Air-gapped: pre-stage the JSON there.
- **`--dataset path/to/prompts.json`** — your own JSON list of prompts or a
  ShareGPT-format file.

---

## 3. Benchmarks from anywhere

**An agentless-served model** (the `--ssh` serve default) — the bench is
agentless too. boxy finds the serve record on your laptop, runs the benchmark
**inside the serving image on the login node** over the same SSH session, and
stores the parsed result laptop-side. No boxy, binary, or anything else on the
cluster:

```console
$ boxy bench                      # or: boxy bench --ssh user1@clustera / boxy bench NAME
  auto: bench backend: vllm-container (benchmark inside the serving image
        docker.io/rocm/vllm:… via podman on clustera — agentless)
###   concurrency 1: 76.5 tok/s (32/32 ok)
    …
### Result saved: ~/.local/share/boxy/results/…   (list: boxy results; plot: boxy plot)
```

(Models served by an older boxy have no image in their record — pass
`--image <the serving image>` once; new serves record it automatically.)

**Install the vllm-bench binary ON a cluster** — also agentless (a curl over
the SSH master; never delegated to the cluster's own boxy):

```console
$ boxy bench --fetch-backend --ssh user1@clustera
```

**A delegated-serve instance** (`--delegate`): `boxy bench --ssh user1@clustera`
runs the bench via the cluster's boxy — keep that checkout current.

**A secured k8s/OpenShift ingress** (vLLM behind `--api-key` — this scripts
what the paper did by hand for its OpenShift columns):

```console
$ boxy bench --url https://vllm-bench.apps.ocp.example.gov --api-key "$VLLM_API_KEY" \
             --label "ocpcluster/h100x2"
```

The token rides the request header / child-process env only — it is never
written into argv, logs, or the saved result.

**Quick sanity pass** (3 levels, small budget):

```console
$ boxy bench --max-concurrency 1,8,64 --num-prompts 64 --max-tokens 32
```

Levels that crash the server record as failed **and the sweep continues** —
the surviving levels still save and plot (the 405B lesson: a 1024-concurrency
crash must not discard the 1..512 measurements).

**Scaling past 256 concurrency.** Two ceilings flatten the top of a sweep:

- *Prompt pool*: boxy sizes it automatically (~10× the level, and always at
  least 3× at the top rungs — 3072 prompts at concurrency 1024), so the
  requested concurrency can genuinely be held in flight. Expect the 512/1024
  levels to take proportionally longer; `--num-prompts` overrides.
- *The server's scheduler*: vLLM only batches `--max-num-seqs` requests at a
  time (256 by default on V0 engines) and **queues the rest** — the plot goes
  flat past that value no matter what the client sends. To measure real
  1024-way scaling, raise it at serve time:

```console
$ boxy serve meta-llama/Llama-3.1-8B-Instruct --ssh clustera -- --max-num-seqs 1024
```

---

## 4. Results persist — `boxy results`

Every bench lands in a per-cluster store (`paths.results_root`, one JSON per
run: provenance envelope + per-level metrics under vLLM `--save-result` key
names, identical for real and synthetic backends):

```console
$ boxy results
  #              created      backend levels peak tok/s  model / label
  1  2026-07-20T14:10:02Z   vllm-bench     11     1059.3  clustera/llama31
  2  2026-07-19T09:02:41Z    synthetic     11      998.1  clusterb/llama31
$ boxy results show 1        # re-print any stored run's table
$ boxy results path          # where the store lives
```

`--no-save` skips persistence; `-o results.csv` still writes the plot-ready
CSV wherever you point it.

---

## 5. Plot — `boxy plot`

```console
$ boxy plot                        # newest result → throughput-vs-concurrency PNG
$ boxy plot 1 2                    # OVERLAY two runs — compare clusters/configs
$ boxy plot --kind latency --metric ttft --stat p99
$ boxy plot --kind frontier        # latency vs throughput, one point per level
$ boxy plot --kind cache           # prefix-cache hit rate (%) per level
$ boxy plot --kind all -o ./figs/  # every figure the results support
```

- The default figure is the paper's: **output tokens/s vs max request
  concurrency, log₂ x-axis**; crashed levels appear as gaps.
- Overlaying N results reproduces the paper's multi-column comparison
  (ClusterA vs ClusterB vs OpenShift) without hand-editing a `.dat`.
- Legends read `<gpu model or accel family>: cluster/name` (mi300a, h100, …)
  — the GPU model comes from the serve record, the node's GRES/Features, or
  the cluster's system card; pin `gpu_type = "mi300a"` in
  `~/.config/boxy/cards/systems/<cluster>.toml` on sites whose scheduler
  doesn't label GPU types.
- **Cache-hit figures need two things**: the server must export prefix-cache
  metrics (`boxy serve ... -- --enable-prefix-caching`; vLLM V1 engines have
  it on by default), and the workload must actually reuse prefixes — bench
  with `--dataset sharegpt`; the `random` dataset shows ~0% by design.
  Rates are sampled from vLLM's `/metrics` around each level, so no extra
  flags at bench time.
- Rendering needs matplotlib: `pip install 'boxy-hpc[plot]'`.
- **No matplotlib? No problem** — emit the paper's exact gnuplot pipeline:

```console
$ boxy plot 1 2 --emit gnuplot -o ./figs/
### Plot: figs/results.dat          # X marks crashed cells, like plots/*/results.dat
### Plot: figs/compare-2-throughput.gp
### Plot: figs/run-gnuplot.sh
```

---

## 6. Scaling sweeps

`boxy sweep` (submit → ready → bench → tear down, one rung per node/replica
count) now persists each rung to the store and can plot the overlay:

```console
$ boxy sweep hf://meta-llama/Llama-3.1-8B-Instruct --scheduler slurm \
      --sweep-replicas 1,2,4 --plot
```

Replica-fleet rungs use the synthetic backend deliberately: it pools raw
latencies across every endpoint, so p50/p95 are true fleet percentiles rather
than one replica's view.

---

## 7. Reproducibility + provenance checklist

- `--seed` (default 12345) fixes the request mix for real backends.
- The envelope records model, endpoint, cluster, backend + version, dataset,
  seed, geometry (nodes/gpus/replicas), and the boxy version — enough to
  reproduce any figure's provenance months later.
- Bench traffic **never** rides the corporate proxy (compute nodes and
  localhost tunnels are direct; subprocess backends get the target host
  appended to `no_proxy` — the same fix the paper's script carried).

## Prove-it checklist

| # | Command | Expect | Automated proof |
|---|---------|--------|-----------------|
| 1 | `boxy bench --dryrun` | backend + dataset + save-path plan | `test_bench_backends.py::test_cli_bench_dryrun_names_backend` |
| 2 | `boxy bench` (served instance) | table + `### Result saved:` | `test_results_store.py::test_bench_persists_by_default_and_no_save_skips` |
| 3 | `boxy bench --backend vllm-bench` | official flags incl. `--seed 12345`, canonical JSON stored | `test_bench_backends.py::test_cli_bench_real_backend_with_record_model` |
| 4 | `boxy results` / `show 1` | listing + stored table | `test_results_store.py::test_results_list_show_path` |
| 5 | `boxy plot 1 2` | overlay PNG, log₂ x | `test_plot.py::test_cli_plot_overlay_compare` |
| 6 | `boxy plot --emit gnuplot` | paper-style dat/gp/runner | `test_plot.py::test_cli_plot_emit_gnuplot` |
| 7 | `boxy doctor` | `bench backend` + `plotting` lines | `test_doctor.py` |
