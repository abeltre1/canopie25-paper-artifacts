# Design: agentic scaling studies on HPC (`boxy study`)

**Status: proposal. Nothing here is implemented.** This document exists to be
argued with before any code is written.

Goal: one command that takes an HPC application from nothing to a defended
statement about where it stops scaling on *this* machine.

```bash
boxy study osu-benchmarks --ssh <cluster>
```

download → build → deploy → benchmark at increasing scale → **find the knee** →
report, with boxy choosing the next scale from the last result rather than
running a grid somebody guessed in advance.

OSU is the exemplar throughout because it is already a packaged app card, it
builds in minutes, its metrics have *known analytic shapes* (so "did it stop
scaling" has a defensible answer), and nobody has to babysit a 70B download to
exercise the machinery.

---

## 1. What already exists

This is not a greenfield feature. The audit below is what a first implementation
must build on, not around.

| Capability | Where | Fit |
| --- | --- | --- |
| App cards (`kind=spack`, `spec`, `sources`, `sha256`, `nodes`, `tasks_per_node`, `setup`, `run`) | `appcards.py:55-136`, `data/cards/apps/*.toml` | **Reuse as-is.** `osu-benchmarks.toml` already pins 7.5.2 with its source URL + digest. |
| Download + build + submit + collect over one SSH master | `_app_agentless_ssh` `cli.py:1704-1991` | **Reuse.** Includes spack bootstrap across 5 setup-env locations, `spack external find`, `--reuse`. |
| Two field self-heals: blocked spack fetch → stage archive into a `file://` mirror; OpenMPI ucx `pml` failure → retry `ob1`/tcp with a perf caveat | `cli.py:1963-1982` | **Reuse.** Both fire on first contact with a real filtered cluster. |
| Per-run geometry as a parameter | `render_app_script(card, ..., nodes=N, tasks_per_node=T)` `appcards.py:235` | **Reuse.** Node count is already a function argument; nothing global blocks varying it in a loop. |
| Per-rung loop precedent | `_rung_serve_args` `cli.py:6578`, `cmd_sweep` `cli.py:6632` | **Reuse the shape**, not the code — `sweep` is LLM-serving-specific. |
| Cluster inventory ceiling | `Resources.total_nodes` / `total_gpu_nodes` `location.py:23-44` | **Reuse.** Already exists to "cap what the geometry solver may ever ask for" — exactly the cap an autonomous ladder needs. |
| Results store, plots | `results.py`, `plotting.py` | **Extend, do not contort.** See §5. |

### The five gaps

1. **`boxy app` results are unstructured log text.** `cli.py:1927` tails 200
   lines and prints them. Nothing is parsed, nothing reaches the results store.
2. **Build-once/run-many is only implicit.** Every `boxy app` invocation
   re-submits the whole script including `spack install --reuse`. A 6-rung study
   pays the spack resolve six times, inside six queue allocations.
3. **`boxy sweep` is a fixed grid with no stopping rule** (`_sweep_axis`
   `cli.py:6548`: "a comma list of ints"). It cannot decide anything.
4. **The results schema is bound to LLM serving.** `boxy-bench/1` requires
   `model` and `runs[].max_concurrency`; every plot's x-axis is
   `max_concurrency`. MPI latency-vs-nodes does not fit.
5. **No efficiency or speedup as a first-class quantity.** The single speedup
   calculation in the tree is a table-render side effect in
   `bench.ScalingReport.to_table` (`bench.py:371-375`) — not stored, not plotted,
   and relative to "whatever rung ran first".

---

## 2. What "agentic" means here — and what it does not

It means **the ladder is chosen from results, with an explicit stopping rule**,
and the reasoning is printed. It does *not* mean an LLM decides how many nodes to
ask for. Every decision below is arithmetic over measured numbers.

That distinction is the whole safety argument. A scaling study submits jobs that
consume real allocation; a policy of "the model will figure it out" is not
something to hand a scheduler. If a language model is involved later, it belongs
at the *interpretation* layer (writing the summary), not the *submission* layer.

Where a language model — or any code boxy did not write — does eventually get
involved, it runs in a sandbox, not on the login node. That is the subject of the
companion proposal on NVIDIA OpenShell, and the reason the two are separate PRs.

---

## 3. The search

### 3.1 Phases

```
  probe    smallest meaningful scale; establishes the baseline
  climb    double the scale until a stopping rule trips
  localize bisect between the last good and first bad rung
  confirm  repeat the knee and its neighbour to separate signal from noise
```

Climbing by doubling then bisecting costs O(log N) jobs to locate a knee that a
linear grid finds in O(N) — the difference between 6 jobs and 32 on a 64-node
question. Repeats happen only at the knee, where the answer is actually
contested, rather than uniformly across the ladder.

### 3.2 Stopping rules

The climb stops at the **first** of:

| Rule | Default | Why |
| --- | --- | --- |
| Efficiency below threshold | `< 0.5` vs the model in §4 | The knee is found; localize it. |
| No improvement | two consecutive rungs within measurement noise | Plateau. |
| Regression | a rung is *worse* than the previous | Past the knee. |
| Node ceiling | `Resources.total_nodes`, or `--max-nodes` | Never ask for more than exists. |
| Budget | `--max-node-hours` (**required**, no default) | See §6. |
| Failure | run fails twice at the same scale | Distinguish "does not scale" from "does not run". |

Every rung prints its verdict and the rule that fired, in the existing `auto:`
decision-line idiom. A study that stops must say which rule stopped it.

### 3.3 Worked example — OSU

```
$ boxy study osu-benchmarks --ssh <cluster> --max-node-hours 4

auto: build: one job, spack osu-micro-benchmarks@7.5.2 -> <prefix> (12m18s)
auto: ladder: osu_allreduce, doubling from 2 nodes, ceiling 64 (cluster inventory)

  nodes   ranks   allreduce(8B)   vs ideal   efficiency   verdict
      2       2         4.1 us         —           —      baseline
      4       4         6.3 us      5.4 us      0.86      climb
      8       8         8.9 us      6.8 us      0.76      climb
     16      16        14.2 us      8.2 us      0.58      climb
     32      32        31.7 us      9.5 us      0.30      STOP: efficiency < 0.5
auto: localize: bisecting 16..32
     24      24        19.1 us      8.9 us      0.47      knee
auto: confirm: repeating 16 and 24 (3x each)

### KNEE  24 nodes — allreduce departs log-scaling above 16 nodes
### LIMIT osu_bw plateaus at 23.8 GB/s from 256 KiB (link saturation)
### COST  11 jobs, 3.2 node-hours of the 4.0 budgeted
```

Illustrative numbers, not measurements.

---

## 4. "Stopped scaling" needs a model, per metric

The weakest possible version of this feature reports "it got slower". The useful
version says *slower than what*. Each metric therefore declares its expected
shape, and efficiency is measured against that shape:

| Benchmark | Expectation | Limit is |
| --- | --- | --- |
| `osu_allreduce`, `osu_alltoall` | latency grows ~`O(log N)` (allreduce) / `O(N)` (alltoall) | the node count where measured/ideal crosses the threshold |
| `osu_bw` | rises with message size, then plateaus | the plateau value and the message size that reaches it |
| `osu_latency` | flat in node count; a floor | the floor, and any node count that raises it |
| STREAM | scales linearly with nodes | departure from linear = memory-bandwidth ceiling |

This lives in the app card, so a new app is data rather than code:

```toml
[[app.metric]]
name    = "allreduce"
binary  = "collective/osu_allreduce"
parse   = "osu_two_column"     # size, value
x       = "nodes"
y       = "avg_latency_us"
model   = "log"                # log | linear | constant | plateau
lower_is_better = true
```

**Open question for review:** `model = "log"` is a coarse instrument. It cannot
tell a genuinely bad interconnect from a correct one whose constant factor is
large, and on a machine whose allreduce is implemented hierarchically the shape
is closer to piecewise-log. An alternative is to fit the curve and report the
exponent with a confidence interval, saying "allreduce scales as N^0.34" and
letting the reader judge. That is more honest and more work. **Which do you
want?** It changes the schema, so it should be decided before implementation.

---

## 5. Results: a second schema, not a contorted first

`boxy-bench/1` is frozen by a golden test and its keys are vLLM's
`--save-result` names. Bending it to hold MPI latency would mean a `model` field
that is not a model and a `max_concurrency` that is a node count. Proposal:

```
boxy-study/1
  schema, boxy_version, created, cluster, app, app_card, spec, build{prefix,hash,seconds}
  metric{name, x, y, model, lower_is_better}
  rungs[]  { x, ranks, tasks_per_node, samples[], mean, stdev, efficiency, verdict, job_id, seconds }
  knee{ x, rule, confidence }
  budget{ node_hours_used, node_hours_budgeted, jobs }
```

Same directory, same atomic write, same `boxy results list` (which already
selects by index/path/fragment). `boxy plot --kind scaling` becomes a new kind
alongside the existing four, reading `boxy-study/1` and drawing measured vs
ideal with the knee marked.

Keeping the schemas separate means the LLM-serving path cannot be broken by this
work — the golden test on `RUN_KEYS` stays exactly as it is.

---

## 6. Spending other people's allocation

An autonomous loop that submits jobs is a different risk class from every
existing boxy command. Non-negotiables:

- **`--max-node-hours` is required.** No default. The study projects each rung's
  cost before submitting and refuses the rung that would exceed the budget,
  rather than discovering it afterwards.
- **`--dryrun` prints the planned ladder and the projected cost** without
  submitting anything. Same contract as every other boxy command.
- **Never exceed cluster inventory.** `Resources.total_nodes` already exists for
  this.
- **Resumable.** A study is a series of scheduler jobs and will outlive its
  terminal (queue waits alone can exceed a working day). State lands in the
  existing per-cluster jobs directory; `boxy study --resume <name>` continues.
  A study that cannot survive a dropped SSH session is not usable on a real
  machine.
- **One study at a time per app+cluster**, or two ladders interleave in the
  queue and contaminate each other's timings.

---

## 7. Build once

Split what `boxy app` does in one script into two:

```
  build job   spack install --reuse <spec>  ->  record {prefix, hash, seconds}
  run jobs    spack load <hash>; srun -N <rung> <binary>
```

The build record keys on `(spec, cluster, compiler)` and lives beside the study
state, so a second study of the same app skips straight to running, and the
report can state *which* build produced the numbers — a scaling curve whose
compiler changed halfway through is worthless.

---

## 8. Scope of a first implementation

**In:** `boxy study <app-card> --ssh <cluster>`; the four phases; the metric
declaration in app cards; an `osu_two_column` parser; `boxy-study/1`;
`--dryrun`, `--max-node-hours`, `--resume`; `--kind scaling` plots; OSU end to
end as the acceptance test.

**Out:** any LLM in the decision path; multi-app comparison; cost models beyond
node-hours; automatic tuning of application parameters (that is a search over a
different space and deserves its own design); weak scaling (§9).

---

## 9. Open questions

1. **Fitted exponent or threshold verdict?** (§4) — schema-affecting, decide first.
2. **Strong scaling only, or weak too?** Weak scaling needs the app card to
   declare how to grow the problem with the node count, which OSU does not
   naturally express. Proposal: strong only in v1, and say so in the output
   rather than letting a reader assume.
3. **How many repeats at the knee?** 3 is a guess. On a shared machine with other
   jobs on the fabric, the variance *is* the finding — possibly worth reporting
   the distribution instead of a mean.
4. **What is the acceptance bar?** A knee number nobody can check is not a
   result. Proposal: the study is accepted when its OSU knee matches a
   hand-run ladder on the same machine on the same day, and the report carries
   enough provenance (build hash, job ids, node lists) to re-run by hand.
5. **Which cluster runs the acceptance test**, and is there a node-hour budget
   for the development loop itself?
