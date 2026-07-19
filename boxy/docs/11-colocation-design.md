# Colocation / bin-packing of services — DESIGN DRAFT (not implemented)

STATUS: shelled only. `boxy app --colocate NAME` / `--nodelist NODE` are parsed
and refuse with a pointer here. Nothing submits. This document is the design
space to evaluate before committing to the feature.

## The ask

Service A runs on cluster A (Slurm put it on node56). Launch service B onto the
SAME node — colocated — as long as it does not interfere with A's resources
(ports, CPUs, memory, GPUs):

    boxy app --image img-b --port 9090:80 --ssh clusterA --colocate app-service-a
    boxy app --image img-b --port 9090:80 --ssh clusterA --nodelist node56

## Mechanisms (preference order)

1. **Step injection into A's allocation** — works on EXCLUSIVE-node partitions
   (most HPC sites): `srun --jobid=<A's job> --overlap [--exact --cpus-per-task=N
   --mem=…] podman run …`, started detached (setsid/nohup) on the login node over
   the ssh master so it survives the session. A's job id/target come from A's
   laptop record; A's node from A's endpoint JSON. B writes its own endpoint +
   record (recording the step/pid so `boxy stop app-B` kills only the step).
   CAVEAT to surface loudly: B dies when A's job ends (walltime, boxy stop A).
2. **Second pinned job** — for shareable partitions: B's normal service script
   plus `#SBATCH -w <node> --oversubscribe`; on rejection ("exclusive",
   "oversubscribe not permitted") fall back to mechanism 1 with a clear message,
   following the existing self-heal resubmit pattern (GRES/license retries in
   _app_agentless_ssh / _serve_agentless_ssh).

## Non-interference guardrails (laptop-side, before submission)

- **Ports**: every record on that cluster whose endpoint host == target node;
  refuse a host-port collision naming the conflicting service.
- **CPU/memory**: budget = system card (cpus_per_node / mem_gb_per_node) minus
  what sibling records declare; services grow `--cpus N --mem-gb N` (small
  defaults) recorded per instance so the arithmetic is possible.
- **GPUs**: services default to no GPU; a GPU service colocating with a serve is
  refused unless disjoint GPUs can be pinned (CUDA_VISIBLE_DEVICES per step).
- Every decision prints as an `auto:` line (the every-choice-is-printed contract).

## Open questions to settle before implementing

- Is step injection acceptable operationally (B's lifetime bound to A's job)?
  If A is long-lived (8h walltime services) this may be fine; if A is a model
  serve with a 1h default walltime, B dies with it.
- Do the target partitions allow OverSubscribe at all? (`scontrol show
  partition` — if yes, mechanism 2 is much cleaner: independent lifetimes.)
- Should packing instead happen at SUBMIT time — one job hosting N services
  (the `--replicas` group_batch_script precedent) — trading "add later" for
  independent-lifetime simplicity?

## Acceptance (when/if built)

- `--colocate`/`--nodelist` dryrun shows the srun-step (or -w) plan and the
  port/CPU/mem arithmetic; nothing submitted.
- Port collision and exclusive-partition cases produce actionable errors/fallbacks.
- `boxy list` shows both services on one node; `boxy stop app-B` kills only B.
- Finite apps (benchmarks) and existing service semantics untouched.
