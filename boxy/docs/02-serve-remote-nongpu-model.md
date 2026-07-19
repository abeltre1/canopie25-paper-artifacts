# 02 — Serve a non-GPU model (laptop, and remotely on a CPU partition)

*The no-accelerator path: run a small model right where you are, or submit it
to a cluster's CPU partition over `--ssh`. Companion to
[01 — serve a GPU model on a cluster](01-serve-gpu-model.md); share the result
with [03 — chisel](03-share-with-chisel.md).*

---

## 1. Laptop / workstation (Podman or Docker)

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct
  auto: gpus: 1 per node (packaged card 'llama-3.1-8b-instruct')
  auto: engine: vllm (packaged card 'llama-3.1-8b-instruct')
  auto: accelerator: cuda (autodetected)
### READY  http://127.0.0.1:8000/v1   (model: meta-llama/Llama-3.1-8B-Instruct)
```

No scheduler, no account — a laptop just runs the container. The model card
still supplies the GPU count, engine, and port. On a machine with no GPU the
accelerator resolves to `none` and the engine runs on CPU — fine for small
models; a 70B is not going to be pleasant.

---

## 2. Remote CPU serve over `--ssh` — `--gpus 0`

A small model doesn't need a GPU allocation (or the GPU queue's wait time).
Pass `--gpus 0` to mark the job **explicitly CPU** and point it at a CPU
partition:

```console
$ boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct --ssh user1@clustera --gpus 0 --partition short
  auto: scheduler: slurm (via detected (Slurm is live — sinfo listed partitions) on clustera)
  auto: account: ab110001 (via myaccounts on clustera — placed in the batch script)
  auto: time: 30:00 (via config site.default_time)
### Batch script (…/boxy-qwen2.5-0.5b-instruct.sh):
    #SBATCH --partition=short           <-- your CPU partition; no GPU directive rendered
    #SBATCH --account=ab110001
    #SBATCH --time=30:00
### Submitted slurm job …
### READY  http://…:8000/v1   (tunnel: printed on the next line)
```

Everything from the GPU walkthrough still applies — agentless rendering,
account/partition/walltime auto-resolution, the SSH tunnel, readiness via
`localhost/health` — the only difference is that no GPU directive is placed in
the batch script and the accelerator checks are skipped.

---

## 3. The pre-deploy accelerator hold (what happens WITHOUT `--gpus 0`)

boxy checks the partition's GPU inventory **before** submitting (one composite
probe over the same SSH session — no allocation is burned). If you point a
GPU-needing job at a partition whose `sinfo -p <part> -h -o %G` shows no GPU
GRES anywhere, boxy **holds the deployment** instead of letting it be rejected
or queue forever:

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh user1@clustera --partition short
boxy: partition 'short' on clustera advertises no GPUs (sinfo -p short -h -o %G shows
no gpu GRES) — a GPU job there would be rejected or queue forever. Pick a GPU
partition (sinfo -s on clustera), drop --partition, or pass an explicit
--accelerator (or --gpus 0 for a CPU run) to override this check.
```

Your three ways out, exactly as the message says:

| You want | Do |
|---|---|
| the model on a GPU after all | drop `--partition` (auto picks GPU partitions, soonest-start) or pick one from `sinfo -s` |
| a CPU run on that partition | add `--gpus 0` (this doc's scenario — the hold does not apply to explicit CPU jobs) |
| to overrule the probe (e.g. the site's GRES labels are wrong) | pass `--accelerator cuda|rocm|…` explicitly |

The hold only fires when the **cluster-wide** inventory corroborates that the
partition really is GPU-less — a site that doesn't label GRES at all is never
blocked on the guess.

---

## 4. Where the accelerator verdict comes from

Over `--ssh`, everything rides the one-shot cluster probe; the `auto:` lines
tell you the provenance:

```
  auto: accelerator: none (via cluster probe: partition short's GPU inventory)
```

Locally, detection survives HPC login nodes where vendor tools are behind
`module load` or an allocation: boxy tries a functional probe (loading the
module if needed) before falling back to filesystem markers, and `boxy doctor`
shows the detection note when the ladder had to work for its answer.

---

## Prove-it checklist

| # | Command | Expect | Automated proof |
|---|---------|--------|-----------------|
| 1 | `boxy serve <small model>` (laptop, no GPU) | serves on CPU, `### READY http://127.0.0.1:…` | `test_capability_matrix.py` serve goldens |
| 2 | `boxy serve <small model> --ssh <host> --gpus 0 --partition <cpu part>` | batch script with the partition + account, **no** GPU directive | `test_turnkey_e2e.py::test_ssh_partition_scoped_accel_probe` |
| 3 | `boxy serve <8B> --ssh <host> --partition <cpu part>` (no `--gpus 0`) | the hold message above, exit code 2, nothing submitted | `test_turnkey_e2e.py::test_ssh_partition_without_gpus_holds_deployment` |
| 4 | `boxy doctor --ssh <host>` | `accelerator` line with the probe's verdict + note | `test_doctor.py` |
