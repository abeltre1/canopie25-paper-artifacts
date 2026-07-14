# boxy turnkey — one command per target

The turnkey promise: **you supply a model name and a scheduler; boxy fills in
everything a job needs to be accepted** — the GPU count and engine (from the
model card), and the charge account (from `mywcid`) placed into the batch
script. No hand-written sbatch/flux directives, no `--account`, no TOML.

This runbook is the one-command story for each target, the exact output to
expect, and a prove-it checklist. Every string below is copied from a real run
(the account table is the field `mywcid` format).

---

## The field failure this fixes

> "you should have gotten the account number using `mywcid` from the HPC system
> and placed it in the batch script to make it work. I didn't see that work."

Root cause: `boxy serve … --ssh <cluster>` delegates the **whole** command to
the *cluster's* boxy, which may predate turnkey — so `mywcid` never ran and the
script carried no account. The fix resolves the account **laptop-side** and
injects it as `--account` into the delegated command (a flag every boxy version
accepts), so the cluster-built script gets the account no matter how old the
cluster's boxy is. On the login node itself, boxy runs `mywcid` directly.

---

## 0. Prove the cluster is ready (before you serve anything)

Agentless — needs **no boxy on the cluster**, just SSH:

```console
$ boxy doctor --ssh ambelt@hops
boxy doctor — remote audit of ambelt@hops (no boxy required on the cluster)

doctor
  container runtime          [OK] podman
  scheduler                  [OK] sbatch, srun
  account discovery          [OK] fy140001 (also: fy140252, fy260064) — turnkey will place
                                  `--account=fy140001` in the batch script
  accelerator (login node)   [OK] none (login nodes often have no GPU — pin --accelerator)
  image registry ghcr.io     [OK] reachable from the login node (HTTP 401) — `podman pull` should work
  cluster boxy               [OK] not installed — fine: `--ssh` resolves the account laptop-side

doctor: all checks OK
```

The `account discovery` line is the one that matters here: it runs the same
`mywcid` probe the serve path uses and shows you the account boxy *will* use. If
it WARNs (`no account parsed …`), the raw `mywcid` output is printed so you can
pass `--account <wcid>` explicitly and file the format.

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
still supplies the GPU count, engine, and port.

---

## 2. HPC over Slurm (the headline case) — from your laptop

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler slurm --ssh ambelt@hops
  auto: account: fy140001 (via mywcid on hops; also: fy140252, fy260064 — placed in the batch script)
  auto: gpus: 1 per node (packaged card 'llama-3.1-8b-instruct')
  auto: engine: vllm (packaged card 'llama-3.1-8b-instruct')
  auto: scheduler: slurm (submitting a batch job — detaches once READY)
### Batch script (…/boxy-llama-3.1-8b-instruct.sh):
    #!/bin/bash
    #SBATCH --job-name=boxy-llama-3.1-8b-instruct
    #SBATCH --nodes=1
    #SBATCH --gpus-per-node=1
    #SBATCH --account=fy140001          <-- from mywcid, no flag typed
    #SBATCH --output=…-%j.log
    …
### Submitted slurm job 1786916  (boxy-llama-3.1-8b-instruct)
### Waiting for the job to start and the server to become ready …
### READY  http://hops-gpu07:8000/v1   (model: …, slurm job 1786916)
###   tunnel: ssh -L 8000:hops-gpu07:8000 hops
###   stop:  boxy stop boxy-llama-3.1-8b-instruct
```

On a login node directly (no `--ssh`), the line reads
`auto: account: fy140001 (via mywcid; also: …)` — same result, `mywcid` just
runs locally.

**Override anytime** — an explicit flag always wins, and boxy stays silent:

```console
$ boxy serve <model> --scheduler slurm --account fy260064 --ssh ambelt@hops
```

### Get it running first — partition selection is automatic

You don't set anything. With **no `--partition` flag**, boxy reads `sinfo` and
submits to every **GPU-bearing** partition, idle-first, so Slurm starts the job
wherever a GPU frees first instead of getting stuck in one queue:

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler slurm --ssh ambelt@hops
  auto: partition: gpu,batch (via sinfo on hops: 2 partitions with GPUs, soonest-start (most idle: gpu (6 nodes)))
  auto: account: fy140001 (via mywcid on hops …)
### Batch script (…):
    #SBATCH --partition=gpu,batch     <-- CPU-only partitions excluded; starts wherever frees FIRST
```

- **Default (no flag)** — GPU partitions only, so a GPU job never parks in a
  CPU-only partition (the "stuck" failure). Slurm's own scheduler starts it in
  whichever frees soonest — native multi-partition behavior, no polling.
- **Power-user overrides** (an explicit flag always wins):
  - `--partition gpu,short` — your exact set.
  - `--partition all` — every up partition, CPU ones included.
  - `--partition off` — the scheduler's own default partition.
  - `BOXY_PARTITION=<name|auto|all|off>` — pin a fixed default.
- **Flux** takes one queue, so it picks the single best (`--queue=…`).
- Over `--ssh` everything is resolved on the cluster **before** delegating, so
  it works even against a cluster whose boxy predates this feature.

Verify what boxy would choose on a real cluster, no serving:

```console
$ boxy doctor --ssh ambelt@hops
  partitions   [OK] --partition auto → gpu,batch (soonest-start; Slurm starts in whichever frees first)
```

---

## 3. HPC over Flux — from your laptop

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler flux --ssh ambelt@eldorado
  auto: account: fy140001 (via mywcid on eldorado; …)
### Batch script (…):
    #!/bin/bash
    # flux: --job-name=boxy-llama-3.1-8b-instruct
    # flux: -N1
    # flux: -g1
    # flux: --bank=fy140001            <-- Flux spells 'account' as 'bank'
```

Flux takes **one** queue, not a Slurm comma-list. If you pass
`--partition short,batch`, boxy trims it to the first and warns:

```
warning: Flux --queue takes ONE queue; using 'short' from 'short,batch'
```

---

## 4. Cloud (SkyPilot) and OpenShift

Cloud and OpenShift don't use `mywcid`; the account concept is a cloud project /
namespace, supplied by the system card. Turnkey still hides the mechanism:

```console
$ boxy generate sky  hf://meta-llama/Llama-3.1-8B-Instruct --system cloud-aws-gpu
$ boxy generate relay <box> --system openshift-… > relay.yaml   # see DEMO-chisel.md
```

---

## Prove-it checklist

Run these against the real systems; each maps to an automated test here.

| # | Command | Expect | Automated proof |
|---|---------|--------|-----------------|
| 0 | `boxy doctor --ssh ambelt@hops` | `account discovery: OK fy140001 …` | `test_doctor.py::test_remote_checks_report_discovered_account` |
| 1 | `boxy serve <8B> --scheduler slurm --ssh ambelt@hops` | `auto: account: fy140001 (via mywcid on hops …)` then `#SBATCH --account=fy140001` | `test_turnkey_e2e.py::test_ssh_probes_mywcid_on_the_cluster` |
| 2 | on hops: `boxy serve <8B> --scheduler slurm` | `#SBATCH --account=fy140001` in the submitted script | `test_turnkey_e2e.py::test_login_node_submit_writes_account_into_the_script` |
| 2b | `boxy serve <8B> --scheduler slurm --partition auto --ssh ambelt@hops` | `#SBATCH --partition=<idle-first list>` — starts wherever frees first | `test_turnkey_e2e.py::test_ssh_resolves_partition_auto_to_concrete_list` |
| 3 | `boxy serve <8B> --scheduler flux --ssh ambelt@eldorado` | `# flux: --bank=fy140001`, one queue | `test_turnkey_e2e.py::test_login_node_flux_bank_and_single_queue` |
| 4 | `boxy list --ssh ambelt@hops` | the job as `RUNNING`, then its endpoint | submit/follow loop in `_serve_submission` |

The account resolution and its placement in the batch script are proven
end-to-end in the sandbox with fake `mywcid`/`sbatch`/`ssh` shims; steps 1–4
against the live hops/eldorado systems are the only part that must run on your
side (the sandbox has no real scheduler).

---

## Optional: update the cluster's boxy

You do **not** need this — `--ssh` injection covers an old cluster boxy. But if
you want the cluster to resolve turnkey natively (e.g. for `boxy serve` typed on
the login node with the newest cards):

```console
$ ssh ambelt@hops
$ cd ~/canopie25-paper-artifacts/boxy && git fetch && git checkout claude/boxy-turnkey && pip install -e .
```

After this, `boxy doctor --ssh ambelt@hops` reports the version instead of
`not installed`, and the `--account` injection becomes a silent no-op (the
cluster resolves the same account itself; the explicit flag still wins).
