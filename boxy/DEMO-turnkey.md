# boxy turnkey — one command per target

The turnkey promise: **you supply a model name (and, over `--ssh`, a host);
boxy fills in everything a job needs to be accepted** — the scheduler
(auto-detected by which control plane is LIVE on the cluster — robust when a Flux system ships slurm shims), the GPU count and engine (from
the model card), the charge account (from `mywcid`), a soonest-start partition
(from `sinfo`), and a 30-minute default walltime — all placed into the batch
script. No `--scheduler`, no hand-written sbatch/flux directives, no `--account`,
no `--partition`, no TOML. Every abstraction has a power-user flag that wins.

This runbook is the one-command story for each target, the exact output to
expect, and a prove-it checklist. Every string below is copied from a real run
(the account table is the field `mywcid` format).

---

## Zero install on the HPC — fully agentless over `--ssh` (default)

> "I've never installed it on the cluster. We should not require installing it on
> the HPC system."

`boxy serve <model> --ssh <cluster>` installs **nothing** on the HPC — no boxy,
no Python, no RamaLama. Your laptop does everything over the one SSH session:

1. detects the live scheduler + resolves the site (`mywcid` account, `sinfo`
   partition, walltime) over SSH,
2. renders a **self-contained `podman run` batch script** — the engine pulls the
   model at container start (`vllm serve meta-llama/Llama-3.1-8B-Instruct`), so no
   RamaLama on the cluster,
3. writes + `sbatch`/`flux batch`-submits it over the same SSH master,
4. polls the shared-FS endpoint file and confirms readiness via
   `localhost/health` **through the tunnel**, then prints `### READY → ### LOCAL →
   ### SHARE`.

The compute node runs only `podman` + a two-line endpoint write. Nothing to
install, ever. Use `--delegate` (or `BOXY_SSH_DELEGATE=1`) to run the cluster's
own boxy instead — needed for `--replicas` / `--distributed` / `--box`, which the
agentless path doesn't cover yet. A pre-staged shared-FS model path (or `--image`)
is served as-is; an `s3://` model still needs staging first.

---

## The earlier field failure this also fixes

> "you should have gotten the account number using `mywcid` from the HPC system
> and placed it in the batch script to make it work. I didn't see that work."

Root cause: `boxy serve … --ssh <cluster>` used to delegate the **whole** command
to the *cluster's* boxy, which may predate turnkey — so `mywcid` never ran and the
script carried no account. Now the script is rendered **laptop-side** (agentless)
with the account/partition/time already resolved over SSH, so it can't be missing.
On the `--delegate` path, the account is still injected as `--account` so even an
old cluster boxy gets it.

---

## 0. Prove the cluster is ready (before you serve anything)

Agentless — needs **no boxy on the cluster**, just SSH:

```console
$ boxy doctor --ssh ambelt@hops
boxy doctor — remote audit of ambelt@hops (no boxy required on the cluster)

doctor
  container runtime          [OK] podman
  scheduler                  [OK] will submit via: slurm — detected (Slurm is live — sinfo listed partitions)
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

You don't even type `--scheduler`: boxy probes the cluster and finds `sbatch`,
so the whole command is just the model and the host.

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops
  auto: scheduler: slurm (via detected (Slurm is live — sinfo listed partitions) on hops)
  auto: partition: gpu,batch (via sinfo on hops: 2 partitions with GPUs, soonest-start)
  auto: account: fy140001 (via mywcid on hops; also: fy140252, fy260064 — placed in the batch script)
  auto: time: 30:00 (via config site.default_time — the scheduler stops the job at this walltime)
  auto: gpus: 1 per node (packaged card 'llama-3.1-8b-instruct')
  auto: engine: vllm (packaged card 'llama-3.1-8b-instruct')
### Batch script (…/boxy-llama-3.1-8b-instruct.sh):
    #!/bin/bash
    #SBATCH --job-name=boxy-llama-3.1-8b-instruct
    #SBATCH --nodes=1
    #SBATCH --gpus-per-node=1           <-- proven default; auto-recovers to --gres if rejected
    #SBATCH --partition=gpu,batch       <-- from sinfo, soonest-start, no flag typed
    #SBATCH --account=fy140001          <-- from mywcid, no flag typed
    #SBATCH --time=30:00                <-- 30-min default, no flag typed
    #SBATCH --output=…-%j.log
    …
### Submitted slurm job 1786916  (boxy-llama-3.1-8b-instruct)
### Waiting for the job to start and the server to become ready …
###   [0:14] job 1786916: RUNNING
###   [1:20] PULLING CONTAINER IMAGE  ›  Copying blob sha256:… 
###   [3:05] LOADING WEIGHTS  [########------]  57%
###   [4:40] CAPTURING CUDA GRAPHS  [############--]  86%
###   [5:02] SERVER STARTING  ›  INFO: Application startup complete.
### READY  http://hops-gpu07:8000/v1   (model: …, slurm job 1786916)
###   tunnel: ssh -L 8000:hops-gpu07:8000 hops
###   stop:  boxy stop boxy-llama-3.1-8b-instruct
```

> **The GPU request self-heals — without changing a working cluster.** boxy uses
> the proven `--gpus-per-node=N` by default (what works on hops/eldorado). It does
> **not** pre-emptively rewrite that on a cluster that's already fine. But some
> sites reject it (`sbatch: error: Invalid generic resource (gres) specification`,
> field report: kahuna) — so if a submit is rejected for the GPU line, boxy
> **re-renders with the portable `--gres=gpu:[type:]N` form (the type probed from
> `sinfo`) and resubmits by itself**, cycling typed → untyped → `--gpus` until one
> is accepted:
>
> ```
> boxy: the site rejected the GPU request; retrying with --gres=gpu:N ...
> ### GPU request accepted as --gres=gpu:N (auto-recovered).
> ```
>
> You do nothing. To pin a form and skip the self-heal, `export
> BOXY_GPU_DIRECTIVE=gres|gpus|gpus-per-node|none` (and optionally `BOXY_GPU_TYPE=a100`).

> **Choosing your WCID (charge account).** When `mywcid` lists several accounts
> and you didn't pass `--account`, boxy shows an inline menu (on a terminal) so you
> pick which one the job charges to — instead of silently taking the first:
>
> ```console
> $ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@hops
>   ...
> Select a charge account (WCID):
>   1) fy140001  system software and tools
>   2) fy140252  common computing environment  [default]
>   3) fy260064  the genesis project
> account [1-3] (Enter = 2): 3
>   auto: account: fy260064 (you picked 3 of 3 from mywcid on hops)
> ```
>
> Your pick is remembered per cluster (hit **Enter** next time to reuse it, shown
> as `[default]`), and validated against the live `mywcid` list so a stale default
> can't charge an account you've lost. Bypass the menu entirely with `--account
> fy…`, `export WCID=fy…` (handy in scripts/CI), or a pinned `BOXY_ACCOUNT`. The
> menu never blocks a non-terminal run (batch/CI): with no TTY it auto-picks the
> first/remembered account and prints how to choose. Force or disable it with
> `--pick-account` / `--no-pick-account`, or `export BOXY_PICK_ACCOUNT=always|never`.

While the model loads, boxy prints a **live progress line** every ~10 s — an
elapsed clock, the current phase (QUEUED → STARTING → PULLING IMAGE → LOADING
WEIGHTS → CAPTURING CUDA GRAPHS → SERVER STARTING → READY), and a bar parsed from
the engine/container log — so a multi-minute load reads as forward motion, not a
silent spinner. boxy waits up to ~20 min for readiness (it prints READY the
instant the server answers, so it never over-waits); raise it with
`--ready-timeout <sec>` / `BOXY_READY_TIMEOUT`, or `--ready-timeout 0` to submit
and detach immediately. Over `--ssh` the wait is raised on the delegated command
too, so an **older cluster boxy doesn't give up at 180 s** on a still-loading model.

**Readiness is checked via `/health`, and the LAPTOP owns it** so an old cluster
boxy can't stall you. The moment the cluster boxy names the compute-node endpoint
(`server starting on cronus5 … http://cronus5:8000/…`), your laptop opens the SSH
tunnel and confirms the server itself by polling **`http://localhost:<localport>/health`
through that tunnel** — the canonical, **unauthenticated** readiness endpoint for
both vLLM and llama.cpp (200 the instant it can serve, no API key even when the
model API is gated), checked as `localhost` from your laptop where the forwarded
port reaches the serving node. It does **not** depend on the cluster boxy's own
probe (which the field report showed looping "still waiting" forever because its
`http://cronus5:8000/v1/models` GET went through the corporate proxy). Fallbacks:
the compute node also flips a `ready` flag on the shared-FS endpoint file after
its own localhost `/health`, and boxy reads the engine's "server is up" line from
the job log (`Application startup complete.` / `server is listening`). The instant
readiness fires, the sequence is exactly:

```
### READY   http://127.0.0.1:<localport>/v1   (confirmed via localhost/health through the tunnel)
### LOCAL   http://127.0.0.1:<localport>/v1   (tunnel over the SSH session)
### SHARE   https://<name>-boxy.apps.<cluster>/v1   (chisel — team URL)
```

and the "still waiting" spam stops. This works **without updating the cluster's
boxy** — it's all laptop-side.

On a login node directly (no `--ssh`), `mywcid`/`sinfo` run locally. The
scheduler is **not** auto-probed there (bare `boxy serve MODEL` keeps the
login-node guard, which prints a clear "add `--scheduler` / `--here`" message so
an LLM is never served on a shared login node by accident) — set
`export BOXY_SCHEDULER=slurm` once, or pass `--scheduler slurm`, to submit
without the flag from the login node.

> **Walltime caps the serve.** The 30-minute default is a *hard stop*: the
> scheduler kills the served job at `--time`. For a longer session pass
> `--time 4:00:00` (Slurm notation; boxy converts it to Flux FSD) or set
> `export BOXY_DEFAULT_TIME=4:00:00`. Set `BOXY_DEFAULT_TIME=` (empty) to fall
> back to the scheduler's own default instead.

**How the scheduler is detected (robust across a mixed fleet).** boxy does **not**
guess from which binaries exist — a Flux system often ships Slurm-compat
`sbatch`/`sinfo`/`scontrol` shims that *proxy to Flux*, so "binary present" (even
"`sinfo` answers") is a lie. It probes which control plane is **live** over `--ssh`
and applies one rule:

- **A reachable SYSTEM Flux broker wins.** Flux runs the machine; any slurm
  commands that also answer are compat shims — submitting through them returns
  *Flux* job ids (`f2c5JAAU8BR1`) that `squeue` can't track. So boxy picks
  **flux** and submits via `flux batch` / tracks via `flux jobs`. boxy reaches the
  system instance via its well-known socket (`local:///run/flux/local`) **even
  when a bare ssh has no `FLUX_URI`** (no profile sourced), and uses
  `flux getattr instance-level` to tell the **system** instance (level 0,
  authoritative) from a personal **nested** one (level ≥1, e.g. a `flux alloc`
  under Slurm — *not* authoritative).
- **Otherwise Slurm** — a real `slurmctld` (`scontrol ping` → "is UP") or `sinfo`
  partitions. A real slurmctld outranks a merely nested flux instance, so a Slurm
  cluster where you happen to have a personal flux running is still detected as
  Slurm.

So `eldorado` (a Flux system whose slurm shims answer too) is correctly detected
as **flux**. The decision line says why:
`auto: scheduler: flux (via detected (Flux broker is live — Flux runs this machine;
slurm commands also answered but on a Flux system those are compat shims that proxy
to Flux. Pass --scheduler slurm / set BOXY_SCHEDULER=slurm if this cluster's primary
really is Slurm))`. Verify any cluster first with `boxy doctor --ssh <host>` — its
`scheduler` line reports the same `will submit via: …` pick.

**Pin it** to skip the probe or for the rare cluster whose primary boxy guessed
wrong: `--scheduler slurm|flux`, or `export BOXY_SCHEDULER=slurm|flux`.
`--scheduler none` / `BOXY_SCHEDULER=none` keeps it a direct serve. `--here`
serves directly on the node you're on (no submission).

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

### Run it again — you get a second instance, not a wall

Re-running the same command while one is already live doesn't block; boxy starts
an **independent** instance (its own job / log / endpoint). You never type
`--unique`:

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler slurm --ssh ambelt@hops
  auto: name: boxy-llama-3.1-8b-instruct-0714-203155-9f2c (boxy-llama-3.1-8b-instruct is already
        slurm job 1786916 / RUNNING — starting an independent instance; stop either with `boxy stop <name>`)
```

- A service that's **already up and ready** still just reports its URL (no
  duplicate) — only a *pending/starting* instance triggers a fork.
- **Power users** keep full control: pass `--unique` to force a fresh instance
  every time; `--no-auto-unique` / `BOXY_AUTO_UNIQUE=false` restores the strict
  refuse-if-one-exists behavior.
- Over `--ssh` the decision is made **laptop-side** (boxy checks `squeue`/`flux
  jobs` on the cluster for a live job of that name), so `--unique` is injected
  into the delegated command and this works **even against an older cluster
  boxy** — no cluster update required.

### The proxy comes along automatically

If your shell exports `http(s)_proxy` (or you set `BOXY_PROXY=http://proxy.<org>:80`
once), boxy carries it into the job's image/model pulls and, over `--ssh`,
forwards it to the cluster — you'll see `### Proxy forwarding http://proxy.<org>:80`.
No `--proxy` flag needed; pass one only to override.

---

## 3. HPC over Flux — from your laptop

On a Flux-only cluster you don't type `--scheduler` either — boxy finds `flux`
on the login node and picks it (pin `--scheduler flux` / `BOXY_SCHEDULER=flux`
only if the cluster ALSO has `sbatch`, where boxy defaults to slurm):

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --ssh ambelt@eldorado
  auto: scheduler: flux (via detected (Flux broker is live) on eldorado)
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
| 1 | `boxy serve <8B> --ssh ambelt@hops` (no `--scheduler`) | `auto: scheduler: slurm (via detected (Slurm is live …) on hops)`, then `#SBATCH --account=fy140001` | `test_turnkey_e2e.py::test_ssh_auto_detects_scheduler_when_flag_absent` |
| 1b | `boxy serve <8B> --scheduler slurm --ssh ambelt@hops` | `auto: account: fy140001 (via mywcid on hops …)` then `#SBATCH --account=fy140001` | `test_turnkey_e2e.py::test_ssh_probes_mywcid_on_the_cluster` |
| 2 | on hops: `BOXY_SCHEDULER=slurm boxy serve <8B>` (no `--scheduler`) | `auto: scheduler: slurm (via config site.scheduler)`, `#SBATCH --account=fy140001` | `test_turnkey_e2e.py::test_login_node_scheduler_from_config` |
| 2b | `boxy serve <8B> --scheduler slurm --partition auto --ssh ambelt@hops` | `#SBATCH --partition=<idle-first list>` — starts wherever frees first | `test_turnkey_e2e.py::test_ssh_resolves_partition_auto_to_concrete_list` |
| 2c | `boxy serve <8B> --ssh ambelt@hops` (no `--time`) | `auto: time: 30:00 …`, `#SBATCH --time=30:00` | `test_turnkey_e2e.py::test_ssh_injects_default_walltime` |
| 3 | `boxy serve <8B> --ssh ambelt@eldorado` (Flux auto-detected) | `# flux: --bank=fy140001`, one queue | `test_turnkey_e2e.py::test_login_node_flux_bank_and_single_queue` |
| 4 | `boxy list --ssh ambelt@hops` | the job as `RUNNING`, then its endpoint | submit/follow loop in `_serve_submission` |

The account resolution and its placement in the batch script are proven
end-to-end in the sandbox with fake `mywcid`/`sbatch`/`ssh` shims; steps 1–4
against the live hops/eldorado systems are the only part that must run on your
side (the sandbox has no real scheduler).

---

## Full progression: one command → a served, team-shared model (with chisel)

Add `--share <name>` to publish an **everyone-URL** through the OpenShift chisel
relay once the model is up (teammates need nothing installed). The whole
deployment, start to finish:

```console
$ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler flux --ssh ambelt@eldorado --share llama8b
Enter OTP Token Value: ······
  auto: partition: pbatch (via flux queue list on eldorado)
  auto: account: FY140001 (via mywcid on eldorado; also: FY140252, FY260064 — placed in the batch script)
  auto: engine args: --max-model-len 8192 (packaged card 'llama-3.1-8b-instruct' — placed after --)
### CA      copied your site CA -> ambelt@eldorado:$HOME/.local/share/boxy/store/laptop-ca.crt  (remote SSL_CERT_FILE)
### Remote  ambelt@eldorado  $ boxy serve hf://meta-llama/Llama-3.1-8B-Instruct --scheduler flux --account FY140001 -- --max-model-len 8192

  auto: model: hf://meta-llama/Llama-3.1-8B-Instruct (transport URI — pulled via RamaLama)
  auto: scheduler: flux (submitting a batch job — detaches once READY)
### Submitted flux job f2c2yFbcbaAK  (boxy-llama-3.1-8b-instruct)
### Waiting for the job to start and the server to become ready ... (Ctrl-C detaches; the job keeps running)
###   job f2c2yFbcbaAK: RUNNING
###   server starting on eldo1001 — waiting up to 20 min for readiness at http://eldo1001:8000/v1/models (Ctrl-C detaches; the job keeps loading)
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Pulling vllm/vllm-openai ... 43%
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Loading safetensors checkpoint shards: 2/5
###   still loading (job f2c2yFbcbaAK: RUNNING)  ›  Capturing CUDA graph shapes: 18/35
### READY  http://eldo1001:8000/v1   (model: meta-llama/Llama-3.1-8B-Instruct, flux job f2c2yFbcbaAK)
###   try:   curl -s http://eldo1001:8000/v1/models
###   stop:  boxy stop boxy-llama-3.1-8b-instruct
### LOCAL   http://127.0.0.1:8000/v1   (tunnel over the SSH session; persists ~12h)
### SHARE   https://llama8b-boxy.apps.eldorado.example.gov/v1   (browser UI: https://llama8b-boxy.apps.eldorado.example.gov/)
```

Three URLs, three audiences — all from that one command:

| URL | Who | How |
|---|---|---|
| `http://eldo1001:8000/v1` | on the cluster | direct compute-node endpoint |
| `http://127.0.0.1:8000/v1` | **you**, on your laptop | auto SSH tunnel (no setup) |
| `https://llama8b-boxy.apps.…/v1` | **your team** | chisel relay everyone-URL (nothing installed) |

```console
# you (laptop):
$ curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# a teammate (anywhere, nothing installed):
$ curl -s https://llama8b-boxy.apps.eldorado.example.gov/v1/models
```

The chisel relay is deployed **once per cluster** (`boxy generate relay … | oc apply`,
see `DEMO-chisel.md`); after that every `--share` just publishes. The progress
lines (`› Loading safetensors …`) are the live tail of the job log, and boxy now
waits up to **20 min** for the weights to load before detaching (raise it with
`--ready-timeout 1800`), so a slow load no longer ends the command early.

---

## Troubleshooting: the server crashes at startup (vLLM `KeyboardInterrupt: terminated`)

If the job log ends with a vLLM `KeyboardInterrupt: terminated` cascade and a
`leaked semaphore` warning, the engine-core worker **died during startup** — the
real error is *above* that cascade in the log (`boxy logs <name>`), almost always
a **GPU OOM while profiling the KV cache**. vLLM defaults to the model's full
context (128K for Llama-3.1), which doesn't fit a single 24–40GB GPU.

boxy's model cards now cap this automatically — every packaged single-GPU vLLM
card sets `max_model_len = 8192`, and boxy places `--max-model-len 8192` in the
job (and injects it over `--ssh` so even an older cluster boxy gets it). You'll
see it in the `auto:` lines:

```
  auto: engine args: --max-model-len 8192 (card 'llama-3.1-8b-instruct')
```

Knobs if you need more context or hit OOM anyway:

- **More context on a big GPU**: `boxy serve <model> --scheduler slurm --ssh … -- --max-model-len 32768`
  (your `--` args always win over the card's).
- **Tight memory**: add `-- --gpu-memory-utilization 0.85` or `-- --enforce-eager`.
- **Land on a bigger GPU**: the default `--partition auto` already prefers GPU
  partitions; pin one with `--partition <name>` if your cluster mixes GPU sizes.

The command holds until the server is **READY** (or reports the failure with the
log tail) — it doesn't detach early. Re-running while an instance is already
serving reports its URL instead of launching a duplicate.

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
