# 15 — Release runbook: NETWORKS

Zero to a served model from a laptop, across the three network shapes a release
actually meets. The command is the same in all three; what changes is what boxy
discovers and what you have to tell it.

| Shape | Cluster's egress | What boxy does |
|---|---|---|
| **A. Direct** | login + compute reach the internet | probes, injects nothing |
| **B. Proxied** | only through a TLS-intercepting proxy (Zscaler-style) | forwards the proxy + stages the site CA |
| **C. Isolated compute** | login node has egress, compute nodes do not | pre-stages on the login node, serves by path |

No egress anywhere → that is not this document: see
[13-runbook-airgapped.md](13-runbook-airgapped.md).

---

## Phase 0 — Laptop (5 min)

```bash
pip install 'boxy-hpc[ramalama]'        # or: uv tool install boxy-hpc
boxy --version
boxy doctor                             # local: runtime, TLS, tokens
boxy doctor --ssh <user>@<login> --net  # the cluster: runtime, scheduler, registries
```

`boxy doctor --ssh` is the cheapest thing you will do all day: it audits the
cluster's container runtime, scheduler, accelerator and registry reachability
and prints OK/WARN/FAIL **with a fix for each**, before you spend an
allocation.

**OTP/YubiKey:** you are prompted **once**. boxy opens one SSH master and every
later command reuses it.

---

## Phase 1 — Which shape are you in?

You do not have to guess; boxy probes. But knowing tells you what to expect:

```bash
ssh <login> 'curl -sIf https://huggingface.co -o /dev/null && echo DIRECT || echo BLOCKED'
ssh <login> 'echo "${https_proxy:-<none>}"'
```

- `DIRECT` → **Shape A**.
- `BLOCKED` + a proxy exported on the login node → **Shape B** (boxy picks it
  up automatically; no flag needed).
- `BLOCKED` + no proxy → ask your admins for the proxy URL and use `--proxy`,
  or go to the air-gap runbook.

---

## Phase 2 — Stage the model (recommended for anything large)

```bash
boxy pull hf://<org>/<model> --ssh <user>@<login>
```

- Sizes the model first and **refuses** if the store cannot hold it.
- Probes egress and downloads with the **login node's own curl** — the exact
  binary, trust store, and network path the probe just verified.
- Runs **detached**: it survives your laptop sleeping, closing the lid, or the
  SSH session dropping. Re-run the same command for status; it resumes rather
  than restarting.

Re-running prints progress; when it finishes it prints the serve line to use.

## Phase 3 — Serve

```bash
boxy serve hf://<org>/<model> --ssh <user>@<login> --time 4:00:00
```

Zero geometry flags: the model card plus the cluster's system card decide GPUs,
nodes, and the engine args. Watch the `auto:` lines — they are boxy telling you
every decision it made.

**What each shape gets:**

- **A (direct):** `auto: egress: <registry> verifies directly from <host> — NOT
  injecting this laptop's proxy`. This matters: forwarding your laptop's proxy
  into a cluster that reaches the registry directly *replaces a working path
  with an unroutable one*. It cost six rounds of in-container TLS failures in
  the field before boxy learned to probe.
- **B (proxied):** `auto: proxy: forwarding …` plus the site CA staged onto the
  shared FS and mounted into the container, so an in-container fetch through
  the interceptor doesn't die with `CERTIFICATE_VERIFY_FAILED`. Override with
  `--proxy URL` — an explicit `--proxy` is never probed away.
- **C (isolated compute):** add `--prestage`. boxy pulls the image *and* the
  model on the login node (which has your SSH session's network) onto the
  shared FS and serves the model **by path**, so the compute node never needs
  to reach Docker Hub or HuggingFace. A completed `boxy pull` already counts as
  the pre-stage, under every mode.

### Certificates, when you need them

```bash
boxy trust huggingface.co            # capture the interceptor's chain
boxy trust <registry-host> --proxy http://proxy:80
```

`SSL_CERT_FILE` **replaces** Python's trust store rather than adding to it, so
a file holding only your site CA breaks every host that is *not* intercepted.
boxy merges automatically (public CAs + your site CA) when `SSL_CERT_FILE` is
set; `boxy trust` writes into that merged bundle. Persist the export — it dies
with the shell.

---

## Phase 4 — Reach the endpoint

The job runs on a **compute node** whose hostname resolves only on the cluster.
Three supported ways, all from the laptop:

```bash
boxy attach --ssh <user>@<login>        # tunnel + wait for READY + print the URL
boxy curl   --ssh <user>@<login> --prompt "hello"   # one-shot, from the cluster side
boxy open   --ssh <user>@<login> --route mymodel    # browser: http://mymodel.localhost:8000/
```

Then, through the tunnel:

```bash
curl -s http://127.0.0.1:8000/v1/models    # the id to put in "model"
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id from above>","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

**On the login node itself** there is no tunnel — curl the compute node
directly: `curl -s http://<node>:8000/v1/models`. `boxy list` shows the node.

**Share it with the team** (zero setup for them):

```bash
boxy open <name> --ssh <user>@<login> --share mymodel
# -> https://mymodel-boxy.apps.<cluster>/v1   for anyone on the corporate network
boxy unshare mymodel
```

---

## Phase 5 — Lifecycle

```bash
boxy list --ssh <user>@<login>     # state probed over SSH; nothing installed there
boxy logs <name>                   # the job log + boxy's crash diagnosis
boxy stop <name>                   # cancel the job
boxy clean                         # sweep finished-job records/scripts/logs
```

`boxy list/logs/stop/attach/curl/open` all answer from **this machine's**
records over the SSH master — an agentless serve installs nothing on the
cluster, so there is no cluster boxy to ask.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CERTIFICATE_VERIFY_FAILED` in the container | site interceptor, CA not trusted inside | `boxy trust <host>`; boxy stages + mounts the merged bundle |
| Image pull 403 from the compute node | node is behind the proxy, registry blocked | boxy forwards the proxy automatically; else `--proxy URL`, or `--registry <site mirror>` |
| It worked on the login node but fails in the job | you are in Shape C | add `--prestage` (or run `boxy pull` first) |
| The serve broke *after* you set a laptop proxy | Shape A cluster, proxy injected | current boxy probes and skips it; `BOXY_PROXY=` clears a configured one |
| `boxy curl/open` says nothing is serving | you are on a different machine than the one that submitted | records live on the submitting laptop; use it, or `boxy list --ssh` there |
| OTP prompted repeatedly | the SSH master lapsed (site session cap) | re-run the command; it re-opens one master |
| Job READY but the model id 404s | the engine advertises a different id | `curl /v1/models` and use what it returns (current boxy serves under the canonical id) |

---

## What is verified where

| Step | How it was verified |
|---|---|
| direct egress → no proxy injected (prefix *and* container env) | e2e test |
| blocked egress → proxy still forwarded | e2e test |
| explicit `--proxy` survives a "direct" verdict | e2e test |
| detached pull: launch, status, resume, refusal on a full store | 18 tests |
| CA staging + merge over SSH | suite |
| curl/open answer from laptop records | e2e test |
| a real proxied/direct cluster | **your systems** — this runbook is the procedure |
