# Design: sandboxed vs direct deployment (NVIDIA OpenShell)

**Status: proposal. Nothing here is implemented.**

Deployment gains an **isolation choice**: run my application in a sandbox, or run
it the way it runs today. Presented to the user the same way boxy already
presents the charge-account choice — a numbered menu when the answer is genuinely
ambiguous, a remembered default per cluster, and an explicit flag that skips the
question entirely.

The unsandboxed path is not a fallback or a legacy mode. It is a first-class,
deliberately-chosen option, and it stays the default for everything boxy composes
itself.

---

## 1. Duplication check — what boxy already has

This section is first because "secure remote execution" sounds like something
already built, and half of it is.

| Property | Where | Status |
| --- | --- | --- |
| One authenticated session, OTP/YubiKey prompted once, multiplexed | ControlMaster, `remote.ensure_master` `remote.py:138` | **Done** |
| Every interpolation into a remote shell quoted | `shlex.quote` throughout `_remote_command` `remote.py:262-282` | **Done** — the PR #5 audit found no `shell=True`, no `os.system`, no `eval`/`exec`, no unquoted user input |
| Remote files created private, no world-readable window | `push_file` under `umask 077` `remote.py:352` | **Done** (PR #5) |
| Secrets kept out of displayed commands and public docs | `redact.py` (PR #5) | **Done** |
| Site CA propagated so remote TLS trusts what the laptop trusts | `propagate_ca` `remote.py:211` | **Done** |
| Nothing installed on the cluster; no daemon, no agent | agentless batch scripts, `deploy.render_agentless_script` | **Done** |

**boxy's transport is not the gap.** Replacing `ssh` with something else would be
duplicated work and strictly worse — it would lose ControlMaster's single-prompt
OTP handling, which is what makes boxy usable behind a YubiKey.

### The gap

Every command boxy runs remotely is one **boxy composed itself** — a rendered
batch script, a container invocation, a scheduler call — golden-tested token by
token. The trust model is "we wrote it, we quoted it, the tests assert it".

There is no way to run anything under a restriction weaker than the user's full
shell. Anything boxy launches can read `~/.ssh`, reach any host the login node
reaches, and write anywhere the user can write. Correct for a rendered `sbatch`
script. Wrong for a container image someone else built, an app card downloaded
from a colleague, or a snippet an agent produced sixty seconds ago.

---

## 2. What OpenShell provides

NVIDIA OpenShell (Apache 2.0, open-sourced March 2026) executes workloads inside
kernel-level sandboxes governed by declarative policy:

- **Landlock LSM** confines a process to declared paths — `allowed_reads` /
  `allowed_writes`.
- **seccomp** filters syscalls in two phases (a narrow supervisor prelude, then a
  broader runtime filter after privilege drop).
- **Default-deny network egress**, with policy-declared exceptions.
- **Policy as data** — static sections (`filesystem_policy`, `landlock`,
  `process`) locked at sandbox creation; dynamic sections (`network_policies`,
  `network_middlewares`) hot-reloadable on a running sandbox.
- **Root is forbidden** — `run_as_user` / `run_as_group` may not be 0.
- **Gateways**: local, remote over SSH, or cloud-registered.

The load-bearing property for HPC: Landlock and seccomp sit *below* the container
layer. On a cluster where an unprivileged user cannot get a privileged container
runtime, kernel-level confinement of an ordinary process is the only mechanism
actually available.

---

## 3. The choice, modeled on the account picker

boxy already solved this UX problem once. `picker.py` discovers the charge
accounts a user may use, and — instead of silently taking the first — presents a
numbered menu, remembers the answer per cluster, and validates the remembered
value against the live list so a stale default cannot charge an account the user
has lost. Isolation gets the same treatment.

### 3.1 Precedence

Identical in shape to `site.pick_account`:

| Rank | Source | Effect |
| --- | --- | --- |
| 1 | `--isolation sandbox` / `--isolation direct` | wins, no menu |
| 2 | `BOXY_ISOLATION` | wins, no menu |
| 3 | config `site.isolation` | wins, no menu |
| 4 | remembered per-cluster choice | offered as the menu default |
| 5 | the menu | only when interactive **and** both modes are actually available |
| 6 | the built-in default | `direct` — see §4 |

`site.pick_isolation` takes `always` / `never` / `auto`, exactly like
`site.pick_account`. `auto` means "menu only when stdin and stdout are both a
TTY", so CI and batch scripts never block on a prompt. That property is
non-negotiable and already proven in `picker.is_interactive` (`picker.py:26-39`).

### 3.2 What the user sees

```
$ boxy app osu-benchmarks --ssh <cluster>

  How should this run on <cluster>?

    1) direct    — full user shell, as boxy runs today          [remembered]
    2) sandbox   — Landlock + seccomp confinement via OpenShell

  Choose [1]:

  auto: isolation: direct (remembered for <cluster>; --isolation sandbox to change)
```

And when the question has only one honest answer, there is no menu — the same
rule the account picker already follows (`len(names) > 1`):

```
  auto: isolation: direct (sandbox unavailable on <cluster>: kernel 4.18 has no Landlock)
```

### 3.3 Where the analogy breaks — and it must

For accounts, a stale remembered value is a nuisance: you charge the wrong
project and someone emails you. **For isolation, a silent fallback is a security
downgrade**, and the pattern must diverge here:

- **A remembered `sandbox` that is no longer available does NOT quietly become
  `direct`.** It stops and says so. The user re-chooses explicitly.
- **`--isolation sandbox` is a hard requirement.** If OpenShell or Landlock is
  missing, the command fails loudly. There is no automatic downgrade, ever.
- **Only `direct` may be implicit.** Choosing *less* isolation is always an
  explicit act by a human.

The general principle: the menu exists to make the choice visible, not to make
either answer automatic when it matters.

---

## 4. Defaults, and where there is no choice at all

Offering the choice everywhere would be wrong. The default depends on **who wrote
the code being run**:

| What is running | Default | Choice offered? |
| --- | --- | --- |
| A batch script boxy rendered (`serve`, `sweep`) | `direct` | Yes — sandbox available for defence in depth |
| A packaged app card's `run` lines (`osu-benchmarks`) | `direct` | Yes |
| A **user-supplied or third-party** app card | `direct`, with a warning naming the card's source | Yes, and the warning recommends `sandbox` |
| Code boxy did **not** write — a model-authored command, an agent-proposed parameter change | **none — `sandbox` required** | **No.** Refuse to run unsandboxed. |

That last row is the point of the whole proposal. Everywhere else the choice is a
genuine user preference between two legitimate options; there, it is not a
preference, and the design should not pretend otherwise by offering a menu.

Backwards compatibility falls out: every command that works today keeps working
identically, because `direct` is what they already do and stays their default. No
existing user is opted into a new execution path by upgrading.

---

## 5. Provenance — the mode is part of the result

A benchmark produced under a default-deny egress policy with a confined
filesystem is **a different claim** from the same benchmark produced with a full
shell. Anyone comparing two numbers needs to know which they have.

- The job record gains `isolation: "direct" | "sandbox"` and, when sandboxed, the
  policy digest.
- The results envelope carries the same, so `boxy results show` and any plot
  legend can state it.
- `boxy list` shows it per instance.

This is cheap now and impossible to reconstruct later. It also makes the honest
performance question answerable: seccomp filtering costs something per syscall,
and a study that silently changed isolation mode halfway through its ladder would
be worthless in a way nobody could detect after the fact.

---

## 6. Where it attaches

boxy already has the right seam. `exposers/` is a name-keyed registry of
pluggable components (`relay`, `hosts`), mirroring `backends/` and `schedulers/`.
Same shape:

```
  executors/
    base.py        Executor: run(argv, policy) -> result;  available() -> (bool, why)
    direct.py      today's behaviour — ssh + shlex.quote, no confinement
    openshell.py   the same command inside a policy-confined sandbox
```

`available()` returns a *reason* alongside the boolean, because "sandbox
unavailable" must be able to say **why** — missing binary, kernel too old,
Landlock disabled — both in the menu and in the hard-failure message.

A registry rather than a flag threaded through the call graph means the decision
is made once, in one place, and is testable with a fake executor exactly as
`BOXY_SSH` / `BOXY_OC` shims already make the transport testable.

### The policy boxy would generate

Derivable from what boxy already knows about a job, which is the argument for
generating it rather than asking a user to hand-write policy:

| Policy field | boxy already knows |
| --- | --- |
| `allowed_reads` | the app's spack prefix, the model/data staging path |
| `allowed_writes` | the job's scratch dir, the results dir — nothing else |
| network | default-deny; allow only the registry or endpoint the job needs |
| `run_as_user` | the invoking user, never root (OpenShell forbids root anyway) |

Useful consequence: `~/.ssh` and cloud credentials fall outside `allowed_reads`
**by construction**. An agent that goes wrong cannot read the key that would let
it reach another machine.

---

## 7. Open questions

1. **Is Landlock available on the target clusters?** It needs a recent kernel
   with the LSM enabled; HPC login nodes run conservative enterprise kernels.
   **Question zero** — if the answer is no, §3's menu will only ever show one
   option and this proposal should be replaced with a different mechanism. Cheap
   to determine, and worth determining before anything else.
2. **Login node, compute node, or both?** The threat models differ: a login node
   is shared and holds credentials; a compute node is exclusively allocated.
   Confining only the login-node half may be sufficient and is certainly cheaper.
3. **Does the sandbox survive the scheduler?** boxy submits batch jobs that
   outlive the SSH session. It is not obvious whether the sandbox should wrap the
   submitting command, the submitted script, or both — and a sandbox that only
   wraps an interactive process covers none of the interesting cases.
4. **Does OpenShell's remote deployment require a gateway to be running?** If
   that means something long-lived on the cluster, it collides directly with
   boxy's "nothing installed, no daemon, no agent" property, which is one of its
   genuinely differentiating characteristics.
5. **What does confinement cost?** seccomp filtering is per-syscall. If it is
   measurable on an I/O-heavy job, that belongs in the menu text, not discovered
   later in a benchmark comparison.
6. **Should the remembered choice be per-cluster or per-cluster-and-command?**
   Per-cluster matches the account picker. But "sandbox my agent work, run my
   own serves direct" is a plausible thing to want, and per-cluster cannot
   express it.
7. **Does this beat the cheaper alternatives?** A restricted user account, a
   dropped-capability container, or simply not running agent-written code on a
   cluster are all less work. This proposal should have to win on merit.

---

## 8. Scope of a first implementation, if approved

**In:** the `executors/` registry with `direct` as the unchanged default;
`openshell` behind explicit opt-in; the picker integration (`--isolation`,
`BOXY_ISOLATION`, `site.isolation`, `site.pick_isolation`, per-cluster recall);
policy generation from what boxy already knows; `available()` reporting a reason;
isolation recorded in job records and result envelopes; a fake executor for
tests.

**Out:** replacing the SSH transport (not the gap); any change to the default
execution path; a gateway boxy operates; sandboxing boxy's own rendered scripts
by default.

**Not started until:** question 1 is answered on a real target machine.

---

## Sources

- [NVIDIA/OpenShell](https://github.com/NVIDIA/openshell) — Apache 2.0
- [Developer guide](https://docs.nvidia.com/openshell/latest/get-started),
  [sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies),
  [security best practices](https://docs.nvidia.com/openshell/security/best-practices),
  [gateways](https://docs.nvidia.com/openshell/sandboxes/manage-gateways)

The OpenShell behaviour above is taken from published documentation summaries;
`docs.nvidia.com` and `perspectives.nvidia.com` are both blocked by this
environment's egress policy, so **no exact CLI surface is quoted here and none
should be inferred**. Confirming command names, the policy file format, and
kernel prerequisites against the real documentation is task one of any
implementation.
