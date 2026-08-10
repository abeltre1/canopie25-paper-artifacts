# Design: sandboxed remote execution with NVIDIA OpenShell

**Status: proposal. Nothing here is implemented.** Written to answer one
question before any code exists: *does boxy need this, or does it already have
it?*

Short answer: boxy already has a **remote execution transport**. It has no
**sandbox**. Those are different problems, and boxy only acquires the second
problem the moment it starts running code it did not write — which is exactly
what the companion `boxy study` proposal introduces.

---

## 1. What boxy already does — the duplication check

This section exists because "secure remote execution" sounds like something
already built, and half of it is.

| Property | Where | Status |
| --- | --- | --- |
| One authenticated session, OTP/YubiKey prompted once, multiplexed | OpenSSH ControlMaster, `remote.ensure_master` `remote.py:138`, `control_persist` `:59` | **Done** |
| Every interpolation into a remote shell quoted | `shlex.quote` throughout `_remote_command` `remote.py:262-282` | **Done** — a security audit (PR #5) confirmed no `shell=True`, no `os.system`, no `eval`/`exec`, no unquoted user input |
| Remote files created private, with no world-readable window | `push_file` under `umask 077` `remote.py:352` | **Done** (PR #5) |
| Secrets kept out of displayed commands and public docs | `redact.py` (PR #5) | **Done** |
| Site CA propagated so remote TLS trusts what the laptop trusts | `propagate_ca` `remote.py:211` | **Done** |
| Nothing installed on the cluster; no daemon, no agent | agentless batch scripts, `deploy.render_agentless_script` | **Done** |

So: **boxy's transport is not the gap.** An OpenShell integration that replaced
`ssh` with something else would be duplicated work, and worse than what exists —
it would lose ControlMaster's single-prompt OTP handling, which is the thing that
makes boxy usable behind a YubiKey.

### What boxy does *not* have

Every command boxy executes remotely is a command **boxy composed itself**: a
rendered batch script, a container invocation, a scheduler call. They are
golden-tested token by token. The trust model is "we wrote it, we quoted it, the
tests assert it".

There is no mechanism for running code that boxy did **not** write, under a
restriction weaker than "this user's full shell". Today, anything boxy runs
remotely can read `~/.ssh`, reach any host the login node can reach, and write
anywhere the user can write. That is correct for a rendered `sbatch` script. It
is not correct for a snippet some agent produced sixty seconds ago.

---

## 2. What OpenShell is

NVIDIA OpenShell (Apache 2.0, open-sourced March 2026) is a runtime that executes
agent workloads inside **kernel-level** sandboxes governed by declarative policy.
The parts that matter here:

- **Landlock LSM** confines a process to declared filesystem paths —
  `allowed_reads` / `allowed_writes`.
- **seccomp** filters syscalls, applied in two phases (a narrow supervisor
  prelude, then a broader runtime filter after privilege drop).
- **Default-deny network egress**, with policy-declared exceptions.
- **Policy as data**, split into static sections (`filesystem_policy`,
  `landlock`, `process`) locked at sandbox creation, and dynamic sections
  (`network_policies`, `network_middlewares`) that hot-reload on a running
  sandbox.
- **Cannot request root** — `run_as_user` / `run_as_group` may not be 0.
- **Gateways**: local, **remote over SSH**, or cloud-registered.

The load-bearing property for HPC is that Landlock and seccomp sit *below* the
container layer. On a cluster where an unprivileged user cannot get a privileged
container runtime, kernel-level confinement of an ordinary process is the only
mechanism actually available.

---

## 3. Why boxy would want it — the concrete trigger

Not "agents are a security risk" in the abstract. One specific thing:

The `boxy study` proposal (doc 13) has boxy deciding what to run next from
measured results. Everything in *that* design is arithmetic boxy performs, so it
stays inside the existing trust model. But the moment anyone wants:

- an LLM writing the summary of a study, with tool access to the run outputs, or
- an agent proposing an application-parameter change and testing it, or
- a user-supplied `[app.run]` line from a card they downloaded,

...boxy is executing code it did not write, on a login node, as a user with
credentials. That is a new trust boundary, and it is the one thing boxy has no
answer for.

**Stating it plainly: boxy does not need OpenShell today.** It needs it exactly
when the agentic loop stops being arithmetic. Sequencing the two proposals in
that order is deliberate — this document should not be implemented ahead of a
demonstrated need for it.

---

## 4. Where it would attach

boxy already has the right seam. `exposers/` is a registry of pluggable
components (`relay`, `hosts`) chosen by name, mirroring `backends/` and
`schedulers/`. The same shape fits here:

```
  executors/
    base.py        Executor: run(argv, policy) -> result;  available() -> bool
    direct.py      today's behaviour — ssh + shlex.quote, no confinement
    openshell.py   the same command inside a policy-confined sandbox
```

`direct` is the default and stays the default. Nothing that works today changes
its execution path. `openshell` is opt-in per command:

```bash
boxy study osu-benchmarks --ssh <cluster> --executor openshell
```

An `Executor` registry rather than a flag threaded through the call graph means
the policy decision is made once, in one place, and is testable with a fake
executor exactly as `BOXY_SSH`/`BOXY_OC` shims already make the transport
testable.

### The policy boxy would declare

Derivable from what boxy already knows about a job, which is the argument for
generating it rather than asking the user to write one:

| Policy field | boxy already knows |
| --- | --- |
| `allowed_reads` | the app's spack prefix, the model/data staging path |
| `allowed_writes` | the job's scratch dir, the results dir — nothing else |
| network | default-deny; allow only the registry/endpoint the job legitimately needs |
| `run_as_user` | the invoking user, never root (OpenShell forbids root anyway) |

The interesting consequence: `~/.ssh` and the user's cloud credentials fall
outside `allowed_reads` by construction. An agent that goes wrong cannot read the
key that would let it move to another machine.

---

## 5. The open questions, which are the point of this document

1. **Does OpenShell run on the target clusters at all?** Landlock needs a
   sufficiently recent kernel with the LSM enabled. HPC login nodes run
   conservative enterprise kernels. **This is question zero** — if Landlock is
   unavailable on the machines in question, the rest of this document is moot and
   the honest answer is a different mechanism. Cheap to determine and worth
   determining before anything else.

2. **Login node, compute node, or both?** The threat model differs: a login node
   is shared and holds credentials; a compute node is exclusively allocated.
   Confining the login-node half may be sufficient and is certainly cheaper.

3. **Does the sandbox survive the scheduler?** boxy's remote work is submitted as
   batch jobs that outlive the SSH session. A sandbox that only wraps an
   interactive process does not cover a `sbatch` script — and it is not obvious
   whether the sandbox should wrap the submitting command, the submitted script,
   or both.

4. **What is the failure mode when OpenShell is absent?** Consistent with the
   rest of boxy: degrade with an explanation, never traceback. But for a
   *security* feature, silently falling back to unconfined execution is the wrong
   default. Proposal: `--executor openshell` is a hard requirement that fails
   loudly if unavailable; there is no automatic fallback.

5. **Does this pull in a gateway to operate?** OpenShell's remote deployment goes
   through a gateway. If that means something long-running on the cluster, it
   collides directly with boxy's "nothing installed, no daemon, no agent"
   property — which is one of its genuinely differentiating characteristics.
   Worth knowing before committing.

6. **Is the added complexity worth it versus the alternatives?** A restricted
   user account, a container with dropped capabilities, or simply not running
   agent-written code on a cluster are all cheaper. This proposal should have to
   beat them, not just exist.

---

## 6. Scope of a first implementation, if approved

**In:** the `executors/` registry with `direct` as the unchanged default; an
`openshell` executor behind an explicit opt-in; policy generation from what boxy
already knows about a job; a preflight that reports whether kernel support and
the binary are actually present; a fake executor for tests.

**Out:** replacing the SSH transport (it is not the gap); any change to the
default execution path; a gateway boxy operates; sandboxing boxy's own rendered
scripts, which are golden-tested and do not need it.

**Not started until:** question 1 is answered on a real target machine, and doc
13's agentic loop actually grows a component that runs code boxy did not write.

---

## Sources

- [NVIDIA/OpenShell](https://github.com/NVIDIA/openshell) — Apache 2.0
- [Developer guide](https://docs.nvidia.com/openshell/latest/get-started),
  [sandbox policies](https://docs.nvidia.com/openshell/sandboxes/policies),
  [security best practices](https://docs.nvidia.com/openshell/security/best-practices),
  [gateways](https://docs.nvidia.com/openshell/sandboxes/manage-gateways)

The OpenShell behaviour described above is taken from published documentation
summaries; `docs.nvidia.com` and `perspectives.nvidia.com` are both blocked by
this environment's egress policy, so **no exact CLI surface is quoted here and
none should be inferred**. Confirming the command names, policy file format, and
kernel prerequisites against the real docs is the first task of any
implementation.
