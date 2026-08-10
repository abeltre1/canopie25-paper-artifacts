# Wormhole early-access kit

Getting a first user publishing a web service through [Wormhole](https://github.com/LLNL/wormhole-cli),
in three commands. Deliberately **standalone** — nothing here imports boxy or
changes it. Wormhole is evaluated on its own first; wiring it into `boxy serve`
is a separate decision to make after this works.

## What Wormhole is (and what it is not)

Wormhole lets an HPC user publish a web application or API running on a cluster
at a **stable URL**, with the center's multi-factor authentication in front of
it and optional token access for automated clients. It replaces the usual pile
of SSH tunnels and ad-hoc port forwards.

Two things shape everything below:

1. **It is operated by center administrators, not by you.** The gateway, route
   registry, and token service are deployed by the center. Your side is one CLI
   (`wh`) plus a token. You cannot stand the platform up yourself.
2. **It is early access.** Availability is granted per user. Most first-run
   failures are "you are not enabled yet", not a mistake in your command — which
   is exactly why `preflight.sh` exists and reports that case separately.

## The three commands

```bash
./preflight.sh                          # am I enabled, and is anything missing?
./install-wh.sh                         # only if your site has not shipped `wh`
./smoke-test.sh --allowed-users "$USER" # publish a marker file, prove the URL works
```

### 1. Preflight

Checks the client, the config layering, the credential, and — the part that
actually matters — whether the route registry answers for *you*:

```
wormhole preflight — 2026-08-04T18:22:03Z

client
  [ ok ]   wh found at /usr/local/bin/wh

configuration
  [ ok ]   site config present: /etc/wormhole/cli.toml
  [note]   no user config at ~/.config/wormhole/cli.toml (optional)
  [ ok ]   endpoint comes from a config file

credential
  [ ok ]   WORMHOLE_TOKEN is set (44 chars, sha256:9f2c41ab)

service reachability
  [ ok ]   route registry answered — you are enabled

summary: 5 ok, 0 failed, 1 note(s)
```

The output is safe to paste into a support ticket: the token is never printed,
only a SHA-256 prefix, which is enough for a support engineer to confirm you are
holding the credential they issued without either side disclosing it.

### 2. Install the client (usually unnecessary)

`wormhole-cli`'s own README says the binary is *"expected to be distributed as a
user-facing `wh` binary on the target clusters"*. On a site-managed system it is
already there. Use `install-wh.sh` only when it is not: it builds from source
(Go; there are no released binaries and no `go install` line upstream) into
`~/.local/bin/wh` with no sudo.

### 3. Smoke test

Publishes a temporary directory containing one marker file. The design point is
the **split** — it verifies the service on `127.0.0.1` *before* involving
Wormhole:

```
### 1/3  serving a marker file locally on port 41397
     local check passed (served the marker over 127.0.0.1)

### 2/3  publishing it through Wormhole as 'boxy-smoke-user1'
     wh open --name boxy-smoke-user1 --app-port 41397 --allowed-users user1
```

So when it fails you already know which half broke. A passing local check and a
failing Wormhole URL is a platform or authorization problem, and the local check
failing on its own has nothing to do with Wormhole at all.

## Two rules the scripts enforce

**The token never goes on the command line.** `wh --token <value>` puts a live
credential in `argv`, and `ps` is world-readable on a shared login node — any
other user on that node can read it. Every script here reads `WORMHOLE_TOKEN`
from the environment instead, so the commands they print are safe to copy into a
ticket. `cli.toml.example` explains why a token in a config file is also a poor
second choice.

**No publish without an access list.** Wormhole requires at least one
`--allowed-users` or `--allowed-groups`, and `smoke-test.sh` refuses to invent
one. Who can reach a service is not a good default to guess at, especially for a
model endpoint.

## When it does not work

Run `./preflight.sh` first and read the `[FAIL]` lines — they distinguish "no
client", "no endpoint", "no token", and "the registry rejected you", which are
four different tickets.

If the registry rejects you, early access is granted per user: contact your
center's support desk and quote the token fingerprint from the preflight output.
For questions about Wormhole itself, the project points contributors at
<wormhole-dev@llnl.gov> and the repositories below.

## Upstream

| Repository | Role |
| --- | --- |
| [wormhole-cli](https://github.com/LLNL/wormhole-cli) | the `wh` end-user client |
| [wormhole-airlock](https://github.com/LLNL/wormhole-airlock) | translation layer to downstream web services |
| [wormhole-holepunch](https://github.com/LLNL/wormhole-holepunch) | proxy and gateway layers |
| [wormhole-route-registry](https://github.com/LLNL/wormhole-route-registry) | manages service routes |
| [wormhole-token-service](https://github.com/LLNL/wormhole-token-service) | authentication and authorization |

## Status of this kit

Written against the documented `wh` interface (`wh open --name --app-port
--allowed-users/--allowed-groups [-- command]`, `wh route list`, the
`WORMHOLE_*` environment variables, and the `/etc/wormhole/cli.toml` config
layering).

**Not yet executed against a live Wormhole deployment** — there is no reachable
instance from the machine this was authored on. The local half of the smoke test
and the argument handling are tested; the `wh` invocation itself is not. Treat
the first real run as the test, and please correct anything that does not match.
