# Releasing boxy-hpc

A push of a `boxy-v*` tag runs `.github/workflows/boxy-release.yml`, which does
**four** things — three of them with zero external setup:

| Job | Ships | Setup needed |
|---|---|---|
| `build` | wheel + sdist, `twine check`ed, tag verified against `boxy.__version__` | none |
| `github-release` | a GitHub Release with wheel + sdist attached, install instructions in the notes | none |
| `container` | GHCR images: hardened + slim, python 3.11–3.14, `:VERSION`, `:latest`, `:VERSION-pyX.Y` | none |
| `publish` | PyPI via **Trusted Publishing** (OIDC — no stored tokens) | **opt-in**, see below |

So the FIRST release is one tag away: it produces installable artifacts even
before PyPI is configured. The PyPI job is gated on a repository variable and
is skipped until you set it:

```
Settings -> Secrets and variables -> Actions -> Variables:  PYPI_PUBLISH = true
```

## One-time setup (a maintainer does this once)

1. **Create the PyPI project + pending publisher.** On <https://pypi.org>, go to
   your account → *Publishing* → *Add a pending publisher*:
   - PyPI Project Name: `boxy-hpc`
   - Owner: `abeltre1`
   - Repository name: `canopie25-paper-artifacts`
   - Workflow name: `boxy-release.yml`
   - Environment name: `pypi`

2. **Create the GitHub environment.** In the repo → *Settings* → *Environments*
   → *New environment* named `pypi`. Optionally add required reviewers so a human
   approves each publish.

That's it — no secrets to paste.

## Cutting a release

1. Bump the version in **`src/boxy/__init__.py`** (`__version__`). `pyproject.toml`
   reads it dynamically, so there is only one place to edit.
2. Commit, and let CI go green on the branch (`boxy-ci`). Note the branch
   filter: pushes run CI on `main`, `master`, and `claude/**` only — any other
   branch name gets CI **only once a pull request exists**. Open the PR (or
   trigger `workflow_dispatch`) if you are working on a differently-named
   branch.
3. Tag and push — the tag must match the version, prefixed `boxy-v`:
   ```bash
   git tag boxy-v0.1.0
   git push origin boxy-v0.1.0
   ```
4. `boxy-release.yml` verifies the tag equals `boxy.__version__`, builds,
   `twine check`s, attaches the artifacts to a GitHub Release, pushes the GHCR
   images, and (when `PYPI_PUBLISH` is set) publishes to PyPI. Watch it in the
   Actions tab.

### Before you tag — the 5-minute smoke test

The one thing CI cannot tell you is whether the packaged DATA survived, so
check an installed copy rather than your checkout:

```bash
cd boxy && python -m pip wheel . --no-deps -w /tmp/bw
python3 -m venv /tmp/v && /tmp/v/bin/pip install /tmp/bw/*.whl
/tmp/v/bin/boxy --version        # EXPECT: "<version> (installed copy)"
/tmp/v/bin/boxy cards            # EXPECT: the packaged model + system cards
/tmp/v/bin/boxy doctor           # EXPECT: a report, exit 0, no traceback
cd /tmp && /tmp/v/bin/boxy serve --box examples/boxes/vllm-hf.toml \
    --location examples/locations/slurm-podman-cuda.toml --dryrun
                                 # EXPECT: a full plan from an unrelated directory
```

The `boxy-v*` prefix (not a bare `v*`) keeps boxy's tags from colliding with other
artifacts in this monorepo.

## Publishing to a local (private) PyPI

For an internal index (devpi, Nexus, Artifactory, `pypiserver`, …) skip the tag
flow entirely — the `Makefile` in `boxy/` does build → `twine check` → upload in
one step:

```bash
cd boxy
make wheel                                     # just build: dist/boxy_hpc-*.whl
make publish LOCAL_PYPI=https://pypi.example.gov/   # build + check + upload
```

`LOCAL_PYPI` accepts either the index's **upload endpoint URL** (passed to twine
as `--repository-url`) or a **section name from `~/.pypirc`** (passed as
`--repository`), and can be exported once in your shell instead of repeated on
the command line. A typical `~/.pypirc` for an internal index:

```ini
[distutils]
index-servers = site

[site]
repository = https://pypi.example.gov/
username = __token__          # or your LDAP user, per your index
password = <token>
```

Credentials can also ride the environment (`TWINE_USERNAME` / `TWINE_PASSWORD`),
which is friendlier for CI. If your index sits behind the site proxy, twine
honors `https_proxy`; a custom CA goes in `TWINE_CERT=/path/to/ca-bundle.crt`.

Installing from the local index on a cluster login node:

```bash
pip install --index-url https://pypi.example.gov/simple boxy-hpc
```

Note the upload endpoint and the `/simple` install index are usually *different
paths* on the same server — check your index's docs for both.

## Extracting boxy into its own repository

boxy is self-contained under `boxy/` (own `LICENSE`, `README.md`, `pyproject.toml`,
tests, and packaged examples), so it lifts out cleanly:

```bash
# carve out just boxy/ with its history
git clone https://github.com/abeltre1/canopie25-paper-artifacts boxy-standalone
cd boxy-standalone
git filter-repo --subdirectory-filter boxy      # pip install git-filter-repo

# the standalone workflow templates are ready to go:
mkdir -p .github/workflows
mv .github-export/workflows/*.yml .github/workflows/
```

> The `.github-export/` templates are a SEPARATE copy of the monorepo
> workflows (no `boxy/` path prefix, plain `v*` tags). They do not update
> themselves — diff them against `.github/workflows/boxy-*.yml` before relying
> on them, or the standalone repo silently ships an older release pipeline.

Then update `[project.urls]` in `pyproject.toml` and the absolute GitHub links in
`README.md`/`docs/08-releasing.md` to the new repository, and re-point the PyPI pending
publisher's *Repository name* / *Workflow name* to match. The live monorepo
workflows stay at the repo root; the `.github-export/` copies are the
standalone-repo versions (no `boxy/` path prefix, no `working-directory`).

## Standing up boxy as a private repository (one command)

Repo creation needs YOUR authenticated `gh` (the CI integration cannot create
repos). From the monorepo root:

```bash
gh repo create <you>/boxy --private --source boxy --push
```

Then follow "Extracting boxy into its own repository" above for the history
carve-out and workflow templates.
