# Releasing boxy-hpc

boxy publishes to PyPI via **Trusted Publishing** (OpenID Connect) — no API
tokens are stored anywhere. A push of a `boxy-v*` tag builds the wheel/sdist and
uploads them through `.github/workflows/boxy-release.yml`.

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
2. Commit, and let CI go green on the branch (`boxy-ci`).
3. Tag and push — the tag must match the version, prefixed `boxy-v`:
   ```bash
   git tag boxy-v0.1.0
   git push origin boxy-v0.1.0
   ```
4. `boxy-release.yml` verifies the tag equals `boxy.__version__`, builds,
   `twine check`s, and publishes to PyPI. Watch it in the Actions tab.

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
index-servers = sandia

[sandia]
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

Then update `[project.urls]` in `pyproject.toml` and the absolute GitHub links in
`README.md`/`RELEASING.md` to the new repository, and re-point the PyPI pending
publisher's *Repository name* / *Workflow name* to match. The live monorepo
workflows stay at the repo root; the `.github-export/` copies are the
standalone-repo versions (no `boxy/` path prefix, no `working-directory`).
