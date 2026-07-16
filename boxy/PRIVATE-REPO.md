# Standing up boxy as a private GitHub repo

boxy currently lives in the `boxy/` subdirectory of the
`canopie25-paper-artifacts` monorepo, where CI runs green
(`.github/workflows/boxy-ci.yml`, matrix Python 3.11/3.12/3.13 + a package
build). This doc extracts it into its own **private** repository.

## Why this is a one-command step for you, not something already done

Creating the GitHub repo was attempted from the assistant's session and is
**blocked by the session's GitHub integration**, which returns
`403 Resource not accessible by integration` on `POST /user/repos` — repo
*creation* is outside its granted scopes (it can only read/write the two repos
the session was scoped to). So the repo has to be created from your own
authenticated `gh`/GitHub account. Everything needed for it is prepared and
committed here.

## One command (from the monorepo root, the parent of `boxy/`)

```bash
boxy/scripts/make-standalone-repo.sh --branch claude/boxy-turnkey --push abeltre1/boxy
```

That script (`boxy/scripts/make-standalone-repo.sh`):
1. `git subtree split --prefix=boxy` — extracts boxy's history with boxy at the
   repo root (no `boxy/` prefixes);
2. activates CI by moving the dormant `.github-export/workflows/{ci,release}.yml`
   into the active `.github/workflows/` (the CI template is already written for a
   root-level boxy: `ruff check src tests`, `pytest`, `python -m build`);
3. runs a local ruff+pytest sanity check;
4. `gh repo create abeltre1/boxy --private --source=. --remote=origin --push`.

Drop `--push abeltre1/boxy` to only produce a local `./boxy-standalone/` repo
and publish it yourself later:

```bash
boxy/scripts/make-standalone-repo.sh --branch claude/boxy-turnkey
cd boxy-standalone
gh repo create abeltre1/boxy --private --source=. --remote=origin --push
```

## After the push

CI (`.github/workflows/ci.yml`) runs automatically on the first push — the same
suite that is green in the monorepo today. The release workflow
(`.github/workflows/release.yml`) publishes on a tag.

Nothing about the monorepo changes: `.github-export/` is dormant there (GitHub
only runs workflows from the *root* `.github/workflows/`), so it exists purely to
seed the standalone repo.
