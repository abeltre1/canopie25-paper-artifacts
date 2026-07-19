#!/usr/bin/env bash
# make-standalone-repo.sh — extract boxy/ from the canopie25-paper-artifacts
# monorepo into a self-contained repository and (optionally) push it to a new
# PRIVATE GitHub repo. boxy becomes the repo root: the CI template that ships
# dormant at .github-export/workflows/ is moved into the active .github/workflows/.
#
# Usage (run from the monorepo root, i.e. the parent of boxy/):
#   boxy/scripts/make-standalone-repo.sh [--branch <src-branch>] [--out <dir>] \
#                                        [--push abeltre1/boxy]
#
# Examples:
#   # just produce a local standalone repo in ./boxy-standalone:
#   boxy/scripts/make-standalone-repo.sh --branch claude/boxy-turnkey
#   # produce it AND create+push a private GitHub repo (needs gh authed):
#   boxy/scripts/make-standalone-repo.sh --branch claude/boxy-turnkey --push abeltre1/boxy
set -euo pipefail

branch="$(git rev-parse --abbrev-ref HEAD)"
out="boxy-standalone"
push_slug=""
while [ $# -gt 0 ]; do
  case "$1" in
    --branch) branch="$2"; shift 2 ;;
    --out)    out="$2"; shift 2 ;;
    --push)   push_slug="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -d boxy ] || { echo "error: run from the monorepo root (the parent of boxy/)." >&2; exit 1; }
[ -e "$out" ] && { echo "error: output path '$out' already exists." >&2; exit 1; }

echo "==> splitting boxy/ history from '$branch' (git subtree)…"
split_sha="$(git subtree split --prefix=boxy "$branch")"

echo "==> materializing the standalone repo at '$out'…"
git init -q "$out"
git -C "$out" pull -q "$(pwd)/.git" "$split_sha"

echo "==> activating CI (.github-export/workflows -> .github/workflows)…"
mkdir -p "$out/.github/workflows"
git -C "$out" mv .github-export/workflows/ci.yml      .github/workflows/ci.yml
git -C "$out" mv .github-export/workflows/release.yml .github/workflows/release.yml 2>/dev/null || true
rmdir "$out/.github-export/workflows" "$out/.github-export" 2>/dev/null || true
git -C "$out" commit -qm "ci: activate standalone workflows (boxy is the repo root)"

echo "==> sanity check (ruff + tests) in the standalone tree…"
( cd "$out" && python -m pip install -q -e '.[test]' ruff >/dev/null 2>&1 || true
  ruff check src tests && python -m pytest -q --ignore=tests/test_degraded_and_live.py ) || {
    echo "warning: standalone sanity check did not fully pass here (often just missing deps); "
    echo "         CI will run it cleanly on push." >&2; }

if [ -n "$push_slug" ]; then
  command -v gh >/dev/null || { echo "error: --push needs the gh CLI, authenticated." >&2; exit 1; }
  echo "==> creating PRIVATE GitHub repo '$push_slug' and pushing…"
  ( cd "$out" && gh repo create "$push_slug" --private --source=. --remote=origin --push )
  echo "==> done: https://github.com/$push_slug (CI runs on the first push)."
else
  echo "==> done. Local standalone repo: $out"
  echo "    To publish it privately:"
  echo "      cd $out && gh repo create abeltre1/boxy --private --source=. --remote=origin --push"
fi
