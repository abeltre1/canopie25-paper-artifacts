#!/usr/bin/env bash
# Build the Wormhole CLI from source into ~/.local/bin/wh — user level, no sudo.
#
# On a site-managed system you should NOT need this: wormhole-cli's own README
# says the binary is "expected to be distributed as a user-facing `wh` binary on
# the target clusters". Use this when you are on a system the site has not set
# up yet, or you want a newer build than the one installed.
#
# There are no released binaries and no `go install` line upstream, so building
# from the repository is the only path.
#
# Usage:
#   ./install-wh.sh                 # build and install to ~/.local/bin/wh
#   ./install-wh.sh --prefix ~/opt  # install to ~/opt/bin/wh
#   ./install-wh.sh --ref v1.2.3    # build a specific tag/branch/commit

set -euo pipefail

REPO="https://github.com/LLNL/wormhole-cli"
prefix="$HOME/.local"
ref=""

die() { printf 'install-wh: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) prefix="${2:?--prefix needs a path}"; shift 2 ;;
        --ref)    ref="${2:?--ref needs a git ref}"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

command -v go  >/dev/null 2>&1 || die "go is not on PATH — wormhole-cli is written in Go.
    On an HPC system try: module avail go && module load go"
command -v git >/dev/null 2>&1 || die "git is not on PATH"

echo "### go: $(go version)"

# Build in a temp tree so a failed build leaves nothing behind, and so this is
# safe to re-run.
src=$(mktemp -d)
trap 'rm -rf "$src"' EXIT INT TERM

echo "### cloning $REPO"
if [ -n "$ref" ]; then
    git clone --quiet --depth 1 --branch "$ref" "$REPO" "$src/wormhole-cli" ||
        die "clone failed. On a filtered-egress system set https_proxy, or clone on a
    machine with egress and copy the tree here, then run: cd <tree> && make build"
else
    git clone --quiet --depth 1 "$REPO" "$src/wormhole-cli" ||
        die "clone failed. On a filtered-egress system set https_proxy, or clone on a
    machine with egress and copy the tree here, then run: cd <tree> && make build"
fi

cd "$src/wormhole-cli"
echo "### building (make build)"
# The Makefile emits binaries/wormhole-cli; sites rename or symlink it to `wh`.
# Fall back to a direct `go build` if the Makefile target ever moves.
if ! make build; then
    echo "### make build failed; falling back to: go build -o wh ."
    go build -o wh .
fi

built=""
for candidate in binaries/wormhole-cli wh binaries/wh; do
    if [ -x "$candidate" ]; then built="$candidate"; break; fi
done
[ -n "$built" ] || die "build produced no binary (looked for binaries/wormhole-cli, wh)"

mkdir -p "$prefix/bin"
install -m 0755 "$built" "$prefix/bin/wh"
echo "### installed $prefix/bin/wh"

case ":$PATH:" in
    *":$prefix/bin:"*) ;;
    *) echo "### NOTE: $prefix/bin is not on your PATH. Add it:"
       echo "    export PATH=\"$prefix/bin:\$PATH\"" ;;
esac

echo
echo "next: configure an endpoint and token, then run ./preflight.sh"
