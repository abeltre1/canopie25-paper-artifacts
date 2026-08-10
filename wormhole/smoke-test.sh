#!/usr/bin/env bash
# Publish a trivial web service through Wormhole and prove the URL serves it.
#
# The point of this script is the SPLIT: it verifies the app is reachable on
# localhost BEFORE involving Wormhole at all. When something fails you then know
# which half broke — a working local check plus a failing Wormhole URL is a
# platform/authorization problem, not your application.
#
# Usage:
#   ./smoke-test.sh --allowed-users "$USER"
#   ./smoke-test.sh --allowed-groups project-a --name my-test-app
#
# Requires WORMHOLE_TOKEN in the environment (never pass a token as an argument;
# argv is world-readable in ps on a shared login node).

set -euo pipefail

name=""
app_port=""
allowed_users=""
allowed_groups=""

die() { printf 'smoke-test: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --name)           name="${2:?--name needs a value}"; shift 2 ;;
        --app-port)       app_port="${2:?--app-port needs a value}"; shift 2 ;;
        --allowed-users)  allowed_users="${2:?--allowed-users needs a value}"; shift 2 ;;
        --allowed-groups) allowed_groups="${2:?--allowed-groups needs a value}"; shift 2 ;;
        -h|--help)        sed -n '2,14p' "$0"; exit 0 ;;
        *)                die "unknown argument: $1" ;;
    esac
done

# Wormhole requires at least one allowed user or group. Refuse to guess one:
# publishing a service to the wrong audience is not a convenient default.
[ -n "$allowed_users$allowed_groups" ] ||
    die "pass --allowed-users and/or --allowed-groups (Wormhole requires at least one;
     this script will not pick a default — that decides who can reach your service)"

[ -n "${WORMHOLE_TOKEN:-}" ] ||
    die "WORMHOLE_TOKEN is not set. export it first; run ./preflight.sh to check the rest"

command -v wh >/dev/null 2>&1 || die "wh is not on PATH (see ./install-wh.sh)"

# A per-user default name keeps two people smoke-testing at once from colliding
# on the same route, which would otherwise look like a permissions bug.
name="${name:-boxy-smoke-${USER:-anon}}"

# Ask the kernel for a free port rather than hoping a hardcoded one is idle.
if [ -z "$app_port" ]; then
    app_port=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
fi

workdir=$(mktemp -d)
marker="wormhole-smoke-ok-$$"
printf '%s\n' "$marker" > "$workdir/index.html"

http_pid=""
cleanup() {
    [ -n "$http_pid" ] && kill "$http_pid" 2>/dev/null || true
    rm -rf "$workdir"
}
trap cleanup EXIT INT TERM

echo "### 1/3  serving a marker file locally on port $app_port"
python3 -m http.server "$app_port" --bind 127.0.0.1 --directory "$workdir" >/dev/null 2>&1 &
http_pid=$!

for _ in $(seq 1 40); do
    if curl -fsS "http://127.0.0.1:$app_port/" >/dev/null 2>&1; then break; fi
    sleep 0.25
done

got=$(curl -fsS "http://127.0.0.1:$app_port/" 2>/dev/null || true)
if [ "$got" != "$marker" ]; then
    die "the LOCAL service never came up on port $app_port — this is not a Wormhole
     problem. Check that python3 can bind that port here."
fi
echo "     local check passed (served the marker over 127.0.0.1)"

echo
echo "### 2/3  publishing it through Wormhole as '$name'"
set -- open --name "$name" --app-port "$app_port"
if [ -n "$allowed_users" ]; then
    set -- "$@" --allowed-users "$allowed_users"
fi
if [ -n "$allowed_groups" ]; then
    set -- "$@" --allowed-groups "$allowed_groups"
fi

# The token is read from WORMHOLE_TOKEN in the environment; it is deliberately
# absent from this argv so the printed command is safe to copy into a ticket.
echo "     wh $*"
echo
echo "### 3/3  when the URL appears, verify it from a browser or another shell:"
echo "     curl -fsS <the-wormhole-url>/     # expect: $marker"
echo
echo "     Ctrl-C here tears down both the route and the local server."
echo

# Deliberately NOT exec: this shell has to survive so the EXIT trap can stop the
# background http.server. exec would replace it and orphan that process.
wh "$@"
