#!/usr/bin/env bash
# Wormhole early-access preflight: answer "can I publish yet?" in one command,
# and produce a report that is safe to paste into a support ticket.
#
# Wormhole is deployed and operated by center administrators, so most early-user
# failures are environmental (not enabled, no token, site config missing) rather
# than anything the user did. This separates those cases instead of letting them
# all surface as one opaque error from `wh open`.
#
# Exit status: 0 = ready to publish, 1 = something is missing (details above).
#
# NEVER prints the token. It prints a SHA-256 prefix instead, which is enough to
# confirm two people are holding the same credential without disclosing it.

set -uo pipefail

CONFIG_SITE="/etc/wormhole/cli.toml"
CONFIG_USER="${XDG_CONFIG_HOME:-$HOME/.config}/wormhole/cli.toml"

pass=0
fail=0
warn=0

ok()   { printf '  [ ok ]   %s\n' "$*"; pass=$((pass + 1)); }
bad()  { printf '  [FAIL]   %s\n' "$*"; fail=$((fail + 1)); }
note() { printf '  [note]   %s\n' "$*"; warn=$((warn + 1)); }
hint() { printf '           -> %s\n' "$*"; }

# A credential fingerprint: irreversible, but stable, so a support engineer can
# ask "is your token fingerprint abcd1234?" without either side pasting a secret.
fingerprint() {
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$1" | sha256sum | cut -c1-8
    elif command -v shasum >/dev/null 2>&1; then
        printf '%s' "$1" | shasum -a 256 | cut -c1-8
    else
        printf 'unavailable'
    fi
}

echo "wormhole preflight — $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo
echo "client"

if wh_path=$(command -v wh 2>/dev/null); then
    ok "wh found at $wh_path"
    # --version is not documented in the CLI reference; probe it without letting a
    # non-zero exit or a missing flag look like a hard failure.
    if wh_ver=$(wh --version 2>&1 | head -1) && [ -n "$wh_ver" ]; then
        printf '           %s\n' "$wh_ver"
    fi
else
    bad "wh is not on PATH"
    hint "your site normally provides it; if not, build one: ./install-wh.sh"
    hint "(some sites install it as 'wormhole-cli' — symlink it to 'wh')"
fi

echo
echo "configuration"

if [ -r "$CONFIG_SITE" ]; then
    ok "site config present: $CONFIG_SITE"
else
    note "no site config at $CONFIG_SITE"
    hint "expected on a site-managed system; harmless if you set --endpoint yourself"
fi

if [ -r "$CONFIG_USER" ]; then
    ok "user config present: $CONFIG_USER"
else
    note "no user config at $CONFIG_USER (optional)"
    hint "copy cli.toml.example there to pin your endpoint and defaults"
fi

# The endpoint may legitimately come from a config file rather than the
# environment, so an unset variable is only worth reporting when NO config
# supplies one either.
endpoint="${WORMHOLE_ENDPOINT:-}"
if [ -n "$endpoint" ]; then
    ok "WORMHOLE_ENDPOINT is set ($endpoint)"
elif grep -qs '^[[:space:]]*endpoint[[:space:]]*=' "$CONFIG_USER" "$CONFIG_SITE" 2>/dev/null; then
    ok "endpoint comes from a config file"
else
    bad "no route-registry endpoint from the environment or any config file"
    hint "export WORMHOLE_ENDPOINT=https://<your-site-route-registry>"
fi

echo
echo "credential"

if [ -n "${WORMHOLE_TOKEN:-}" ]; then
    ok "WORMHOLE_TOKEN is set (${#WORMHOLE_TOKEN} chars, sha256:$(fingerprint "$WORMHOLE_TOKEN"))"
elif grep -qs '^[[:space:]]*token[[:space:]]*=' "$CONFIG_USER" "$CONFIG_SITE" 2>/dev/null; then
    ok "a token comes from a config file"
    note "prefer WORMHOLE_TOKEN in the environment over a token in a config file"
    hint "a file persists; check its mode is 600 if you keep it there"
else
    bad "no token from the environment or any config file"
    hint "export WORMHOLE_TOKEN=...   (never pass --token on the command line:"
    hint " argv is world-readable in ps on a shared login node)"
fi

# Warn about the specific mistake this whole kit is designed to prevent.
if pgrep -af 'wh .*--token' 2>/dev/null | grep -qv preflight; then
    bad "a running process has its token on the command line (visible in ps)"
    hint "stop it and use WORMHOLE_TOKEN instead"
fi

echo
echo "service reachability"

if command -v wh >/dev/null 2>&1 && { [ -n "${WORMHOLE_TOKEN:-}" ] || [ -r "$CONFIG_USER" ] || [ -r "$CONFIG_SITE" ]; }; then
    # `route list` is the cheapest call that exercises the whole chain:
    # endpoint resolution, TLS, and token authorization.
    if out=$(wh route list 2>&1); then
        ok "route registry answered — you are enabled"
        n=$(printf '%s\n' "$out" | grep -c . || true)
        printf '           %s line(s) of route output\n' "$n"
    else
        bad "route registry call failed"
        printf '%s\n' "$out" | sed 's/^/           | /' | head -10
        hint "early access is granted per user: if this says unauthorized, contact"
        hint " your center's support desk to be added, and quote the fingerprint above"
    fi
else
    note "skipped: needs wh plus a token or config"
fi

echo
printf 'summary: %d ok, %d failed, %d note(s)\n' "$pass" "$fail" "$warn"
if [ "$fail" -gt 0 ]; then
    echo "not ready to publish — fix the [FAIL] lines above"
    exit 1
fi
echo "ready: try ./smoke-test.sh --allowed-users \"\$USER\""
