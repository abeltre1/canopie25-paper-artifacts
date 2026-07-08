#!/usr/bin/env bash
# Install the Tier-2 naming authority (Headscale) on OpenShift.
#   SERVER_URL=https://headscale.apps.<cluster> BASE_DOMAIN=boxy.ts.net ./install.sh
set -euo pipefail
SERVER_URL="${SERVER_URL:?set SERVER_URL to the public Route host, e.g. https://headscale.apps.<cluster>}"
BASE_DOMAIN="${BASE_DOMAIN:-boxy.ts.net}"
NS="${NS:-headscale}"

oc new-project "$NS" 2>/dev/null || oc project "$NS"
helm upgrade --install headscale "$(dirname "$0")" \
  --namespace "$NS" \
  --set serverUrl="$SERVER_URL" \
  --set baseDomain="$BASE_DOMAIN" \
  ${PREAUTH_KEY:+--set preAuthKey="$PREAUTH_KEY"} \
  ${DERP_UDP:+--set derp.udp.enabled=true}

echo "Headscale installed in $NS. Mint a reusable pre-auth key with:"
echo "  oc -n $NS exec deploy/headscale -- headscale preauthkeys create --reusable --user boxy"
echo "Then enroll a client:  tailscale up --login-server $SERVER_URL --authkey <key>"
