#!/usr/bin/env bash
# deploy-relay.sh — one-shot: stand up the boxy chisel relay on OpenShift.
#
# Deploy ONCE per cluster. After this, `boxy ... --share <name>` publishes a
# model at https://<name>-boxy.apps.<cluster>/ that ANYONE on the corporate
# network reaches with NOTHING installed, and the sharing side runs the chisel
# client in a container (zero install there too).
#
# Usage:
#   deploy-relay.sh --host relay-boxy.apps.<cluster> [options]
#
# Options:
#   --host HOST         REQUIRED. The relay's Route hostname, under the cluster's
#                       *.apps wildcard (e.g. relay-boxy.apps.ocpcluster.example.com).
#   --namespace NS      OpenShift namespace (default: boxy-relay).
#   --image IMG         chisel image (default: boxy's images.relay). Point at your
#                       mirror / <registry>/user/chisel:1.10.1 for air-gapped sites.
#   --auth USER:PASS    Tunnel credential gating who may CREATE shares
#                       (default: boxy:<random>).
#   --dry-run           Print the manifest and the plan; apply nothing.
#
# Requires: boxy, oc (logged in), openssl.
set -euo pipefail

HOST="" NS="boxy-relay" IMAGE="" AUTH="" DRYRUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --host)      HOST="$2"; shift 2 ;;
    --namespace) NS="$2"; shift 2 ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --auth)      AUTH="$2"; shift 2 ;;
    --dry-run)   DRYRUN=1; shift ;;
    -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$HOST" ] || { echo "error: --host is required (e.g. --host relay-boxy.apps.<cluster>)" >&2; exit 2; }
# oc is only needed to APPLY; --dry-run just renders the manifest.
need=(boxy openssl); [ "$DRYRUN" = 1 ] || need+=(oc)
for bin in "${need[@]}"; do
  command -v "$bin" >/dev/null || { echo "error: '$bin' not found on PATH" >&2; exit 2; }
done

AUTH="${AUTH:-boxy:$(openssl rand -hex 16)}"
KEY_SEED="$(openssl rand -hex 16)"
IMG_ARGS=(); [ -n "$IMAGE" ] && IMG_ARGS=(--image "$IMAGE")

echo "### boxy relay deploy"
echo "    namespace : $NS"
echo "    route host: $HOST   ->  https://$HOST"
echo "    image     : ${IMAGE:-<boxy images.relay default>}"
echo "    auth      : ${AUTH%%:*}:********"
echo

if [ "$DRYRUN" = 1 ]; then
  echo "### DRY RUN — manifest that WOULD be applied:"
  boxy generate relay --host "$HOST" --namespace "$NS" --auth "$AUTH" --key-seed "$KEY_SEED" "${IMG_ARGS[@]}"
  echo
  echo "### would then: oc rollout status deploy/boxy-relay -n $NS; wait for Route admission"
  exit 0
fi

# namespace (idempotent)
oc get namespace "$NS" >/dev/null 2>&1 || oc create namespace "$NS"

# apply Secret + Deployment + Service + Route
echo "### applying relay manifest ..."
boxy generate relay --host "$HOST" --namespace "$NS" --auth "$AUTH" --key-seed "$KEY_SEED" "${IMG_ARGS[@]}" \
  | oc apply -n "$NS" -f -

echo "### waiting for the relay pod to roll out ..."
oc rollout status deploy/boxy-relay -n "$NS" --timeout=120s

echo "### waiting for the Route to be admitted ..."
for _ in $(seq 1 40); do
  admitted="$(oc get route boxy-relay -n "$NS" \
      -o 'jsonpath={.status.ingress[0].conditions[?(@.type=="Admitted")].status}' 2>/dev/null || true)"
  [ "$admitted" = "True" ] && break
  sleep 3
done
if [ "${admitted:-}" != "True" ]; then
  echo "warning: Route not admitted yet — check: oc describe route boxy-relay -n $NS" >&2
fi

echo
echo "### RELAY READY"
echo "    https://$HOST   (admitted: ${admitted:-unknown})"
echo
echo "### On the machine that will SHARE (Mac / HPC login node) — zero install:"
echo "    export BOXY_SHARE_ENABLED=1"
echo "    # boxy auto-discovers the relay URL + credential from your logged-in oc."
echo "    # No oc there? export them explicitly instead:"
echo "    export BOXY_RELAY_URL=https://$HOST"
echo "    export BOXY_RELAY_AUTH='$AUTH'"
[ -n "$IMAGE" ] && echo "    export BOXY_RELAY_IMAGE='$IMAGE'   # same image for the client container"
echo
echo "### Then share a served model:"
echo "    boxy serve <model> --ssh you@login-node --share demo"
echo "    #   ### SHARE   https://demo-boxy.${HOST#*.}/v1   (reachable by ANYONE on the network)"
echo
echo "### Verify readiness anytime:  boxy doctor | grep 'share relay'"
