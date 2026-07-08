#!/usr/bin/env bash
set -euo pipefail
NS="${NS:-headscale}"
helm uninstall headscale --namespace "$NS"
# the PVC has helm.sh/resource-policy: keep — delete it explicitly if you want the DB gone:
#   oc -n "$NS" delete pvc headscale-data
