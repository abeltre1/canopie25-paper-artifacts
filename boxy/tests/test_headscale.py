"""Tier-2 naming authority: `boxy generate headscale` emits the OpenShift
artifacts for a self-hosted Headscale (the MagicDNS control plane behind
`boxy open --publish`). Pure emitters, mirroring tests/test_router.py."""

import pytest

from boxy import headscale
from boxy.cli import main

SERVER = "https://headscale.apps.ocp.example.com"


def test_emit_values_has_openshift_route_and_magicdns():
    v = headscale.emit_values(SERVER, "boxy.ts.net")
    assert f"serverUrl: {SERVER}" in v and "baseDomain: boxy.ts.net" in v
    assert "magicDns: true" in v
    assert "termination: edge" in v                                 # works with headscale plain-HTTP :8080
    assert "timeout: 3600s" in v                                    # long-lived control conn
    assert "runAsNonRoot: true" in v
    assert "accessMode: ReadWriteOnce" in v                         # small RWO PVC for SQLite


def test_emit_manifest_is_valid_multidoc_yaml_with_expected_kinds():
    yaml = pytest.importorskip("yaml")
    docs = [d for d in yaml.safe_load_all(headscale.emit_manifest(SERVER, "boxy.ts.net", "hs")) if d]
    assert [d["kind"] for d in docs] == \
        ["ConfigMap", "Secret", "PersistentVolumeClaim", "Deployment", "Service", "Route"]
    route = next(d for d in docs if d["kind"] == "Route")
    assert route["spec"]["tls"]["termination"] == "edge"            # default that actually works
    assert route["spec"]["host"] == "headscale.apps.ocp.example.com"
    assert route["metadata"]["annotations"]["haproxy.router.openshift.io/timeout"] == "3600s"
    # reencrypt is still available for a TLS-serving backend
    r2 = [d for d in yaml.safe_load_all(
        headscale.emit_manifest(SERVER, "boxy.ts.net", termination="reencrypt")) if d]
    assert next(d for d in r2 if d["kind"] == "Route")["spec"]["tls"]["termination"] == "reencrypt"
    dep = next(d for d in docs if d["kind"] == "Deployment")
    assert dep["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
    cfg = next(d for d in docs if d["kind"] == "ConfigMap")["data"]["config.yaml"]
    assert "magic_dns: true" in cfg and "base_domain: boxy.ts.net" in cfg
    assert all(ns == "hs" for ns in (d["metadata"]["namespace"] for d in docs))


def test_emit_manifest_derp_udp_adds_loadbalancer():
    yaml = pytest.importorskip("yaml")
    docs = [d for d in yaml.safe_load_all(headscale.emit_manifest(SERVER, "t.net", derp_udp=True)) if d]
    lb = [d for d in docs if d["kind"] == "Service" and d["spec"].get("type") == "LoadBalancer"]
    assert lb and lb[0]["spec"]["ports"][0]["protocol"] == "UDP"       # STUN/3478 opt-in
    # default (no derp_udp) has no LoadBalancer -> relays over the :443 Route
    docs2 = [d for d in yaml.safe_load_all(headscale.emit_manifest(SERVER, "t.net")) if d]
    assert not any(d.get("spec", {}).get("type") == "LoadBalancer" for d in docs2)


def test_cli_generate_headscale_manifest_default(capsys):
    rc = main(["generate", "headscale", "--server-url", SERVER, "--base-domain", "boxy.ts.net"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kind: Route" in out and "termination: edge" in out and "oc apply -f -" in out


def test_cli_generate_headscale_requires_server_url(capsys):
    rc = main(["generate", "headscale"])
    assert rc == 2
    assert "--server-url is required" in capsys.readouterr().err


def test_cli_generate_sky_still_requires_box_location(capsys):
    # relaxing required= for headscale must not let sky/slurm run without box+location
    rc = main(["generate", "sky"])
    assert rc == 2
    assert "--box and --location are required" in capsys.readouterr().err
