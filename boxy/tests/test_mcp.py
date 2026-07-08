"""Persistent MCP services: `boxy generate flux-mcp` emits the Flux MCP server
(https://github.com/converged-computing/flux-mcp) as an OpenShift Deployment +
Service + Route. Pure emitter, mirroring tests/test_headscale.py."""

import pytest

from boxy import mcp
from boxy.cli import main

HOST = "flux-mcp.apps.ocp.example.com"


def test_emit_flux_mcp_manifest_shape():
    yaml = pytest.importorskip("yaml")
    docs = [d for d in yaml.safe_load_all(mcp.emit_flux_mcp_manifest(HOST, "fm")) if d]
    assert [d["kind"] for d in docs] == ["Deployment", "Service", "Route"]
    c = docs[0]["spec"]["template"]["spec"]["containers"][0]
    assert c["command"] == ["python3", "-m", "flux_mcp.server.fastmcp"]
    assert c["ports"][0]["containerPort"] == mcp.FLUX_MCP_PORT == 8089
    assert c["image"] == mcp.FLUX_MCP_IMAGE
    assert docs[1]["spec"]["ports"][0]["port"] == 8089
    assert docs[2]["spec"]["host"] == HOST and docs[2]["spec"]["tls"]["termination"] == "edge"
    assert all(d["metadata"]["namespace"] == "fm" for d in docs)
    # runAsNonRoot for the restricted-v2 SCC
    assert docs[0]["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True


def test_emit_flux_mcp_flux_uri_env_optional():
    yaml = pytest.importorskip("yaml")
    with_uri = [d for d in yaml.safe_load_all(
        mcp.emit_flux_mcp_manifest(HOST, flux_uri="ssh://eldo/run/flux/local")) if d]
    env = with_uri[0]["spec"]["template"]["spec"]["containers"][0]["env"]
    assert env == [{"name": "FLUX_URI", "value": "ssh://eldo/run/flux/local"}]
    # omitted -> no env block
    without = [d for d in yaml.safe_load_all(mcp.emit_flux_mcp_manifest(HOST)) if d]
    assert "env" not in without[0]["spec"]["template"]["spec"]["containers"][0]


def test_cli_generate_flux_mcp(capsys):
    rc = main(["generate", "flux-mcp", "--host", HOST])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kind: Deployment" in out and "flux_mcp.server.fastmcp" in out and "oc apply -f -" in out


def test_cli_generate_flux_mcp_requires_host(capsys):
    rc = main(["generate", "flux-mcp"])
    assert rc == 2
    assert "--host is required" in capsys.readouterr().err
