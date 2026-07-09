"""Pluggable exposers (RUNBOOK §0.993): the registry mirrors backends/schedulers,
and the relay's OpenShift manifests are pure golden-testable strings — no chisel,
no oc, no cluster."""

import random

import pytest

from boxy.cli import main
from boxy.exposers import EXPOSERS, ExposeError, get_exposer
from boxy.exposers import relay as relay_mod
from boxy.exposers.hosts import HostsExposer

RELAY_URL = "https://relay-boxy.apps.goodall.sandia.gov"


# ---- registry (same contract as backends/schedulers) ---------------------------


def test_registry_members_and_factory():
    assert set(EXPOSERS) == {"relay", "hosts"}
    assert get_exposer("relay").name == "relay"
    assert get_exposer("hosts").name == "hosts"


def test_registry_unknown_name_error_shape():
    with pytest.raises(ValueError, match=r"unknown exposer 'nope' \(available: relay, hosts\)"):
        get_exposer("nope")


def test_hosts_member_is_always_available_and_local_only():
    h = HostsExposer()
    assert h.available()
    url, note = h.expose("mymodel", 8090)
    assert url == "http://mymodel:8090/v1"
    assert "/etc/hosts" in note and "THIS machine" in note
    h.unexpose("mymodel")  # no-op, must not raise


# ---- pure helpers ----------------------------------------------------------------


def test_apps_domain_strips_relay_label():
    assert relay_mod.apps_domain_from_url(RELAY_URL) == "apps.goodall.sandia.gov"
    assert relay_mod.apps_domain_from_url("relay.apps.x.y/") == "apps.x.y"
    with pytest.raises(ExposeError):
        relay_mod.apps_domain_from_url("https://nodomain")


def test_share_host_suffix_and_alias_validation():
    assert relay_mod.share_host("nemotron", "apps.x.y") == "nemotron-boxy.apps.x.y"
    assert relay_mod.share_host("a", "apps.x.y") == "a-boxy.apps.x.y"
    for bad in ("Bad_Name!", "-lead", "trail-", "UPPER", "a" * 41):
        with pytest.raises(ExposeError):
            relay_mod.share_host(bad, "apps.x.y")


def test_pick_relay_port_range_and_exhaustion():
    rand = random.Random(7)
    p = relay_mod.pick_relay_port({31000, 31001}, rand)
    assert p in relay_mod.RELAY_PORT_RANGE and p not in {31000, 31001}
    with pytest.raises(ExposeError, match="all .* relay ports are taken"):
        relay_mod.pick_relay_port(set(relay_mod.RELAY_PORT_RANGE), rand)


# ---- golden manifests -------------------------------------------------------------


def test_relay_manifest_shape_and_security():
    yaml = pytest.importorskip("yaml")
    docs = [d for d in yaml.safe_load_all(
        relay_mod.emit_relay_manifest("relay-boxy.apps.goodall.sandia.gov", "boxy-relay",
                                      auth="boxy:s3cret", key_seed="seed1")) if d]
    assert [d["kind"] for d in docs] == ["Secret", "Deployment", "Service", "Route"]
    c = docs[1]["spec"]["template"]["spec"]["containers"][0]
    assert c["args"][:2] == ["server", "--reverse"]
    # the credential travels via secretKeyRef env — NEVER argv
    assert "s3cret" not in " ".join(c["args"])
    assert c["env"][0]["name"] == "AUTH"
    assert c["env"][0]["valueFrom"]["secretKeyRef"] == {"name": "boxy-relay", "key": "auth"}
    assert docs[1]["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
    route = docs[3]
    assert route["spec"]["tls"]["termination"] == "edge"
    ann = route["metadata"]["annotations"]
    # BOTH websocket timers: `timeout` pre-upgrade, `timeout-tunnel` live tunnel
    assert ann["haproxy.router.openshift.io/timeout"] == "3600s"
    assert ann["haproxy.router.openshift.io/timeout-tunnel"] == "3600s"
    assert docs[0]["stringData"]["auth"] == "boxy:s3cret"


def test_relay_manifest_placeholder_auth_hint():
    text = relay_mod.emit_relay_manifest("relay-boxy.apps.x.y")
    assert "REPLACE_ME:REPLACE_ME" in text and "create secret generic" in text


def test_share_manifest_shape():
    yaml = pytest.importorskip("yaml")
    docs = [d for d in yaml.safe_load_all(
        relay_mod.emit_share_manifest("nemotron", "nemotron-boxy.apps.x.y", 31234, "boxy-relay")) if d]
    assert [d["kind"] for d in docs] == ["Service", "Route"]
    svc, route = docs
    assert svc["spec"]["selector"] == {"app": "boxy-relay"}
    assert svc["spec"]["ports"][0]["targetPort"] == 31234
    assert svc["metadata"]["labels"]["boxy.share"] == "nemotron"
    assert route["spec"]["host"] == "nemotron-boxy.apps.x.y"
    assert route["metadata"]["labels"]["boxy.share"] == "nemotron"
    assert route["metadata"]["annotations"]["haproxy.router.openshift.io/timeout-tunnel"] == "3600s"


# ---- boxy generate relay -----------------------------------------------------------


def test_cli_generate_relay(capsys):
    rc = main(["generate", "relay", "--host", "relay-boxy.apps.goodall.sandia.gov",
               "--auth", "boxy:pw"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kind: Deployment" in out and "--reverse" in out and "oc apply -f -" in out
    assert "boxy:pw" in out                                # lands in the Secret stringData
    assert "brew install chisel" in out                    # laptop-side hint in the header


def test_cli_generate_relay_requires_host(capsys):
    rc = main(["generate", "relay"])
    assert rc == 2
    assert "--host is required" in capsys.readouterr().err
