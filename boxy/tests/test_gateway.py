"""The OpenSSH gateway exposer (RUNBOOK §0.994): no chisel, no tunnel binary —
a pod runs `ssh -L` to the login node. Pure emitters are golden-tested without a
cluster; the expose→share.json→unshare lifecycle is driven through a fake-oc
shim (the same pattern as test_share.py's relay coverage)."""

import pytest

from boxy.cli import main
from boxy.exposers import ShareContext, get_exposer
from boxy.exposers import gateway as gw
from boxy.exposers.base import ExposeError

APPS = "apps.goodall.sandia.gov"


# ---- registry ------------------------------------------------------------------


def test_gateway_is_registered_and_default_needs_no_laptop_binary():
    g = get_exposer("gateway")
    assert g.name == "gateway"
    assert g.binary == ""  # nothing on the laptop but oc


# ---- pure helpers --------------------------------------------------------------


def test_login_target_prefers_explicit_then_derives_service_user():
    # explicit BOXY_GW_LOGIN wins verbatim
    assert gw.login_target("svc@hops.example", "boxy", "ambelt@hops") == "svc@hops.example"
    # else: keep the login HOST from --ssh, swap in the service account
    assert gw.login_target("", "boxy-svc", "ambelt@hops.example") == "boxy-svc@hops.example"
    assert gw.login_target("", "boxy", "hops") == "boxy@hops"


def test_login_target_without_any_host_errors():
    with pytest.raises(ExposeError, match="cannot resolve the login node"):
        gw.login_target("", "boxy", "")


def test_ssh_command_pins_hostkey_and_keeps_key_off_argv():
    cmd = gw._ssh_command("boxy@hops", "hops18", 8090, 8090)
    assert "ssh -NT -i /tmp/gwkey" in cmd
    assert "StrictHostKeyChecking=yes" in cmd            # host key pinned (no MITM)
    assert "-L 0.0.0.0:8090:hops18:8090 boxy@hops" in cmd
    assert "while true" in cmd and "sleep 5" in cmd       # self-healing on drop
    # the private key material is a file mount, never inlined
    assert "BEGIN" not in cmd


# ---- golden manifests ----------------------------------------------------------


def test_share_manifest_shape_and_security():
    yaml = pytest.importorskip("yaml")
    text = gw.emit_share_manifest("nemotron", "nemotron-boxy." + APPS, "boxy@hops",
                                  "hops18", 8090, namespace="boxy-gw")
    docs = [d for d in yaml.safe_load_all(text) if d]
    assert [d["kind"] for d in docs] == ["Deployment", "Service", "Route"]
    dep, svc, route = docs
    # pod runs OpenSSH only, non-root, key mounted read-only from the Secret
    c = dep["spec"]["template"]["spec"]["containers"][0]
    assert c["command"][0] == "/bin/sh"
    assert "ssh -NT" in c["command"][2] and "hops18:8090" in c["command"][2]
    assert dep["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
    vol = dep["spec"]["template"]["spec"]["volumes"][0]
    assert vol["secret"]["secretName"] == "boxy-gw-ssh"
    assert vol["secret"]["defaultMode"] == 0o400
    # every object labelled for one-shot teardown
    for d in docs:
        assert d["metadata"]["labels"]["boxy.share"] == "nemotron"
    assert route["spec"]["host"] == "nemotron-boxy." + APPS
    assert route["spec"]["tls"]["termination"] == "edge"
    # NO chisel image, NO Docker Hub
    assert "chisel" not in text and "docker.io" not in text


def test_setup_manifest_emits_secret_networkpolicy_and_build_steps():
    yaml = pytest.importorskip("yaml")
    text = gw.emit_setup_manifest("boxy-svc@hops.sandia.gov")
    docs = [d for d in yaml.safe_load_all(text) if d]
    assert [d["kind"] for d in docs] == ["Secret", "NetworkPolicy"]
    np = docs[1]
    assert np["spec"]["egress"][0]["ports"][0]["port"] == 22   # only login-node egress
    # the header carries the non-YAML one-time steps
    assert "ubi9" in text and "openssh-clients" in text        # native image build
    assert "permitopen" in text and "restrict" in text          # locked-down authorized_keys
    assert "ssh-keyscan hops.sandia.gov" in text                # pin the host key


# ---- expose -> share.json -> unshare, through a fake oc ------------------------

OC_SHIM = r"""#!/bin/bash
echo "$*" >> "$OC_LOG"
case "$*" in
  *"get route boxy-gw-"*)   echo -n "${FAKE_OC_ADMITTED:-True}" ;;
  *"get deploy boxy-gw-"*)  echo -n "${FAKE_OC_READY:-1}" ;;
  apply*)                   cat > "$OC_APPLIED" ;;
  delete*)                  ;;
esac
exit 0
"""


@pytest.fixture
def gw_env(tmp_path, monkeypatch):
    oc = tmp_path / "oc"
    oc.write_text(OC_SHIM)
    oc.chmod(0o755)
    (tmp_path / "oc.log").write_text("")
    monkeypatch.setenv("BOXY_OC", str(oc))  # gateway resolves oc via relay.oc_bin() -> BOXY_OC
    monkeypatch.setenv("OC_LOG", str(tmp_path / "oc.log"))
    monkeypatch.setenv("OC_APPLIED", str(tmp_path / "applied.yaml"))
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv(gw.ENV_APPS_DOMAIN, APPS)
    monkeypatch.setenv(gw.ENV_LOGIN, "boxy-svc@hops")
    monkeypatch.setattr(gw, "ADMIT_TIMEOUT", 1.0)
    monkeypatch.setattr(gw, "ADMIT_POLL", 0.05)
    return tmp_path


def test_expose_full_lifecycle_and_unshare(gw_env):
    from boxy import jobs

    ctx = ShareContext(ssh_host="ambelt@hops", node="hops18", remote_port=8090)
    url, note = get_exposer("gateway").expose("nemotron", 8090, ctx)
    assert url == "https://nemotron-boxy." + APPS + "/v1"
    assert "boxy unshare nemotron" in note

    rec = jobs.read_share("nemotron")
    assert rec["exposer"] == "gateway"
    assert rec["node"] == "hops18" and rec["login"] == "boxy-svc@hops"

    # what actually got applied is a real Deployment aimed at the compute node
    applied = (gw_env / "applied.yaml").read_text()
    assert "kind: Deployment" in applied and "hops18:8090" in applied

    assert get_exposer("gateway").is_live(rec) is True
    get_exposer("gateway").unexpose("nemotron")
    assert jobs.read_share("nemotron") is None


def test_expose_requires_cluster_side_address(gw_env):
    # with oc present but no --ssh (no node/port), the gateway can't build the forward
    with pytest.raises(ExposeError, match="needs the cluster-side address"):
        get_exposer("gateway").expose("x", 8090, ShareContext(ssh_host="", node="", remote_port=0))


def test_expose_requires_apps_domain(gw_env, monkeypatch):
    monkeypatch.delenv(gw.ENV_APPS_DOMAIN, raising=False)
    ctx = ShareContext(ssh_host="ambelt@hops", node="hops18", remote_port=8090)
    with pytest.raises(ExposeError, match="BOXY_GW_APPS_DOMAIN"):
        get_exposer("gateway").expose("nemotron", 8090, ctx)


# ---- boxy generate gateway -----------------------------------------------------


def test_cli_generate_gateway(capsys):
    rc = main(["generate", "gateway", "--login", "boxy-svc@hops.sandia.gov"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "kind: NetworkPolicy" in out and "openssh-clients" in out
    assert "chisel" not in out


def test_cli_generate_gateway_requires_login(capsys):
    rc = main(["generate", "gateway"])
    assert rc == 2
    assert "--login is required" in capsys.readouterr().err
