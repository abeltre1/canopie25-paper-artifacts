"""--lb: containerized gateway (nginx/LiteLLM) fronting an agentless --replicas
pool — the AMD Instinct multi-node inference LB architecture (inference pool +
utility-node gateway), boxy edition: K single-instance agentless jobs with
pool-unique ports, gateway container on the login node over the ssh master."""

import argparse
import json

import pytest

from boxy import config, jobs, lb
from boxy.cli import UsageError, _serve_replicas_agentless, cmd_serve


# ---- pure renderers ----------------------------------------------------------


def test_nginx_conf_golden():
    conf = lb.nginx_conf([("node1", 8000), ("node2", 8001)], 9000)
    assert "least_conn;" in conf
    assert "server node1:8000 max_fails=3 fail_timeout=10s;" in conf
    assert "server node2:8001 max_fails=3 fail_timeout=10s;" in conf
    assert "listen 9000;" in conf
    # streaming-safe: SSE tokens must not be buffered, long generations must
    # not trip the read timeout
    assert "proxy_buffering off;" in conf
    assert "proxy_read_timeout 1h;" in conf
    assert "proxy_http_version 1.1;" in conf


def test_litellm_config_golden():
    y = lb.litellm_config("meta-llama/Llama-3.2-1B-Instruct", [("n1", 8000), ("n2", 8001)])
    assert y.count("model_name: meta-llama/Llama-3.2-1B-Instruct") == 2
    assert "api_base: http://n1:8000/v1" in y
    assert "api_base: http://n2:8001/v1" in y
    assert "routing_strategy: least-busy" in y


def test_gateway_container_cmd_nginx():
    cmd = lb.gateway_container_cmd("podman", "pool-lb", "/home/u/pool-lb.conf", "nginx", 9000)
    assert cmd.startswith("podman rm -f pool-lb")            # replace-on-rerun
    assert "podman run -d --rm --name pool-lb --network=host" in cmd
    assert "-v /home/u/pool-lb.conf:/etc/nginx/nginx.conf:ro" in cmd
    assert config.get_str("images.nginx_lb") in cmd


def test_gateway_container_cmd_litellm():
    cmd = lb.gateway_container_cmd("docker", "pool-lb", "/home/u/pool-lb.yaml", "litellm", 9000)
    assert "-v /home/u/pool-lb.yaml:/etc/litellm/config.yaml:ro" in cmd
    assert cmd.endswith("--config /etc/litellm/config.yaml --port 9000")
    assert config.get_str("images.litellm_lb") in cmd


# ---- CLI validation ----------------------------------------------------------


def _args(**over):
    base = dict(model="m", name=None, replicas=1, lb=None, ssh=None, dryrun=True)
    base.update(over)
    return argparse.Namespace(**base)


def test_lb_requires_replicas():
    with pytest.raises(UsageError, match="--replicas"):
        cmd_serve(_args(lb="nginx"))


def test_lb_requires_ssh(monkeypatch):
    monkeypatch.delenv("BOXY_SSH_HOST", raising=False)
    with pytest.raises(UsageError, match="--ssh"):
        cmd_serve(_args(lb="nginx", replicas=2))


# ---- pool orchestration (fake ssh + stubbed single-serve) --------------------


def _pool_args(**over):
    base = dict(model="meta-llama/Llama-3.2-1B-Instruct", name=None, scheduler="flux",
                replicas=2, lb="nginx", dryrun=False, ready_timeout=1.0,
                share=None, share_auto=False, port=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_pool_submits_replicas_and_starts_gateway(monkeypatch):
    from boxy import cli as _cli
    from boxy import remote as _remote

    launched: list[tuple[str, int]] = []
    ssh_cmds: list[str] = []
    pushed: dict[str, str] = {}

    def fake_single(per, target, follow=True):
        assert follow is False
        launched.append((per.name, per.port))
        ep = f"/remote/agentless/cluster/{per.name}.json"
        jobs.write_record(per.name, {"name": per.name, "submitted_from": "agentless-ssh",
                                     "target": target, "endpoint_remote": ep, "job": "1"})
        return 0

    def fake_capture(target, cmd, timeout=30):
        ssh_cmds.append(cmd)
        if "__BXEP__" in cmd:
            out = []
            for nm, port in launched:
                if nm in cmd:
                    out.append(f"__BXEP__{nm}")
                    out.append(json.dumps({"name": nm, "host": f"c{port}", "port": port}))
            return 0, "\n".join(out) + "\n"
        if cmd.startswith("command -v"):
            return 0, "/usr/bin/podman\n"
        return 0, "ok\n"

    monkeypatch.setattr(_cli, "_serve_agentless_ssh", fake_single)
    monkeypatch.setattr(_remote, "ssh_capture", fake_capture)
    monkeypatch.setattr(_remote, "push_file",
                        lambda t, path, content: pushed.__setitem__(path, content) or 0)

    rc = _serve_replicas_agentless(_pool_args(), "user@cluster", 2)
    assert rc == 0
    # replicas got pool-unique ports and -rN names
    base_port = config.get_int("network.replica_port_base")
    assert [p for _, p in launched] == [base_port, base_port + 1]
    assert [n.endswith("-r0") for n, _ in launched][0] and launched[1][0].endswith("-r1")
    # the nginx conf was pushed with BOTH live upstreams
    conf = next(iter(pushed.values()))
    assert f"server c{base_port}:{base_port} max_fails=3" in conf
    assert f"server c{base_port + 1}:{base_port + 1} max_fails=3" in conf
    # the gateway container was replace-run on the login node
    gw = [c for c in ssh_cmds if "run -d --rm --name" in c]
    assert gw and "--network=host" in gw[0] and config.get_str("images.nginx_lb") in gw[0]
    # the pool record ties it together for `boxy stop <base>`
    base = launched[0][0][: -len("-r0")]
    rec = jobs.read_record(base)
    assert rec and rec["submitted_from"] == "agentless-pool"
    assert rec["replicas"] == [n for n, _ in launched]
    assert rec["lb"]["kind"] == "nginx" and rec["lb"]["container"] == f"{base}-lb"


def test_pool_without_lb_prints_endpoints_only(monkeypatch):
    from boxy import cli as _cli
    from boxy import remote as _remote

    ssh_cmds: list[str] = []

    def fake_single(per, target, follow=True):
        ep = f"/remote/agentless/cluster/{per.name}.json"
        jobs.write_record(per.name, {"name": per.name, "submitted_from": "agentless-ssh",
                                     "target": target, "endpoint_remote": ep, "job": "1"})
        return 0

    def fake_capture(target, cmd, timeout=30):
        ssh_cmds.append(cmd)
        if "__BXEP__" in cmd:
            lines = []
            for tag in [t for t in cmd.split(";") if "echo __BXEP__" in t]:
                nm = tag.split("__BXEP__", 1)[1].strip()
                lines += [f"__BXEP__{nm}", json.dumps({"host": "n0", "port": 8000})]
            return 0, "\n".join(lines) + "\n"
        return 0, ""

    monkeypatch.setattr(_cli, "_serve_agentless_ssh", fake_single)
    monkeypatch.setattr(_remote, "ssh_capture", fake_capture)

    rc = _serve_replicas_agentless(_pool_args(lb=None), "user@cluster", 2)
    assert rc == 0
    # no gateway container was launched
    assert not any("run -d --rm" in c for c in ssh_cmds)
