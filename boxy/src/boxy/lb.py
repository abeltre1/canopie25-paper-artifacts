"""Gateway (load balancer) in front of a --replicas pool.

The AMD Instinct multi-node inference LB architecture: an inference POOL of
independent engine servers (one per replica, tensor-parallel within its node)
fronted by a battle-tested API GATEWAY on a utility node — nginx (reverse
proxy, passive health checks / failover) or LiteLLM (model-aware routing).
boxy places the gateway on the LOGIN node: it reaches the compute fabric (the
same route the readiness probe uses), runs containerized under the user's
podman/docker, and lives outside any single job's allocation so replica jobs
can come and go beneath it.

Only PURE renderers live here (golden-testable); the orchestration — waiting
for replica endpoints, pushing the config, running the container — is driven
from cli.py's pool branch over the existing SSH master.
"""

from __future__ import annotations

from boxy import config


def nginx_conf(upstreams: list[tuple[str, int]], listen_port: int) -> str:
    """A complete nginx.conf fronting the pool: least_conn balancing across the
    replica endpoints with passive health checks (max_fails/fail_timeout evicts
    a dead replica for a cooldown, so a lost node degrades throughput instead
    of erroring 1/K of requests), and streaming-safe proxying — buffering OFF
    (SSE token streams must not be batched) with generous read timeouts (long
    generations idle the upstream socket between tokens)."""
    servers = "\n".join(
        f"        server {host}:{port} max_fails=3 fail_timeout=10s;"
        for host, port in upstreams
    )
    return f"""\
worker_processes auto;
events {{ worker_connections 4096; }}
http {{
    upstream boxy_pool {{
        least_conn;
{servers}
    }}
    server {{
        listen {listen_port};
        location / {{
            proxy_pass http://boxy_pool;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_buffering off;
            proxy_request_buffering off;
            proxy_read_timeout 1h;
            proxy_send_timeout 1h;
            client_max_body_size 0;
        }}
    }}
}}
"""


def litellm_config(model: str, upstreams: list[tuple[str, int]]) -> str:
    """A LiteLLM proxy config fronting the pool: every replica is a deployment
    of the SAME model_name, so LiteLLM's router load-balances across them
    (least-busy) with health checks — the model-aware alternative to nginx."""
    entries = "\n".join(
        f"""\
  - model_name: {model}
    litellm_params:
      model: hosted_vllm/{model}
      api_base: http://{host}:{port}/v1"""
        for host, port in upstreams
    )
    return f"""\
model_list:
{entries}
router_settings:
  routing_strategy: least-busy
  num_retries: 2
  allowed_fails: 3
  cooldown_time: 10
general_settings:
  master_key: null
"""


def gateway_container_cmd(runtime: str, name: str, conf_remote: str, kind: str,
                          listen_port: int) -> str:
    """The one-line shell command that (re)starts the gateway container on the
    login node: replace-on-rerun (rm -f first), host networking (the gateway
    must reach compute-node endpoints on the cluster fabric), config mounted
    read-only, detached. Returns a shell string for `ssh target '<cmd>'`."""
    if kind == "nginx":
        image = config.get_str("images.nginx_lb")
        mount = f"-v {conf_remote}:/etc/nginx/nginx.conf:ro"
        inner = ""
    else:  # litellm
        image = config.get_str("images.litellm_lb")
        mount = f"-v {conf_remote}:/etc/litellm/config.yaml:ro"
        inner = f" --config /etc/litellm/config.yaml --port {listen_port}"
    return (f"{runtime} rm -f {name} >/dev/null 2>&1; "
            f"{runtime} run -d --rm --name {name} --network=host {mount} {image}{inner}")
