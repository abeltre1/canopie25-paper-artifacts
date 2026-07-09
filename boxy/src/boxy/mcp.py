"""Emit deployment artifacts for MCP (Model Context Protocol) servers as
persistent services — starting with flux-mcp (agentic Flux job control:
https://github.com/converged-computing/flux-mcp). flux-mcp serves MCP over
HTTP/SSE on :8089 and needs to reach a Flux instance (FLUX_URI); the OpenShift
manifest here runs the published image and exposes it behind a Route.

Pure string builders (like router.emit_nginx) — no I/O, no pyyaml dependency.
On HPC (where Flux actually lives) run it as a normal container via the scheduler
and reach it with a tunnel (`boxy open … --ssh <login>`); this manifest is the
OpenShift persistent-service form for a cluster that can reach Flux over FLUX_URI.
"""

from __future__ import annotations

FLUX_MCP_IMAGE = "ghcr.io/converged-computing/flux-mcp:latest"
FLUX_MCP_PORT = 8089  # FastMCP HTTP/SSE transport


def emit_flux_mcp_manifest(host: str, namespace: str = "flux-mcp",
                           *, image: str = FLUX_MCP_IMAGE, flux_uri: str = "",
                           port: int = FLUX_MCP_PORT) -> str:
    """A self-contained OpenShift manifest (Deployment + Service + Route) running
    flux-mcp's FastMCP server. `host` is the Route hostname; `flux_uri` (optional)
    is exported as FLUX_URI so the server can reach a remote Flux instance."""
    env_block = ""
    if flux_uri:
        env_block = f"""
          env:
            - name: FLUX_URI
              value: {flux_uri}"""
    docs = [
        f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: flux-mcp
  namespace: {namespace}
  labels: {{app: flux-mcp}}
spec:
  replicas: 1
  selector:
    matchLabels: {{app: flux-mcp}}
  template:
    metadata:
      labels: {{app: flux-mcp}}
    spec:
      securityContext:
        runAsNonRoot: true
      containers:
        - name: flux-mcp
          image: {image}
          command: ["python3", "-m", "flux_mcp.server.fastmcp"]
          ports:
            - {{containerPort: {port}, name: mcp}}{env_block}""",
        f"""apiVersion: v1
kind: Service
metadata:
  name: flux-mcp
  namespace: {namespace}
spec:
  selector: {{app: flux-mcp}}
  ports:
    - {{name: mcp, port: {port}, targetPort: {port}}}""",
        f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: flux-mcp
  namespace: {namespace}
  annotations:
    haproxy.router.openshift.io/timeout: "3600s"
spec:
  host: {host}
  to: {{kind: Service, name: flux-mcp}}
  port: {{targetPort: mcp}}
  tls: {{termination: edge}}""",
    ]
    header = (f"# flux-mcp (MCP server for Flux) as a persistent OpenShift service.\n"
              f"# apply:  boxy generate flux-mcp --host {host} | oc apply -f -\n"
              f"# MCP endpoint (HTTP/SSE): https://{host}/  (agents connect here)\n")
    return header + "\n---\n".join(docs) + "\n"
