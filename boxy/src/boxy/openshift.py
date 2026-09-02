"""Emit an OpenShift manifest that runs the MODEL SERVER itself.

boxy already talks to OpenShift for auxiliary infrastructure — the chisel relay
(exposers/relay.py) and flux-mcp (mcp.py) — but the inference workload always ran
on a login or compute node under podman/docker/apptainer. This module is the
missing half: a Deployment + Service that serve a model INSIDE an OpenShift
project, sized from the same model cards the HPC path uses.

Pure string builders (like mcp.emit_flux_mcp_manifest and router.emit_nginx): no
I/O, no cluster, no pyyaml dependency, so every manifest is golden-testable.

USER LEVEL, deliberately. Everything emitted is namespace-scoped and creatable by
an ordinary developer with `oc apply` in their own project:

  * no cluster-scoped objects (no ClusterRole, no SCC, no operator, no CRD)
  * no privileged / hostPath / hostNetwork / hostPID, no NET_ADMIN
  * no runAsUser: OpenShift's default restricted-v2 SCC assigns the namespace's
    own UID range, and pinning a UID is what gets a Deployment REJECTED there
  * `namespace:` is omitted unless asked for, so the manifest lands in whatever
    project the user is currently on (`oc project`) rather than a hardcoded one

Three OpenShift-specific failure modes are designed out here, because each one
costs an afternoon to diagnose from inside a CrashLoopBackOff:

  1. Random UID, unwritable HOME. restricted-v2 runs the container as an
     arbitrary high UID that owns nothing in the image, so a vLLM image whose
     HOME is /root cannot write its cache and dies during startup on a
     permission error that names no path. HOME and the HF/XDG caches are pointed
     at a writable emptyDir.
  2. 64MiB /dev/shm. Kubernetes' default shm is 64MiB; vLLM's tensor-parallel
     workers communicate through shared memory and hang or abort once past one
     GPU. A Memory-backed emptyDir at /dev/shm replaces it.
  3. Credentials in the manifest. A manifest gets committed, pasted and shared,
     so the HF token is referenced from a Secret the user creates separately and
     never appears in the emitted YAML.
"""

from __future__ import annotations

import re

# Device-plugin resource names. Requesting these needs no admin: the GPU operator
# advertises them and any pod in any namespace may request them, which is what
# keeps GPU serving inside the user-level envelope.
GPU_RESOURCES = {
    "cuda": "nvidia.com/gpu",
    "rocm": "amd.com/gpu",
    "intel": "gpu.intel.com/i915",
}

# vLLM's tensor-parallel workers talk over /dev/shm. Kubernetes gives a pod 64MiB
# by default, which is enough for one GPU and nothing more.
DEFAULT_SHM_SIZE = "8Gi"

# Written into the pod so the engine's caches land somewhere the assigned UID can
# actually write. Keys are env vars; values are paths under the scratch volume.
_CACHE_ENV = {
    "HOME": "/scratch",
    "HF_HOME": "/scratch/huggingface",
    "XDG_CACHE_HOME": "/scratch/.cache",
    "TRITON_CACHE_DIR": "/scratch/.triton",
    "VLLM_CACHE_ROOT": "/scratch/.vllm",
}


# A Service name must be an RFC 1123 LABEL, which forbids dots and uppercase —
# stricter than the subdomain rule a Deployment name follows. Both objects here
# share one name, so the label rule governs. Getting this wrong surfaces as an
# apply-time rejection, not a boxy error, so validate before emitting.
_RFC1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def k8s_name(model: str, prefix: str = "boxy") -> str:
    """A valid object name derived from a model id: dots and case are the two
    things that bite ('Llama-3.3-70B' -> 'boxy-llama-3-3-70b')."""
    base = model.rstrip("/").rsplit("/", 1)[-1]
    base = re.sub(r"\.(gguf|safetensors)$", "", base, flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-") or "model"
    return f"{prefix}-{slug}"[:63].strip("-")


def _yaml_str(value: str) -> str:
    """Quote a scalar so a value containing ':' or '#' cannot break the document."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def serve_command(engine: str, model: str, port: int, engine_args=(), *,
                  gpus: int = 0, served_name: str = "") -> list[str]:
    """The argv the container runs — built by the SAME engines.py builders the
    HPC path uses, not hand-composed.

    Hand-composing drifted from the real thing in ways that silently break a
    manifest: no --tensor-parallel-size, so a Deployment requesting N GPUs
    loaded the whole model onto GPU 0 and OOM'd; `llama-server` as an absolute
    command, which does not exist on $PATH in the upstream image (its binary is
    /app/llama-server, so the image's own ENTRYPOINT must be used); and no
    --served-model-name, so a PVC-staged model answered to its mount path.
    Going through the builders means every default the field taught boxy —
    eager safetensors on network filesystems included — lands here too.

    An empty FIRST element means "defer to the image's ENTRYPOINT" (llama.cpp);
    the caller turns that into k8s `args:` with no `command:`."""
    from boxy import engines
    from boxy.box import Box
    from boxy.location import Location, Resources, Staging

    box = Box(name="serve", image="", engine=("vllm" if engine == "vllm" else "llama.cpp"),
              entrypoint=("vllm" if engine == "vllm" else ""), ports=[port],
              model=model, served_name=served_name)
    loc = Location(name="openshift", scheduler="none", accelerator="cuda", runtime="podman",
                   resources=Resources(nodes=1, gpus_per_node=max(1, int(gpus or 0))),
                   staging=Staging(models_dir="."))
    return engines.build_serve_cmd(box, loc, model, host="0.0.0.0", port=port,
                                   extra_args=[str(a) for a in engine_args])


def emit_serve_manifest(
    name: str,
    image: str,
    model: str,
    *,
    namespace: str = "",
    engine: str = "vllm",
    port: int = 8000,
    gpus: int = 0,
    accelerator: str = "cuda",
    engine_args=(),
    model_pvc: str = "",
    model_mount: str = "/models",
    served_name: str = "",
    hf_secret: str = "",
    replicas: int = 1,
    cpu: str = "",
    memory: str = "",
    shm_size: str = DEFAULT_SHM_SIZE,
    route_host: str = "",
) -> str:
    """A self-contained manifest (Deployment + Service [+ Route]) serving `model`.

    `model` is the path the ENGINE opens — a path under `model_mount` when
    `model_pvc` carries staged weights, or a Hugging Face id when the pod should
    download them (which needs `hf_secret` for gated repos and egress).

    A Route is emitted only when `route_host` is given. Without one the service
    stays inside the cluster and is reached with `oc port-forward`, which is the
    lowest-privilege option and the only one guaranteed to work for a user who
    cannot create Routes.
    """
    if not name:
        raise ValueError("a deployment name is required")
    if not _RFC1123_LABEL.match(name):
        raise ValueError(
            f"{name!r} is not a valid Kubernetes object name: it must be lowercase "
            f"alphanumeric with '-' (no dots, no underscores, no capitals). "
            f"Try {k8s_name(name)!r}")
    if not image:
        raise ValueError("an image is required (boxy resolves one from the model card)")
    if not model:
        raise ValueError("a model path or id is required")

    ns = f"\n  namespace: {namespace}" if namespace else ""
    labels = f"{{app: {name}, boxy.serve: {name}}}"

    # --- container command -----------------------------------------------------
    argv = serve_command(engine, model, port, engine_args, gpus=gpus, served_name=served_name)
    # An empty first element means the image's own ENTRYPOINT runs the server
    # (the upstream llama.cpp image keeps its binary off $PATH at
    # /app/llama-server, so naming it as `command:` makes the pod CrashLoop).
    # k8s spells that as args-without-command.
    _key = "command" if argv[0] else "args"
    cmd_yaml = f"{_key}: [" + ", ".join(_yaml_str(a) for a in argv if a) + "]"

    # --- environment -----------------------------------------------------------
    env_lines = [f"            - {{name: {k}, value: {_yaml_str(v)}}}" for k, v in _CACHE_ENV.items()]
    if hf_secret:
        # secretKeyRef, never a literal: this manifest is meant to be committed
        # and pasted. Create the secret separately (the header shows how).
        env_lines.append(
            f"            - name: HF_TOKEN\n"
            f"              valueFrom:\n"
            f"                secretKeyRef: {{name: {hf_secret}, key: token}}")
    env_block = "\n          env:\n" + "\n".join(env_lines)

    # --- resources -------------------------------------------------------------
    limits: list[str] = []
    if gpus:
        resource = GPU_RESOURCES.get(accelerator)
        if not resource:
            raise ValueError(
                f"no Kubernetes device-plugin resource is known for accelerator "
                f"{accelerator!r} (known: {', '.join(sorted(GPU_RESOURCES))}); "
                f"pass --gpus 0 to serve on CPU")
        # GPUs are only ever a LIMIT: the device plugin makes them non-shareable,
        # and setting a request as well is redundant (Kubernetes copies it).
        limits.append(f'{_yaml_str(resource)}: {gpus}')
    if cpu:
        limits.append(f"cpu: {_yaml_str(cpu)}")
    if memory:
        limits.append(f"memory: {_yaml_str(memory)}")
    res_block = ""
    if limits:
        res_block = ("\n          resources:\n            limits:\n"
                     + "\n".join(f"              {line}" for line in limits))

    # --- volumes ---------------------------------------------------------------
    mounts = [
        "            - {name: scratch, mountPath: /scratch}",
        "            - {name: dshm, mountPath: /dev/shm}",
    ]
    volumes = [
        "        - {name: scratch, emptyDir: {}}",
        f"        - name: dshm\n          emptyDir: {{medium: Memory, sizeLimit: {shm_size}}}",
    ]
    if model_pvc:
        # readOnly: many pods may mount the same staged weights concurrently, and
        # a read-only mount is what makes ReadOnlyMany binding legal.
        mounts.insert(0, f"            - {{name: models, mountPath: {model_mount}, readOnly: true}}")
        volumes.insert(0, f"        - name: models\n          persistentVolumeClaim: "
                          f"{{claimName: {model_pvc}, readOnly: true}}")

    deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}{ns}
  labels: {labels}
spec:
  replicas: {replicas}
  selector:
    matchLabels: {{app: {name}}}
  template:
    metadata:
      labels: {labels}
    spec:
      securityContext:
        # No runAsUser: restricted-v2 assigns the namespace's UID range, and
        # pinning one is what gets the pod rejected.
        runAsNonRoot: true
        seccompProfile: {{type: RuntimeDefault}}
      containers:
        - name: server
          image: {image}
          {cmd_yaml}
          securityContext:
            allowPrivilegeEscalation: false
            capabilities: {{drop: ["ALL"]}}
          ports:
            - {{containerPort: {port}, name: http}}{env_block}{res_block}
          volumeMounts:
{chr(10).join(mounts)}
          # A big model can take many minutes to load; a startupProbe is what
          # stops the kubelet restarting it mid-load in a loop, while still
          # failing eventually if it never comes up.
          startupProbe:
            httpGet: {{path: /v1/models, port: http}}
            periodSeconds: 10
            failureThreshold: 180
          readinessProbe:
            httpGet: {{path: /v1/models, port: http}}
            periodSeconds: 10
      volumes:
{chr(10).join(volumes)}"""

    service = f"""apiVersion: v1
kind: Service
metadata:
  name: {name}{ns}
  labels: {labels}
spec:
  selector: {{app: {name}}}
  ports:
    - {{name: http, port: {port}, targetPort: http}}"""

    docs = [deployment, service]
    if route_host:
        docs.append(f"""apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: {name}{ns}
  labels: {labels}
  annotations:
    # Token generation streams for a long time; the router's 30s default cuts
    # the connection mid-response.
    haproxy.router.openshift.io/timeout: "3600s"
spec:
  host: {route_host}
  to: {{kind: Service, name: {name}}}
  port: {{targetPort: http}}
  tls: {{termination: edge}}""")

    header = [f"# boxy: {model} on OpenShift ({engine}"
              + (f", {gpus}x {GPU_RESOURCES.get(accelerator, accelerator)}" if gpus else ", CPU")
              + ")",
              "# apply:  boxy generate openshift ... | oc apply -f -"]
    if hf_secret:
        header.append("# needs a secret first (the token never enters this file):")
        header.append(f"#   oc create secret generic {hf_secret} --from-literal=token=\"$HF_TOKEN\"")
    if route_host:
        header.append(f"# reach it at:  https://{route_host}/v1/models")
    else:
        header.append("# reach it (no Route needed, lowest privilege):")
        header.append(f"#   oc port-forward svc/{name} {port}:{port}")
        header.append(f"#   curl -s http://127.0.0.1:{port}/v1/models")
    return "\n".join(header) + "\n" + "\n---\n".join(docs) + "\n"
