"""OpenShift serve manifests: parse as YAML, stay inside the user-level
envelope, and design out the three failure modes that cost an afternoon each.

No cluster is involved — the emitters are pure string builders, exactly like
mcp.emit_flux_mcp_manifest and router.emit_nginx, so every claim below is
checkable in CI.
"""

import pytest

from boxy import openshift
from boxy.cli import main

yaml = pytest.importorskip("yaml", reason="pyyaml is a test-only dep for parsing emitted manifests")


def _docs(**kw):
    kw.setdefault("gpus", 4)
    text = openshift.emit_serve_manifest("m", "img:1", "/models/x", **kw)
    return text, [d for d in yaml.safe_load_all(text) if d]


def _pod(docs):
    return docs[0]["spec"]["template"]["spec"]


# ---- the manifest is real YAML, with the objects it claims ---------------------------


def test_emits_deployment_and_service_and_route_only_when_asked():
    _, docs = _docs()
    assert [d["kind"] for d in docs] == ["Deployment", "Service"]
    # A Route needs permission a user may not have, so it is opt-in. Without one
    # the service is reached with `oc port-forward`, which always works.
    _, routed = _docs(route_host="m.apps.example.gov")
    assert [d["kind"] for d in routed] == ["Deployment", "Service", "Route"]
    assert routed[2]["spec"]["host"] == "m.apps.example.gov"
    assert routed[2]["spec"]["tls"] == {"termination": "edge"}


def test_namespace_is_omitted_so_the_manifest_lands_in_the_current_project():
    _, docs = _docs()
    assert "namespace" not in docs[0]["metadata"]
    _, pinned = _docs(namespace="my-project")
    assert all(d["metadata"]["namespace"] == "my-project" for d in pinned)


def test_service_selector_matches_the_deployment_pod_labels():
    # a selector/label mismatch yields a Service with no endpoints, which looks
    # exactly like a model that failed to start
    _, docs = _docs()
    assert docs[1]["spec"]["selector"].items() <= docs[0]["spec"]["template"]["metadata"]["labels"].items()


# ---- user level: nothing here needs an administrator ---------------------------------


@pytest.mark.parametrize("kw", [
    {"gpus": 0},
    {"gpus": 4, "accelerator": "cuda"},
    {"gpus": 8, "accelerator": "rocm", "model_pvc": "w"},
    {"gpus": 1, "hf_secret": "s", "route_host": "h.example.gov", "namespace": "p"},
])
def test_nothing_requires_elevated_privilege(kw):
    """OpenShift's default restricted-v2 SCC rejects all of these outright, and a
    user cannot grant themselves another SCC. Asserted on the PARSED object, not
    the text, so the module's own prose about runAsUser cannot satisfy it."""
    _, docs = _docs(**kw)
    pod = _pod(docs)
    assert "runAsUser" not in pod.get("securityContext", {})
    assert pod["securityContext"]["runAsNonRoot"] is True
    for key in ("hostNetwork", "hostPID", "hostIPC"):
        assert not pod.get(key)
    assert not [v for v in pod["volumes"] if "hostPath" in v]
    for c in pod["containers"]:
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False
        assert sc["capabilities"] == {"drop": ["ALL"]}
        assert not sc.get("privileged")
        assert "runAsUser" not in sc
    # cluster-scoped objects would need an admin to apply
    assert all(d["kind"] in ("Deployment", "Service", "Route") for d in docs)


# ---- the three designed-out failure modes --------------------------------------------


def test_caches_are_redirected_so_a_random_uid_can_write():
    """restricted-v2 runs the pod as an arbitrary UID owning nothing in the
    image. A vLLM image whose HOME is /root then dies during startup on a
    permission error that names no path."""
    _, docs = _docs()
    env = {e["name"]: e.get("value") for e in _pod(docs)["containers"][0]["env"]}
    assert env["HOME"] == "/scratch"
    assert env["HF_HOME"].startswith("/scratch")
    mounts = {m["name"]: m["mountPath"] for m in _pod(docs)["containers"][0]["volumeMounts"]}
    assert mounts["scratch"] == "/scratch"
    # ...and the target must actually be writable: an emptyDir, not the PVC,
    # which is mounted read-only
    scratch = [v for v in _pod(docs)["volumes"] if v["name"] == "scratch"][0]
    assert "emptyDir" in scratch


def test_dev_shm_is_replaced_because_the_64mib_default_breaks_tensor_parallel():
    _, docs = _docs(gpus=4)
    dshm = [v for v in _pod(docs)["volumes"] if v["name"] == "dshm"][0]
    assert dshm["emptyDir"]["medium"] == "Memory"
    assert dshm["emptyDir"]["sizeLimit"] == openshift.DEFAULT_SHM_SIZE
    assert any(m["mountPath"] == "/dev/shm" for m in _pod(docs)["containers"][0]["volumeMounts"])


def test_the_token_is_referenced_never_written_into_the_manifest(monkeypatch):
    """A manifest gets committed and pasted. Emitting a token value into one is a
    disclosure, so only a Secret NAME may appear."""
    monkeypatch.setenv("HF_TOKEN", "hf_a_real_looking_secret_value")
    text, docs = _docs(hf_secret="hf-token")
    assert "hf_a_real_looking_secret_value" not in text
    hf = [e for e in _pod(docs)["containers"][0]["env"] if e["name"] == "HF_TOKEN"][0]
    assert hf["valueFrom"]["secretKeyRef"] == {"name": "hf-token", "key": "token"}
    assert "value" not in hf
    # and the header tells the user how to create it
    assert "oc create secret generic hf-token" in text


# ---- resources and geometry ----------------------------------------------------------


def test_gpu_resource_name_follows_the_accelerator():
    for accel, resource in openshift.GPU_RESOURCES.items():
        _, docs = _docs(gpus=2, accelerator=accel)
        assert _pod(docs)["containers"][0]["resources"]["limits"][resource] == 2


def test_cpu_serving_requests_no_gpu_at_all():
    _, docs = _docs(gpus=0)
    limits = _pod(docs)["containers"][0].get("resources", {}).get("limits", {})
    assert not any(k.endswith("/gpu") or "gpu" in k for k in limits)


def test_unknown_accelerator_is_refused_rather_than_guessed():
    # emitting a made-up resource name yields a pod that is Pending forever with
    # "0/N nodes are available: insufficient <nonsense>"
    with pytest.raises(ValueError, match="no Kubernetes device-plugin resource"):
        openshift.emit_serve_manifest("m", "i:1", "/m", gpus=1, accelerator="ascend")


def test_pvc_is_mounted_read_only_so_replicas_can_share_it():
    _, docs = _docs(model_pvc="weights")
    vol = [v for v in _pod(docs)["volumes"] if v["name"] == "models"][0]
    assert vol["persistentVolumeClaim"] == {"claimName": "weights", "readOnly": True}
    mount = [m for m in _pod(docs)["containers"][0]["volumeMounts"] if m["name"] == "models"][0]
    assert mount["readOnly"] is True


def test_startup_probe_tolerates_a_long_weight_load():
    """A 70B can take 20 minutes to load. With only a readinessProbe the kubelet
    restarts it mid-load, forever."""
    _, docs = _docs()
    c = _pod(docs)["containers"][0]
    budget = c["startupProbe"]["periodSeconds"] * c["startupProbe"]["failureThreshold"]
    assert budget >= 1800, "the startup budget must outlast a big model's load"


# ---- names ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("meta-llama/Llama-3.3-70B-Instruct", "boxy-llama-3-3-70b-instruct"),
    ("hf://org/Mixtral-8x7B", "boxy-mixtral-8x7b"),
    ("/scratch/models/Qwen2.5-72B-Instruct/", "boxy-qwen2-5-72b-instruct"),
    ("x.gguf", "boxy-x"),
])
def test_k8s_name_strips_what_a_service_name_forbids(model, expected):
    # dots and capitals are the two that bite: a Service name is an RFC 1123
    # LABEL, stricter than the subdomain rule a Deployment name follows
    assert openshift.k8s_name(model) == expected
    assert openshift._RFC1123_LABEL.match(openshift.k8s_name(model))


def test_invalid_name_is_refused_with_the_fix():
    with pytest.raises(ValueError, match="not a valid Kubernetes object name"):
        openshift.emit_serve_manifest("Llama_3.3", "i:1", "/m")


# ---- the CLI ------------------------------------------------------------------------


def test_cli_resolves_geometry_and_image_from_the_model_card(capsys):
    rc = main(["generate", "openshift", "--model", "meta-llama/Llama-3.3-70B-Instruct",
               "--model-pvc", "weights", "--accelerator", "rocm"])
    cap = capsys.readouterr()
    assert rc == 0
    # stdout stays pure YAML so it can be piped straight into `oc apply -f -`
    docs = [d for d in yaml.safe_load_all(cap.out) if d]
    assert [d["kind"] for d in docs] == ["Deployment", "Service"]
    c = docs[0]["spec"]["template"]["spec"]["containers"][0]
    # the card supplied the GPU count and the context cap — the user typed neither
    assert c["resources"]["limits"]["amd.com/gpu"] == 4
    assert "--max-model-len" in c["command"]
    # decisions go to stderr, where they cannot corrupt the YAML
    assert "auto: gpus: 4 per node" in cap.err


def test_cli_requires_a_model(capsys):
    assert main(["generate", "openshift"]) == 2
    assert "--model is required" in capsys.readouterr().err
