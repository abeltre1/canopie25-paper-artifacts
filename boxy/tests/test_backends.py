"""Golden-argv tests: backend commands must match the prototype's
known-good commands from hpc-workflow/common_boxy.sh."""

from boxy import envs
from boxy.backends import get_backend

MOUNTS = [("./models", "/vllm-workspace/models", "")]
INNER = ["vllm", "serve", "Llama-4-Scout-17B-16E-Instruct", "--port=8000"]


def test_podman_cuda(vllm_box, clustera):
    cmd = get_backend("podman").build_command(vllm_box, clustera, INNER, {}, MOUNTS, "cuda")
    assert cmd[:2] == ["podman", "run"]
    assert "--rm" in cmd and "--name=vllm" in cmd
    assert "--network=host" in cmd and "--ipc=host" in cmd
    assert "--entrypoint=vllm" in cmd
    assert "--workdir=/vllm-workspace/models" in cmd
    assert "--volume=./models:/vllm-workspace/models" in cmd
    # Prototype: PODMAN_ARGS+=("--device nvidia.com/gpu=all")
    i = cmd.index("--device")
    assert cmd[i + 1] == "nvidia.com/gpu=all"
    # image before inner args, entrypoint via flag (prototype behavior)
    assert cmd.index("vllm/vllm-openai:v0.9.1") < cmd.index("serve")
    assert cmd[-1] == "--port=8000"


def test_podman_rocm(vllm_box, clusterb):
    cmd = get_backend("podman").build_command(vllm_box, clusterb, INNER, {}, MOUNTS, "rocm")
    # Prototype ROCm set: --group-add=video --cap-add=SYS_PTRACE
    #   --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined
    assert "--group-add=video" in cmd
    assert "--cap-add=SYS_PTRACE" in cmd
    devices = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--device"]
    assert devices == ["/dev/kfd", "/dev/dri"]
    i = cmd.index("--security-opt")
    assert cmd[i + 1] == "seccomp=unconfined"


def test_apptainer_rocm(vllm_box, clusterb):
    backend = get_backend("apptainer")
    cmd = backend.build_command(vllm_box, clusterb, INNER, {}, MOUNTS, "rocm")
    assert cmd[:2] == ["apptainer", "exec"]
    # Prototype APPTAINER_ARGS + --rocm
    for flag in ("--fakeroot", "--writable-tmpfs", "--cleanenv", "--no-home", "--rocm"):
        assert flag in cmd
    i = cmd.index("--cwd")
    assert cmd[i + 1] == "/vllm-workspace/models"
    i = cmd.index("--bind")
    assert cmd[i + 1] == "./models:/vllm-workspace/models"
    assert "HF_HOME=/root/.cache/huggingface" in cmd
    # Prototype SIF name: vllm-rocm.sif; inner command follows the SIF
    sif = cmd.index("vllm-rocm.sif")
    assert cmd[sif + 1 : sif + 3] == ["vllm", "serve"]


def test_apptainer_cuda_uses_nv(vllm_box, clustera):
    cmd = get_backend("apptainer").build_command(vllm_box, clustera, INNER, {}, MOUNTS, "cuda")
    assert "--nv" in cmd and "--rocm" not in cmd
    assert "vllm-cuda.sif" in cmd


def test_apptainer_prepare_builds_sif(vllm_box, clusterb):
    prepare = get_backend("apptainer").prepare(vllm_box, clusterb)
    # Prototype: apptainer build --force ${SHORT_NAME}.sif docker://$IMAGE_NAME
    assert prepare == [["apptainer", "build", "--force", "vllm-rocm.sif", "docker://vllm/vllm-openai:v0.9.1"]]


def test_docker_cuda_uses_gpus_all(vllm_box, clustera):
    cmd = get_backend("docker").build_command(vllm_box, clustera, INNER, {}, MOUNTS, "cuda")
    i = cmd.index("--gpus")
    assert cmd[i + 1] == "all"


def test_env_injection(vllm_box, clusterb):
    env = envs.build_env(vllm_box.env, "rocm", offline=True)
    cmd = get_backend("apptainer").build_command(vllm_box, clusterb, INNER, env, MOUNTS, "rocm")
    joined = " ".join(cmd)
    # Offline set (prototype ENV_VARS) + ROCm vLLM quirks
    for var in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "VLLM_NO_USAGE_STATS=1", "VLLM_USE_V1=1"):
        assert var in joined


def test_registry_prefix(vllm_box, clustera):
    clustera.registry = "registry.example.com/"
    cmd = get_backend("podman").build_command(vllm_box, clustera, INNER, {}, MOUNTS, "cuda")
    assert "registry.example.com/vllm/vllm-openai:v0.9.1" in cmd
