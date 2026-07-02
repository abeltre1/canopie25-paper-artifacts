"""Tests for the llama.cpp engine builder and the boxy -> SkyPilot transpiler."""

import pytest

from boxy import engines, sky_export
from boxy.box import Box
from boxy.location import Location, Resources


@pytest.fixture
def llamacpp_box() -> Box:
    return Box(
        name="llamacpp",
        image="boxy-demo/llamacpp:local",
        engine="llama.cpp",
        model="tiny.gguf",
        workdir="/models",
        ports=[8090],
    )


@pytest.fixture
def cloud_gpu() -> Location:
    return Location(
        name="cloud-gpu",
        scheduler="none",
        accelerator="cuda",
        runtime="docker",
        resources=Resources(nodes=1, gpus_per_node=4, accelerator_type="H100"),
    )


def test_llamacpp_serve_cmd_space_style(llamacpp_box, cloud_gpu):
    cmd = engines.build_serve_cmd(llamacpp_box, cloud_gpu, "tiny.gguf")
    # No explicit entrypoint => defer to the image ENTRYPOINT ("" sentinel):
    # the upstream llama.cpp image keeps its binary off $PATH (field finding).
    assert cmd[:3] == ["", "-m", "tiny.gguf"]
    i = cmd.index("--port")
    assert cmd[i + 1] == "8090"
    assert not any(a.startswith("--port=") for a in cmd)


def test_llamacpp_user_args_win(llamacpp_box, cloud_gpu):
    cmd = engines.build_serve_cmd(llamacpp_box, cloud_gpu, "tiny.gguf", extra_args=["--port", "9999"])
    assert cmd.count("--port") == 1
    assert cmd[cmd.index("--port") + 1] == "9999"


def test_engine_dispatch_vllm_default(vllm_box, cloud_gpu):
    cmd = engines.build_serve_cmd(vllm_box, cloud_gpu, "model-x")
    assert cmd[:3] == ["vllm", "serve", "model-x"]


def test_box_rejects_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        Box(name="x", image="y", engine="tgi")


def test_sky_export_task(vllm_box, cloud_gpu):
    yaml_text = sky_export.to_sky_task(vllm_box, cloud_gpu)
    assert "image_id: docker:vllm/vllm-openai:v0.9.1" in yaml_text
    assert "accelerators: H100:4" in yaml_text
    assert "ports: [8000]" in yaml_text
    assert "vllm serve" in yaml_text
    assert "service:" not in yaml_text  # no service block without --serve


def test_sky_export_serve_block(vllm_box, cloud_gpu):
    yaml_text = sky_export.to_sky_task(vllm_box, cloud_gpu, serve=True)
    assert "service:" in yaml_text
    assert "path: /v1/models" in yaml_text
    assert "replicas: 1" in yaml_text


def test_sky_export_no_accel_type_omits_accelerators(vllm_box):
    loc = Location(name="cpu", scheduler="none", runtime="docker")
    yaml_text = sky_export.to_sky_task(vllm_box, loc)
    assert "accelerators:" not in yaml_text
