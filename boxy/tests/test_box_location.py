import pytest

from boxy.box import Box
from boxy.location import Location
from tests.conftest import EXAMPLES


def test_box_from_toml():
    box = Box.from_toml(EXAMPLES / "boxes" / "vllm.toml")
    assert box.name == "vllm"
    assert box.image == "vllm/vllm-openai:v0.24.0"
    assert box.volumes[0].source == "${MODELS_DIR}"
    assert box.args["tensor_parallel_size"] == 4
    assert not box.model_is_transport_uri


def test_box_transport_uri_detection(vllm_box):
    vllm_box.model = "hf://meta-llama/Llama-4-Scout-17B-16E-Instruct"
    assert vllm_box.model_is_transport_uri


def test_location_from_toml():
    loc = Location.from_toml(EXAMPLES / "locations" / "eldorado.toml")
    assert loc.scheduler == "flux"
    assert loc.accelerator == "rocm"
    assert loc.runtime == "apptainer"
    assert loc.modules == ["rocm/6.4.0"]
    assert loc.tuning["gpu_memory_utilization"] == 0.7
    assert loc.resources.nodes == 2


def test_location_rejects_unknown_scheduler():
    with pytest.raises(ValueError, match="unknown scheduler"):
        Location(name="bad", scheduler="pbs")


def test_location_explicit_accelerator_wins(eldorado):
    # No autodetect call should be needed when accelerator is explicit.
    assert eldorado.resolve_accelerator() == "rocm"


def test_location_autodetect_accelerator_falls_back():
    # In this test container there is no GPU: autodetect must return "none"
    # (via ramalama get_accel) or degrade to "none" without ramalama.
    loc = Location(name="local", scheduler="none")
    assert loc.resolve_accelerator() == "none"
