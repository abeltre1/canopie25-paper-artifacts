import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from boxy.box import Box, Volume
from boxy.location import Location, Resources, Staging

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Keep the developer's real ~/.config/boxy/config.toml out of every test, and
    clear config.py's cached file parse between tests (it is process-global). Tests
    that exercise the file layer re-point XDG_CONFIG_HOME/BOXY_CONFIG and call
    config.reset() themselves."""
    from boxy import config

    monkeypatch.delenv("BOXY_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-none"))
    config.reset()
    yield
    config.reset()


@pytest.fixture
def vllm_box() -> Box:
    return Box(
        name="vllm",
        image="vllm/vllm-openai:v0.9.1",
        entrypoint="vllm",
        model="Llama-4-Scout-17B-16E-Instruct",
        workdir="/vllm-workspace/models",
        ports=[8000],
        volumes=[Volume(source="${MODELS_DIR}", target="/vllm-workspace/models")],
        args={"tensor_parallel_size": 4, "seed": 12345},
    )


@pytest.fixture
def hops() -> Location:
    return Location(
        name="hops",
        scheduler="slurm",
        accelerator="cuda",
        runtime="podman",
        offline=True,
        resources=Resources(nodes=2, gpus_per_node=4),
        staging=Staging(models_dir="./models"),
    )


@pytest.fixture
def eldorado() -> Location:
    return Location(
        name="eldorado",
        scheduler="flux",
        accelerator="rocm",
        runtime="apptainer",
        offline=True,
        resources=Resources(nodes=2, gpus_per_node=4),
        modules=["rocm/6.4.0"],
        staging=Staging(models_dir="./models"),
        tuning={"gpu_memory_utilization": 0.7},
    )
