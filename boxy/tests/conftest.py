import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from boxy.box import Box, Volume
from boxy.location import Location, Resources, Staging

EXAMPLES = Path(__file__).parent.parent / "src" / "boxy" / "data" / "examples"


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Keep the developer's real ~/.config/boxy/config.toml out of every test, and
    clear config.py's cached file parse between tests (it is process-global). Tests
    that exercise the file layer re-point XDG_CONFIG_HOME/BOXY_CONFIG and call
    config.reset() themselves."""
    from boxy import config

    monkeypatch.delenv("BOXY_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-none"))
    # remote CA propagation is opt-in per test: off by default so a delegated
    # command never tries to `cat >` a CA into the test runner's real $HOME.
    monkeypatch.setenv("BOXY_NO_CA_PROPAGATE", "1")
    # ditto proxy forwarding: the CI/sandbox runner may export proxy vars, which
    # would otherwise be injected into every delegated command; e2e tests opt in.
    monkeypatch.setenv("BOXY_NO_PROXY_PROPAGATE", "1")
    # ditto remote account injection (a dev box may carry sacctmgr/mywcid,
    # making --ssh serve tests nondeterministic); e2e tests opt back in.
    monkeypatch.setenv("BOXY_NO_REMOTE_ACCOUNT", "1")
    # ditto auto-share: off by default so every --ssh serve test doesn't emit a
    # team-URL decision line / attempt a relay; the auto-share test opts back in.
    monkeypatch.setenv("BOXY_AUTO_SHARE", "false")
    # agentless-over-ssh is the production DEFAULT, but the existing --ssh e2e
    # tests exercise the DELEGATION path; keep that the test default and let the
    # agentless tests opt in with BOXY_AGENTLESS_SSH=true (mirrors the opt-outs above).
    monkeypatch.setenv("BOXY_AGENTLESS_SSH", "false")
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
