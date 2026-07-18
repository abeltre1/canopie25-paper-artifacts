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
    # network.proxy ships a site default (proxy.sandia.gov); blank it so goldens
    # stay site-agnostic. Proxy tests pass --proxy explicitly.
    monkeypatch.setenv("BOXY_PROXY", "")
    # relay.apps_domain ships a site default (apps.goodall.sandia.gov); blank it
    # so the oc-discovery paths are what the suite exercises.
    monkeypatch.setenv("BOXY_APPS_DOMAIN", "")
    # the HF architecture preflight does a live Hub fetch; keep it off for the
    # suite (no network calls per test) — the preflight tests opt back in.
    monkeypatch.setenv("BOXY_NO_PREFLIGHT", "1")
    # ditto serve-time card AUTOGEN (a live HuggingFace fetch for uncarded
    # models): off for the suite; the autogen tests opt in with a fixture Hub.
    monkeypatch.setenv("BOXY_CARD_AUTOGEN", "false")
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
    # the interactive WCID picker is a TTY feature; keep it OFF for the suite so no
    # test can block on input() and multi-account discovery keeps its silent
    # first-pick. Picker tests opt in with BOXY_PICK_ACCOUNT=always + a fake input().
    monkeypatch.setenv("BOXY_PICK_ACCOUNT", "never")
    monkeypatch.setenv("BOXY_PICK_PARTITION", "never")
    # model-store discovery probes the login node (mkdir/df over ssh) for a big
    # scratch FS — nondeterministic on a dev/CI box (the fake ssh runs it
    # LOCALLY, and a root runner could really create /tscratch). Pin the store
    # per test; the discovery tests blank this to opt back in.
    monkeypatch.setenv("BOXY_MODEL_DIR", str(tmp_path / "model-store"))
    # site.license defaults to tscratch:1 (Sandia); neutralize it for the broad
    # suite so goldens/e2e stay site-agnostic. The license tests opt back in.
    monkeypatch.setenv("BOXY_LICENSE", "")
    monkeypatch.delenv("WCID", raising=False)
    # the Slurm GRES auto-detect override is process-global; clear it so a value
    # set by one --ssh agentless test can't leak into the next (default 'auto').
    from boxy.schedulers import slurm as _slurm

    _slurm.reset_auto_gres()
    # the agentless CA-mount override is process-global too; clear it so a value set
    # by one --ssh agentless test can't leak the laptop/cluster CA path into the next.
    from boxy import deploy as _deploy

    _deploy.set_agentless_ca(None)
    # boxy's local HPC accel ladder (the detect_accel fallback) runs /bin/sh
    # probes on THIS host — a runner with sinfo, Lmod, or a real GPU would make
    # decision lines nondeterministic. Seed the memo with "nothing found" (the
    # historical behavior); ladder tests reset it to None to opt back in.
    from boxy import site as _site

    monkeypatch.setattr(_site, "_local_accel_cache", ("", ""))
    _deploy.set_airgap(False)
    # agentless pre-staging is on-by-default in prod, but the existing agentless e2e
    # tests assert the engine-pull render; keep it off for the suite and let the
    # prestage tests opt in with BOXY_AGENTLESS_PRESTAGE=auto/always.
    monkeypatch.setenv("BOXY_AGENTLESS_PRESTAGE", "never")
    config.reset()
    yield
    config.reset()
    _deploy.set_agentless_ca(None)
    _deploy.set_airgap(False)


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
