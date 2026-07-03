"""v2 CLI surface: `boxy serve MODEL` end-to-end through argparse (dry-run),
profile interop, stop-by-name, pull-by-URI."""

import pytest

from boxy.box import Box
from boxy.cli import main
from boxy.location import Location
from tests.conftest import EXAMPLES


@pytest.fixture
def gguf(tmp_path):
    model = tmp_path / "tiny-llama.q4_k_m.gguf"
    model.write_bytes(b"GGUF")
    return model


def test_serve_model_dryrun_prints_decisions_and_command(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "auto: model:" in out
    assert "auto: engine: llama.cpp (model is GGUF)" in out
    assert "auto: runtime: docker (--runtime)" in out
    assert "### Running Command:" in out
    assert "docker run" in out and "demo:1" in out
    assert "boxy-tiny-llama.q4_k_m" in out  # container name printed for stop/list


def test_serve_without_model_or_box_is_usage_error(capsys):
    assert main(["serve", "--dryrun"]) == 1
    assert "usage: boxy serve MODEL" in capsys.readouterr().err


def test_serve_missing_local_file_suggests_transports(capsys):
    rc = main(["serve", "no-such-model.gguf", "--runtime", "docker"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ollama://no-such-model.gguf" in err and "hf://" in err


def test_serve_gpus_without_scheduler_is_error(gguf, capsys):
    assert main(["serve", str(gguf), "--runtime", "docker", "--gpus", "4", "--dryrun"]) == 1
    assert "--scheduler" in capsys.readouterr().err


def test_serve_scheduler_submission_dryrun(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "podman", "--scheduler", "slurm",
               "--accelerator", "cuda", "--gpus", "2", "--nodes", "1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "srun" in out and "--gpus-per-node=2" in out
    assert "auto: scheduler: slurm (--scheduler)" in out


def test_serve_save_profile_round_trips(gguf, tmp_path, capsys):
    prefix = tmp_path / "snap"
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1",
               "--dryrun", "--save-profile", str(prefix)])
    assert rc == 0
    box = Box.from_toml(f"{prefix}.box.toml")
    loc = Location.from_toml(f"{prefix}.location.toml")
    assert box.model == str(gguf) and box.image == "demo:1"
    assert loc.runtime == "docker" and loc.scheduler == "none"
    # the snapshot itself must dry-run
    rc = main(["serve", "--box", f"{prefix}.box.toml", "--location", f"{prefix}.location.toml", "--dryrun"])
    assert rc == 0


def test_serve_profile_mode_extra_args_still_reach_engine(capsys):
    rc = main(["serve",
               "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "local.toml"),
               "--dryrun", "--", "--max-model-len", "2048"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--max-model-len 2048" in out


def test_stop_by_positional_name(capsys):
    rc = main(["stop", "boxy-tiny-llama", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "docker stop boxy-tiny-llama" in capsys.readouterr().out


def test_stop_without_name_or_box_is_usage_error(capsys):
    assert main(["stop", "--runtime", "docker"]) == 1
    assert "boxy stop NAME" in capsys.readouterr().err


def test_pull_accepts_positional_uri(monkeypatch, capsys):
    seen = {}

    def fake_pull(uri, dryrun=False, quiet=False):
        seen["uri"] = uri
        return "/store/blob"

    monkeypatch.setattr("boxy.ramalama_shim.pull_model", fake_pull)
    rc = main(["pull", "hf://org/repo/file.gguf"])
    assert rc == 0
    assert seen["uri"] == "hf://org/repo/file.gguf"
    assert "/store/blob" in capsys.readouterr().out


def test_pull_local_path_is_noop(capsys):
    rc = main(["pull", "/shared/models/m.gguf"])
    assert rc == 0
    assert "nothing to pull" in capsys.readouterr().out
