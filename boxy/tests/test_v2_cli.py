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


BASE = "boxy-tiny-llama.q4_k_m"  # slug of the gguf fixture


class _ReclaimHarness:
    """Drive cmd_serve's collision policy without containers: pin the
    inspect helpers and stub execution."""

    def __init__(self, monkeypatch, exists=(), ours=None, running=False,
                 ready_id=None, container_port=None):
        import boxy.cli as cli
        import boxy.deploy as deploy
        import boxy.readiness as readiness

        exist_set = set(exists)
        ours_set = set(exists if ours is None else ours)
        monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "none")
        monkeypatch.setattr(cli, "_container_exists", lambda r, n: n in exist_set)
        monkeypatch.setattr(cli, "_container_running", lambda r, n: running)
        monkeypatch.setattr(cli, "_container_label", lambda r, n: n if n in ours_set else "")
        monkeypatch.setattr(cli, "_container_port", lambda r, n: container_port)
        monkeypatch.setattr(cli, "_dump_logs", lambda r, n, tail=50: None)
        self.commands = []
        monkeypatch.setattr(cli.subprocess, "run",
                            lambda cmd, **kw: self.commands.append(cmd) or
                            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        monkeypatch.setattr(deploy, "execute", lambda d: 0)
        self.ready_urls = []

        def fake_wait(url, **kw):
            self.ready_urls.append(url)
            return ready_id

        monkeypatch.setattr(readiness, "wait_ready", fake_wait)


def test_serve_rerun_is_idempotent_when_already_ready(gguf, monkeypatch, capsys):
    """Field finding 14: rerunning `boxy serve MODEL` while it is already
    serving must report the endpoint (exit 0), not error — and must probe the
    CONTAINER's port, not the fresh resolution's (which scans past the busy
    port the running instance itself occupies)."""
    h = _ReclaimHarness(monkeypatch, exists=[BASE], running=True,
                        ready_id="granite3-moe", container_port=7777)
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALREADY SERVING" in out and ":7777/v1" in out and "boxy stop" in out
    assert h.ready_urls == ["http://127.0.0.1:7777"]
    assert "### Running Command" not in out  # nothing was launched


def test_serve_rerun_while_still_loading_reports_and_exits_1(gguf, monkeypatch, capsys):
    _ReclaimHarness(monkeypatch, exists=[BASE], running=True, ready_id=None)
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    assert rc == 1
    assert "not answering yet" in capsys.readouterr().err


def test_serve_reclaims_exited_leftover_and_relaunches(gguf, monkeypatch, capsys):
    harness = _ReclaimHarness(monkeypatch, exists=[BASE], running=False, ready_id="m")
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "### READY" in captured.out            # relaunch went through
    assert "removed" in captured.err and "relaunching" in captured.err
    assert any(cmd[:2] == ["docker", "rm"] for cmd in harness.commands)  # leftover removed


def test_serve_suffixes_when_name_owned_by_foreign_container(gguf, monkeypatch, capsys):
    """User request: a name held by a container boxy did NOT create must not
    block serving — auto-suffix and proceed."""
    _ReclaimHarness(monkeypatch, exists=[BASE], ours=[], running=False, ready_id="m")
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert f"auto: name: {BASE}-2" in out and "not created by boxy" in out
    assert f"--name={BASE}-2" in out and f"--label=boxy.box={BASE}-2" in out
    assert f"boxy stop {BASE}-2" in out           # hints use the real name


def test_serve_rerun_finds_our_suffixed_instance(gguf, monkeypatch, capsys):
    """After a suffixed launch, a rerun must reclaim boxy's -2 instance
    (idempotent), not stack a -3 duplicate."""
    _ReclaimHarness(monkeypatch, exists=[BASE, f"{BASE}-2"], ours=[f"{BASE}-2"],
                    running=True, ready_id="m", container_port=8090)
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALREADY SERVING" in out and f"boxy stop {BASE}-2" in out
    assert "### Running Command" not in out


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
