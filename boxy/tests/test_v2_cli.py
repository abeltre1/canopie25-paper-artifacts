"""v2 CLI surface: `boxy serve MODEL` end-to-end through argparse (dry-run),
profile interop, stop-by-name, pull-by-URI."""

import pytest

from boxy.box import Box
from boxy.cli import main
from boxy.location import Location
from tests.conftest import EXAMPLES


@pytest.fixture(autouse=True)
def laptop_host(monkeypatch):
    """These tests assert laptop/workstation behavior; the test environment
    may itself have a scheduler installed (the sandbox runs a real Slurm for
    the live E2E), which would trip the login-node guard. Hide srun/flux from
    RESOLUTION only — guard-specific tests patch `which` themselves."""
    import shutil as _shutil

    real_which = _shutil.which
    monkeypatch.setattr("boxy.resolve.shutil.which",
                        lambda name: None if name in ("srun", "flux") else real_which(name))


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
    assert main(["serve", "--dryrun"]) == 2  # usage errors exit 2 (finding 51)
    assert "usage: boxy serve MODEL" in capsys.readouterr().err


def test_serve_missing_local_file_suggests_transports(capsys):
    rc = main(["serve", "no-such-model.gguf", "--runtime", "docker"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ollama://no-such-model.gguf" in err and "hf://" in err


def test_serve_gpus_without_scheduler_is_error(gguf, capsys):
    assert main(["serve", str(gguf), "--runtime", "docker", "--gpus", "4", "--dryrun"]) == 1
    assert "--scheduler" in capsys.readouterr().err


def test_serve_scheduler_submits_batch_job_dryrun(gguf, capsys, monkeypatch, tmp_path):
    """MODEL + --scheduler now SUBMITS (sbatch) instead of wrapping srun:
    the batch script re-invokes boxy on the compute node."""
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--gpus", "2", "--nodes", "1",
               "--partition", "short", "--account", "ab110003",
               "--scheduler-arg=--license=sitescratch:1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#SBATCH --gpus-per-node=2" in out and "#SBATCH --nodes=1" in out
    assert "#SBATCH --partition=short" in out and "#SBATCH --account=ab110003" in out
    assert "#SBATCH --license=sitescratch:1" in out
    assert "sbatch --parsable" in out
    assert "--foreground --here" in out and "--endpoint-file" in out  # inner re-resolution
    assert "resolved on the compute node" in out
    assert "srun" not in out


def test_serve_scheduler_foreground_keeps_attached_srun(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "podman", "--scheduler", "slurm",
               "--accelerator", "cuda", "--gpus", "2", "--nodes", "1",
               "--foreground", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "srun" in out and "--gpus-per-node=2" in out
    assert "#SBATCH" not in out


def test_serve_flux_submission_uses_flux_spellings(gguf, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--gpus", "4",
               "--partition", "pbatch", "--account", "guests", "--time", "4h", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    # flux batch: lowercase `# flux:` sentinel; GPUs via -n/-g slots, not --gpus-per-node
    assert "# flux: -g4" in out and "# flux: -n" in out
    assert "#FLUX:" not in out                   # the uppercase spelling is silently ignored by flux
    assert "# flux: --queue=pbatch" in out       # partition -> queue
    assert "# flux: --bank=guests" in out        # account -> bank
    assert "# flux: -t 4h" in out                # time -> -t
    assert "flux batch" in out


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


def test_serve_rerun_replaces_running_instance(gguf, monkeypatch, capsys):
    """User request (2026-07): without --unique the name is a per-model
    singleton, so rerunning `boxy serve MODEL` REPLACES the running instance
    (rm -f + relaunch fresh) and points at --unique for a second one — it does
    NOT leave a duplicate and does NOT just report idempotently."""
    h = _ReclaimHarness(monkeypatch, exists=[BASE], running=True, ready_id="m")
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "### READY" in captured.out                       # a fresh instance launched
    assert "replacing the running instance" in captured.err and "--unique" in captured.err
    assert ["docker", "rm", "-f", BASE] in h.commands        # the duplicate was force-removed


def test_serve_rerun_replaces_even_if_old_not_ready_yet(gguf, monkeypatch, capsys):
    # a running instance is replaced without first probing it — boxy no longer
    # branches on whether the OLD one was answering; it just redeploys.
    h = _ReclaimHarness(monkeypatch, exists=[BASE], running=True, ready_id=None)
    main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    assert ["docker", "rm", "-f", BASE] in h.commands          # old one replaced
    assert "### Running Command" in capsys.readouterr().out    # fresh launch attempted


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


def test_serve_rerun_replaces_our_suffixed_instance(gguf, monkeypatch, capsys):
    """When BASE is foreign-owned, boxy uses its own -2 instance; a rerun then
    REPLACES that -2 (still a per-name singleton), it does not stack a -3."""
    h = _ReclaimHarness(monkeypatch, exists=[BASE, f"{BASE}-2"], ours=[f"{BASE}-2"],
                        running=True, ready_id="m")
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "demo:1"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "### READY" in captured.out
    assert ["docker", "rm", "-f", f"{BASE}-2"] in h.commands   # our -2 replaced
    assert f"--name={BASE}-2" in captured.out                  # relaunched under the same -2 name
    assert f"--name={BASE}-3" not in captured.out              # not stacked


def test_stop_by_positional_name(capsys):
    rc = main(["stop", "boxy-tiny-llama", "--runtime", "docker", "--dryrun"])
    assert rc == 0
    assert "docker stop boxy-tiny-llama" in capsys.readouterr().out


def test_stop_without_name_or_box_is_usage_error(capsys):
    assert main(["stop", "--runtime", "docker"]) == 2  # usage errors exit 2 (finding 51)
    assert "boxy stop NAME" in capsys.readouterr().err


def test_pull_accepts_positional_uri(monkeypatch, capsys):
    seen = {}

    def fake_pull(uri, dryrun=False, quiet=False, force=False):
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


def test_nodes_per_replica_maps_to_nodes_for_single_instance(tmp_path, monkeypatch, capfd):
    """Field: `--nodes-per-replica 2` without --replicas was silently ignored —
    a 2-node Maverick plan ran single-node and OOMed. With one instance the
    flag now means --nodes 2 (announced), so the Ray plan engages."""
    from boxy.cli import main

    rc = main(["serve", "hf://org/repo", "--nodes-per-replica", "2", "--gpus", "4",
               "--scheduler", "slurm", "--runtime", "podman", "--accelerator", "cuda",
               "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "auto: nodes: 2 (--nodes-per-replica with a single instance = --nodes)" in out
