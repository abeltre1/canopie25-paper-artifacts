"""The seamless scheduler path: submit -> track -> rendezvous -> READY.
Drives the full login-side state machine against a fake scheduler; the same
flow runs live against real Slurm in the E2E environment (see RUNBOOK §0)."""

import json

import pytest

from boxy import jobs
from boxy.cli import main
from boxy.location import Location


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    return tmp_path / "jobs"


@pytest.fixture
def gguf(tmp_path):
    model = tmp_path / "m.q4.gguf"
    model.write_bytes(b"GGUF")
    return model


def test_jobs_record_and_endpoint_roundtrip(jobs_dir):
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "42"})
    assert jobs.read_record("boxy-m")["job"] == "42"
    jobs.write_endpoint("boxy-m", 8090, job_id="42")
    endpoint = jobs.read_endpoint("boxy-m")
    assert endpoint["port"] == 8090 and endpoint["url"].startswith("http://")
    assert [r["name"] for r in jobs.list_records()] == ["boxy-m"]
    jobs.remove("boxy-m")
    assert jobs.read_record("boxy-m") is None and jobs.read_endpoint("boxy-m") is None


def test_location_profile_carries_scheduler_args(tmp_path):
    profile = tmp_path / "site.toml"
    profile.write_text(
        '[location]\nname = "hops"\nscheduler = "slurm"\naccelerator = "cuda"\n'
        'scheduler_args = ["--partition=short", "--license=tscratch:1"]\n'
        "[location.resources]\nnodes = 1\ngpus_per_node = 2\n"
    )
    loc = Location.from_toml(profile)
    assert loc.scheduler_args == ["--partition=short", "--license=tscratch:1"]


def test_model_plus_location_profile_submits_with_site_args(gguf, tmp_path, jobs_dir, capsys):
    profile = tmp_path / "site.toml"
    profile.write_text(
        '[location]\nname = "hops"\nscheduler = "slurm"\naccelerator = "cuda"\n'
        'scheduler_args = ["--partition=short", "--account=fy260064"]\n'
        "[location.resources]\nnodes = 1\ngpus_per_node = 2\n"
    )
    rc = main(["serve", str(gguf), "--location", str(profile), "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#SBATCH --partition=short" in out and "#SBATCH --account=fy260064" in out
    assert "#SBATCH --gpus-per-node=2" in out
    assert "sbatch --parsable" in out


class _FakeSlurm:
    """subprocess.run stand-in: scripted sbatch/squeue/scancel behavior.
    When the fake job reaches RUNNING it publishes the endpoint file, exactly
    like the real compute-node boxy does."""

    def __init__(self, monkeypatch, states, submit_stdout="77\n", submit_rc=0,
                 publish_on_running=None):
        import boxy.cli as cli

        self.states = list(states)
        self.calls = []
        self.publish_on_running = publish_on_running

        def fake_run(cmd, **kwargs):
            self.calls.append(list(cmd))
            out = ""
            rc = 0
            if cmd[0] == "sbatch":
                out, rc = submit_stdout, submit_rc
            elif cmd[0] == "squeue":
                out = self.states.pop(0) if len(self.states) > 1 else self.states[0]
                if "RUNNING" in out and self.publish_on_running:
                    name, port = self.publish_on_running
                    if not jobs.read_endpoint(name):
                        jobs.write_endpoint(name, port, job_id="77")
            return type("R", (), {"returncode": rc, "stdout": out, "stderr": ""})()

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)


def test_submission_reaches_ready(gguf, jobs_dir, monkeypatch, capsys):
    fake = _FakeSlurm(monkeypatch, states=["PENDING\n", "RUNNING\n"],
                      publish_on_running=("boxy-m", 9291))
    monkeypatch.setattr("boxy.readiness.wait_ready",
                        lambda url, **kw: "the-model" if jobs.read_endpoint("boxy-m") else None)

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "### Submitted slurm job 77" in out
    assert "### READY" in out and ":9291/v1" in out and "slurm job 77" in out
    assert "ssh -L 9291" in out and "boxy stop boxy-m" in out
    assert jobs.read_record("boxy-m")["job"] == "77"
    assert any(c[0] == "sbatch" for c in fake.calls)
    # the submitted script exists and re-invokes boxy with rendezvous flags
    script = jobs.script_path("boxy-m").read_text()
    assert "--foreground --here" in script and "--endpoint-file" in script


def test_submission_job_dies_dumps_log(gguf, jobs_dir, monkeypatch, capsys):
    _FakeSlurm(monkeypatch, states=["PENDING\n", "\n"])  # PENDING then gone from queue
    monkeypatch.setattr("boxy.readiness.wait_ready", lambda url, **kw: None)
    jobs._dir()
    jobs.log_path("boxy-m").write_text("srun: error: GPU allocation failed\n")

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "ended before the server became ready" in err
    assert "GPU allocation failed" in err          # the actual job log, surfaced
    assert jobs.read_record("boxy-m") is None      # record reaped


def test_submission_is_idempotent_per_name(gguf, jobs_dir, monkeypatch, capsys):
    _FakeSlurm(monkeypatch, states=["RUNNING\n"])
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "55"})
    jobs.write_endpoint("boxy-m", 9291, job_id="55")
    monkeypatch.setattr("boxy.readiness.wait_ready", lambda url, **kw: "m")

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ALREADY SERVING" in out and "slurm job 55" in out
    assert "sbatch" not in out  # nothing was resubmitted


def test_stop_cancels_scheduler_job(jobs_dir, monkeypatch, capsys):
    import boxy.cli as cli

    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "88"})
    calls = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, **kw: calls.append(list(cmd)) or
                        type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    rc = main(["stop", "boxy-m"])
    assert rc == 0
    assert ["scancel", "88"] in calls
    assert jobs.read_record("boxy-m") is None


def test_list_shows_scheduler_jobs(jobs_dir, monkeypatch, capsys):
    import boxy.cli as cli

    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "99"})
    jobs.write_endpoint("boxy-m", 9291, job_id="99")
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    monkeypatch.setattr(cli, "_container_runtime", lambda loc: (_ for _ in ()).throw(RuntimeError("none")))
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "boxy-m" in out and "slurm job 99" in out and "RUNNING" in out and ":9291/v1" in out


def test_dynamic_scheduler_flags_flow_into_the_script(gguf, jobs_dir, capsys):
    """User request: pass ANY scheduler flag without boxy needing to know it.
    --slurm-FLAG[=VALUE] / --flux-FLAG[=VALUE] translate 1:1 into directives."""
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun",
               "--slurm-qos=long", "--slurm-exclusive", "--slurm-C=gpu_h100",
               "--flux-queue=pbatch"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "#SBATCH --qos=long" in captured.out
    assert "#SBATCH --exclusive" in captured.out          # value-less flag
    assert "#SBATCH -C gpu_h100" in captured.out          # single-char flag
    assert "ignoring --flux-queue" in captured.err        # other scheduler's flag: loud, not silent


def test_dynamic_flags_apply_to_attached_srun_too(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "podman", "--scheduler", "slurm",
               "--accelerator", "cuda", "--gpus", "1", "--foreground", "--dryrun",
               "--slurm-partition=short", "--slurm-C=gpu_h100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "srun" in out and "--partition=short" in out and "-C gpu_h100" in out


def test_typos_still_rejected(gguf, capsys):
    assert main(["serve", str(gguf), "--dryrun", "--not-a-flag"]) == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --not-a-flag" in err
    assert "--slurm-FLAG" in err  # the pass-through convention is advertised


def test_endpoint_file_written_by_serving_side(gguf, jobs_dir, monkeypatch, tmp_path, capsys):
    """The inner (compute-node) serve publishes its endpoint before launching."""
    import boxy.deploy as deploy

    monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "none")
    monkeypatch.setattr(deploy, "execute", lambda d: 0)
    endpoint_file = tmp_path / "ep.json"
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1",
               "--foreground", "--here", "--endpoint-file", str(endpoint_file), "--port", "9391"])
    assert rc == 0
    data = json.loads(endpoint_file.read_text())
    assert data["port"] == 9391 and data["url"].endswith(":9391")
