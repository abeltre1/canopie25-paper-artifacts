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
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: True)  # CI runners have no squeue
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    monkeypatch.setattr(cli, "_container_runtime", lambda loc: (_ for _ in ()).throw(RuntimeError("none")))
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "boxy-m" in out and "slurm job 99" in out and "RUNNING" in out and ":9291/v1" in out


def test_list_reveals_exited_containers_with_oom_note(monkeypatch, capsys):
    """Field report: `--unique` instances kept 'disappearing'. Plain `ps` hides
    EXITED containers, so a server killed seconds after READY vanished from view.
    `boxy list` must surface them with exit code + an OOM note when exit==137."""
    import boxy.cli as cli

    def fake_run(cmd, **kw):
        out = ""
        if cmd[1:3] == ["ps", "-a"]:
            out = "boxy-a-0707\tExited (137) 2 minutes ago\nboxy-b-0707\tExited (0) 1 minute ago\n"
        elif cmd[1] == "inspect":
            out = "137 true" if "boxy-a-0707" in cmd else "0 false"
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cli._report_exited_containers("podman")
    out = capsys.readouterr().out
    assert "boxy-a-0707" in out and "exit 137" in out and "OOM/SIGKILL" in out
    assert "boxy-b-0707" in out and "exit 0" in out
    assert "podman machine set --memory" in out          # OOM fix shown (137 present)
    assert "podman logs <name>" in out


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


def test_sched_neutral_flags_follow_the_active_scheduler(gguf, jobs_dir, capsys):
    """--sched-FLAG[=VALUE] is scheduler-NEUTRAL: the SAME command renders under
    whichever --scheduler is active (the scheduler flag dictates what to do)."""
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun",
               "--sched-license=tscratch:1", "--sched-exclusive"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "#SBATCH --license=tscratch:1" in out and "#SBATCH --exclusive" in out

    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun",
               "--sched-license=tscratch:1", "--sched-exclusive"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "# flux: --license=tscratch:1" in captured.out and "# flux: --exclusive" in captured.out
    assert "ignoring" not in captured.err                  # neutral flags are never dropped


def test_dynamic_flags_apply_to_attached_srun_too(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "podman", "--scheduler", "slurm",
               "--accelerator", "cuda", "--gpus", "1", "--foreground", "--dryrun",
               "--slurm-partition=short", "--slurm-C=gpu_h100"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "srun" in out and "--partition=short" in out and "-C gpu_h100" in out


def test_submission_hint_for_account_partition_rejection():
    from boxy.cli import _submission_hint

    hint = _submission_hint("sbatch: error: Batch job submission failed: "
                            "Invalid account or account/partition combination specified")
    assert "sacctmgr show assoc" in hint and "sinfo -s" in hint
    assert "single partition" in hint
    assert _submission_hint("some other failure") == ""


def test_bare_flags_pass_to_the_active_scheduler(gguf, jobs_dir, capsys):
    """The user's spelling: NO prefix — any flag boxy doesn't own goes to the
    active scheduler verbatim; the portable trio is translated internally."""
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun",
               "--account=fy260064", "--partition=short,batch", "--license=tscratch:1",
               "--time", "30:00", "--qos=long"])
    out = capsys.readouterr().out
    assert rc == 0
    for d in ("--partition=short,batch", "--account=fy260064", "--time=30:00",
              "--license=tscratch:1", "--qos=long"):
        assert f"#SBATCH {d}" in out

    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun",
               "--account=fy260064", "--partition=short,batch", "--license=tscratch:1"])
    out = capsys.readouterr().out
    assert rc == 0
    # SAME command; flux spellings chosen internally
    assert "# flux: --queue=short,batch" in out and "# flux: --bank=fy260064" in out
    assert "# flux: --license=tscratch:1" in out


def test_typo_of_a_boxy_flag_errors_with_suggestion(gguf, capsys):
    # a near-miss of a real boxy flag must NEVER silently become a scheduler flag
    assert main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun", "--replcias=3"]) == 2
    err = capsys.readouterr().err
    assert "did you mean --replicas?" in err


def test_unknown_flag_without_scheduler_warns_loudly(gguf, capsys):
    rc = main(["serve", str(gguf), "--here", "--runtime", "docker",
               "--accelerator", "none", "--dryrun", "--not-a-flag"])
    assert rc == 0
    assert "ignoring --not-a-flag" in capsys.readouterr().err  # loud, never silent


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


# ---- round 2: submission auditor ----

def test_r2_unreachable_scheduler_is_unknown_not_done(monkeypatch):
    """r2-S1 (major, reproduced live): squeue's connect-failure signature
    (rc!=0, empty stdout) was read as DONE — live jobs got reaped, duplicated,
    and boxy stop cancelled the wrong job."""
    import boxy.cli as cli
    from boxy.schedulers import get_scheduler

    def fake_run(cmd, **kw):
        return type("R", (), {"returncode": 1, "stdout": "",
                              "stderr": "slurm_load_jobs error: Unable to contact slurm controller"})()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._job_state(get_scheduler("slurm"), "11") == "UNKNOWN"

    def fake_run2(cmd, **kw):
        return type("R", (), {"returncode": 1, "stdout": "",
                              "stderr": "slurm_load_jobs error: Invalid job id specified"})()

    monkeypatch.setattr(cli.subprocess, "run", fake_run2)
    assert cli._job_state(get_scheduler("slurm"), "11") == "DONE"


def test_r2_unknown_state_never_resubmits(gguf, jobs_dir, monkeypatch, capsys):
    import boxy.cli as cli

    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "11"})
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "UNKNOWN")
    monkeypatch.setattr("time.sleep", lambda s: None)
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m"])
    err = capsys.readouterr()
    assert rc == 1
    assert "Not resubmitting" in err.err
    assert jobs.read_record("boxy-m") is not None        # record NOT reaped
    assert "sbatch" not in err.out                       # no duplicate job


def test_r2_wait_loop_bails_after_unknown_streak(gguf, jobs_dir, monkeypatch, capsys):
    """r2-S2 (major, reproduced): the wait loop spun forever, silently."""
    _FakeSlurm(monkeypatch, states=["SUSPENDEDX\n"])  # unmapped -> UNKNOWN forever
    monkeypatch.setattr("boxy.readiness.wait_ready", lambda url, **kw: None)
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "cannot determine job" in err and "boxy stop boxy-m" in err


def test_r2_suspended_is_alive_not_unknown():
    from boxy.schedulers import get_scheduler

    slurm = get_scheduler("slurm")
    for state in ("SUSPENDED", "REQUEUED", "RESIZING"):
        assert slurm.interpret_state(state) == "PENDING"


def test_r2_junk_endpoint_json_is_not_published(jobs_dir):
    jobs.endpoint_path("boxy-m").parent.mkdir(parents=True, exist_ok=True)
    jobs.endpoint_path("boxy-m").write_text('{"name": "boxy-m", "host": "vm", "port": 1}')  # no url
    assert jobs.read_endpoint("boxy-m") is None          # was: KeyError downstream
    jobs.endpoint_path("boxy-m").write_text('"just a string"')
    assert jobs.read_endpoint("boxy-m") is None
    jobs.write_endpoint("boxy-m", 8090, job_id="1")      # the real writer round-trips
    assert jobs.read_endpoint("boxy-m")["url"].endswith(":8090")


def test_r2_directive_values_with_spaces_are_quoted(gguf, jobs_dir, capsys):
    rc = main(["serve", str(gguf), "--scheduler", "slurm",
               "--scheduler-arg=--comment=hello world", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '#SBATCH --comment="hello world"' in out      # was: 'Invalid directive: world'


def test_r2_dryrun_does_not_reap_job_state(gguf, jobs_dir, monkeypatch, capsys):
    """r2-S6: --dryrun deleted stale records and endpoint files."""
    import boxy.cli as cli

    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "24"})
    jobs.write_endpoint("boxy-m", 8090)
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "DONE")
    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--name", "boxy-m", "--dryrun"])
    assert rc == 0
    assert jobs.read_record("boxy-m") is not None        # untouched by dryrun
    rc = main(["list", "--dryrun", "--runtime", "docker"])
    assert jobs.read_record("boxy-m") is not None


def test_list_labels_foreign_cluster_records(jobs_dir, monkeypatch, capsys):
    """Shared $HOME: another cluster's record (a scheduler this host can't even
    speak) must list as FOREIGN(origin), not UNKNOWN — and must not be probed.
    Field report: an eldorado flux job listed on hops as UNKNOWN."""
    from boxy import cli, jobs

    jobs.write_record("boxy-eldo", {"name": "boxy-eldo", "scheduler": "flux",
                                    "job": "f2agHnM4psaw", "submitted_from": "eldorado-login2"})
    jobs.write_record("boxy-here", {"name": "boxy-here", "scheduler": "slurm", "job": "77",
                                    "submitted_from": "hops-login1"})
    probed = []
    # cluster identity decides (deterministic on any host, incl. CI runners):
    # this host "is" hops, so the eldorado-submitted record is the foreign one
    monkeypatch.setenv("BOXY_CLUSTER", "hops")
    monkeypatch.setattr(cli, "_job_state", lambda s, j: probed.append(j) or "RUNNING")
    rc = main(["list", "--runtime", "docker", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "FOREIGN(eldorado-login2)" in out          # labeled with its origin
    assert "boxy-here" in out and "RUNNING" in out    # local record still probed
    assert probed == ["77"]                           # the foreign job was NOT probed
    assert "shares this $HOME" in out                 # the explainer footnote
    assert jobs.read_record("boxy-eldo") is not None  # never reaped


def test_cluster_identity_and_fallback(monkeypatch):
    """FOREIGN classification keys on WHERE the record was submitted, not on
    which scheduler binaries happen to be on PATH (field report: hops ships
    `flux` too, so an eldorado flux record passed the binary check and boxy
    curl chased eldo1025). Legacy records keep the binary fallback."""
    from boxy import cli

    assert cli._cluster_id("eldorado-login2") == "eldorado"
    assert cli._cluster_id("eldorado-login1.sandia.gov") == "eldorado"
    assert cli._cluster_id("hops12") == "hops"
    assert cli._cluster_id("HOPS-LOGIN5") == "hops"
    monkeypatch.setenv("BOXY_CLUSTER", "hops")   # this host "is" hops
    foreign, origin = cli._record_is_foreign({"scheduler": "flux", "submitted_from": "eldorado-login2"})
    assert foreign and origin == "eldorado-login2"
    foreign, _ = cli._record_is_foreign({"scheduler": "slurm", "submitted_from": "hops-login1"})
    assert not foreign
    # legacy record without an origin: binary-presence fallback still applies
    monkeypatch.setattr(cli, "_scheduler_reachable", lambda s: False)
    foreign, origin = cli._record_is_foreign({"scheduler": "slurm"})
    assert foreign and origin == "another cluster"


def test_boxy_logs_newest_named_and_diagnosed(jobs_dir, monkeypatch, capsys):
    """boxy logs: newest log by default, NAME-prefix filter, crash diagnosis
    appended — and it works after the record was reaped (files outlive jobs)."""
    import time

    from boxy import jobs

    d = jobs._dir()
    (d / "boxy-old-1.log").write_text("old noise\n")
    time.sleep(0.01)
    (d / "boxy-tiny-99.log").write_text(
        "loading model\nRuntimeError: HIP error: no kernel image is available for execution\n")
    rc = main(["logs"])                      # no name -> newest
    out = capsys.readouterr().out
    assert rc == 0
    assert "boxy-tiny-99.log" in out and "HIP error" in out
    assert "boxy diagnosis:" in out and "gfx" in out    # diagnosis appended
    rc = main(["logs", "boxy-old"])          # prefix filter
    out = capsys.readouterr().out
    assert rc == 0 and "old noise" in out
    rc = main(["logs", "nope"])              # helpful error
    assert rc == 2
    assert "boxy-tiny-99.log" in capsys.readouterr().err
