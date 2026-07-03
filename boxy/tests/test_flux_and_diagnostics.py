"""Field-report fixes: flux batch directive format + GPU spelling, stale
cross-scheduler records, and engine-startup log diagnostics.

Root causes (all reproduced from a live eldorado run):
  1. boxy emitted `#FLUX:` directives; flux's sentinel is lowercase `# flux:`,
     so every directive (nodes/gpus/queue/job-name) was silently dropped and
     the job ran with default resources — "flux is not working".
  2. `flux batch` speaks resource SLOTS, not the `--gpus-per-node` spelling
     that `flux run`/`alloc` accept; boxy emitted the wrong GPU flag.
  3. A stale slurm record was probed with the flux state command, which can
     never recognise a slurm job id, so resubmission wedged on a bogus
     "slurm job ... unreachable".
"""

import pytest

from boxy import diagnostics, jobs
from boxy.cli import main
from boxy.location import Location, Resources
from boxy.schedulers import get_scheduler


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    return tmp_path / "jobs"


@pytest.fixture
def gguf(tmp_path):
    model = tmp_path / "m.q4.gguf"
    model.write_bytes(b"GGUF")
    return model


# ---- flux batch directive format --------------------------------------------

def _flux_script(nodes=2, gpus=4):
    loc = Location(name="e", scheduler="flux",
                   resources=Resources(nodes=nodes, gpus_per_node=gpus))
    return get_scheduler("flux").batch_script(
        "boxy serve m --foreground", loc, "boxy-m", "/tmp/boxy-m.log", [])


def test_flux_directives_use_lowercase_sentinel():
    """flux's directive sentinel is `flux:` (lowercase); `#FLUX:` is ignored."""
    script = _flux_script()
    assert "#FLUX:" not in script
    assert "# flux: --job-name=boxy-m" in script
    assert "# flux: -N2" in script


def test_flux_batch_gpus_use_slot_spelling_not_gpus_per_node():
    """`flux batch` has no --gpus-per-node; GPUs go on slots (-n nodes, -g n)."""
    script = _flux_script(nodes=2, gpus=4)
    assert "--gpus-per-node" not in script
    assert "# flux: -n2" in script
    assert "# flux: -g4" in script


def test_flux_batch_no_gpus_omits_slot_flags():
    script = _flux_script(nodes=1, gpus=0)
    assert "# flux: -N1" in script
    assert "-g" not in script and "# flux: -n" not in script


def test_flux_run_still_uses_gpus_per_node():
    """`flux run`/`alloc` DO accept --gpus-per-node — that path is unchanged."""
    loc = Location(name="e", scheduler="flux", resources=Resources(nodes=2, gpus_per_node=4))
    assert get_scheduler("flux").alloc_command(loc) == ["flux", "alloc", "-N2", "--gpus-per-node=4"]
    assert "--gpus-per-node=4" in get_scheduler("flux").launch_prefix(loc)


def test_flux_queue_directive_lands_in_script(gguf, jobs_dir, capsys):
    """--flux-queue=batch becomes a real, recognised directive under flux."""
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--flux-queue=batch"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# flux: --queue=batch" in out


# ---- stale cross-scheduler record -------------------------------------------

def test_stale_slurm_record_probed_with_slurm_not_flux(gguf, jobs_dir, monkeypatch, capsys):
    """A leftover slurm record must be queried with the slurm state command and
    must not wedge a new flux submission behind a bogus 'unreachable'. Here the
    OTHER scheduler IS installed locally, so boxy stays safe and blocks."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "1786916"})

    seen = {}

    def fake_job_state(scheduler, job_id):
        seen["scheduler_name"] = scheduler.name
        seen["job"] = job_id
        return "UNKNOWN"  # slurm controller not reachable

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", fake_job_state)
    monkeypatch.setattr(cli.shutil, "which", lambda b: "/usr/bin/" + b)  # both schedulers present

    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 1
    # probed the RECORD's scheduler (slurm), not the requested one (flux)
    assert seen["scheduler_name"] == "slurm"
    assert seen["job"] == "1786916"
    err = capsys.readouterr().err
    assert "slurm job 1786916" in err
    assert "submitted under 'slurm'" in err and "you asked for 'flux'" in err
    assert "boxy stop boxy-m" in err


def test_foreign_record_from_another_cluster_does_not_block(gguf, jobs_dir, monkeypatch, capsys):
    """A flux record on a slurm-only host (shared $HOME across clusters) must not
    block a slurm submission — flux isn't installed here, so that job belongs to
    another cluster. boxy takes over the name and submits. Field report: hops."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f2ZoxD9R5fqh"})

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "UNKNOWN")
    # sbatch present (slurm host), flux absent (foreign scheduler)
    monkeypatch.setattr(cli.shutil, "which", lambda b: "/usr/bin/sbatch" if b == "sbatch" else None)

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun", "--name", "boxy-m"])
    assert rc == 0                                   # proceeded, did NOT block
    err = capsys.readouterr().err
    assert "ignoring a stale flux record" in err
    assert "another cluster" in err


def test_matching_scheduler_record_message_unchanged(gguf, jobs_dir, monkeypatch, capsys):
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f123"})
    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already submitted as flux job f123" in err


# ---- --unique: launch multiple of the same model ----------------------------

def test_unique_flag_gives_distinct_coherent_instance_names(gguf, jobs_dir, monkeypatch, capsys):
    """--unique keys the WHOLE instance (job-name, --output log, endpoint file,
    inner --name) off one fresh suffix so N launches of the same model coexist."""
    import re
    import secrets

    tokens = iter(["aaaa", "bbbb"])
    monkeypatch.setattr(secrets, "token_hex", lambda n: next(tokens))

    outs = []
    for _ in range(2):
        assert main(["serve", str(gguf), "--scheduler", "flux", "--gpus", "1",
                     "--unique", "--dryrun"]) == 0
        outs.append(capsys.readouterr().out)

    names = [re.search(r"auto: name: (\S+) \(--unique", o).group(1) for o in outs]
    assert names[0] != names[1]                        # distinct per launch
    for name, out in zip(names, outs):
        assert f"# flux: --job-name={name}" in out      # job carries the name
        assert f"--output=" in out and f"{name}.log" in out  # its own log file
        assert f"--name {name}" in out                  # inner serve + endpoint
        assert f"{name}.endpoint.json" in out


def test_without_unique_name_is_deterministic(gguf, jobs_dir, capsys):
    """Default stays a stable singleton name (reconnect/idempotent submit)."""
    import re

    main(["serve", str(gguf), "--scheduler", "flux", "--dryrun"])
    out1 = capsys.readouterr().out
    main(["serve", str(gguf), "--scheduler", "flux", "--dryrun"])
    out2 = capsys.readouterr().out
    assert "(--unique" not in out1
    j1 = re.search(r"--job-name=(\S+)", out1).group(1)
    j2 = re.search(r"--job-name=(\S+)", out2).group(1)
    assert j1 == j2


def test_unique_instance_name_avoids_existing_record(jobs_dir, monkeypatch):
    """The suffix loop skips a token already taken by a live record."""
    import secrets
    import time

    from boxy import cli, jobs

    monkeypatch.setattr(time, "strftime", lambda fmt: "0101-000000")  # freeze the stamp
    tokens = iter(["dead", "beef"])
    monkeypatch.setattr(secrets, "token_hex", lambda n: next(tokens))
    # pre-seat the record the first token would produce
    first = "boxy-m-0101-000000-dead"
    jobs.write_record(first, {"name": first, "scheduler": "flux", "job": "1"})
    name = cli._unique_instance_name("boxy-m")
    assert name == "boxy-m-0101-000000-beef"   # skipped the taken 'dead' token


# ---- engine-startup diagnostics ---------------------------------------------

VLLM_WEIGHTS_ERR = """
(EngineCore pid=79) ERROR ... EngineCore failed to start.
ValueError: Following weights were not initialized from checkpoint:
{'model.layers.16.input_layernorm.weight', 'model.layers.9.post_attention_layernorm.weight',
 'model.layers.19.input_layernorm.weight'}
"""


def test_diagnose_vllm_weights_not_initialized():
    hint = diagnostics.diagnose(VLLM_WEIGHTS_ERR)
    assert hint is not None
    assert "version mismatch" in hint.lower()
    # names the layernorm-only signature and the concrete fixes
    assert "layernorm.weight" in hint
    assert "--image vllm/vllm-openai" in hint
    assert "llama.cpp" in hint


def test_diagnose_cuda_oom():
    hint = diagnostics.diagnose("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...")
    assert hint is not None and "out of memory" in hint.lower()
    assert "--gpu-memory-utilization" in hint


def test_diagnose_rocm_hip_error():
    hint = diagnostics.diagnose("RuntimeError: HIP error: no kernel image is available for execution")
    assert hint is not None and "gfx" in hint and "rocminfo" in hint


def test_diagnose_engine_core_wrapper_alone_points_up():
    """The outer 'Engine core initialization failed' wrapper with no recognised
    root cause tells the user the real error is higher in the log."""
    wrapper = ("RuntimeError: Engine core initialization failed. See root cause above. "
               "Failed core proc(s): {}")
    hint = diagnostics.diagnose(wrapper)
    assert hint is not None and "actionable error is higher up" in hint


def test_specific_signature_beats_generic_wrapper():
    """When the real cause AND the wrapper are both present, the specific rule
    wins (ordering: generic wrapper is last)."""
    both = (VLLM_WEIGHTS_ERR + "\n"
            "RuntimeError: Engine core initialization failed. See root cause above.")
    hint = diagnostics.diagnose(both)
    assert "version mismatch" in hint.lower()  # weights rule, not the generic wrapper


def test_diagnose_weights_on_nfs_leads_with_load_bug(monkeypatch):
    """On an NFS-backed checkpoint the diagnosis must lead with the eager/re-pull
    fix, not 'version mismatch' (field report: standard Llama-3.1-8B on NFS)."""
    log = (VLLM_WEIGHTS_ERR +
           "\n[weight_utils.py:849] Filesystem type for checkpoints: NFS. Checkpoint size: 10.30 GiB."
           "\n[weight_utils.py:811] Prefetching checkpoint files into page cache started")
    hint = diagnostics.diagnose(log)
    assert "NETWORK filesystem" in hint or "network" in hint.lower()
    assert "--safetensors-load-strategy eager" in hint
    assert "--force" in hint  # re-pull path


def test_diagnose_unknown_load_strategy_flag():
    hint = diagnostics.diagnose("vllm serve: error: unrecognized arguments: "
                                "--safetensors-load-strategy=eager")
    assert hint is not None and "BOXY_NO_VLLM_EAGER=1" in hint


def test_vllm_defaults_to_eager_load_strategy(monkeypatch):
    from boxy.box import Box
    from boxy.engines import build_vllm_serve_cmd
    from boxy.location import Location

    monkeypatch.delenv("BOXY_NO_VLLM_EAGER", raising=False)
    box = Box(name="b", engine="vllm", model="/m", entrypoint="vllm")
    cmd = build_vllm_serve_cmd(box, Location(name="l"), "/m")
    assert "--safetensors-load-strategy=eager" in cmd


def test_vllm_eager_is_overridable_and_env_disablable(monkeypatch):
    from boxy.box import Box
    from boxy.engines import build_vllm_serve_cmd
    from boxy.location import Location

    box = Box(name="b", engine="vllm", model="/m", entrypoint="vllm")
    # user value wins (no duplicate, no override)
    monkeypatch.delenv("BOXY_NO_VLLM_EAGER", raising=False)
    cmd = build_vllm_serve_cmd(box, Location(name="l"), "/m",
                               extra_args=["--safetensors-load-strategy", "prefetch"])
    assert "prefetch" in cmd and "--safetensors-load-strategy=eager" not in cmd
    # env disables it entirely (e.g. vLLM < 0.24)
    monkeypatch.setenv("BOXY_NO_VLLM_EAGER", "1")
    cmd = build_vllm_serve_cmd(box, Location(name="l"), "/m")
    assert not any("safetensors-load-strategy" in a for a in cmd)


def test_pull_force_removes_then_repulls(monkeypatch, capsys):
    """boxy pull --force wipes a cached (possibly corrupt) snapshot before pulling."""
    from boxy import ramalama_shim as s

    calls = {"removed": False, "pulled": False}

    class FakeTransport:
        def remove(self, args):
            calls["removed"] = getattr(args, "ignore", False)  # boxy sets ignore=True
            return True

        def ensure_model_exists(self, args):
            calls["pulled"] = True

        def _get_entry_model_path(self, *a):
            return "/store/model"

    monkeypatch.setattr(s, "ensure_trust_bundle", lambda: None)
    monkeypatch.setattr(s, "_store_args", lambda uri, **k: __import__("types").SimpleNamespace())
    import ramalama.transports.transport_factory as tf

    monkeypatch.setattr(tf, "New", lambda uri, args: FakeTransport())
    path = s.pull_model("hf://o/m", force=True)
    assert path == "/store/model"
    assert calls["removed"] is True and calls["pulled"] is True
    assert "removed cached" in capsys.readouterr().err


def test_diagnose_unknown_returns_none():
    assert diagnostics.diagnose("Server started on port 8000. Ready.") is None
    assert diagnostics.diagnose("") is None
