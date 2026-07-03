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
    must not wedge a new flux submission behind a bogus 'unreachable'."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "1786916"})

    seen = {}

    def fake_job_state(scheduler, job_id):
        seen["scheduler_name"] = scheduler.name
        seen["job"] = job_id
        return "UNKNOWN"  # slurm not reachable on this flux-only login node

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", fake_job_state)

    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 1
    # probed the RECORD's scheduler (slurm), not the requested one (flux)
    assert seen["scheduler_name"] == "slurm"
    assert seen["job"] == "1786916"
    err = capsys.readouterr().err
    assert "slurm job 1786916" in err
    assert "submitted under 'slurm'" in err and "you asked for 'flux'" in err
    assert "boxy stop boxy-m" in err


def test_matching_scheduler_record_message_unchanged(gguf, jobs_dir, monkeypatch, capsys):
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f123"})
    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already submitted as flux job f123" in err


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


def test_diagnose_unknown_returns_none():
    assert diagnostics.diagnose("Server started on port 8000. Ready.") is None
    assert diagnostics.diagnose("") is None
