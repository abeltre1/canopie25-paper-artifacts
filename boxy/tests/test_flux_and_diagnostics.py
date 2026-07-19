"""Field-report fixes: flux batch directive format + GPU spelling, stale
cross-scheduler records, and engine-startup log diagnostics.

Root causes (all reproduced from a live clusterb run):
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


def test_output_log_is_unique_per_job_via_scheduler_token(gguf, jobs_dir, capsys):
    """The scheduler --output carries the job-id token (%j / {{id}}) so repeated
    submissions never overwrite each other's logs."""
    main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun"])
    out = capsys.readouterr().out
    assert "#SBATCH --output=" in out and "-%j.log" in out
    main(["serve", str(gguf), "--scheduler", "flux", "--dryrun"])
    out = capsys.readouterr().out
    assert "# flux: --output=" in out and "-{{id}}.log" in out


def test_flux_queue_directive_lands_in_script(gguf, jobs_dir, capsys):
    """--flux-queue=batch becomes a real, recognised directive under flux."""
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--flux-queue=batch"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# flux: --queue=batch" in out


# ---- stale cross-scheduler record -------------------------------------------

def test_stale_record_probed_with_its_own_scheduler(gguf, jobs_dir, monkeypatch, capsys):
    """A leftover slurm record must be queried with the SLURM state command, not
    the requested flux one (a slurm job id is meaningless to `flux jobs`)."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "1786916"})

    seen = {}

    def fake_job_state(scheduler, job_id):
        seen["scheduler_name"] = scheduler.name
        seen["job"] = job_id
        return "UNKNOWN"

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", fake_job_state)
    main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    # probed the RECORD's scheduler (slurm), not the requested one (flux)
    assert seen["scheduler_name"] == "slurm"
    assert seen["job"] == "1786916"


def test_foreign_record_from_another_cluster_does_not_block(gguf, jobs_dir, monkeypatch, capsys):
    """A different-scheduler job we can't reach (shared $HOME across clusters:
    an clusterb flux record on a clustera slurm login node) must NOT block a local
    submission — boxy takes over the name and submits. Field report: clustera."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f2Zp2gRTakud"})

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "UNKNOWN")  # flux job unreachable here

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun", "--name", "boxy-m"])
    assert rc == 0                                   # proceeded, did NOT block
    err = capsys.readouterr().err
    assert "ignoring a stale flux record" in err
    assert "another cluster" in err


def test_same_scheduler_unknown_still_blocks(gguf, jobs_dir, monkeypatch, capsys):
    """Safety unchanged: a SAME-scheduler job we can't confirm (controller flap)
    must NOT be resubmitted — that would double-submit a maybe-live job."""
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "slurm", "job": "42"})

    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "UNKNOWN")

    rc = main(["serve", str(gguf), "--scheduler", "slurm", "--dryrun", "--name", "boxy-m"])
    assert rc == 1                                   # blocked, not resubmitted
    err = capsys.readouterr().err
    assert "Not resubmitting" in err and "scheduler unreachable" in err


def test_no_auto_unique_keeps_singleton_block_message(gguf, jobs_dir, monkeypatch, capsys):
    # With auto-unique disabled, a live same-scheduler job still hard-blocks
    # (the strict singleton) with the unchanged guidance message.
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f123"})
    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--no-auto-unique",
               "--dryrun", "--name", "boxy-m"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already submitted as flux job f123" in err


def test_auto_unique_default_forks_on_live_scheduler_job(gguf, jobs_dir, monkeypatch, capsys):
    # DEFAULT (no flag): a live same-scheduler instance with NOTHING serving yet
    # (no endpoint published) is not blocked — boxy forks a fresh instance.
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f123"})
    from boxy import cli
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "starting an independent instance" in out
    assert "# flux: --job-name=boxy-m-" in out            # forked, not the base 'boxy-m'


def test_auto_unique_does_not_duplicate_when_endpoint_published(gguf, jobs_dir, monkeypatch, capsys):
    # Adversarial-review regression: a RUNNING instance that already published an
    # endpoint must NOT be forked into a duplicate GPU job just because the 2s
    # readiness probe was slow — report the live endpoint instead.
    jobs.write_record("boxy-m", {"name": "boxy-m", "scheduler": "flux", "job": "f123"})
    jobs.write_endpoint("boxy-m", 8000, "f123")           # server came up (endpoint published)
    from boxy import cli, readiness
    monkeypatch.setattr(cli, "_job_state", lambda s, j: "RUNNING")
    monkeypatch.setattr(readiness, "wait_ready", lambda *a, **k: None)  # transient probe miss
    rc = main(["serve", str(gguf), "--scheduler", "flux", "--dryrun", "--name", "boxy-m"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ALREADY SERVING" in out
    assert "starting an independent instance" not in out  # NOT forked
    assert "# flux: --job-name=boxy-m-" not in out        # no second job built


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
        assert "--output=" in out and (name + "-{{id}}.log") in out  # its own per-job log
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


def test_diagnose_dockerhub_403_registry_block():
    # the field failure: an agentless vLLM job on clustera dies pulling from Docker Hub
    # (Zscaler 403). boxy must recognize it and point at the fixes.
    log = ('Trying to pull docker.io/vllm/vllm-openai:latest...\n'
           'Error: initializing source docker://vllm/vllm-openai:latest: pinging container '
           'registry registry-1.docker.io: StatusCode: 403, "<html>...Zs...')
    hint = diagnostics.diagnose(log)
    assert hint is not None
    assert "registry blocked" in hint.lower()
    assert "Zscaler" in hint or "air-gapped" in hint
    assert "--registry" in hint            # names the mirror fix
    assert "pre-pull" in hint.lower()      # and the login-node pre-pull fix


def test_looks_like_pull_block_recognizes_the_field_error():
    from boxy.cli import _looks_like_pull_block
    assert _looks_like_pull_block(
        "initializing source docker://vllm/vllm-openai: pinging container registry "
        "registry-1.docker.io: StatusCode: 403")
    assert not _looks_like_pull_block("vllm: Application startup complete.")


def test_diagnose_missing_python_package():
    log = ("ImportError: This modeling file requires the following packages that were not found "
           "in your environment: open_clip. Run `pip install open_clip`")
    hint = diagnostics.diagnose(log)
    assert hint is not None
    assert "open_clip" in hint
    assert "--image" in hint and "pip install" in hint


def test_diagnose_cert_verify_failed():
    log = ("'[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local "
           "issuer certificate (_ssl.c:1010)' thrown while requesting HEAD https://huggingface.co/...")
    hint = diagnostics.diagnose(log)
    assert hint is not None and "site CA" in hint
    assert "SSL_CERT_FILE" in hint


def test_diagnose_trust_remote_code():
    log = ("pydantic_core._pydantic_core.ValidationError: The repository /mnt/models/x contains "
           "custom code which must be executed to correctly load the model. Please pass the argument "
           "`trust_remote_code=True` to allow custom code to be run.")
    hint = diagnostics.diagnose(log)
    assert hint is not None and "trust_remote_code" in hint
    assert "--trust-remote-code" in hint


def test_diagnose_cuda_oom():
    hint = diagnostics.diagnose("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...")
    assert hint is not None and "out of memory" in hint.lower()
    assert "--gpu-memory-utilization" in hint


def test_diagnose_rocm_hip_error():
    hint = diagnostics.diagnose("RuntimeError: HIP error: no kernel image is available for execution")
    assert hint is not None and "gfx" in hint and "rocminfo" in hint


def test_diagnose_image_pull_403_not_gguf(monkeypatch):
    # The exact clustera compute-node failure: ghcr.io 403 via Zscaler. The log's
    # model repo name ends in '-GGUF', which must NOT trigger the gguf-load rule;
    # boxy must diagnose the IMAGE pull block and point at pre-pull / --registry.
    log = ("Trying to pull ghcr.io/ggml-org/llama.cpp:server-cuda...\n"
           "Error: initializing source docker://ghcr.io/ggml-org/llama.cpp:server-cuda: "
           "pinging container registry ghcr.io: StatusCode: 403, \"<html>...Zs...\"\n"
           "  auto: model: hf://hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/"
           "llama-3.2-1b-instruct-q4_k_m.gguf")
    hint = diagnostics.diagnose(log)
    assert hint is not None
    assert "IMAGE could not be pulled" in hint and "registry blocked" in hint
    assert "podman pull ghcr.io/ggml-org/llama.cpp:server-cuda" in hint
    assert "--registry" in hint
    assert "could not load the GGUF" not in hint          # the old misdiagnosis is gone


def test_diagnose_gguf_load_still_matches_real_load_error():
    # a genuine llama.cpp load failure still gets the GGUF advice
    hint = diagnostics.diagnose("llama_load_model_from_file: failed to load model")
    assert hint is not None and "could not load the GGUF" in hint
    # but a bare repo name mentioning GGUF with no load error does NOT
    assert diagnostics.diagnose("pulling hugging-quants/...-Q4_K_M-GGUF ok") is None


def test_diagnose_host_oom_exit_137():
    # a second local instance reaped by the podman/docker VM OOM killer: often
    # empty logs, so boxy synthesizes 'OOMKilled exit 137' from the exit code.
    hint = diagnostics.diagnose("some early load lines\nOOMKilled exit 137")
    assert hint is not None and "HOST/VM memory" in hint
    assert "podman machine set --memory" in hint and "--unique" in hint


def test_diagnose_host_oom_beats_nothing_but_not_gpu_oom():
    # host-OOM matches 'Killed'/alloc-failure signatures...
    assert "HOST/VM memory" in (diagnostics.diagnose("llama_model_load: Killed") or "")
    # ...but a GPU OOM still gets the GPU-specific advice (cuda-oom is earlier)
    gpu = diagnostics.diagnose("torch.cuda.OutOfMemoryError: CUDA out of memory")
    assert gpu is not None and "--gpu-memory-utilization" in gpu


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


def test_version_string_reports_git_commit_in_a_checkout(tmp_path, monkeypatch):
    """boxy info / --version must expose the checkout's commit so a stale editable
    install is obvious (field: `boxy 0.1.0` alone can't tell old from new)."""
    import boxy

    # fake a checkout: <root>/.git/HEAD -> a branch ref with a slash in it
    git = tmp_path / "repo" / ".git"
    (git / "refs" / "heads" / "claude").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/claude/my-branch\n")
    (git / "refs" / "heads" / "claude" / "my-branch").write_text("abcdef1234567890\n")
    pkg = tmp_path / "repo" / "boxy" / "src" / "boxy"
    pkg.mkdir(parents=True)
    monkeypatch.setattr(boxy, "__file__", str(pkg / "__init__.py"))

    sha, branch = boxy._read_git_revision()
    assert sha == "abcdef1" and branch == "claude/my-branch"   # full branch, short sha
    assert "git abcdef1" in boxy.version_string() and "claude/my-branch" in boxy.version_string()


def test_version_string_plain_outside_a_checkout(tmp_path, monkeypatch):
    import boxy

    loose = tmp_path / "site-packages" / "boxy"
    loose.mkdir(parents=True)
    monkeypatch.setattr(boxy, "__file__", str(loose / "__init__.py"))
    assert boxy.version_string() == boxy.__version__


# ---- incomplete-checkpoint guard --------------------------------------------

def _write_safetensors(path, tensor_names):
    """Minimal valid safetensors: 8-byte header length + JSON header."""
    import json
    import struct

    header = {name: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for name in tensor_names}
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\x00\x00")


def _write_index(path, weight_map):
    import json

    json.dump({"weight_map": weight_map}, open(path, "w"))


def test_verify_safetensors_complete_passes_when_all_tensors_present(tmp_path):
    from boxy import ramalama_shim as s

    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", ["a", "b"])
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", ["c"])
    _write_index(tmp_path / "model.safetensors.index.json",
                 {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors",
                  "c": "model-00002-of-00002.safetensors"})
    assert s.verify_safetensors_complete(str(tmp_path)) == []


def test_verify_flags_missing_shard_file(tmp_path):
    from boxy import ramalama_shim as s

    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", ["a"])  # shard 2 never written
    _write_index(tmp_path / "model.safetensors.index.json",
                 {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"})
    problems = s.verify_safetensors_complete(str(tmp_path))
    assert problems and "shard file" in problems[0]


def test_verify_flags_tensor_present_in_index_absent_in_shards(tmp_path):
    """The field case: shard files all present, but they don't contain every
    tensor the index declares (the layernorms)."""
    from boxy import ramalama_shim as s

    _write_safetensors(tmp_path / "model-00001-of-00001.safetensors", ["a"])  # missing 'b'
    _write_index(tmp_path / "model.safetensors.index.json",
                 {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"})
    problems = s.verify_safetensors_complete(str(tmp_path))
    assert problems and "incomplete checkpoint" in problems[0] and "b" in problems[0]


def test_verify_noop_for_non_safetensors(tmp_path):
    from boxy import ramalama_shim as s

    assert s.verify_safetensors_complete(str(tmp_path / "nope")) == []   # missing dir
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    assert s.verify_safetensors_complete(str(tmp_path)) == []            # no index -> skip


def test_serve_fast_fails_on_incomplete_checkpoint(tmp_path, monkeypatch):
    """plan_serve raises before launch when a vLLM safetensors checkpoint is
    incomplete, instead of letting vLLM burn the allocation."""
    import pytest

    from boxy import deploy
    from boxy.box import Box
    from boxy.location import Location

    model = tmp_path / "Llama"
    model.mkdir()
    _write_safetensors(model / "model-00001-of-00001.safetensors", ["a"])
    _write_index(model / "model.safetensors.index.json",
                 {"a": "model-00001-of-00001.safetensors", "b": "model-00001-of-00001.safetensors"})
    box = Box(name="b", engine="vllm", model=str(model), image="vllm/vllm-openai", entrypoint="vllm")
    monkeypatch.delenv("BOXY_NO_MODEL_VERIFY", raising=False)
    with pytest.raises(RuntimeError, match="incomplete/corrupt"):
        deploy.plan_serve(box, Location(name="l", accelerator="cuda", runtime="podman"))
    # opt-out lets it through
    monkeypatch.setenv("BOXY_NO_MODEL_VERIFY", "1")
    deploy.plan_serve(box, Location(name="l", accelerator="cuda", runtime="podman"))  # no raise


def test_diagnose_unknown_returns_none():
    assert diagnostics.diagnose("Server started on port 8000. Ready.") is None
    assert diagnostics.diagnose("") is None


def test_flux_time_converted_to_fsd():
    """Field report: `--time 30:00` reached flux as `-t 30:00` -> 'invalid Flux
    standard duration'. The portable --time converts formats, not just names."""
    from boxy.schedulers.flux import _to_fsd

    assert _to_fsd("30:00") == "1800s"        # MM:SS (the failing case)
    assert _to_fsd("1:30:00") == "5400s"      # HH:MM:SS
    assert _to_fsd("30") == "1800s"           # bare number = minutes (Slurm)
    assert _to_fsd("2-12") == "216000s"       # D-HH
    assert _to_fsd("2-12:30:15") == "217815s" # D-HH:MM:SS
    assert _to_fsd("30m") == "30m"            # already FSD: untouched
    assert _to_fsd("1.5h") == "1.5h"
    assert _to_fsd("nonsense") == "nonsense"  # let flux produce its own error

    from boxy.schedulers import get_scheduler
    assert get_scheduler("flux").site_directive("time", "30:00") == "-t 1800s"
    assert get_scheduler("slurm").site_directive("time", "30:00") == "--time=30:00"  # slurm untouched


def test_engine_core_wrapper_extracts_root_cause():
    # FIELD (clusterb): the user pasted 60 lines of the vLLM wrapper traceback;
    # the real exception had scrolled away. The generic wrapper diagnosis now
    # extracts the inner error line itself when the window contains it.
    from boxy import diagnostics

    log = ("(EngineCore_DP0 pid=44) Traceback (most recent call last):\n"
           '(EngineCore_DP0 pid=44)   File "worker.py", line 12, in init_device\n'
           "(EngineCore_DP0 pid=44) Weird.ObscureError: /dev/shm too small for MoE buffers\n"
           "(APIServer pid=1) RuntimeError: Engine core initialization failed. "
           "See root cause above. Failed core proc(s): {}\n")
    out = diagnostics.diagnose(log)
    assert "root cause extracted" in out
    assert "/dev/shm too small for MoE buffers" in out
    # window too small to contain it: still the pointer, never a fabricated cause
    out = diagnostics.diagnose("RuntimeError: Engine core initialization failed. "
                               "See root cause above.")
    assert "actionable error is higher up" in out and "extracted" not in out


def test_diagnose_nccl_shm_too_small_beats_generic_wrapper():
    # FIELD (clusterb, Nemotron-3 Nano TP=2 on ROCm): both workers died at
    # ncclCommInitRank with 'NCCL error: unhandled system error' — RCCL could
    # not create its shared-memory segments in podman's default 64MB /dev/shm
    # (the Mac-rendered agentless script had dropped --ipc=host). The specific
    # remedy must beat the generic engine-core wrapper even when both appear.
    log = (
        "(Worker pid=120) ERROR [multiproc_executor.py:898] WorkerProc failed to start.\n"
        "(Worker pid=120) ERROR ...   self.comm: ncclComm_t = self.nccl.ncclCommInitRank(\n"
        "(Worker pid=120) ERROR ... RuntimeError: NCCL error: unhandled system error "
        "(run with NCCL_DEBUG=INFO for details)\n"
        "(EngineCore pid=85) ERROR ... Engine core initialization failed.\n"
    )
    hint = diagnostics.diagnose(log)
    assert hint is not None
    assert "shared memory" in hint
    assert "--ipc=host" in hint and "--shm-size" in hint
    assert "NCCL_DEBUG=INFO" in hint
