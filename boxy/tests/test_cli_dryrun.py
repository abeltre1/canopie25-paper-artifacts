"""End-to-end dry-run: the full pipeline must reproduce the prototype's
command shape for the paper's Eldorado (flux+apptainer+rocm) and HOPS
(slurm+podman+cuda) deployments."""

from boxy import deploy
from boxy.cli import main
from tests.conftest import EXAMPLES


def test_plan_serve_eldorado_end_to_end(vllm_box, eldorado):
    d = deploy.plan_serve(vllm_box, eldorado, dryrun=True)
    cmd = d.command
    # Scheduler wrap outermost
    assert cmd[:2] == ["flux", "run"]
    # Module preamble (bash -lc "module load rocm/6.4.0 && exec apptainer ...")
    assert "bash" in cmd and any("module load rocm/6.4.0" in part for part in cmd)
    script = cmd[-1]
    # Apptainer + GPU + SIF + inner vllm command inside the wrapped script
    for token in ("apptainer exec", "--rocm", "--fakeroot", "vllm-rocm.sif", "vllm serve"):
        assert token in script, f"missing {token!r} in: {script}"
    # Offline env + determinism + tack-ons (never overriding user args)
    for token in ("HF_HUB_OFFLINE=1", "--tensor-parallel-size=4", "--seed=12345", "--gpu-memory-utilization=0.7"):
        assert token in script, f"missing {token!r} in: {script}"
    # SIF auto-build planned (prototype build_apptainer_image)
    assert d.prepare_commands[0][:3] == ["apptainer", "build", "--force"]


def test_plan_serve_hops_end_to_end(vllm_box, hops):
    d = deploy.plan_serve(vllm_box, hops, dryrun=True)
    cmd = d.command
    assert cmd[:3] == ["srun", "--nodes=2", "--gpus-per-node=4"]
    assert "podman" in cmd and "run" in cmd
    i = cmd.index("--device")
    assert cmd[i + 1] == "nvidia.com/gpu=all"
    # Model is a relative path in the shared models dir (paper flow)
    assert "Llama-4-Scout-17B-16E-Instruct" in cmd
    assert d.prepare_commands == []  # OCI runtime: nothing to build


def test_user_args_never_overridden(vllm_box, eldorado):
    d = deploy.plan_serve(vllm_box, eldorado, extra_args=["--seed=7", "--tensor-parallel-size=2"], dryrun=True)
    script = d.command[-1]
    assert "--seed=7" in script and "--seed=12345" not in script
    assert "--tensor-parallel-size=2" in script and "--tensor-parallel-size=4" not in script


def test_cli_serve_dryrun_examples(capsys):
    rc = main(
        [
            "serve",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "eldorado.toml"),
            "--dryrun",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Running Command:" in out
    assert "flux run" in out and "apptainer" in out and "vllm-rocm.sif" in out


def test_cli_run_passthrough_dryrun(capsys):
    rc = main(
        [
            "run",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "hops.toml"),
            "--dryrun",
            "--",
            "serve", "some-model", "--max-model-len=4096",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "srun" in out and "podman" in out and "--max-model-len=4096" in out


def test_cli_info_runs(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "accelerator:" in out


def test_cli_build_oci_noop(capsys):
    rc = main(
        [
            "build",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "hops.toml"),
            "--dryrun",
        ]
    )
    assert rc == 0
    assert "nothing to build" in capsys.readouterr().out


def test_cli_pull_path_model_noop(capsys):
    rc = main(["pull", "--box", str(EXAMPLES / "boxes" / "vllm.toml"), "--dryrun"])
    assert rc == 0
    assert "nothing to pull" in capsys.readouterr().out


def test_cli_stub_commands(capsys):
    assert main(["alloc"]) == 2
    assert "not implemented in the MVP" in capsys.readouterr().err


def test_stage_without_target_shows_usage(capsys, monkeypatch):
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    assert main(["stage"]) == 2
    assert "usage: boxy stage" in capsys.readouterr().err
