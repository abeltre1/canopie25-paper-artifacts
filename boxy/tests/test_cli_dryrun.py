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
    # eldorado is a 2-node vLLM location, so it now auto-distributes; --no-distributed
    # degrades it to the classic single wrapped container (flux run -> apptainer SIF).
    rc = main(
        [
            "serve",
            "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
            "--location", str(EXAMPLES / "locations" / "eldorado.toml"),
            "--no-distributed",
            "--dryrun",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Running Command:" in out
    assert "flux run" in out and "apptainer" in out and "vllm-rocm.sif" in out


def test_cli_serve_dryrun_distributed_flux(capsys):
    # the same 2-node vLLM location, distributed: a Ray head (runs directly) plus a
    # flux-run worker fan-out to the other node, with TP/PP derived from geometry.
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
    assert "### Head" in out and "### Worker" in out
    assert "ray start --head" in out and "ray start --address=${BOXY_RAY_HEAD}" in out
    assert "--tensor-parallel-size=4" in out and "--pipeline-parallel-size=2" in out
    assert "--distributed-executor-backend=ray" in out
    # flux location -> workers placed with flux run (not local containers)
    assert "flux run" in out and "apptainer" in out and "vllm-rocm.sif" in out


def test_ca_bundle_propagated_into_container(vllm_box, hops, tmp_path, monkeypatch):
    # boxy's merged CA is mounted into the container + the TLS env points at it, so
    # in-container HuggingFace downloads trust the site CA.
    ca = tmp_path / "ca-merged.crt"
    ca.write_text("-----BEGIN CERTIFICATE-----\nx\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    d = deploy.plan_serve(vllm_box, hops, dryrun=True)
    cmd = " ".join(d.command)
    assert f"{ca}:/etc/ssl/certs/boxy-ca-merged.pem:ro" in cmd
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        assert f"{var}=/etc/ssl/certs/boxy-ca-merged.pem" in cmd


def test_ca_bundle_not_propagated_for_bare_site_cert(vllm_box, hops, tmp_path, monkeypatch):
    # a bare site CA (not boxy's merged bundle) must NOT be mounted — replacing the
    # container trust with a site-only cert would break public HTTPS (huggingface.co).
    bare = tmp_path / "site.crt"
    bare.write_text("-----BEGIN CERTIFICATE-----\nx\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(bare))
    d = deploy.plan_serve(vllm_box, hops, dryrun=True)
    assert "boxy-ca-merged.pem" not in " ".join(d.command)


def test_pip_builds_derived_image(vllm_box, hops):
    # --pip layers the package onto the base image via a build prepare command; the
    # run command then references the derived tag.
    d = deploy.plan_serve(vllm_box, hops, dryrun=True, pip=["open_clip_torch"])
    assert d.prepare_commands, "expected a build prepare command"
    prep = " ".join(d.prepare_commands[0])
    assert "pip install --no-cache-dir" in prep and "open_clip_torch" in prep
    assert "image exists" in prep and "flock" in prep  # idempotent + race-safe
    assert "localhost/boxy-ext:" in " ".join(d.command)


def test_pip_tag_is_deterministic_and_order_independent(vllm_box, hops):
    a = deploy.plan_serve(vllm_box, hops, dryrun=True, pip=["a", "b"])
    b = deploy.plan_serve(vllm_box, hops, dryrun=True, pip=["b", "a"])
    ta = next(x for x in " ".join(a.command).split() if x.startswith("localhost/boxy-ext:"))
    tb = next(x for x in " ".join(b.command).split() if x.startswith("localhost/boxy-ext:"))
    assert ta == tb  # content hash over sorted packages


def test_pip_apptainer_warns_and_does_not_build(vllm_box, eldorado):
    d = deploy.plan_serve(vllm_box, eldorado, dryrun=True, pip=["open_clip_torch"])
    assert any("needs an OCI runtime" in w for w in d.warnings)
    assert "localhost/boxy-ext:" not in " ".join(d.command)


def test_cli_serve_pip_dryrun_and_forwarding(capsys):
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "hops.toml"),
               "--no-distributed", "--pip", "open_clip_torch", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Prepare:" in out
    assert "pip install --no-cache-dir" in out and "open_clip_torch" in out
    assert "localhost/boxy-ext:" in out


def test_cli_serve_trust_remote_code(capsys):
    # --trust-remote-code adds the vLLM flag; the scheduler path forwards it to the
    # compute-node inner serve (re-applied engine-aware there).
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "hops.toml"),
               "--no-distributed", "--trust-remote-code", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--trust-remote-code" in out
    assert "trust-remote-code: enabled" in out


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
