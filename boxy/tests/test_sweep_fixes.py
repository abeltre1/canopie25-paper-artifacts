"""Regression tests for the round-1 corner-case sweep (61 findings, 7 auditor
agents + adversarial verification). One test per fixed behavior, numbered by
finding."""

import shutil as _shutil

import pytest

from boxy import deploy, engines, ramalama_shim, readiness, resolve
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location, Resources
from tests.conftest import EXAMPLES


@pytest.fixture(autouse=True)
def laptop_host(monkeypatch):
    real_which = _shutil.which
    monkeypatch.setattr("boxy.resolve.shutil.which",
                        lambda name: None if name in ("srun", "flux") else real_which(name))
    monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "none")


@pytest.fixture
def gguf(tmp_path):
    model = tmp_path / "m.q4.gguf"
    model.write_bytes(b"GGUF")
    return model


# ---- port coherence (findings 2/10/16/25/47/55) ----

def test_one_llamacpp_port_default_everywhere(gguf, capsys):
    """Portless llama.cpp box: banner, probe, and server must agree (8090)."""
    box = Box(name="np", image="i:1", engine="llama.cpp", model=str(gguf))
    loc = Location(name="l", runtime="docker", accelerator="none")
    d = deploy.plan_serve(box, loc, dryrun=True)
    assert "--port" in d.command and "8090" in d.command
    assert d.port == 8090


def test_engine_level_port_override_wins_everywhere(gguf):
    """--port after `--` is honored by the server; boxy must probe THAT port."""
    box = Box(name="np", image="i:1", engine="llama.cpp", model=str(gguf))
    loc = Location(name="l", runtime="docker", accelerator="none")
    d = deploy.plan_serve(box, loc, extra_args=["--port", "19123"], dryrun=True)
    assert d.port == 19123
    assert d.command.count("--port") == 1  # user's flag, not a duplicate


def test_serve_banner_uses_deployment_port(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1",
               "--foreground", "--here", "--dryrun", "--", "--port", "19124"])
    assert rc == 0
    # dryrun prints the command; the port in it is the user's
    assert "19124" in capsys.readouterr().out


def test_box_args_port_and_host_win_over_defaults():
    """finding 59: defaults were tacked BEFORE box.args, silently overriding."""
    box = Box(name="b", entrypoint="srv", engine="llama.cpp", model="",
              args={"port": 9090, "host": "127.0.0.1"})
    loc = Location(name="l", runtime="docker", accelerator="none")
    cmd = engines.build_llamacpp_serve_cmd(box, loc, "/m.gguf")
    assert cmd.count("--port") == 1 and "9090" in cmd
    assert cmd.count("--host") == 1 and "127.0.0.1" in cmd
    assert engines.serving_port(cmd, box) == 9090


# ---- per-engine tuning (findings 4/36) ----

def test_vllm_only_tuning_never_reaches_llamacpp():
    eldorado = Location.from_toml(EXAMPLES / "locations" / "eldorado.toml")
    box = Box(name="g", image="i:1", engine="llama.cpp", model="m.gguf")
    cmd = engines.build_llamacpp_serve_cmd(box, eldorado, "/m.gguf")
    assert "--gpu-memory-utilization" not in cmd  # llama-server exits 2 on it
    vbox = Box(name="v", image="i:1", engine="vllm", model="m")
    vcmd = engines.build_vllm_serve_cmd(vbox, eldorado, "m")
    assert "--gpu-memory-utilization=0.7" in vcmd  # vLLM still gets it


def test_nested_tuning_targets_each_engine():
    loc = Location(name="l", runtime="docker", accelerator="none",
                   tuning={"vllm": {"seed": 1}, "llama.cpp": {"ctx_size": 4096}})
    assert engines.tuning_for_engine(loc, "vllm") == {"seed": 1}
    assert engines.tuning_for_engine(loc, "llama.cpp") == {"ctx_size": 4096}


# ---- boxy run passthrough (findings 44/52/56) ----

def test_run_defers_to_image_entrypoint_and_uses_engine_style():
    box = Box(name="d", image="i:1", engine="llama.cpp", model="",
              args={"ctx_size": 4096})
    loc = Location(name="l", runtime="docker", accelerator="none")
    cmd = engines.build_raw_cmd(box, ["-m", "/m.gguf"], loc)
    assert cmd[0] == ""                       # deferral sentinel survives
    assert cmd[1:3] == ["-m", "/m.gguf"]      # first user arg is NOT the entrypoint
    assert "--ctx-size" in cmd and "4096" in cmd
    assert "--ctx-size=4096" not in cmd       # llama.cpp style is space-separated


def test_run_profile_dryrun_no_args_does_not_crash(capsys):
    rc = main(["run", "--box", str(EXAMPLES / "boxes" / "llamacpp-demo.toml"),
               "--location", str(EXAMPLES / "locations" / "local-docker.toml"), "--dryrun"])
    assert rc == 0  # was: IndexError traceback


# ---- model classification (findings 17/20/22/35) ----

def test_gguf_extension_not_substring():
    assert resolve.looks_like_gguf("/data/m.Q4.gguf")
    assert not resolve.looks_like_gguf("/data/models.gguf/llama-3-safetensors")
    assert not resolve.looks_like_gguf("model.gguf.bak")


def test_empty_scheme_rest_is_rejected():
    with pytest.raises(RuntimeError, match="malformed model URI"):
        resolve._classify_model("hf://", require_exists=False)


def test_unknown_scheme_is_rejected():
    with pytest.raises(RuntimeError, match="unsupported model scheme"):
        resolve._classify_model("s3://bucket/model.gguf", require_exists=False)


def test_file_scheme_becomes_local_path(gguf):
    resolved, note = resolve._classify_model(f"file://{gguf}", require_exists=True)
    assert resolved == str(gguf) and "local file" in note


def test_mixed_case_scheme_is_normalized():
    resolved, _ = resolve._classify_model("OLLAMA://granite3-moe", require_exists=True)
    assert resolved == "ollama://granite3-moe"
    engine, _ = resolve.infer_engine(resolved, "none")
    assert engine == "llama.cpp"


def test_dryrun_missing_path_not_misdiagnosed_as_safetensors(capsys):
    rc = main(["serve", "models/tiny-lama-demo.ggu", "--runtime", "docker",
               "--image", "i:1", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT PRESENT" in out and "did you mean" in out
    assert "assumed — model file not present" in out  # not "is a safetensors/HF repo"


def test_gpus_request_does_not_bypass_vllm_accel_gate():
    with pytest.raises(RuntimeError, match="GGUF"):
        resolve.infer_engine("hf://org/repo", "vulkan", gpus=4)  # finding 21


# ---- profile/flag layering (findings 11/12/13/39/40/46) ----

def test_flags_override_box_profile(capsys):
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "llamacpp-demo.toml"),
               "--location", str(EXAMPLES / "locations" / "local-docker.toml"),
               "--name", "my-name", "--image", "my/image:tag", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--name=my-name" in out and "my/image:tag" in out
    assert "flag overrides profile" in out


def test_flags_override_location_profile(capsys):
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "qwen-gguf.toml"),
               "--location", str(EXAMPLES / "locations" / "local-podman.toml"),
               "--runtime", "docker", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "docker run" in out and "podman run" not in out


def test_location_profile_scheduler_skips_login_guard(gguf, tmp_path, monkeypatch, capsys):
    """A profile pinning scheduler=slurm IS the sanctioned flow — it must not
    trip the login-node guard (finding 11/39)."""
    monkeypatch.setattr("boxy.resolve.shutil.which",
                        lambda name: f"/usr/bin/{name}")  # srun everywhere
    profile = tmp_path / "site.toml"
    profile.write_text('[location]\nname = "hops"\nscheduler = "slurm"\naccelerator = "cuda"\n'
                       'runtime = "podman"\n[location.resources]\nnodes = 1\ngpus_per_node = 2\n')
    rc = main(["serve", str(gguf), "--location", str(profile), "--foreground", "--dryrun"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "srun" in out.out  # attached wrap with the profile's scheduler


def test_model_plus_box_is_usage_error(gguf, capsys):
    rc = main(["serve", str(gguf), "--box", str(EXAMPLES / "boxes" / "qwen-gguf.toml"), "--dryrun"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_submission_flags_override_profile_geometry(gguf, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    profile = tmp_path / "site.toml"
    profile.write_text('[location]\nname = "e"\nscheduler = "flux"\naccelerator = "rocm"\n'
                       "[location.resources]\nnodes = 2\ngpus_per_node = 4\n")
    rc = main(["serve", str(gguf), "--location", str(profile), "--gpus", "8", "--nodes", "1", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "#FLUX: -N1" in out and "#FLUX: --gpus-per-node=8" in out  # finding 40
    assert "--location" in out  # profile forwarded to the compute node (finding 38)


# ---- lifecycle (findings 23/27) ----

def test_stop_refuses_foreign_containers(monkeypatch, capsys):
    import boxy.cli as cli

    monkeypatch.setattr(cli, "_container_exists", lambda r, n: True)
    monkeypatch.setattr(cli, "_container_label", lambda r, n: "")  # not ours
    rc = main(["stop", "someone-elses-db", "--runtime", "docker"])
    assert rc == 1
    assert "not created by boxy" in capsys.readouterr().err


def test_ready_timeout_zero_means_dont_wait(gguf, monkeypatch, capsys):
    import boxy.cli as cli
    import boxy.deploy as dep

    monkeypatch.setattr(cli, "_container_exists", lambda r, n: False)
    monkeypatch.setattr(dep, "execute", lambda d: 0)
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1",
               "--ready-timeout", "0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not waiting" in out and "endpoint once loaded" in out


# ---- readiness robustness (finding 26) ----

def test_readiness_survives_non_openai_responders(monkeypatch):
    import json as _json

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    responses = iter([_json.loads("[]"), {"data": ["notadict"]}, {"data": [{"id": "ok"}]}])

    def fake_load(resp):
        return next(responses)

    monkeypatch.setattr("boxy.readiness.urllib.request.urlopen", lambda *a, **kw: FakeResp())
    monkeypatch.setattr("boxy.readiness.json.load", fake_load)
    monkeypatch.setattr("boxy.readiness.time.sleep", lambda s: None)
    assert readiness.wait_ready("http://x", timeout_s=30) == "ok"  # was: AttributeError


# ---- TOML loader hardening (findings 1/6/7/8) ----

def test_string_ports_rejected_with_filename(tmp_path):
    bad = tmp_path / "b.toml"
    bad.write_text('[box]\nname="x"\nimage="i"\nengine="llama.cpp"\nports = "8090"\n')
    with pytest.raises(ValueError, match="list of integers"):
        Box.from_toml(bad)


def test_toml_syntax_error_names_the_file(tmp_path):
    bad = tmp_path / "broken.toml"
    bad.write_text('[box]\nname = "x\n')
    with pytest.raises(ValueError, match="broken.toml"):
        Box.from_toml(bad)


def test_volume_missing_target_is_clean_error(tmp_path):
    bad = tmp_path / "vol.toml"
    bad.write_text('[box]\nname="x"\nimage="i"\nengine="llama.cpp"\n[[box.volumes]]\nsource="./m"\n')
    with pytest.raises(ValueError, match="vol.toml"):
        Box.from_toml(bad)


def test_modules_string_treated_as_single_module(tmp_path):
    loc = tmp_path / "l.toml"
    loc.write_text('[location]\nname="m"\nruntime="docker"\nmodules = "rocm/6.4.0"\n')
    assert Location.from_toml(loc).modules == ["rocm/6.4.0"]  # not 10 one-char modules


# ---- apptainer SIF coherence (findings 3/37/54) ----

def test_sif_name_matches_resolved_accelerator(monkeypatch, gguf):
    monkeypatch.setattr("boxy.ramalama_shim.detect_accel", lambda: "rocm")
    box = Box(name="demo", image="i:1", engine="llama.cpp", model=str(gguf), ports=[8080])
    loc = Location(name="s", runtime="apptainer", accelerator="")  # autodetect
    d = deploy.plan_serve(box, loc, dryrun=True)
    sif_in_prepare = [a for a in d.prepare_commands[0] if a.endswith(".sif")][0]
    sif_in_command = [a for a in d.command if a.endswith(".sif")][0]
    assert sif_in_prepare == sif_in_command == "demo-rocm.sif"


# ---- image maps (findings 28/29/53) ----

def test_vllm_plugin_vocab_is_env_var_names(monkeypatch):
    seen = {}

    class FakePlugin:
        def get_container_image(self, config, gpu_type):
            seen["gpu_type"] = gpu_type
            return "docker.io/vllm/vllm-openai-rocm:x"

    import ramalama.plugins.loader as loader

    monkeypatch.setattr(loader, "get_runtime", lambda name: FakePlugin())
    image = ramalama_shim._ramalama_vllm_image("rocm")
    assert seen["gpu_type"] == "HIP_VISIBLE_DEVICES"  # not "HIP" (finding 28)
    assert image == "docker.io/vllm/vllm-openai-rocm:x"


def test_vllm_plugin_cuda_image_rejected_for_rocm(monkeypatch):
    class FakePlugin:
        def get_container_image(self, config, gpu_type):
            return "docker.io/vllm/vllm-openai:latest"  # CUDA-only fallthrough

    import ramalama.plugins.loader as loader

    monkeypatch.setattr(loader, "get_runtime", lambda name: FakePlugin())
    assert ramalama_shim._ramalama_vllm_image("rocm") is None
    assert "rocm" in ramalama_shim.default_image("vllm", "rocm")  # static map wins


def test_llamacpp_rocm_image_is_static_not_vulkan():
    assert ramalama_shim.default_image("llama.cpp", "rocm") == "quay.io/ramalama/rocm:latest"


def test_pinned_quay_image_gets_llama_server_entrypoint(gguf):
    box = Box(name="pin", image="quay.io/ramalama/rocm:latest", engine="llama.cpp", model=str(gguf))
    loc = Location(name="l", runtime="docker", accelerator="rocm")
    d = deploy.plan_serve(box, loc, dryrun=True)
    assert d.box.entrypoint == "llama-server"  # finding 53


# ---- shim robustness (findings 18/30/31/32) ----

def test_oci_pull_rejected_with_boxy_wording(monkeypatch):
    # default policy blocks oci:// before anything else
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    with pytest.raises(RuntimeError, match="registry allowlist"):
        ramalama_shim.pull_model("oci://quay.io/ramalama/smollm:135m", dryrun=True)
    # even when opted in, the store-only pull path can't drive an engine
    monkeypatch.setenv("BOXY_ALLOW_TRANSPORTS", "hf,ollama,oci")
    with pytest.raises(RuntimeError, match="container engine to pull"):
        ramalama_shim.pull_model("oci://quay.io/ramalama/smollm:135m", dryrun=True)


def test_trust_bundle_directory_and_unwritable_store(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path))  # a directory
    assert ramalama_shim.ensure_trust_bundle() is None
    assert "DIRECTORY" in capsys.readouterr().err
    site = tmp_path / "ca.crt"
    site.write_text("x")
    monkeypatch.setenv("SSL_CERT_FILE", str(site))
    monkeypatch.setattr(ramalama_shim, "DEFAULT_STORE", "/proc/definitely-not-writable")
    assert ramalama_shim.ensure_trust_bundle() is None
    assert "could not build" in capsys.readouterr().err


def test_bad_loglevel_env_does_not_crash(monkeypatch):
    import importlib
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import boxy.cli; print('ok')"],
        env={"BOXY_RAMALAMA_LOGLEVEL": "debug", "PYTHONPATH": "src:/home/user/ramalama", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True,
        cwd="/tmp/claude-0/-home-user/b9fd7f3c-6c0b-57ce-9dff-6366c0a08ece/scratchpad/work/boxy",
    )
    assert result.returncode == 0 and "ok" in result.stdout


# ---- sky export (findings 41/42/43) ----

def test_generate_sky_resolves_default_image(capsys):
    rc = main(["generate", "sky", "--box", str(EXAMPLES / "boxes" / "qwen-gguf.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "image_id: docker:\n" not in out          # finding 41
    assert "image_id: docker:" in out                # still docker-form
    assert "--hf-repo" in out                        # finding 43: llama.cpp HF download flags
    assert "-m Qwen/" not in out


def test_generate_sky_quay_image_uses_path_llama_server(tmp_path, capsys):
    box = tmp_path / "rocm.toml"
    box.write_text('[box]\nname = "r"\nengine = "llama.cpp"\nimage = "quay.io/ramalama/rocm:latest"\n'
                   'model = "/shared/m.gguf"\nports = [8090]\n')
    rc = main(["generate", "sky", "--box", str(box),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "llama-server -m /shared/m.gguf" in out
    assert "/app/llama-server" not in out            # finding 42


# ---- bench (findings 45/49/50) ----

def test_bench_p50_is_not_the_max():
    from boxy import bench

    latencies = [0.0138, 0.040]
    assert bench.percentile_ms(latencies, 0.50) == pytest.approx(13.8, rel=0.01)  # was 40.0
    assert bench.percentile_ms(latencies, 0.95) == pytest.approx(40.0, rel=0.01)


def test_bench_dict_dataset_clean_error(tmp_path):
    from boxy import bench

    data = tmp_path / "d.json"
    data.write_text('{"prompts": ["hi"]}')
    with pytest.raises(ValueError, match="no prompts found"):
        bench.load_prompts(str(data))


def test_bench_unreachable_endpoint_is_clean_error(capsys):
    rc = main(["bench", "--box", str(EXAMPLES / "boxes" / "qwen-gguf.toml"),
               "--url", "http://127.0.0.1:1", "--batch-sizes", "1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "cannot reach" in err and "Traceback" not in err


# ---- volumes (finding 60) ----

def test_relative_volume_sources_absolutized_and_colon_warned(tmp_path):
    from boxy.box import Volume

    box = Box(name="v", image="i:1", engine="llama.cpp", model="",
              volumes=[Volume(source="models", target="/m"),
                       Volume(source="/data/run:2026", target="/d")])
    loc = Location(name="l", runtime="docker", accelerator="none")
    d = deploy.plan_run(box, loc, ["--help"], dryrun=True)
    volume_args = [a for a in d.command if a.startswith("--volume=")]
    assert not any(a.startswith("--volume=models:") for a in volume_args)  # absolutized
    assert any("contains ':'" in w for w in d.warnings)


# ---- round 2: robustness auditor ----

def test_r2_tuning_array_is_clean_error(tmp_path):
    loc = tmp_path / "l.toml"
    loc.write_text('[location]\nname="l"\nruntime="docker"\ntuning = ["x=1"]\n')
    with pytest.raises(ValueError, match="l.toml.*tuning"):
        Location.from_toml(loc)


def test_r2_bool_and_out_of_range_ports_rejected():
    with pytest.raises(ValueError, match="1-65535"):
        Box(name="b", image="i", engine="llama.cpp", ports=[True])
    with pytest.raises(ValueError, match="1-65535"):
        Box(name="b", image="i", engine="llama.cpp", ports=[-1])


def test_r2_fifo_ssl_cert_file_does_not_hang(tmp_path, monkeypatch, capsys):
    import os as _os

    fifo = tmp_path / "ca.fifo"
    _os.mkfifo(fifo)
    monkeypatch.setenv("SSL_CERT_FILE", str(fifo))
    monkeypatch.delenv("BOXY_NO_CA_MERGE", raising=False)
    assert ramalama_shim.ensure_trust_bundle() is None  # was: blocks forever
    assert "not a regular file" in capsys.readouterr().err


def test_r2_malformed_job_record_skipped(tmp_path, monkeypatch, capsys):
    from boxy import jobs

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    (tmp_path / "y.json").write_text('{"name":"y"}')          # missing keys
    (tmp_path / "z.json").write_text('{"name":"z","scheduler":"slurm","job":"1"}')
    assert [r["name"] for r in jobs.list_records()] == ["z"]  # was: KeyError in boxy list


def test_r2_bench_batch_sizes_misuse_exits_2(capsys):
    rc = main(["bench", "--box", str(EXAMPLES / "boxes" / "qwen-gguf.toml"),
               "--batch-sizes", "abc", "--dryrun"])
    assert rc == 2
    assert "--batch-sizes" in capsys.readouterr().err


# ---- round 2: port/engine auditor ----

def test_r2_duplicate_port_last_wins_everywhere(gguf):
    """argparse engines honor the LAST --port; probing the first missed a
    live server (reproduced with a 30s timeout against a READY endpoint)."""
    box = Box(name="d", image="i:1", engine="llama.cpp", model=str(gguf))
    loc = Location(name="l", runtime="docker", accelerator="none")
    d = deploy.plan_serve(box, loc, extra_args=["--port", "19301", "--port", "19302"], dryrun=True)
    assert d.port == 19302
    assert engines.parse_port_flag(["--port", "1", "--port=2"]) == 2


def test_r2_explicit_port_flag_beats_box_args_port(gguf):
    box = Box(name="b", image="i:1", engine="llama.cpp", model=str(gguf),
              ports=[8091], args={"port": 9999})
    loc = Location(name="l", runtime="docker", accelerator="none")
    d = deploy.plan_serve(box, loc, port=8888, dryrun=True)
    assert d.port == 8888
    assert "9999" not in d.command


def test_r2_bench_default_honors_box_args_port(tmp_path, capsys):
    box = tmp_path / "b.toml"
    box.write_text('[box]\nname="b"\nimage="i:1"\nengine="llama.cpp"\nmodel="m.gguf"\n'
                   "ports = [8091]\n[box.args]\nport = 9999\n")
    rc = main(["bench", "--box", str(box), "--dryrun"])
    assert rc == 0
    assert "url=http://127.0.0.1:9999" in capsys.readouterr().out


def test_r2_mixed_flat_and_nested_tuning_keeps_vllm_keys():
    loc = Location(name="l", runtime="docker", accelerator="none",
                   tuning={"gpu_memory_utilization": 0.7, "llama.cpp": {"ctx_size": 2048}})
    assert engines.tuning_for_engine(loc, "vllm") == {"gpu_memory_utilization": 0.7}
    assert engines.tuning_for_engine(loc, "llama.cpp") == {"ctx_size": 2048}


def test_r2_extras_port_reflected_in_decisions(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1",
               "--here", "--dryrun", "--", "--port", "19305"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "port: 19305" in out            # decision line matches reality
    assert "llama.cpp default" not in out  # no phantom scanned default


# ---- round 2: layering auditor ----

def _slurm_profile(tmp_path, **extra):
    lines = ['[location]', 'name = "site"', 'scheduler = "slurm"', 'runtime = "podman"',
             'accelerator = "cuda"',
             'scheduler_args = ["--partition=site-default", "--license=tscratch:1"]',
             '[location.resources]', 'nodes = 1', 'gpus_per_node = 2']
    profile = tmp_path / "site.toml"
    profile.write_text("\n".join(lines) + "\n")
    return profile


def test_r2_attached_mode_consumes_scheduler_flags(gguf, tmp_path, monkeypatch, capsys):
    """r2-L1 (high): --partition/--account/--time/--scheduler-arg silently
    vanished in attached (srun) mode — a mis-billed job on a real site."""
    monkeypatch.setattr("boxy.resolve.shutil.which", lambda name: f"/usr/bin/{name}")
    profile = _slurm_profile(tmp_path)
    rc = main(["serve", str(gguf), "--location", str(profile), "--foreground",
               "--partition", "cli-part", "--time", "4:00:00",
               "--scheduler-arg=--qos=high", "--slurm-mem=64G", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--partition=cli-part" in out and "--time=4:00:00" in out
    assert "--qos=high" in out and "--mem=64G" in out
    assert "--license=tscratch:1" in out  # profile args survive too


def test_r2_scheduler_flags_warn_when_no_scheduler(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1",
               "--partition", "foo", "--slurm-mem=64G", "--dryrun"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "ignoring" in err and "--partition" in err and "--slurm-mem" in err


def test_r2_scheduler_none_profile_skips_guard_in_model_mode(gguf, monkeypatch, capsys):
    """r2-L2: a profile pinning scheduler='none' behaves like the flag."""
    monkeypatch.setattr("boxy.resolve.shutil.which", lambda name: f"/usr/bin/{name}")
    rc = main(["serve", str(gguf), "--location", str(EXAMPLES / "locations" / "local-docker.toml"),
               "--image", "i:1", "--dryrun"])
    out = capsys.readouterr()
    assert rc == 0, out.err  # was: login-node guard refusal
    assert "docker run" in out.out


def test_r2_save_profile_list_escaping(tmp_path, gguf, capsys):
    """r2-L3: an apostrophe in scheduler_args wrote unloadable TOML."""
    profile = tmp_path / "l.toml"
    profile.write_text('[location]\nname = "l"\nscheduler = "slurm"\nruntime = "podman"\n'
                       'accelerator = "cuda"\nscheduler_args = ["--comment=bob\'s run"]\n')
    prefix = tmp_path / "snap"
    rc = main(["serve", str(gguf), "--location", str(profile), "--foreground",
               "--dryrun", "--save-profile", str(prefix)])
    assert rc == 0
    loc = Location.from_toml(f"{prefix}.location.toml")   # was: Unclosed array
    assert loc.scheduler_args[0] == "--comment=bob's run"


def test_r2_save_profile_captures_box_mode_port(tmp_path, capsys):
    prefix = tmp_path / "snap"
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "llamacpp-demo.toml"),
               "--location", str(EXAMPLES / "locations" / "local-docker.toml"),
               "--port", "9000", "--dryrun", "--save-profile", str(prefix)])
    assert rc == 0
    assert Box.from_toml(f"{prefix}.box.toml").ports == [9000]  # was: [8090]


def test_r2_save_profile_quotes_dotted_env_keys(tmp_path, gguf, capsys):
    box = tmp_path / "b.toml"
    box.write_text('[box]\nname = "b"\nimage = "i:1"\nengine = "llama.cpp"\nmodel = "m.gguf"\n'
                   '[box.env]\n"MY.DOTTED" = "v1"\n')
    prefix = tmp_path / "snap"
    rc = main(["serve", "--box", str(box),
               "--location", str(EXAMPLES / "locations" / "local-docker.toml"),
               "--dryrun", "--save-profile", str(prefix)])
    assert rc == 0
    reloaded = Box.from_toml(f"{prefix}.box.toml")
    assert reloaded.env == {"MY.DOTTED": "v1"}   # was: {"MY": "{'DOTTED': 'v1'}"}


def test_r2_profile_gpus_feed_engine_inference(tmp_path, monkeypatch, capsys):
    """r2-L6: slurm profile with gpus_per_node=2 must not yield 'no GPU'."""
    monkeypatch.setattr("boxy.resolve.shutil.which", lambda name: f"/usr/bin/{name}")
    profile = tmp_path / "l.toml"
    profile.write_text('[location]\nname = "l"\nscheduler = "slurm"\nruntime = "podman"\n'
                       "[location.resources]\ngpus_per_node = 2\n")
    safetensors = tmp_path / "model-dir"
    safetensors.mkdir()
    rc = main(["serve", str(safetensors), "--location", str(profile), "--foreground",
               "--accelerator", "cuda", "--dryrun"])
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "engine: vllm" in out.out


def test_r2_extras_port_beats_port_flag_with_warning(gguf, capsys):
    rc = main(["serve", str(gguf), "--runtime", "docker", "--image", "i:1", "--here",
               "--port", "9000", "--dryrun", "--", "--port", "9100"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "port: 9100" in captured.out          # decision matches the argv winner
    assert "engine flag wins" in captured.err
    assert "--port 9000" not in captured.out


def test_r2_decision_lines_name_profile_provenance(gguf, capsys):
    rc = main(["serve", str(gguf), "--location", str(EXAMPLES / "locations" / "hops.toml"),
               "--foreground", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "scheduler: slurm (location profile)" in out   # was: "(--scheduler)"
    assert "accelerator: cuda (location profile)" in out


# ---- registry origin policy + auth visibility (user requirement, 2026-07) ----

def test_modelscope_blocked_by_default(monkeypatch, capsys):
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    rc = main(["serve", "ms://qwen/Qwen2-7B-Instruct", "--here", "--dryrun"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "modelscope.cn" in err and "China" in err
    assert "BOXY_ALLOW_TRANSPORTS" in err          # the deliberate override is named


def test_modelscope_blocked_in_pull_and_profile_path(monkeypatch, tmp_path, capsys):
    """The policy must hold on EVERY pull path, not just model-mode serve."""
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    with pytest.raises(RuntimeError, match="modelscope.cn"):
        ramalama_shim.pull_model("ms://qwen/x.gguf", dryrun=True)
    box = tmp_path / "b.toml"
    box.write_text('[box]\nname="b"\nimage="i:1"\nengine="llama.cpp"\nmodel="modelscope://qwen/x.gguf"\n')
    rc = main(["pull", "--box", str(box)])
    assert rc == 1
    assert "modelscope.cn" in capsys.readouterr().err


def test_transport_optin_is_explicit_env(monkeypatch):
    from boxy import policy

    monkeypatch.setenv("BOXY_ALLOW_TRANSPORTS", "hf,ollama,ms")
    policy.check_transport("ms://qwen/x.gguf")     # no raise
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS")
    assert policy.allowed_transports() == ("hf", "ollama")


def test_info_shows_policy_and_auth_status_never_values(monkeypatch, capsys):
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_SUPERSECRETVALUE")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIASECRETID")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "S3CR3T")
    rc = main(["info"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "registries: allowed [hf, ollama]" in out and "ms" in out.split("blocked")[1]
    assert "HuggingFace token: present" in out
    assert "S3 credentials: present" in out
    assert "SUPERSECRETVALUE" not in out and "AKIASECRETID" not in out and "S3CR3T" not in out


def test_info_net_never_probes_blocked_registries(monkeypatch):
    from boxy import policy

    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    urls = [url for _scheme, url in policy.registry_probes()]
    assert not any("modelscope" in u for u in urls)
    assert any("huggingface.co" in u for u in urls) and any("ollama" in u for u in urls)


# ---- HF token precedence + validation (field report: 'HF_TOKEN did not take effect') ----

def test_effective_hf_token_env_wins_outright(monkeypatch, tmp_path):
    """Mirrors RamaLama's verified precedence: HF_TOKEN set => cache IGNORED."""
    cache = tmp_path / "token"
    cache.write_text("hf_CACHED\n")
    monkeypatch.setattr(ramalama_shim.os.path, "expanduser",
                        lambda p: str(cache) if "huggingface" in p else p)
    monkeypatch.setenv("HF_TOKEN", "hf_FROMENV")
    token, source = ramalama_shim.effective_hf_token()
    assert token == "hf_FROMENV" and "HF_TOKEN env" in source
    monkeypatch.delenv("HF_TOKEN")
    token, source = ramalama_shim.effective_hf_token()
    assert token == "hf_CACHED" and "huggingface-cli" in source
    monkeypatch.setenv("HF_TOKEN", "")           # empty forces anonymous
    token, source = ramalama_shim.effective_hf_token()
    assert token is None and "EMPTY" in source


def test_info_names_the_winning_token_source(monkeypatch, tmp_path, capsys):
    cache = tmp_path / "token"
    cache.write_text("hf_CACHED\n")
    monkeypatch.setattr(ramalama_shim.os.path, "expanduser",
                        lambda p: str(cache) if "huggingface" in p else p)
    monkeypatch.setenv("HF_TOKEN", "hf_FROMENV")
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "using HF_TOKEN env var" in out
    assert "IGNORED while HF_TOKEN is set" in out    # both present -> precedence stated
    assert "hf_FROMENV" not in out and "hf_CACHED" not in out


def test_info_net_validates_effective_token(monkeypatch, capsys):
    import urllib.error
    import urllib.request

    monkeypatch.setenv("HF_TOKEN", "hf_SOMETOKEN")
    monkeypatch.setattr(ramalama_shim, "ensure_trust_bundle", lambda: None)
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    seen = {}

    def fake_urlopen(req, timeout=0):
        url = req if isinstance(req, str) else req.full_url
        if "whoami" in url:
            seen["auth"] = req.get_header("Authorization")
            raise urllib.error.HTTPError(url, 401, "Unauthorized", None, None)

        class R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    rc = main(["info", "--net"])
    out = capsys.readouterr().out
    assert seen["auth"] == "Bearer hf_SOMETOKEN"      # the EFFECTIVE token was tested
    assert "INVALID (HTTP 401)" in out and "re-export HF_TOKEN" in out
    assert "hf_SOMETOKEN" not in out                  # never printed
    assert rc == 1                                    # invalid token counts as a failure
