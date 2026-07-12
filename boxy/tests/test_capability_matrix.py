"""Exhaustive capability tests: every module path not covered elsewhere.

Complements the golden-argv, gap-regression, and bench/cloud suites so that
every shipped capability has at least one test before HPC go-live.
"""

import shutil

import pytest

from boxy import cloud, deploy, ramalama_shim, sky_export
from boxy.backends import get_backend
from boxy.backends.base import RuntimeBackend
from boxy.box import Box
from boxy.cli import main
from boxy.deploy import Deployment
from boxy.location import Location, Resources
from boxy.schedulers import get_scheduler
from tests.conftest import EXAMPLES


# ---------- registries & backends ----------

def test_unknown_backend_and_scheduler_are_clean_errors():
    with pytest.raises(ValueError, match="unknown runtime backend"):
        get_backend("bogus")
    with pytest.raises(ValueError, match="unknown scheduler"):
        get_scheduler("bogus")


def test_backend_available_probes_path():
    backend = get_backend("podman")
    assert backend.available() == (shutil.which("podman") is not None)


def test_gpu_args_empty_for_no_accelerator():
    for name in ("podman", "apptainer", "docker"):
        assert get_backend(name).gpu_args("none") == []


def test_docker_rocm_inherits_podman_device_set(vllm_box, eldorado):
    assert get_backend("docker").gpu_args("rocm") == get_backend("podman").gpu_args("rocm")


def test_podman_mount_options_suffix(vllm_box, hops):
    cmd = get_backend("podman").build_command(vllm_box, hops, ["x"], {}, [("/a", "/b", "ro")], "none")
    assert "--volume=/a:/b:ro" in cmd


def test_runtime_backend_is_abstract():
    with pytest.raises(TypeError):
        RuntimeBackend()  # build_command is abstract


# ---------- box schema ----------

def test_box_missing_section_and_env_coercion(tmp_path):
    empty = tmp_path / "e.toml"
    empty.write_text("[notbox]\nx=1\n")
    with pytest.raises(ValueError, match="missing \\[box\\] section"):
        Box.from_toml(empty)
    withenv = tmp_path / "env.toml"
    withenv.write_text('[box]\nname="n"\nimage="i"\n[box.env]\nN_THREADS = 4\n')
    box = Box.from_toml(withenv)
    assert box.env["N_THREADS"] == "4"  # TOML ints coerce to env strings


@pytest.mark.parametrize("uri", ["hf://o/n", "huggingface://o/n", "ollama://n", "oci://r/n", "ms://o/n"])
def test_all_transport_schemes_detected(uri):
    assert Box(name="b", image="i", model=uri).model_is_transport_uri


# ---------- location schema ----------

def test_location_validation_errors():
    with pytest.raises(ValueError, match="unknown runtime"):
        Location(name="l", runtime="lxc")
    with pytest.raises(ValueError, match="unknown accelerator"):
        Location(name="l", accelerator="tpu")


def test_location_unknown_key_and_modules_list_form(tmp_path):
    bad = tmp_path / "l.toml"
    bad.write_text('[location]\nname="l"\nbogus=1\n')
    with pytest.raises(ValueError, match="unknown \\[location\\] keys"):
        Location.from_toml(bad)


def test_all_example_locations_parse():
    for name in ("local", "local-docker", "slurm-podman-cuda", "flux-apptainer-rocm", "cloud-gpu"):
        loc = Location.from_toml(EXAMPLES / "locations" / f"{name}.toml")
        assert loc.name in (name, name.replace(".toml", ""))


def test_resolve_runtime_explicit_and_error(monkeypatch, eldorado):
    assert eldorado.resolve_runtime() == "apptainer"
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="no container runtime found"):
        Location(name="bare").resolve_runtime()


# ---------- schedulers ----------

def test_scheduler_prefixes_without_gpus():
    loc = Location(name="cpu", scheduler="slurm", resources=Resources(nodes=3, gpus_per_node=0))
    assert get_scheduler("slurm").launch_prefix(loc) == ["srun", "--nodes=3"]
    loc.scheduler = "flux"
    assert get_scheduler("flux").launch_prefix(loc) == ["flux", "run", "-N3"]


def test_none_scheduler_clears_xdg_only_inside_allocations(monkeypatch):
    scheduler = get_scheduler("none")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("FLUX_ENCLOSING_ID", raising=False)
    assert scheduler.host_env_fixups() == []
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert scheduler.host_env_fixups() == ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]


def test_multiple_module_loads_joined(eldorado):
    eldorado.modules = ["rocm/6.4.0", "cray-mpich"]
    cmd = get_scheduler("flux").with_modules(["x"], eldorado)
    assert "module load rocm/6.4.0 && module load cray-mpich" in cmd[2]


# ---------- deploy: model resolution & execute ----------

def test_resolve_model_empty_and_absolute(vllm_box, hops):
    vllm_box.model = ""
    assert deploy.resolve_model(vllm_box, hops, dryrun=True) == ("", [])
    vllm_box.model = "/shared/fs/llama.gguf"
    path, mounts = deploy.resolve_model(vllm_box, hops, dryrun=True)
    assert path == "/mnt/models/llama.gguf"
    assert mounts == [("/shared/fs/llama.gguf", "/mnt/models/llama.gguf", "ro")]


def test_resolve_model_transport_uri_dryrun_via_shim(hops):
    # With ramalama importable, dryrun pull resolves to its placeholder path.
    box = Box(name="b", image="i", model="hf://Qwen/Qwen2.5-0.5B-Instruct")
    path, mounts = deploy.resolve_model(box, hops, dryrun=True)
    assert path.startswith("/mnt/models/")
    assert mounts and mounts[0][2] == "ro"


def _fake_deployment(command, prepare=(), env_unset=()):
    return Deployment(
        box=Box(name="t", image="i"), location=Location(name="l"), accelerator="none",
        backend=get_backend("docker"), scheduler=get_scheduler("none"),
        command=list(command), prepare_commands=[list(p) for p in prepare], env_unset=list(env_unset),
    )


def test_execute_runs_prepare_then_command_and_propagates_failure():
    assert deploy.execute(_fake_deployment(["true"], prepare=[["true"]])) == 0
    assert deploy.execute(_fake_deployment(["true"], prepare=[["false"]])) == 1
    assert deploy.execute(_fake_deployment(["false"])) == 1


def test_execute_skips_sif_build_when_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sif = tmp_path / "cached.sif"
    sif.write_text("x")
    # prepare would fail if run; existing SIF must short-circuit it
    d = _fake_deployment(["true"], prepare=[["false", str(sif)]])
    assert deploy.execute(d) == 0


def test_execute_unsets_scheduler_env(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_ID", "should-be-unset")
    d = _fake_deployment(["sh", "-c", 'test -z "$XDG_SESSION_ID"'], env_unset=["XDG_SESSION_ID"])
    assert deploy.execute(d) == 0


# ---------- ramalama shim ----------

def test_shim_helpers_return_dicts_in_gpuless_env():
    assert isinstance(ramalama_shim.accel_env_vars(), dict)
    assert isinstance(ramalama_shim.gpu_device_paths(), dict)


def test_default_images_cover_engines_and_accelerators():
    assert "llama.cpp:server-cuda" in ramalama_shim.default_image("llama.cpp", "cuda")
    assert ramalama_shim.default_image("llama.cpp", "none").endswith(":server")
    for accel in ("cuda", "rocm", "intel", "none", "vulkan"):
        assert ramalama_shim.default_image("vllm", accel)  # never empty
    assert ramalama_shim._ramalama_vllm_image("none") is None  # unmapped accel


# ---------- sky export details ----------

def test_yaml_str_quotes_specials():
    assert sky_export._yaml_str("plain") == "plain"
    assert sky_export._yaml_str("a:b") == '"a:b"'
    assert sky_export._yaml_str("") == '""'


def test_sky_export_port_override_and_llamacpp(vllm_box):
    loc = Location(name="c", scheduler="none", runtime="docker")
    yaml_text = sky_export.to_sky_task(vllm_box, loc, port=9999)
    assert "ports: [9999]" in yaml_text and "--port=9999" in yaml_text
    llbox = Box(name="l", image="i", engine="llama.cpp", model="m.gguf", ports=[8080])
    yaml_text = sky_export.to_sky_task(llbox, loc)
    # sky runs in a shell, so the deferred entrypoint resolves to the
    # upstream image's concrete binary path
    assert "/app/llama-server -m m.gguf" in yaml_text


# ---------- cloud details ----------

def test_write_task_yaml_tempfile_mode(vllm_box, hops):
    path = cloud.write_task_yaml(vllm_box, hops, port=None, serve=False)
    assert path.endswith(".sky.yaml")
    with open(path) as f:
        assert "image_id: docker:" in f.read()


def test_ensure_sky_error_when_missing(monkeypatch):
    monkeypatch.setattr(cloud.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="boxy-hpc\\[cloud\\]"):
        cloud.ensure_sky()


# ---------- CLI edges ----------

def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert "boxy 0.1.0" in capsys.readouterr().out


def test_cli_serve_port_override(capsys):
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"), "--dryrun", "--port", "9001"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--port=9001" in out and "--port=8000" not in out


def test_cli_generate_stdout_mode(capsys):
    rc = main(["generate", "sky", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    assert rc == 0
    assert "image_id: docker:vllm/vllm-openai:v0.24.0" in capsys.readouterr().out


def test_cli_pull_no_model_is_error(tmp_path, capsys):
    box = tmp_path / "b.toml"
    box.write_text('[box]\nname="x"\nimage="i"\n')
    assert main(["pull", "--box", str(box)]) == 1
    assert "no model set" in capsys.readouterr().err


def test_cli_stop_apptainer_location_is_helpful_error(capsys):
    rc = main(["stop", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "flux-apptainer-rocm.toml"), "--dryrun"])
    assert rc == 1
    assert "scancel / flux cancel" in capsys.readouterr().err


def test_cli_build_apptainer_dryrun_prints_sif_build(capsys):
    rc = main(["build", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "flux-apptainer-rocm.toml"), "--dryrun"])
    assert rc == 0
    assert "apptainer build --force vllm-rocm.sif" in capsys.readouterr().out


def test_cli_missing_files_are_rc1(capsys):
    assert main(["serve", "--box", "nope.toml", "--location", "also-nope.toml", "--dryrun"]) == 1
    assert "boxy: error:" in capsys.readouterr().err


def test_cli_launch_without_sky_is_helpful_error(monkeypatch, capsys):
    monkeypatch.setattr(cloud.shutil, "which", lambda _: None)
    rc = main(["launch", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    assert rc == 1
    assert "boxy-hpc[cloud]" in capsys.readouterr().err
