"""Tests for the llama.cpp engine builder and the boxy -> SkyPilot transpiler."""

import pytest

from boxy import engines, sky_export
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location, Resources
from tests.conftest import EXAMPLES


def test_generate_sky_smallest_llama_is_valid_yaml(capsys):
    """Cloud row (verified by construction; a live cloud run needs credentials):
    `boxy generate sky` for the smallest-Llama GGUF box must emit a SkyPilot task
    that PARSES as YAML and carries the llama.cpp image + the in-task GGUF fetch."""
    yaml = pytest.importorskip("yaml")
    rc = main(["generate", "sky", "--box", str(EXAMPLES / "boxes" / "llama-3.2-1b.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    out = capsys.readouterr().out
    assert rc == 0
    task = yaml.safe_load(out)                                   # must be valid YAML
    assert task["name"] == "llama-3.2-1b"
    assert task["resources"]["image_id"] == "docker:ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert task["resources"]["accelerators"] == "H100:4"
    assert task["resources"]["ports"] == [8090]
    # the GGUF is fetched by the engine on the cloud VM (--hf-repo/--hf-file)
    assert "--hf-repo hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF" in task["run"]
    assert "--hf-file llama-3.2-1b-instruct-q4_k_m.gguf" in task["run"]


@pytest.fixture
def llamacpp_box() -> Box:
    return Box(
        name="llamacpp",
        image="boxy-demo/llamacpp:local",
        engine="llama.cpp",
        model="tiny.gguf",
        workdir="/models",
        ports=[8090],
    )


@pytest.fixture
def cloud_gpu() -> Location:
    return Location(
        name="cloud-gpu",
        scheduler="none",
        accelerator="cuda",
        runtime="docker",
        resources=Resources(nodes=1, gpus_per_node=4, accelerator_type="H100"),
    )


def test_llamacpp_serve_cmd_space_style(llamacpp_box, cloud_gpu):
    cmd = engines.build_serve_cmd(llamacpp_box, cloud_gpu, "tiny.gguf")
    # No explicit entrypoint => defer to the image ENTRYPOINT ("" sentinel):
    # the upstream llama.cpp image keeps its binary off $PATH (field finding).
    assert cmd[:3] == ["", "-m", "tiny.gguf"]
    i = cmd.index("--port")
    assert cmd[i + 1] == "8090"
    assert not any(a.startswith("--port=") for a in cmd)


def test_llamacpp_user_args_win(llamacpp_box, cloud_gpu):
    cmd = engines.build_serve_cmd(llamacpp_box, cloud_gpu, "tiny.gguf", extra_args=["--port", "9999"])
    assert cmd.count("--port") == 1
    assert cmd[cmd.index("--port") + 1] == "9999"


def test_engine_dispatch_vllm_default(vllm_box, cloud_gpu):
    cmd = engines.build_serve_cmd(vllm_box, cloud_gpu, "model-x")
    assert cmd[:3] == ["vllm", "serve", "model-x"]


def test_box_rejects_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        Box(name="x", image="y", engine="tgi")


def test_sky_export_task(vllm_box, cloud_gpu):
    yaml_text = sky_export.to_sky_task(vllm_box, cloud_gpu)
    assert "image_id: docker:vllm/vllm-openai:v0.9.1" in yaml_text
    assert "accelerators: H100:4" in yaml_text
    assert "ports: [8000]" in yaml_text
    assert "vllm serve" in yaml_text
    assert "service:" not in yaml_text  # no service block without --serve


def test_sky_export_serve_block(vllm_box, cloud_gpu):
    yaml_text = sky_export.to_sky_task(vllm_box, cloud_gpu, serve=True)
    assert "service:" in yaml_text
    assert "path: /v1/models" in yaml_text
    assert "replicas: 1" in yaml_text


def test_sky_export_no_accel_type_omits_accelerators(vllm_box):
    loc = Location(name="cpu", scheduler="none", runtime="docker")
    yaml_text = sky_export.to_sky_task(vllm_box, loc)
    assert "accelerators:" not in yaml_text


# ---- corporate proxy + CA carriage on the cloud path (serving-matrix audit) ----


def test_sky_proxy_and_ca_carriage(vllm_box, cloud_gpu, monkeypatch, tmp_path):
    """--proxy = "this task runs on-net": the sky YAML must carry the proxy env
    (both cases, no_proxy preserved) AND ship the merged trust bundle via
    file_mounts with SSL_CERT_FILE/REQUESTS_CA_BUNDLE pointing at it. Validated
    once against the real skypilot parser (10/10 boxes, sky 0.12.3)."""
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,.sandia.gov")
    ca = tmp_path / "merged-ca.crt"
    ca.write_text("x")
    text = sky_export.to_sky_task(vllm_box, cloud_gpu,
                                  proxy="http://proxy.sandia.gov:80", ca_bundle=str(ca))
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        assert f'{var}: "http://proxy.sandia.gov:80"' in text   # URL has ":" -> quoted
    assert "NO_PROXY: " in text and "no_proxy: " in text          # intra-VM stays direct
    assert f"SSL_CERT_FILE: {sky_export.CA_MOUNT}" in text
    assert f"REQUESTS_CA_BUNDLE: {sky_export.CA_MOUNT}" in text
    assert "file_mounts:" in text and f"{sky_export.CA_MOUNT}: {ca}" in text


def test_sky_no_network_env_by_default(vllm_box, cloud_gpu):
    # off-net cloud VM: injecting a corporate proxy blindly would break egress
    text = sky_export.to_sky_task(vllm_box, cloud_gpu)
    assert "PROXY" not in text and "file_mounts:" not in text and "SSL_CERT_FILE" not in text


def test_sky_empty_image_is_a_loud_error(cloud_gpu):
    # finding 41 re-caught by the first real sky.Task.from_yaml validation:
    # an unresolved image emitted `image_id: docker:` — invalid YAML sky rejects.
    box = Box(name="noimg", engine="vllm", model="m", ports=[8000])
    with pytest.raises(ValueError, match="has no image"):
        sky_export.to_sky_task(box, cloud_gpu)


def test_cli_generate_sky_proxy_carries_network_env(capsys, monkeypatch, tmp_path):
    from boxy import cli

    ca = tmp_path / "ca.crt"
    ca.write_text("x")
    monkeypatch.setattr(cli.ramalama_shim, "ensure_trust_bundle", lambda: str(ca))
    rc = main(["generate", "sky", "--box", str(EXAMPLES / "boxes" / "llama-3.2-1b.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml"),
               "--proxy", "http://proxy.sandia.gov:80"])
    out = capsys.readouterr().out
    assert rc == 0
    assert 'HTTPS_PROXY: "http://proxy.sandia.gov:80"' in out
    assert f"{sky_export.CA_MOUNT}: {ca}" in out


def test_cli_launch_dryrun_proxy_writes_network_env(capsys, monkeypatch, tmp_path):
    from boxy import cli

    ca = tmp_path / "ca.crt"
    ca.write_text("x")
    monkeypatch.setattr(cli.ramalama_shim, "ensure_trust_bundle", lambda: str(ca))
    out_yaml = tmp_path / "task.sky.yaml"
    rc = main(["launch", "--box", str(EXAMPLES / "boxes" / "llama-3.2-1b.toml"),
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml"),
               "--proxy", "http://proxy.sandia.gov:80", "-o", str(out_yaml), "--dryrun"])
    assert rc == 0
    text = out_yaml.read_text()
    assert 'https_proxy: "http://proxy.sandia.gov:80"' in text
    assert f"SSL_CERT_FILE: {sky_export.CA_MOUNT}" in text
    assert "sky launch -c llama-3.2-1b" in capsys.readouterr().out


def test_sky_yaml_accepted_by_real_skypilot(vllm_box, cloud_gpu, tmp_path):
    """When skypilot is installed (sandbox validation; skipped on CI), the
    generated YAML must parse through sky's OWN Task.from_yaml — this is what
    caught the empty-image bug that pyyaml-level tests missed."""
    sky = pytest.importorskip("sky")
    p = tmp_path / "t.yaml"
    ca = tmp_path / "ca.crt"
    ca.write_text("x")
    p.write_text(sky_export.to_sky_task(vllm_box, cloud_gpu,
                                        proxy="http://p:80", ca_bundle=str(ca)))
    task = sky.Task.from_yaml(str(p))
    assert task.envs["HTTPS_PROXY"] == "http://p:80"
    assert task.file_mounts[sky_export.CA_MOUNT] == str(ca)
