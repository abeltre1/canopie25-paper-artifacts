"""Regression tests for findings from the first real-user run-through
(macOS, Apple Silicon, TLS-intercepting network, podman applehv VM)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from boxy import deploy, ramalama_shim
from boxy.box import Box, Volume
from boxy.cli import main
from tests.conftest import EXAMPLES

ROOT = Path(__file__).parent.parent


def test_finding1_shim_suppresses_podman_gpu_prompt():
    # Fresh subprocess without the var: importing the shim must set it.
    env = {k: v for k, v in os.environ.items()
           if k not in ("RAMALAMA_USER__NO_MISSING_GPU_PROMPT", "PYTHONPATH")}
    env["PYTHONPATH"] = str(ROOT / "src")
    p = subprocess.run(
        [sys.executable, "-c",
         "import boxy.ramalama_shim, os; print(os.environ['RAMALAMA_USER__NO_MISSING_GPU_PROMPT'])"],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert p.returncode == 0 and p.stdout.strip() == "true"


def test_finding2_ssl_failure_message_has_remedy():
    try:
        raise RuntimeError("URL pull failed") from OSError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate"
        )
    except RuntimeError as e:
        msg = ramalama_shim._pull_failure_message("hf://o/n", e)
    assert "SSL_CERT_FILE" in msg and "root cause" in msg and "CERTIFICATE_VERIFY_FAILED" in msg


def test_finding3_cli_fallback_message_names_real_cause():
    try:
        try:
            raise OSError("connection reset by peer")
        except OSError:
            raise NotImplementedError("huggingface cli download not available")
    except NotImplementedError as e:
        msg = ramalama_shim._pull_failure_message("hf://o/n", e)
    assert "unimplemented" in msg               # explains the dead-end fallback
    assert "connection reset by peer" in msg    # surfaces the real error


def test_finding4_workdir_without_volume_warns(hops, tmp_path):
    box = Box(name="w", image="i", model="m", workdir="/vllm-workspace/models")
    d = deploy.plan_serve(box, hops, dryrun=True)
    assert any("workdir" in w and "Podman will refuse" in w for w in d.warnings)
    # and the warning reaches the user on stderr through the CLI
    ok = Box(name="w2", image="i", model="m", workdir="/models",
             volumes=[Volume(source=str(tmp_path), target="/models")])
    d2 = deploy.plan_serve(ok, hops, dryrun=True)
    assert d2.warnings == []


def test_finding4b_cli_prints_workdir_warning(tmp_path, capsys):
    boxfile = tmp_path / "b.toml"
    boxfile.write_text('[box]\nname="w"\nimage="i"\nmodel="m"\nworkdir="/nope"\n')
    rc = main(["serve", "--box", str(boxfile),
               "--location", str(EXAMPLES / "locations" / "hops.toml"), "--dryrun"])
    assert rc == 0
    assert "Podman will refuse" in capsys.readouterr().err


def test_finding5_example_boxes_workdir_rule_holds():
    # No shipped example may set a workdir that no volume provides.
    for path in (EXAMPLES / "boxes").glob("*.toml"):
        box = Box.from_toml(path)
        if box.workdir:
            targets = {v.target for v in box.volumes}
            assert box.workdir in targets, f"{path.name}: workdir {box.workdir} has no volume"


def test_finding7_llamacpp_defers_to_image_entrypoint(hops):
    from boxy.backends import get_backend

    box = Box(name="q", engine="llama.cpp", model="m.gguf", ports=[8090])
    d = deploy.plan_serve(box, hops, dryrun=True)  # hops runtime=podman
    joined = " ".join(d.command)
    assert "--entrypoint" not in joined            # image ENTRYPOINT wins
    image_idx = d.command.index(d.box.image)
    assert d.command[image_idx + 1] == "-m"        # args follow image directly
    # explicit entrypoint still honored
    box2 = Box(name="q2", engine="llama.cpp", entrypoint="/app/llama-server", model="m.gguf")
    d2 = deploy.plan_serve(box2, hops, dryrun=True)
    assert "--entrypoint=/app/llama-server" in d2.command
    # apptainer: deferred entrypoint switches exec -> run (SIF runscript)
    cmd = get_backend("apptainer").build_command(box, hops, ["", "-m", "m.gguf"], {}, [], "cuda")
    assert cmd[:2] == ["apptainer", "run"]
    assert "" not in cmd


def test_finding10_missing_volume_source_warns(hops, tmp_path):
    box = Box(name="v", image="i", model="m",
              volumes=[Volume(source="/definitely/not/there", target="/models")])
    d = deploy.plan_serve(box, hops, dryrun=True)
    assert any("does not exist on this host" in w for w in d.warnings)
    # existing source: no such warning
    box2 = Box(name="v2", image="i", model="m",
               volumes=[Volume(source=str(tmp_path), target="/models")])
    d2 = deploy.plan_serve(box2, hops, dryrun=True)
    assert not any("does not exist on this host" in w for w in d2.warnings)


def test_finding11_macos_publishes_ports_instead_of_host_network(monkeypatch, hops):
    from boxy.backends import get_backend

    box = Box(name="q", image="i", engine="llama.cpp", model="m.gguf", ports=[8090])
    inner = ["", "-m", "m.gguf", "--host", "0.0.0.0", "--port", "8001"]

    monkeypatch.setattr(sys, "platform", "darwin")
    cmd = get_backend("podman").build_command(box, hops, inner, {}, [], "none")
    assert "--network=host" not in cmd
    assert "-p" in cmd
    published = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-p"]
    assert "8090:8090" in published and "8001:8001" in published  # box port + CLI override

    monkeypatch.setattr(sys, "platform", "linux")
    cmd = get_backend("podman").build_command(box, hops, inner, {}, [], "none")
    assert "--network=host" in cmd and "-p" not in cmd  # HPC parity preserved


def test_finding8_prompts_hard_silenced_at_seam():
    ramalama_shim.detect_accel()
    import ramalama.common as rc

    assert rc.confirm_no_gpu("any-machine", "applehv") is True  # patched, no input()


def test_finding6_latest_vllm_and_mac_example():
    hf_box = Box.from_toml(EXAMPLES / "boxes" / "vllm-hf.toml")
    assert hf_box.image == "vllm/vllm-openai:v0.24.0"   # registry-verified latest
    assert not hf_box.workdir
    vllm_box = Box.from_toml(EXAMPLES / "boxes" / "vllm.toml")
    assert vllm_box.image == "vllm/vllm-openai:v0.24.0"
    gguf = Box.from_toml(EXAMPLES / "boxes" / "qwen-gguf.toml")
    assert gguf.engine == "llama.cpp"
    assert gguf.model.endswith(".gguf")   # single-file pull: no HF CLI needed
    assert gguf.image == ""               # exercises default-image resolution
