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


def test_finding4_workdir_without_volume_warns(hops):
    box = Box(name="w", image="i", model="m", workdir="/vllm-workspace/models")
    d = deploy.plan_serve(box, hops, dryrun=True)
    assert any("workdir" in w and "Podman will refuse" in w for w in d.warnings)
    # and the warning reaches the user on stderr through the CLI
    ok = Box(name="w2", image="i", model="m", workdir="/models",
             volumes=[Volume(source="/x", target="/models")])
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
