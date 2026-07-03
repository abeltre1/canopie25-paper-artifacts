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


def test_finding12_chainless_retry_error_still_surfaces_ssl_remedy(monkeypatch):
    """ramalama's downloader logs the SSL error per retry, then raises a FRESH
    ConnectionError with no chain ('Download failed after multiple attempts').
    boxy must still name the root cause and the remedy, from the log tap.
    (Field finding: ollama:// pull on Mac, 2026-07.)"""
    logged = [
        "❌ Network Error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "unable to get local issuer certificate (_ssl.c:1028)"
    ] * 4
    err = ConnectionError("\nDownload failed after multiple attempts.\nPossible causes:\n- Internet ...")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    msg = ramalama_shim._pull_failure_message("ollama://granite3-moe", err, logged=logged)
    assert "root cause" in msg and "CERTIFICATE_VERIFY_FAILED" in msg
    assert "SSL_CERT_FILE" in msg and "persist" in msg  # sticky remedy, not a one-shell export
    # with SSL_CERT_FILE already set, the remedy explains REPLACE semantics instead
    monkeypatch.setenv("SSL_CERT_FILE", "/some/site-ca.crt")
    msg = ramalama_shim._pull_failure_message("ollama://granite3-moe", err, logged=logged)
    assert "REPLACES" in msg and "boxy info --net" in msg and "certifi" in msg


def test_finding13_trust_bundle_merges_site_ca_with_certifi(monkeypatch, tmp_path, capsys):
    """SSL_CERT_FILE holding only the site CA breaks non-intercepted registries
    (verified: SSL_CERT_FILE REPLACES the trust store). boxy merges public CAs
    with the site CA before pulling. (Field finding: Mac run-through #3.)"""
    site = tmp_path / "site-ca.crt"
    site.write_text("-----BEGIN CERTIFICATE-----\nSITECA\n-----END CERTIFICATE-----\n")
    public = tmp_path / "certifi.pem"
    public.write_text("-----BEGIN CERTIFICATE-----\nPUBLIC\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(site))
    monkeypatch.delenv("BOXY_NO_CA_MERGE", raising=False)
    monkeypatch.setattr(ramalama_shim, "DEFAULT_STORE", str(tmp_path / "store"))
    import certifi

    monkeypatch.setattr(certifi, "where", lambda: str(public))
    merged = ramalama_shim.ensure_trust_bundle()
    assert merged and os.environ["SSL_CERT_FILE"] == merged
    text = Path(merged).read_text()
    assert "SITECA" in text and "PUBLIC" in text  # both trust roots survive
    assert "merged" in capsys.readouterr().err     # the decision is printed
    # idempotent: a second call sees the merged file and leaves it alone
    assert ramalama_shim.ensure_trust_bundle() == merged


def test_finding13b_trust_bundle_edge_cases(monkeypatch, tmp_path, capsys):
    # missing site file -> loud warning, no merge (OpenSSL ignores bad paths silently)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "nope.crt"))
    assert ramalama_shim.ensure_trust_bundle() is None
    assert "does not exist" in capsys.readouterr().err
    # opt-out respected
    site = tmp_path / "ca.crt"
    site.write_text("x")
    monkeypatch.setenv("SSL_CERT_FILE", str(site))
    monkeypatch.setenv("BOXY_NO_CA_MERGE", "1")
    assert ramalama_shim.ensure_trust_bundle() is None
    # unset -> nothing to do
    monkeypatch.delenv("BOXY_NO_CA_MERGE")
    monkeypatch.delenv("SSL_CERT_FILE")
    assert ramalama_shim.ensure_trust_bundle() is None


def test_finding13c_info_net_probes_registries(monkeypatch, capsys):
    import urllib.error
    import urllib.request

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url, timeout=0):
        if "ollama" in url:
            raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ramalama_shim, "ensure_trust_bundle", lambda: None)
    monkeypatch.delenv("BOXY_ALLOW_TRANSPORTS", raising=False)
    rc = main(["info", "--net"])
    out = capsys.readouterr().out
    assert rc == 0  # HTTP errors still prove TLS; nothing actually failed
    assert "hf://" in out and "OK (HTTP 200)" in out
    assert "OK (TLS fine; HTTP 403)" in out
    assert "modelscope" not in out                   # blocked registries are not probed


def test_finding12b_log_tap_captures_ramalama_errors(monkeypatch):
    """End-to-end through pull_model: a transport that logs then raises
    chain-less must still produce the SSL remedy."""
    import logging

    class FakeTransport:
        def ensure_model_exists(self, args):
            logging.getLogger("ramalama").error(
                "❌ Network Error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
            )
            raise ConnectionError("\nDownload failed after multiple attempts.\n")

        def _get_entry_model_path(self, *a):
            raise AssertionError("unreachable")

    import ramalama.transports.transport_factory as tf

    monkeypatch.setattr(tf, "New", lambda uri, args: FakeTransport())
    with pytest.raises(RuntimeError) as e:
        ramalama_shim.pull_model("ollama://granite3-moe")
    assert "SSL_CERT_FILE" in str(e.value) and "CERTIFICATE_VERIFY_FAILED" in str(e.value)


def test_finding12c_info_reports_tls_state(monkeypatch, capsys):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert main(["info"]) == 0
    assert "tls: system default CA store" in capsys.readouterr().out
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/ca.crt")
    assert main(["info"]) == 0
    assert "MISSING FILE" in capsys.readouterr().out


def _hf_401_error():
    try:
        try:
            raise OSError("HTTP Error 401: Unauthorized")
        except OSError:
            raise NotImplementedError("huggingface cli download not available")
    except NotImplementedError as e:
        return e


@pytest.mark.parametrize("verdict,expect", [
    ("public", "EXISTS and is PUBLIC"),
    ("gated", "GATED"),
    ("missing", "does not exist under this"),
    (None, "could not reach the HF API"),
])
def test_finding15_hf_401_probe_gives_a_verdict(monkeypatch, verdict, expect):
    """HF 401 has three unrelated causes (stale token / nonexistent repo /
    gated repo) — boxy probes anonymously and names the actual one.
    (Field findings 15+16: three 401s in a row on a Mac, 2026-07.)"""
    monkeypatch.setattr(ramalama_shim, "_hf_repo_info", lambda repo: (verdict, []))
    monkeypatch.setattr(ramalama_shim, "_hf_token_sources", lambda: [])
    msg = ramalama_shim._pull_failure_message(
        "hf://bartowski/Llama-3.1-8B-GGUF/Llama-3.1-8B-Q4_K_M.gguf", _hf_401_error())
    assert expect in msg
    assert "bartowski/Llama-3.1-8B-GGUF" in msg


def test_finding16_stale_token_named_as_401_cause(monkeypatch):
    """A stale cached token 401s EVERY repo — when a token source exists, the
    message must name it and give the anonymous-retry command."""
    monkeypatch.setattr(ramalama_shim, "_hf_repo_info", lambda repo: ("public", []))
    monkeypatch.setattr(ramalama_shim, "_hf_token_sources",
                        lambda: ["~/.cache/huggingface/token (huggingface-cli login)"])
    msg = ramalama_shim._pull_failure_message("hf://org/repo/f.gguf", _hf_401_error())
    assert "token IS being sent" in msg and "HF_TOKEN=''" in msg
    assert "EVERY repo, even public ones" in msg


def _hf_404_error():
    try:
        raise RuntimeError('"Failed to pull model: \'... HTTP Error 404: Not Found\'"')
    except RuntimeError as e:
        return e


def test_finding17_hf_404_lists_actual_gguf_files(monkeypatch):
    """404 with working auth = wrong path. Quantizers name files unpredictably
    (TheBloke lowercases, bartowski doesn't) — list what the repo really has.
    (Field finding: guessed file name, Mac 2026-07.)"""
    monkeypatch.setattr(ramalama_shim, "_hf_repo_info",
                        lambda repo: ("public", ["tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                                                 "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"]))
    msg = ramalama_shim._pull_failure_message(
        "hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/TinyLlama-WRONG.gguf", _hf_404_error())
    assert "has no file named" in msg
    assert "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" in msg    # the real files, listed
    assert "boxy serve hf://TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" in msg


def test_finding17b_hf_404_missing_repo_points_at_search(monkeypatch):
    monkeypatch.setattr(ramalama_shim, "_hf_repo_info", lambda repo: ("missing", []))
    msg = ramalama_shim._pull_failure_message("hf://bartowski/Nope-GGUF/x.gguf", _hf_404_error())
    assert "does not exist" in msg and "models?search=Nope-GGUF" in msg and "ollama://" in msg


def test_finding17c_hf_404_repo_without_ggufs(monkeypatch):
    monkeypatch.setattr(ramalama_shim, "_hf_repo_info", lambda repo: ("public", []))
    msg = ramalama_shim._pull_failure_message("hf://meta-llama/Llama-3-8B/x.gguf", _hf_404_error())
    assert "contains no .gguf files" in msg and "tree/main" in msg


def test_finding16b_token_sources_detection(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(ramalama_shim.os.path, "expanduser",
                        lambda p: str(tmp_path / "token") if "huggingface" in p else p)
    assert ramalama_shim._hf_token_sources() == []
    monkeypatch.setenv("HF_TOKEN", "hf_stale")
    (tmp_path / "token").write_text("hf_old")
    sources = ramalama_shim._hf_token_sources()
    assert len(sources) == 2 and "HF_TOKEN" in sources[0]


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
    # a --port in the command REPLACES the declared box port: publishing the
    # stale 8090 would bind the very host port the user was avoiding
    # (sweep finding 58)
    assert published == ["8001:8001"]

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
