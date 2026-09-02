"""Two claimed capabilities that need special harnesses:

1. Graceful degradation WITHOUT ramalama installed (air-gapped bootstrap):
   verified in a subprocess whose PYTHONPATH contains only boxy.
2. Live end-to-end against real Docker: serve -> endpoint -> list -> stop.
   Runs when Docker and the demo image are present (this sandbox); skips
   cleanly elsewhere (e.g. a login node without the demo image).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
SRC = str(ROOT / "src")
# The example profiles are PACKAGED (src/boxy/data/examples), not a top-level
# examples/ dir — they moved there when the repo was slimmed. Derive the path
# once so a future move breaks one line, not every test in this file.
EXAMPLES = ROOT / "src" / "boxy" / "data" / "examples"


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run python with ONLY boxy on the path (no ramalama importable)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = SRC
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=120)


def _ramalama_isolatable() -> bool:
    """True when the isolated harness can actually hide ramalama. Setting
    PYTHONPATH=SRC does NOT exclude site-packages, so when ramalama is pip-installed
    (as in CI's `.[ramalama]` env, or this sandbox) the subprocess still imports it
    and the degraded-mode precondition can't be met. CI excludes this whole file
    for exactly that reason; here we SKIP (not fail) so a full local `pytest` on a
    ramalama-present machine stays green."""
    p = _run_isolated("import importlib.util as u; print(u.find_spec('ramalama') is None)")
    return p.returncode == 0 and p.stdout.strip() == "True"


@pytest.mark.skipif(not _ramalama_isolatable(),
                    reason="ramalama is importable in the harness (pip-installed); the "
                           "no-ramalama degradation tests need it truly absent (CI excludes "
                           "this file). Run in a venv/image without ramalama to exercise them.")
class TestDegradedWithoutRamalama:
    def test_ramalama_really_absent_in_harness(self):
        p = _run_isolated("import importlib.util as u; print(u.find_spec('ramalama'))")
        assert p.stdout.strip() == "None"

    def test_accelerator_detection_works_without_ramalama(self):
        """Detection is boxy's own since accel.py landed, so this is no longer a
        degradation test — without ramalama you get a real answer, not a stub.

        Asserted host-independently on purpose. A CI runner has no GPU but DOES
        expose /dev/dri (a virtual DRM node with nothing behind it), and a dev
        box may have CUDA_VISIBLE_DEVICES exported; an equality assertion on
        either dict passes on one machine and fails on the next."""
        p = _run_isolated(
            "import json\n"
            "from boxy import accel, ramalama_shim as s\n"
            "print(json.dumps({'ramalama': s.ramalama_available(), 'accel': s.detect_accel(),\n"
            "                  'env': s.accel_env_vars(), 'devices': s.gpu_device_paths(),\n"
            "                  'known_env': sorted(accel.VISIBLE_DEVICE_VARS.values())}))\n"
        )
        assert p.returncode == 0
        got = json.loads(p.stdout.strip().splitlines()[-1])
        assert got["ramalama"] is False
        # 'none' is the honest answer on a GPU-less runner; the point is that it
        # was reached natively, with ramalama absent, rather than by giving up.
        assert got["accel"] == "none"
        assert set(got["env"]) <= set(got["known_env"])
        assert all(node == f"/dev/{name}" for name, node in got["devices"].items())

    def test_pull_transport_uri_gives_guidance(self):
        p = _run_isolated(
            "from boxy import ramalama_shim as s\n"
            "try:\n"
            "    s.pull_model('hf://o/n')\n"
            "except RuntimeError as e:\n"
            "    print('OK:', e)\n"
        )
        assert p.returncode == 0
        assert "OK:" in p.stdout and "boxy-hpc[ramalama]" in p.stdout

    def test_serve_dryrun_works_with_explicit_location(self):
        box = EXAMPLES / "boxes" / "vllm.toml"
        loc = EXAMPLES / "locations" / "flux-apptainer-rocm.toml"
        p = _run_isolated(
            "from boxy.cli import main; import sys; "
            f"sys.exit(main(['serve', '--box', {str(box)!r}, "
            f"'--location', {str(loc)!r}, '--dryrun']))"
        )
        assert p.returncode == 0
        assert "apptainer exec" in p.stdout and "vllm-rocm.sif" in p.stdout

    def test_info_reports_ramalama_not_installed(self):
        p = _run_isolated("from boxy.cli import main; import sys; sys.exit(main(['info']))")
        assert p.returncode == 0
        assert "ramalama library:" in p.stdout and "not installed" in p.stdout

    def test_default_image_fallback_map_without_ramalama(self):
        p = _run_isolated("from boxy import ramalama_shim as s; print(s.default_image('vllm', 'rocm'))")
        assert p.returncode == 0
        assert "vllm" in p.stdout.lower() or "rocm" in p.stdout.lower()


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        images = subprocess.run(["docker", "images", "-q", "boxy-demo/llamacpp:local"],
                                capture_output=True, text=True, timeout=20)
        return images.returncode == 0 and images.stdout.strip() != ""
    except Exception:
        return False


needs_live_docker = pytest.mark.skipif(
    not (_docker_ready() and (ROOT / "models" / "tiny-llama-demo.gguf").exists()),
    reason="live suite needs Docker + boxy-demo/llamacpp:local image + demo model",
)


@needs_live_docker
class TestLiveDockerCycle:
    BOX = str(EXAMPLES / "boxes" / "llamacpp-demo.toml")
    LOC = str(EXAMPLES / "locations" / "local-docker.toml")
    URL = "http://127.0.0.1:8090"

    def _boxy(self, *args, background=False):
        cmd = [sys.executable, "-m", "boxy.cli", *args]
        env = dict(os.environ, PYTHONPATH=f"{SRC}:{os.environ.get('PYTHONPATH', '')}")
        if background:
            return subprocess.Popen(cmd, env=env, cwd=ROOT,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)

    def test_full_serve_query_list_stop_cycle(self):
        subprocess.run(["docker", "rm", "-f", "llamacpp-demo"], capture_output=True)
        proc = self._boxy("serve", "--box", self.BOX, "--location", self.LOC, background=True)
        try:
            deadline = time.time() + 90
            models = None
            while time.time() < deadline:
                try:
                    with urllib.request.urlopen(f"{self.URL}/v1/models", timeout=2) as r:
                        models = json.load(r)
                    break
                except Exception:
                    time.sleep(1)
            assert models is not None, "endpoint never came up"
            assert models["data"][0]["id"] == "tiny-llama-demo.gguf"

            # real inference through the OpenAI completions route
            req = urllib.request.Request(
                f"{self.URL}/v1/completions",
                data=json.dumps({"prompt": "hpc", "max_tokens": 4}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                usage = json.load(r)["usage"]
            assert usage["completion_tokens"] == 4

            listed = self._boxy("list")
            assert listed.returncode == 0 and "llamacpp-demo" in listed.stdout

            stopped = self._boxy("stop", "--box", self.BOX)
            assert stopped.returncode == 0
        finally:
            proc.terminate()
            subprocess.run(["docker", "rm", "-f", "llamacpp-demo"], capture_output=True)

        ps = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
        assert "llamacpp-demo" not in ps.stdout


def test_dryrun_plans_without_the_ramalama_extra(monkeypatch, capsys):
    """A --dryrun PRINTS A PLAN: it must never need the network, the store, or
    an optional extra. It used to raise 'requires the ramalama package' before
    even consulting dryrun — so the documented first command failed on the
    documented first install (bare `pip install boxy-hpc`, whose README
    examples are all --dryrun)."""
    import sys

    from boxy import ramalama_shim

    # make the transport import fail exactly as it does without the extra
    monkeypatch.setitem(sys.modules, "ramalama.transports.transport_factory", None)
    path = ramalama_shim.pull_model("hf://Qwen/Qwen2.5-0.5B-Instruct", dryrun=True)
    assert path.endswith("qwen-qwen2.5-0.5b-instruct")
    out = capsys.readouterr().out
    assert "would be pulled to" in out and "plan only" in out

    # (a REAL hf:// pull no longer refuses either — boxy downloads it itself;
    # see test_hf_pull_needs_no_optional_extra)


def _fake_hub(monkeypatch, files, bodies):
    """Serve a fake Hub through boxy's own opener seam (stdlib only)."""
    import io
    import json as _json

    class _R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            url = req.full_url
            if "/api/models/" in url:
                return _R(_json.dumps({"siblings": [{"rfilename": f} for f in files]}).encode())
            return _R(bodies[url.split("/resolve/main/", 1)[1]])

    from boxy import cardgen

    monkeypatch.setattr(cardgen, "_opener", lambda: _Opener())


def test_hf_pull_needs_no_optional_extra(monkeypatch, tmp_path, capsys):
    """boxy already downloads models two ways it wrote itself (host curl over
    --ssh, huggingface_hub for bundles). The LOCAL pull was the only path that
    reached for the optional 'ramalama' extra — so a laptop had to carry an
    extra dependency closure, and an air-gap transfer its wheels, to do
    something boxy can do with certifi and the stdlib."""
    import sys

    from boxy import ramalama_shim

    monkeypatch.setitem(sys.modules, "ramalama.transports.transport_factory", None)
    monkeypatch.setenv("BOXY_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(ramalama_shim, "DEFAULT_STORE", str(tmp_path / "store"))
    _fake_hub(monkeypatch,
              ["config.json", "model.safetensors", "original/consolidated.pth", "w.gguf"],
              {"config.json": b'{"a":1}', "model.safetensors": b"W" * 64})

    path = ramalama_shim.pull_model("hf://acme/Demo-Model")
    got = sorted(os.listdir(path))
    assert got == ["config.json", "model.safetensors"]      # .pth/.gguf/original skipped
    assert (Path(path) / "config.json").read_bytes() == b'{"a":1}'
    assert path.endswith("acme-demo-model")
    # a rerun is a no-op: complete files are skipped, nothing re-downloaded
    assert ramalama_shim.pull_model("hf://acme/Demo-Model", quiet=True) == path


def test_ollama_and_oci_still_name_the_extra(monkeypatch):
    # those ARE ramalama transports; the message must say so without implying
    # hf:// needs it too
    import sys

    from boxy import ramalama_shim

    monkeypatch.setitem(sys.modules, "ramalama.transports.transport_factory", None)
    with pytest.raises(RuntimeError) as e:
        ramalama_shim.pull_model("ollama://llama3")
    assert "boxy downloads hf:// itself" in str(e.value)
    assert "ollama:// and oci:// are ramalama transports" in str(e.value)
