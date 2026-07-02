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


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    """Run python with ONLY boxy on the path (no ramalama importable)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = SRC
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=120)


class TestDegradedWithoutRamalama:
    def test_ramalama_really_absent_in_harness(self):
        p = _run_isolated("import importlib.util as u; print(u.find_spec('ramalama'))")
        assert p.stdout.strip() == "None"

    def test_detect_accel_degrades_to_none(self):
        p = _run_isolated("from boxy import ramalama_shim as s; "
                          "print(s.ramalama_available(), s.detect_accel(), s.accel_env_vars(), s.gpu_device_paths())")
        assert p.returncode == 0
        assert p.stdout.strip() == "False none {} {}"

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
        p = _run_isolated(
            "from boxy.cli import main; import sys; "
            "sys.exit(main(['serve', '--box', 'examples/boxes/vllm.toml', "
            "'--location', 'examples/locations/eldorado.toml', '--dryrun']))"
        )
        assert p.returncode == 0
        assert "apptainer exec" in p.stdout and "vllm-rocm.sif" in p.stdout

    def test_info_reports_ramalama_not_installed(self):
        p = _run_isolated("from boxy.cli import main; import sys; sys.exit(main(['info']))")
        assert p.returncode == 0
        assert "ramalama library: not installed" in p.stdout

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
    BOX = str(ROOT / "examples" / "boxes" / "llamacpp-demo.toml")
    LOC = str(ROOT / "examples" / "locations" / "local-docker.toml")
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
