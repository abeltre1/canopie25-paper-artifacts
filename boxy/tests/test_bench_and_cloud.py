"""Phase 4 (bench) and Phase 5 (cloud launch) tests.

The bench tests run against a real local HTTP server implementing the two
OpenAI endpoints — the full request/measure/aggregate path is exercised for
real, no mocks of boxy code.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from boxy import bench, cloud
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location, Resources
from tests.conftest import EXAMPLES


class _FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep test output quiet
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.path == "/v1/models"
        self._send({"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

    def do_POST(self):
        assert self.path == "/v1/completions"
        n = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(n))
        assert payload["model"] == "fake-model"
        self._send(
            {
                "choices": [{"text": "ok", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": payload["max_tokens"], "total_tokens": 7},
            }
        )


@pytest.fixture
def fake_endpoint():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_bench_end_to_end_against_real_http(fake_endpoint):
    report = bench.run_bench(fake_endpoint, batch_sizes=[1, 4], max_tokens=8)
    assert report.model == "fake-model"
    assert [r.batch_size for r in report.results] == [1, 4]
    r = report.results[1]
    assert r.ok == 4 and r.errors == 0
    assert r.completion_tokens == 32  # 4 requests x max_tokens=8
    assert r.tokens_per_s > 0 and r.latency_p95_ms >= r.latency_p50_ms > 0


def test_bench_csv_shape(fake_endpoint):
    report = bench.run_bench(fake_endpoint, batch_sizes=[2], max_tokens=4)
    csv = report.to_csv()
    header, row = csv.strip().split("\n")
    assert header.startswith("batch_size,requests,ok,errors")
    assert row.startswith("2,2,2,0")


def test_bench_counts_errors_when_endpoint_down():
    result = bench.run_level("http://127.0.0.1:1", "m", ["p"], batch_size=2, max_tokens=4)
    assert result.errors == 2 and result.ok == 0


def test_bench_prompt_loading_sharegpt(tmp_path):
    dataset = tmp_path / "sharegpt.json"
    dataset.write_text(json.dumps([
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]},
        {"conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "there"}]},
    ]))
    assert bench.load_prompts(str(dataset)) == ["hi", "there"]
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(["a", "b"]))
    assert bench.load_prompts(str(plain)) == ["a", "b"]
    assert bench.load_prompts(None) == bench.SYNTHETIC_PROMPTS


def test_bench_cli_dryrun(capsys):
    rc = main(["bench", "--box", str(EXAMPLES / "boxes" / "vllm.toml"), "--batch-sizes", "1,2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "url=http://127.0.0.1:8000" in out and "batch_sizes=[1, 2]" in out


def test_launch_command_shapes():
    box = Box(name="svc", image="i")
    assert cloud.launch_command(box, "/t.yaml", serve=False) == \
        ["sky", "launch", "-c", "svc", "/t.yaml", "--yes"]
    assert cloud.launch_command(box, "/t.yaml", serve=True) == \
        ["sky", "serve", "up", "-n", "svc", "/t.yaml", "--yes"]
    assert cloud.launch_command(box, "", serve=False, down=True) == ["sky", "down", "svc", "--yes"]
    assert cloud.launch_command(box, "", serve=True, down=True) == \
        ["sky", "serve", "down", "svc", "--yes"]


def test_launch_writes_valid_yaml(tmp_path):
    box = Box(name="svc", image="img:1", model="m", ports=[8000])
    loc = Location(name="cloud", scheduler="none", accelerator="cuda", runtime="docker",
                   resources=Resources(nodes=1, gpus_per_node=2, accelerator_type="A100"))
    cloud.write_task_yaml(box, loc, port=None, serve=True, output=str(tmp_path / "t.yaml"))
    text = (tmp_path / "t.yaml").read_text()
    assert "accelerators: A100:2" in text and "service:" in text


def test_launch_cli_dryrun(capsys):
    rc = main([
        "launch",
        "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
        "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml"),
        "--serve", "--dryrun",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sky serve up -n vllm" in out and "### Task YAML:" in out


def test_launch_cli_warns_on_hpc_scheduler(capsys):
    rc = main([
        "launch",
        "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
        "--location", str(EXAMPLES / "locations" / "hops.toml"),
        "--dryrun",
    ])
    assert rc == 0
    assert "use `boxy serve` for Slurm/Flux" in capsys.readouterr().err
