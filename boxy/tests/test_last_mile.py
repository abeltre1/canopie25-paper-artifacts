"""Last-mile coverage: the remaining in-process branches."""

import json
import threading
from http.server import HTTPServer

import pytest

from boxy import bench, engines
from boxy.box import Box
from boxy.cli import main
from boxy.location import Location
from boxy.schedulers.base import Scheduler
from tests.test_bench_and_cloud import _FakeOpenAI


def test_box_unknown_keys_from_toml(tmp_path):
    bad = tmp_path / "b.toml"
    bad.write_text('[box]\nname="x"\nimage="i"\nbogus="y"\n')
    with pytest.raises(ValueError, match="unknown \\[box\\] keys"):
        Box.from_toml(bad)


def test_location_missing_section(tmp_path):
    bad = tmp_path / "l.toml"
    bad.write_text("[notlocation]\nx=1\n")
    with pytest.raises(ValueError, match="missing \\[location\\] section"):
        Location.from_toml(bad)


def test_scheduler_base_defaults():
    class Bare(Scheduler):
        name = "bare"

    loc = Location(name="l")
    assert Bare().wrap(["x"], loc) == ["x"]
    assert Bare().host_env_fixups() == []


def test_tack_on_bool_values_both_styles():
    cmd = engines._tack_on_last(["e"], {"enforce_eager": True, "disable_thing": False})
    assert cmd == ["e", "--enforce-eager"]  # True -> bare flag, False -> omitted
    cmd = engines._tack_on_last(["e"], {"verbose": True}, style="space")
    assert cmd == ["e", "--verbose"]


def test_bench_rejects_promptless_dataset(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps([{"conversations": [{"from": "gpt", "value": "x"}]}]))
    with pytest.raises(ValueError, match="no prompts found"):
        bench.load_prompts(str(empty))


@pytest.fixture
def fake_endpoint():
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_cli_bench_table_and_json_and_csv(fake_endpoint, tmp_path, capsys):
    from tests.conftest import EXAMPLES

    csv_out = tmp_path / "r.csv"
    rc = main(["bench", "--box", str(EXAMPLES / "boxes" / "vllm.toml"), "--url", fake_endpoint,
               "--batch-sizes", "1,2", "--max-tokens", "4", "-o", str(csv_out)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model=fake-model" in out and "tok/s" in out
    assert csv_out.read_text().startswith("batch_size,")
    rc = main(["bench", "--box", str(EXAMPLES / "boxes" / "vllm.toml"), "--url", fake_endpoint,
               "--batch-sizes", "1", "--json"])
    assert rc == 0
    assert '"results"' in capsys.readouterr().out
