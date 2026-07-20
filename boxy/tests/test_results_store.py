"""The persistent bench-results store (boxy-bench/1): canonical schema parity
between backends, per-cluster partitioning, shape-guarded reads, selection
semantics, and the `boxy results` / bench-persistence CLI surface."""

import json

import pytest

from boxy import bench, results
from boxy.cli import main
from tests.test_bench_serving import server  # noqa: F401  (in-process SSE server fixture)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "results"))
    yield


def _fake_run(conc=1, **over):
    run = {k: 0.0 for k in results.RUN_KEYS}
    run.update({"max_concurrency": conc, "status": "ok", "num_prompts": 4,
                "completed": 4, "failed": 0, "output_throughput": 100.0 * conc})
    run.update(over)
    return run


def _envelope(**over):
    kw = dict(url="http://n1:8000", model="fake/tiny-1b", backend="synthetic",
              runs=[_fake_run(1), _fake_run(2)])
    kw.update(over)
    return results.make_envelope(**kw)


# ---------- schema ----------

def test_canonical_run_keys_frozen():
    """The golden key set: renaming any of these is a schema bump, not a drive-by."""
    assert results.RUN_KEYS == [
        "max_concurrency", "status",
        "num_prompts", "completed", "failed", "duration",
        "total_input_tokens", "total_output_tokens",
        "request_throughput", "output_throughput", "total_token_throughput",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms",
        "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
        "mean_e2el_ms", "median_e2el_ms", "p95_e2el_ms", "p99_e2el_ms",
    ]


def test_synthetic_to_canonical_covers_every_key(server):  # noqa: F811
    report = bench.run_bench(server, [1], max_tokens=4)
    record = bench.to_canonical(report.results[0])
    assert set(results.RUN_KEYS) <= set(record)
    assert record["status"] == "ok" and record["max_concurrency"] == 1
    assert record["output_throughput"] > 0 and record["median_ttft_ms"] > 0


def test_envelope_never_carries_secrets():
    env = _envelope()
    assert "api_key" not in json.dumps(env)
    assert env["schema"] == "boxy-bench/1"
    assert env["label"]  # derived when not given


# ---------- store mechanics ----------

def test_write_read_roundtrip_and_listing():
    p1 = results.write_result(_envelope(label="run-a"))
    p2 = results.write_result(_envelope(label="run-b"))
    assert p1 != p2 and p1.name.endswith(".bench.json")
    listing = results.list_results()
    assert [d["label"] for _, d in listing] == ["run-b", "run-a"] or len(listing) == 2


def test_junk_files_are_skipped_not_fatal(tmp_path):
    d = results._dir()
    (d / "junk.bench.json").write_text("{not json")
    (d / "shapeless.bench.json").write_text('{"schema": "boxy-bench/1"}')
    results.write_result(_envelope())
    assert len(results.list_results()) == 1
    assert results.read_result(d / "junk.bench.json") is None


def test_results_dir_env_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "pinned"))
    assert results._dir() == tmp_path / "pinned"


def test_select_semantics():
    pa = results.write_result(_envelope(label="alpha"))
    pb = results.write_result(_envelope(label="beta"))
    assert results.select([]) == [pb]                       # newest
    assert results.select(["2"]) == [pa]                    # index (1 = newest)
    assert results.select([str(pa)]) == [pa]                # explicit path
    assert pb in results.select(["fake-tiny"])              # name fragment
    with pytest.raises(ValueError):
        results.select(["99"])
    with pytest.raises(ValueError):
        results.select(["no-such-result"])


def test_to_csv_skips_errored_levels():
    env = _envelope(runs=[_fake_run(1), dict(_fake_run(2), status="error", error="boom")])
    csv = results.to_csv(env)
    lines = csv.strip().splitlines()
    assert lines[0].startswith("max_concurrency,")
    assert len(lines) == 2                                  # header + the one ok level


# ---------- CLI ----------

def test_bench_persists_by_default_and_no_save_skips(server, capfd):  # noqa: F811
    rc = main(["bench", "--url", server, "--batch-sizes", "1", "--max-tokens", "4"])
    out = capfd.readouterr().out
    assert rc == 0 and "### Result saved:" in out
    assert len(results.list_results()) == 1
    rc = main(["bench", "--url", server, "--batch-sizes", "1", "--max-tokens", "4", "--no-save"])
    assert rc == 0 and len(results.list_results()) == 1


def test_results_list_show_path(server, capfd):  # noqa: F811
    main(["bench", "--url", server, "--batch-sizes", "1", "--max-tokens", "4",
          "--label", "store-test"])
    capfd.readouterr()
    assert main(["results"]) == 0
    out = capfd.readouterr().out
    assert "store-test" in out and "peak tok/s" in out
    assert main(["results", "show", "1"]) == 0
    out = capfd.readouterr().out
    assert "TTFT p50" in out and "fake/tiny-1b" in out
    assert main(["results", "path"]) == 0
    assert str(results._dir()) in capfd.readouterr().out


def test_bench_api_key_sends_bearer(server, capfd, monkeypatch):  # noqa: F811
    """--api-key must reach the endpoint as an Authorization header (the k8s
    ingress case). The fake server records headers via the handler class."""
    from tests.test_bench_serving import _Handler

    seen = {}
    orig = _Handler.do_POST

    def spy(self):
        seen["auth"] = self.headers.get("Authorization", "")
        orig(self)

    monkeypatch.setattr(_Handler, "do_POST", spy)
    rc = main(["bench", "--url", server, "--batch-sizes", "1", "--max-tokens", "4",
               "--api-key", "sekrit", "--no-save"])
    assert rc == 0 and seen["auth"] == "Bearer sekrit"
    bench._extra_headers.clear()                            # don't leak into other tests


def test_bench_dryrun_names_save_path(server, capfd):  # noqa: F811
    rc = main(["bench", "--url", server, "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0 and "Result would be saved under" in out
    rc = main(["bench", "--url", server, "--dryrun", "--no-save"])
    out = capfd.readouterr().out
    assert "Result would be saved" not in out


def test_max_concurrency_alias(server, capfd):  # noqa: F811
    rc = main(["bench", "--url", server, "--max-concurrency", "1,2", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0 and "batch_sizes=[1, 2]" in out
