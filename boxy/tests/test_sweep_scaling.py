"""Phase 3: scaling sweep + multi-endpoint bench aggregation.

Pure ScalingReport/summarize tests, real multi-endpoint aggregation against
local HTTP servers, and the `boxy sweep` CLI dryrun + guards.
"""

import threading
from http.server import HTTPServer

import pytest

from boxy import bench
from boxy.cli import main
from tests.test_bench_and_cloud import _FakeOpenAI


@pytest.fixture
def two_endpoints():
    servers = [HTTPServer(("127.0.0.1", 0), _FakeOpenAI) for _ in range(2)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield [f"http://127.0.0.1:{s.server_address[1]}" for s in servers]
    for s in servers:
        s.shutdown()


# ---- pure: ScalingReport / summarize_point ----------------------------------


def _pt(value, tok):
    return bench.ScalingPoint(label=f"nodes={value}", axis="nodes", value=value, endpoints=1,
                              peak_batch=64, requests_per_s=tok / 10, tokens_per_s=tok,
                              latency_p50_ms=100.0, latency_p95_ms=200.0)


def test_scaling_report_table_has_speedup_vs_first_rung():
    report = bench.ScalingReport(axis="nodes", model="m", max_tokens=32,
                                 points=[_pt(1, 100.0), _pt(2, 180.0), _pt(4, 320.0)])
    table = report.to_table()
    assert "nodes" in table and "tok/s" in table
    assert "1.80x" in table  # 180/100
    assert "3.20x" in table  # 320/100


def test_scaling_report_csv_and_json():
    report = bench.ScalingReport(axis="replicas", model="m", max_tokens=8, points=[_pt(1, 50.0)])
    csv = report.to_csv()
    assert csv.splitlines()[0].startswith("label,axis,value,endpoints,peak_batch")
    import json
    obj = json.loads(report.to_json())
    assert obj["axis"] == "replicas" and obj["points"][0]["tokens_per_s"] == 50.0


def test_summarize_point_picks_peak_tokens_per_s():
    report = bench.BenchReport(url="u", model="m", max_tokens=8)
    for bs, tok in [(1, 10.0), (8, 90.0), (64, 250.0), (128, 240.0)]:
        report.results.append(bench.BenchResult(
            batch_size=bs, requests=bs, ok=bs, errors=0, elapsed_s=1.0, requests_per_s=bs,
            prompt_tokens=0, completion_tokens=0, tokens_per_s=tok,
            latency_mean_ms=1.0, latency_p50_ms=1.0, latency_p95_ms=2.0))
    pt = bench.summarize_point("nodes=2", "nodes", 2, 1, report)
    assert pt.peak_batch == 64 and pt.tokens_per_s == 250.0  # not the 128 rung (240)


# ---- real multi-endpoint aggregation ----------------------------------------


def test_run_level_endpoints_spreads_load_across_endpoints(two_endpoints):
    eps = [(u, "fake-model") for u in two_endpoints]
    r = bench.run_level_endpoints(eps, ["p"], batch_size=8, max_tokens=4)
    assert r.ok == 8 and r.errors == 0
    assert r.completion_tokens == 32  # 8 requests x 4 tokens, pooled across both


def test_run_scaling_point_aggregates_fleet(two_endpoints):
    rep = bench.run_scaling_point(two_endpoints, batch_sizes=[2, 4], max_tokens=8)
    assert rep.model == "fake-model"
    assert [r.batch_size for r in rep.results] == [2, 4]
    assert rep.results[-1].ok == 4 and rep.results[-1].tokens_per_s > 0
    assert "," in rep.url  # both endpoints recorded in the fleet url


# ---- boxy sweep CLI ----------------------------------------------------------


def test_sweep_nodes_dryrun_plans_each_rung(capsys):
    rc = main(["sweep", "Meta-Llama-3.1-8B", "--scheduler", "slurm", "--gpus", "4",
               "--sweep-nodes", "1,2,4", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Scaling sweep: nodes = 1, 2, 4" in out
    for tag in ("-n1", "-n2", "-n4"):
        assert f"--job-name=boxy-meta-llama-3.1-8b{tag}" in out
    # nodes>1 rungs are distributed (one Ray task per node); nodes=1 is not
    assert out.count("#SBATCH --ntasks-per-node=1") == 2
    assert "3 rungs planned" in out


def test_sweep_replicas_dryrun_fans_out(capsys):
    rc = main(["sweep", "Meta-Llama-3.1-8B", "--scheduler", "slurm", "--gpus", "4",
               "--sweep-replicas", "1,2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    # replicas=2 rung fans out to two per-replica jobs, tagged x2 then -r0/-r1
    assert "--job-name=boxy-meta-llama-3.1-8b-x2-r0" in out
    assert "--job-name=boxy-meta-llama-3.1-8b-x2-r1" in out


def test_sweep_requires_exactly_one_axis(capsys):
    rc = main(["sweep", "M", "--scheduler", "slurm", "--sweep-nodes", "1,2",
               "--sweep-replicas", "1,2", "--dryrun"])
    assert rc == 2
    assert "exactly one of --sweep-nodes or --sweep-replicas" in capsys.readouterr().err


def test_sweep_needs_scheduler(capsys):
    rc = main(["sweep", "M", "--sweep-nodes", "1,2", "--dryrun"])
    assert rc == 2
    assert "needs --scheduler slurm|flux" in capsys.readouterr().err


def test_sweep_rejects_nonint_list(capsys):
    rc = main(["sweep", "M", "--scheduler", "slurm", "--sweep-nodes", "1,two", "--dryrun"])
    assert rc == 2
    assert "must be a comma list of integers" in capsys.readouterr().err
