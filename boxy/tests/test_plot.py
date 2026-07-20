"""`boxy plot` + plotting.py: series building (pure stdlib), matplotlib
rendering (Agg, figure introspection), gap handling for crashed levels, and
the gnuplot emitter's paper-format artifacts (results.dat with X cells)."""

import json

import pytest

from boxy import plotting, results


def _env(label, tps_by_conc, crash=()):
    runs = []
    for conc, tps in tps_by_conc.items():
        if conc in crash:
            runs.append({"max_concurrency": conc, "status": "error", "error": "crash"})
            continue
        runs.append({
            "max_concurrency": conc, "status": "ok", "num_prompts": 10, "completed": 10,
            "failed": 0, "duration": 1.0, "total_input_tokens": 0, "total_output_tokens": 100,
            "request_throughput": 1.0, "output_throughput": tps, "total_token_throughput": tps,
            "mean_ttft_ms": 20.0, "median_ttft_ms": 18.0, "p99_ttft_ms": 40.0,
            "mean_tpot_ms": 5.0, "median_tpot_ms": 5.0, "p99_tpot_ms": 9.0,
            "mean_itl_ms": 5.0, "median_itl_ms": 5.0, "p99_itl_ms": 9.0,
            "mean_e2el_ms": 100.0, "median_e2el_ms": 90.0, "p95_e2el_ms": 150.0,
            "p99_e2el_ms": 180.0})
    return results.make_envelope(url="http://n:1", model="m/x", backend="synthetic",
                                 runs=runs, label=label)


TWO = [
    _env("clustera/run1", {1: 76.5, 2: 145.2, 4: 258.9, 1024: 900.0}, crash=(1024,)),
    _env("clusterb/run1", {1: 71.2, 2: 139.9, 4: 250.1, 1024: 1500.0}),
]


# ---------- series building (no matplotlib needed) ----------

def test_throughput_series_marks_crashes_as_none():
    s = plotting.throughput_series(TWO)
    assert [x.label for x in s] == ["clustera/run1", "clusterb/run1"]
    assert s[0].xs == [1, 2, 4, 1024]
    assert s[0].ys[-1] is None and s[1].ys[-1] == 1500.0


def test_latency_series_metric_selection():
    s = plotting.latency_series(TWO, metric="ttft", stat="p99")
    assert s[0].ys[0] == 40.0
    with pytest.raises(ValueError, match="unknown metric"):
        plotting.latency_series(TWO, metric="bogus", stat="p99")


def test_frontier_series_skips_crashes():
    s = plotting.frontier_series(TWO)
    assert len(s[0].xs) == 3 and len(s[1].xs) == 4


# ---------- matplotlib rendering ----------

def test_render_throughput_png(tmp_path):
    pytest.importorskip("matplotlib")
    out = plotting.render("throughput", plotting.throughput_series(TWO),
                          tmp_path / "t.png", title="demo")
    assert out.exists() and out.stat().st_size > 1000


def test_render_axes_and_series(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calls = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, ax = orig(*a, **k)
        calls["ax"] = ax
        return fig, ax

    plt.subplots, _ = spy, None
    try:
        plotting.render("throughput", plotting.throughput_series(TWO), tmp_path / "s.png")
    finally:
        plt.subplots = orig
    ax = calls["ax"]
    assert ax.get_xscale() == "log"
    assert ax.get_ylabel() == "Output Token Throughput (tokens/s)"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["clustera/run1", "clusterb/run1"]


def test_render_without_matplotlib_names_the_extra(tmp_path, monkeypatch):
    import builtins

    orig_import = builtins.__import__

    def block(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("nope")
        return orig_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(RuntimeError, match=r"boxy-hpc\[plot\]"):
        plotting.render("throughput", plotting.throughput_series(TWO), tmp_path / "x.png")


# ---------- gnuplot emitter ----------

def test_emit_gnuplot_paper_format(tmp_path):
    files = plotting.emit_gnuplot(plotting.throughput_series(TWO), tmp_path, "compare")
    dat = (tmp_path / "results.dat").read_text()
    lines = dat.strip().splitlines()
    assert lines[-1].split()[0] == "1024"
    assert "X" in lines[-1]                                 # crashed cell = literal X
    assert "1500.00" in lines[-1]
    gp = (tmp_path / "compare.gp").read_text()
    assert "set logscale x 2" in gp and "postscript eps" in gp
    assert "linespoints lw 3" in gp and "lc rgb 'red'" in gp
    assert "clusterb/run1" in gp
    sh = tmp_path / "run-gnuplot.sh"
    assert sh.exists() and "gnuplot compare.gp" in sh.read_text()
    assert set(files) == {tmp_path / "results.dat", tmp_path / "compare.gp", sh}


# ---------- CLI ----------

@pytest.fixture()
def _store_with_result(tmp_path, monkeypatch):
    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "res"))
    p1 = results.write_result(TWO[0])
    p2 = results.write_result(TWO[1])
    return tmp_path, p1, p2


def test_cli_plot_newest_default(_store_with_result, capfd):
    pytest.importorskip("matplotlib")
    from boxy.cli import main

    rc = main(["plot"])
    out = capfd.readouterr().out
    assert rc == 0 and "### Plot:" in out
    path = out.split("### Plot:", 1)[1].strip()
    assert path.endswith("-throughput.png")


def test_cli_plot_overlay_compare(_store_with_result, tmp_path, capfd):
    pytest.importorskip("matplotlib")
    from boxy.cli import main

    rc = main(["plot", "1", "2", "-o", str(tmp_path / "cmp.png")])
    out = capfd.readouterr().out
    assert rc == 0 and "cmp.png" in out


def test_cli_plot_emit_gnuplot(_store_with_result, tmp_path, capfd):
    from boxy.cli import main

    rc = main(["plot", "1", "2", "--emit", "gnuplot", "-o", str(tmp_path / "gp")])
    out = capfd.readouterr().out
    assert rc == 0 and "results.dat" in out
    assert (tmp_path / "gp" / "results.dat").exists()


def test_cli_plot_no_results_is_clean_error(tmp_path, monkeypatch, capfd):
    from boxy.cli import main

    monkeypatch.setenv("BOXY_RESULTS_DIR", str(tmp_path / "empty"))
    rc = main(["plot"])
    err = capfd.readouterr().err
    assert rc != 0 and "no bench results" in err


def test_cli_plot_kind_all(_store_with_result, tmp_path, capfd):
    pytest.importorskip("matplotlib")
    from boxy.cli import main

    rc = main(["plot", "--kind", "all", "-o", str(tmp_path)])
    out = capfd.readouterr().out
    assert rc == 0
    for kind in ("throughput", "latency", "frontier"):
        assert f"-{kind}.png" in out


def test_default_ticks_are_explicit_values(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, ax = orig(*a, **k)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = spy
    try:
        plotting.render("throughput", plotting.throughput_series(TWO), tmp_path / "v.png")
    finally:
        plt.subplots = orig
    ax = captured["ax"]
    assert list(ax.get_xticks()) == [1, 2, 4, 1024]         # the series' levels, verbatim
    labels = [ax.xaxis.get_major_formatter()(v, None) for v in ax.get_xticks()]
    assert labels == ["1", "2", "4", "1024"]                # plain values, not 2^n


def test_pow2_ticks_option_keeps_log_formatter(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, ax = orig(*a, **k)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = spy
    try:
        plotting.render("throughput", plotting.throughput_series(TWO), tmp_path / "p.png",
                        ticks="pow2")
    finally:
        plt.subplots = orig
    from matplotlib.ticker import FixedLocator

    assert not isinstance(captured["ax"].xaxis.get_major_locator(), FixedLocator)


def test_legend_enriches_old_results_with_accelerator():
    """Envelopes stored before the accelerator field existed still legend as
    '<accel>: cluster/name' — inferred from the serving image recorded in
    backend_detail. Unknown images stay untouched."""
    old = dict(TWO[0])
    old["label"] = "clusterb/boxy-llama-3.2-1b-instruct"
    old["accelerator"] = ""
    old["backend_detail"] = ("benchmark inside the serving image "
                             "docker.io/vllm/vllm-openai-rocm via podman on clusterb (agentless)")
    assert results.display_label(old) == "rocm: clusterb/boxy-llama-3.2-1b-instruct"
    s = plotting.throughput_series([old])
    assert s[0].label == "rocm: clusterb/boxy-llama-3.2-1b-instruct"
    plain = dict(TWO[0])
    plain["accelerator"] = ""
    plain["backend_detail"] = "vllm-bench binary on somewhere"
    assert ":" not in results.display_label(plain).split("/")[0]


def test_legend_never_guesses_across_clusters(tmp_path, monkeypatch):
    """Field regression: the SAME instance name served later on another cluster
    overwrote the laptop-side record, and the clusterb (ROCm) result relabeled
    itself 'cuda' from clusterc's record. A serve record may only inform the
    label when its target sits on the cluster the result was measured on."""
    jobsdir = tmp_path / "jobs"
    jobsdir.mkdir()
    monkeypatch.setenv("BOXY_JOBS_DIR", str(jobsdir))
    (jobsdir / "boxy-llama.json").write_text(json.dumps({
        "name": "boxy-llama", "target": "user1@clusterc-login",
        "image": "docker.io/vllm/vllm-openai:v0.9.1", "accelerator": "cuda",
        "gpu_type": "h100"}))
    env = dict(TWO[0])
    env.update(label="clusterb/boxy-llama", instance="boxy-llama",
               accelerator="", gpu_type="", backend_detail="")
    lbl = results.display_label(env)
    assert "cuda" not in lbl and "h100" not in lbl
    # the result's OWN cluster's record is authoritative — and the GPU model
    # (mi300a) outranks the family (rocm)
    (jobsdir / "boxy-llama.json").write_text(json.dumps({
        "name": "boxy-llama", "target": "user1@clusterb-login",
        "image": "docker.io/vllm/vllm-openai-rocm", "accelerator": "rocm",
        "gpu_type": "mi300a"}))
    assert results.display_label(env) == "mi300a: clusterb/boxy-llama"


def test_legend_reads_gpu_model_from_system_card(tmp_path, monkeypatch):
    """Sites whose scheduler doesn't label GPU types can pin gpu_type in the
    cluster's system card (~/.config/boxy/cards/systems/) — the legend reads
    it when neither the envelope nor a same-cluster record names the model."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    sysdir = tmp_path / "cfg" / "boxy" / "cards" / "systems"
    sysdir.mkdir(parents=True)
    (sysdir / "clusterb.toml").write_text(
        '[location]\nname = "clusterb"\n\n[location.resources]\n'
        'gpus_per_node = 4\ngpu_type = "AMD Instinct MI300A APU"\n')
    env = dict(TWO[0])
    env.update(label="clusterb/boxy-llama", instance="",
               accelerator="", gpu_type="", backend_detail="")
    assert results.display_label(env) == "mi300a: clusterb/boxy-llama"


def test_legend_prefers_envelope_gpu_type():
    env = dict(TWO[0])
    env.update(label="clusterb/boxy-llama", instance="",
               accelerator="rocm", gpu_type="mi300a", backend_detail="")
    assert results.display_label(env) == "mi300a: clusterb/boxy-llama"


def test_cache_kind_plots_hit_rate(tmp_path):
    env = dict(TWO[1])
    env["runs"] = [dict(r, prefix_cache_hit_rate=10.0 * i)
                   for i, r in enumerate(env["runs"], 1)]
    s = plotting.cache_series([env])
    assert s[0].ys == [10.0, 20.0, 30.0, 40.0]
    old = dict(TWO[0])                                       # runs without the metric gap out
    assert all(y is None for y in plotting.cache_series([old])[0].ys)
    pytest.importorskip("matplotlib")
    out = plotting.render("cache", s, tmp_path / "c.png")
    assert out.exists()


def test_cache_kind_renders_grouped_bars(tmp_path):
    """Hit rates cluster near 100% and overlapping lines were unreadable
    (field) — the cache figure draws grouped BARS on a full 0..100 axis; a
    level without the metric has no bar."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = dict(TWO[1])
    env["runs"] = [dict(r, prefix_cache_hit_rate=50.0) for r in env["runs"]]
    partial = dict(TWO[1])                     # one level lacks the metric
    partial["runs"] = ([dict(r, prefix_cache_hit_rate=99.0) for r in env["runs"][:3]]
                       + [dict(env["runs"][3])])
    partial["runs"][3].pop("prefix_cache_hit_rate", None)
    captured = {}
    orig = plt.subplots

    def spy(*a, **k):
        fig, ax = orig(*a, **k)
        captured["ax"] = ax
        return fig, ax

    plt.subplots = spy
    try:
        plotting.render("cache", plotting.cache_series([env, partial]), tmp_path / "c.png")
    finally:
        plt.subplots = orig
    ax = captured["ax"]
    assert len(ax.patches) == 7                # 4 bars + 3 bars (missing level gaps)
    assert ax.get_ylim() == (0.0, 105.0)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "4", "1024"]


def test_duplicate_overlay_labels_get_run_stamps():
    """Overlaying two results of the SAME instance must not draw two identical
    legend entries (field: twin 'h200: ...' lines nobody could tell apart)."""
    a = _env("clusterc/run1", {1: 70.0})
    b = _env("clusterc/run1", {1: 75.0})
    a["created"] = "2026-07-19T09:00:00Z"
    b["created"] = "2026-07-20T14:10:02Z"
    s = plotting.throughput_series([a, b])
    assert s[0].label != s[1].label
    assert s[0].label.endswith("[2026-07-19 09:00:00]")
    assert s[1].label.endswith("[2026-07-20 14:10:02]")
    lone = plotting.throughput_series([a])     # no duplicate -> no stamp
    assert lone[0].label == "clusterc/run1"
