"""Post-bench plotting: turn persisted `boxy-bench/1` results into figures
(matplotlib, optional `[plot]` extra) or into the paper's exact gnuplot
pipeline (`--emit gnuplot` writes results.dat + .gp + run-gnuplot.sh, the
artifacts that were hand-maintained under plots/ for the paper).

Series building is pure stdlib so it is unit-testable without matplotlib;
only render() touches matplotlib (lazy import, Agg backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("throughput", "latency", "cache", "frontier")

# one metric family per --metric choice; stat prefixes follow the canonical keys
_METRIC_KEY = {"ttft": "ttft_ms", "tpot": "tpot_ms", "itl": "itl_ms", "e2e": "e2el_ms"}
_STAT_PREFIX = {"mean": "mean_", "median": "median_", "p99": "p99_"}

# the paper's series colors, in its order (plots/*/llama4-scout.gp)
GNUPLOT_COLORS = ["red", "orange", "green", "blue", "purple", "gray", "black"]

GNUPLOT_PREAMBLE = """\
set terminal postscript eps enhanced color 20
set output '{stem}.eps'

set xlabel '{xlabel}'
set ylabel '{ylabel}'

set pointsize 1.25
set logscale x 2
set xrange [.9:1050]
set yrange [10:]
set key at screen 0.7,0.93

datafile = 'results.dat'
"""


@dataclass
class Series:
    """One plotted line: xs = concurrency levels, ys parallel (None = the level
    errored/crashed — a gap in matplotlib, a literal X in the gnuplot dat, the
    paper's crash marker)."""
    label: str
    xs: list[int] = field(default_factory=list)
    ys: list[float | None] = field(default_factory=list)


def _metric_key(metric: str, stat: str) -> str:
    try:
        return _STAT_PREFIX[stat] + _METRIC_KEY[metric]
    except KeyError:
        raise ValueError(f"unknown metric/stat {metric!r}/{stat!r} — metric one of "
                         f"{'|'.join(_METRIC_KEY)}, stat one of {'|'.join(_STAT_PREFIX)}") from None


def _display_labels(envelopes: list[dict]) -> list[str]:
    """Legend labels, disambiguated: overlaying two results of the SAME
    instance (a before/after rerun) must not draw two identical legend entries
    nobody can tell apart (field: twin 'h200: ...' lines) — repeats get the
    run's creation stamp appended."""
    from boxy import results as _results

    labels = [_results.display_label(e) for e in envelopes]
    dup = {lbl for lbl in labels if labels.count(lbl) > 1}
    out, nth = [], {}
    for lbl, env in zip(labels, envelopes):
        if lbl in dup:
            nth[lbl] = nth.get(lbl, 0) + 1
            stamp = (env.get("created") or "").replace("T", " ").removesuffix("Z")
            out.append(f"{lbl} [{stamp or f'run {nth[lbl]}'}]")
        else:
            out.append(lbl)
    return out


def _series(envelopes: list[dict], value_key: str) -> list[Series]:
    out = []
    for env, label in zip(envelopes, _display_labels(envelopes)):
        s = Series(label=label)
        for run in env.get("runs", []):
            s.xs.append(int(run.get("max_concurrency", 0)))
            s.ys.append(float(run[value_key]) if run.get("status") == "ok"
                        and run.get(value_key) is not None else None)
        out.append(s)
    return out


def throughput_series(envelopes: list[dict]) -> list[Series]:
    """The paper's figure: Output token throughput vs max request concurrency."""
    return _series(envelopes, "output_throughput")


def latency_series(envelopes: list[dict], metric: str = "ttft", stat: str = "p99") -> list[Series]:
    return _series(envelopes, _metric_key(metric, stat))


def cache_series(envelopes: list[dict]) -> list[Series]:
    """Server-side prefix-cache hit rate (%) per concurrency level, sampled
    from vLLM's /metrics around each level; levels without the metric gap."""
    return _series(envelopes, "prefix_cache_hit_rate")


def frontier_series(envelopes: list[dict]) -> list[Series]:
    """Latency-throughput frontier: one point per level — x = output tok/s,
    y = median E2E latency. (xs carry throughputs here, not concurrency.)"""
    out = []
    for env, label in zip(envelopes, _display_labels(envelopes)):
        s = Series(label=label)
        for run in env.get("runs", []):
            if run.get("status") != "ok":
                continue
            s.xs.append(run.get("output_throughput", 0.0))
            s.ys.append(run.get("median_e2el_ms", 0.0))
        out.append(s)
    return out


def _axis_labels(kind: str, metric: str, stat: str) -> tuple[str, str]:
    if kind == "throughput":
        return "Maximum Request Concurrency", "Output Token Throughput (tokens/s)"
    if kind == "cache":
        return "Maximum Request Concurrency", "Prefix Cache Hit Rate (%)"
    if kind == "latency":
        return "Maximum Request Concurrency", f"{stat} {metric.upper()} (ms)"
    return "Output Token Throughput (tokens/s)", "Median E2E Latency (ms)"


# vivid primaries for print-quality figures (the paper's gnuplot order)
MPL_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#7f7f7f", "#000000"]


def render(kind: str, series: list[Series], out: Path, fmt: str = "png",
           title: str = "", logx2: bool = True, ticks: str = "values",
           dpi: int = 300) -> Path:
    """Render one figure with matplotlib (Agg). Raises RuntimeError with the
    install hint when matplotlib is absent — `--emit gnuplot` never needs it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("plotting needs matplotlib: pip install 'boxy-hpc[plot]' — "
                           "or use `boxy plot --emit gnuplot` (no python deps)") from None
    fig, ax = plt.subplots(figsize=(10, 6))
    if kind == "cache":
        # hit rates cluster near 100% — lines overlap into one unreadable band
        # (field). Grouped bars on a full 0..100 axis show each level's rate
        # per series; a level without the metric simply has no bar.
        levels = sorted({x for s in series for x in s.xs})
        pos = {x: i for i, x in enumerate(levels)}
        width = 0.8 / max(1, len(series))
        for idx, s in enumerate(series):
            xs = [pos[x] + (idx - (len(series) - 1) / 2) * width
                  for x, y in zip(s.xs, s.ys) if y is not None]
            ys = [y for y in s.ys if y is not None]
            ax.bar(xs, ys, width=width * 0.92, color=MPL_COLORS[idx % len(MPL_COLORS)],
                   label=s.label)
        ax.set_xticks(range(len(levels)), [str(x) for x in levels])
        ax.set_ylim(0, 105)
        logx2 = False                              # categorical positions, not values
    for idx, s in enumerate(series if kind != "cache" else []):
        series_color = MPL_COLORS[idx % len(MPL_COLORS)]
        if kind == "frontier":
            ax.plot(s.xs, s.ys, marker="o", linewidth=2.5, color=series_color, label=s.label)
            continue
        # split at None gaps so crashed levels break the line (paper's X cells)
        xs, ys = [], []
        segments = []
        for x, y in zip(s.xs, s.ys):
            if y is None:
                if xs:
                    segments.append((xs, ys))
                xs, ys = [], []
            else:
                xs.append(x)
                ys.append(y)
        if xs:
            segments.append((xs, ys))
        for i, (sx, sy) in enumerate(segments):
            ax.plot(sx, sy, marker="o", linewidth=2.5, color=series_color,
                    label=s.label if i == 0 else None)
    xlabel, ylabel = _axis_labels(kind, "", "")
    if kind == "latency":
        ylabel = "Latency (ms)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx2 and kind != "frontier":
        ax.set_xscale("log", base=2)
        if ticks == "values":
            # explicit concurrency values (1, 2, 4, ... 1024) beat exponent
            # notation for reading a specific level off the figure; --ticks
            # pow2 restores matplotlib's 2^n labels.
            from matplotlib.ticker import FixedLocator, FuncFormatter

            levels = sorted({x for s in series for x in s.xs if x})
            if levels:
                ax.xaxis.set_major_locator(FixedLocator(levels))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):d}"))
                ax.xaxis.set_minor_locator(FixedLocator([]))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y" if kind == "cache" else "both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, format=fmt, dpi=dpi)
    plt.close(fig)
    return out


def emit_gnuplot(series: list[Series], outdir: Path, stem: str,
                 ylabel: str = "Output Token Throughput (tokens/s)",
                 xlabel: str = "Maximum Request Concurrency") -> list[Path]:
    """Write the paper-pipeline artifacts: results.dat (col 1 = union of
    concurrency levels; one column per series; literal X for crashed/missing
    cells — exactly plots/*/results.dat), <stem>.gp with the centralized
    preamble, and run-gnuplot.sh."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    xs = sorted({x for s in series for x in s.xs})
    lookup = [{x: y for x, y in zip(s.xs, s.ys)} for s in series]
    lines = ["#" + " ".join(f"{s.label:>20}" for s in series),
             f"#{'max_concurrency':>15} " + " ".join(f"{'tokens/s':>12}" for _ in series)]
    for x in xs:
        cells = []
        for lk in lookup:
            y = lk.get(x)
            cells.append(f"{y:>12.2f}" if isinstance(y, (int, float)) and y is not None
                         else f"{'X':>12}")
        lines.append(f"{x:>16} " + " ".join(cells))
    dat = outdir / "results.dat"
    dat.write_text("\n".join(lines) + "\n")

    plot_rows = []
    for i, s in enumerate(series):
        color = GNUPLOT_COLORS[i % len(GNUPLOT_COLORS)]
        plot_rows.append(f"    datafile using 1:{i + 2} title \"{s.label}\" "
                         f"with linespoints lw 3 lc rgb '{color}', \\")
    gp = outdir / f"{stem}.gp"
    gp.write_text(GNUPLOT_PREAMBLE.format(stem=stem, xlabel=xlabel, ylabel=ylabel)
                  + "\nplot \\\n" + "\n".join(plot_rows) + "\n\nset output\n")
    sh = outdir / "run-gnuplot.sh"
    sh.write_text(f"#!/bin/sh\ngnuplot {stem}.gp\n")
    sh.chmod(0o755)
    return [dat, gp, sh]
