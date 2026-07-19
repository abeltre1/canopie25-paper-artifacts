"""Readiness-wait progress display: the pure helpers that turn a job log into a
live progress line (elapsed clock + phase + a bar parsed from the engine/container
output), so a long model load reads as forward motion instead of a silent spinner
(field request: "show something that shows progress")."""

from boxy.cli import _fmt_elapsed, _parse_load_progress, _phase_for, _progress_bar


def _log(tmp_path, text):
    p = tmp_path / "job.log"
    p.write_text(text)
    return p


def test_parse_vllm_weight_shards_ratio(tmp_path):
    p = _log(tmp_path, "INFO ...\nLoading safetensors checkpoint shards:  40% Completed | 2/5 [00:03\n")
    label, frac = _parse_load_progress(p)
    assert label == "loading weights" and abs(frac - 2 / 5) < 1e-6


def test_parse_vllm_cuda_graph_pct(tmp_path):
    p = _log(tmp_path, "Capturing CUDA graph shapes:  60%|######    | 6/10\n")
    label, frac = _parse_load_progress(p)
    assert label == "capturing CUDA graphs" and abs(frac - 0.6) < 1e-6


def test_parse_server_starting_is_most_progressed(tmp_path):
    # even with an earlier weight-load line present, "server starting" wins — it's
    # the most-progressed signal (nearly ready).
    p = _log(tmp_path, "Loading safetensors checkpoint shards: 100% | 5/5\n"
                       "INFO: Application startup complete.\n")
    label, frac = _parse_load_progress(p)
    assert label == "server starting" and frac is None


def test_parse_image_pull(tmp_path):
    p = _log(tmp_path, "Trying to pull ghcr.io/...\nCopying blob sha256:abcd\n")
    label, frac = _parse_load_progress(p)
    assert label == "pulling container image" and frac is None


def test_parse_download_pct(tmp_path):
    p = _log(tmp_path, "Downloading model.safetensors: 25%\n")
    label, frac = _parse_load_progress(p)
    assert label == "downloading model" and abs(frac - 0.25) < 1e-6


def test_parse_nothing_recognized(tmp_path):
    assert _parse_load_progress(_log(tmp_path, "random line\nanother\n")) == ("", None)


def test_parse_missing_file(tmp_path):
    assert _parse_load_progress(tmp_path / "nope.log") == ("", None)


def test_progress_bar_fraction():
    bar = _progress_bar(0.5, width=10)
    assert bar.count("#") == 5 and bar.count("-") == 5 and "50%" in bar


def test_progress_bar_clamps_and_indeterminate():
    assert "100%" in _progress_bar(1.5)          # clamped
    assert _progress_bar(-1.0).endswith("0%")    # clamped low
    assert _progress_bar(None).startswith("[·")  # indeterminate meter


def test_fmt_elapsed():
    assert _fmt_elapsed(0) == "0:00"
    assert _fmt_elapsed(9) == "0:09"
    assert _fmt_elapsed(75) == "1:15"
    assert _fmt_elapsed(-3) == "0:00"


def test_phase_for():
    assert _phase_for("PENDING", False, "") == "QUEUED"
    assert _phase_for("RUNNING", False, "") == "STARTING"
    assert _phase_for("RUNNING", True, "") == "LOADING"
    assert _phase_for("RUNNING", True, "loading weights") == "LOADING WEIGHTS"
