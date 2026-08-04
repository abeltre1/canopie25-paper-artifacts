"""boxy's own accelerator detection (accel.py).

The point of owning this is that it becomes testable: the whole matrix runs
from fixtures instead of only on the metal it describes. A synthetic KFD
topology stands in for an MI300A node, and the vendor CLIs are injected.
"""

import pytest

from boxy import accel


# ---- AMD / ROCm via a synthetic KFD topology ----------------------------------------


def _kfd_node(root, idx, *, gfx=90402, banks=((1, 128 * 1024**3),)):
    """Write one KFD topology node. `banks` is (heap_type, size_in_bytes) pairs.
    heap_type 1/2 are framebuffer (VRAM); 0 is system memory and must not count."""
    node = root / f"{idx}"
    node.mkdir(parents=True)
    (node / "properties").write_text(
        f"gfx_target_version {gfx}\nmem_banks_count {len(banks)}\nsimd_count 304\n")
    for b, (heap, size) in enumerate(banks):
        bank = node / "mem_banks" / f"{b}"
        bank.mkdir(parents=True)
        (bank / "properties").write_text(f"heap_type {heap}\nsize_in_bytes {size}\n")
    return node


def test_rocm_detects_mi300a_class_node(tmp_path, monkeypatch):
    monkeypatch.setattr(accel, "is_arm", lambda: False)
    for i in range(4):                       # a 4-GPU node
        _kfd_node(tmp_path, i)
    info = accel.check_rocm(topology=str(tmp_path / "*"))
    assert info is not None
    assert info.kind == "rocm"               # boxy's spelling, not the runtime's "hip"
    assert info.count == 4
    assert info.vram_gb == 128
    assert "4 AMD GPU(s)" in info.detail


def test_rocm_counts_only_framebuffer_memory(tmp_path, monkeypatch):
    """System memory (heap_type 0) sits in the same table and must not be
    counted as VRAM — otherwise every CPU node looks like a big GPU."""
    monkeypatch.setattr(accel, "is_arm", lambda: False)
    _kfd_node(tmp_path, 0, banks=((1, 64 * 1024**3), (0, 512 * 1024**3)))
    info = accel.check_rocm(topology=str(tmp_path / "*"))
    assert info.vram_gb == 64                # not 576


def test_rocm_skips_cpu_nodes_and_ancient_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(accel, "is_arm", lambda: False)
    _kfd_node(tmp_path, 0, gfx=0)                       # CPU node
    _kfd_node(tmp_path, 1, gfx=80300)                   # pre-gfx900: not ROCm-capable
    assert accel.check_rocm(topology=str(tmp_path / "*")) is None


def test_rocm_ignores_display_adapters(tmp_path, monkeypatch):
    """Below MIN_VRAM_BYTES is a console adapter, not something to serve on."""
    monkeypatch.setattr(accel, "is_arm", lambda: False)
    _kfd_node(tmp_path, 0, banks=((1, 256 * 1024**2),))
    assert accel.check_rocm(topology=str(tmp_path / "*")) is None


def test_rocm_declines_on_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(accel, "is_arm", lambda: True)
    _kfd_node(tmp_path, 0)
    assert accel.check_rocm(topology=str(tmp_path / "*")) is None


def test_kfd_props_survive_malformed_files(tmp_path):
    """Odd kernels have produced unparseable property files; a probe that
    tracebacks would take the whole CLI down with it."""
    (tmp_path / "properties").write_text("gfx_target_version notanumber\n")
    assert accel.parse_kfd_props(str(tmp_path / "properties")) == {}
    assert accel.parse_kfd_props("/nonexistent/properties") == {}


# ---- NVIDIA -------------------------------------------------------------------------


def test_nvidia_reports_count_and_vram():
    out = "0, 81920\n1, 81920\n2, 81920\n3, 81920\n"
    info = accel.check_nvidia(run=lambda cmd: out)
    assert info.kind == "cuda" and info.count == 4 and info.vram_gb == 80
    assert info.visible_devices == "0,1,2,3"


def test_nvidia_absent_when_smi_missing_or_empty():
    assert accel.check_nvidia(run=lambda cmd: None) is None
    assert accel.check_nvidia(run=lambda cmd: "\n") is None


def test_nvidia_survives_smi_without_memory_column():
    """Older nvidia-smi builds answer a narrower query; the GPU is still real."""
    info = accel.check_nvidia(run=lambda cmd: "0\n1\n")
    assert info.kind == "cuda" and info.count == 2 and info.vram_gb == 0


# ---- ordering, degradation, and the no-side-effects promise -------------------------


def test_detect_returns_none_kind_when_nothing_present():
    assert accel.detect(checks=()).kind == "none"


def test_detect_takes_the_first_match_in_order():
    hit = accel.AccelInfo("cuda", "0", "fixture")
    assert accel.detect(checks=(lambda: None, lambda: hit, lambda: accel.AccelInfo("rocm"))).kind == "cuda"


def test_a_broken_probe_cannot_break_detection():
    """A vendor tool that raises must not take the CLI down — the next probe
    still runs and 'none' is a valid answer."""
    def boom():
        raise RuntimeError("vendor tool exploded")

    assert accel.detect(checks=(boom,)).kind == "none"
    assert accel.detect(checks=(boom, lambda: accel.AccelInfo("rocm", "0", "x"))).kind == "rocm"


def test_detection_never_mutates_the_environment(tmp_path, monkeypatch):
    """The deliberate divergence from prior art: probing must not export
    CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES behind the caller's back. boxy
    probes on one host and runs on another; a self-modifying probe is a trap."""
    for var in accel.VISIBLE_DEVICE_VARS.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(accel, "is_arm", lambda: False)
    _kfd_node(tmp_path, 0)
    accel.check_rocm(topology=str(tmp_path / "*"))
    accel.check_nvidia(run=lambda cmd: "0, 81920\n")
    accel.detect(checks=(lambda: accel.AccelInfo("cuda", "0", "x"),))
    leaked = [v for v in accel.VISIBLE_DEVICE_VARS.values() if v in __import__("os").environ]
    assert not leaked, f"detection exported {leaked}"


def test_accel_env_vars_reports_but_does_not_invent(monkeypatch):
    for var in accel.VISIBLE_DEVICE_VARS.values():
        monkeypatch.delenv(var, raising=False)
    assert accel.accel_env_vars() == {}
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    assert accel.accel_env_vars() == {"HIP_VISIBLE_DEVICES": "2"}


# ---- the seam the rest of boxy uses --------------------------------------------------


def test_shim_delegates_to_boxy_detection(monkeypatch):
    """ramalama_shim.detect_accel() is the seam every caller uses; it must now
    resolve through accel.py and NOT through the optional package."""
    from boxy import ramalama_shim
    monkeypatch.setattr(accel, "detect_kind", lambda: "rocm")
    assert ramalama_shim._ramalama_accel() == "rocm"


@pytest.mark.parametrize("engine,acc,expected", [
    ("vllm", "cuda", "vllm/vllm-openai:latest"),
    ("vllm", "rocm", "rocm/vllm:latest"),
    ("llama.cpp", "cuda", "ghcr.io/ggml-org/llama.cpp:server-cuda"),
    ("llama.cpp", "rocm", "quay.io/ramalama/rocm:latest"),
    ("llama.cpp", "none", "ghcr.io/ggml-org/llama.cpp:server"),
])
def test_default_image_is_deterministic(engine, acc, expected):
    """The image must not depend on whether an optional package is installed.
    It used to: RamaLama's plugin map answered when importable and a static map
    otherwise, so rocm resolved to vllm-openai-rocm on one machine and
    rocm/vllm on another — different builds, different performance, nothing in
    the command line to show which you got."""
    from boxy import ramalama_shim
    assert ramalama_shim.default_image(engine, acc) == expected
