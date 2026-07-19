"""Cross-platform / cross-site hardening: SELinux relabel, GPU device
pass-through for intel/vulkan/asahi, the metal accelerator, the Windows guard,
portable primary-IP discovery, and registry-derived scheduler/runtime validation."""

import pytest

from boxy.backends.apptainer import ApptainerBackend
from boxy.backends.base import relabel_option, warn_cpu_only
from boxy.backends.podman import PodmanBackend
from boxy.box import Box
from boxy.location import Location


# ---- SELinux relabel (pure; caller supplies `enforcing`) ------------------------


@pytest.mark.parametrize("options,mode,enforcing,expected", [
    ("ro", "auto", True, "ro,z"),      # enforcing + auto -> add z
    ("ro", "auto", False, "ro"),       # not enforcing + auto -> leave alone
    ("ro", "always", False, "ro,z"),   # always -> add even when not enforcing
    ("ro", "never", True, "ro"),       # never -> never add
    ("ro,z", "always", True, "ro,z"),  # user already set z -> untouched
    ("ro,Z", "auto", True, "ro,Z"),    # user set private-label Z -> untouched
    ("", "always", True, "z"),         # empty options -> just z
])
def test_relabel_option_matrix(options, mode, enforcing, expected):
    assert relabel_option(options, mode, enforcing) == expected


def test_podman_build_command_adds_relabel_when_forced(monkeypatch):
    monkeypatch.setenv("BOXY_SELINUX_RELABEL", "always")
    box = Box(name="b", image="img", engine="llama.cpp", model="m")
    cmd = PodmanBackend().build_command(box, Location(name="l"), ["", "-m", "/m"],
                                        {}, [("/data", "/models", "ro")], "none")
    assert "--volume=/data:/models:ro,z" in cmd


def test_podman_build_command_no_relabel_when_never(monkeypatch):
    monkeypatch.setenv("BOXY_SELINUX_RELABEL", "never")
    box = Box(name="b", image="img", engine="llama.cpp", model="m")
    cmd = PodmanBackend().build_command(box, Location(name="l"), ["", "-m", "/m"],
                                        {}, [("/data", "/models", "ro")], "none")
    assert "--volume=/data:/models:ro" in cmd


# ---- GPU device pass-through (the silent-CPU fix) -------------------------------


@pytest.mark.parametrize("accel", ["intel", "vulkan", "asahi"])
def test_dri_accelerators_get_device_podman(accel):
    assert PodmanBackend().gpu_args(accel) == ["--device", "/dev/dri"]


@pytest.mark.parametrize("accel", ["intel", "vulkan", "asahi"])
def test_dri_accelerators_get_bind_apptainer(accel):
    assert ApptainerBackend().gpu_args(accel) == ["--bind", "/dev/dri"]


def test_cuda_rocm_unchanged():
    assert PodmanBackend().gpu_args("cuda") == ["--device", "nvidia.com/gpu=all"]
    assert PodmanBackend().gpu_args("rocm")[0] == "--group-add=video"
    assert ApptainerBackend().gpu_args("cuda") == ["--nv"]


def test_unmapped_accelerator_warns_cpu_only(capsys):
    # musa/ascend/metal are valid location values with no device mapping: warn,
    # don't silently run CPU and burn the allocation.
    assert PodmanBackend().gpu_args("musa") == []
    assert "CPU-only" in capsys.readouterr().err


def test_none_accelerator_is_silent(capsys):
    assert PodmanBackend().gpu_args("none") == []
    assert capsys.readouterr().err == ""


def test_warn_cpu_only_names_backend(capsys):
    warn_cpu_only("ascend", "podman")
    err = capsys.readouterr().err
    assert "ascend" in err and "podman" in err


# ---- metal accelerator + registry-derived validation ---------------------------


def test_metal_accelerator_accepted():
    Location(name="mac", accelerator="metal")  # must not raise


def test_unknown_scheduler_rejected_with_registry_list():
    with pytest.raises(ValueError, match="unknown scheduler 'pbs'"):
        Location(name="x", scheduler="pbs")


def test_unknown_runtime_rejected_with_registry_list():
    with pytest.raises(ValueError, match="unknown runtime 'containerd'"):
        Location(name="x", runtime="containerd")


def test_known_scheduler_and_runtime_still_accepted():
    Location(name="ok", scheduler="slurm", runtime="apptainer")  # must not raise


# ---- portable primary IP (replaces GNU-only `hostname -I`) ----------------------


def test_primary_ip_returns_an_address():
    from boxy import distributed

    ip = distributed._primary_ip()
    assert isinstance(ip, str) and ip.count(".") == 3


def test_primary_ip_falls_back_to_loopback(monkeypatch):
    from boxy import distributed

    class _Boom:
        def __enter__(self): raise OSError("no socket")
        def __exit__(self, *a): return False

    monkeypatch.setattr(distributed.socket, "socket", lambda *a, **k: _Boom())
    monkeypatch.setattr(distributed.socket, "gethostbyname",
                        lambda *_: (_ for _ in ()).throw(OSError()))
    assert distributed._primary_ip() == "127.0.0.1"


# ---- Windows guard --------------------------------------------------------------


def test_windows_is_a_clean_error(monkeypatch, capsys):
    from boxy import cli

    monkeypatch.setattr(cli.sys, "platform", "win32")
    rc = cli.main(["info"])
    assert rc == 2
    assert "Windows is not supported" in capsys.readouterr().err
