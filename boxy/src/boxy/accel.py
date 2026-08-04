"""Accelerator detection — boxy's own, with no library dependency.

This is a headline capability: `boxy serve MODEL` picking the right engine and
image for the hardware is most of what boxy does, so boxy implements it rather
than delegating it. Delegation made the feature conditional on an optional
package AND made the result vary with whether that package happened to be
installed — unacceptable in a tool used to compare accelerators.

Implemented directly against the public interfaces: the KFD sysfs topology
(documented in linux/kfd_sysfs.h), `nvidia-smi --query-gpu`, PCI device ids,
and `/dev/{dri,kfd,accel}`. Those are kernel and vendor ABIs, stable on a
scale of years.

Design notes — why this looks the way it does:

  * detection NEVER mutates os.environ. Setting CUDA_VISIBLE_DEVICES as a side
    effect of asking a question is a trap when you probe on a login node,
    submit elsewhere, and report for a third host. Selected devices are
    returned instead.
  * NVIDIA detection does not require a CDI configuration. boxy frequently
    detects on a machine that will never run the workload, so absent CDI says
    nothing about the compute nodes; container wiring is the backend's job.
  * it reports VRAM and device COUNT, not just a name. boxy's geometry solver
    needs those numbers and otherwise has to infer them from a GRES-token
    lookup table or a hand-written system card.
  * every probe is injectable, so the whole matrix is testable from fixtures
    rather than only on the metal it describes.

`boxy accel` and `boxy info` report what this returns.
"""

from __future__ import annotations

import glob
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

# Heap types from /usr/include/linux/kfd_sysfs.h — framebuffer memory counts as
# VRAM; system/GDS/LDS/scratch heaps do not.
HEAP_TYPE_FB_PUBLIC = 1
HEAP_TYPE_FB_PRIVATE = 2

# Anything smaller is an integrated display adapter, not a compute GPU. 1GiB is
# comfortably below any accelerator worth serving on and comfortably above the
# framebuffer of a management/console adapter.
MIN_VRAM_BYTES = 1024 * 1024 * 1024

# Accelerator -> the env var that scopes visible devices for it. boxy sets these
# on the CONTAINER (see envs.py); detection only reports them.
VISIBLE_DEVICE_VARS = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "rocm": "HIP_VISIBLE_DEVICES",
    "intel": "INTEL_VISIBLE_DEVICES",
    "ascend": "ASCEND_VISIBLE_DEVICES",
    "asahi": "ASAHI_VISIBLE_DEVICES",
    "musa": "MUSA_VISIBLE_DEVICES",
}


@dataclass(frozen=True)
class AccelInfo:
    """What detection found. `kind` is boxy's accelerator spelling (the same
    strings location.ACCELERATORS uses), NOT the vendor runtime name: the
    runtimes call these 'hip'/'cann' while boxy says 'rocm'/'ascend'
    everywhere, so detection emits boxy's vocabulary directly."""
    kind: str = "none"
    visible_devices: str = ""      # e.g. "0" — advisory; never exported by us
    detail: str = ""               # human-readable provenance for `boxy info`
    devices: dict = field(default_factory=dict)   # /dev nodes present (see gpu_device_paths)
    count: int = 0                 # how many accelerators of this kind
    vram_gb: int = 0               # per-device memory, GiB (0 = not determined)


def _run(cmd: list[str], timeout: int = 10) -> str | None:
    """Run a probe binary, returning stdout, or None when it is absent or fails.
    Never raises: a missing vendor tool is the normal case, not an error."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.stdout if p.returncode == 0 else None


def is_arm() -> bool:
    return platform.machine() in ("arm64", "aarch64")


# ---- AMD / ROCm ---------------------------------------------------------------------

def parse_kfd_props(path: str) -> dict:
    """A KFD properties file is `key value` lines, all integers."""
    try:
        with open(path) as fh:
            return {k: int(v) for k, _, v in (ln.partition(" ") for ln in fh) if v.strip()}
    except (OSError, ValueError):
        return {}


def kfd_gpus(topology: str = "/sys/devices/virtual/kfd/kfd/topology/nodes/*"):
    """Yield (node_path, properties) for each GPU in the KFD topology, skipping
    CPU nodes (which report gfx_target_version 0)."""
    for node in sorted(glob.glob(topology)):
        props = parse_kfd_props(node + "/properties")
        if props.get("gfx_target_version", 0) == 0:
            continue
        yield node, props


def check_rocm(topology: str | None = None) -> AccelInfo | None:
    """AMD GPUs via the KFD topology. Picks the node with the most framebuffer
    memory, ignoring pre-gfx900 parts (Polaris and older are not ROCm-capable)."""
    if is_arm():
        return None            # no ROCm on arm64 — Vulkan is the path there
    kwargs = {"topology": topology} if topology else {}
    best_idx, best_bytes, found = 0, 0, 0
    for i, (node, props) in enumerate(kfd_gpus(**kwargs)):
        if props.get("gfx_target_version", 0) < 90000:
            continue
        vram = 0
        for bank in range(props.get("mem_banks_count", 0)):
            bp = parse_kfd_props(f"{node}/mem_banks/{bank}/properties")
            if bp.get("heap_type") in (HEAP_TYPE_FB_PUBLIC, HEAP_TYPE_FB_PRIVATE):
                vram += bp.get("size_in_bytes", 0)
        if vram > MIN_VRAM_BYTES:
            found += 1
            if vram > best_bytes:
                best_idx, best_bytes = i, vram
    if not best_bytes:
        return None
    gb = best_bytes // (1024 ** 3)
    return AccelInfo("rocm", str(best_idx),
                     f"{found} AMD GPU(s) via KFD topology, {gb}GB each",
                     count=found, vram_gb=gb)


# ---- NVIDIA -------------------------------------------------------------------------

def check_nvidia(run=_run) -> AccelInfo | None:
    """NVIDIA GPUs via nvidia-smi.

    Deliberately does NOT require a CDI configuration. A tool that launches the
    container itself has reason to insist on one; boxy often detects on a login
    node that will never run the workload, where missing CDI says nothing about
    the compute nodes. Container wiring is the runtime backend's job
    (podman.py), which fails loudly at run time if the toolkit is absent."""
    out = run(["nvidia-smi", "--query-gpu=index,memory.total",
               "--format=csv,noheader,nounits"])
    if not out:
        return None
    indices, mib = [], 0
    for line in (ln for ln in out.splitlines() if ln.strip()):
        parts = [f.strip() for f in line.split(",")]
        indices.append(parts[0])
        if len(parts) > 1 and parts[1].isdigit():
            mib = max(mib, int(parts[1]))
    if not indices:
        return None
    gb = mib // 1024
    detail = f"{len(indices)} NVIDIA GPU(s) via nvidia-smi" + (f", {gb}GB each" if gb else "")
    return AccelInfo("cuda", ",".join(indices), detail, count=len(indices), vram_gb=gb)


# ---- the rest -----------------------------------------------------------------------

def check_ascend(run=_run) -> AccelInfo | None:
    return AccelInfo("ascend", "0", "Ascend NPU via npu-smi") if run(["npu-smi", "info"]) else None


def check_asahi() -> AccelInfo | None:
    """Apple Silicon under Asahi Linux."""
    if os.path.exists("/proc/device-tree/compatible"):
        try:
            with open("/proc/device-tree/compatible", "rb") as fh:
                if b"apple" in fh.read():
                    return AccelInfo("asahi", "1", "Apple Silicon (Asahi)")
        except OSError:
            pass
    return None


def check_intel() -> AccelInfo | None:
    """Intel discrete/integrated GPUs via the i915/xe DRM device ids. Only Arc
    and Data Center parts are worth an accelerator image; the id prefixes below
    are the discrete families."""
    for path in sorted(glob.glob("/sys/bus/pci/drivers/i915/*/device")
                       + glob.glob("/sys/bus/pci/drivers/xe/*/device")):
        try:
            with open(path) as fh:
                dev_id = fh.read().strip().lower()
        except OSError:
            continue
        # 0x56xx = Arc (DG2/Alchemist), 0x0bd* = Data Center GPU Max (Ponte Vecchio)
        if dev_id.startswith("0x56") or dev_id.startswith("0x0bd"):
            return AccelInfo("intel", "0", f"Intel GPU (PCI id {dev_id})")
    return None


def check_musa(run=_run) -> AccelInfo | None:
    return AccelInfo("musa", "0", "Moore Threads GPU via mthreads-gmi") \
        if run(["mthreads-gmi"]) else None


def gpu_device_paths() -> dict:
    """Which GPU device nodes this host actually exposes. Reported, not wired:
    each backend derives its own flags from the accelerator KIND (podman's
    `--device /dev/kfd`, apptainer's `--bind`), because the spelling differs per
    runtime and CUDA passes through nvidia.com/gpu rather than a /dev path.

    The value of reporting it is diagnostic — `boxy info` shows it, so "boxy
    says none but I have a GPU" can be told apart from "the node isn't there at
    all" (missing driver) in one look. Note the converse case: a CI runner
    exposes /dev/dri with nothing behind it, so a node present is not a GPU."""
    return {d: f"/dev/{d}" for d in ("dri", "kfd", "accel") if os.path.exists(f"/dev/{d}")}


# Order matters: the most specific / least ambiguous probe first. Asahi leads
# because an Apple machine also exposes a DRM node that looser checks match.
_CHECKS = (check_asahi, check_nvidia, check_ascend, check_rocm, check_intel, check_musa)


def detect(checks=None) -> AccelInfo:
    """Detect the local accelerator. Returns AccelInfo(kind='none') when there
    is nothing — never raises, because 'no GPU here' is a normal answer on a
    login node and on a laptop."""
    for check in (checks or _CHECKS):
        try:
            found = check()
        except Exception:  # noqa: BLE001 — a broken vendor tool must not break boxy
            continue
        if found:
            return AccelInfo(found.kind, found.visible_devices, found.detail,
                             gpu_device_paths(), found.count, found.vram_gb)
    return AccelInfo()


def detect_kind(checks=None) -> str:
    """Just the accelerator name — the shape most call sites want."""
    return detect(checks).kind


def accel_env_vars() -> dict:
    """Visible-device vars already set in THIS environment, passed through to
    the container so an operator's scoping is honoured. Detection does not add
    to these; it only reports what the user set."""
    return {v: os.environ[v] for v in VISIBLE_DEVICE_VARS.values() if os.environ.get(v)}
