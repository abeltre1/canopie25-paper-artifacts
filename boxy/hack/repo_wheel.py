#!/usr/bin/env python3
"""Keep ONE installable boxy wheel committed at wheels/, and prove it matches src/.

Why a wheel in the repo at all: on a site whose package index terminates TLS with
a private CA, `pip install boxy-hpc` and even `pip install ./boxy` (which needs a
build backend) both fail before they start. A committed wheel needs no index, no
network and no build backend — `git pull` then

    uv pip install --no-deps wheels/boxy_hpc-<version>-py3-none-any.whl

is the whole install. It sits beside the dependency wheels already tracked there.

Why this script exists: a committed binary silently goes stale, and a stale wheel
is WORSE than no wheel — someone installs old code and has no way to notice. So
CI runs `check`, which rebuilds from src/ and compares the package payload. Drift
fails the build with the command that fixes it.

    python3 hack/repo_wheel.py build    # refresh the committed wheel
    python3 hack/repo_wheel.py check    # CI: assert it matches src/
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BOXY = Path(__file__).resolve().parent.parent
WHEELS = BOXY.parent / "wheels"
GLOB = "boxy_hpc-*.whl"

# Compared for equality. The wheel's own metadata is NOT compared: METADATA and
# WHEEL carry the generating tool's version, and RECORD carries hashes of the
# rest, so all three differ between build hosts without the code differing. The
# payload under boxy/ plus the entry points are what actually get installed.
def _payload(whl: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with zipfile.ZipFile(whl) as z:
        for name in sorted(z.namelist()):
            if name.endswith("/"):
                continue
            if ".dist-info/" in name and not name.endswith("entry_points.txt"):
                continue
            out[name] = hashlib.sha256(z.read(name)).hexdigest()
    return out


def _version() -> str:
    sys.path.insert(0, str(BOXY / "src"))
    import boxy

    return boxy.__version__


def _build(dest: Path) -> Path:
    subprocess.run([sys.executable, "-m", "build", "--wheel", "-o", str(dest), str(BOXY)],
                   check=True, stdout=subprocess.DEVNULL)
    built = sorted(dest.glob(GLOB))
    if len(built) != 1:
        raise SystemExit(f"expected exactly one wheel in {dest}, got {[p.name for p in built]}")
    return built[0]


def _committed() -> list[Path]:
    return sorted(WHEELS.glob(GLOB))


def build() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        fresh = _build(Path(tmp))
        WHEELS.mkdir(parents=True, exist_ok=True)
        # Remove every older boxy wheel first: a version bump changes the
        # filename, so without this the repo accumulates wheels and `install
        # wheels/boxy_hpc-*.whl` becomes ambiguous.
        for old in _committed():
            if old.name != fresh.name:
                old.unlink()
                print(f"removed stale {old.name}")
        shutil.copy2(fresh, WHEELS / fresh.name)
    print(f"wrote wheels/{fresh.name}")
    return 0


def check() -> int:
    have = _committed()
    if len(have) != 1:
        print(f"FAIL: expected exactly one {GLOB} in wheels/, found {[p.name for p in have]}\n"
              f"  fix: python3 boxy/hack/repo_wheel.py build", file=sys.stderr)
        return 1
    committed = have[0]

    expected_name = f"boxy_hpc-{_version()}-py3-none-any.whl"
    if committed.name != expected_name:
        print(f"FAIL: wheels/{committed.name} does not match boxy.__version__ "
              f"(expected {expected_name})\n"
              f"  fix: python3 boxy/hack/repo_wheel.py build", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        fresh = _build(Path(tmp))
        a, b = _payload(committed), _payload(fresh)

    if a == b:
        print(f"ok: wheels/{committed.name} matches src/ ({len(a)} files)")
        return 0

    only_committed = sorted(set(a) - set(b))
    only_fresh = sorted(set(b) - set(a))
    changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    print(f"FAIL: wheels/{committed.name} is STALE — it does not match src/", file=sys.stderr)
    for label, items in (("only in the committed wheel", only_committed),
                         ("missing from the committed wheel", only_fresh),
                         ("content differs", changed)):
        for item in items[:10]:
            print(f"  {label}: {item}", file=sys.stderr)
        if len(items) > 10:
            print(f"  ... and {len(items) - 10} more {label}", file=sys.stderr)
    print("  fix: python3 boxy/hack/repo_wheel.py build && git add wheels/", file=sys.stderr)
    return 1


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    if action not in ("build", "check"):
        raise SystemExit(f"usage: {Path(__file__).name} [build|check]")
    raise SystemExit(build() if action == "build" else check())
