"""Phase 5: cloud delegation — boxy invokes SkyPilot directly.

`boxy launch` transpiles box+location to a SkyPilot task (sky_export) and
hands it to the `sky` CLI: `sky launch` for batch, `sky serve up` for managed
serving (SkyServe replicas + readiness probe). SkyPilot stays an OPTIONAL
dependency invoked as a subprocess — its API-server architecture never
touches boxy's air-gapped HPC path (SPEC §6c).
"""

from __future__ import annotations

import os
import shutil
import tempfile

from boxy import sky_export
from boxy.box import Box
from boxy.location import Location


def sky_available() -> bool:
    return shutil.which("sky") is not None


def write_task_yaml(box: Box, location: Location, port: int | None, serve: bool, output: str | None = None,
                    proxy: str | None = None, ca_bundle: str | None = None) -> str:
    """Write the generated SkyPilot task YAML; returns its path. `proxy`/`ca_bundle`
    carry the corporate network env onto the task (see sky_export._network_env)."""
    yaml_text = sky_export.to_sky_task(box, location, port=port, serve=serve,
                                       proxy=proxy, ca_bundle=ca_bundle)
    if output is None:
        fd, output = tempfile.mkstemp(prefix=f"boxy-{box.name}-", suffix=".sky.yaml")
        with os.fdopen(fd, "w") as f:
            f.write(yaml_text)
    else:
        with open(output, "w") as f:
            f.write(yaml_text)
    return output


def launch_command(box: Box, yaml_path: str, serve: bool, down: bool = False) -> list[str]:
    """The `sky` argv boxy delegates to.

    serve=False: one cluster named after the box (`sky launch -c <name>`).
    serve=True:  managed service (`sky serve up -n <name>`).
    down=True:   teardown instead of launch.
    """
    if down:
        if serve:
            return ["sky", "serve", "down", box.name, "--yes"]
        return ["sky", "down", box.name, "--yes"]
    if serve:
        return ["sky", "serve", "up", "-n", box.name, yaml_path, "--yes"]
    return ["sky", "launch", "-c", box.name, yaml_path, "--yes"]


def ensure_sky() -> None:
    if not sky_available():
        raise RuntimeError(
            "the cloud path requires the SkyPilot CLI (pip install 'boxy-hpc[cloud]' "
            "or pip install skypilot); the HPC path (`boxy serve`) does not need it"
        )
