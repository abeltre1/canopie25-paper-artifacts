"""Every shipped example box must dry-run cleanly against every shipped
example location (the full engine x runtime x scheduler matrix that
examples/MATRIX.md documents). Adding a new example automatically extends
this test."""

import pytest

from boxy.cli import main
from tests.conftest import EXAMPLES

LOCATIONS = sorted(p.name for p in (EXAMPLES / "locations").glob("*.toml"))
BOXES = ["qwen-gguf.toml", "vllm.toml", "vllm-hf.toml", "llamacpp-demo.toml"]


@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("box", BOXES)
def test_matrix_serve_dryrun(box, location, capsys):
    rc = main([
        "serve",
        "--box", str(EXAMPLES / "boxes" / box),
        "--location", str(EXAMPLES / "locations" / location),
        "--dryrun",
    ])
    out = capsys.readouterr().out
    assert rc == 0, f"{box} x {location} failed"
    assert "### Running Command:" in out


def test_matrix_doc_exists_and_covers_all_combos():
    text = (EXAMPLES / "MATRIX.md").read_text()
    for engine in ("llama.cpp", "vllm"):
        for runtime in ("podman", "docker", "apptainer"):
            for sched in ("none", "slurm", "flux"):
                assert f"{engine} + {runtime} + {sched}" in text, \
                    f"MATRIX.md missing combo {engine}+{runtime}+{sched}"
