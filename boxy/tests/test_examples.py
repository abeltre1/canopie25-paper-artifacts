"""The packaged example profiles ship inside the wheel and are reachable via
`boxy examples` (importlib.resources — works from an installed package, not just
a source checkout)."""

from importlib.resources import files

from boxy.cli import main


def test_examples_are_packaged_data():
    root = files("boxy.data") / "examples"
    boxes = [p.name for p in (root / "boxes").iterdir() if p.name.endswith(".toml")]
    locs = [p.name for p in (root / "locations").iterdir() if p.name.endswith(".toml")]
    assert "vllm.toml" in boxes
    assert "slurm-podman-cuda.toml" in locs
    # the site-named profiles were renamed away
    assert "clustera.toml" not in locs and "clusterb.toml" not in locs


def test_examples_list(capsys):
    rc = main(["examples"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "boxes" in out and "locations" in out and "vllm.toml" in out


def test_examples_show(capsys):
    rc = main(["examples", "show", "slurm-podman-cuda.toml"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[location]" in out and 'scheduler = "slurm"' in out


def test_examples_show_accepts_name_without_suffix(capsys):
    rc = main(["examples", "show", "vllm"])
    assert rc == 0
    assert "[box]" in capsys.readouterr().out


def test_examples_show_unknown_errors(capsys):
    rc = main(["examples", "show", "nope.toml"])
    assert rc == 1
    assert "no example named" in capsys.readouterr().err


def test_examples_export(tmp_path, capsys):
    rc = main(["examples", "export", str(tmp_path / "ex")])
    assert rc == 0
    assert (tmp_path / "ex" / "boxes" / "vllm.toml").exists()
    assert (tmp_path / "ex" / "locations" / "slurm-podman-cuda.toml").exists()
