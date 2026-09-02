"""The packaged example profiles ship inside the wheel and are reachable via
`boxy examples` (importlib.resources — works from an installed package, not just
a source checkout)."""

from importlib.resources import files
from pathlib import Path

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


def test_documented_example_paths_resolve_to_the_packaged_profiles(tmp_path, monkeypatch):
    """Every doc line spells these `examples/boxes/x.toml`, but the files ship
    INSIDE the package — so the first command a new user copy-pastes died with
    'No such file or directory' from a checkout AND from an install. The
    documented spelling must work from any directory; a real file on disk still
    wins."""
    from boxy.cli import _packaged_profile

    monkeypatch.chdir(tmp_path)                      # nowhere near a checkout
    resolved = _packaged_profile("examples/boxes/vllm.toml")
    assert resolved.endswith("vllm.toml") and Path(resolved).is_file()
    assert _packaged_profile("examples/locations/local.toml").endswith("local.toml")
    # a bare filename resolves too, and an unknown name is returned untouched
    assert Path(_packaged_profile("vllm.toml")).is_file()
    assert _packaged_profile("examples/boxes/not-a-real-example.toml") == \
        "examples/boxes/not-a-real-example.toml"
    # a REAL file on disk always wins over the packaged copy
    local = tmp_path / "examples" / "boxes"
    local.mkdir(parents=True)
    (local / "vllm.toml").write_text("[box]\nname='x'\n")
    assert _packaged_profile("examples/boxes/vllm.toml") == "examples/boxes/vllm.toml"


def test_every_example_path_named_in_the_docs_actually_ships():
    """The docs referenced three location profiles that do not exist in the
    package at all (clusterA/clusterB/example.toml) — copy-paste failures that
    no amount of path resolution can fix."""
    import re
    from importlib.resources import files

    root = files("boxy.data") / "examples"
    have = {e.name for kind in ("boxes", "locations") for e in (root / kind).iterdir()}
    docs = Path(__file__).parent.parent
    missing = set()
    for f in [*(docs / "docs").glob("*.md"), docs / "README.md"]:
        for m in re.finditer(r"examples/(?:boxes|locations)/([\w.\-]+\.toml)", f.read_text()):
            if m.group(1) not in have:
                missing.add(f"{f.name}: {m.group(1)}")
    assert not missing, f"docs reference example profiles that do not ship: {sorted(missing)}"
