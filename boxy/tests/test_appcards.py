"""App cards (src/boxy/appcards.py + `boxy app`): the deployment-OS namespace
for classic HPC applications/benchmarks. Card loading/precedence, the rendered
batch script (spack bootstrap, launcher geometry, container+proxy), and the CLI
list/dryrun/error paths — all against packaged cards or tmp user cards, never a
real spack or scheduler."""

import pytest

from boxy import appcards
from boxy.cli import main


@pytest.fixture
def user_cards(tmp_path, monkeypatch):
    d = tmp_path / "cfg" / "boxy" / "cards" / "apps"
    d.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return d


# ---- loading + precedence -----------------------------------------------------------


def test_packaged_cards_load_and_validate():
    cards = appcards.load_cards()
    names = {c.name for c in cards}
    assert {"osu-benchmarks", "stream", "miniem"} <= names
    osu = appcards.find_card("osu-benchmarks")
    assert osu.kind == "spack" and osu.spec == "osu-micro-benchmarks"
    assert osu.nodes == 2 and osu.tasks_per_node == 1
    assert osu.source == "packaged"


def test_user_card_wins_over_packaged(user_cards):
    (user_cards / "osu-benchmarks.toml").write_text(
        '[app]\nname = "osu-benchmarks"\nkind = "spack"\nspec = "osu-micro-benchmarks@7.4"\n'
        'run = ["osu_bw"]\n')
    card = appcards.find_card("osu-benchmarks")
    assert card.source == "user" and card.spec == "osu-micro-benchmarks@7.4"


def test_malformed_user_card_raises_with_its_path(user_cards):
    (user_cards / "broken.toml").write_text("[app]\nkind = 'spack'\n")   # no name
    with pytest.raises(ValueError) as e:
        appcards.load_cards()
    assert "broken.toml" in str(e.value) and "name" in str(e.value)


def test_unknown_kind_rejected(user_cards):
    (user_cards / "weird.toml").write_text(
        '[app]\nname = "weird"\nkind = "cmake"\nrun = ["x"]\n')
    with pytest.raises(ValueError) as e:
        appcards.load_cards()
    assert "kind" in str(e.value)


# ---- script rendering ---------------------------------------------------------------


def test_spack_script_slurm_geometry_and_bootstrap():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x-%j.log", ["--account=fy1"])
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --account=fy1" in text
    assert "spack install --reuse -y osu-micro-benchmarks" in text
    assert "spack load osu-micro-benchmarks" in text
    assert "srun -N 2 -n 2 " in text                     # geometry rides the launcher
    assert "setup-env.sh" in text                        # spack probed on the node


def test_flux_spelling_and_flag_overrides():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "flux", "app-osu", "/tmp/x.log", [],
                                      nodes=4, tasks_per_node=2)
    assert "flux run -N 4 -n 8 " in text                 # flags override the card
    assert "srun" not in text


def test_container_card_renders_podman_with_proxy(user_cards):
    (user_cards / "hello.toml").write_text(
        '[app]\nname = "hello"\nkind = "container"\nimage = "quay.io/podman/hello:latest"\n'
        'run = ["echo hi"]\nnodes = 1\n')
    card = appcards.find_card("hello")
    text = appcards.render_app_script(card, "slurm", "app-hello", "/tmp/x.log", [],
                                      proxy_prefix="env https_proxy=http://p:80 ")
    assert "podman run --rm quay.io/podman/hello:latest echo hi" in text
    assert "env https_proxy=http://p:80 srun -N 1 -n 1 " in text
    assert "spack" not in text


def test_modules_and_setup_lines_precede_run():
    card = appcards.AppCard(name="x", card_name="x", source="user", kind="spack",
                            spec="stream", modules=["gcc/12"], setup=["export OMP_NUM_THREADS=16"],
                            run=["stream_c.exe"])
    text = appcards.render_app_script(card, "slurm", "app-x", "/tmp/x.log", [])
    body = text[text.index("set -e"):]
    assert body.index("module load gcc/12") < body.index("spack install")
    assert body.index("export OMP_NUM_THREADS=16") < body.index("stream_c.exe")


# ---- the CLI ------------------------------------------------------------------------


def test_cli_bare_app_lists_cards(capfd):
    rc = main(["app"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "osu-benchmarks" in out and "stream" in out and "miniem" in out


def test_cli_unknown_card_is_a_clear_error(capfd):
    rc = main(["app", "does-not-exist"])
    err = capfd.readouterr().err
    assert rc == 2
    assert "no app card named" in err and "osu-benchmarks" in err


def test_cli_dryrun_prints_script_without_scheduler(capfd, monkeypatch, tmp_path):
    # --dryrun renders even on a machine with NO scheduler (the laptop case).
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["app", "stream", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --job-name=app-stream" in out
    assert "spack install --reuse -y stream" in out
    assert "srun -N 1 -n 1 stream_c.exe" in out


def test_cli_cards_listing_includes_apps(capfd):
    rc = main(["cards"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "app cards" in out and "osu-benchmarks" in out
