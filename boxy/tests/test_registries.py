"""Pluggable image registries: --registry / [location.image_mirrors] rewrite
every image reference through ONE module (registries.py) — site mirrors, local
registries, and localhost images without touching backend/engine code."""

from boxy import registries
from boxy.cli import main
from boxy.location import Location
from tests.conftest import EXAMPLES


# ---- pure resolution ----------------------------------------------------------


def test_split_registry_conventions():
    assert registries.split_registry("vllm/vllm-openai:v1") == ("", "vllm/vllm-openai:v1")
    assert registries.split_registry("docker.io/vllm/vllm-openai") == ("docker.io", "vllm/vllm-openai")
    assert registries.split_registry("ghcr.io/ggml-org/llama.cpp:server") == ("ghcr.io", "ggml-org/llama.cpp:server")
    assert registries.split_registry("localhost/vllm-extra:latest") == ("localhost", "vllm-extra:latest")
    assert registries.split_registry("reg:5000/img") == ("reg:5000", "img")


def test_blanket_registry_replaces_or_prefixes():
    # bare name: prefixed (legacy behavior preserved)
    assert (registries.resolve_image("vllm/vllm-openai:v1", registry="registry.site.gov/")
            == "registry.site.gov/vllm/vllm-openai:v1")
    # an existing registry component is REPLACED, never concatenated
    assert (registries.resolve_image("docker.io/vllm/vllm-openai", registry="registry.site.gov")
            == "registry.site.gov/vllm/vllm-openai")
    assert (registries.resolve_image("ghcr.io/ggml-org/llama.cpp:server", registry="registry.site.gov")
            == "registry.site.gov/ggml-org/llama.cpp:server")


def test_mirrors_map_exact_wildcard_and_precedence():
    mirrors = {"docker.io": "registry.site.gov/dockerhub", "*": "registry.site.gov/mirror"}
    # bare names imply docker.io
    assert (registries.resolve_image("vllm/vllm-openai:v1", mirrors=mirrors)
            == "registry.site.gov/dockerhub/vllm/vllm-openai:v1")
    # other registries fall to the wildcard
    assert (registries.resolve_image("ghcr.io/ggml-org/llama.cpp:server", mirrors=mirrors)
            == "registry.site.gov/mirror/ggml-org/llama.cpp:server")
    # mirrors WIN over the blanket registry
    assert (registries.resolve_image("vllm/x", registry="other.gov", mirrors=mirrors)
            == "registry.site.gov/dockerhub/vllm/x")
    # no wildcard: unmatched registries (e.g. localhost) stay put
    assert (registries.resolve_image("localhost/vllm-extra", mirrors={"docker.io": "m.gov/d"})
            == "localhost/vllm-extra")
    assert registries.resolve_image("vllm/x") == "vllm/x"  # nothing configured: unchanged


# ---- integration: flag, profile, apptainer build ------------------------------


def test_serve_registry_flag_rewrites_image(capsys):
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(EXAMPLES / "locations" / "slurm-podman-cuda.toml"),
               "--no-distributed", "--registry", "registry.example.gov/mirror", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "registry.example.gov/mirror/vllm/vllm-openai:v0.24.0" in out
    assert "image registry: registry.example.gov/mirror" in out  # auto: line


def test_location_image_mirrors_from_toml(tmp_path, capsys):
    loc = tmp_path / "mirrored.toml"
    loc.write_text(
        '[location]\nname = "mirrored"\nscheduler = "none"\naccelerator = "none"\n'
        'runtime = "docker"\noffline = true\n'
        '[location.image_mirrors]\n"docker.io" = "registry.site.gov/dockerhub"\n'
    )
    parsed = Location.from_toml(loc)
    assert parsed.image_mirrors == {"docker.io": "registry.site.gov/dockerhub"}
    rc = main(["serve", "--box", str(EXAMPLES / "boxes" / "vllm.toml"),
               "--location", str(loc), "--dryrun"])
    assert rc == 0
    assert "registry.site.gov/dockerhub/vllm/vllm-openai:v0.24.0" in capsys.readouterr().out


def test_apptainer_sif_build_uses_rewritten_registry(vllm_box, clusterb):
    from dataclasses import replace

    from boxy.backends import get_backend

    loc = replace(clusterb, registry="registry.site.gov")
    prepare = get_backend("apptainer").prepare(vllm_box, loc)
    assert prepare[0][-1] == "docker://registry.site.gov/vllm/vllm-openai:v0.9.1"


def test_sbatch_flux_wrapper_hint():
    # clusterb-class systems ship an sbatch that wraps flux; --scheduler slurm
    # there fails with flux-batch usage errors — boxy must say "use flux".
    from boxy.cli import _submission_hint

    hint = _submission_hint("Unknown option: parsable\nusage: flux batch [OPTIONS...] "
                            "[SCRIPT]\nflux batch: error: unrecognized arguments: --gpus-per-node=4")
    assert "--scheduler flux" in hint and "wrapper" in hint
