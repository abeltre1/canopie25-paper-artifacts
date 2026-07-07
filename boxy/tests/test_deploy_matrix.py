"""Deployment matrix: the SAME smallest-Llama box/model serves on every platform —
only the location/scheduler changes (boxy's box+location design). Golden dryrun
shapes so the four targets can't silently regress.

Model: Llama 3.2 1B (smallest in the family) as a Q4 GGUF -> llama.cpp, portable
across CPU desktop and GPU nodes.
"""

from boxy.cli import main
from tests.conftest import EXAMPLES

BOX = str(EXAMPLES / "boxes" / "llama-3.2-1b.toml")
MODEL = "hf://hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF/llama-3.2-1b-instruct-q4_k_m.gguf"


def test_deploy_local_desktop_cpu(capsys):
    # 1) local / baremetal: no scheduler -> a container right here (CPU llama.cpp).
    rc = main(["serve", "--box", BOX,
               "--location", str(EXAMPLES / "locations" / "local-docker.toml"), "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Running Command:" in out
    assert "ghcr.io/ggml-org/llama.cpp:server" in out       # llama.cpp engine, CPU image
    assert "-m /mnt/models/model" in out and "--port 8090" in out
    assert "srun" not in out and "sbatch" not in out         # nothing scheduler-y locally


def test_deploy_slurm(capsys):
    # 2) Slurm: submit a batch job that re-runs boxy on the compute node.
    rc = main(["serve", MODEL, "--scheduler", "slurm", "--gpus", "1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "#SBATCH --nodes=1" in out and "#SBATCH --gpus-per-node=1" in out
    assert "sbatch --parsable" in out
    assert "--foreground --here" in out                      # compute-node re-invocation


def test_deploy_flux(capsys):
    # 3) Flux: identical UX, flux's own directive spelling + flux batch.
    rc = main(["serve", MODEL, "--scheduler", "flux", "--gpus", "1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# flux: -N1" in out and "# flux: -g1" in out
    assert "flux batch" in out


def test_deploy_any_other_platform_cloud(capsys):
    # 4) any other platform: transpile to a SkyPilot task (cloud/K8s), or write a
    # --location <site>.toml for an on-prem site. Here: the cloud/Sky path.
    rc = main(["generate", "sky", "--box", BOX,
               "--location", str(EXAMPLES / "locations" / "cloud-gpu.toml")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "resources:" in out and "run:" in out             # a SkyPilot task YAML
    assert "llama-server" in out                             # the llama.cpp serve command


def test_same_box_every_location_is_portable(capsys):
    # The modular promise: ONE box file dry-runs cleanly against every shipped
    # location (CPU + GPU, podman/docker/apptainer, none/slurm/flux).
    for loc in sorted((EXAMPLES / "locations").glob("*.toml")):
        rc = main(["serve", "--box", BOX, "--location", str(loc), "--dryrun"])
        out = capsys.readouterr().out
        assert rc == 0, f"{loc.name} failed"
        assert "### Running Command:" in out or "### Head" in out, f"{loc.name}: no command"
