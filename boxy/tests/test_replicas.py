"""Phase 2: --replicas K data-parallel fan-out.

K independent instances of the model, each its own batch job named <base>-r0..r{K-1}
with its own endpoint/log. Composes with --nodes>1 (each replica is itself a
distributed instance). K==1 is byte-identical to the single-submission path.
"""

from boxy.cli import main


def test_replicas_fans_out_distinct_named_jobs(capsys):
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--gpus", "4", "--replicas", "3", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    # three replicas, each with its own job name and endpoint file
    assert out.count("### Replica ") == 3
    for i in range(3):
        assert f"-r{i}" in out
        assert f"--job-name=boxy-meta-llama-3.1-8b-r{i}" in out
        assert f"-r{i}.endpoint.json" in out
    assert "replicas: 3 independent instances" in out
    assert "nothing submitted" in out  # dryrun


def test_replicas_compose_with_distributed(capsys):
    # each replica is itself a 2-node distributed instance: the per-replica batch
    # script requests one Ray task per node.
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--nodes", "2", "--gpus", "4", "--replicas", "2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("#SBATCH --ntasks-per-node=1") == 2  # one per replica script
    assert out.count("#SBATCH --nodes=2") == 2
    # each replica's inner serve forwards the geometry so it re-derives Ray on the head
    assert out.count("--nodes 2 --gpus 4") == 2


def test_replicas_one_is_single_submission_path(capsys):
    # K==1 must not enter the fan-out (no "### Replica" header, no -r0 suffix)
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--gpus", "4", "--replicas", "1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Replica " not in out
    assert "-r0" not in out
    assert "### Batch script" in out and "### Submit Command:" in out


def test_replicas_without_scheduler_is_guarded(capsys):
    # no scheduler in play: --replicas can't fan out batch jobs; clear guidance.
    rc = main(["serve", "Meta-Llama-3.1-8B", "--replicas", "3", "--dryrun"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--replicas" in err
    assert "--scheduler slurm|flux" in err
    assert "--unique" in err  # points at the local workaround


def test_replicas_flux(capsys):
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "flux",
               "--gpus", "4", "--replicas", "2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    # flux directives use the `# flux:` sentinel; each replica its own job name
    for i in range(2):
        assert f"# flux: --job-name=boxy-meta-llama-3.1-8b-r{i}" in out
    assert "flux batch" in out  # flux submit command per replica
