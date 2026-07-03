"""--replicas K data-parallel fan-out with GPU bin-packing.

By default replicas BIN-PACK onto a node's GPUs: rpn = --gpus // --gpus-per-replica
per node, each replica pinned to its own GPU(s) on its own port, so K replicas take
ceil(K/rpn) node jobs — not one whole node each. With --nodes>1 each replica is a
multi-node distributed instance. K==1 is the single-submission path.
"""

from boxy.cli import main


def test_replicas_bin_pack_onto_one_node(capsys):
    # 4 replicas x 1 GPU on a 4-GPU node = ONE job, not four nodes.
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--gpus", "4", "--replicas", "4", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "packed 4/node -> 1 node job(s)" in out
    assert out.count("### Node job") == 1
    assert out.count("#SBATCH --nodes=1") == 1
    assert "#SBATCH --gpus-per-node=4" in out
    # four co-located, GPU-pinned servers on distinct ports, backgrounded + waited
    for i in range(4):
        assert f"--name boxy-meta-llama-3.1-8b-r{i}" in out
        assert f"--visible-gpus {i}" in out
        assert f"--port {8000 + i}" in out
    assert out.count(" &\n") >= 4 or out.count(" &") >= 4
    assert "\n    wait" in out or out.rstrip().endswith("wait") or "    wait\n" in out


def test_replicas_gpus_per_replica_sets_tensor_parallel(capsys):
    # 2 replicas x 2 GPUs on a 4-GPU node: each pinned to a GPU pair, TP=2.
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm", "--gpus", "4",
               "--replicas", "2", "--gpus-per-replica", "2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "packed 2/node -> 1 node job(s)" in out
    assert "--visible-gpus 0,1" in out and "--visible-gpus 2,3" in out
    assert out.count("--tensor-parallel-size 2") == 2


def test_replicas_overflow_to_multiple_node_jobs(capsys):
    # 6 replicas, 4 per node -> 2 node jobs (4 + 2).
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--gpus", "4", "--replicas", "6", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "packed 4/node -> 2 node job(s)" in out
    assert "--job-name=boxy-meta-llama-3.1-8b-n0" in out
    assert "--job-name=boxy-meta-llama-3.1-8b-n1" in out
    assert "#SBATCH --gpus-per-node=4" in out and "#SBATCH --gpus-per-node=2" in out
    for i in range(6):
        assert f"--name boxy-meta-llama-3.1-8b-r{i}" in out


def test_replicas_guard_gpus_per_replica_exceeds_budget(capsys):
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm", "--gpus", "2",
               "--replicas", "2", "--gpus-per-replica", "4", "--dryrun"])
    assert rc == 2
    assert "gpus-per-replica" in capsys.readouterr().err


def test_replicas_compose_with_distributed(capsys):
    # --nodes>1: each replica is itself a 2-node distributed instance (one Ray task
    # per node), so it stays one distributed job per replica.
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--nodes", "2", "--gpus", "4", "--replicas", "2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("### Replica ") == 2
    assert out.count("#SBATCH --ntasks-per-node=1") == 2
    assert out.count("#SBATCH --nodes=2") == 2
    assert out.count("--nodes 2 --gpus 4") == 2


def test_replicas_one_is_single_submission_path(capsys):
    # K==1 must not enter the fan-out (no packing header, no -r0 suffix)
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "slurm",
               "--gpus", "4", "--replicas", "1", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "### Node job" not in out and "### Replica " not in out
    assert "-r0" not in out
    assert "### Batch script" in out and "### Submit Command:" in out


def test_replicas_without_scheduler_is_guarded(capsys):
    rc = main(["serve", "Meta-Llama-3.1-8B", "--replicas", "3", "--dryrun"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--replicas" in err
    assert "--scheduler slurm|flux" in err
    assert "--unique" in err  # points at the local workaround


def test_list_and_stop_group_record(capsys, tmp_path, monkeypatch):
    import boxy.cli as climod
    from boxy import jobs
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    jobs.write_record("grp", {"name": "grp", "scheduler": "slurm", "job": "555", "model": "m",
                              "replicas": ["grp-r0", "grp-r1"]})
    jobs.write_endpoint("grp-r0", 8000)
    jobs.write_endpoint("grp-r1", 8001)
    # list shows the one job and expands its replica endpoints
    main(["list", "--runtime", "docker", "--dryrun"])
    out = capsys.readouterr().out
    assert "grp  slurm job 555" in out and "(2 replicas)" in out
    assert "grp-r0" in out and "grp-r1" in out
    # stop cancels the single job and cleans the replica endpoint files (stub the
    # scancel exec — the scheduler binary isn't present in the test env)
    monkeypatch.setattr(climod, "_run_or_print", lambda cmd, dryrun: 0)
    assert main(["stop", "grp"]) == 0
    assert not (tmp_path / "grp-r0.endpoint.json").exists()
    assert not (tmp_path / "grp-r1.endpoint.json").exists()
    assert jobs.read_record("grp") is None


def test_replicas_flux_bin_packs(capsys):
    rc = main(["serve", "Meta-Llama-3.1-8B", "--scheduler", "flux",
               "--gpus", "4", "--replicas", "2", "--dryrun"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "packed 4/node -> 1 node job(s)" in out
    assert "# flux: --job-name=boxy-meta-llama-3.1-8b" in out  # ONE flux job
    assert "flux batch" in out
    for i in range(2):
        assert f"--name boxy-meta-llama-3.1-8b-r{i}" in out
        assert f"--visible-gpus {i}" in out
