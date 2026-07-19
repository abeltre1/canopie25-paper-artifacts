from boxy.schedulers import get_scheduler


def test_none_is_passthrough(clusterb):
    assert get_scheduler("none").wrap(["echo", "hi"], clusterb) == ["echo", "hi"]


def test_slurm_srun_prefix(clustera):
    cmd = get_scheduler("slurm").wrap(["podman", "run"], clustera)
    assert cmd == ["srun", "--nodes=2", "--gpus-per-node=4", "podman", "run"]


def test_slurm_parse_job_id_ignores_sbatch_info_noise():
    s = get_scheduler("slurm")
    # field: clustera merges sbatch's stderr INFO line into the captured output, and it
    # can land AFTER the job id — the id must still be extracted, not the INFO line.
    noisy = ("1819345\nsbatch: INFO: Adding filesystem licenses to job: "
             "gpfs:1,sitescratch:1,pscratch:1,rlfs01:1,rnfs01:1")
    assert s.parse_job_id(noisy) == "1819345"
    assert s.parse_job_id("sbatch: INFO: Adding filesystem licenses\n1819345") == "1819345"
    assert s.parse_job_id("1818768") == "1818768"
    assert s.parse_job_id("1818768;clusterA") == "1818768"
    assert s.parse_job_id("") == ""


def test_slurm_clears_xdg(clustera):
    # Prototype check_podman: unset XDG_SESSION_ID / XDG_RUNTIME_DIR in jobs.
    assert get_scheduler("slurm").host_env_fixups() == ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]


def test_flux_run_prefix(clusterb):
    cmd = get_scheduler("flux").wrap(["apptainer", "exec"], clusterb)
    assert cmd == ["flux", "run", "-N2", "--gpus-per-node=4", "apptainer", "exec"]


def test_module_loading_wraps_command(clusterb):
    # Prototype check_apptainer: module load rocm/6.4.0 before --rocm runs.
    cmd = get_scheduler("flux").with_modules(["apptainer", "exec", "x.sif"], clusterb)
    assert cmd[:2] == ["bash", "-lc"]
    assert "module load rocm/6.4.0" in cmd[2]
    assert "exec apptainer exec x.sif" in cmd[2]


def test_no_modules_no_wrap(clustera):
    assert get_scheduler("slurm").with_modules(["podman"], clustera) == ["podman"]


def test_alloc_commands(clustera, clusterb):
    assert get_scheduler("slurm").alloc_command(clustera) == ["salloc", "--nodes=2", "--gpus-per-node=4"]
    assert get_scheduler("flux").alloc_command(clusterb) == ["flux", "alloc", "-N2", "--gpus-per-node=4"]


# ---- site GRES convention: --gpus-per-node vs --gres=gpu:N (field: clusterd) ----------

import pytest  # noqa: E402


@pytest.mark.parametrize("directive,gtype,expected", [
    ("gpus-per-node", "", "--gpus-per-node=4"),          # default
    ("gres",          "", "--gres=gpu:4"),               # the portable form clusterd wanted
    ("gres",     "a100", "--gres=gpu:a100:4"),           # typed GRES
    ("gpus",          "", "--gpus=4"),
    ("gpus-per-node", "h100", "--gpus-per-node=h100:4"),
])
def test_slurm_gpu_directive_forms(clustera, monkeypatch, directive, gtype, expected):
    # a config env adapts the GPU request to any site's GRES convention — the
    # batch directive, the srun prefix, and salloc all follow it.
    from boxy import config

    monkeypatch.setenv("BOXY_GPU_DIRECTIVE", directive)
    monkeypatch.setenv("BOXY_GPU_TYPE", gtype)
    config.reset()
    sched = get_scheduler("slurm")
    assert f"#SBATCH {expected}" in sched.resource_directives(clustera)
    assert expected in sched.launch_prefix(clustera)
    assert expected in sched.alloc_command(clustera)


def test_slurm_gpu_directive_none_omits_the_request(clustera, monkeypatch):
    from boxy import config

    monkeypatch.setenv("BOXY_GPU_DIRECTIVE", "none")
    config.reset()
    sched = get_scheduler("slurm")
    assert not any("gpu" in ln.lower() or "gres" in ln.lower() for ln in sched.resource_directives(clustera))
