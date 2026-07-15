from boxy.schedulers import get_scheduler


def test_none_is_passthrough(eldorado):
    assert get_scheduler("none").wrap(["echo", "hi"], eldorado) == ["echo", "hi"]


def test_slurm_srun_prefix(hops):
    cmd = get_scheduler("slurm").wrap(["podman", "run"], hops)
    assert cmd == ["srun", "--nodes=2", "--gpus-per-node=4", "podman", "run"]


def test_slurm_clears_xdg(hops):
    # Prototype check_podman: unset XDG_SESSION_ID / XDG_RUNTIME_DIR in jobs.
    assert get_scheduler("slurm").host_env_fixups() == ["XDG_SESSION_ID", "XDG_RUNTIME_DIR"]


def test_flux_run_prefix(eldorado):
    cmd = get_scheduler("flux").wrap(["apptainer", "exec"], eldorado)
    assert cmd == ["flux", "run", "-N2", "--gpus-per-node=4", "apptainer", "exec"]


def test_module_loading_wraps_command(eldorado):
    # Prototype check_apptainer: module load rocm/6.4.0 before --rocm runs.
    cmd = get_scheduler("flux").with_modules(["apptainer", "exec", "x.sif"], eldorado)
    assert cmd[:2] == ["bash", "-lc"]
    assert "module load rocm/6.4.0" in cmd[2]
    assert "exec apptainer exec x.sif" in cmd[2]


def test_no_modules_no_wrap(hops):
    assert get_scheduler("slurm").with_modules(["podman"], hops) == ["podman"]


def test_alloc_commands(hops, eldorado):
    assert get_scheduler("slurm").alloc_command(hops) == ["salloc", "--nodes=2", "--gpus-per-node=4"]
    assert get_scheduler("flux").alloc_command(eldorado) == ["flux", "alloc", "-N2", "--gpus-per-node=4"]


# ---- site GRES convention: --gpus-per-node vs --gres=gpu:N (field: kahuna) ----------

import pytest  # noqa: E402


@pytest.mark.parametrize("directive,gtype,expected", [
    ("gpus-per-node", "", "--gpus-per-node=4"),          # default
    ("gres",          "", "--gres=gpu:4"),               # the portable form kahuna wanted
    ("gres",     "a100", "--gres=gpu:a100:4"),           # typed GRES
    ("gpus",          "", "--gpus=4"),
    ("gpus-per-node", "h100", "--gpus-per-node=h100:4"),
])
def test_slurm_gpu_directive_forms(hops, monkeypatch, directive, gtype, expected):
    # a config env adapts the GPU request to any site's GRES convention — the
    # batch directive, the srun prefix, and salloc all follow it.
    from boxy import config

    monkeypatch.setenv("BOXY_GPU_DIRECTIVE", directive)
    monkeypatch.setenv("BOXY_GPU_TYPE", gtype)
    config.reset()
    sched = get_scheduler("slurm")
    assert f"#SBATCH {expected}" in sched.resource_directives(hops)
    assert expected in sched.launch_prefix(hops)
    assert expected in sched.alloc_command(hops)


def test_slurm_gpu_directive_none_omits_the_request(hops, monkeypatch):
    from boxy import config

    monkeypatch.setenv("BOXY_GPU_DIRECTIVE", "none")
    config.reset()
    sched = get_scheduler("slurm")
    assert not any("gpu" in ln.lower() or "gres" in ln.lower() for ln in sched.resource_directives(hops))
