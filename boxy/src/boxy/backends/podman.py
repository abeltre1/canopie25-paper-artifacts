"""Podman backend — mirrors the prototype's build_podman_command
(hpc-workflow/common_boxy.sh) and RamaLama's engine device wiring."""

from __future__ import annotations

import sys

from boxy.backends.base import (
    DRI_ACCELERATORS,
    RuntimeBackend,
    relabel_option,
    selinux_enforcing,
    warn_cpu_only,
)
from boxy.box import Box
from boxy.location import Location

# GPU pass-through per accelerator (prototype "Podman magic args").
CUDA_ARGS = ["--device", "nvidia.com/gpu=all"]
ROCM_ARGS = [
    "--group-add=video",
    "--cap-add=SYS_PTRACE",
    "--device", "/dev/kfd",
    "--device", "/dev/dri",
    "--security-opt", "seccomp=unconfined",
]


# Where the built command will RUN, when that differs from where it is BUILT.
# The agentless renderer builds the podman command on the laptop (often macOS)
# for execution on a Linux compute node — sys.platform must not decide there.
# FIELD (clusterb, TP=2 Nemotron): a Mac-rendered agentless script took the
# darwin branch below (-p port publishing, no --ipc=host), so the container
# ran with podman's default 64MB /dev/shm and RCCL died at ncclCommInitRank
# with 'NCCL error: unhandled system error' the moment a second GPU joined.
_target_os: str | None = None  # None = this host; "linux" pinned by the agentless renderer


def set_target_os(value: str | None) -> None:
    """Pin the platform the rendered command targets ('linux'), or None to
    follow sys.platform again. Process-global (same pattern as the Slurm
    auto-GRES override); the agentless renderer resets it in a finally."""
    global _target_os
    _target_os = value


def _serve_ports(box: Box, inner_cmd: list[str]) -> list[int]:
    """Ports to publish. A --port in the engine args is the port the server
    will ACTUALLY bind, so it replaces (not augments) the box's declared
    ports — publishing the stale declared port binds the host port the user
    was avoiding (finding 58)."""
    from boxy.engines import parse_port_flag

    from_cmd = parse_port_flag(inner_cmd)  # LAST occurrence = the bound port
    return [from_cmd] if from_cmd is not None else sorted(set(box.ports))


class PodmanBackend(RuntimeBackend):
    name = "podman"
    image_format = "oci"
    run_verb = "run"

    def network_args(self, box: Box, inner_cmd: list[str]) -> list[str]:
        """Linux (HPC nodes): host networking + host IPC, per the paper's
        prototype — --ipc=host also gives the container the host's /dev/shm,
        which NCCL/RCCL need for multi-GPU communicators (podman's default
        64MB is fatally small: 'NCCL error: unhandled system error').
        macOS: podman-machine/Docker Desktop run containers in a Linux VM,
        where --network=host binds inside the VM and is unreachable from the
        host — publish ports instead. (Field finding #11, 2026-07.)
        The platform is the TARGET's when pinned (agentless render for a
        Linux cluster from a Mac), else this host's."""
        if (_target_os or sys.platform) == "darwin":
            args: list[str] = []
            for port in _serve_ports(box, inner_cmd):
                args += ["-p", f"{port}:{port}"]
            return args
        return ["--network=host", "--ipc=host"]

    def gpu_args(self, accelerator: str) -> list[str]:
        if accelerator == "cuda":
            return list(CUDA_ARGS)
        if accelerator == "rocm":
            return list(ROCM_ARGS)
        if accelerator in DRI_ACCELERATORS:  # intel / vulkan / asahi via /dev/dri
            return ["--device", "/dev/dri"]
        if accelerator and accelerator != "none":  # known but no device pass-through
            warn_cpu_only(accelerator, self.name)
        return []

    def build_command(
        self,
        box: Box,
        location: Location,
        inner_cmd: list[str],
        env: dict[str, str],
        mounts: list[tuple[str, str, str]],
        accelerator: str,
    ) -> list[str]:
        entrypoint, inner_args = inner_cmd[0], inner_cmd[1:]
        # --pids-limit=-1: podman's default pid ceiling (2048) is far too small for
        # inference serving on big nodes — field: on 192-CPU MI300A nodes, Ray
        # prestarts one worker process per declared CPU when the first driver
        # registers, the fork storm hits the ceiling mid-spawn, and the raylet
        # wedges with the driver blocked forever in RegisterClient's recv().
        # vLLM/llama.cpp thread pools scale with cores too. RamaLama lifts the
        # limit the same way.
        cmd = [self.name, self.run_verb, "--rm", "--pids-limit=-1", f"--name={box.name}"]
        cmd += self.network_args(box, inner_cmd)
        cmd += [f"--label=boxy.box={box.name}"]  # lets `boxy list` find boxy-launched containers
        if entrypoint:  # "" => keep the image's own ENTRYPOINT, pass args only
            cmd += [f"--entrypoint={entrypoint}"]
        if box.workdir:
            cmd += [f"--workdir={box.workdir}"]
        from boxy import config

        relabel_mode, enforcing = config.get("mounts.selinux_relabel"), selinux_enforcing()
        for source, target, options in mounts:
            # add ':z' on SELinux-enforcing hosts so rootless bind mounts don't
            # get 'permission denied' (config mounts.selinux_relabel: auto|always|never).
            options = relabel_option(options, relabel_mode, enforcing)
            spec = f"{source}:{target}"
            if options:
                spec += f":{options}"
            cmd += [f"--volume={spec}"]
        cmd += self.gpu_args(accelerator)
        for key, value in env.items():
            cmd += ["--env", f"{key}={value}"]
        cmd += [self.image_ref(box, location)]
        cmd += inner_args
        return cmd
