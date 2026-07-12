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
        """Linux (HPC nodes): host networking, per the paper's prototype.
        macOS: podman-machine/Docker Desktop run containers in a Linux VM,
        where --network=host binds inside the VM and is unreachable from the
        host — publish ports instead. (Field finding #11, 2026-07.)"""
        if sys.platform == "darwin":
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
        cmd = [self.name, self.run_verb, "--rm", f"--name={box.name}"]
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
