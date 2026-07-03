"""Podman backend — mirrors the prototype's build_podman_command
(hpc-workflow/common_boxy.sh) and RamaLama's engine device wiring."""

from __future__ import annotations

import sys

from boxy.backends.base import RuntimeBackend
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
    """Ports to publish: the box's declared ports plus any --port in the
    engine args (covers `boxy serve --port N` overrides)."""
    ports = set(box.ports)
    for i, arg in enumerate(inner_cmd):
        if arg == "--port" and i + 1 < len(inner_cmd) and inner_cmd[i + 1].isdigit():
            ports.add(int(inner_cmd[i + 1]))
        elif arg.startswith("--port="):
            value = arg.split("=", 1)[1]
            if value.isdigit():
                ports.add(int(value))
    return sorted(ports)


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
        for source, target, options in mounts:
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
