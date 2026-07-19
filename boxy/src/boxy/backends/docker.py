"""Docker backend — thin variant of the Podman builder for dev/non-HPC sites."""

from __future__ import annotations

from boxy.backends.podman import PodmanBackend


class DockerBackend(PodmanBackend):
    name = "docker"

    def gpu_args(self, accelerator: str) -> list[str]:
        if accelerator == "cuda":
            return ["--gpus", "all"]
        return super().gpu_args(accelerator)
