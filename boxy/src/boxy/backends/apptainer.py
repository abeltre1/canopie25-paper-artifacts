"""Apptainer/Singularity backend — mirrors the prototype's
build_apptainer_command + build_apptainer_image (hpc-workflow/common_boxy.sh).

Apptainer args mimic Podman behavior for GenAI stacks that expect root:
vllm needs to write to /root, and mapping host env vars breaks Python, hence
--fakeroot --writable-tmpfs --cleanenv --no-home (the exact set the paper
found necessary)."""

from __future__ import annotations

from boxy.backends.base import DRI_ACCELERATORS, RuntimeBackend, warn_cpu_only
from boxy.box import Box
from boxy.location import Location

BASE_ARGS = ["--fakeroot", "--writable-tmpfs", "--cleanenv", "--no-home"]


class ApptainerBackend(RuntimeBackend):
    name = "apptainer"
    image_format = "sif"

    def sif_name(self, box: Box, accelerator: str) -> str:
        # Prototype: SHORT_NAME="vllm-${TARGET}" -> vllm-cuda.sif
        return f"{box.name}-{accelerator}.sif"

    def prepare(self, box: Box, location: Location, dryrun: bool = False,
                accelerator: str | None = None) -> list[list[str]]:
        """Auto-build the SIF from the OCI image if it doesn't exist.

        Uses the RESOLVED accelerator (sweep findings 3/37/54: naming the SIF
        from the raw location field built '<name>-none.sif' on autodetect
        locations while the run command exec'd '<name>-<detected>.sif')."""
        resolved = accelerator if accelerator is not None else location.resolve_accelerator()
        sif = self.sif_name(box, resolved)
        return [[self.name, "build", "--force", sif, f"docker://{self.image_ref(box, location)}"]]

    def gpu_args(self, accelerator: str) -> list[str]:
        if accelerator == "cuda":
            return ["--nv"]
        if accelerator == "rocm":
            return ["--rocm"]
        if accelerator in DRI_ACCELERATORS:  # intel / vulkan / asahi via /dev/dri
            return ["--bind", "/dev/dri"]
        if accelerator and accelerator != "none":
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
        # Empty entrypoint => `apptainer run` (executes the SIF runscript,
        # i.e. the original image ENTRYPOINT) instead of `exec <command>`.
        defer_to_image = inner_cmd and inner_cmd[0] == ""
        cmd = [self.name, "run" if defer_to_image else "exec"]
        cmd += BASE_ARGS
        if box.workdir:
            cmd += ["--cwd", box.workdir]
        for source, target, options in mounts:
            spec = f"{source}:{target}"
            if "ro" in options.split(","):
                spec += ":ro"
            cmd += ["--bind", spec]
        # vllm needs a writable HF cache under /root (prototype rule).
        cmd += ["--env", "HF_HOME=/root/.cache/huggingface"]
        cmd += self.gpu_args(accelerator)
        for key, value in env.items():
            cmd += ["--env", f"{key}={value}"]
        cmd += [self.sif_name(box, accelerator)]
        cmd += inner_cmd[1:] if defer_to_image else inner_cmd
        return cmd
