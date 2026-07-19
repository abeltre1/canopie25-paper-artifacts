"""CharlieCloud backend (ch-run) — EXPERIMENTAL.

CharlieCloud is a fully unprivileged HPC container runtime with NO daemon and no
setuid helper, favored on security-conscious clusters. Like Apptainer it builds
an image artifact ahead of time and then runs it directly:

    ch-image pull docker://<image>          # into ch-image storage
    ch-convert <image> <name>-<accel>.dir   # unpack to a directory image
    ch-fromhost --nvidia <dir>              # inject host CUDA libs (NVIDIA idiom)
    ch-run <dir> -- <command>               # run, unprivileged, no daemon

Marked experimental and golden-argv-tested only (no live CI dependency): GPU
visibility and writable-home behavior are site-dependent, so a site may need to
tune `binaries.*`/mounts. Everything above the RuntimeBackend seam (cards,
scheduler, resolver) is unchanged — this is one class + a registration.
"""

from __future__ import annotations

import shutil

from boxy.backends.base import DRI_ACCELERATORS, RuntimeBackend, warn_cpu_only
from boxy.box import Box
from boxy.location import Location


class CharlieCloudBackend(RuntimeBackend):
    name = "charliecloud"        # registry key / --runtime value
    binary = "ch-run"            # the actual runner (build tools: ch-image/ch-convert)
    image_format = "charliecloud"  # a directory image built by ch-convert

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def dir_name(self, box: Box, accelerator: str) -> str:
        # like Apptainer's per-accelerator SIF: a directory image per target
        return f"{box.name}-{accelerator}.dir"

    def prepare(self, box: Box, location: Location, dryrun: bool = False,
                accelerator: str | None = None) -> list[list[str]]:
        """Pull + unpack the OCI image to a directory, injecting host CUDA libs
        for cuda (the documented CharlieCloud NVIDIA method)."""
        resolved = accelerator if accelerator is not None else location.resolve_accelerator()
        ref = self.image_ref(box, location)
        image_dir = self.dir_name(box, resolved)
        cmds = [
            ["ch-image", "pull", f"docker://{ref}"],
            ["ch-convert", ref, image_dir],
        ]
        if resolved == "cuda":
            cmds.append(["ch-fromhost", "--nvidia", image_dir])
        return cmds

    def gpu_args(self, accelerator: str) -> list[str]:
        # NVIDIA libs are injected at build (ch-fromhost --nvidia), so no run
        # flag. ROCm/DRI GPUs are reached by binding the device nodes.
        if accelerator == "cuda":
            return []
        if accelerator == "rocm":
            return ["-b", "/dev/kfd", "-b", "/dev/dri"]
        if accelerator in DRI_ACCELERATORS:
            return ["-b", "/dev/dri"]
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
        # ch-run has no `run vs exec` split (no runscript concept); an empty
        # entrypoint means the image's own default, so drop the placeholder.
        defer_to_image = bool(inner_cmd) and inner_cmd[0] == ""
        argv = inner_cmd[1:] if defer_to_image else inner_cmd

        cmd = [self.binary, "--write"]  # writable image (vLLM writes caches); unprivileged
        if box.workdir:
            cmd += ["--cd", box.workdir]
        for source, target, options in mounts:
            spec = f"{source}:{target}"
            cmd += ["-b", spec]  # CharlieCloud binds are read-write by design
        cmd += ["--set-env=HF_HOME=/root/.cache/huggingface"]
        cmd += self.gpu_args(accelerator)
        for key, value in env.items():
            cmd += [f"--set-env={key}={value}"]
        cmd += [self.dir_name(box, accelerator), "--"]
        cmd += argv
        return cmd
