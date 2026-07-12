"""Container-IMAGE registry resolution — one pluggable module.

Sites rarely pull straight from docker.io/ghcr.io: air-gapped clusters run a
local mirror (Harbor/Artifactory/registry:2), and laptops use localhost images.
boxy therefore resolves every image reference through THIS module, driven by
data, so swapping registries never touches engine/backend code:

  1. `[location.image_mirrors]` — a rewrite map, per-registry, mirroring how
     containers' registries.conf mirrors work:
         [location.image_mirrors]
         "docker.io" = "registry.example.com/dockerhub"
         "ghcr.io"   = "registry.example.com/ghcr"
         "*"         = "registry.example.com/mirror"     # catch-all
     `docker.io/vllm/vllm-openai` (or the bare `vllm/vllm-openai`, which implies
     docker.io) becomes `registry.example.com/dockerhub/vllm/vllm-openai`.
  2. `--registry HOST[/path]` (or `registry=` in [location]) — send EVERYTHING
     to one registry: replaces the image's registry component, or prefixes a
     bare name. Mirrors win over this when both match.

Resolution is pure and applies uniformly: podman/docker run, apptainer's
OCI->SIF build, and the SkyPilot export all go through Backend.image_ref ->
resolve_image. `localhost/...` images stay local unless explicitly mirrored.
"""

from __future__ import annotations

DOCKERHUB = "docker.io"  # the implied registry of bare names ("vllm/vllm-openai")


def split_registry(image: str) -> tuple[str, str]:
    """(registry, remainder). Container convention: the first path component is
    a registry only if it looks like a host — contains a dot or a port, or is
    exactly `localhost`. Otherwise the name is Docker-Hub-implied."""
    first, sep, rest = image.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        return first, rest
    return "", image


def resolve_image(image: str, registry: str = "", mirrors: dict[str, str] | None = None) -> str:
    """Rewrite `image` for the site's registries. Precedence: an image_mirrors
    entry (exact registry key, then "*") > the blanket `registry` > unchanged."""
    if not image:
        return image
    host, rest = split_registry(image)
    if mirrors:
        target = mirrors.get(host or DOCKERHUB) or mirrors.get("*")
        if target:
            return f"{target.rstrip('/')}/{rest}"
    if registry:
        return f"{registry.rstrip('/')}/{rest}"
    return image
