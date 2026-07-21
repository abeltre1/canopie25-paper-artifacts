"""Air-gapped deployments: build a BUNDLE on a connected machine, carry it
across the gap, serve entirely from it.

    connected$  boxy bundle nvidia/NVIDIA-Nemotron-Parse-v1.2 -o nemotron-bundle/
    # move nemotron-bundle/ to the cluster (scp before the gap, or media)
    airgap$     boxy serve nvidia/NVIDIA-Nemotron-Parse-v1.2 \
                    --bundle /projects/me/nemotron-bundle --ssh cluster

A bundle is a plain directory (rsync/scp/tar-able):

    hfcache/        HF_HOME cache tree: the model AND its auxiliary custom-code
                    repos (a VLM like Nemotron-Parse dynamically fetches its
                    vision encoder repo — fatal air-gapped unless pre-cached)
    image.oci.tar   the engine container image (podman save --format oci-archive)
    wheels/         the card's pip deps as wheels (pip download)
    manifest.toml   what's inside (model, image, aux repos, pip, versions)

Serving --bundle: the batch script `podman load`s the image from the bundle,
mounts hfcache/ as the container's HF_HOME with HF_HUB_OFFLINE=1 (the engine and
transformers resolve the repo id FROM CACHE — custom code included), and the
pip wrapper installs from wheels/ with --no-index. No proxy, no egress, nothing
fetched — the bundle is the whole world."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class BundleError(RuntimeError):
    pass


def _run(cmd: list[str], env: dict | None = None, what: str = "") -> None:
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BundleError(f"{what or cmd[0]} failed:\n{(proc.stderr or proc.stdout).strip()[-1200:]}")


def build_wheelhouse(out: str, *, platform: str = "linux/amd64", python: str = "3.12",
                     extras: str = "ramalama,plot", runtime: str = "",
                     ca_file: str = "", verify: bool = True) -> Path:
    """TURNKEY offline wheel set: boxy's own wheel plus the full dependency
    closure of the requested extras, built INSIDE a python:<ver> container so
    every compiled wheel matches the TARGET (--platform), not this machine —
    the field lesson: an Apple-Silicon laptop's podman VM silently produced an
    aarch64 set that no x86_64 cluster could install. The site CA (boxy's
    ca-merged.crt) and network.proxy ride along; a --network=none install
    verifies the set is complete before anyone carries it across a gap."""
    import boxy
    from boxy import config, ramalama_shim

    src = Path(boxy.__file__).resolve().parents[2]
    if not (src / "pyproject.toml").is_file():
        raise BundleError(
            f"`boxy wheels` builds boxy's wheel from its git checkout, but this boxy is a "
            f"frozen wheel install ({Path(boxy.__file__).parent}) — run it where boxy is "
            f"installed editable from the repo (`pip install -e ./boxy`)")
    rt = shutil.which(runtime) if runtime else (shutil.which("podman") or shutil.which("docker"))
    if not rt:
        raise BundleError("building a platform-correct wheel set needs podman or docker — "
                          "the wheels must match the TARGET system, not this machine")
    d = Path(os.path.expanduser(out))
    d.mkdir(parents=True, exist_ok=True)
    req = f"boxy-hpc[{extras}]" if extras else "boxy-hpc"
    image = f"docker.io/library/python:{python}"
    opts = ["-v", f"{src}:/src:ro", "-v", f"{d}:/wheels"]
    # the container must trust the site's TLS interceptor: --ca wins, then the
    # same SSL_CERT_FILE the user's other tools use, then boxy's merged bundle
    ca = next((p for p in (ca_file, os.environ.get("SSL_CERT_FILE", ""),
                           str(Path(ramalama_shim.DEFAULT_STORE) / "ca-merged.crt"))
               if p and Path(os.path.expanduser(p)).is_file()), "")
    if ca:
        ca = os.path.expanduser(ca)
        print(f"### wheels: site CA into the container: {ca}")
        opts += ["-v", f"{ca}:/ca.crt:ro", "-e", "PIP_CERT=/ca.crt",
                 "-e", "SSL_CERT_FILE=/ca.crt", "-e", "REQUESTS_CA_BUNDLE=/ca.crt"]
    pfx = config.get("network.proxy")
    if pfx:
        opts += ["-e", f"https_proxy={pfx}", "-e", f"http_proxy={pfx}"]
    spec = f"/tmp/b[{extras}]" if extras else "/tmp/b"
    # the checkout mounts read-only; the build runs on a copy so a stale
    # *.egg-info from an old editable install can't fail it (field)
    script = ("cp -r /src /tmp/b && rm -rf /tmp/b/src/*.egg-info && "
              f"pip -q download '{spec}' -d /wheels && "
              "pip -q wheel /tmp/b --no-deps -w /wheels")
    print(f"### wheels: building the {platform} py{python} set into {d}/ (inside {image}) ...")
    _run([rt, "run", "--rm", "--platform", platform, *opts, image, "bash", "-c", script],
         what="wheelhouse build")
    if verify:
        print("### wheels: verifying a fully OFFLINE install (--network=none) ...")
        _run([rt, "run", "--rm", "--platform", platform, "--network=none",
              "-v", f"{d}:/wheels:ro", image, "bash", "-c",
              f"pip -q install --no-index --find-links /wheels '{req}'"],
             what="offline verification")
        print("### wheels: offline install verified — the set is complete")
    return d


def _hf_download(repo: str, hf_home: str, token: str = "") -> None:
    """Populate the bundle's HF cache with a full repo snapshot. Prefers the
    huggingface_hub Python API (ships with vLLM/ramalama installs); falls back
    to the huggingface-cli binary."""
    env = {**os.environ, "HF_HOME": hf_home}
    if token:
        env["HF_TOKEN"] = token
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        old = os.environ.get("HF_HOME")
        os.environ["HF_HOME"] = hf_home
        try:
            snapshot_download(repo, token=token or None)
        finally:
            if old is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = old
        return
    except ImportError:
        pass
    cli = shutil.which("huggingface-cli") or shutil.which("hf")
    if not cli:
        raise BundleError(
            "downloading the model needs huggingface_hub (pip install huggingface_hub) "
            "or the huggingface-cli binary on PATH")
    _run([cli, "download", repo], env=env, what=f"huggingface-cli download {repo}")


def build_bundle(model_id: str, dest: str, image: str, *, aux_repos: list[str] | None = None,
                 pip_pkgs: list[str] | None = None, token: str = "",
                 runtime: str = "podman", skip_image: bool = False,
                 bake: bool = False) -> str:
    """Create <dest>/ with everything an air-gapped serve needs. Returns the
    manifest path. Each stage prints what it's doing; a missing tool fails with
    the exact remedy."""
    d = Path(os.path.expanduser(dest))
    d.mkdir(parents=True, exist_ok=True)
    hf_home = str(d / "hfcache")

    repos = [model_id, *(aux_repos or [])]
    for repo in repos:
        print(f"### bundle: caching {repo} into hfcache/ ...")
        _hf_download(repo, hf_home, token)

    if pip_pkgs:
        wheels = d / "wheels"
        wheels.mkdir(exist_ok=True)
        pip = [sys.executable, "-m", "pip"]
        print(f"### bundle: downloading wheels: {' '.join(pip_pkgs)} ...")
        _run([*pip, "download", "--dest", str(wheels), *pip_pkgs], what="pip download")

    if not skip_image:
        rt = shutil.which(runtime)
        if not rt:
            raise BundleError(f"{runtime} not on PATH — needed to pull+save the engine image "
                              f"(or pass --skip-image and load the image out of band)")
        print(f"### bundle: pulling {image} ...")
        _run([rt, "pull", image], what=f"{runtime} pull")
        if bake and pip_pkgs:
            # BAKE the card's pip deps into a derived image so the air-gapped
            # container starts instantly with no pip step at all (the wheels/
            # dir still ships as belt-and-suspenders for user-added deps).
            slug = model_id.rsplit("/", 1)[-1].lower()
            baked = f"localhost/boxy-{slug}:baked"
            cf = d / "Containerfile.boxy"
            cf.write_text(f"FROM {image}\n"
                          f"RUN pip install --no-cache-dir {' '.join(pip_pkgs)}\n")
            print(f"### bundle: baking {', '.join(pip_pkgs)} into {baked} ...")
            _run([rt, "build", "-t", baked, "-f", str(cf), str(d)], what=f"{runtime} build")
            image = baked
        print("### bundle: saving image.oci.tar ...")
        _run([rt, "save", "--format", "oci-archive", "-o", str(d / "image.oci.tar"), image],
             what=f"{runtime} save")

    manifest = d / "manifest.toml"
    lines = [
        "# boxy air-gap bundle — see `boxy serve MODEL --bundle <this dir>`",
        "[bundle]",
        f'model = "{model_id}"',
        f'image = "{image}"',
        f'created = "{datetime.now(timezone.utc).isoformat()}"',
    ]
    if aux_repos:
        lines.append("aux_repos = [" + ", ".join(f'"{r}"' for r in aux_repos) + "]")
    if pip_pkgs:
        lines.append("pip = [" + ", ".join(f'"{p}"' for p in pip_pkgs) + "]")
    manifest.write_text("\n".join(lines) + "\n")
    print(f"### bundle ready: {d}")
    print("###   move it across the gap (scp -r / tar), then:")
    print(f"###   boxy serve {model_id} --bundle /path/on/cluster/{d.name} --ssh <cluster>")
    return str(manifest)


def read_remote_manifest(target: str, bundle_dir: str) -> dict:
    """The bundle's manifest, read off the CLUSTER over the ssh master. Empty
    dict when unreadable (the caller degrades to flags/defaults)."""
    import tomllib

    from boxy import remote

    rc, out = remote.ssh_capture(
        target, f"cat {shlex.quote(bundle_dir.rstrip('/') + '/manifest.toml')}", timeout=15)
    if rc != 0 or not out.strip():
        return {}
    try:
        return tomllib.loads(out).get("bundle", {})
    except tomllib.TOMLDecodeError:
        return {}
