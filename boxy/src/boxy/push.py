"""`boxy push` — publish a HuggingFace model into SITE storage, once, so every
later serve pulls from infrastructure you control (fast, filter-proof, and the
feeder for air-gapped mirrors):

    boxy push meta-llama/Llama-3.1-8B-Instruct s3://models/llama31-8b
    boxy push nvidia/NVIDIA-Nemotron-Parse-v1.2 oci://registry.site.gov/models/nemotron-parse:v1.2

Then serve from the site copy anywhere boxy runs:

    boxy serve s3://models/llama31-8b/Llama-3.1-8B-Instruct --ssh cluster
    boxy serve oci://registry.site.gov/models/nemotron-parse:v1.2 --ssh cluster

The HF download reuses boxy's trust/proxy plumbing (run `boxy trust
huggingface.co` first on an intercepted network). S3 uploads ride the same
backend ladder as s3:// staging (boto3 -> aws CLI); OCI artifacts are pushed
with RamaLama's model-OCI support (`ramalama push`)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class PushError(RuntimeError):
    pass


def snapshot(model_id: str, token: str = "") -> str:
    """Download (or reuse) the model snapshot in the local HF cache; returns the
    snapshot directory path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise PushError("downloading needs huggingface_hub — pip install huggingface_hub "
                        "(or pip install 'boxy-hpc[ramalama]')") from None
    try:
        return snapshot_download(model_id, token=token or None)
    except Exception as e:  # noqa: BLE001 — every hub error gets the same remedy path
        raise PushError(f"could not download {model_id}: {e}\n"
                        "  intercepted network? run: boxy trust huggingface.co\n"
                        "  gated repo? export HF_TOKEN") from None


def _s3_upload(local_dir: str, uri: str, endpoint: str | None, dryrun: bool) -> None:
    from boxy import s3 as _s3

    bucket, prefix = _s3.parse_s3_uri(uri)
    ep = _s3.endpoint_url(endpoint)
    backend = _s3.choose_backend()
    if backend in ("awscli",) or (backend == "container" and shutil.which("aws")):
        backend = "awscli"
    if backend == "boto3":
        import boto3

        client = boto3.client("s3", endpoint_url=ep)
        files = sorted(p for p in Path(local_dir).rglob("*") if p.is_file())
        for i, p in enumerate(files, 1):
            key = f"{prefix.rstrip('/')}/{p.relative_to(local_dir)}" if prefix else str(
                p.relative_to(local_dir))
            print(f"###   [{i}/{len(files)}] s3://{bucket}/{key}")
            if not dryrun:
                client.upload_file(str(p), bucket, key)
        return
    if backend == "awscli":
        cmd = ["aws"] + (["--endpoint-url", ep] if ep else []) + \
              ["s3", "cp", "--recursive", local_dir, uri]
        print(f"###   {' '.join(cmd)}")
        if dryrun:
            return
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise PushError(f"aws s3 cp failed:\n{(proc.stderr or proc.stdout).strip()[-1200:]}")
        return
    raise PushError("no S3 backend: pip install boto3 (or 'boxy-hpc[s3]'), or put the aws "
                    "CLI on PATH; set the endpoint with --endpoint / S3_ENDPOINT_URL")


def _oci_push(model_id: str, local_dir: str, uri: str, dryrun: bool) -> None:
    """Publish the model as an OCI artifact via RamaLama's model-OCI support —
    the same transport `boxy serve oci://...` pulls with."""
    rl = shutil.which("ramalama")
    if not rl:
        raise PushError("pushing to a registry needs the ramalama CLI "
                        "(pip install ramalama, or 'boxy-hpc[ramalama]') — it packages the "
                        "model as an OCI artifact and pushes it")
    cmd = [rl, "push", f"hf://{model_id}", uri]
    print(f"###   {' '.join(cmd)}")
    if dryrun:
        return
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PushError(f"ramalama push failed:\n{(proc.stderr or proc.stdout).strip()[-1200:]}\n"
                        "  (registry auth: podman login <registry> first)")


def push_model(model_id: str, dest: str, *, endpoint: str | None = None,
               token: str = "", dryrun: bool = False) -> str:
    """Download MODEL from HF and publish it at DEST (s3://bucket/prefix or
    oci://registry/repo:tag). Returns the URI to serve from."""
    low = dest.lower()
    if low.startswith("s3://"):
        print(f"### push: downloading {model_id} from HuggingFace ...")
        local = snapshot(model_id, token) if not dryrun else "<snapshot>"
        print(f"### push: uploading to {dest} ...")
        _s3_upload(local, dest, endpoint, dryrun)
        served = dest.rstrip("/") + "/" + (Path(local).name if not dryrun else model_id.rsplit("/", 1)[-1])
        print(f"### pushed. serve it from site storage:\n"
              f"    boxy serve {dest.rstrip('/')} --ssh <cluster>")
        return served
    if low.startswith(("oci://", "docker://")):
        uri = dest if low.startswith("oci://") else "oci://" + dest.split("://", 1)[1]
        print(f"### push: {model_id} -> {uri} (OCI model artifact via ramalama)")
        _oci_push(model_id, "", uri, dryrun)
        print(f"### pushed. serve it from the registry:\n"
              f"    boxy serve {uri} --ssh <cluster>")
        return uri
    raise PushError(f"unsupported destination {dest!r} — expected s3://bucket/prefix or "
                    f"oci://registry/repo:tag")


def main_push(args) -> int:
    tok = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN") or ""
    try:
        push_model(args.model, args.dest, endpoint=getattr(args, "endpoint", None),
                   token=tok, dryrun=args.dryrun)
    except PushError as e:
        print(f"boxy push: {e}", file=sys.stderr)
        return 1
    return 0
