"""Stage models from an S3-compatible bucket (AWS S3, MinIO, Ceph RGW, ...).

Reads the SAME environment a Kubernetes vLLM deployment uses, so an existing
secret set works unchanged:

    S3_ENDPOINT_URL          on-prem/MinIO endpoint (empty => real AWS)
    S3_BUCKET_NAME           default bucket when the URI omits one
    S3_PATH                  default key prefix (e.g. meta-llama/Llama-3.1-8B-Instruct)
    AWS_ACCESS_KEY_ID        credentials (also AWS_SECRET_ACCESS_KEY)
    AWS_SECRET_ACCESS_KEY
    AWS_EC2_METADATA_DISABLED  set true for non-AWS endpoints (avoids slow probes)

Three interchangeable backends, chosen automatically (override with
BOXY_S3_BACKEND or --s3-backend):

    boto3       in-process, if the library is importable (fast, honours a
                custom endpoint_url for on-prem S3)
    awscli      the `aws` CLI if it is on PATH
    container   the aws CLI in a container via boxy's runtime (podman/docker/
                apptainer) — the paper's approach; needs NO host Python/CLI
                deps, just the container engine boxy already uses. Image is
                public.ecr.aws/aws-cli/aws-cli:latest (override BOXY_AWSCLI_IMAGE).

`auto` prefers boto3 -> host aws -> container, so a bare HPC node with only
podman still stages via the container. None of the three is a hard dependency.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

S3_SCHEME = "s3://"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key/prefix` -> ('bucket', 'key/prefix'). A bare `s3://` (or
    `s3://` with only a prefix and no bucket) falls back to S3_BUCKET_NAME /
    S3_PATH from the environment, mirroring the K8s secret layout."""
    rest = uri[len(S3_SCHEME):] if uri.startswith(S3_SCHEME) else uri
    rest = rest.strip("/")
    bucket, _, key = rest.partition("/")
    if not bucket:
        bucket = os.environ.get("S3_BUCKET_NAME", "")
    if not key:
        key = os.environ.get("S3_PATH", "")
    return bucket, key


def endpoint_url(explicit: str | None = None) -> str | None:
    return explicit or os.environ.get("S3_ENDPOINT_URL") or None


def credentials_present() -> bool:
    return bool(
        (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
        or os.path.exists(os.path.expanduser("~/.aws/credentials"))
    )


def _dest_for(key: str, models_dir: str) -> Path:
    """Local staging directory: <models_dir>/<last path component of the key>.
    e.g. key 'meta-llama/Llama-3.1-8B-Instruct' -> <models_dir>/Llama-3.1-8B-Instruct."""
    leaf = key.rstrip("/").rsplit("/", 1)[-1] or "model"
    return Path(os.path.abspath(models_dir)) / leaf


CONTAINER_IMAGE_DEFAULT = "public.ecr.aws/aws-cli/aws-cli:latest"
_CRED_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
             "AWS_DEFAULT_REGION", "AWS_REGION", "AWS_EC2_METADATA_DISABLED")


def container_image() -> str:
    return os.environ.get("BOXY_AWSCLI_IMAGE", CONTAINER_IMAGE_DEFAULT)


def choose_backend(runtime: str | None = None) -> str:
    """Resolve the effective staging backend from BOXY_S3_BACKEND or auto:
    boto3 -> host aws -> container."""
    forced = os.environ.get("BOXY_S3_BACKEND", "auto").lower()
    if forced in ("boto3", "awscli", "container"):
        return forced
    if _boto3_available():
        return "boto3"
    if shutil.which("aws"):
        return "awscli"
    if runtime and shutil.which(runtime):
        return "container"
    return "none"


def stage_model(uri: str, models_dir: str, endpoint: str | None = None, dryrun: bool = False,
                runtime: str = "podman", backend: str = "") -> str:
    """Download every object under the S3 prefix into a local directory and
    return that directory's path (boxy then serves it as a shared-FS model).
    Idempotent: unchanged objects are skipped.

    `runtime` is boxy's container engine, used by the 'container' backend.
    `backend` overrides selection ('' = auto / BOXY_S3_BACKEND)."""
    bucket, key = parse_s3_uri(uri)
    if not bucket:
        raise RuntimeError(
            "S3 model staging needs a bucket: pass s3://BUCKET/PREFIX, or set S3_BUCKET_NAME "
            "(and S3_PATH) in the environment (the same secret keys your K8s deployment uses)."
        )
    endpoint = endpoint_url(endpoint)
    # A custom endpoint means non-AWS: disable the EC2 metadata probe unless the
    # user already decided (matches AWS_EC2_METADATA_DISABLED in the K8s config).
    if endpoint and "AWS_EC2_METADATA_DISABLED" not in os.environ:
        os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    chosen = (backend or "").lower()
    if chosen in ("", "auto"):
        chosen = choose_backend(runtime)
    dest = _dest_for(key, models_dir)
    where = endpoint or "AWS S3"
    if dryrun:
        print(f"### Stage: s3://{bucket}/{key}  ({where}, via {chosen})  ->  {dest}")
        return str(dest)
    if not credentials_present():
        raise RuntimeError(
            "no S3 credentials found: export AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY "
            "(or AWS_PROFILE, or ~/.aws/credentials). boxy reads the same variables your "
            "K8s secret provides."
        )
    dest.mkdir(parents=True, exist_ok=True)
    print(f"### Staging s3://{bucket}/{key} from {where} -> {dest}  (via {chosen})", file=sys.stderr)
    if chosen == "boto3":
        _stage_boto3(bucket, key, dest, endpoint)
    elif chosen == "awscli":
        _stage_awscli(bucket, key, dest, endpoint)
    elif chosen == "container":
        _stage_container(bucket, key, dest, endpoint, runtime)
    else:
        raise RuntimeError(
            "no S3 staging backend available: install boto3 (pip install boto3), or the aws CLI, "
            f"or a container runtime for the aws-cli container ({container_image()}). "
            "Alternatively pre-stage the model to the shared filesystem and serve it by path."
        )
    return str(dest)


def _boto3_available() -> bool:
    try:
        import boto3  # noqa: F401

        return True
    except Exception:
        return False


def _stage_boto3(bucket: str, key: str, dest: Path, endpoint: str | None) -> None:
    import boto3

    client = boto3.client("s3", endpoint_url=endpoint)
    paginator = client.get_paginator("list_objects_v2")
    prefix = key if key.endswith("/") or not key else key + "/"
    found = False
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
            k, size = obj["Key"], obj.get("Size", 0)
            if k.endswith("/"):
                continue  # directory marker
            rel = k[len(prefix):] if k.startswith(prefix) else k.rsplit("/", 1)[-1]
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == size:
                found = True
                continue  # already staged, unchanged
            client.download_file(bucket, k, str(target))
            found = True
    if not found:
        raise RuntimeError(f"nothing found under s3://{bucket}/{key} — check the bucket/prefix")


def _aws_sync_args(bucket: str, key: str, endpoint: str | None, dest_in_container: str) -> list[str]:
    args = []
    if endpoint:
        args += ["--endpoint-url", endpoint]
    args += ["s3", "sync", f"s3://{bucket}/{key.rstrip('/')}/", dest_in_container]
    return args


def _stage_awscli(bucket: str, key: str, dest: Path, endpoint: str | None) -> None:
    cmd = ["aws"] + _aws_sync_args(bucket, key, endpoint, f"{dest}/")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"aws s3 sync failed (rc={result.returncode}) for s3://{bucket}/{key}")


def _stage_container(bucket: str, key: str, dest: Path, endpoint: str | None, runtime: str) -> None:
    """Run the aws CLI inside a container (the paper's approach): no host Python
    or aws-CLI dependency, just boxy's container engine. Credentials + endpoint
    flow through the environment, exactly like the K8s pod's secret env."""
    image = container_image()
    present = [v for v in _CRED_ENV if os.environ.get(v)]
    if runtime == "apptainer":
        # apptainer exec forwards host env by default; call `aws` explicitly
        # (its image ENTRYPOINT is not run by exec).
        cmd = ["apptainer", "exec", "--bind", f"{dest}:/dest", f"docker://{image}",
               "aws"] + _aws_sync_args(bucket, key, endpoint, "/dest/")
    else:  # podman / docker: image ENTRYPOINT is `aws`
        cmd = [runtime, "run", "--rm", "--network=host"]
        for var in present:
            cmd += ["-e", var]  # forward the value from boxy's environment
        cmd += ["-v", f"{dest}:/dest", image]
        cmd += _aws_sync_args(bucket, key, endpoint, "/dest/")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"aws-cli container sync failed (rc={result.returncode}) for s3://{bucket}/{key} "
            f"[{runtime} {image}]"
        )
