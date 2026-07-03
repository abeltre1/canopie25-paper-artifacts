"""S3 model staging: fetch from a site-local S3-compatible bucket (MinIO/Ceph/
AWS), then serve by path. Reads the same env a K8s vLLM deployment uses."""

import sys
import types

import pytest

from boxy import s3
from boxy.box import Box
from boxy.cli import main


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("S3_ENDPOINT_URL", "S3_BUCKET_NAME", "S3_PATH", "AWS_ACCESS_KEY_ID",
              "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE", "AWS_EC2_METADATA_DISABLED"):
        monkeypatch.delenv(k, raising=False)


def test_parse_uri_explicit_and_env(monkeypatch):
    assert s3.parse_s3_uri("s3://bkt/meta-llama/Llama-3.1-8B") == ("bkt", "meta-llama/Llama-3.1-8B")
    monkeypatch.setenv("S3_BUCKET_NAME", "vllm-models")
    monkeypatch.setenv("S3_PATH", "meta-llama/Llama-3.1-8B-Instruct")
    assert s3.parse_s3_uri("s3://") == ("vllm-models", "meta-llama/Llama-3.1-8B-Instruct")
    # explicit bucket, prefix falls back to env
    assert s3.parse_s3_uri("s3://other") == ("other", "meta-llama/Llama-3.1-8B-Instruct")


def test_endpoint_auto_disables_ec2_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.local:9000")
    s3.stage_model("s3://b/prefix", str(tmp_path), dryrun=True)
    assert __import__("os").environ["AWS_EC2_METADATA_DISABLED"] == "true"


def test_stage_requires_bucket(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="needs a bucket"):
        s3.stage_model("s3://", str(tmp_path))


def test_stage_requires_credentials(monkeypatch, tmp_path):
    # bucket present, but no creds -> clear error (not a boto3 stack trace)
    with pytest.raises(RuntimeError, match="no S3 credentials"):
        s3.stage_model("s3://bkt/prefix", str(tmp_path))


class _FakeS3Client:
    """Minimal boto3 s3 client: one bucket/prefix with two objects."""

    def __init__(self, objects):
        self.objects = objects  # {key: bytes}
        self.endpoint = None

    def get_paginator(self, _op):
        client = self

        class _P:
            def paginate(self, Bucket, Prefix):
                contents = [{"Key": k, "Size": len(v)} for k, v in client.objects.items()
                            if k.startswith(Prefix)]
                yield {"Contents": contents}

        return _P()

    def download_file(self, Bucket, Key, target):
        with open(target, "wb") as f:
            f.write(self.objects[Key])


def _install_fake_boto3(monkeypatch, objects, capture=None):
    fake = types.ModuleType("boto3")

    def client(_name, endpoint_url=None):
        c = _FakeS3Client(objects)
        if capture is not None:
            capture["endpoint_url"] = endpoint_url
        return c

    fake.client = client
    monkeypatch.setitem(sys.modules, "boto3", fake)


def test_stage_downloads_prefix_with_boto3(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.local:9000")
    objects = {
        "meta-llama/Llama-3.1-8B-Instruct/config.json": b"{}",
        "meta-llama/Llama-3.1-8B-Instruct/model-00001-of-00001.safetensors": b"AAAA",
        "meta-llama/Llama-3.1-8B-Instruct/": b"",  # dir marker, skipped
    }
    cap = {}
    _install_fake_boto3(monkeypatch, objects, cap)
    dest = s3.stage_model("s3://bkt/meta-llama/Llama-3.1-8B-Instruct", str(tmp_path))
    assert cap["endpoint_url"] == "https://s3.local:9000"        # custom endpoint honoured
    assert (tmp_path / "Llama-3.1-8B-Instruct" / "config.json").read_bytes() == b"{}"
    assert (tmp_path / "Llama-3.1-8B-Instruct" / "model-00001-of-00001.safetensors").exists()
    assert dest.endswith("Llama-3.1-8B-Instruct")


def test_serve_s3_stages_then_serves_by_path(monkeypatch, tmp_path, capsys):
    """`boxy serve s3://...` stages to the shared FS and serves the local dir."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.local:9000")
    objects = {"meta-llama/Llama/config.json": b"{}"}
    _install_fake_boto3(monkeypatch, objects)
    loc = tmp_path / "loc.toml"
    loc.write_text('[location]\nname="l"\nruntime="docker"\naccelerator="cuda"\n'
                   f'[location.staging]\nmodels_dir="{tmp_path / "staged"}"\n')
    # explicit engine/accel: an s3:// prefix has no extension to infer from
    rc = main(["serve", "s3://bkt/meta-llama/Llama", "--location", str(loc),
               "--engine", "vllm", "--here", "--dryrun"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "S3 bucket — staged to the shared filesystem" in out
    assert str(tmp_path / "staged" / "Llama") in out           # served the staged dir


def test_box_model_is_s3():
    assert Box(name="b", engine="vllm", model="s3://bkt/x").model_is_s3
    assert not Box(name="b", engine="vllm", model="hf://a/b").model_is_s3


def test_choose_backend_prefers_boto3_then_aws_then_container(monkeypatch):
    monkeypatch.delenv("BOXY_S3_BACKEND", raising=False)
    monkeypatch.setattr(s3, "_boto3_available", lambda: True)
    assert s3.choose_backend("podman") == "boto3"
    monkeypatch.setattr(s3, "_boto3_available", lambda: False)
    monkeypatch.setattr(s3.shutil, "which", lambda b: "/usr/bin/aws" if b == "aws" else None)
    assert s3.choose_backend("podman") == "awscli"
    monkeypatch.setattr(s3.shutil, "which", lambda b: "/usr/bin/podman" if b == "podman" else None)
    assert s3.choose_backend("podman") == "container"   # bare HPC node: only the runtime
    assert s3.choose_backend(None) == "none"


def test_backend_forced_via_env(monkeypatch):
    monkeypatch.setenv("BOXY_S3_BACKEND", "container")
    monkeypatch.setattr(s3, "_boto3_available", lambda: True)  # would win in auto
    assert s3.choose_backend("podman") == "container"


def test_container_backend_builds_podman_command(monkeypatch, tmp_path):
    """The paper's approach: aws CLI in a container via boxy's runtime, creds
    forwarded through the env, endpoint honoured."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.local:9000")
    monkeypatch.setenv("BOXY_S3_BACKEND", "container")
    monkeypatch.setenv("BOXY_AWSCLI_IMAGE", "public.ecr.aws/aws-cli/aws-cli:2.15")
    seen = {}

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        import types as _t
        return _t.SimpleNamespace(returncode=0)

    monkeypatch.setattr(s3.subprocess, "run", fake_run)
    s3.stage_model("s3://bkt/meta-llama/Llama", str(tmp_path), runtime="podman")
    cmd = seen["cmd"]
    assert cmd[:3] == ["podman", "run", "--rm"]
    assert "-e" in cmd and "AWS_ACCESS_KEY_ID" in cmd            # creds forwarded
    assert "public.ecr.aws/aws-cli/aws-cli:2.15" in cmd          # image override
    assert "--endpoint-url" in cmd and "https://s3.local:9000" in cmd
    assert "sync" in cmd and "s3://bkt/meta-llama/Llama/" in cmd
    assert f"{tmp_path / 'Llama'}:/dest" in cmd                  # bind mount


def test_no_sign_skips_credentials_and_adds_flag_awscli(monkeypatch, tmp_path):
    """Public bucket: --no-sign-request needs no creds and passes the flag through."""
    monkeypatch.setenv("BOXY_S3_BACKEND", "awscli")
    monkeypatch.setattr(s3.shutil, "which", lambda b: "/usr/bin/aws")
    seen = {}
    monkeypatch.setattr(s3.subprocess, "run",
                        lambda cmd, *a, **k: seen.update(cmd=cmd) or __import__("types").SimpleNamespace(returncode=0))
    # no AWS_* creds set at all -> would normally raise; --no-sign-request allows it
    s3.stage_model("s3://public/models/x", str(tmp_path), no_sign=True)
    assert "--no-sign-request" in seen["cmd"]


def test_no_sign_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("S3_NO_SIGN_REQUEST", "1")
    monkeypatch.setenv("BOXY_S3_BACKEND", "container")
    seen = {}
    monkeypatch.setattr(s3.subprocess, "run",
                        lambda cmd, *a, **k: seen.update(cmd=cmd) or __import__("types").SimpleNamespace(returncode=0))
    s3.stage_model("s3://public/models/x", str(tmp_path), runtime="podman")
    assert "--no-sign-request" in seen["cmd"]


def test_no_sign_boto3_uses_unsigned_config(monkeypatch, tmp_path):
    pytest.importorskip("botocore")  # UNSIGNED sentinel comes from botocore
    monkeypatch.setenv("BOXY_S3_BACKEND", "boto3")
    cap = {}
    fake = types.ModuleType("boto3")

    def client(_n, endpoint_url=None, config=None):
        cap["config"] = config
        return _FakeS3Client({"models/x/config.json": b"{}"})

    fake.client = client
    monkeypatch.setitem(sys.modules, "boto3", fake)
    # provide a real botocore UNSIGNED sentinel
    s3.stage_model("s3://public/models/x", str(tmp_path), no_sign=True)
    from botocore import UNSIGNED
    assert cap["config"] is not None and cap["config"].signature_version is UNSIGNED


def test_container_backend_apptainer_calls_aws_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s")
    monkeypatch.setenv("BOXY_S3_BACKEND", "container")
    seen = {}
    monkeypatch.setattr(s3.subprocess, "run",
                        lambda cmd, *a, **k: seen.update(cmd=cmd) or __import__("types").SimpleNamespace(returncode=0))
    s3.stage_model("s3://bkt/prefix", str(tmp_path), runtime="apptainer")
    cmd = seen["cmd"]
    assert cmd[:2] == ["apptainer", "exec"]
    assert "docker://public.ecr.aws/aws-cli/aws-cli:latest" in cmd
    assert cmd[cmd.index("--bind") + 1] == f"{tmp_path / 'prefix'}:/dest"
    assert "aws" in cmd and "s3" in cmd and "sync" in cmd
