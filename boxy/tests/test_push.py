"""`boxy push` — HF model -> site S3 bucket / OCI registry artifact. Backends
are exercised against shims (fake aws / ramalama on PATH); the HF snapshot is
monkeypatched. Nothing touches the network."""

import os
import stat

import pytest

from boxy import push
from boxy.cli import main


def _shim(directory, name, body):
    p = directory / name
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_text("weights")
    monkeypatch.setattr(push, "snapshot", lambda mid, token="": str(d))
    return d


def test_push_s3_via_aws_cli(model_dir, tmp_path, monkeypatch, capfd):
    binp = tmp_path / "bin"
    binp.mkdir()
    log = tmp_path / "aws.log"
    _shim(binp, "aws", f"#!/bin/bash\necho \"$@\" >> {log}\n")
    monkeypatch.setenv("PATH", f"{binp}:{os.environ['PATH']}")
    monkeypatch.setenv("BOXY_S3_BACKEND", "awscli")
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.site.gov")
    rc = main(["push", "hf://acme/Tiny-1B", "s3://models/tiny"])
    out = capfd.readouterr().out
    assert rc == 0
    logged = log.read_text()
    assert "--endpoint-url https://s3.site.gov" in logged
    assert f"s3 cp --recursive {model_dir} s3://models/tiny" in logged
    assert "boxy serve s3://models/tiny" in out            # the serve-back hint


def test_push_oci_via_ramalama(model_dir, tmp_path, monkeypatch, capfd):
    binp = tmp_path / "bin"
    binp.mkdir()
    log = tmp_path / "rl.log"
    _shim(binp, "ramalama", f"#!/bin/bash\necho \"$@\" >> {log}\n")
    monkeypatch.setenv("PATH", f"{binp}:{os.environ['PATH']}")
    rc = main(["push", "acme/Tiny-1B", "oci://registry.site.gov/models/tiny:1.0"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "push hf://acme/Tiny-1B oci://registry.site.gov/models/tiny:1.0" in log.read_text()
    assert "boxy serve oci://registry.site.gov/models/tiny:1.0" in out


def test_push_rejects_unknown_destination(model_dir, capfd):
    rc = main(["push", "acme/Tiny-1B", "ftp://nope"])
    assert rc == 1
    assert "expected s3://" in capfd.readouterr().err


def test_push_oci_without_ramalama_is_actionable(model_dir, monkeypatch, capfd):
    monkeypatch.setattr("shutil.which", lambda n: None)
    rc = main(["push", "acme/Tiny-1B", "oci://reg/models/t:1"])
    assert rc == 1
    assert "ramalama" in capfd.readouterr().err
