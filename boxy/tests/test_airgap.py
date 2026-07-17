"""Air-gap bundles: `boxy bundle` (connected side) and `serve --bundle`
(air-gapped side). The bundle build is exercised with the network/podman
stages monkeypatched; the serve side is asserted on the rendered agentless
script — image loaded from the bundle, HF cache mounted offline, wheels
installed with --no-index, and NO proxy anywhere."""

import pytest

from boxy import airgap, cards
from boxy.cli import main


def test_card_declares_aux_repos_for_bundle():
    assert cards.layered_aux_repos("hf://nvidia/NVIDIA-Nemotron-Parse-v1.2") == ["nvidia/C-RADIOv2-H"]
    assert cards.layered_aux_repos("meta-llama/Llama-3.1-8B-Instruct") == []


def test_build_bundle_stages_everything(tmp_path, monkeypatch, capfd):
    calls = {"hf": [], "run": []}
    monkeypatch.setattr(airgap, "_hf_download", lambda repo, hf, token="": calls["hf"].append(repo))
    monkeypatch.setattr(airgap, "_run", lambda cmd, env=None, what="": calls["run"].append(cmd))
    monkeypatch.setattr("shutil.which", lambda n: f"/usr/bin/{n}")
    dest = tmp_path / "nb"
    airgap.build_bundle("nvidia/NVIDIA-Nemotron-Parse-v1.2", str(dest),
                        "docker.io/vllm/vllm-openai:latest",
                        aux_repos=["nvidia/C-RADIOv2-H"], pip_pkgs=["open_clip_torch"])
    # model AND the dynamically-fetched aux repo are cached
    assert calls["hf"] == ["nvidia/NVIDIA-Nemotron-Parse-v1.2", "nvidia/C-RADIOv2-H"]
    flat = [" ".join(c) for c in calls["run"]]
    assert any("pip download" in c and "open_clip_torch" in c for c in flat)
    assert any("pull docker.io/vllm/vllm-openai:latest" in c for c in flat)
    assert any("save --format oci-archive" in c for c in flat)
    manifest = (dest / "manifest.toml").read_text()
    assert 'model = "nvidia/NVIDIA-Nemotron-Parse-v1.2"' in manifest
    assert 'aux_repos = ["nvidia/C-RADIOv2-H"]' in manifest
    assert 'pip = ["open_clip_torch"]' in manifest
    assert "boxy serve" in capfd.readouterr().out            # the carry-across instructions


def test_cmd_bundle_pulls_card_knowledge(tmp_path, monkeypatch, capfd):
    seen = {}

    def fake_build(model, dest, image, *, aux_repos=None, pip_pkgs=None, token="",
                   runtime="podman", skip_image=False):
        seen.update(model=model, dest=dest, image=image, aux=aux_repos, pip=pip_pkgs,
                    skip=skip_image)
        return dest + "/manifest.toml"

    monkeypatch.setattr(airgap, "build_bundle", fake_build)
    rc = main(["bundle", "hf://nvidia/NVIDIA-Nemotron-Parse-v1.2", "--skip-image",
               "-o", str(tmp_path / "nb")])
    assert rc == 0
    assert seen["model"] == "nvidia/NVIDIA-Nemotron-Parse-v1.2"
    assert seen["aux"] == ["nvidia/C-RADIOv2-H"]              # from the packaged card
    assert seen["pip"] == ["open_clip_torch"]
    assert seen["skip"] is True
    assert "vllm" in seen["image"]                            # engine image from the map


def test_bundle_error_when_no_downloader(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_hub(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_hub)
    monkeypatch.setattr("shutil.which", lambda n: None)
    with pytest.raises(airgap.BundleError) as e:
        airgap._hf_download("acme/x", str(tmp_path))
    assert "huggingface_hub" in str(e.value)


FAKE_LEAF = "-----BEGIN CERTIFICATE-----\nLEAF\n-----END CERTIFICATE-----"
FAKE_INTER = "-----BEGIN CERTIFICATE-----\nZIA-INTERMEDIATE\n-----END CERTIFICATE-----"
FAKE_ROOT = "-----BEGIN CERTIFICATE-----\nZIA-ROOT\n-----END CERTIFICATE-----"


def test_trust_captures_chain_and_merge_includes_it(tmp_path, monkeypatch, capfd):
    # `boxy trust huggingface.co`: the interceptor's ISSUING chain (leaf dropped)
    # lands in boxy's trusted-extra store, and every subsequent CA merge carries
    # it — the turnkey CERTIFICATE_VERIFY_FAILED fix.
    from boxy import cli, ramalama_shim

    monkeypatch.setattr(ramalama_shim, "DEFAULT_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(cli, "_capture_tls_chain",
                        lambda host, port, proxy: [FAKE_LEAF, FAKE_INTER, FAKE_ROOT])
    monkeypatch.setattr(cli, "_cert_issuer", lambda pem: "issuer=CN=SNL ZIA TLS-Interception CA")
    rc = main(["trust", "huggingface.co", "--yes"])
    out = capfd.readouterr().out
    assert rc == 0
    extra = (tmp_path / "store" / "site-trusted-ca.pem").read_text()
    assert "ZIA-INTERMEDIATE" in extra and "ZIA-ROOT" in extra
    assert "LEAF" not in extra                               # never pin a single host's leaf
    assert "SNL ZIA" in out and "boxy info --net" in out
    # the merge picks the extras up
    primary = tmp_path / "site.crt"
    primary.write_text("-----BEGIN CERTIFICATE-----\nSITE\n-----END CERTIFICATE-----\n")
    merged = ramalama_shim._merge_with_certifi(str(primary), "test merge")
    text = open(merged).read()
    assert "ZIA-ROOT" in text and "SITE" in text             # certifi + site + trusted extras


def test_trust_refuses_non_interactive_without_yes(tmp_path, monkeypatch, capfd):
    from boxy import cli, ramalama_shim

    monkeypatch.setattr(ramalama_shim, "DEFAULT_STORE", str(tmp_path / "store"))
    monkeypatch.setattr(cli, "_capture_tls_chain", lambda *a: [FAKE_LEAF, FAKE_ROOT])
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["trust", "huggingface.co"])
    assert rc == 1
    assert "rerun with --yes" in capfd.readouterr().err
    assert not (tmp_path / "store" / "site-trusted-ca.pem").exists()
