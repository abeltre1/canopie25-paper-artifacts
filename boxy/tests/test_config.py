"""Layered config resolution: CLI flag > env > file > default. The flag layer
lives in cli.py; here we prove env > file > default and the file-discovery rules.
The autouse `_isolate_config` fixture (conftest) already points XDG at an empty
dir and resets the cache per test."""

import os

import pytest

from boxy import config


# ---- registry invariants --------------------------------------------------------


def test_setting_keys_and_env_names_are_unique():
    keys = list(config.SETTINGS)
    assert len(keys) == len(set(keys))
    envs = [s.env for s in config.SETTINGS.values()]
    assert len(envs) == len(set(envs)), "duplicate env var names in SETTINGS"


def test_every_default_survives_its_own_cast():
    # a str default fed back through the env cast must round-trip to the same type
    for s in config.SETTINGS.values():
        assert isinstance(s.cast(str(s.default)), type(s.default)), s.key


def test_legacy_env_names_preserved():
    # the pre-existing BOXY_* vars must keep their exact spelling (back-compat)
    by_env = {s.env for s in config.SETTINGS.values()}
    for legacy in ("BOXY_SSH_PERSIST", "BOXY_RELAY_NAMESPACE", "BOXY_OC", "BOXY_CHISEL",
                   "BOXY_SSH", "BOXY_REMOTE_COMMAND", "BOXY_JOBS_ROOT", "BOXY_STORE",
                   "BOXY_MODELS_DIR", "BOXY_AWSCLI_IMAGE"):
        assert legacy in by_env, legacy


# ---- precedence -----------------------------------------------------------------


def test_default_when_nothing_set():
    assert config.get("network.bind_host") == "0.0.0.0"
    assert config.get_int("network.ray_port") == 6379
    assert config.get_float("timeouts.readiness") == 180.0
    assert config.source("network.bind_host") == ("0.0.0.0", "default")


def test_env_overrides_default_and_is_typed(monkeypatch):
    monkeypatch.setenv("BOXY_RAY_PORT", "7000")
    assert config.get_int("network.ray_port") == 7000
    assert config.source("network.ray_port") == (7000, "env BOXY_RAY_PORT")


def _write_cfg(tmp_path, body):
    d = tmp_path / "boxy"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.toml").write_text(body)
    return tmp_path


def test_file_overrides_default_via_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_write_cfg(tmp_path, """
[network]
bind_host = "127.0.0.1"
ray_port = 6400
""")))
    config.reset()
    assert config.get("network.bind_host") == "127.0.0.1"
    assert config.get_int("network.ray_port") == 6400
    assert config.source("network.bind_host") == ("127.0.0.1", "file")


def test_env_beats_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_write_cfg(tmp_path, """
[network]
bind_host = "127.0.0.1"
""")))
    monkeypatch.setenv("BOXY_BIND_HOST", "10.0.0.5")
    config.reset()
    assert config.get("network.bind_host") == "10.0.0.5"
    assert config.source("network.bind_host")[1] == "env BOXY_BIND_HOST"


def test_boxy_config_explicit_path(monkeypatch, tmp_path):
    cfg = tmp_path / "custom.toml"
    cfg.write_text('[ssh]\ncontrol_persist = "4h"\n')
    monkeypatch.setenv("BOXY_CONFIG", str(cfg))
    config.reset()
    assert config.get("ssh.control_persist") == "4h"


def test_boxy_config_invalid_is_fatal_when_explicit(monkeypatch, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = not valid = toml =\n")
    monkeypatch.setenv("BOXY_CONFIG", str(bad))
    config.reset()
    with pytest.raises(ValueError, match="BOXY_CONFIG"):
        config.get("ssh.control_persist")


def test_discovered_bad_file_is_silently_skipped(monkeypatch, tmp_path):
    # a broken XDG file must NOT crash every command (unlike an explicit BOXY_CONFIG)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_write_cfg(tmp_path, "nonsense = = =\n")))
    config.reset()
    assert config.get("network.bind_host") == "0.0.0.0"  # falls back to default


def test_unknown_file_key_warns_not_fatal(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_write_cfg(tmp_path, """
[network]
bind_host = "127.0.0.1"
made_up_key = 3
""")))
    config.reset()
    assert config.get("network.bind_host") == "127.0.0.1"
    assert "unknown config keys" in capsys.readouterr().err


def test_file_type_mismatch_names_the_key(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(_write_cfg(tmp_path, """
[network]
ray_port = "not-a-number"
""")))
    config.reset()
    with pytest.raises(ValueError, match="network.ray_port"):
        config.get_int("network.ray_port")


def test_unregistered_key_is_programmer_error():
    with pytest.raises(KeyError):
        config.get("does.not.exist")


def test_render_template_covers_every_setting():
    tpl = config.render_template()
    for s in config.SETTINGS.values():
        assert s.env in tpl


# ---- back-compat: legacy env vars still move the consuming code -----------------


def test_legacy_jobs_root_env_still_moves_jobs_dir(monkeypatch, tmp_path):
    from boxy import jobs

    monkeypatch.setenv("BOXY_JOBS_ROOT", str(tmp_path / "j"))
    monkeypatch.delenv("BOXY_JOBS_DIR", raising=False)
    monkeypatch.setenv("BOXY_CLUSTER", "clusterX")
    assert jobs._dir() == tmp_path / "j" / "clusterX"


def test_legacy_ssh_persist_env_still_honored(monkeypatch):
    from boxy import remote

    monkeypatch.setenv("BOXY_SSH_PERSIST", "3h")
    assert remote.control_persist() == "3h"


def test_bind_host_reaches_engine_serve_cmd(monkeypatch, vllm_box, hops):
    from boxy import engines

    monkeypatch.setenv("BOXY_BIND_HOST", "127.0.0.1")
    cmd = engines.build_vllm_serve_cmd(vllm_box, hops, "/models/m")
    assert "--host=127.0.0.1" in cmd


def test_apply_bind_host_env_exports_flag(monkeypatch):
    from types import SimpleNamespace

    from boxy import cli

    monkeypatch.delenv("BOXY_BIND_HOST", raising=False)
    cli._apply_bind_host_env(SimpleNamespace(bind_host="0.0.0.0"))
    assert os.environ["BOXY_BIND_HOST"] == "0.0.0.0"


# ---- the `boxy config` subcommand ----------------------------------------------


def test_cli_config_shows_value_and_source(monkeypatch, capsys):
    from boxy.cli import main

    monkeypatch.setenv("BOXY_RAY_PORT", "7100")
    rc = main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "network.ray_port" in out and "7100" in out and "env BOXY_RAY_PORT" in out


def test_cli_config_init_emits_parseable_toml(capsys):
    import tomllib

    from boxy.cli import main

    rc = main(["config", "--init"])
    assert rc == 0
    tomllib.loads(capsys.readouterr().out)  # comment-only template must still parse
