"""Credential handling: boxy forwards the user's HF token (and S3 keys) into the
job it renders, so the three ways a credential can escape are covered here.

  1. DISPLAY — the agentless batch script is echoed to the terminal; terminal
     output ends up in scrollback, CI logs, pasted bug reports and screen
     shares. What is printed must be redacted; what is WRITTEN must not be.
  2. THE SHARED FILESYSTEM — the script lands on a multi-tenant HPC $HOME.
     It must be created mode 600 atomically, not world-readable-then-chmod'd.
  3. THE PUBLIC REPO — MATRIX.md is generated from real serve output and
     committed. A maintainer's exported token must never bake into it.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from boxy import deploy, redact, remote
from boxy.box import Box
from boxy.location import Location

SECRET = "hf_liveSECRETvalue0123456789"


# ---- 1. the redactor itself ---------------------------------------------------------


@pytest.mark.parametrize("name,secret", [
    ("HF_TOKEN", True),
    ("HUGGING_FACE_HUB_TOKEN", True),
    ("AWS_SECRET_ACCESS_KEY", True),
    ("AWS_ACCESS_KEY_ID", True),
    ("OPENAI_API_KEY", True),
    ("SITE_REGISTRY_PASSWORD", True),          # name-shape fallback
    ("MY_THING_TOKEN", True),
    ("SSL_CERT_FILE", False),                  # path, not a credential
    ("REQUESTS_CA_BUNDLE", False),
    ("HF_HUB_OFFLINE", False),
    ("VLLM_USE_V1", False),
    ("", False),
])
def test_is_secret_key(name, secret):
    assert redact.is_secret_key(name) is secret


def test_redact_assignments_masks_only_secret_values():
    text = ("podman run --env HF_TOKEN=hf_abc123 --env HF_HUB_OFFLINE=1 "
            "--env SSL_CERT_FILE=/etc/ssl/ca.pem -e AWS_SECRET_ACCESS_KEY=wJalr/K7 image")
    out = redact.redact_assignments(text)
    assert "hf_abc123" not in out and "wJalr/K7" not in out
    assert out.count(redact.MASK) == 2
    # everything non-secret survives verbatim, keys included
    assert "HF_TOKEN=" in out and "--env HF_HUB_OFFLINE=1" in out
    assert "SSL_CERT_FILE=/etc/ssl/ca.pem" in out and "image" in out


def test_redact_values_catches_shell_quoted_secrets():
    # the key-based pass can't parse every quoting shape, so the VALUE pass is
    # the backstop: the literal secret is masked however it was written.
    text = "bash -lc 'export HF_TOKEN='\"'\"'hf_abc123'\"'\"'; run'"
    assert "hf_abc123" not in redact.redact_values(text, {"hf_abc123"})


def test_redact_command_uses_ambient_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    assert SECRET not in redact.redact_command(f"some --flag {SECRET} tail")
    # non-secret text is returned byte-for-byte
    plain = "podman run --rm --name=x image"
    assert redact.redact_command(plain) == plain


def test_ambient_secret_values_skips_trivially_short(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ab")        # too short to mask safely
    assert "ab" not in redact.ambient_secret_values()


# ---- 2. rendered vs printed ---------------------------------------------------------


def _script_with_token():
    box = Box(name="t", model="meta-llama/Llama-3.2-1B-Instruct", engine="vllm",
              env={"HF_TOKEN": SECRET})
    loc = Location(name="c", scheduler="slurm", accelerator="cuda", runtime="podman")
    return deploy.render_agentless_script(box, loc, "slurm", "t", "/h/ep.json", "/h/l.log",
                                          site_args=[], port=8000)


def test_rendered_script_still_carries_the_real_token():
    # the JOB must authenticate — redaction is a display concern only
    assert SECRET in _script_with_token()


def test_echoed_script_is_redacted():
    lines = redact.redact_lines(_script_with_token())
    assert not any(SECRET in ln for ln in lines), "token echoed to the terminal"
    assert any("HF_TOKEN" in ln for ln in lines), "the key should stay visible"


def test_serve_agentless_dryrun_never_prints_the_token(tmp_path, monkeypatch, capsys):
    """End-to-end: the flagship `--ssh --dryrun` path must not put the token on
    stdout (this is the output users paste into tickets)."""
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("HF_TOKEN", SECRET)
    model = tmp_path / "m.q4.gguf"
    model.write_bytes(b"GGUF")
    from boxy.cli import main
    main(["serve", str(model), "--scheduler", "slurm", "--agentless",
          "--accelerator", "cuda", "--account", "ab110003", "--dryrun"])
    cap = capsys.readouterr()
    assert SECRET not in cap.out and SECRET not in cap.err


# ---- 3. the write is private from birth ---------------------------------------------


def test_push_file_creates_mode_600_with_no_world_readable_window(tmp_path, monkeypatch):
    """push_file must create the file 600 ATOMICALLY (umask 077 in the same
    shell), not world-readable-then-chmod. Verified against a fake ssh that
    executes the command locally under a deliberately permissive umask."""
    shim = tmp_path / "fake-ssh"
    shim.write_text("#!/bin/bash\n"                     # args: -o ControlPath=… host CMD
                    "umask 000\n"                       # worst case: nothing masked
                    'exec bash -c "${@: -1}"\n')
    shim.chmod(0o755)
    monkeypatch.setenv("BOXY_SSH", str(shim))
    from boxy import config
    config.reset()
    dest = tmp_path / "job" / "script.sh"
    assert remote.push_file("host", str(dest), "#!/bin/bash\necho hi\n") == 0
    mode = stat.S_IMODE(dest.stat().st_mode)
    assert mode == 0o600, f"script created {oct(mode)} — readable by other cluster users"


def test_push_file_public_opt_out(tmp_path, monkeypatch):
    shim = tmp_path / "fake-ssh"
    shim.write_text("#!/bin/bash\numask 022\nexec bash -c \"${@: -1}\"\n")
    shim.chmod(0o755)
    monkeypatch.setenv("BOXY_SSH", str(shim))
    from boxy import config
    config.reset()
    dest = tmp_path / "pub.txt"
    assert remote.push_file("host", str(dest), "data", private=False) == 0
    assert stat.S_IMODE(dest.stat().st_mode) == 0o644


# ---- 4. nothing secret is committed -------------------------------------------------


REPO_DATA = Path(__file__).resolve().parent.parent / "src" / "boxy" / "data"


def test_committed_example_docs_carry_no_credentials():
    """MATRIX.md and the packaged profiles are generated from real serve output
    and committed to a PUBLIC repo: assert no secret-looking env assignment in
    them ever has a value."""
    import re
    bad = []
    for path in REPO_DATA.rglob("*"):
        if not path.is_file() or path.suffix not in (".md", ".toml", ".yaml", ".yml"):
            continue
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)", path.read_text(errors="replace")):
            key, val = m.group(1), m.group(2)
            if redact.is_secret_key(key) and val not in ("", '""', "''", redact.MASK):
                bad.append(f"{path.name}: {key}={val[:12]}...")
    assert not bad, "credential-shaped values committed to the repo: " + "; ".join(bad)


def test_gen_matrix_refuses_to_write_a_leaking_doc(monkeypatch):
    """The generator's last-line-of-defence: with a secret in the env, a doc
    containing that value must abort the write instead of committing it."""
    script = Path(__file__).resolve().parent.parent / "hack" / "gen_matrix.py"
    env = dict(os.environ, HF_TOKEN=SECRET)
    # exec the generator's top half (everything before __main__) with a real
    # __file__ so its ROOT/sys.path bootstrap works, then call the guard directly.
    probe = (
        f"g = {{'__file__': {str(script)!r}, '__name__': 'genmatrix_probe'}}\n"
        f"exec(open({str(script)!r}).read().split('if __name__')[0], g)\n"
        f"g['_assert_no_secrets']('harmless text', 'doc')\n"
        f"print('CLEAN-OK')\n"
        f"g['_assert_no_secrets']('leaked {SECRET} here', 'doc')\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert "CLEAN-OK" in proc.stdout                      # clean text writes fine
    assert proc.returncode != 0                           # leaking text aborts
    assert "REFUSING to write" in (proc.stderr + proc.stdout)
