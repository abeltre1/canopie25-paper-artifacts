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
    ("AWS_ACCESS_KEY_ID", True),            # ends _ID — only the explicit list catches it
    ("TAILSCALE_AUTHKEY", True),            # no underscore before KEY — likewise
    ("LITELLM_MASTER_KEY", True),           # name-shape rule (_KEY)
    ("SITE_REGISTRY_PASSWORD", True),       # name-shape rule (_PASSWORD)
    ("MY_THING_TOKEN", True),
    ("SSL_CERT_FILE", False),               # path boxy sets itself — must stay readable
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


@pytest.mark.parametrize("shape", [
    "podman run --env HF_TOKEN={s} img",
    "podman run --env 'HF_TOKEN={s} trailing' img",
    'podman run --env "HF_TOKEN={s}" img',
    "bash -lc 'export HF_TOKEN={s}; vllm serve'",
    # nested shell escaping: `export K='"'"'secret'"'"'`. A quote-aware value
    # pattern used to match only a short prefix here and leave the token in the
    # output — this asserts the simpler `\\S*` pattern does not regress to that.
    """sh -c 'export HF_TOKEN='"'"'{s}'"'"'; vllm serve'""",
])
def test_key_pass_alone_never_leaks_across_quoting_shapes(shape, monkeypatch):
    """The KEY pass must stand on its own: with the secret ABSENT from the
    ambient environment the value pass cannot help, so any shape that survives
    here is a real disclosure."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    text = shape.format(s=SECRET)
    assert SECRET not in redact.redact_command(text), f"leaked in: {text}"


def test_value_pass_catches_non_assignment_positions(monkeypatch):
    """The VALUE pass is load-bearing, not decoration: a credential outside a
    KEY=VALUE assignment (boxy builds an Authorization header for the HF whoami
    check) is invisible to the key pass."""
    monkeypatch.setenv("HF_TOKEN", SECRET)
    header = f'curl -H "Authorization: Bearer {SECRET}" https://huggingface.co/api/whoami'
    assert SECRET in redact.redact_assignments(header)      # key pass alone cannot see it
    assert SECRET not in redact.redact_command(header)      # both passes together do


def test_redact_command_uses_ambient_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", SECRET)
    assert SECRET not in redact.redact_command(f"some --flag {SECRET} tail")
    # non-secret text is returned byte-for-byte
    plain = "podman run --rm --name=x image"
    assert redact.redact_command(plain) == plain


def test_short_values_are_not_masked_by_value(monkeypatch):
    """Masking by value must not scribble over ordinary words: a short value
    that happens to equal a command token would blank it. Such a value is not a
    credential anyway, and the KEY pass still covers it in assignments."""
    monkeypatch.setenv("HF_TOKEN", "podman")
    assert redact.redact_command("podman run --rm img") == "podman run --rm img"
    # ...while the same short value in an assignment IS still masked by key
    assert "podman" not in redact.redact_command("run --env HF_TOKEN=podman img").split("--env")[1]



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


def test_distributed_dryrun_never_prints_the_token(tmp_path, monkeypatch, capsys):
    """The multi-node Ray path echoes the HEAD and every WORKER command, which
    carry the container env. Redacting only the agentless script left this one
    open — a box profile with a token in [box.env] printed it in the clear.

    Written as an end-to-end assertion on stdout+stderr rather than against a
    specific print site, so a NEW echo added later is caught too."""
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    box = tmp_path / "box.toml"
    box.write_text(f'[box]\nname = "lt"\nimage = "img:1"\nengine = "vllm"\n'
                   f'model = "{model}"\n[box.env]\nHF_TOKEN = "{SECRET}"\n')
    loc = tmp_path / "loc.toml"
    loc.write_text('[location]\nname = "l"\nscheduler = "none"\naccelerator = "cuda"\n'
                   'runtime = "podman"\n[location.resources]\nnodes = 2\n'
                   'gpus_per_node = 4\ndistributed = true\n')
    from boxy.cli import main
    main(["serve", "--box", str(box), "--location", str(loc), "--here", "--dryrun"])
    cap = capsys.readouterr()
    assert "### Head" in cap.out, "expected the distributed path to run"
    assert SECRET not in cap.out and SECRET not in cap.err


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
