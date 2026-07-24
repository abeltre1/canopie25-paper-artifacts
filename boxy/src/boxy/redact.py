"""Secret redaction for anything boxy PRINTS.

boxy forwards the user's credentials (HF token for gated repos, S3 keys) into
the container env, and on the agentless --ssh path those land in the batch
script it renders. That script is also ECHOED to the terminal so the user can
see exactly what will run — and terminal output travels: scrollback, `tee`d
logs, CI job output, pasted bug reports, screen shares, tickets.

So: the text boxy WRITES keeps real values (the job must authenticate); the
text boxy DISPLAYS runs through redact_command() first. The two are
deliberately separated — never redact on the way to the remote file, never
print without redacting.

Redaction is by KEY (an --env/-e/export assignment whose name looks secret)
AND by VALUE (the ambient secrets are masked wherever they appear, however
they got quoted). Over-redacting a display string is harmless; under-redacting
is a credential disclosure, so both passes run.
"""

from __future__ import annotations

import os
import re

MASK = "<redacted-by-boxy>"

# Exact env names boxy itself propagates or that carry site credentials.
SECRET_ENV_KEYS = frozenset({
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LITELLM_MASTER_KEY",
    "LITELLM_SALT_KEY",
    "VLLM_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "REGISTRY_AUTH_TOKEN",
    "S3_SECRET_ACCESS_KEY",
    "TAILSCALE_AUTHKEY",
})

# Name-shape fallback for keys we don't enumerate (site-specific spellings).
_SECRET_SUFFIX = re.compile(r"(?:_TOKEN|_SECRET|_PASSWORD|_PASSWD|_APIKEY|_API_KEY|_ACCESS_KEY)$",
                            re.IGNORECASE)

# Names that LOOK secret by shape but are paths/flags boxy sets itself — keeping
# them visible matters for debugging (a CA path is the thing you need to see).
_NOT_SECRET = frozenset({
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_DIR",
    "BOXY_SSH_KEY_PATH",
})

# KEY=VALUE as it appears in a rendered command: `--env K=V`, `-e K=V`, bare
# `K=V` (env prefixes / export lines). VALUE runs to the next unquoted space, or
# to the closing quote when quoted — shlex.join quotes anything with specials.
_ASSIGN_RE = re.compile(
    r"""(?P<pre>(?:--env[= ]|-e[= ]|\bexport\s+)?)      # optional flag/keyword
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)                 # env NAME
        =                                               # =
        (?P<val>'(?:[^']|'\\'')*'|"(?:\\.|[^"\\])*"|\S*) # quoted or bare value
    """,
    re.VERBOSE)


def is_secret_key(name: str) -> bool:
    """Does this env var name carry a credential? Exact list first, then the
    name-shape fallback, minus the known-benign paths."""
    key = (name or "").strip()
    if not key or key.upper() in _NOT_SECRET:
        return False
    return key.upper() in SECRET_ENV_KEYS or bool(_SECRET_SUFFIX.search(key))


def ambient_secret_values(env: dict[str, str] | None = None) -> set[str]:
    """The VALUES of every secret-looking var in `env` (default: this process's
    environment). These are masked wherever they appear in displayed text —
    including inside shell quoting that the key-based pass can't parse.

    Very short values are skipped: masking a 1-3 char string would scribble over
    unrelated text, and a credential that short isn't one."""
    src = os.environ if env is None else env
    return {v for k, v in src.items() if is_secret_key(k) and v and len(v) > 3}


def redact_values(text: str, values) -> str:
    """Mask literal secret VALUES anywhere in `text` (longest first, so a value
    that contains another is replaced whole)."""
    out = text
    for val in sorted({v for v in values if v}, key=len, reverse=True):
        out = out.replace(val, MASK)
    return out


def redact_assignments(text: str) -> str:
    """Mask the value of every secret-looking KEY=VALUE assignment in `text`,
    leaving the key (and everything else) readable."""
    def sub(m: re.Match) -> str:
        if not is_secret_key(m.group("key")):
            return m.group(0)
        return f"{m.group('pre')}{m.group('key')}={MASK}"

    return _ASSIGN_RE.sub(sub, text)


def redact_command(text: str, env: dict[str, str] | None = None) -> str:
    """THE function to call before printing any rendered command or script.
    Runs both passes: secret-looking assignments by key, then ambient secret
    values by literal match. Returns `text` unchanged when there is nothing to
    hide, so non-secret output is untouched byte-for-byte."""
    if not text:
        return text
    return redact_values(redact_assignments(text), ambient_secret_values(env))


def redact_lines(text: str, env: dict[str, str] | None = None) -> list[str]:
    """redact_command() split into lines — the shape the CLI echoes scripts in."""
    return redact_command(text, env).splitlines()
