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

Two passes, and BOTH are load-bearing:
  - by KEY, for `--env NAME=value` / `-e NAME=value` / `export NAME=value`.
    Catches the common case even when the value is unknown to us.
  - by VALUE, masking ambient secrets wherever they appear. This is the only
    pass that can catch a credential in a NON-assignment position, e.g. an
    `Authorization: Bearer <token>` header (boxy builds one for the HF whoami
    check), which the key pass cannot see.
"""

from __future__ import annotations

import os
import re

MASK = "<redacted-by-boxy>"

# Names that do NOT match _SECRET_SUFFIX and so must be listed to be caught at
# all. Keep this minimal — a name already covered by the suffix rule does not
# need an entry:
#   AWS_ACCESS_KEY_ID  — ends in _ID
#   TAILSCALE_AUTHKEY  — "AUTHKEY" has no underscore before KEY
SECRET_ENV_KEYS = frozenset({
    "AWS_ACCESS_KEY_ID",
    "TAILSCALE_AUTHKEY",
    # Also caught by the suffix rule, but listed for auditability: these are
    # the credentials boxy itself handles, so a reviewer can grep this file
    # and see them.
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
})

# Name-shape rule — this does most of the work, including for site-specific
# spellings we have never seen.
_SECRET_SUFFIX = re.compile(
    r"(?:_TOKEN|_SECRET|_PASSWORD|_PASSWD|_APIKEY|_API_KEY|_ACCESS_KEY|_KEY)$",
    re.IGNORECASE)

# KEY=VALUE as it appears in a rendered command: `--env K=V`, `-e K=V`, or a
# bare `K=V` (export lines, env prefixes, and assignments inside shell quoting
# — the bare form still matches those because `pre` is optional).
#
# The value is deliberately `\S*`, "up to the next whitespace", and NOT a
# quote-aware pattern. A quote-aware version was tried and measured WORSE:
# against nested shell escaping (`export K='"'"'secret'"'"'`) it matched a
# short prefix and left the real secret in the output. Simpler is both smaller
# and safer here — over-matching only masks a little extra display text, while
# under-matching is a credential disclosure.
_ASSIGN_RE = re.compile(
    r"""(?P<pre>(?:--env[= ]|-e[= ]|\bexport\s+)?)   # optional flag/keyword
        (?P<key>[A-Za-z_][A-Za-z0-9_]*)              # env NAME
        =
        (?P<val>\S*)                                 # value, to the next space
    """,
    re.VERBOSE)

# Below this length a "secret" is not a credential, and masking it by value
# would scribble over unrelated words in the displayed command (a value that
# happened to equal "podman" would blank the runtime name). Real credentials
# are far longer: an HF token is `hf_` + 34 chars, an AWS key id is 20.
_MIN_MASKABLE_VALUE = 8


def is_secret_key(name: str) -> bool:
    """Does this env var name carry a credential? Explicit list first, then the
    name-shape rule."""
    key = (name or "").strip()
    if not key:
        return False
    return key.upper() in SECRET_ENV_KEYS or bool(_SECRET_SUFFIX.search(key))


def ambient_secret_values(env: dict[str, str] | None = None) -> set[str]:
    """VALUES of every secret-looking var in `env` (default: this process's
    environment) that are long enough to mask safely — see
    _MIN_MASKABLE_VALUE."""
    src = os.environ if env is None else env
    return {v for k, v in src.items()
            if is_secret_key(k) and v and len(v) >= _MIN_MASKABLE_VALUE}


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
