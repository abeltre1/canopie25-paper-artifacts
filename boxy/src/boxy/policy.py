"""Registry origin policy: which model transports boxy may pull from.

Default allowlist: Hugging Face and Ollama. Everything else — notably
ModelScope (modelscope.cn, operated by Alibaba from China) — is BLOCKED
unless the operator deliberately opts in via the BOXY_ALLOW_TRANSPORTS
environment variable. The policy is env-only ON PURPOSE: a TOML profile in
a repo must not be able to widen it silently; an export in your shell is an
auditable, deliberate act.

    export BOXY_ALLOW_TRANSPORTS=hf,ollama          # the default
    export BOXY_ALLOW_TRANSPORTS=hf,ollama,ms       # deliberate opt-in
"""

from __future__ import annotations

import os

# scheme -> (host, origin note). Schemes sharing a registry share a policy key.
REGISTRIES = {
    "hf": ("huggingface.co", "Hugging Face"),
    "huggingface": ("huggingface.co", "Hugging Face"),
    "ollama": ("registry.ollama.ai", "Ollama"),
    "ms": ("modelscope.cn", "ModelScope — operated by Alibaba (China)"),
    "modelscope": ("modelscope.cn", "ModelScope — operated by Alibaba (China)"),
    "rlcr": ("rlcr.io", "RamaLama container registry"),
    "oci": ("(arbitrary OCI registry)", "any OCI registry the URI names"),
}

# aliases normalize to one policy key
_CANONICAL = {"huggingface": "hf", "modelscope": "ms"}

DEFAULT_ALLOWED = ("hf", "ollama")


def _canonical(scheme: str) -> str:
    return _CANONICAL.get(scheme.lower(), scheme.lower())


def allowed_transports() -> tuple[str, ...]:
    raw = os.environ.get("BOXY_ALLOW_TRANSPORTS", "")
    if not raw.strip():
        return DEFAULT_ALLOWED
    return tuple(sorted({_canonical(s.strip()) for s in raw.split(",") if s.strip()}))


def check_transport(model_uri: str) -> None:
    """Raise with the registry's origin spelled out when a URI's transport is
    outside the allowlist. Local paths and file:// never hit this."""
    if "://" not in model_uri:
        return
    scheme = _canonical(model_uri.split("://", 1)[0])
    allowed = allowed_transports()
    if scheme in allowed:
        return
    host, origin = REGISTRIES.get(scheme, ("(unknown)", "unknown origin"))
    raise RuntimeError(
        f"transport '{scheme}://' pulls from {host} ({origin}), which is outside boxy's "
        f"registry allowlist [{', '.join(allowed)}].\n"
        f"  This is a deliberate origin-control default. To enable it anyway:\n"
        f"      export BOXY_ALLOW_TRANSPORTS={','.join(allowed)},{scheme}\n"
        f"  (env-only on purpose: a repo's TOML profile cannot widen the policy silently)"
    )


def registry_probes() -> list[tuple[str, str]]:
    """(scheme://, https URL) pairs for `boxy info --net` — allowed registries
    only; a blocked registry must not even be probed."""
    seen = set()
    probes = []
    for scheme in allowed_transports():
        host, _origin = REGISTRIES.get(scheme, (None, None))
        if not host or host in seen or host.startswith("("):
            continue
        seen.add(host)
        url = {"registry.ollama.ai": "https://registry.ollama.ai/v2/"}.get(host, f"https://{host}")
        probes.append((f"{scheme}://", url))
    return probes
