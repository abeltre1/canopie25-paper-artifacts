"""The single seam between boxy and RamaLama.

ALL `ramalama` imports live in this module, and every import is lazy, so:
  * boxy works (dry-run, explicit locations) with no ramalama installed;
  * a RamaLama internals change breaks exactly one boxy file;
  * air-gapped bootstrap needs nothing beyond the stdlib.

RamaLama internals carry no API promise (plugins are `v1alpha`), so boxy pins
`ramalama==0.23.*` and treats the Protocols in ramalama/arg_types.py as the
input contract: args objects are duck-typed SimpleNamespaces, not argparse.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

# ramalama configures an INFO-level logger on import (proxy discovery etc.);
# keep boxy's output clean unless the user opts into verbosity.
_loglevel = os.environ.get("BOXY_RAMALAMA_LOGLEVEL", "WARNING").upper()
if _loglevel not in ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"):
    _loglevel = "WARNING"  # a verbosity knob must never crash boxy (finding 32)
logging.getLogger("ramalama").setLevel(_loglevel)

# ramalama's config prompts interactively on macOS when the podman machine has
# no GPU (applehv). boxy must never block on a prompt — a pull doesn't need a
# GPU. Users can still opt back in by exporting the variable themselves.
# (Field finding: Mac run-through, 2026-07.)
os.environ.setdefault("RAMALAMA_USER__NO_MISSING_GPU_PROMPT", "true")

DEFAULT_STORE = os.path.expanduser(os.environ.get("BOXY_STORE", "~/.local/share/boxy/store"))


def _silence_prompts() -> None:
    """Hard guarantee that RamaLama never blocks boxy on an interactive prompt.

    The env-var route (above) depends on RamaLama's config layering; field
    testing showed the macOS applehv prompt can still fire. Patching
    confirm_no_gpu at the seam is unconditional. (Field finding #8, 2026-07.)
    """
    try:
        import ramalama.common as _rc

        _rc.confirm_no_gpu = lambda name, provider: True
    except Exception:
        pass


def ramalama_available() -> bool:
    try:
        import ramalama.common  # noqa: F401

        return True
    except Exception:
        return False


# get_accel() speaks GPU-runtime dialect ("hip" for AMD, "cann" for Ascend);
# boxy's vocabulary is the platform name ("rocm", "ascend") — location.toml,
# image maps, and backend GPU args all use it. Normalize at the seam, or every
# v2 command dead-ends on a ROCm node with "unknown accelerator 'hip'".
_ACCEL_NORMALIZE = {"hip": "rocm", "cann": "ascend"}


def detect_accel() -> str:
    """GPU autodetect via ramalama.common.get_accel() (probes nvidia-smi, ROCm, ...).

    NOTE: get_accel() mutates os.environ (sets CUDA_VISIBLE_DEVICES etc.);
    that is desirable here — the backends read those vars for pass-through.
    Returns "none" when ramalama is unavailable or no accelerator is found.
    """
    try:
        _silence_prompts()
        from ramalama.common import get_accel
    except Exception:
        import shutil
        import sys

        if shutil.which("nvidia-smi") or shutil.which("rocm-smi"):
            print(
                "warning: a GPU tool is on PATH but the 'ramalama' package is not importable, "
                "so boxy cannot autodetect the accelerator (a CPU image would be chosen). "
                "Install boxy with the [ramalama] extra, or pass --accelerator explicitly.",
                file=sys.stderr,
            )
        return "none"
    try:
        accel = str(get_accel())
    except Exception as e:
        # malformed sysfs/KFD entries crash ramalama's probes on odd kernels;
        # autodetect must degrade, not traceback (finding 33)
        import sys

        print(f"warning: GPU autodetect failed ({e.__class__.__name__}: {e}); "
              f"assuming none — pass --accelerator to override", file=sys.stderr)
        return "none"
    # ramalama returns e.g. "cuda", "hip", "intel", "cann", ... or "none"
    accel = accel.split(":")[0] if accel else "none"
    return _ACCEL_NORMALIZE.get(accel, accel)


def accel_env_vars() -> dict[str, str]:
    """Accelerator visibility env vars (CUDA_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, ...)."""
    try:
        from ramalama.common import get_accel_env_vars, set_accel_env_vars

        set_accel_env_vars()
        return {k: str(v) for k, v in get_accel_env_vars().items()}
    except Exception:
        return {}


def gpu_device_paths() -> dict[str, str]:
    """Host GPU device nodes ({"dri": "/dev/dri", "kfd": "/dev/kfd", ...})."""
    try:
        from ramalama.common import get_gpu_devices

        return dict(get_gpu_devices())
    except Exception:
        return {}


def _store_args(model: str, dryrun: bool = False, quiet: bool = False) -> SimpleNamespace:
    """Duck-typed args satisfying ramalama's StoreArgType + pull needs.

    Required by transport_factory.New / Transport.ensure_model_exists:
    store, engine, container, MODEL, pull, quiet, verify, dryrun.
    engine=None/container=False: boxy owns the launch; ramalama only pulls.
    """
    return SimpleNamespace(
        store=DEFAULT_STORE,
        engine=None,
        container=False,
        MODEL=model,
        pull="missing",
        quiet=quiet,
        verify=True,
        dryrun=dryrun,
    )


# OS trust stores, in the order distros ship them. A TLS-inspecting proxy's
# root CA is installed HERE (so curl/yum/git work) but is NOT in certifi's
# bundle, which requests/huggingface_hub use — that split is why pulls fail
# with "unable to get local issuer certificate" on an otherwise-working login
# node. We discover the OS bundle and merge it with certifi.
_OS_CA_BUNDLES = (
    "/etc/pki/tls/certs/ca-bundle.crt",                    # RHEL / CentOS / Fedora
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  # RHEL (extracted)
    "/etc/ssl/certs/ca-certificates.crt",                 # Debian / Ubuntu / SLES
    "/etc/ssl/ca-bundle.pem",                             # openSUSE
    "/etc/ssl/cert.pem",                                  # Alpine / BSD / macOS
)


def discover_os_ca_bundle() -> str | None:
    """Return the first existing, non-empty OS CA bundle — the trust store the
    system's own tools use. Checked in distro order, then OpenSSL's compiled-in
    default as a fallback."""
    candidates = list(_OS_CA_BUNDLES)
    try:
        import ssl

        default = ssl.get_default_verify_paths().cafile
        if default:
            candidates.append(default)
    except Exception:
        pass
    for path in candidates:
        try:
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        except OSError:
            continue
    return None


def _merge_with_certifi(primary: str, note: str) -> str | None:
    """Write certifi's public CAs + `primary` into boxy's ca-merged.crt and point
    this process at it. Superset of both stores — never drops a trust anchor.
    `note` is the one-line explanation printed to stderr. Returns the merged
    path, or None if certifi is unavailable or the write fails."""
    import sys

    try:
        import certifi

        public = certifi.where()
    except Exception:
        return None
    try:
        merged = os.path.join(DEFAULT_STORE, "ca-merged.crt")
        os.makedirs(DEFAULT_STORE, exist_ok=True)
        with open(public, "rb") as f:
            bundle = f.read()
        with open(primary, "rb") as f:
            bundle += b"\n" + f.read()
        with open(merged, "wb") as f:
            f.write(bundle)
    except OSError as e:
        # diagnosing TLS must never introduce its own crash (finding 31)
        print(f"warning: could not build the merged CA bundle ({e})", file=sys.stderr)
        return None
    os.environ["SSL_CERT_FILE"] = merged
    print(f"{note} -> {merged}  (disable: BOXY_NO_CA_MERGE=1)", file=sys.stderr)
    return merged


def ensure_trust_bundle() -> str | None:
    """Repair the TLS trust store for pulls. Verified OpenSSL behaviors that
    bite HPC users (field findings #13, Mac run-through #3, eldorado 2026-07):

      * SSL_CERT_FILE REPLACES the trust store — a file holding only the
        site/proxy CA breaks every registry that is NOT intercepted with that
        CA (hf:// works, ollama:// fails, same shell);
      * a missing SSL_CERT_FILE path is SILENTLY ignored — everything fails;
      * certifi (used by requests/huggingface_hub) does NOT include a site's
        TLS-inspecting-proxy CA even when the OS trust store does, so pulls
        fail with "unable to get local issuer certificate" on a login node
        where curl works fine.

    Two paths, both merging with certifi so public + site roots coexist:
      1. SSL_CERT_FILE set  -> merge that site CA with certifi.
      2. SSL_CERT_FILE unset -> auto-discover the OS trust store and merge it,
         so boxy trusts exactly what the system's own tools trust.
    Opt out with BOXY_NO_CA_MERGE=1. Returns the merged path, or None."""
    import sys

    if os.environ.get("BOXY_NO_CA_MERGE"):
        return None
    site = os.environ.get("SSL_CERT_FILE")
    if not site:
        os_bundle = discover_os_ca_bundle()
        if not os_bundle:
            return None
        return _merge_with_certifi(
            os_bundle,
            f"tls: no SSL_CERT_FILE set — merged the OS trust store ({os_bundle}) with certifi's "
            f"public CAs so pulls trust the same roots as the system",
        )
    if not os.path.exists(site):
        print(
            f"warning: SSL_CERT_FILE={site} does not exist — OpenSSL silently ignores missing "
            f"paths, so ALL TLS verification will fail. Fix the path (ls -l \"$SSL_CERT_FILE\").",
            file=sys.stderr,
        )
        return None
    if os.path.isdir(site):
        print(f"warning: SSL_CERT_FILE={site} is a DIRECTORY — you probably meant SSL_CERT_DIR. "
              f"Point SSL_CERT_FILE at a PEM file.", file=sys.stderr)
        return None
    import stat

    if not stat.S_ISREG(os.stat(site).st_mode):
        # a FIFO with no writer blocks open() FOREVER (r2 audit)
        print(f"warning: SSL_CERT_FILE={site} is not a regular file — skipping the CA merge.",
              file=sys.stderr)
        return None
    if site.endswith("ca-merged.crt"):  # already ours (idempotent within a process)
        return site
    return _merge_with_certifi(
        site, f"tls: merged your SSL_CERT_FILE ({site}) with certifi's public CAs (site CA kept)")


class _LogTap(logging.Handler):
    """Capture RamaLama's WARNING+ log records during a pull. Its downloader
    logs the real error (e.g. the SSL failure) on every retry but then raises
    a FRESH ConnectionError('Download failed after multiple attempts') with no
    exception chain — without the tap, boxy's remedies never see the root
    cause on the ollama path. (Field finding: Mac run-through #2, 2026-07.)"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def _safetensors_tensor_names(path: str):
    """The tensor names declared in a safetensors file's header, read cheaply
    (8-byte little-endian header length + that many JSON bytes — no data, no
    dependency). None if the header can't be read (truncated/corrupt)."""
    import json
    import struct

    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            header = json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return None
    return {k for k in header if k != "__metadata__"}


def verify_safetensors_complete(model_dir: str) -> list[str]:
    """Cross-check a sharded-safetensors model dir against its index. Returns a
    list of problems (empty = OK / not a sharded-safetensors model). Catches the
    silent partial pull that makes vLLM crash with 'weights were not initialized
    from checkpoint' only after a multi-minute load (field: an incomplete
    RamaLama store copy shared across two clusters)."""
    import glob
    import json

    if not os.path.isdir(model_dir):
        return []
    indexes = glob.glob(os.path.join(model_dir, "*.safetensors.index.json"))
    if not indexes:
        return []  # single-file or non-safetensors model — nothing to cross-check
    try:
        weight_map = json.load(open(indexes[0])).get("weight_map", {})
    except (OSError, ValueError):
        return []
    if not weight_map:
        return []
    referenced = set(weight_map.values())
    present = {os.path.basename(p) for p in glob.glob(os.path.join(model_dir, "*.safetensors"))}
    missing_shards = sorted(referenced - present)
    if missing_shards:
        shown = ", ".join(missing_shards[:3]) + (" ..." if len(missing_shards) > 3 else "")
        return [f"{len(missing_shards)} of {len(referenced)} safetensors shard file(s) missing: {shown}"]
    expected = set(weight_map.keys())
    found: set[str] = set()
    for shard in sorted(referenced):
        names = _safetensors_tensor_names(os.path.join(model_dir, shard))
        if names is None:
            return [f"shard {shard} is truncated/corrupt (unreadable header)"]
        found |= names
    missing = sorted(expected - found)
    if missing:
        shown = ", ".join(missing[:3]) + (" ..." if len(missing) > 3 else "")
        return [f"{len(missing)} weight tensor(s) in the index are absent from the shards "
                f"(incomplete checkpoint): {shown}"]
    return []


def pull_model(model_uri: str, dryrun: bool = False, quiet: bool = False, force: bool = False) -> str:
    """Pull a model via RamaLama transports (hf://, ollama://, oci://, ...).

    Returns the resolved host path of the model inside boxy's store.
    Raises RuntimeError with guidance if ramalama is not installed.

    force=True removes any cached snapshot first, so a partial/corrupt copy
    (e.g. a download interrupted by an earlier TLS/network error, which then
    fails to load with 'weights were not initialized from checkpoint') is
    replaced by a clean re-pull.
    """
    try:
        _silence_prompts()
        from ramalama.transports.transport_factory import New
    except Exception as e:
        raise RuntimeError(
            "pulling transport URIs requires the 'ramalama' package "
            "(pip install 'boxy-hpc[ramalama]'); for air-gapped sites set "
            "box.model to a path on the shared filesystem instead"
        ) from e
    from boxy import policy

    policy.check_transport(model_uri)  # registry origin allowlist (env-controlled)
    if model_uri.startswith(("oci://", "docker://")):
        raise RuntimeError(
            f"{model_uri.split('://')[0]}:// models need a container engine to pull, which "
            "boxy's store-only pull path doesn't drive yet (SPEC roadmap). Use hf:// or "
            "ollama:// for the model weights, or pull the OCI artifact with podman/docker "
            "directly and serve the extracted GGUF by path."
        )
    ensure_trust_bundle()
    args = _store_args(model_uri, dryrun=dryrun, quiet=quiet)
    transport = New(model_uri, args)
    if force and not dryrun:
        args.ignore = True  # remove() must not raise when nothing is cached yet
        try:
            if transport.remove(args):
                print(f"boxy: removed cached {model_uri} — re-pulling clean", file=sys.stderr)
        except Exception as e:  # a failed cleanup must not block the re-pull
            print(f"boxy: could not remove the cached copy ({e}); pulling over it", file=sys.stderr)
    tap = _LogTap()
    ramalama_logger = logging.getLogger("ramalama")
    ramalama_logger.addHandler(tap)
    try:
        transport.ensure_model_exists(args)
    except Exception as e:
        raise RuntimeError(_pull_failure_message(model_uri, e, logged=tap.lines)) from e
    finally:
        ramalama_logger.removeHandler(tap)
    # use_container=False, should_generate=False => host blob/snapshot path.
    return transport._get_entry_model_path(False, False, dryrun)


def _mask_credentials(url: str) -> str:
    import re

    return re.sub(r"//[^/@]+@", "//<credentials>@", url)


def active_proxies() -> dict[str, str]:
    """The proxy map urllib ACTUALLY uses for every pull/probe (env vars; on
    macOS also the system proxy settings), credentials masked for display.
    Empty exports (the `https_proxy="${http_proxy}"`-before-http_proxy-is-set
    ordering bug) vanish here — so showing this map exposes them. Only the
    schemes OUR requests use (getproxies also surfaces docker_https_proxy-style
    vars from other tools)."""
    import urllib.request

    return {scheme: _mask_credentials(url) for scheme, url in sorted(urllib.request.getproxies().items())
            if scheme in ("http", "https", "all", "no")}


def raw_proxy_env(override: str = "") -> dict[str, str]:
    """The ACTUAL proxy vars to PROPAGATE into a job/container (unmasked, for
    real use — unlike active_proxies which masks for display). `override` (a
    single proxy URL, e.g. from --proxy) sets http+https; otherwise the current
    environment's proxy vars are captured. Emitted in BOTH cases so host tools
    (podman/curl read lower- or upper-case) and Python all see them; no_proxy is
    carried through so intra-cluster/localhost traffic stays direct."""
    out: dict[str, str] = {}
    if override:
        for name in ("http_proxy", "https_proxy"):
            out[name] = out[name.upper()] = override
        no = os.environ.get("no_proxy") or os.environ.get("NO_PROXY")
        if no:
            out["no_proxy"] = out["NO_PROXY"] = no
        return out
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        val = os.environ.get(name) or os.environ.get(name.upper())
        if val:
            out[name] = out[name.upper()] = val
    return out


def net_failure_kind(reason: object) -> str:
    """Classify a transport failure: 'dns' | 'proxy' | 'conn' | 'tls' | ''.
    The distinction matters because DNS is UPSTREAM of TLS — no certificate can
    fix a name that never resolved — and with a proxy configured a LOCAL
    dns/conn error is about the PROXY host (the proxy, not this machine,
    resolves and reaches the target)."""
    text = str(reason)
    if "CERTIFICATE_VERIFY_FAILED" in text:
        return "tls"
    if any(sig in text for sig in ("nodename nor servname", "Name or service not known",
                                   "Temporary failure in name resolution", "getaddrinfo failed",
                                   "No address associated with hostname")):
        return "dns"
    if "Tunnel connection failed" in text or "Proxy Authentication Required" in text:
        return "proxy"
    if any(sig in text for sig in ("Connection refused", "timed out", "Network is unreachable",
                                   "Connection reset by peer")):
        return "conn"
    return ""


_OFFLINE_ESCAPE = ("no-network escape: a LOCAL model file needs none of this — download the GGUF "
                   "on a machine that can, copy it over, then: boxy serve /path/to/model.gguf")


def network_remedy(kind: str) -> str:
    """One tailored remedy per net_failure_kind, shared by `boxy info --net`
    and the pull-failure path so both explain a failure the SAME way."""
    proxies = active_proxies()
    proxy_line = "  ".join(f"{k}={v}" for k, v in proxies.items())
    if kind == "dns":
        if proxies:
            return ("DNS failed WITH a proxy configured — the PROXY host itself did not resolve "
                    "(the proxy, not this machine, resolves the target). Check the value for a typo, "
                    f"and whether that host only resolves on the corporate network/VPN: {proxy_line}. "
                    + _OFFLINE_ESCAPE)
        return ("DNS failed and NO proxy is configured: this shell cannot resolve the registry at all "
                "(VPN down, or a corporate network that REQUIRES a proxy — export "
                "https_proxy/http_proxy=http://<proxy-host>:<port>, https_proxy is the one registries "
                "use). A cert in SSL_CERT_FILE cannot fix this — DNS happens before TLS. "
                + _OFFLINE_ESCAPE)
    if kind == "proxy":
        return ("the proxy REFUSED the tunnel (CONNECT): it requires authentication or blocks this "
                f"destination by policy. Active: {proxy_line or 'none'}. Credentials go in the URL: "
                "export https_proxy=http://user:pass@host:port. " + _OFFLINE_ESCAPE)
    if kind == "conn":
        if proxies:
            return (f"the TCP connection to the PROXY was refused/timed out (active: {proxy_line}) — "
                    "wrong port, or a proxy that is only reachable from inside the corporate "
                    "network/VPN. Unset the proxy vars when off-network. " + _OFFLINE_ESCAPE)
        return ("the TCP connection to the registry was refused/timed out — firewall, captive portal, "
                "or a network that requires a proxy (export https_proxy=...). " + _OFFLINE_ESCAPE)
    if kind == "tls":
        return ("TLS verification failed: the issuer of the certificate this shell SEES is not in your "
                "bundle. Interceptors (Zscaler, proxies) swap the chain per-host — and routing THROUGH "
                "a proxy can intercept hosts that were clean when direct. Append the interceptor's "
                "root CA to SSL_CERT_FILE (boxy merges it with certifi's public CAs; RUNBOOK §2.1).")
    return ""


def _pull_failure_message(model_uri: str, error: Exception, logged: list[str] | None = None) -> str:
    """Actionable message with the ROOT cause. RamaLama's repo-pull fallback
    masks the original URL error behind 'cli download not available'
    (NotImplementedError in its v0.23 HF transport), and its retrying
    downloader masks it behind a chain-less ConnectionError — so scan both
    the exception chain and the captured log lines.
    (Field findings: Mac run-throughs, 2026-07.)"""
    chain: list[str] = []
    seen = 0
    cursor: BaseException | None = error
    while cursor is not None and seen < 6:
        chain.append(str(cursor))
        cursor = cursor.__cause__ or cursor.__context__
        seen += 1
    distinct_logged = list(dict.fromkeys(logged or []))
    combined = " | ".join(chain + distinct_logged)
    msg = f"failed to pull {model_uri}: {chain[0].strip()}"
    if len(chain) > 1:
        msg += f"\n  root cause: {chain[-1]}"
    elif distinct_logged:
        msg += f"\n  root cause: {distinct_logged[0]}"
    if "CERTIFICATE_VERIFY_FAILED" in combined:
        if os.environ.get("SSL_CERT_FILE"):
            msg += (
                "\n  remedy: SSL_CERT_FILE is set but verification still failed. SSL_CERT_FILE"
                " REPLACES Python's trust store, so:\n"
                "    1. missing file? OpenSSL silently ignores bad paths:  ls -l \"$SSL_CERT_FILE\"\n"
                "    2. site-CA-only file? registries that are NOT intercepted by that CA fail"
                " (hf:// can work while ollama:// fails).\n"
                "       Fix: pip install certifi — boxy then merges public CAs with your site CA"
                " automatically (BOXY_NO_CA_MERGE=1 disables).\n"
                "  Diagnose per registry: boxy info --net"
            )
        else:
            msg += (
                "\n  remedy: your Python has no usable CA bundle (common with uv/standalone builds"
                " and TLS-intercepting proxies). Run:\n"
                "    pip install certifi && export SSL_CERT_FILE=$(python3 -m certifi)\n"
                "  or point SSL_CERT_FILE at your site's CA bundle."
                "\n  NOTE: an `export` only lives in that one shell — if this worked before, persist it:\n"
                "    echo 'export SSL_CERT_FILE=<path-to-ca.crt>' >> ~/.zshrc   # or ~/.bashrc,"
                " or your venv's bin/activate"
                "\n  Diagnose per registry: boxy info --net"
            )
    kind = net_failure_kind(combined)
    if kind and kind != "tls":  # tls has the richer SSL_CERT_FILE remedy above
        msg += "\n  remedy: " + network_remedy(kind)
    if "401" in combined and model_uri.startswith(("hf://", "huggingface://")):
        msg += _hf_401_diagnosis(model_uri)
    elif "404" in combined and model_uri.startswith(("hf://", "huggingface://")):
        msg += _hf_404_diagnosis(model_uri)
    if "cli download not available" in combined:
        msg += (
            "\n  note: RamaLama 0.23's HuggingFace full-repo CLI fallback is unimplemented;"
            " the direct download above is the path that must succeed."
        )
    return msg


def effective_hf_token() -> tuple[str | None, str]:
    """(token, source) — THE token an hf:// pull will send, resolved with
    RamaLama's own precedence (verified in its huggingface_token(), used by
    all five of its request paths): the HF_TOKEN env var WINS outright; the
    huggingface-cli cache file is never read while HF_TOKEN is set."""
    if "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"].strip() or None
        return token, "HF_TOKEN env var" if token else "HF_TOKEN env var (set but EMPTY — forces anonymous)"
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        try:
            with open(token_path) as f:
                return f.read().strip() or None, "~/.cache/huggingface/token (huggingface-cli login)"
        except OSError:
            pass
    return None, "not configured"


def _hf_token_sources() -> list[str]:
    """Where RamaLama's HF transport will find a token (its huggingface_token():
    HF_TOKEN env var first, then the huggingface-cli login cache)."""
    sources = []
    if os.environ.get("HF_TOKEN"):
        sources.append("the HF_TOKEN env var")
    if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
        sources.append("~/.cache/huggingface/token (huggingface-cli login)")
    return sources


def _hf_repo_info(repo: str) -> tuple[str | None, list[str]]:
    """Anonymous check via the public HF API. Returns (status, gguf_files):
    status 'public' | 'gated' | 'missing' | None (offline — no verdict);
    gguf_files = the .gguf filenames the repo actually contains."""
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}",
                                     headers={"User-Agent": "boxy"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.load(resp)
        files = [s.get("rfilename", "") for s in data.get("siblings", [])]
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        return ("gated" if data.get("gated") else "public"), ggufs
    except urllib.error.HTTPError as e:
        if e.code in (401, 404):
            return "missing", []
        return None, []
    except Exception:
        return None, []


def _hf_401_diagnosis(model_uri: str) -> str:
    """HF returns 401 for three unrelated causes: a stale token sent
    automatically (fails EVERY repo, even public), a nonexistent repo
    (anonymous requests never see 404), or a gated repo. Probe and give a
    verdict instead of a guessing list. (Field finding #16: three 401s in a
    row on the user's Mac, 2026-07.)"""
    repo = "/".join(model_uri.split("://", 1)[1].split("/")[:2])
    lines = []
    sources = _hf_token_sources()
    if sources:
        lines.append(
            f"  note: a HuggingFace token IS being sent (from {' and '.join(sources)}). A stale or\n"
            f"  revoked token makes HF return 401 for EVERY repo, even public ones. Retry without it:\n"
            f"      HF_TOKEN='' boxy serve {model_uri}"
        )
    status, _files = _hf_repo_info(repo)
    if status == "public":
        lines.append(
            f"  probe: {repo} EXISTS and is PUBLIC (anonymous API check just succeeded), so the 401\n"
            f"  is your token or proxy — the HF_TOKEN='' retry above should work."
        )
    elif status == "gated":
        lines.append(
            f"  probe: {repo} exists but is GATED — accept its license at\n"
            f"  https://huggingface.co/{repo} then export HF_TOKEN=<token from hf.co/settings/tokens>."
        )
    elif status == "missing":
        lines.append(
            f"  probe: an anonymous check of {repo} also failed — that repo does not exist under this\n"
            f"  exact name (or is private). Verify https://huggingface.co/{repo} in a browser;\n"
            f"  search huggingface.co for the model name to find the right owner/repo."
        )
    else:
        lines.append(
            f"  (could not reach the HF API to probe {repo} — check https://huggingface.co/{repo}\n"
            f"  in a browser: nonexistent repos get 401 anonymously; gated ones need HF_TOKEN.)"
        )
    return "\n  remedy: HuggingFace answered 401.\n" + "\n".join(lines)


def _hf_404_diagnosis(model_uri: str) -> str:
    """404 with working auth means the PATH is wrong. Quantizers name files
    unpredictably (TheBloke lowercases, bartowski doesn't), so list the GGUF
    files the repo ACTUALLY contains instead of making the user guess.
    (Field finding #17: three guessed repo/file names in a row, 2026-07.)"""
    parts = model_uri.split("://", 1)[1].split("/")
    repo = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    status, ggufs = _hf_repo_info(repo)
    if status in ("public", "gated") and filename:
        if ggufs:
            shown = ggufs[:12]
            listing = "\n      ".join(shown)
            more = "" if len(ggufs) <= 12 else f"\n      ... and {len(ggufs) - 12} more"
            return (
                f"\n  remedy: the repo {repo} exists, but has no file named {filename!r}.\n"
                f"  GGUF files it DOES contain:\n      {listing}{more}\n"
                f"  e.g.:  boxy serve hf://{repo}/{shown[0]}"
            )
        return (
            f"\n  remedy: the repo {repo} exists but contains no .gguf files — it is probably a\n"
            f"  safetensors repo (vLLM/GPU territory) or the wrong repo:\n"
            f"  https://huggingface.co/{repo}/tree/main"
        )
    if status == "missing":
        name = repo.split("/")[-1]
        return (
            f"\n  remedy: the repo {repo} does not exist. Find the right owner/repo:\n"
            f"      https://huggingface.co/models?search={name}\n"
            f"  — or skip repo-name guessing entirely: boxy serve ollama://<model-name>"
        )
    return (
        f"\n  remedy: not found — browse https://huggingface.co/{repo}/tree/main for the exact\n"
        f"  repo and filename (could not reach the HF API to list it for you)."
    )


def vllm_image_for(accelerator: str) -> str:
    """Default vLLM serving image per accelerator (mirrors VllmPlugin.get_container_image)."""
    return {
        "cuda": "vllm/vllm-openai:latest",
        "rocm": "rocm/vllm:latest",
        "intel": "intel/vllm:latest",
    }.get(accelerator, "vllm/vllm-openai:latest")


def _ramalama_vllm_image(accelerator: str) -> str | None:
    """Ask RamaLama's vLLM plugin for its accelerator->image mapping (deeper
    leverage than the static map); returns None when unavailable."""
    try:
        from ramalama.config import DefaultConfig
        from ramalama.plugins.loader import get_runtime

        # the plugin map is keyed by env-var names; "CUDA"/"HIP" silently fell
        # through to the CUDA-only default image on ROCm/Intel (finding 28)
        gpu_type = {"cuda": "CUDA_VISIBLE_DEVICES", "rocm": "HIP_VISIBLE_DEVICES",
                    "intel": "INTEL_VISIBLE_DEVICES"}.get(accelerator)
        if gpu_type is None:
            return None
        image = get_runtime("vllm").get_container_image(DefaultConfig(), gpu_type)
        # never accept a CUDA image for a non-CUDA accelerator (fallthrough guard)
        if accelerator != "cuda" and image and "vllm-openai:" in image:
            return None
        return image
    except Exception:
        return None


# llama.cpp on non-CUDA GPUs: the upstream ghcr server image is CPU-only, so a
# GGUF on a ROCm/Intel node would silently burn the allocation on CPU inference.
# Use RamaLama's accelerator images (llama-server on $PATH) for those.
_LLAMACPP_ACCEL_IMAGES = {
    "rocm": "quay.io/ramalama/rocm:latest",
    "intel": "quay.io/ramalama/intel-gpu:latest",
    "musa": "quay.io/ramalama/musa:latest",
    "ascend": "quay.io/ramalama/cann:latest",
    "vulkan": "quay.io/ramalama/ramalama:latest",
    "asahi": "quay.io/ramalama/asahi:latest",
}


def _ramalama_llamacpp_image(accelerator: str) -> str | None:
    """Ask RamaLama's llama.cpp plugin for its accelerator->image mapping
    (version-tagged quay.io/ramalama images); returns None when unavailable."""
    try:
        from ramalama.config import DefaultConfig
        from ramalama.plugins.loader import get_runtime

        gpu_type = {
            "rocm": "HIP_VISIBLE_DEVICES",
            "intel": "INTEL_VISIBLE_DEVICES",
            "musa": "MUSA_VISIBLE_DEVICES",
            "ascend": "ASCEND_VISIBLE_DEVICES",
            "vulkan": "GGML_VK_VISIBLE_DEVICES",
            "asahi": "ASAHI_VISIBLE_DEVICES",
        }.get(accelerator)
        if gpu_type is None:
            return None
        return get_runtime("llama.cpp").get_container_image(DefaultConfig(), gpu_type)
    except Exception:
        return None


def default_entrypoint(engine: str, image: str) -> str:
    """"" means "defer to the image ENTRYPOINT" (ghcr llama.cpp keeps its binary
    at /app/llama-server, off $PATH). RamaLama's images put llama-server ON
    $PATH but their ENTRYPOINT is not the server — name it explicitly there."""
    if engine == "llama.cpp" and image.startswith("quay.io/ramalama/"):
        return "llama-server"
    return ""


def default_image(engine: str, accelerator: str) -> str:
    """Default container image when a box omits `image`, per engine+accelerator.

    Defaults come from RamaLama's own plugin mappings when importable (falling
    back to static maps). llama.cpp keeps the field-tested upstream ghcr images
    for cuda/cpu; GPU accelerators map to RamaLama's accel images.
    """
    if engine == "llama.cpp":
        if accelerator == "cuda":
            return "ghcr.io/ggml-org/llama.cpp:server-cuda"
        if accelerator in _LLAMACPP_ACCEL_IMAGES:
            # static map is authoritative: RamaLama's 'auto' backend resolves
            # HIP/INTEL to the generic vulkan image on Linux (finding 29)
            return _LLAMACPP_ACCEL_IMAGES[accelerator]
        return "ghcr.io/ggml-org/llama.cpp:server"
    return _ramalama_vllm_image(accelerator) or vllm_image_for(accelerator)
