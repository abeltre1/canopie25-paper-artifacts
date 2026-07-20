"""Layered configuration — the single place a hardcoded default can be overridden.

Resolution order (highest wins):

    CLI flag  >  environment variable  >  config file  >  built-in default

CLI flags stay in cli.py (`args.x or config.get("...")`); this module owns the
lower three layers. The config file is TOML, discovered at (first hit wins):

    1. $BOXY_CONFIG                       (explicit; a bad/missing file here is fatal)
    2. $XDG_CONFIG_HOME/boxy/config.toml  (default ~/.config/boxy/config.toml)

Design rules that keep the rest of the codebase (and its 550+ tests) working:
  * this module imports NOTHING from boxy — no import cycles, ever;
  * the environment is read PER CALL (so a test's monkeypatch of BOXY_* is
    honored without cache-busting); only the file parse is cached;
  * every setting declares its legacy env-var spelling explicitly, so the 24
    pre-existing BOXY_* vars keep their exact names;
  * unknown file keys warn (forward-compat), they don't crash.

Call `config.get("network.bind_host")` inside functions — never to initialize a
module-level constant (that would freeze the value at import and defeat layering).
The old module constants stay as the built-in DEFAULT, reached through the
Setting registry below.
"""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Setting:
    key: str                                   # dotted, e.g. "network.bind_host"
    env: str                                   # primary env var (legacy spelling preserved)
    default: Any
    cast: Callable[[str], Any] = str           # parse the env-var string with this
    help: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)  # extra legacy env spellings


# One entry per overridable constant. Grouped by section only for readability;
# the dict is flat and keyed by the dotted config key.
SETTINGS: dict[str, Setting] = {s.key: s for s in [
    # -- network ---------------------------------------------------------------
    Setting("network.bind_host", "BOXY_BIND_HOST", "0.0.0.0",
            help="host the engine/router binds inside the container. 0.0.0.0 is "
                 "required for the compute-node rendezvous and multi-node Ray; "
                 "set 127.0.0.1 only for a purely local, single-host serve."),
    Setting("network.ray_port", "BOXY_RAY_PORT", 6379, int,
            help="Ray head port for multi-node (distributed) vLLM serving."),
    Setting("network.replica_port_base", "BOXY_REPLICA_PORT_BASE", 8000, int,
            help="first port for --replicas fan-out (replica N binds base+N)."),
    Setting("network.proxy", "BOXY_PROXY", "",
            help="corporate proxy URL for the job's image/model pulls, propagated to the "
                 "compute node and, over --ssh, to the cluster. No default: the ambient "
                 "http(s)_proxy env applies; set BOXY_PROXY / [network].proxy for a site "
                 "proxy, or --proxy <url> per job."),

    # -- per-engine default ports ---------------------------------------------
    Setting("ports.vllm", "BOXY_PORT_VLLM", 8000, int,
            help="default listen port for the vLLM engine."),
    Setting("ports.llama_cpp", "BOXY_PORT_LLAMACPP", 8090, int,
            help="default listen port for the llama.cpp engine."),

    # -- container images ------------------------------------------------------
    Setting("relay.apps_domain", "BOXY_APPS_DOMAIN", "",
            help="the OpenShift apps domain share/relay URLs are minted under "
                 "(<name>.<apps domain>). Empty (the default) auto-discovers from the "
                 "logged-in cluster (oc ingress config, else api.->apps. off "
                 "`oc whoami --show-server`); set it to pin a domain."),
    Setting("images.relay", "BOXY_RELAY_IMAGE", "docker.io/jpillora/chisel:1.10.1",
            help="chisel image for the share relay CLIENT + OpenShift relay server "
                 "(one override mirrors both). Point BOXY_RELAY_IMAGE / [images].relay "
                 "at a site mirror when isolated/air-gapped compute nodes 403 on "
                 "Docker Hub."),
    Setting("images.flux_mcp", "BOXY_FLUX_MCP_IMAGE", "ghcr.io/converged-computing/flux-mcp:latest",
            help="flux-mcp image for the persistent OpenShift MCP service."),
    Setting("images.awscli", "BOXY_AWSCLI_IMAGE", "public.ecr.aws/aws-cli/aws-cli:latest",
            help="aws-cli image for the container S3 staging backend."),

    # -- timeouts (seconds) ----------------------------------------------------
    Setting("timeouts.readiness", "BOXY_READY_TIMEOUT", 180.0, float,
            help="how long `serve` waits for the endpoint to answer /v1 before giving up."),
    Setting("timeouts.route_admit", "BOXY_ROUTE_ADMIT_TIMEOUT", 10.0, float,
            help="how long the relay waits for an OpenShift Route to be admitted."),

    # -- ssh -------------------------------------------------------------------
    Setting("ssh.control_path", "BOXY_SSH_CONTROL_PATH", "~/.ssh/boxy-cm-%C",
            help="OpenSSH ControlPath for the multiplexed master (%C keeps it "
                 "under the ~104-char unix-socket limit)."),
    Setting("ssh.control_persist", "BOXY_SSH_PERSIST", "12h",
            help="idle lifetime of the ssh master + its tunnels (OpenSSH time "
                 "format: 30m, 8h, ...). One OTP/touch buys this much access."),
    Setting("ssh.server_alive_interval", "BOXY_SSH_ALIVE_INTERVAL", 30, int,
            help="ssh ServerAliveInterval (seconds) for the master."),

    # -- share relay -----------------------------------------------------------
    Setting("share.enabled", "BOXY_SHARE_ENABLED", True, _as_bool,
            help="whether `--share` publishes a team URL. Set false to turn team "
                 "sharing off (e.g. until the relay client is installed/approved); "
                 "the local tunnel and --route still work."),
    Setting("serve.auto_share", "BOXY_AUTO_SHARE", True, _as_bool,
            help="over --ssh, auto-publish a team URL for the served model without "
                 "typing --share (turnkey) — the alias is derived from the model's "
                 "instance name. Best-effort: no relay => it degrades quietly to the "
                 "local tunnel. --share <alias> pins a name; set false to opt out."),
    Setting("relay.namespace", "BOXY_RELAY_NAMESPACE", "boxy-relay",
            help="OpenShift namespace the share relay lives in."),
    Setting("relay.client_mode", "BOXY_RELAY_CLIENT_MODE", "auto",
            help="how the chisel relay CLIENT runs for `--share`: host (a chisel "
                 "binary on PATH) | container (run it in podman/docker/apptainer — "
                 "ZERO install on the laptop/login node) | auto (host if a chisel "
                 "binary is present, else container). Container mode needs only a "
                 "container runtime; point images.relay at a mirror for air-gapped sites."),
    Setting("relay.port_min", "BOXY_RELAY_PORT_MIN", 31000, int,
            help="low end of the per-share reverse-tunnel port range."),
    Setting("relay.port_max", "BOXY_RELAY_PORT_MAX", 32000, int,
            help="high end (exclusive) of the per-share reverse-tunnel port range."),

    # -- flux-mcp --------------------------------------------------------------
    Setting("mcp.flux_port", "BOXY_FLUX_MCP_PORT", 8089, int,
            help="HTTP/SSE port the flux-mcp service listens on."),

    # -- paths -----------------------------------------------------------------
    Setting("paths.jobs_root", "BOXY_JOBS_ROOT", "~/.local/share/boxy/jobs",
            help="base dir for job records/endpoints (partitioned per cluster). "
                 "Point at shared scratch on sites where $HOME is not on compute nodes."),
    Setting("paths.store", "BOXY_STORE", "~/.local/share/boxy/store",
            help="boxy's own store dir (merged CA bundle, staged models)."),
    Setting("paths.models_dir", "BOXY_MODELS_DIR", "./models",
            help="default destination for staged models."),
    Setting("paths.results_root", "BOXY_RESULTS_ROOT", "~/.local/share/boxy/results",
            help="base dir for persisted bench results (partitioned per cluster; "
                 "BOXY_RESULTS_DIR pins an exact dir)."),
    Setting("paths.datasets", "BOXY_DATASETS_DIR", "~/.local/share/boxy/datasets",
            help="cache for downloaded bench datasets (ShareGPT)."),

    # -- bench -----------------------------------------------------------------
    Setting("bench.backend", "BOXY_BENCH_BACKEND", "auto",
            help="benchmark load generator: auto (vllm-bench binary > vllm CLI > "
                 "container > synthetic), or pin one of synthetic|vllm-bench|"
                 "vllm-cli|vllm-container."),
    Setting("bench.seed", "BOXY_BENCH_SEED", 12345, int,
            help="dataset sampling seed for real backends (the paper's seed) — "
                 "same seed = same request mix, comparable runs."),
    Setting("bench.api_key", "BOXY_BENCH_API_KEY", "",
            help="Bearer token for benching a secured endpoint (e.g. a k8s/OpenShift "
                 "ingress fronting vLLM --api-key). Never written to results."),
    Setting("binaries.vllm_bench", "BOXY_VLLM_BENCH", "vllm-bench",
            help="the vllm-bench load-generator binary (name on PATH or full path); "
                 "boxy also looks in <paths.store>/bin. `boxy bench --fetch-backend` "
                 "downloads it."),
    Setting("urls.vllm_bench",
            "BOXY_VLLM_BENCH_URL",
            "https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-{arch}-linux-musl",
            help="download URL for the static vllm-bench binary ({arch} = x86_64|aarch64); "
                 "point at an internal mirror on air-gapped sites."),
    Setting("datasets.sharegpt_url",
            "BOXY_SHAREGPT_URL",
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
            "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
            help="where `--dataset sharegpt` downloads the ShareGPT corpus from "
                 "(cached in paths.datasets; pre-stage the file there when air-gapped)."),

    # -- mounts ----------------------------------------------------------------
    Setting("mounts.selinux_relabel", "BOXY_SELINUX_RELABEL", "auto",
            help="add ':z' to bind mounts for SELinux: auto (only when enforcing) "
                 "| always | never."),

    # -- site defaults (turnkey: fill account/partition/time when flags absent) --
    Setting("site.account", "BOXY_ACCOUNT", "",
            help="Slurm account / Flux bank for job submission. Empty => auto-discover "
                 "(site.account_command, then $SBATCH_ACCOUNT, then sacctmgr). --account wins."),
    Setting("site.account_command", "BOXY_ACCOUNT_COMMAND", "myaccounts",
            help="site command that prints the user's charge account(s) (the site: myaccounts). "
                 "boxy takes the first account-looking token. Set empty to skip it."),
    Setting("site.pick_account", "BOXY_PICK_ACCOUNT", "auto",
            help="when several charge accounts (ACCOUNT_IDs) are discovered and none was named, "
                 "show an interactive numbered menu to pick one: 'auto' (default) shows it "
                 "only on a TTY (else auto-picks the first / remembered), 'always' forces the "
                 "prompt, 'never' disables it (silent first-pick). --account / $ACCOUNT_ID / "
                 "site.account bypass the menu; --pick-account / --no-pick-account override."),
    Setting("site.partition", "BOXY_PARTITION", "",
            help="default Slurm partition / Flux queue. Empty => AUTO: boxy picks the "
                 "soonest-start GPU partitions from sinfo. Also accepts 'all' (every "
                 "partition), 'off' (the scheduler's own default), or a name/comma-list."),
    Setting("site.pick_partition", "BOXY_PICK_PARTITION", "auto",
            help="when 2+ partitions are available and none was named, show an interactive "
                 "menu to pick ONE (parallel to site.pick_account): 'auto' (default) prompts "
                 "only on a TTY (else keeps the soonest-start comma-list), 'always' forces "
                 "the prompt, 'never' disables it. --partition / --pick-partition / "
                 "--no-pick-partition override."),
    Setting("site.license", "BOXY_LICENSE", "",
            help="Slurm license(s) to request as `#SBATCH --license=<val>` (e.g. "
                 "'scratchfs:1' or 'scratchfs:1,pscratch:1'), for sites that gate filesystems "
                 "behind licenses. No default — set BOXY_LICENSE (or [site].license) on sites "
                 "that need one, or pass --license per job. --license wins."),
    Setting("serve.agentless_ssh", "BOXY_AGENTLESS_SSH", "true",
            help="over --ssh, serve a model with NOTHING installed on the HPC (no boxy/Python/"
                 "RamaLama): the laptop renders a self-contained podman batch script, submits + "
                 "polls it over SSH. Set false (or --delegate / BOXY_SSH_DELEGATE=1) to run the "
                 "cluster's own boxy instead (needed for --replicas/--distributed/--box)."),
    Setting("serve.agentless_prestage", "BOXY_AGENTLESS_PRESTAGE", "never",
            help="agentless --ssh only: on a truly ISOLATED compute node (no external network at "
                 "runtime) the engine can't pull an hf:// model or the container image. Pass "
                 "--prestage (or set this to 'auto'/'always') to PRE-STAGE both from the LOGIN "
                 "node (which has your SSH session's network + the forwarded proxy) onto the "
                 "shared filesystem, then serve the model by path — nothing installed on the "
                 "cluster. Default 'never': the compute node pulls the image/model itself, which "
                 "is FASTER (no up-front login-node download) and works whenever the node has "
                 "network — the common case. 'auto' stages a transport URI (hf://…); 'always' "
                 "also pre-pulls a path model's image."),
    Setting("serve.auto_unique", "BOXY_AUTO_UNIQUE", "true",
            help="when a live instance of the same model already exists, start an "
                 "independent instance instead of blocking (turnkey). Set false to "
                 "restore the strict singleton (re-run reports 'already submitted')."),
    Setting("site.default_time", "BOXY_DEFAULT_TIME", "1:00:00",
            help="default walltime when --time is absent (Slurm colon notation, e.g. "
                 "'1:00:00' = 1 h, '30:00' = 30 min; boxy converts it to Flux FSD). "
                 "NOTE: the scheduler KILLS the served job at the walltime, so raise this "
                 "for long serving sessions. Empty => the scheduler's own default."),
    Setting("site.gpu_directive", "BOXY_GPU_DIRECTIVE", "auto",
            help="how Slurm asks for GPUs (sites differ): 'auto' (default) uses the proven "
                 "--gpus-per-node=N and, IF a site rejects it at submit ('Invalid generic "
                 "resource (gres) specification'), auto-recovers by resubmitting with the "
                 "portable --gres=gpu:[type:]N (type probed from sinfo) — so a working cluster "
                 "is never changed pre-emptively. Pin 'gres'/'gpus'/'gpus-per-node'/'none' to "
                 "force one form (disables the self-heal)."),
    Setting("site.gpu_type", "BOXY_GPU_TYPE", "",
            help="GPU type token in the GRES request (e.g. 'a100', 'h100'): boxy emits gpu:<type>:N. "
                 "Empty => auto-detected from sinfo (or untyped). Find it with `sinfo -o %G`."),
    Setting("site.scheduler", "BOXY_SCHEDULER", "auto",
            help="scheduler for a cluster serve when --scheduler is absent: 'auto' detects the "
                 "LIVE control plane over --ssh — a reachable Flux broker wins (Flux runs the "
                 "machine; slurm-compat sbatch/sinfo shims proxy to it), else a real slurmctld / "
                 "sinfo => slurm. Not a guess from which binary exists. Pin 'slurm'/'flux'/'none' "
                 "if a cluster's primary differs; --scheduler always wins."),
    Setting("cardgen.gpu_class_gb", "BOXY_GPU_CLASS_GB", "80",
            help="GPU memory (GB) `boxy generate card` sizes against when deriving gpus / "
                 "min_vram_gb from a model's weights (default 80 = A100/H100-class; set 40 "
                 "for a 40GB-class site so cards spread across more GPUs)."),
    Setting("site.default_accelerator", "BOXY_DEFAULT_ACCELERATOR", "cuda",
            help="accelerator assumed for a GPU job submitted from a GPU-less login node "
                 "(where detection sees no device). --accelerator / a --location profile win."),

    # -- model cards -----------------------------------------------------------
    Setting("cards.autogen", "BOXY_CARD_AUTOGEN", "true",
            help="when no card matches a served model, GENERATE one deterministically "
                 "from its HuggingFace metadata (config.json + safetensors index) and "
                 "write it to the user cards dir — the name-size heuristic then only "
                 "fires when the Hub is unreachable, loudly labeled a guess. false "
                 "disables the Hub lookup (air-gapped sites; HF_HUB_OFFLINE=1 also skips it)."),

    # -- node hardware (the geometry solver's supply side) ---------------------
    Setting("site.gpus_per_node", "BOXY_GPUS_PER_NODE", 0, int,
            help="GPUs per compute node on the target system, for the model-card "
                 "geometry solver. 0 = take it from a system card matching the cluster, "
                 "else assume 4. A system card is the durable home for this."),
    Setting("site.gpu_vram_gb", "BOXY_GPU_VRAM_GB", 0, int,
            help="per-GPU memory (GB) on the target system, for the model-card geometry "
                 "solver (e.g. 140 for H200-class parts). 0 = system card, else the "
                 "cardgen.gpu_class_gb 80GB-class assumption."),

    # -- model storage on the cluster ------------------------------------------
    Setting("storage.model_dir", "BOXY_MODEL_DIR", "",
            help="cluster directory for the model cache (HF downloads) on --ssh serves. "
                 "Empty = auto-discover a big shared scratch FS ($SCRATCH, /sitescratch, "
                 "/pscratch, /scratch — first writable one with room) so multi-GB models "
                 "never land on the $HOME quota. Set to pin an exact path."),
    Setting("storage.min_free_gb", "BOXY_MIN_FREE_GB", 100, int,
            help="minimum free space (GB) a discovered scratch FS must have to be picked "
                 "for the model cache; below this the next candidate is tried (the best "
                 "one is still used, with a warning, when none clears the bar)."),

    # -- external binaries (test shims / site spellings) -----------------------
    Setting("binaries.ssh", "BOXY_SSH", "ssh", help="ssh binary."),
    Setting("binaries.oc", "BOXY_OC", "oc", help="OpenShift oc binary."),
    Setting("binaries.chisel", "BOXY_CHISEL", "chisel", help="chisel binary for the share relay."),
    Setting("binaries.remote_command", "BOXY_REMOTE_COMMAND", "boxy",
            help="how boxy is spelled on the remote cluster (for --ssh)."),
]}


# ---- file discovery + cache -----------------------------------------------------

_file_cache: dict[str, Any] | None = None


def _discover() -> tuple[Path | None, bool]:
    """Return (config_path_or_None, explicit). `explicit` is True when BOXY_CONFIG
    was set, so an unreadable/invalid file there is fatal rather than skipped."""
    explicit = os.environ.get("BOXY_CONFIG")
    if explicit:
        return Path(os.path.expanduser(explicit)), True
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    path = Path(xdg) / "boxy" / "config.toml"
    return (path if path.exists() else None), False


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in data.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _warn_unknown(flat: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(flat) - set(SETTINGS))
    if unknown:
        print(f"boxy: warning: {path}: unknown config keys ignored: {', '.join(unknown)}",
              file=sys.stderr)


def _load_file() -> dict[str, Any]:
    global _file_cache
    if _file_cache is not None:
        return _file_cache
    path, explicit = _discover()
    flat: dict[str, Any] = {}
    if path is not None:
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError) as e:
            if explicit:
                raise ValueError(f"BOXY_CONFIG={path}: {e}") from None
            data = {}
        else:
            flat = _flatten(data)
            _warn_unknown(flat, path)
    _file_cache = flat
    return flat


def _coerce(s: Setting, value: Any, path_hint: str) -> Any:
    """Coerce a TOML file value to the setting's declared type (type(default))."""
    want = type(s.default)
    if want is bool:
        return bool(value)
    try:
        if want is int:
            return int(value)
        if want is float:
            return float(value)
        if want is str:
            return str(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"config key {s.key!r} ({path_hint}): expected {want.__name__}, got {value!r}"
        ) from None
    return value


# ---- public accessors -----------------------------------------------------------


def get(key: str) -> Any:
    """Resolve `key` through env > file > default. Env is read per call (test
    shims work); the file is cached. KeyError means an unregistered key — a
    programmer error, surfaced immediately."""
    s = SETTINGS[key]
    for name in (s.env, *s.aliases):
        raw = os.environ.get(name)
        if raw is not None:
            return s.cast(raw)
    file_vals = _load_file()
    if key in file_vals:
        return _coerce(s, file_vals[key], "config file")
    return s.default


def get_str(key: str) -> str:
    return str(get(key))


def get_int(key: str) -> int:
    return int(get(key))


def get_float(key: str) -> float:
    return float(get(key))


def get_bool(key: str) -> bool:
    v = get(key)
    return _as_bool(v) if isinstance(v, str) else bool(v)


def source(key: str) -> tuple[Any, str]:
    """(value, provenance) where provenance is 'env <NAME>', 'file', or 'default'.
    Powers `boxy config` so a user can see why a value took effect."""
    s = SETTINGS[key]
    for name in (s.env, *s.aliases):
        raw = os.environ.get(name)
        if raw is not None:
            return s.cast(raw), f"env {name}"
    file_vals = _load_file()
    if key in file_vals:
        return _coerce(s, file_vals[key], "config file"), "file"
    return s.default, "default"


def reset() -> None:
    """Drop the cached file parse (tests re-point BOXY_CONFIG/XDG between cases)."""
    global _file_cache
    _file_cache = None


def render_template() -> str:
    """A commented starter config.toml covering every setting (for `boxy config --init`)."""
    by_section: dict[str, list[Setting]] = {}
    for s in SETTINGS.values():
        section = s.key.split(".", 1)[0]
        by_section.setdefault(section, []).append(s)
    lines = ["# boxy config — every value here is also overridable by its BOXY_* env",
             "# var (which wins) and by a CLI flag (which wins over that).", ""]
    for section in by_section:
        lines.append(f"[{section}]")
        for s in by_section[section]:
            leaf = s.key.split(".", 1)[1]
            for hl in _wrap(s.help):
                lines.append(f"# {hl}")
            lines.append(f"# env: {s.env}")
            lines.append(f"# {_toml_kv(leaf, s.default)}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _toml_kv(leaf: str, value: Any) -> str:
    key = leaf if leaf.isidentifier() else f'"{leaf}"'
    if isinstance(value, str):
        return f'{key} = "{value}"'
    if isinstance(value, bool):
        return f"{key} = {str(value).lower()}"
    return f"{key} = {value}"


def _wrap(text: str, width: int = 76) -> list[str]:
    if not text:
        return []
    words, line, out = text.split(), "", []
    for w in words:
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        out.append(line)
    return out
