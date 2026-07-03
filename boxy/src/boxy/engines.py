"""Inner-command builders: the command that runs *inside* the container.

boxy builds these itself (pure functions) rather than calling RamaLama's
VllmPlugin._cmd_serve, because that builder is impure — it resolves the model
through the RamaLama store, which the paper's shared-filesystem flow doesn't
use. For store-pulled models the resolved path flows in the same way.
"""

from __future__ import annotations

from boxy.box import Box
from boxy.location import Location

# One default per engine, used everywhere (resolve, engines, banners, bench).
# Sweep finding: three different llama.cpp defaults (8000/8080/8090) made the
# printed endpoint disagree with the port the server actually bound.
DEFAULT_PORTS = {"llama.cpp": 8090, "vllm": 8000}


def default_port(engine: str) -> int:
    return DEFAULT_PORTS.get(engine, 8000)


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _tack_on_last(cmd: list[str], extra: dict[str, object], style: str = "eq") -> list[str]:
    """Append args unless the user already set them (prototype rule from
    common_boxy.sh: 'If user has already set any of these args, don't tack
    them on (don't override the user).').

    style="eq" emits --key=value (vLLM style); style="space" emits
    --key value as two tokens (llama.cpp's llama-server style)."""
    present = {a.split("=", 1)[0] for a in cmd if a.startswith("--")}
    for key, value in extra.items():
        flag = _flag(str(key))
        if flag in present:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif style == "space":
            cmd += [flag, str(value)]
        else:
            cmd.append(f"{flag}={value}")
    return cmd


def tuning_for_engine(location: Location, engine: str) -> dict[str, object]:
    """Site tuning scoped to the engine that understands it. Flat keys are
    vLLM-only (the paper's prototype tuned vLLM on MI300a — llama-server
    exits 2 on unknown flags, burning the allocation); nested tables
    ([location.tuning.vllm] / [location.tuning."llama.cpp"]) select per
    engine and MERGE with the flat keys for vLLM (r2 audit: adding a
    llama.cpp table must not silently drop the flat vLLM tuning)."""
    tuning = location.tuning or {}
    nested = {k: v for k, v in tuning.items() if isinstance(v, dict)}
    flat = {k: v for k, v in tuning.items() if not isinstance(v, dict)}
    if engine == "vllm":
        return {**flat, **nested.get("vllm", {})}
    return dict(nested.get(engine, {}))


def build_serve_cmd(
    box: Box,
    location: Location,
    model_path: str,
    host: str = "0.0.0.0",
    port: int | None = None,
    extra_args: list[str] | None = None,
    parallelism: tuple[int, int] | None = None,
) -> list[str]:
    """Dispatch to the box's inference engine (box.engine). `parallelism` is a
    (tensor_parallel_size, pipeline_parallel_size) derived from the allocation
    geometry for multi-node distributed vLLM (ignored by llama.cpp)."""
    if box.engine == "llama.cpp":
        return build_llamacpp_serve_cmd(box, location, model_path, host, port, extra_args)
    return build_vllm_serve_cmd(box, location, model_path, host, port, extra_args, parallelism)


def build_llamacpp_serve_cmd(
    box: Box,
    location: Location,
    model_path: str,
    host: str = "0.0.0.0",
    port: int | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    """llama.cpp OpenAI-compatible server argv (`llama-server -m <model> ...`).

    An empty first element means "defer to the image's own ENTRYPOINT": the
    upstream ghcr.io/ggml-org/llama.cpp:server image keeps its binary at
    /app/llama-server, NOT on $PATH, so overriding the entrypoint by bare
    name fails under crun. (Field finding: Mac run-through, 2026-07.)
    """
    entrypoint = box.entrypoint  # "" => image ENTRYPOINT + args
    cmd = [entrypoint, "-m", model_path]
    cmd += list(extra_args or [])
    # user-supplied sources first (box.args, site tuning), THEN the defaults:
    # _tack_on_last skips flags already present, so this order is what makes
    # "user args always win" true for host/port too. (Sweep finding 59.)
    # An EXPLICIT --port flag outranks a profile's [box.args] port (r2 audit).
    box_args, tuning = dict(box.args), tuning_for_engine(location, "llama.cpp")
    if port is not None:
        box_args.pop("port", None)
        tuning.pop("port", None)
    cmd = _tack_on_last(cmd, box_args, style="space")
    cmd = _tack_on_last(cmd, tuning, style="space")
    resolved_port = port or (box.ports[0] if box.ports else default_port("llama.cpp"))
    cmd = _tack_on_last(cmd, {"host": host, "port": resolved_port}, style="space")
    return cmd


def build_vllm_serve_cmd(
    box: Box,
    location: Location,
    model_path: str,
    host: str = "0.0.0.0",
    port: int | None = None,
    extra_args: list[str] | None = None,
    parallelism: tuple[int, int] | None = None,
) -> list[str]:
    """`vllm serve <model> ...` argv, with box.args then location.tuning
    applied, then the defaults last (user-supplied args always win).

    `parallelism` = (tensor_parallel_size, pipeline_parallel_size) derived from
    the multi-node allocation geometry; tacked on last so a user/box/tuning value
    for either still wins."""
    entrypoint = box.entrypoint or "vllm"
    cmd = [entrypoint, "serve", model_path]
    cmd += list(extra_args or [])
    box_args, tuning = dict(box.args), tuning_for_engine(location, "vllm")
    if port is not None:  # explicit --port outranks a profile's [box.args] port
        box_args.pop("port", None)
        tuning.pop("port", None)
    cmd = _tack_on_last(cmd, box_args)
    cmd = _tack_on_last(cmd, tuning)
    # Multi-node distributed serving: derive tensor/pipeline parallelism from the
    # geometry (TP within a node, PP across nodes) and use the Ray backend. Only
    # for nodes>1 (the caller passes parallelism); a user/box/tuning value wins.
    if parallelism is not None:
        tp, pp = parallelism
        cmd = _tack_on_last(cmd, {"tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                                  "distributed_executor_backend": "ray"})
    # HPC default: model stores live on NFS/Lustre, where vLLM >= 0.24 auto-enables
    # the 'prefetch' safetensors strategy — which has misloaded checkpoints (field
    # report: 'weights were not initialized from checkpoint' on a standard
    # Llama-3.1-8B served from an NFS store). 'eager' reads each shard fully before
    # load and is vLLM's own recommendation for network filesystems. A user/box
    # value wins; disable entirely with BOXY_NO_VLLM_EAGER=1 (e.g. vLLM < 0.24,
    # which does not know this flag).
    import os

    if not os.environ.get("BOXY_NO_VLLM_EAGER"):
        cmd = _tack_on_last(cmd, {"safetensors_load_strategy": "eager"})
    resolved_port = port or (box.ports[0] if box.ports else default_port("vllm"))
    cmd = _tack_on_last(cmd, {"host": host, "port": resolved_port})
    return cmd


def parse_port_flag(argv: list) -> int | None:
    """LAST --port/--port= in an argv (argparse engines honor the last
    occurrence — probing the first missed a live server; r2 audit)."""
    found = None
    for i, arg in enumerate(argv):
        if arg == "--port" and i + 1 < len(argv) and str(argv[i + 1]).isdigit():
            found = int(argv[i + 1])
        elif isinstance(arg, str) and arg.startswith("--port=") and arg.split("=", 1)[1].isdigit():
            found = int(arg.split("=", 1)[1])
    return found


def serving_port(inner_cmd: list[str], box: Box) -> int:
    """The port the built command will actually serve on: the LAST --port in
    the argv wins (user extra args are honored by _tack_on_last), else the
    box's declared port, else the engine default. This is THE port for
    banners, readiness probes, and publishing. (Sweep findings 2/10/25/47/55.)"""
    from_cmd = parse_port_flag(inner_cmd)
    if from_cmd is not None:
        return from_cmd
    if box.ports:
        return box.ports[0]
    return default_port(box.engine)


def build_raw_cmd(box: Box, user_args: list[str], location: Location) -> list[str]:
    """`boxy run` passthrough: entrypoint + user args + tack-ons, mirroring
    the prototype's boxy-run-vllm.sh "$@" behavior. An empty entrypoint is
    the deferral sentinel, same as the serve path (sweep findings 44/52) —
    dropping it made the first user arg the container entrypoint."""
    style = "space" if box.engine == "llama.cpp" else "eq"
    cmd = [box.entrypoint] + list(user_args)
    cmd = _tack_on_last(cmd, box.args, style=style)
    cmd = _tack_on_last(cmd, tuning_for_engine(location, box.engine), style=style)
    return cmd
