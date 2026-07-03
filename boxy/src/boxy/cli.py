"""boxy CLI: one tool to deploy/serve containerized GenAI across HPC sites.

    boxy serve hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf

Everything (engine, image, runtime, port) is auto-resolved and printed;
every choice is overridable by a flag or a --box/--location TOML profile.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import socket
import subprocess
import sys

from boxy import __version__, ramalama_shim
from boxy.backends import BACKENDS
from boxy.box import TRANSPORT_SCHEMES, Box
from boxy.location import ACCELERATORS, Location
from boxy.schedulers import SCHEDULERS

NOT_IN_MVP = "not implemented in the MVP — see SPEC.md §8 (roadmap) for the phase that adds it"


def _add_common(parser: argparse.ArgumentParser, location_required: bool = True) -> None:
    parser.add_argument("--box", required=True, help="path to a box TOML definition")
    parser.add_argument("--location", required=location_required, help="path to a location TOML definition")
    parser.add_argument("--dryrun", action="store_true", help="print the command instead of executing it")


def _load(args: argparse.Namespace) -> tuple[Box, Location]:
    return Box.from_toml(args.box), Location.from_toml(args.location)


def _emit(deployment, dryrun: bool) -> int:
    from boxy import deploy

    for warning in deployment.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for prep in deployment.prepare_commands:
        print(f"### Prepare: {shlex.join(prep)}")
    print(f"### Running Command:\n    {shlex.join(deployment.command)}")
    if dryrun:
        return 0
    return deploy.execute(deployment)


def cmd_info(args: argparse.Namespace) -> int:
    print(f"boxy {__version__}")
    print(f"ramalama library: {'available' if ramalama_shim.ramalama_available() else 'not installed'}")
    print(f"accelerator: {ramalama_shim.detect_accel()}")
    runtimes = [name for name in BACKENDS if shutil.which(name)]
    print(f"container runtimes: {', '.join(runtimes) or 'none found'}")
    launchers = [name for name, probe in (("slurm", "srun"), ("flux", "flux")) if shutil.which(probe)]
    print(f"schedulers: {', '.join(launchers) or 'none found'}")
    return 0


def _resolve_or_load(args: argparse.Namespace):
    """v2 front door: positional MODEL with full auto-resolution; --box/--location
    profiles still honored. MODEL and --box are alternatives: when --box is
    given, positionals are extra engine args (the profile names the model)."""
    from boxy import resolve

    if args.box and args.model:
        # profile mode: the MODEL slot swallowed the first extra engine arg
        args.args = [args.model] + list(args.args)
        args.model = None
    if not args.model and not args.box:
        raise ValueError(
            f"usage: boxy {args.subcommand} MODEL   "
            f"(or: boxy {args.subcommand} --box box.toml [--location loc.toml])"
        )
    if args.box:
        box = Box.from_toml(args.box)
        if args.location:
            return box, Location.from_toml(args.location), []
        location, decisions = resolve.auto_location(
            runtime=args.runtime,
            scheduler=args.scheduler,
            accelerator=args.accelerator,
            gpus=args.gpus,
            nodes=args.nodes,
            here=args.here,
        )
        return box, location, decisions
    r = resolve.resolve(
        args.model,
        engine=args.engine,
        runtime=args.runtime,
        scheduler=args.scheduler,
        image=args.image,
        port=args.port,
        gpus=args.gpus,
        nodes=args.nodes,
        name=args.name,
        accelerator=args.accelerator,
        here=args.here,
        require_exists=not args.dryrun,
    )
    if args.location:  # location profile + bare model is a valid mix (cluster profiles)
        location = Location.from_toml(args.location)
        return r.box, location, r.decisions
    return r.box, r.location, r.decisions


def _save_profile(prefix: str, box, location) -> None:
    """Snapshot the resolved configuration to TOML profiles (reproducibility,
    air-gapped sites, code review of what will run)."""
    header = (
        "# written by `boxy --save-profile`: values autodetected on the node where\n"
        "# it ran — review accelerator/runtime before reusing on a different node.\n"
    )
    box_lines = [header + "[box]"]
    for key in ("name", "image", "engine", "entrypoint", "model", "workdir"):
        value = getattr(box, key)
        if value:
            box_lines.append(f'{key} = "{value}"')
    if box.ports:
        box_lines.append(f"ports = {box.ports}")
    loc_lines = [header + "[location]"]
    for key in ("name", "scheduler", "accelerator", "runtime", "registry"):
        value = getattr(location, key)
        if value:
            loc_lines.append(f'{key} = "{value}"')
    if location.offline:
        loc_lines.append("offline = true")
    loc_lines += ["[location.resources]", f"nodes = {location.resources.nodes}",
                  f"gpus_per_node = {location.resources.gpus_per_node}"]
    with open(f"{prefix}.box.toml", "w") as f:
        f.write("\n".join(box_lines) + "\n")
    with open(f"{prefix}.location.toml", "w") as f:
        f.write("\n".join(loc_lines) + "\n")
    print(f"profiles written: {prefix}.box.toml, {prefix}.location.toml")


def _inside_allocation() -> bool:
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("FLUX_ENCLOSING_ID")
                or os.environ.get("FLUX_JOB_ID"))


def _detachable(deployment) -> bool:
    """Detach only where a runtime daemon owns the container lifetime: laptop /
    workstation podman|docker with no scheduler in play. Inside an allocation
    the job step must own the server (epilog would reap a daemonized one), and
    under srun/flux wrap `-d` would end the job step immediately."""
    return (deployment.location.scheduler == "none"
            and not _inside_allocation()
            and deployment.command[:2] and deployment.command[1] == "run"
            and deployment.command[0] in ("podman", "docker"))


def _container_exists(runtime: str, name: str) -> bool:
    result = subprocess.run([runtime, "inspect", "--format", "{{.Id}}", name],
                            capture_output=True, text=True)
    return result.returncode == 0


def _container_running(runtime: str, name: str) -> bool:
    result = subprocess.run([runtime, "inspect", "--format", "{{.State.Running}}", name],
                            capture_output=True, text=True)
    return result.returncode == 0 and "true" in result.stdout


def _dump_logs(runtime: str, name: str, tail: int = 50) -> None:
    result = subprocess.run([runtime, "logs", "--tail", str(tail), name],
                            capture_output=True, text=True)
    for stream in (result.stdout, result.stderr):
        if stream.strip():
            print(stream.rstrip(), file=sys.stderr)


def cmd_serve(args: argparse.Namespace) -> int:
    from boxy import deploy, readiness

    box, location, decisions = _resolve_or_load(args)
    for line in decisions:
        print(f"  auto: {line}")
    deployment = deploy.plan_serve(box, location, port=args.port, extra_args=args.args, dryrun=args.dryrun)
    if getattr(args, "save_profile", None):
        _save_profile(args.save_profile, deployment.box, deployment.location)

    port = args.port or (deployment.box.ports[0] if deployment.box.ports else 8000)
    url = f"http://127.0.0.1:{port}"
    runtime_bin = deployment.command[0]
    cname = deployment.box.name
    detach = _detachable(deployment) and not args.foreground and not args.dryrun and args.model
    if detach:
        if _container_exists(runtime_bin, cname):
            raise RuntimeError(
                f"a container named {cname!r} already exists — already serving this model? "
                f"Stop it with `boxy stop {cname}`, or serve under a different --name."
            )
        if "--rm" in deployment.command:
            # keep the container after a crash so its logs stay inspectable;
            # `boxy stop` removes it.
            deployment.command.remove("--rm")
        deployment.command.insert(2, "-d")  # runtime-managed daemon; boxy waits for readiness

    for warning in deployment.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for prep in deployment.prepare_commands:
        print(f"### Prepare: {shlex.join(prep)}")
    print(f"### Running Command:\n    {shlex.join(deployment.command)}")
    if args.dryrun:
        return 0

    if not detach:
        if deployment.location.scheduler == "none":
            host = socket.gethostname() if _inside_allocation() else "127.0.0.1"
            print(f"### Endpoint (once the model loads): http://{host}:{port}/v1")
            if _inside_allocation():
                print(f"###   from your workstation: ssh -L {port}:{host}:{port} <login-node>")
        return deploy.execute(deployment)

    rc = deploy.execute(deployment)  # returns immediately (-d)
    if rc != 0:
        return rc
    print(f"### Waiting for readiness at {url}/v1/models ...")
    try:
        model_id = readiness.wait_ready(
            url, timeout_s=args.ready_timeout,
            still_alive=lambda: _container_running(runtime_bin, cname),
        )
    except RuntimeError:
        print(f"boxy: server exited during startup; last log lines from {cname}:", file=sys.stderr)
        _dump_logs(runtime_bin, cname)
        subprocess.run([runtime_bin, "rm", "-f", cname], capture_output=True)
        return 1
    if model_id is None:
        print(f"boxy: endpoint not ready within {args.ready_timeout:.0f}s — the container is still "
              f"running (large models load slowly). Last log lines:", file=sys.stderr)
        _dump_logs(runtime_bin, cname)
        print(f"  follow: {runtime_bin} logs -f {cname}\n  stop:   boxy stop {cname}", file=sys.stderr)
        return 1
    print(f"### READY  {url}/v1   (model: {model_id})")
    print(f"###   try:  curl -s {url}/v1/models")
    print(f"###   stop: boxy stop {cname}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from boxy import deploy

    box, location = _load(args)
    # argparse.REMAINDER keeps the literal "--" separator; it must not reach
    # the container command.
    user_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    deployment = deploy.plan_run(box, location, user_args, dryrun=args.dryrun)
    return _emit(deployment, args.dryrun)


def cmd_pull(args: argparse.Namespace) -> int:
    """Pre-stage a model (login-node flow: pull where the network is, serve
    where the GPUs are — the store defaults to shared $HOME)."""
    model = args.model
    if not model and args.box:
        box = Box.from_toml(args.box)
        if not box.model:
            print(f"box {box.name}: no model set", file=sys.stderr)
            return 1
        model = box.model
    if not model:
        raise ValueError("usage: boxy pull MODEL   (or: boxy pull --box box.toml)")
    if not model.startswith(TRANSPORT_SCHEMES):
        print(f"model is a path ({model}); nothing to pull (shared-FS flow)")
        return 0
    path = ramalama_shim.pull_model(model, dryrun=args.dryrun)
    print(f"model available at: {path}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from boxy import deploy
    from boxy.backends import get_backend

    box, location = _load(args)
    backend = get_backend(location.resolve_runtime())
    prepare = backend.prepare(box, location, args.dryrun)
    if not prepare:
        print(f"runtime {backend.name}: uses {backend.image_format} images directly; nothing to build")
        return 0
    for prep in prepare:
        print(f"### Build: {shlex.join(prep)}")
    if args.dryrun:
        return 0
    import subprocess

    for prep in prepare:
        result = subprocess.run(prep)
        if result.returncode != 0:
            return result.returncode
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from boxy import sky_export

    if args.format != "sky":
        print(f"boxy generate: unknown format {args.format!r} (available: sky)", file=sys.stderr)
        return 1
    box, location = _load(args)
    if location.scheduler != "none":
        print(
            f"warning: location {location.name!r} uses scheduler={location.scheduler!r}; the SkyPilot "
            "path targets cloud/K8s — boxy serves Slurm/Flux natively (use `boxy serve`)",
            file=sys.stderr,
        )
    yaml_text = sky_export.to_sky_task(box, location, port=args.port, serve=args.serve)
    if args.output:
        with open(args.output, "w") as f:
            f.write(yaml_text)
        print(f"wrote {args.output}  (launch: sky {'serve up' if args.serve else 'launch'} {args.output})")
    else:
        print(yaml_text, end="")
    return 0


def _container_runtime(location: Location | None) -> str:
    if location is not None and location.runtime:
        if location.runtime == "apptainer":
            raise RuntimeError(
                "apptainer runs are foreground in the MVP: Ctrl-C the process, "
                "or cancel the job (scancel / flux cancel)"
            )
        return location.runtime
    for candidate in ("podman", "docker"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("no container runtime found on host (looked for podman, docker)")


def _run_or_print(cmd: list[str], dryrun: bool) -> int:
    print(f"### Running Command:\n    {shlex.join(cmd)}")
    if dryrun:
        return 0
    import subprocess

    return subprocess.run(cmd).returncode


def cmd_stop(args: argparse.Namespace) -> int:
    if args.name:
        target = args.name
    elif args.box:
        target = Box.from_toml(args.box).name
    else:
        raise ValueError("usage: boxy stop NAME   (names are printed at serve time and by `boxy list`)")
    location = Location.from_toml(args.location) if args.location else None
    runtime = args.runtime or _container_runtime(location)
    rc = _run_or_print([runtime, "stop", target], args.dryrun)
    if rc == 0 and not args.dryrun:
        # detached serves drop --rm so crash logs survive; clean up here
        subprocess.run([runtime, "rm", target], capture_output=True)
    return rc


def cmd_list(args: argparse.Namespace) -> int:
    location = Location.from_toml(args.location) if args.location else None
    runtime = args.runtime or _container_runtime(location)
    return _run_or_print([runtime, "ps", "--filter", "label=boxy.box"], args.dryrun)


def cmd_bench(args: argparse.Namespace) -> int:
    from boxy import bench

    box = Box.from_toml(args.box)
    url = args.url or f"http://127.0.0.1:{box.ports[0] if box.ports else 8000}"
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")] if args.batch_sizes else bench.DEFAULT_BATCH_SIZES
    if args.dryrun:
        print(f"### Bench plan: url={url} batch_sizes={batch_sizes} max_tokens={args.max_tokens} "
              f"dataset={args.dataset or 'synthetic'}")
        return 0
    report = bench.run_bench(url, batch_sizes, max_tokens=args.max_tokens, dataset=args.dataset)
    if args.json:
        print(report.to_json())
    else:
        print(f"# model={report.model} url={report.url} max_tokens={report.max_tokens}")
        print(f"{'batch':>6} {'ok':>4} {'err':>4} {'req/s':>8} {'tok/s':>9} {'p50 ms':>9} {'p95 ms':>9}")
        for r in report.results:
            print(f"{r.batch_size:>6} {r.ok:>4} {r.errors:>4} {r.requests_per_s:>8.2f} "
                  f"{r.tokens_per_s:>9.1f} {r.latency_p50_ms:>9.1f} {r.latency_p95_ms:>9.1f}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(report.to_csv())
        print(f"wrote {args.output}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    from boxy import cloud

    box, location = _load(args)
    if location.scheduler != "none":
        print(
            f"warning: location {location.name!r} uses scheduler={location.scheduler!r}; "
            "`boxy launch` delegates to SkyPilot (cloud) — use `boxy serve` for Slurm/Flux",
            file=sys.stderr,
        )
    if args.down:
        cmd = cloud.launch_command(box, "", serve=args.serve, down=True)
    else:
        yaml_path = cloud.write_task_yaml(box, location, args.port, args.serve, output=args.output)
        print(f"### Task YAML: {yaml_path}")
        cmd = cloud.launch_command(box, yaml_path, serve=args.serve)
    print(f"### Running Command:\n    {shlex.join(cmd)}")
    if args.dryrun:
        return 0
    cloud.ensure_sky()
    import subprocess

    return subprocess.run(cmd).returncode


def _stub(name: str):
    def handler(args: argparse.Namespace) -> int:
        print(f"boxy {name}: {NOT_IN_MVP}", file=sys.stderr)
        return 2

    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boxy", description=__doc__)
    parser.add_argument("--version", action="version", version=f"boxy {__version__}")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("info", help="show detected accelerator, runtimes, and schedulers")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser(
        "serve",
        help="serve MODEL as an OpenAI-compatible endpoint (engine/image/runtime/port auto-resolved)",
    )
    p.add_argument("model", nargs="?", default=None,
                   help="local path or transport URI (hf://, ollama://, oci://); alternative: --box")
    p.add_argument("--box", default=None, help="serve from a box TOML profile instead of MODEL")
    p.add_argument("--location", default=None, help="site TOML profile (scheduler/runtime/accelerator/tuning)")
    p.add_argument("--engine", choices=["llama.cpp", "vllm"], default=None,
                   help="inference engine (default: inferred — GGUF/ollama -> llama.cpp, else vLLM)")
    p.add_argument("--runtime", choices=["podman", "docker", "apptainer"], default=None,
                   help="container runtime (default: first WORKING one found)")
    p.add_argument("--scheduler", choices=["none", "slurm", "flux"], default=None,
                   help="submit as a job via this scheduler (never done automatically)")
    p.add_argument("--accelerator", choices=list(ACCELERATORS), default=None,
                   help="pin the accelerator (needed when submitting GPU jobs from GPU-less login nodes)")
    p.add_argument("--image", default=None, help="container image (default: per engine+accelerator)")
    p.add_argument("--port", type=int, default=None, help="serving port (default: engine default, next free)")
    p.add_argument("--gpus", type=int, default=0, help="GPUs per node for the --scheduler job request")
    p.add_argument("--nodes", type=int, default=1, help="node count for the --scheduler job request")
    p.add_argument("--name", default=None, help="container name (default: derived from the model)")
    p.add_argument("--here", action="store_true",
                   help="allow serving directly on a scheduler login node (bypasses the guard)")
    p.add_argument("--foreground", action="store_true",
                   help="stay attached with engine logs (default inside allocations / under --scheduler)")
    p.add_argument("--ready-timeout", type=float, default=180.0,
                   help="seconds to wait for the endpoint when detached (default 180)")
    p.add_argument("--save-profile", default=None, metavar="PREFIX",
                   help="write the resolved config to PREFIX.box.toml + PREFIX.location.toml")
    p.add_argument("--dryrun", action="store_true", help="print the command instead of executing it")
    p.add_argument("args", nargs="*", help="extra engine args (put them after --)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run", help="run the box with explicit arguments (raw passthrough; profile mode)")
    _add_common(p)
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the box entrypoint")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="pre-stage a model via RamaLama transports (pull on the login node)")
    p.add_argument("model", nargs="?", default=None,
                   help="transport URI (hf://, ollama://, oci://); alternative: --box")
    p.add_argument("--box", default=None, help="pull the model named by a box TOML profile")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("build", help="build/convert the image for the location's runtime (OCI->SIF)")
    _add_common(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("generate", help="transpile box+location to another orchestrator (cloud path)")
    p.add_argument("format", choices=["sky"], help="output format (sky = SkyPilot task YAML)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--serve", action="store_true", help="add a SkyServe service block (sky serve up)")
    p.add_argument("-o", "--output", default=None, help="write YAML to file instead of stdout")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("bench", help="throughput/latency sweep against a served box (paper's step 5)")
    p.add_argument("--box", required=True)
    p.add_argument("--url", default=None, help="endpoint (default: http://127.0.0.1:<box port>)")
    p.add_argument("--batch-sizes", default=None, help="comma list, default 1,2,4,...,1024")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--dataset", default=None, help="JSON list of prompts or ShareGPT JSON")
    p.add_argument("-o", "--output", default=None, help="write plot-ready CSV here")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("launch", help="launch the box on cloud via SkyPilot (delegated)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", required=True)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--serve", action="store_true", help="managed serving via SkyServe (sky serve up)")
    p.add_argument("--down", action="store_true", help="tear down instead of launching")
    p.add_argument("-o", "--output", default=None, help="also keep the task YAML at this path")
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("stop", help="stop (and remove) a boxy-served container")
    p.add_argument("name", nargs="?", default=None,
                   help="container name from the READY banner or `boxy list`; alternative: --box")
    p.add_argument("--box", default=None, help="stop the container named by a box TOML profile")
    p.add_argument("--location", default=None)
    p.add_argument("--runtime", choices=["podman", "docker"], default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("list", help="list running boxy-launched containers")
    p.add_argument("--location", default=None)
    p.add_argument("--runtime", choices=["podman", "docker"], default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_list)

    for name, help_text in (
        ("alloc", "request nodes via the location's scheduler"),
        ("stage", "stage models to shared FS / site-local S3"),
    ):
        p = sub.add_parser(name, help=f"{help_text} (post-MVP)")
        p.set_defaults(func=_stub(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    # Everything after a standalone `--` is engine args, verbatim. argparse
    # cannot express this next to optional positionals (a `*` positional only
    # matches one contiguous chunk), so split before parsing.
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)
    try:
        if extra:
            if not hasattr(args, "args"):
                raise ValueError(f"'boxy {args.subcommand}' takes no engine args after --")
            args.args = list(args.args or []) + extra
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"boxy: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
