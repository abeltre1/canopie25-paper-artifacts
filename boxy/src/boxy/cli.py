"""boxy CLI: one tool to deploy/serve containerized GenAI across HPC sites.

    boxy serve --box boxes/vllm.toml --location locations/eldorado.toml
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys

from boxy import __version__, ramalama_shim
from boxy.backends import BACKENDS
from boxy.box import Box
from boxy.location import Location
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


def cmd_serve(args: argparse.Namespace) -> int:
    from boxy import deploy

    box, location = _load(args)
    deployment = deploy.plan_serve(box, location, port=args.port, extra_args=args.args, dryrun=args.dryrun)
    return _emit(deployment, args.dryrun)


def cmd_run(args: argparse.Namespace) -> int:
    from boxy import deploy

    box, location = _load(args)
    # argparse.REMAINDER keeps the literal "--" separator; it must not reach
    # the container command.
    user_args = args.args[1:] if args.args[:1] == ["--"] else args.args
    deployment = deploy.plan_run(box, location, user_args, dryrun=args.dryrun)
    return _emit(deployment, args.dryrun)


def cmd_pull(args: argparse.Namespace) -> int:
    box = Box.from_toml(args.box)
    if not box.model:
        print(f"box {box.name}: no model set", file=sys.stderr)
        return 1
    if not box.model_is_transport_uri:
        print(f"box {box.name}: model is a path ({box.model}); nothing to pull (shared-FS flow)")
        return 0
    path = ramalama_shim.pull_model(box.model, dryrun=args.dryrun)
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
    box = Box.from_toml(args.box)
    location = Location.from_toml(args.location) if args.location else None
    runtime = _container_runtime(location)
    return _run_or_print([runtime, "stop", box.name], args.dryrun)


def cmd_list(args: argparse.Namespace) -> int:
    location = Location.from_toml(args.location) if args.location else None
    runtime = _container_runtime(location)
    return _run_or_print([runtime, "ps", "--filter", "label=boxy.box"], args.dryrun)


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

    p = sub.add_parser("serve", help="serve the box as an OpenAI-compatible endpoint")
    _add_common(p)
    p.add_argument("--port", type=int, default=None, help="override the box's serving port")
    p.add_argument("args", nargs="*", help="extra engine args (after --)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("run", help="run the box with explicit arguments (prototype passthrough)")
    _add_common(p)
    p.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the box entrypoint")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pull", help="fetch the box's model via RamaLama transports")
    p.add_argument("--box", required=True)
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

    p = sub.add_parser("stop", help="stop a running boxy container (by box name)")
    p.add_argument("--box", required=True)
    p.add_argument("--location", default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("list", help="list running boxy-launched containers")
    p.add_argument("--location", default=None)
    p.add_argument("--dryrun", action="store_true")
    p.set_defaults(func=cmd_list)

    for name, help_text in (
        ("alloc", "request nodes via the location's scheduler"),
        ("stage", "stage models to shared FS / site-local S3"),
        ("bench", "throughput/latency sweep"),
    ):
        p = sub.add_parser(name, help=f"{help_text} (post-MVP)")
        p.set_defaults(func=_stub(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"boxy: error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
