"""Application cards — the third card namespace in boxy's deployment OS.

boxy treats deployments like an operating system treats packages: CARDS are the
package format, schedulers + container runtimes are the drivers, and the
agentless layer is the syscall interface. Model cards describe LLM serving;
SYSTEM cards describe machines; APP cards (this module) describe classic HPC
applications and benchmarks — built with spack or pulled as containers — so

    boxy app osu-benchmarks --ssh kahuna

builds (spack install --reuse), loads, and runs the app as a batch job with the
same zero-flag site resolution (account/partition/time) the serve path has.

Cards are DATA, mirroring model cards:

    packaged  src/boxy/data/cards/apps/*.toml   (ships in the wheel)
    user      ~/.config/boxy/cards/apps/*.toml  (wins over packaged)

Card format (TOML):

    [app]
    name = "osu-benchmarks"
    summary = "OSU MPI micro-benchmarks (bandwidth + latency)"
    kind = "spack"                    # spack | container
    spec = "osu-micro-benchmarks"     # spack: the spec to install/load
    # image = "..."                   # container kind: image to run
    nodes = 2                         # default geometry (flags win)
    tasks_per_node = 1
    gpus_per_node = 0
    time = "30:00"                    # walltime for the batch job
    modules = []                      # module load ... before the run
    setup = []                        # extra shell lines after spack load
    run = ["osu_bw", "osu_latency"]   # each launched via srun/flux run
"""

from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# spack bootstrap locations probed ON THE COMPUTE NODE, in order, when
# $SPACK_ROOT isn't already exported by the user's profile.
_SPACK_SETUP_CANDIDATES = (
    "$SPACK_ROOT/share/spack/setup-env.sh",
    "$HOME/spack/share/spack/setup-env.sh",
    "/opt/spack/share/spack/setup-env.sh",
    "/projects/spack/share/spack/setup-env.sh",
    "/usr/share/spack/setup-env.sh",
)


@dataclass(frozen=True)
class AppCard:
    name: str
    card_name: str                  # file stem — provenance for decision lines
    source: str                     # "user" | "packaged"
    summary: str = ""
    kind: str = "spack"             # "spack" | "container"
    spec: str = ""                  # spack spec (spack kind)
    image: str = ""                 # container image (container kind)
    nodes: int = 1
    tasks_per_node: int = 1
    gpus_per_node: int = 0
    time: str = ""
    modules: list = field(default_factory=list)
    setup: list = field(default_factory=list)
    run: list = field(default_factory=list)
    # source-archive provenance (spack kind): with these, boxy PRE-STAGES the
    # archive into the job's file:// mirror before the first submit — the
    # turnkey path on clusters whose egress filter blocks spack's own fetch.
    sources: list = field(default_factory=list)   # candidate download URLs
    sha256: str = ""                              # the archive's digest (spack's)

    @property
    def label(self) -> str:
        return f"{self.source} app card '{self.card_name}'"


def _user_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(xdg) / "boxy" / "cards" / "apps"


def _parse_card(text: str, card_name: str, source: str, path: str) -> AppCard:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path}: {e}") from None
    a = data.get("app")
    if not isinstance(a, dict) or not a.get("name"):
        raise ValueError(f"{path}: an app card needs an [app] section with a 'name'")
    kind = str(a.get("kind", "spack"))
    if kind not in ("spack", "container"):
        raise ValueError(f"{path}: [app] kind must be 'spack' or 'container', not {kind!r}")
    if kind == "spack" and not a.get("spec"):
        raise ValueError(f"{path}: a spack app card needs [app] spec (the spack spec to install)")
    if kind == "container" and not a.get("image"):
        raise ValueError(f"{path}: a container app card needs [app] image")
    if not a.get("run"):
        raise ValueError(f"{path}: an app card needs [app] run (commands to launch)")
    return AppCard(
        name=str(a["name"]), card_name=card_name, source=source,
        summary=str(a.get("summary", "")), kind=kind,
        spec=str(a.get("spec", "")), image=str(a.get("image", "")),
        nodes=int(a.get("nodes", 1)), tasks_per_node=int(a.get("tasks_per_node", 1)),
        gpus_per_node=int(a.get("gpus_per_node", 0)), time=str(a.get("time", "")),
        modules=list(a.get("modules", [])), setup=list(a.get("setup", [])),
        run=[str(r) for r in a.get("run", [])],
        sources=[str(s) for s in a.get("sources", [])], sha256=str(a.get("sha256", "")),
    )


def load_cards() -> list[AppCard]:
    """User cards first (they win by name), then packaged. A malformed USER card
    raises with its path; a malformed PACKAGED card is a boxy bug but must never
    take down the CLI — it is skipped."""
    out: list[AppCard] = []
    user_dir = _user_dir()
    if user_dir.is_dir():
        for p in sorted(user_dir.glob("*.toml")):
            out.append(_parse_card(p.read_text(), p.stem, "user", str(p)))
    from importlib import resources

    try:
        base = resources.files("boxy") / "data" / "cards" / "apps"
        for entry in sorted(base.iterdir(), key=lambda e: e.name):
            if entry.name.endswith(".toml"):
                try:
                    out.append(_parse_card(entry.read_text(), entry.name[:-5], "packaged", entry.name))
                except ValueError:
                    continue
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    return out


def find_card(name: str) -> AppCard | None:
    """First card whose name or file stem matches (user cards precede packaged)."""
    want = name.strip().lower()
    for c in load_cards():
        if want in (c.name.lower(), c.card_name.lower()):
            return c
    return None


def _launcher(scheduler_name: str, nodes: int, ntasks: int) -> str:
    """The per-command MPI launcher prefix in the active scheduler's spelling.
    Geometry rides on the LAUNCHER (srun -N/-n works inside the allocation) so
    the batch directives stay scheduler-portable."""
    if scheduler_name == "slurm":
        return f"srun -N {nodes} -n {ntasks} "
    if scheduler_name == "flux":
        return f"flux run -N {nodes} -n {ntasks} "
    return ""  # scheduler 'none': run bare (the app decides its own parallelism)


def _spack_bootstrap(spec: str, mirror_dir: str = "") -> list[str]:
    """Shell lines that find spack on the COMPUTE node, build the spec once
    (--reuse makes reruns a no-op against the shared-FS install tree), and load
    it into the environment. Fails with a clear message when no spack exists.

    `mirror_dir` registers a boxy-owned file:// source mirror BEFORE the install:
    when the cluster's egress filter blocks spack's fetch (field: Zscaler
    CATEGORY_DENIED on mirror.spack.io AND the upstream), boxy downloads the
    archive laptop-side and drops it here, so the resubmitted job finds the
    source locally. An empty/missing mirror is harmless — spack falls through."""
    probes = " ".join(f'"{c}"' for c in _SPACK_SETUP_CANDIDATES)
    lines = [
        "_SP=''",
        f"for _c in {probes}; do",
        '  if [ -f "$_c" ]; then _SP="$_c"; break; fi',
        "done",
        'if [ -z "$_SP" ]; then',
        '  echo "boxy: spack not found on this node (checked \\$SPACK_ROOT and the usual'
        ' prefixes) — install spack (github.com/spack/spack) or export SPACK_ROOT" >&2',
        "  exit 3",
        "fi",
        '. "$_SP"',
    ]
    if mirror_dir:
        lines += [
            f'mkdir -p "{mirror_dir}"',
            f'spack mirror add boxy-local "file://{mirror_dir}" 2>/dev/null || '
            f'spack mirror set-url boxy-local "file://{mirror_dir}" 2>/dev/null || true',
        ]
    lines += [
        # register the SYSTEM's build tools (gmake/cmake/autotools/…) as spack
        # externals so spack doesn't rebuild the toolchain bottom-up — building
        # gmake with a site's Intel classic compiler dies on gnulib's __malloc__
        # attributes (field: flux cluster, icc). Idempotent; quick.
        "spack external find >/dev/null 2>&1 || true",
        # first try the site's default compiler; if the build fails and gcc is
        # registered, retry ONCE with %gcc — the fix for classic-compiler (icc)
        # gnulib breakage without permanently overriding the site's toolchain.
        f"if ! spack install --reuse -y {spec}; then",
        '  if spack compilers 2>/dev/null | grep -qi gcc; then',
        '    echo "boxy: the build failed with the default compiler — retrying with %gcc'
        ' (classic Intel compilers cannot build gnulib-based tools)" >&2',
        f"    spack install --reuse -y {spec} %gcc",
        "  else",
        "    exit 1",
        "  fi",
        "fi",
        f"spack load {spec}",
    ]
    return lines


def render_app_script(card: AppCard, scheduler_name: str, name: str, log_file: str,
                      site_args: list[str], *, nodes: int = 0, tasks_per_node: int = 0,
                      proxy_prefix: str = "", spack_mirror_dir: str = "",
                      proxy_env: dict | None = None) -> str:
    """A fully self-contained batch script for an app card — the agentless
    contract: NOTHING boxy-side on the cluster; the compute node needs only
    spack (spack kind) or a container runtime (container kind).

    `proxy_env` is EXPORTED at the top of the job so spack's own source fetch
    (and a container kind's podman pull) goes through the corporate proxy —
    the direct-egress path is what the site filter 403s (field: Zscaler
    noauth-useragent block on mirror.spack.io from the compute node)."""
    from boxy.location import Location, Resources
    from boxy.schedulers import get_scheduler

    n = nodes or card.nodes
    tpn = tasks_per_node or card.tasks_per_node
    ntasks = n * tpn
    body_lines: list[str] = ["set -e"]
    for k, v in (proxy_env or {}).items():
        body_lines.append(f"export {k}={shlex.quote(v)}")
    for m in card.modules:
        body_lines.append(f"module load {m}")
    if card.kind == "spack":
        body_lines += _spack_bootstrap(card.spec, mirror_dir=spack_mirror_dir)
    body_lines += [str(s) for s in card.setup]
    launch = _launcher(scheduler_name, n, ntasks)
    for cmd in card.run:
        if card.kind == "container":
            # an empty run line means "the image's own entrypoint" (the ad-hoc
            # `boxy app --image REF` case)
            inner = f"podman run --rm {shlex.quote(card.image)}" + (f" {cmd}" if cmd else "")
            body_lines.append(f"{proxy_prefix}{launch}{inner}")
        else:
            body_lines.append(f"{launch}{cmd}")
    location = Location(name=card.name, scheduler=scheduler_name,
                        resources=Resources(nodes=n, gpus_per_node=card.gpus_per_node))
    scheduler = get_scheduler(scheduler_name)
    return scheduler.batch_script("", location, name, log_file, site_args,
                                  body="\n".join(body_lines))
