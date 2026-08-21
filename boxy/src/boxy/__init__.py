"""boxy: unified, site-portable, offline-first CLI for containerized GenAI serving on HPC."""

import os

__version__ = "0.2.1"


def _read_git_revision():
    """Best-effort (short_sha, branch) of the source checkout, read straight from
    .git files (no git subprocess). Editable installs (pip install -e boxy) point
    at the checkout, so this reflects the LIVE commit — the answer to 'is my boxy
    up to date?'. Returns (None, None) when not a git checkout (wheel install)."""
    # walk up from this file to the enclosing git checkout (the package may be a
    # subdir of the repo, e.g. <repo>/boxy/src/boxy), so a fixed depth won't do
    d = os.path.dirname(os.path.abspath(__file__))
    git = None
    while True:
        cand = os.path.join(d, ".git")
        if os.path.isdir(cand):
            git = cand
            break
        parent = os.path.dirname(d)
        if parent == d:
            return None, None  # reached filesystem root, no checkout
        d = parent
    try:
        with open(os.path.join(git, "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref:"):
            ref = head[4:].strip()
            # keep the full branch name, which may contain slashes
            # (refs/heads/claude/boxy-... -> claude/boxy-...)
            branch = ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref
            ref_file = os.path.join(git, ref)
            if os.path.exists(ref_file):
                with open(ref_file) as f:
                    return f.read().strip()[:7], branch
            packed = os.path.join(git, "packed-refs")  # ref may be packed
            if os.path.exists(packed):
                with open(packed) as f:
                    for line in f:
                        if line.rstrip().endswith(ref):
                            return line.split(" ", 1)[0][:7], branch
            return None, branch
        return head[:7], None  # detached HEAD
    except OSError:
        return None, None


def version_string() -> str:
    """`__version__`, annotated with the checkout's git commit/branch — but ONLY
    when the RUNNING code actually lives in that checkout.

    A copied install whose venv sits INSIDE the repo (site-packages under
    <repo>/.venv/...) walks up to <repo>/.git and used to wear the repo's sha as
    if it were its own. Field: six fixes 'confirmed current' by --version while
    a frozen snapshot predating all of them did the work. A frozen copy now says
    so, loudly, with the remedy."""
    here = os.path.abspath(os.path.dirname(__file__))
    frozen = "site-packages" in here.split(os.sep) or "dist-packages" in here.split(os.sep)
    sha, branch = _read_git_revision()
    if frozen:
        if sha:
            at = f"git {sha}{', ' + branch if branch else ''}"
            return (f"{__version__} (INSTALLED COPY — not tracking the checkout at {at}; "
                    f"`pip install -e ./boxy` to run the checkout)")
        return f"{__version__} (installed copy)"
    if sha and branch:
        return f"{__version__} (git {sha}, {branch})"
    if sha:
        return f"{__version__} (git {sha})"
    return __version__
