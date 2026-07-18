"""App cards (src/boxy/appcards.py + `boxy app`): the deployment-OS namespace
for classic HPC applications/benchmarks. Card loading/precedence, the rendered
batch script (spack bootstrap, launcher geometry, container+proxy), and the CLI
list/dryrun/error paths — all against packaged cards or tmp user cards, never a
real spack or scheduler."""

import hashlib

import pytest

from boxy import appcards
from boxy.cli import main


@pytest.fixture
def user_cards(tmp_path, monkeypatch):
    d = tmp_path / "cfg" / "boxy" / "cards" / "apps"
    d.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    return d


# ---- loading + precedence -----------------------------------------------------------


def test_packaged_cards_load_and_validate():
    cards = appcards.load_cards()
    names = {c.name for c in cards}
    assert {"osu-benchmarks", "stream", "miniem"} <= names
    osu = appcards.find_card("osu-benchmarks")
    assert osu.kind == "spack" and osu.spec.startswith("osu-micro-benchmarks")
    assert osu.nodes == 2 and osu.tasks_per_node == 1
    assert osu.source == "packaged"


def test_user_card_wins_over_packaged(user_cards):
    (user_cards / "osu-benchmarks.toml").write_text(
        '[app]\nname = "osu-benchmarks"\nkind = "spack"\nspec = "osu-micro-benchmarks@7.4"\n'
        'run = ["osu_bw"]\n')
    card = appcards.find_card("osu-benchmarks")
    assert card.source == "user" and card.spec == "osu-micro-benchmarks@7.4"


def test_malformed_user_card_raises_with_its_path(user_cards):
    (user_cards / "broken.toml").write_text("[app]\nkind = 'spack'\n")   # no name
    with pytest.raises(ValueError) as e:
        appcards.load_cards()
    assert "broken.toml" in str(e.value) and "name" in str(e.value)


def test_unknown_kind_rejected(user_cards):
    (user_cards / "weird.toml").write_text(
        '[app]\nname = "weird"\nkind = "cmake"\nrun = ["x"]\n')
    with pytest.raises(ValueError) as e:
        appcards.load_cards()
    assert "kind" in str(e.value)


# ---- script rendering ---------------------------------------------------------------


def test_spack_script_slurm_geometry_and_bootstrap():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x-%j.log", ["--account=fy1"])
    assert "#SBATCH --nodes=2" in text
    assert "#SBATCH --account=fy1" in text
    assert "spack install --reuse -y osu-micro-benchmarks" in text
    assert "spack load osu-micro-benchmarks" in text
    assert "srun -N 2 -n 2 " in text                     # geometry rides the launcher
    assert "setup-env.sh" in text                        # spack probed on the node


def test_flux_spelling_and_flag_overrides():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "flux", "app-osu", "/tmp/x.log", [],
                                      nodes=4, tasks_per_node=2)
    assert "flux run -N 4 -n 8 " in text                 # flags override the card
    assert "srun" not in text


def test_container_card_renders_podman_with_proxy(user_cards):
    (user_cards / "hello.toml").write_text(
        '[app]\nname = "hello"\nkind = "container"\nimage = "quay.io/podman/hello:latest"\n'
        'run = ["echo hi"]\nnodes = 1\n')
    card = appcards.find_card("hello")
    text = appcards.render_app_script(card, "slurm", "app-hello", "/tmp/x.log", [],
                                      proxy_prefix="env https_proxy=http://p:80 ")
    assert "podman run --rm quay.io/podman/hello:latest echo hi" in text
    assert "env https_proxy=http://p:80 srun -N 1 -n 1 " in text
    assert "spack" not in text


def test_modules_and_setup_lines_precede_run():
    card = appcards.AppCard(name="x", card_name="x", source="user", kind="spack",
                            spec="stream", modules=["gcc/12"], setup=["export OMP_NUM_THREADS=16"],
                            run=["stream_c.exe"])
    text = appcards.render_app_script(card, "slurm", "app-x", "/tmp/x.log", [])
    body = text[text.index("set -e"):]
    assert body.index("module load gcc/12") < body.index("spack install")
    assert body.index("export OMP_NUM_THREADS=16") < body.index("stream_c.exe")


# ---- the CLI ------------------------------------------------------------------------


def test_cli_bare_app_lists_cards(capfd):
    rc = main(["app"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "osu-benchmarks" in out and "stream" in out and "miniem" in out


def test_cli_unknown_card_is_a_clear_error(capfd):
    rc = main(["app", "does-not-exist"])
    err = capfd.readouterr().err
    assert rc == 2
    assert "no app card named" in err and "osu-benchmarks" in err


def test_cli_dryrun_prints_script_without_scheduler(capfd, monkeypatch, tmp_path):
    # --dryrun renders even on a machine with NO scheduler (the laptop case).
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["app", "stream", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --job-name=app-stream" in out
    assert "spack install --reuse -y stream" in out
    assert "srun -N 1 -n 1 stream_c.exe" in out


def test_cli_cards_listing_includes_apps(capfd):
    rc = main(["cards"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "app cards" in out and "osu-benchmarks" in out


# ---- egress-filter (Zscaler) source-fetch heal --------------------------------------

TARBALL = b"fake-osu-tarball-bytes"
DIGEST = hashlib.sha256(TARBALL).hexdigest()

# condensed from the real kahuna/hops failure: every fetcher (spack mirror AND
# upstream) 403'd into the site filter's block page — CATEGORY_DENIED.
ZSCALER_LOG = f"""\
spack.error.FetchError: All fetchers failed for spack-stage-osu-micro-benchmarks-7.5.2-frmvtwk7ojga4y4kl5nfrc5bamxwrrz5
        https://mirror.spack.io/_source-cache/archive/{DIGEST[:2]}/{DIGEST}.tar.gz: DetailedHTTPError: GET http://block-message.ca.sandia.gov/block-page.html?url=https%3a%2f%2fmirror%2espack%2eio%2f_source-cache%2farchive%2f{DIGEST[:2]}%2f{DIGEST}%2etar%2egz&reason=Not+allowed returned 403: Forbidden
    https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.2.tar.gz: DetailedHTTPError: GET http://block-message.ca.sandia.gov/block-page.html?url=https%3a%2f%2fmvapich%2ecse%2eohio%2dstate%2eedu%2fdownload%2fmvapich%2fosu-micro-benchmarks-7.5.2%2etar%2egz&reason=Not+allowed returned 403: Forbidden
==> Error: The following packages failed to install:
osu-micro-benchmarks@7.5.2/frmvtwk: /tmp/spack-stage.log
"""


def test_spack_fetch_block_detected_and_sources_extracted():
    from boxy import cli

    assert cli._spack_fetch_blocked(ZSCALER_LOG)
    assert not cli._spack_fetch_blocked("CUDA out of memory")
    urls, rel = cli._extract_spack_sources(ZSCALER_LOG)
    assert urls[0].startswith("https://mirror.spack.io/")            # sha-addressed first
    assert any("mvapich.cse.ohio-state.edu" in u for u in urls)      # upstream unwrapped
    assert not any("block-message" in u for u in urls)               # block pages excluded
    assert rel == f"_source-cache/archive/{DIGEST[:2]}/{DIGEST}.tar.gz"


def test_spack_sources_fall_back_to_package_layout():
    from boxy import cli

    log = ("spack.error.FetchError: All fetchers failed for "
           "spack-stage-stream-5.10-abcdefghijklmnopqrstuvwxyz012345\n"
           "    https://www.cs.virginia.edu/stream/FTP/Code/stream-5.10.tar.gz: "
           "returned 403: Forbidden\n")
    urls, rel = cli._extract_spack_sources(log)
    assert urls == ["https://www.cs.virginia.edu/stream/FTP/Code/stream-5.10.tar.gz"]
    assert rel == "stream/stream-5.10.tar.gz"


def test_spack_source_heal_stages_verified_archive(monkeypatch, capfd):
    from boxy import cli, remote

    pushed = {}
    monkeypatch.setattr(remote, "push_file", lambda t, path, data: pushed.update({path: data}) or 0)
    ok = cli._maybe_spack_source_heal("user@hops", "hops", ZSCALER_LOG, "/home/u/mir",
                                      downloader=lambda u: TARBALL)
    err = capfd.readouterr().err
    assert ok
    dest = f"/home/u/mir/_source-cache/archive/{DIGEST[:2]}/{DIGEST}.tar.gz"
    assert pushed == {dest: TARBALL}
    assert "staging it into a spack mirror" in err and "resubmitting" in err


def test_spack_source_heal_rejects_sha_mismatch(monkeypatch, capfd):
    # the filter may serve an HTML block page WITH a 200 — never stage bytes that
    # don't match spack's digest.
    from boxy import cli, remote

    monkeypatch.setattr(remote, "push_file",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not push")))
    ok = cli._maybe_spack_source_heal("user@hops", "hops", ZSCALER_LOG, "/m",
                                      downloader=lambda u: b"<html>blocked</html>")
    assert not ok
    assert "does not match spack's sha256" in capfd.readouterr().err


def test_render_includes_local_mirror_when_given():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x.log", [],
                                      spack_mirror_dir="/home/u/.local/share/boxy/agentless/h/spack-mirror")
    assert 'spack mirror add boxy-local "file:///home/u/.local/share/boxy/agentless/h/spack-mirror"' in text
    assert text.index("spack mirror add") < text.index("spack install")
    plain = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x.log", [])
    assert "spack mirror" not in plain


def test_stage_source_pushes_at_its_own_digest(tmp_path, monkeypatch, capfd):
    # --stage-source: a browser-downloaded archive lands in the mirror at its OWN
    # sha256 — correct file => spack finds it by digest; wrong file => ignored.
    import argparse

    from boxy import cli, remote

    tarball = tmp_path / "osu-micro-benchmarks-7.5.2.tar.gz"
    tarball.write_bytes(TARBALL)
    pushed = {}
    monkeypatch.setattr(remote, "push_file", lambda t, path, data: pushed.update({path: data}) or 0)
    ns = argparse.Namespace(stage_source=str(tarball))
    assert cli._stage_source_file(ns, "user@hops", "hops", "/m") is True
    assert pushed == {f"/m/_source-cache/archive/{DIGEST[:2]}/{DIGEST}.tar.gz": TARBALL}
    assert "digest-addressed" in capfd.readouterr().out


def test_stage_source_rejects_non_archive(tmp_path, capfd):
    import argparse

    from boxy import cli

    junk = tmp_path / "notes.txt"
    junk.write_text("hi")
    ns = argparse.Namespace(stage_source=str(junk))
    assert cli._stage_source_file(ns, "u@h", "h", "/m") is False
    assert "not a source archive" in capfd.readouterr().err
    ns2 = argparse.Namespace(stage_source=None)
    assert cli._stage_source_file(ns2, "u@h", "h", "/m") is None


def test_osu_card_carries_pinned_source_provenance():
    # turnkey on filtered sites: the packaged card pins the version and names the
    # archive + digest, so boxy can stage the source without ever needing a
    # failed job to learn the URL from.
    card = appcards.find_card("osu-benchmarks")
    assert card.spec == "osu-micro-benchmarks@7.5.2"
    assert card.sha256 == "618de3d0b1122f73a9229177d2da1e5cd62e431190580cb915f2605849cbbbdc"
    assert any("mvapich.cse.ohio-state.edu" in u for u in card.sources)
    assert any("mirror.spack.io" in u for u in card.sources)


def test_render_exports_proxy_env_before_spack():
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x.log", [],
                                      proxy_env={"https_proxy": "http://proxy.sandia.gov:80"})
    assert "export https_proxy=http://proxy.sandia.gov:80" in text
    assert text.index("export https_proxy") < text.index("spack install")
    plain = appcards.render_app_script(card, "slurm", "app-osu", "/tmp/x.log", [])
    assert "export https_proxy" not in plain


# ---- ad-hoc container apps (--image / --container) ----------------------------------


def test_cli_adhoc_image_runs_entrypoint(capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["app", "--image", "quay.io/podman/hello:latest", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --job-name=app-hello" in out            # name derived from the image
    assert "srun -N 1 -n 1 podman run --rm quay.io/podman/hello:latest\n" in out
    assert "spack" not in out


def test_cli_adhoc_container_alias_cmd_and_geometry(capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["app", "--container", "quay.io/podman/hello:latest", "--cmd", "echo hi",
               "--nodes", "2", "--tasks-per-node", "4", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --nodes=2" in out
    assert "srun -N 2 -n 8 podman run --rm quay.io/podman/hello:latest echo hi" in out


def test_cli_adhoc_image_with_explicit_name(capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["app", "smoke", "--image", "quay.io/podman/hello:latest", "--dryrun"])
    out = capfd.readouterr().out
    assert rc == 0
    assert "#SBATCH --job-name=app-smoke" in out            # the positional names the job


def test_spack_bootstrap_uses_externals_and_gcc_retry():
    # FIELD (flux cluster, icc): spack rebuilt gmake bottom-up and the site's
    # Intel classic compiler died on gnulib's __malloc__ attributes. The script
    # must (a) register system build tools as externals so gmake isn't built at
    # all, and (b) retry the install ONCE with %gcc when the default compiler
    # fails and gcc is registered.
    card = appcards.find_card("osu-benchmarks")
    text = appcards.render_app_script(card, "flux", "app-osu", "/tmp/x.log", [])
    assert "spack external find" in text
    assert text.index("spack external find") < text.index("spack install")
    assert "if ! spack install --reuse -y osu-micro-benchmarks@7.5.2; then" in text
    assert "spack install --reuse -y osu-micro-benchmarks@7.5.2 %gcc" in text
    assert "spack compilers" in text                      # gated on gcc being registered


OMPI_UCX_LOG = """\
A requested component was not found, or was unable to be opened.
Host:      cronus1
Framework: pml
Component: ucx
--------------------------------------------------------------------------
  mca_base_framework_open on ompi_pml failed
  --> Returned "Not found" (-13) instead of "Success" (0)
*** An error occurred in MPI_Init
srun: error: cronus1: task 0: Exited with exit code 14
"""


def test_ompi_ucx_failure_detected_and_card_gets_tcp_transport():
    from boxy import cli

    assert cli._looks_like_ompi_ucx_failure(OMPI_UCX_LOG)
    assert not cli._looks_like_ompi_ucx_failure("Segmentation fault (core dumped)")
    card = appcards.find_card("osu-benchmarks")
    healed = cli._card_with_tcp_mpi(card)
    assert "export OMPI_MCA_pml=ob1" in healed.setup
    assert "export OMPI_MCA_btl=self,vader,tcp" in healed.setup
    text = appcards.render_app_script(healed, "slurm", "app-osu", "/tmp/x.log", [])
    body = text[text.index("set -e"):]
    assert body.index("export OMPI_MCA_pml=ob1") < body.index("osu_bw")   # before the ranks launch


def test_stop_all_and_clean_lifecycle(tmp_path, monkeypatch, capfd):
    # the panic button + the sweep: `boxy stop --all` cancels every recorded
    # job; `boxy clean` removes finished-job debris and keeps live jobs.
    import json as _json

    from boxy.cli import main as _main

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path))
    rc = _main(["clean", "--dryrun"])
    assert rc == 0 and "would clean 0" in capfd.readouterr().out
    rc = _main(["stop", "--all"])
    assert rc == 0 and "stopped 0 job(s)" in capfd.readouterr().out
    # a finished (unreachable-scheduler => not live... FOREIGN) record is KEPT;
    # a local-cluster DONE record is cleaned
    import socket

    (tmp_path / "dead.json").write_text(_json.dumps(
        {"name": "dead", "scheduler": "slurm", "job": "1",
         "submitted_from": socket.gethostname()}))
    (tmp_path / "dead-1.log").write_text("old log")
    rc = _main(["clean"])
    out = capfd.readouterr().out
    assert rc == 0 and "cleaned 1 finished job(s)" in out
    assert not (tmp_path / "dead.json").exists()
    assert not (tmp_path / "dead-1.log").exists()


# ---- SERVICES: long-running containers (web/MCP/DB/microservices) --------------------


@pytest.fixture
def ssh_cluster(monkeypatch, tmp_path):
    """A fake --ssh target for dryrun renders: master always up, scheduler probe
    answers slurm, everything else empty (defaults)."""
    from boxy import remote

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(remote, "ensure_master", lambda host: 0)

    def cap(target, cmd, timeout=30):
        if "$HOME" in cmd:
            return 0, "/home/user"
        if "sbatch" in cmd or "instance-level" in cmd:
            return 0, "slurm-bin\nslurm-ctld\nslurm-live\n"
        return 1, ""

    monkeypatch.setattr(remote, "ssh_capture", cap)


SVC_CARD = """
[app]
name = "whoami"
summary = "tiny test web service"
kind = "container"
service = true
image = "docker.io/traefik/whoami:latest"
port = 8080
container_port = 80
[app.env]
WHOAMI_NAME = "boxy"
"""


def test_service_card_parses_and_validates(tmp_path):
    card = appcards._parse_card(SVC_CARD, "whoami", "user", "x")
    assert card.service and card.port == 8080 and card.env == {"WHOAMI_NAME": "boxy"}
    with pytest.raises(ValueError, match="port"):
        appcards._parse_card('[app]\nname="x"\nkind="container"\nimage="i"\nservice=true\n',
                             "x", "user", "x")
    with pytest.raises(ValueError, match="container"):
        appcards._parse_card('[app]\nname="x"\nkind="spack"\nspec="s"\nservice=true\nport=1\n'
                             'run=["r"]\n', "x", "user", "x")


def test_service_script_publishes_endpoint_and_execs_foreground(tmp_path):
    card = appcards._parse_card(SVC_CARD, "whoami", "user", "x")
    script = appcards.render_app_script(card, "slurm", "app-whoami", "/x/%j.log", [],
                                        endpoint_file="/shared/app-whoami.endpoint.json")
    assert 'cat > "${_EP}.tmp"' in script and '"port": 8080' in script
    # -p HOST:CONTAINER, never --network=host: rootless podman can't bind the
    # image's own privileged port on the HOST (field: whoami 'listen tcp :80:
    # bind: permission denied'); inside its netns the container may bind 80.
    assert "exec podman run --rm --name=app-whoami -p 8080:80" in script
    assert "--network=host" not in script
    assert "-e 'WHOAMI_NAME=boxy'" in script or "-e WHOAMI_NAME=boxy" in script
    assert "srun" not in script                            # foreground, not a launcher fan-out
    with pytest.raises(ValueError, match="endpoint"):
        appcards.render_app_script(card, "slurm", "n", "/x.log", [])


def test_cli_adhoc_image_with_port_becomes_service(ssh_cluster, capfd, monkeypatch):
    # `boxy app --image X --port N --ssh HOST`: the user's "deploy any
    # microservice" command — long-running, endpoint published, URL awaited.
    rc = main(["app", "--image", "docker.io/traefik/whoami:latest", "--port", "8080:80",
               "--env", "WHOAMI_NAME=boxy", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless SERVICE" in cap.out and "boxy stop app-whoami" in cap.out
    assert '"url": "http://${_H}:8080"' in cap.out
    assert "-p 8080:80" in cap.out
    # exec'd in the FOREGROUND (a proxy env prefix may sit between exec and podman)
    assert any(ln.strip().startswith("exec ") and "--name=app-whoami" in ln
               for ln in cap.out.splitlines())


def test_cli_service_privileged_host_port_fails_laptop_side(ssh_cluster, capfd):
    # FIELD: whoami's entrypoint bound :80 under host networking and died with
    # 'bind: permission denied' ON THE NODE. Now the mistake is caught before
    # anything is submitted, with the mapping syntax in the message.
    rc = main(["app", "--image", "docker.io/traefik/whoami:latest", "--port", "80",
               "--ssh", "user@hops", "--dryrun"])
    err = capfd.readouterr().err
    assert rc == 2
    assert "privileged" in err and "--port 8080:80" in err


def test_svc_ports_parsing():
    from boxy import cli

    assert cli._svc_ports(None) == (0, 0)
    assert cli._svc_ports("8080") == (8080, 0)
    assert cli._svc_ports("8080:80") == (8080, 80)
    with pytest.raises(cli.UsageError):
        cli._svc_ports("eighty")


def test_cli_service_without_ssh_is_a_clear_error(capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.delenv("BOXY_SSH_HOST", raising=False)
    rc = main(["app", "--image", "docker.io/traefik/whoami:latest", "--port", "8080"])
    assert rc == 2
    assert "needs a cluster" in capfd.readouterr().err


def test_flux_mcp_ported_to_a_service_card(ssh_cluster, capfd, monkeypatch):
    # the packaged flux-mcp SERVICE card replaces `generate flux-mcp` for clusters
    rc = main(["app", "flux-mcp", "--ssh", "user@hops", "--dryrun"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "Agentless SERVICE" in cap.out
    assert "ghcr.io/converged-computing/flux-mcp" in cap.out
    assert '"port": 8089' in cap.out


def test_generate_flux_mcp_prints_deprecation(capfd, monkeypatch, tmp_path):
    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    rc = main(["generate", "flux-mcp", "--host", "flux-mcp.apps.example"])
    cap = capfd.readouterr()
    assert rc == 0
    assert "DEPRECATED" in cap.err and "boxy app flux-mcp" in cap.err
    assert "kind: Deployment" in cap.out or "apiVersion" in cap.out    # manifest still emitted


def test_service_follow_opens_local_tunnel_at_ready(monkeypatch, tmp_path, capfd):
    # FIELD: 'I am not able to hit it with a URL or localhost' — compute nodes
    # aren't routable from a laptop. At READY the service follow now opens the
    # forward ON THE LIVE MASTER itself and prints the working localhost URL.
    from boxy import remote

    monkeypatch.setenv("BOXY_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(remote, "ensure_master", lambda host: 0)
    monkeypatch.setattr(remote, "push_file", lambda t, p, c: 0)
    forwards = []
    monkeypatch.setattr(remote, "add_forward",
                        lambda t, lp, rh, rp: (forwards.append((t, lp, rh, rp)), 0)[1])
    monkeypatch.setattr(remote, "_local_port_free", lambda p: True)

    def cap(target, cmd, timeout=30):
        if cmd.startswith("cat ") and "endpoint" in cmd:
            return 0, ('{"name": "app-whoami", "host": "nid00123", "port": 8080, '
                       '"url": "http://nid00123:8080", "job": "77"}')
        if "--parsable" in cmd:
            return 0, "77\n"
        if "$HOME" in cmd:
            return 0, "/home/user"
        if "sbatch" in cmd or "instance-level" in cmd:
            return 0, "slurm-bin\nslurm-ctld\nslurm-live\n"
        return 0, ""

    monkeypatch.setattr(remote, "ssh_capture", cap)
    rc = main(["app", "--image", "docker.io/traefik/whoami:latest", "--port", "8080:80",
               "--ssh", "user@cronus"])
    out = capfd.readouterr().out
    assert rc == 0
    assert forwards == [("user@cronus", 8080, "nid00123", 8080)]   # the tunnel WAS opened
    assert "### LOCAL  http://127.0.0.1:8080/" in out
    assert "curl -s http://127.0.0.1:8080/" in out
    assert "NOT reachable from your laptop directly" in out
