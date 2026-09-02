"""`boxy pull hf://… --ssh HOST` is ONE command that abstracts the resources.

Field (Kimi-K3, 1.56TB in 96 shards): the pull went to the transport store under
a 301GB $HOME at 93% — died at shard 2 on quota, silently — and the ssh session
was a single point of failure for an 8-hour download. The user's own words:
"remember we are supposed to have a simple command that abstracts all resources."
These tests pin the hazards behind the one command:

  * the model lands in the shared-FS store the agentless SERVE reads (never $HOME),
  * a download the filesystem cannot hold is refused BEFORE it starts,
  * the download runs detached (setsid) and re-running the command is a
    progress report / resume, not a restart,
  * and — after SIX rounds of in-container CERTIFICATE_VERIFY_FAILED — the
    download is plain curl ON THE LOGIN NODE: the same binary, trust store, and
    network path as the egress probe that verifies. No container, no Python TLS
    stack, no certifi anywhere in the download.
"""

import argparse

from boxy import cli

TARGET = "user@clusterb"
STORE = "/scratch/u/boxy"
STAGE = f"{STORE}/models/moonshotai-kimi-k3"
SCRIPT = "/home/u/.local/share/boxy/agentless/clusterb/pull-moonshotai-kimi-k3.sh"


def _args(**kw):
    d = dict(model="hf://moonshotai/Kimi-K3", box=None, force=False, dryrun=False,
             image=None, proxy=None, ssh=TARGET)
    d.update(kw)
    return argparse.Namespace(**d)


def _wire(monkeypatch, *, state="STATE=IDLE\nGOT=\nSHARDS=0", df_kb="5000000000",
          size=(1560.0, 96)):
    """Fake the cluster: ssh replies keyed on a command fragment; records the
    commands in `calls` and any pushed file in `pushed` (path -> content)."""
    calls = []
    pushed = {}

    def fake_capture(target, cmd, timeout=20):
        calls.append(cmd)
        if "printf %s" in cmd:
            return 0, "/home/u"
        if "STATE=" in cmd:
            return 0, state
        if "df -Pk" in cmd:
            return 0, f"{df_kb}\n"
        if "setsid" in cmd:
            return 0, "12345\n"
        return 0, ""

    monkeypatch.setattr("boxy.remote.ensure_master", lambda t: 0)
    monkeypatch.setattr("boxy.remote.ssh_capture", fake_capture)
    monkeypatch.setattr("boxy.remote.remote_proxy_env", lambda: {})
    monkeypatch.setattr("boxy.remote.push_file",
                        lambda t, p, data: pushed.update({p: data}) or 0)
    monkeypatch.setattr(cli, "_remote_model_store", lambda *a, **k: STORE)
    monkeypatch.setattr(cli, "_stage_agentless_ca", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_hf_size_probe", lambda repo: size)
    monkeypatch.setattr(cli, "_system_card_accel", lambda host: "rocm")
    return calls, pushed


def test_fresh_pull_launches_detached_into_the_serve_store(monkeypatch, capsys):
    calls, pushed = _wire(monkeypatch)
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    out = capsys.readouterr().out
    # the announced destination is the SERVE's stage dir, not the $HOME transport store
    assert STAGE in out and "pull started on clusterb (pid 12345)" in out
    assert "laptop may sleep or disconnect" in out
    assert "~1560GB in 96 weight shards" in out
    assert "boxy pull hf://moonshotai/Kimi-K3 --ssh user@clusterb" in out
    launch = next(c for c in calls if "setsid" in c)
    # detached, log rotated per attempt, script mode-tightened, marker read by the probe
    for frag in ("setsid", "nohup", SCRIPT, ".prev", "chmod 600"):
        assert frag in launch, frag
    script = pushed[SCRIPT]
    # the script IS the download: host curl, resume via .part, skip patterns, marker
    for frag in ("curl -sfL", "resolve/main", "-C -", '.part"', ".boxy-pull-complete",
                 "original/*", STAGE):
        assert frag in script, frag
    assert "podman pull" in script and "|| true" in script     # image warmed, never fatal
    assert "python" not in script.lower(), "no Python TLS stack anywhere in the download"
    assert "rm -rf" not in script                              # no --force -> no clean restart


def test_direct_egress_downloads_with_plain_host_curl(monkeypatch, capsys):
    """FIELD (the whole CERTIFICATE_VERIFY_FAILED saga): the login node's own
    curl verified HF every single time while every in-container Python attempt
    failed. On a verified-direct cluster the script must carry NO proxy and NO
    --cacert — it runs exactly the configuration the probe just proved."""
    calls, pushed = _wire(monkeypatch)
    monkeypatch.setattr("boxy.remote.remote_proxy_env",
                        lambda: {"https_proxy": "http://laptop-proxy:80"})
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    assert "login node's own curl" in capsys.readouterr().out
    script = pushed[SCRIPT]
    assert " -x " not in script, "direct egress must not inherit the laptop proxy"
    assert "--cacert" not in script, "the probe verified with default trust; add nothing"


def test_blocked_egress_gives_curl_the_proxy_and_staged_ca(monkeypatch, capsys):
    # the OTHER kind of site: direct probe fails -> curl goes through the proxy
    # with the staged merged CA, still on the host, still resumable
    calls, pushed = _wire(monkeypatch)
    monkeypatch.setattr("boxy.remote.remote_proxy_env",
                        lambda: {"https_proxy": "http://site-proxy:80"})
    monkeypatch.setattr(cli, "_stage_agentless_ca", lambda *a, **k: "/rdir/boxy-ca-merged.pem")
    orig = __import__("boxy.remote", fromlist=["ssh_capture"]).ssh_capture

    def blocked_probe(target, cmd, timeout=20):
        if cmd.startswith("curl -sIf https://huggingface.co"):
            return 6, ""
        return orig(target, cmd, timeout)

    monkeypatch.setattr("boxy.remote.ssh_capture", blocked_probe)
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    assert "direct HF probe failed" in capsys.readouterr().out
    script = pushed[SCRIPT]
    # the proxy reaches curl through the 0600 config file, NOT on argv, where
    # inline credentials would be world-readable in /proc/<pid>/cmdline
    assert "-x http://site-proxy:80" not in script
    assert 'proxy = "$BOXY_PROXY_URL"' in script
    assert "BOXY_PROXY_URL=http://site-proxy:80" in script
    assert '-K "$CURLRC"' in script
    assert "--cacert /rdir/boxy-ca-merged.pem" in script


def test_hf_token_rides_as_an_auth_header_not_in_the_url(monkeypatch):
    calls, pushed = _wire(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    script = pushed[SCRIPT]
    assert "Authorization: Bearer $HF_TOKEN" in script and "hf_secret_token" in script
    launch = next(c for c in calls if "setsid" in c)
    assert "chmod 600" in launch, "the script carries the token; tighten it before running"
    # ...and the token never reaches curl's ARGV: a shared login node lets any
    # user read /proc/<pid>/cmdline for the whole multi-GB download
    assert "-H \"Authorization" not in script
    assert '-K "$CURLRC"' in script
    assert 'chmod 600 "$CURLRC"' in script
    assert 'trap ' in script and 'rm -f' in script       # removed on every exit path


def test_launch_backgrounds_a_single_redirected_command(monkeypatch):
    """FIELD (eldorado, first Kimi-K3 launch): `mkdir && setsid ... &` made the
    non-interactive remote shell background the WHOLE and-list — a child that
    WAITS on the 8-hour download with the ssh session's stdout/stderr still
    open. ssh never returned; ssh_capture killed it at 30s (rc=124, no output);
    nothing was diagnosed. The backgrounded job must be a single command with
    all three fds redirected, and everything else must run in the foreground."""
    calls, _pushed = _wire(monkeypatch)
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    launch = next(c for c in calls if "setsid" in c)
    assert "&& setsid" not in launch, "and-list backgrounding holds the ssh channel open"
    bg = launch.split(" & ")[0]                       # the job that gets backgrounded
    assert bg.startswith("mkdir -p") and "; setsid" in bg
    for fd in (">>", "2>&1", "< /dev/null"):
        assert fd in bg.split("; setsid", 1)[1], f"background job must redirect {fd}"


def test_launch_timeout_is_diagnosed_not_swallowed(monkeypatch, capsys):
    # rc=124 with empty output used to print a blank error. It now says what
    # happened and that a re-run will find the download if it did start.
    calls, _pushed = _wire(monkeypatch)
    orig = __import__("boxy.remote", fromlist=["ssh_capture"]).ssh_capture

    def timeout_on_launch(target, cmd, timeout=20):
        if "setsid" in cmd:
            calls.append(cmd)
            return 124, ""
        return orig(target, cmd, timeout)

    monkeypatch.setattr("boxy.remote.ssh_capture", timeout_on_launch)
    assert cli._pull_agentless_ssh(_args(), TARGET) == 1
    err = capsys.readouterr().err
    assert "re-run this command to check" in err and "(no output)" in err


def test_stopped_attempt_surfaces_its_log_before_resuming(monkeypatch, capsys):
    """FIELD: the same old traceback was read three times as three new failures
    — an appended log never says which attempt it belongs to. On IDLE-with-log,
    boxy prints WHY the last attempt stopped, then resumes; and every launch
    rotates the log so one file never mixes two attempts."""
    calls, _pushed = _wire(monkeypatch, state="STATE=IDLE\nGOT=16G\nSHARDS=1")
    orig = __import__("boxy.remote", fromlist=["ssh_capture"]).ssh_capture

    def with_tail(target, cmd, timeout=20):
        if cmd.startswith("tail -n"):
            calls.append(cmd)
            return 0, "requests.exceptions.SSLError: CERTIFICATE_VERIFY_FAILED ...\n"
        return orig(target, cmd, timeout)

    monkeypatch.setattr("boxy.remote.ssh_capture", with_tail)
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    out = capsys.readouterr().out
    assert "the previous attempt stopped" in out
    assert "| requests.exceptions.SSLError" in out          # the reason, inline
    launch = next(c for c in calls if "setsid" in c)
    assert ".prev" in launch                                # the log is rotated per attempt


def test_stage_agentless_ca_builds_the_merged_bundle_itself(monkeypatch, tmp_path):
    """FIELD: the site CA never reached the download. The old precondition
    wanted SSL_CERT_FILE to ALREADY be boxy's merged bundle, but the merge is
    process-local, so a fresh invocation never arrives pre-merged and the CA
    silently stayed home. Staging must merge first."""
    import os

    site_ca = tmp_path / "site-ca.crt"
    site_ca.write_text("SITE")
    merged = tmp_path / "ca-merged.crt"
    monkeypatch.delenv("BOXY_NO_CA_PROPAGATE", raising=False)   # suite opts out; opt in
    monkeypatch.setenv("SSL_CERT_FILE", str(site_ca))

    def fake_merge():
        merged.write_text("MERGED")
        os.environ["SSL_CERT_FILE"] = str(merged)
        return str(merged)

    monkeypatch.setattr(cli.ramalama_shim, "ensure_trust_bundle", fake_merge)
    pushed = {}
    monkeypatch.setattr("boxy.remote.push_file",
                        lambda t, p, data: pushed.update(path=p, data=data) or 0)
    appended = []
    monkeypatch.setattr("boxy.remote.ssh_capture",
                        lambda t, cmd, timeout=20: (appended.append(cmd),
                                                    (0, "/etc/pki/tls/certs/ca-bundle.crt"))[1])
    out = cli._stage_agentless_ca("user@c", "c", "/home/u/agentless/c")
    assert out == "/home/u/agentless/c/boxy-ca-merged.pem"
    assert pushed["data"] == "MERGED"                 # the MERGED bundle, not the bare site CA
    # ... and the CLUSTER's own OS trust store is appended remote-side: the
    # laptop CA answers for the laptop's network path, the cluster's egress may
    # be intercepted by a different root that only ITS bundle carries.
    assert appended and "cat" in appended[0] and "boxy-ca-merged.pem" in appended[0]
    assert "/etc/pki/tls/certs/ca-bundle.crt" in appended[0]


def test_stage_agentless_ca_still_noop_when_no_bundle_can_be_built(monkeypatch, tmp_path):
    monkeypatch.delenv("BOXY_NO_CA_PROPAGATE", raising=False)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "site.crt"))
    monkeypatch.setattr(cli.ramalama_shim, "ensure_trust_bundle", lambda: None)
    assert cli._stage_agentless_ca("user@c", "c", "/x") is None


def test_too_small_filesystem_is_refused_before_a_byte_moves(monkeypatch, capsys):
    # 100GB free vs a 1560GB model: the exact Kimi-on-$HOME failure, now refused
    # up front with the fix named — instead of dying at shard 2.
    calls, pushed = _wire(monkeypatch, df_kb="100000000")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 1
    err = capsys.readouterr().err
    assert "refusing to start" in err and "1560" in err and "BOXY_MODEL_DIR" in err
    assert not any("setsid" in c for c in calls) and not pushed


def test_unknown_size_never_blocks(monkeypatch, capsys):
    # egress-filtered laptop: the Hub is unreachable, size unknown -> pull proceeds
    calls, _pushed = _wire(monkeypatch, size=(0.0, 0), df_kb="100000000")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    assert any("setsid" in c for c in calls)


def test_rerun_while_running_reports_progress_not_a_second_download(monkeypatch, capsys):
    calls, pushed = _wire(monkeypatch, state="STATE=RUNNING\nGOT=800G\nSHARDS=50")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    out = capsys.readouterr().out
    assert "pull RUNNING on clusterb: 800G of ~1560GB, 50/96 shards" in out
    assert not any("setsid" in c for c in calls) and not pushed


def test_rerun_after_interruption_resumes(monkeypatch, capsys):
    # 1 shard landed, then the session died (the field case): the rerun relaunches
    # and says it is resuming — complete shards are kept, not re-fetched.
    calls, pushed = _wire(monkeypatch, state="STATE=IDLE\nGOT=16G\nSHARDS=1")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    assert "RESUMING" in capsys.readouterr().out
    assert any("setsid" in c for c in calls)
    assert "rm -rf" not in pushed[SCRIPT]             # resume never wipes the stage


def test_done_reports_the_staged_path_and_the_serve_line(monkeypatch, capsys):
    calls, pushed = _wire(monkeypatch, state="STATE=DONE\nGOT=1560G\nSHARDS=96")
    assert cli._pull_agentless_ssh(_args(), TARGET) == 0
    out = capsys.readouterr().out
    assert f"model staged at: clusterb:{STAGE} (96/96 shards, 1560G)" in out
    assert "serve it:  boxy serve hf://moonshotai/Kimi-K3 --ssh user@clusterb" in out
    assert not any("setsid" in c for c in calls) and not pushed


def test_force_restarts_clean_even_when_done(monkeypatch):
    calls, pushed = _wire(monkeypatch, state="STATE=DONE\nGOT=1560G\nSHARDS=96")
    assert cli._pull_agentless_ssh(_args(force=True), TARGET) == 0
    assert any("setsid" in c for c in calls)
    assert 'rm -rf "$STAGE"' in pushed[SCRIPT]        # --force wipes, then re-pulls


def test_dryrun_plans_but_never_launches_or_pushes(monkeypatch, capsys):
    calls, pushed = _wire(monkeypatch)
    assert cli._pull_agentless_ssh(_args(dryrun=True), TARGET) == 0
    assert "would launch on clusterb (detached" in capsys.readouterr().out
    assert not any("setsid" in c for c in calls) and not pushed


def test_explicit_image_wins(monkeypatch):
    calls, pushed = _wire(monkeypatch)
    assert cli._pull_agentless_ssh(_args(image="quay.io/my/vllm:x"), TARGET) == 0
    assert "podman pull quay.io/my/vllm:x" in pushed[SCRIPT]


def test_local_pull_refuses_a_store_that_cannot_hold_the_model(monkeypatch, capsys):
    # The LOCAL path gets the same guard (the $HOME quota trap, caught up front).
    monkeypatch.delenv("BOXY_SSH_HOST", raising=False)
    monkeypatch.delenv("BOXY_PULL_IGNORE_SPACE", raising=False)
    monkeypatch.setattr(cli, "_hf_size_probe", lambda repo: (1560.0, 96))
    monkeypatch.setattr(cli, "_store_free_gb", lambda store: 23.0)
    monkeypatch.setattr(cli.ramalama_shim, "pull_model",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not pull")))
    rc = cli.cmd_pull(_args(ssh=None))
    assert rc == 1
    err = capsys.readouterr().err
    assert "refusing to pull" in err and "BOXY_STORE" in err


def test_local_pull_proceeds_when_space_is_fine(monkeypatch, capsys):
    monkeypatch.delenv("BOXY_SSH_HOST", raising=False)
    monkeypatch.setattr(cli, "_hf_size_probe", lambda repo: (16.0, 1))
    monkeypatch.setattr(cli, "_store_free_gb", lambda store: 500.0)
    monkeypatch.setattr(cli.ramalama_shim, "pull_model", lambda *a, **k: "/store/x")
    assert cli.cmd_pull(_args(ssh=None)) == 0
    assert "model available at: /store/x" in capsys.readouterr().out


def test_listing_failure_names_the_cause_not_six_useless_words():
    """FIELD (cronus, Kimi-K3): the pull died with 'boxy-pull: could not list
    the repo files' — six words that fit a gated repo, a blocked API path, a
    rate limit, a typo and a DNS failure equally well. `curl -sf` threw away
    both the body and the status, and the pipeline discarded curl's exit code
    on top."""
    script = cli._hf_curl_script("moonshotai/Kimi-K3", "/scratch/stage", "img:tag")
    # the listing keeps the status instead of -f throwing it away...
    assert "%{http_code}" in script
    # ...but STILL follows redirects: the Hub redirects some API paths, and a
    # captive/intercepting proxy answers with a 3xx too (field: cronus returned
    # 307 on /api/models and the download died on the body that came back)
    assert "curl -sSL" in script
    assert "captive portal" in script
    # ...and every status maps to the actual remedy
    assert "401|403)" in script and "GATED" in script and "HF_TOKEN" in script
    assert "404)" in script and "check the id for a typo" in script
    assert "429)" in script and "rate-limited" in script
    assert "000)" in script and "DNS/TLS/proxy, not the model" in script
    # a reply that is not Hub JSON is its own case, and it NAMES where the
    # request actually landed (field: cronus answered 307 on /api/models)
    assert "did not come back as HuggingFace JSON" in script
    assert "url_effective" in script and ".boxy-repo.url" in script


def test_a_dead_pull_reports_FAILED_not_RUNNING(monkeypatch, capsys):
    """The status inferred liveness from the LOG'S MTIME, so a script that had
    just crashed still reported 'pull RUNNING ... 0/96 shards' and sent the user
    away to wait for a download that had already given up."""
    calls, _pushed = _wire(monkeypatch)
    orig = __import__("boxy.remote", fromlist=["ssh_capture"]).ssh_capture

    def failed_probe(target, cmd, timeout=20):
        if "STATE=" in cmd:
            return 0, ("STATE=FAILED\nWHY=HuggingFace refused the repo listing (HTTP 403). "
                       "moonshotai/Kimi-K3 is GATED\nGOT=1G\nSHARDS=0\n")
        return orig(target, cmd, timeout)

    monkeypatch.setattr("boxy.remote.ssh_capture", failed_probe)
    rc = cli._pull_agentless_ssh(_args(), TARGET)
    err = capsys.readouterr().err
    assert rc == 1, "a failed pull must not exit 0"
    assert "pull FAILED" in err
    assert "GATED" in err                      # the cause travels back to the laptop
    assert "tail -40" in err and "resumes" in err
    assert "RUNNING" not in err
