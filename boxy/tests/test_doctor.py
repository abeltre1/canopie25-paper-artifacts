"""boxy doctor — the executable half of the SPEC §8b known-issues registry.
Each check reuses an existing boxy helper and reports OK/WARN/FAIL + a fix;
`boxy doctor` exits non-zero only on FAIL."""

from boxy import doctor
from boxy.cli import main


def test_proxy_check_flags_http_without_https(monkeypatch):
    from boxy import ramalama_shim

    monkeypatch.setattr(ramalama_shim, "active_proxies",
                        lambda: {"http": "http://proxy.site.gov:80"})
    r = doctor._check_proxy()
    assert r.status == doctor.WARN and "https_proxy is NOT" in r.detail and "BYPASS" in r.fix
    # both set -> OK
    monkeypatch.setattr(ramalama_shim, "active_proxies",
                        lambda: {"http": "http://p:80", "https": "http://p:80"})
    assert doctor._check_proxy().status == doctor.OK
    # none -> OK direct
    monkeypatch.setattr(ramalama_shim, "active_proxies", lambda: {})
    assert doctor._check_proxy().status == doctor.OK


def test_tls_check_fails_on_missing_cert_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "nope.crt"))
    r = doctor._check_tls()
    assert r.status == doctor.FAIL and "does not exist" in r.detail and "SILENTLY ignores" in r.fix
    real = tmp_path / "ca.crt"
    real.write_text("x")
    monkeypatch.setenv("SSL_CERT_FILE", str(real))
    monkeypatch.delenv("BOXY_NO_CA_MERGE", raising=False)
    assert doctor._check_tls().status == doctor.OK


def test_exited_containers_flags_oom_137(monkeypatch):
    def fake_run(cmd, **kw):
        out = ""
        if cmd[1:3] == ["ps", "-a"]:
            out = "boxy-a Exited (137) 2 minutes ago\n"
        elif cmd[1] == "inspect":
            out = "137 true"
        return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    monkeypatch.setattr(doctor.shutil, "which", lambda b: "/usr/bin/podman")
    r = doctor._check_exited_containers("podman")
    assert r.status == doctor.WARN and "137" in r.detail and "podman machine set --memory" in r.fix


def test_image_registry_403_is_a_fail(monkeypatch):
    import urllib.error
    import urllib.request

    def boom(url, timeout=0):
        raise urllib.error.HTTPError(url, 403, "Forbidden", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    results = doctor._check_image_registries()
    assert results and all(r.status == doctor.FAIL for r in results)
    assert any("403" in r.detail and "Zscaler" in r.detail for r in results)
    assert any("--proxy" in r.fix or "pre-pull" in r.fix for r in results)


def test_image_registry_401_404_is_ok(monkeypatch):
    import urllib.error
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout=0: (_ for _ in ()).throw(
                            urllib.error.HTTPError(url, 401, "Unauthorized", None, None)))
    assert all(r.status == doctor.OK for r in doctor._check_image_registries())


def test_doctor_exit_code_zero_when_no_fail(monkeypatch, capsys):
    # force every check OK/WARN (no FAIL) -> exit 0
    monkeypatch.setattr(doctor, "run_checks",
                        lambda net=False: [doctor.Result("x", doctor.OK, "fine"),
                                           doctor.Result("y", doctor.WARN, "heads up", "do z")])
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "all critical checks OK" in out and "WARN y: do z" in out


def test_doctor_exit_code_nonzero_on_fail(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks",
                        lambda net=False: [doctor.Result("net", doctor.FAIL, "blocked", "fix it")])
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL net: fix it" in out and "1 FAIL" in out


def test_doctor_runtime_none_is_fail(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda b: None)
    r = doctor._check_runtime()
    assert r.status == doctor.FAIL and "none of" in r.detail
