"""Photon S8 — scoped sidecar-token handoff (``runtime_credentials.py``).

Presence (and other local consumers) must be able to read the sidecar's
``X-Hermes-Sidecar-Token`` without parsing ``~/.hermes/.env``. The adapter
materializes ONLY the token into a dedicated runtime credential file:

* ``$XDG_RUNTIME_DIR/hermes/photon-sidecar.token``                 (default)
* ``$XDG_RUNTIME_DIR/hermes/profiles/<name>/photon-sidecar.token`` (profile)
* ``$XDG_RUNTIME_DIR/hermes/custom-<digest>/photon-sidecar.token`` (custom home)
* ``<HERMES_HOME>/run/photon-sidecar.token``                       (fallback)

Directory mode 0700, file mode 0600, atomic rotation, cleanup on stop, and
the token never appears in logs.
"""
from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from hermes_constants import _get_platform_default_hermes_home
from plugins.platforms.photon import runtime_credentials as rc

_POSIX = sys.platform != "win32"


@pytest.fixture
def xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return runtime


def _default_home(monkeypatch: pytest.MonkeyPatch) -> Path:
    home = _get_platform_default_hermes_home()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


# ---------------------------------------------------------------------------
# Path selection


def test_token_path_default_profile(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    assert rc.sidecar_token_path() == xdg / "hermes" / "photon-sidecar.token"


def test_token_path_named_profile(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _get_platform_default_hermes_home()
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    assert rc.sidecar_token_path() == (
        xdg / "hermes" / "profiles" / "coder" / "photon-sidecar.token"
    )


def test_token_path_custom_home_is_scoped_and_stable(
    xdg: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-a"))
    path_a = rc.sidecar_token_path()
    assert path_a == rc.sidecar_token_path()  # stable
    assert path_a.parent.parent == xdg / "hermes"
    assert path_a.parent.name.startswith("custom-")
    assert path_a.name == "photon-sidecar.token"

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-b"))
    path_b = rc.sidecar_token_path()
    # Two custom homes never collide on one runtime dir.
    assert path_b != path_a


def test_token_path_falls_back_without_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rc.sidecar_token_path() == home / "run" / "photon-sidecar.token"


def test_token_path_falls_back_when_xdg_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "does-not-exist"))
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert rc.sidecar_token_path() == home / "run" / "photon-sidecar.token"


# ---------------------------------------------------------------------------
# Write / rotate / clear


def test_write_creates_private_file_with_exact_token(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    path = rc.write_sidecar_token("tok-alpha-000")
    assert path == rc.sidecar_token_path()
    # File contains exactly the token — no trailing data, no other secrets.
    assert path.read_text(encoding="utf-8") == "tok-alpha-000"
    if _POSIX:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_write_secures_profile_scope_dirs(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _get_platform_default_hermes_home()
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    path = rc.write_sidecar_token("tok-profile-111")
    assert path.read_text(encoding="utf-8") == "tok-profile-111"
    if _POSIX:
        # Every directory from the hermes base down is owner-only.
        current = path.parent
        while current != xdg:
            assert stat.S_IMODE(current.stat().st_mode) == 0o700, current
            current = current.parent


def test_write_is_atomic_and_leaves_no_temp_files(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    path = rc.write_sidecar_token("tok-atomic-222")
    leftovers = [
        p for p in path.parent.iterdir() if p.name != path.name
    ]
    assert leftovers == []


def test_write_rotates_atomically(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    path = rc.write_sidecar_token("tok-old-333")
    rc.write_sidecar_token("tok-new-444")
    assert path.read_text(encoding="utf-8") == "tok-new-444"
    if _POSIX:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_write_rejects_empty_token(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    with pytest.raises(ValueError):
        rc.write_sidecar_token("")
    assert not rc.sidecar_token_path().exists()


def test_write_uses_fallback_dir_without_xdg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = rc.write_sidecar_token("tok-fallback-555")
    assert path == home / "run" / "photon-sidecar.token"
    assert path.read_text(encoding="utf-8") == "tok-fallback-555"
    if _POSIX:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_clear_removes_token_and_stale_temps(
    xdg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _default_home(monkeypatch)
    path = rc.write_sidecar_token("tok-clear-666")
    # Simulate a crashed write that left a temp file behind.
    stale = path.parent / (rc._TMP_PREFIX + "stale")
    stale.write_text("tok-stale", encoding="utf-8")
    rc.clear_sidecar_token()
    assert not path.exists()
    assert not stale.exists()
    # Idempotent.
    rc.clear_sidecar_token()


# ---------------------------------------------------------------------------
# Adapter wiring — token materialized on sidecar readiness, cleared on stop.


def _make_adapter(monkeypatch: pytest.MonkeyPatch):
    from gateway.config import PlatformConfig
    from plugins.platforms.photon.adapter import PhotonAdapter

    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


class _FakeProc:
    pid = 4321
    stdout = None
    stdin = None
    returncode = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def _stub_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, healthy: bool
) -> Dict[str, Any]:
    from plugins.platforms.photon import adapter as photon_adapter

    (tmp_path / "sidecar" / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(photon_adapter, "_SIDECAR_DIR", tmp_path / "sidecar")

    class _PatchResult:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        photon_adapter.subprocess, "run", lambda *a, **k: _PatchResult()
    )

    spawned: Dict[str, Any] = {}

    if healthy:
        proc = _FakeProc()
    else:

        class _DeadProc(_FakeProc):
            returncode = 3

            def poll(self):
                return 3

        proc = _DeadProc()

    def _fake_popen(cmd: List[str], **kwargs: Any):
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(photon_adapter.subprocess, "Popen", _fake_popen)

    class _Client:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        async def post(self, *a: Any, **k: Any):
            if not healthy:
                raise photon_adapter.httpx.ConnectError("refused")

            class _Resp:
                status_code = 200

            return _Resp()

    monkeypatch.setattr(photon_adapter.httpx, "AsyncClient", _Client)
    return spawned


@pytest.mark.asyncio
async def test_start_sidecar_materializes_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from plugins.platforms.photon import adapter as photon_adapter

    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    _stub_spawn(monkeypatch, tmp_path, healthy=True)

    await adapter._start_sidecar()
    try:
        token_path = rc.sidecar_token_path()
        assert token_path.exists()
        assert token_path.read_text(encoding="utf-8") == adapter._sidecar_token
        if _POSIX:
            assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
    finally:
        await adapter._stop_sidecar()
    # Stop must clean the credential up.
    assert not rc.sidecar_token_path().exists()


@pytest.mark.asyncio
async def test_failed_sidecar_start_writes_no_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    _stub_spawn(monkeypatch, tmp_path, healthy=False)

    with pytest.raises(RuntimeError):
        await adapter._start_sidecar()
    assert not rc.sidecar_token_path().exists()


@pytest.mark.asyncio
async def test_token_never_logged_during_start_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _make_adapter(monkeypatch)

    async def _no_reap() -> None:
        pass

    monkeypatch.setattr(adapter, "_reap_stale_sidecar", _no_reap)
    _stub_spawn(monkeypatch, tmp_path, healthy=True)

    with caplog.at_level("DEBUG"):
        await adapter._start_sidecar()
        await adapter._stop_sidecar()
    assert adapter._sidecar_token not in caplog.text
