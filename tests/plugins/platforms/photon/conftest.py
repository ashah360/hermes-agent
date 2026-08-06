"""Shared fixtures for the Photon platform plugin tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_xdg_runtime_dir(tmp_path, monkeypatch):
    """Redirect ``XDG_RUNTIME_DIR`` to a per-test tempdir.

    The Photon adapter materializes its sidecar token into
    ``$XDG_RUNTIME_DIR/hermes/...`` (see ``runtime_credentials.py``).  The
    global hermetic fixture redirects ``HERMES_HOME`` but not
    ``XDG_RUNTIME_DIR``, so without this fixture any test that starts the
    sidecar would write credential files into the developer's real
    ``/run/user/<uid>``.
    """
    runtime = tmp_path / "xdg-runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return runtime
