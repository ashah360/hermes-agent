"""Scoped Photon sidecar-token handoff for local consumers (S8).

The Photon adapter authenticates loopback calls to its Node sidecar with a
per-run shared token (``X-Hermes-Sidecar-Token``). Sibling local services —
e.g. Hermes Presence, which consumes the sidecar's read-only location
endpoints (see ``sidecar/LOCATIONS_PROTOCOL.md``) — need that token, but
must not parse ``~/.hermes/.env``: that file holds unrelated secrets.

This module materializes ONLY the sidecar token into a dedicated runtime
credential file:

* ``$XDG_RUNTIME_DIR/hermes/photon-sidecar.token``                  (default profile)
* ``$XDG_RUNTIME_DIR/hermes/profiles/<name>/photon-sidecar.token``  (named profile)
* ``$XDG_RUNTIME_DIR/hermes/custom-<sha256[:12]>/photon-sidecar.token``
  (custom ``HERMES_HOME`` — the digest keeps two custom deployments on one
  machine from clobbering each other's token)
* ``<HERMES_HOME>/run/photon-sidecar.token``                        (fallback when
  ``XDG_RUNTIME_DIR`` is unset or unusable; inherently profile-scoped)

Guarantees:

* directories ``0700``, file ``0600`` (best-effort on Windows);
* the file contains exactly the token — nothing else is ever copied there;
* writes are atomic (same-directory temp file + ``os.replace``), so a reader
  never observes a partial token and rotation is a single swap;
* :func:`clear_sidecar_token` removes the file and any stale temp files.

The adapter writes the file once its sidecar passes the readiness health
check and clears it when the sidecar stops (see ``adapter.py``). The token
value itself must never be logged.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

from hermes_constants import (
    _get_platform_default_hermes_home,
    get_hermes_home,
)

_TOKEN_FILENAME = "photon-sidecar.token"
# Leading dot + distinctive prefix so clear_sidecar_token() can sweep
# leftovers from a crashed write without touching anything else.
_TMP_PREFIX = ".photon-sidecar.token."


def _sanitize_component(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    return cleaned or "profile"


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _scope_parts() -> tuple[str, ...]:
    """Per-deployment discriminator under the shared runtime base dir.

    Empty for the platform-default home, ``("profiles", <name>)`` for a
    standard profile, and a stable digest directory for custom homes.
    """
    home = _resolve(get_hermes_home())
    default_root = _resolve(_get_platform_default_hermes_home())
    if home == default_root:
        return ()
    if home.parent.name == "profiles" and home.parent.parent == default_root:
        return ("profiles", _sanitize_component(home.name))
    digest = hashlib.sha256(str(home).encode("utf-8")).hexdigest()[:12]
    return (f"custom-{digest}",)


def _token_location() -> tuple[Path, Path]:
    """Return ``(anchor, token_path)``.

    ``anchor`` is the pre-existing directory we never chmod (the runtime dir
    itself, or the Hermes home); everything created below it is made
    owner-only.
    """
    raw = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if raw:
        runtime_root = Path(raw)
        try:
            usable = runtime_root.is_dir()
        except OSError:
            usable = False
        if usable:
            base = runtime_root / "hermes"
            for part in _scope_parts():
                base = base / part
            return runtime_root, base / _TOKEN_FILENAME
    home = get_hermes_home()
    return home, home / "run" / _TOKEN_FILENAME


def sidecar_token_path() -> Path:
    """The runtime credential file path for the active profile."""
    return _token_location()[1]


def _ensure_private_dirs(anchor: Path, parent: Path) -> None:
    anchor.mkdir(parents=True, exist_ok=True)
    relative = parent.relative_to(anchor)
    current = anchor
    for part in relative.parts:
        current = current / part
        current.mkdir(mode=0o700, exist_ok=True)
        if sys.platform != "win32":
            # mkdir mode is masked by umask; enforce owner-only explicitly.
            os.chmod(current, 0o700)


def write_sidecar_token(token: str) -> Path:
    """Atomically materialize *token* (and only the token) for consumers.

    Returns the credential file path. Raises ``ValueError`` for an empty
    token and ``OSError`` on filesystem failure — callers treat both as
    non-fatal (the sidecar itself is unaffected).
    """
    if not token or not isinstance(token, str):
        raise ValueError("sidecar token must be a non-empty string")
    anchor, path = _token_location()
    _ensure_private_dirs(anchor, path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=str(path.parent))
    try:
        if sys.platform != "win32":
            os.fchmod(fd, 0o600)  # mkstemp already uses 0600; be explicit
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def clear_sidecar_token() -> None:
    """Remove the credential file and any stale temp files. Never raises."""
    path = sidecar_token_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        for stale in path.parent.glob(f"{_TMP_PREFIX}*"):
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass
