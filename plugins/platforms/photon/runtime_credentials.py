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
* **symlink-safe**: the anchor (runtime root / Hermes home) is canonicalized
  once, and every component below it is traversed with descriptor-relative
  ``O_NOFOLLOW`` opens on POSIX (created with ``mkdir(dir_fd=...)``); a
  symlinked ``hermes``/``profiles``/profile/leaf component, a non-directory,
  or a directory owned by another user aborts the write, and the opened
  directory chain is verified against the canonical path before the token
  is swapped in. Windows falls back to explicit per-component symlink
  rejection (``lstat``-based);
* :func:`clear_sidecar_token` is **path-bound**: callers that recorded the
  exact path a write returned pass it back, so cleanup removes that file
  regardless of the ambient profile scope at clear time. Clear never
  follows symlinks — a parent chain that no longer resolves to itself is
  refused, and the unlink happens descriptor-relative on POSIX.

The adapter writes the file once its sidecar passes the readiness health
check, records the returned path, and clears exactly that path when the
sidecar stops (see ``adapter.py``). The token value itself must never be
logged.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import stat as stat_module
import sys
import tempfile
import uuid
from pathlib import Path

from hermes_constants import (
    _get_platform_default_hermes_home,
    get_hermes_home,
)

logger = logging.getLogger(__name__)

_TOKEN_FILENAME = "photon-sidecar.token"
# Leading dot + distinctive prefix so clear_sidecar_token() can sweep
# leftovers from a crashed write without touching anything else.
_TMP_PREFIX = ".photon-sidecar.token."

_POSIX = os.name == "posix"


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


def _canonical(path: Path) -> Path:
    return Path(os.path.realpath(path))


def _token_location() -> tuple[Path, Path]:
    """Return ``(anchor, token_path)``.

    ``anchor`` is the approved canonical runtime root (realpath'd once) that
    we never chmod; every component below it is created owner-only and must
    be symlink-free.
    """
    raw = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if raw:
        runtime_root = Path(raw)
        try:
            usable = runtime_root.is_dir()
        except OSError:
            usable = False
        if usable:
            anchor = _canonical(runtime_root)
            base = anchor / "hermes"
            for part in _scope_parts():
                base = base / part
            return anchor, base / _TOKEN_FILENAME
    anchor = _canonical(get_hermes_home())
    return anchor, anchor / "run" / _TOKEN_FILENAME


def sidecar_token_path() -> Path:
    """The runtime credential file path for the active profile scope."""
    return _token_location()[1]


# ---------------------------------------------------------------------------
# POSIX: descriptor-relative, O_NOFOLLOW traversal


def _verify_private_dir_fd(fd: int, *, chmod: bool) -> None:
    st = os.fstat(fd)
    if not stat_module.S_ISDIR(st.st_mode):
        raise RuntimeError("runtime credential path component is not a directory")
    if st.st_uid != os.geteuid():
        raise RuntimeError(
            "runtime credential path component has an unsafe owner"
        )
    if chmod:
        os.fchmod(fd, 0o700)


def _open_child_dir(parent_fd: int, name: str) -> int:
    """Open (creating if needed) a child directory without following links."""
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)

    def _open() -> int:
        try:
            return os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            import errno

            if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EMLINK):
                raise RuntimeError(
                    "refusing symlinked or non-directory runtime credential "
                    "path component"
                ) from exc
            raise

    try:
        fd = _open()
    except FileNotFoundError:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass  # racing creator — the O_NOFOLLOW reopen still vets it
        fd = _open()
    try:
        _verify_private_dir_fd(fd, chmod=True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _write_posix(anchor: Path, path: Path, token: str) -> None:
    parent = path.parent
    anchor.mkdir(parents=True, exist_ok=True)
    fds: list[int] = []
    try:
        anchor_fd = os.open(str(anchor), os.O_RDONLY | os.O_DIRECTORY)
        fds.append(anchor_fd)
        # Never chmod the anchor itself (XDG runtime root / Hermes home).
        _verify_private_dir_fd(anchor_fd, chmod=False)
        for part in parent.relative_to(anchor).parts:
            fds.append(_open_child_dir(fds[-1], part))
        leaf_fd = fds[-1]

        # The opened descriptor chain must still be the canonical parent —
        # the final artifact stays under the approved runtime root.
        st_fd = os.fstat(leaf_fd)
        st_path = os.lstat(parent)
        if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
            raise RuntimeError(
                "runtime credential directory escaped its canonical root"
            )

        # Reject a symlinked token leaf rather than replacing it.
        try:
            leaf_st = os.stat(path.name, dir_fd=leaf_fd, follow_symlinks=False)
            if stat_module.S_ISLNK(leaf_st.st_mode):
                raise RuntimeError(
                    "refusing to replace a symlinked runtime credential file"
                )
        except FileNotFoundError:
            pass

        tmp_name = _TMP_PREFIX + uuid.uuid4().hex
        tmp_fd = os.open(
            tmp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=leaf_fd,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                handle.write(token)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                tmp_name, path.name, src_dir_fd=leaf_fd, dst_dir_fd=leaf_fd
            )
        except BaseException:
            try:
                os.unlink(tmp_name, dir_fd=leaf_fd)
            except OSError:
                pass
            raise
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Windows fallback: Path-based with explicit symlink rejection


def _reject_symlink(path: Path) -> None:
    try:
        if path.is_symlink():
            raise RuntimeError(
                "refusing symlinked runtime credential path component"
            )
    except OSError:
        pass


def _write_fallback(anchor: Path, path: Path, token: str) -> None:
    anchor.mkdir(parents=True, exist_ok=True)
    current = anchor
    for part in path.parent.relative_to(anchor).parts:
        current = current / part
        _reject_symlink(current)
        current.mkdir(mode=0o700, exist_ok=True)
        _reject_symlink(current)
    _reject_symlink(path)
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=str(path.parent))
    try:
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


# ---------------------------------------------------------------------------
# Public API


def write_sidecar_token(token: str) -> Path:
    """Atomically materialize *token* (and only the token) for consumers.

    Returns the credential file path — callers that clear later must record
    it and pass it back to :func:`clear_sidecar_token`, so cleanup stays
    bound to this write even if the ambient profile scope changes. Raises
    ``ValueError`` for an empty token and ``OSError``/``RuntimeError`` on
    filesystem failure or an unsafe (symlinked / foreign-owned) path —
    callers treat these as non-fatal (the sidecar itself is unaffected).
    """
    if not token or not isinstance(token, str):
        raise ValueError("sidecar token must be a non-empty string")
    anchor, path = _token_location()
    if _POSIX:
        _write_posix(anchor, path, token)
    else:
        _write_fallback(anchor, path, token)
    return path


def _parent_is_canonical(parent: Path) -> bool:
    """True when *parent* resolves to itself (no symlinked components).

    Paths handed out by :func:`write_sidecar_token` are canonical, so any
    divergence means a symlink appeared underneath — refuse to touch it.
    """
    try:
        return Path(os.path.realpath(parent)) == parent
    except OSError:
        return False


def clear_sidecar_token(path: Path | None = None) -> None:
    """Remove the credential file and any stale temp files. Never raises.

    ``path`` should be the exact path a previous :func:`write_sidecar_token`
    returned; when omitted, the ambient profile scope's path is used. The
    removal never follows symlinks: a parent chain that no longer resolves
    to itself is refused, and the unlink is descriptor-relative on POSIX.
    """
    try:
        target = Path(path) if path is not None else sidecar_token_path()
    except OSError:
        return
    parent = target.parent
    if not _parent_is_canonical(parent):
        logger.debug(
            "[photon] refusing to clear runtime credential under a "
            "non-canonical (symlinked?) directory"
        )
        return
    try:
        if _POSIX:
            _clear_posix(parent, target.name)
        else:
            _clear_fallback(parent, target.name)
    except OSError:
        pass


def _clear_posix(parent: Path, leaf_name: str) -> None:
    try:
        dir_fd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return  # nothing to clear
    try:
        # The opened directory must be the canonical parent we just checked.
        st_fd = os.fstat(dir_fd)
        st_path = os.lstat(parent)
        if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
            return
        for name in (leaf_name, *_stale_temp_names(dir_fd)):
            try:
                os.unlink(name, dir_fd=dir_fd)
            except OSError:
                pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def _stale_temp_names(dir_fd: int) -> list[str]:
    try:
        return [
            name
            for name in os.listdir(dir_fd)
            if name.startswith(_TMP_PREFIX)
        ]
    except OSError:
        return []


def _clear_fallback(parent: Path, leaf_name: str) -> None:
    try:
        (parent / leaf_name).unlink(missing_ok=True)
    except OSError:
        pass
    try:
        for stale in parent.glob(f"{_TMP_PREFIX}*"):
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass
