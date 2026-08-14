"""Tests for PluginContext.current_session_key() session routing exposure."""

import contextvars
import os
import threading
from unittest.mock import patch

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from tools.approval import reset_current_session_key, set_current_session_key


def _context(name: str = "notify-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, key=name, source="user")
    return PluginContext(manifest, manager), manager


def test_current_session_key_returns_bound_key_then_empty_after_reset():
    context, _ = _context()

    token = set_current_session_key("agent:main:telegram:dm:42")
    try:
        assert context.current_session_key() == "agent:main:telegram:dm:42"
    finally:
        reset_current_session_key(token)

    assert context.current_session_key() == ""


def test_current_session_key_is_empty_without_active_session():
    context, _ = _context()

    assert context.current_session_key() == ""


def test_current_session_key_is_context_local_across_copied_contexts():
    context, _ = _context()

    def _bound_read(session_key: str) -> str:
        token = set_current_session_key(session_key)
        try:
            return context.current_session_key()
        finally:
            reset_current_session_key(token)

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    assert ctx_a.run(_bound_read, "agent:main:telegram:dm:1") == (
        "agent:main:telegram:dm:1"
    )
    assert ctx_b.run(_bound_read, "agent:main:discord:dm:2") == (
        "agent:main:discord:dm:2"
    )
    # Neither copied context leaks its binding into the caller's context.
    assert context.current_session_key() == ""


def test_current_session_key_ignores_process_global_env_on_fresh_thread():
    """A plugin background thread must never inherit HERMES_SESSION_KEY.

    The process-global env var can hold a stale or unrelated route key (CLI,
    cron, or a previously active gateway session). A fresh thread with no
    copied session context has no active turn, so the resolver must return
    ``""`` — never the env fallback — or a plugin could inject into the
    wrong conversation.
    """
    context, _ = _context()

    sentinel = object()
    prior = os.environ.get("HERMES_SESSION_KEY", sentinel)
    os.environ["HERMES_SESSION_KEY"] = "stale-or-unrelated-session"
    try:
        results: list[str] = []
        # threading.Thread targets run in a fresh contextvars.Context — no
        # copied active session context, exactly like a plugin's own
        # background worker thread.
        thread = threading.Thread(
            target=lambda: results.append(context.current_session_key())
        )
        thread.start()
        thread.join()
        assert results == [""]
    finally:
        if prior is sentinel:
            os.environ.pop("HERMES_SESSION_KEY", None)
        else:
            os.environ["HERMES_SESSION_KEY"] = prior


def test_current_session_key_fails_closed_when_resolver_raises():
    context, _ = _context()

    with patch(
        "tools.approval.get_context_bound_session_key",
        side_effect=RuntimeError("resolver unavailable"),
    ):
        assert context.current_session_key() == ""
