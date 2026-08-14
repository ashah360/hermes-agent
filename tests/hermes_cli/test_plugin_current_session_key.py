"""Tests for PluginContext.current_session_key() session routing exposure."""

import contextvars
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


def test_current_session_key_fails_closed_when_resolver_raises():
    context, _ = _context()

    with patch(
        "tools.approval.get_current_session_key",
        side_effect=RuntimeError("resolver unavailable"),
    ):
        assert context.current_session_key() == ""
