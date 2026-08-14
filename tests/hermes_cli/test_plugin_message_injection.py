"""Tests for plugin message injection across CLI and gateway hosts."""

from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(name: str = "notify-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, key=name, source="user")
    return PluginContext(manifest, manager), manager


def _write_plugin_config(tmp_path, monkeypatch, entry: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": entry}}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


def test_cli_idle_injection_keeps_existing_queue_behaviour():
    context, manager = _context()
    cli = SimpleNamespace(
        _agent_running=False,
        _pending_input=SimpleQueue(),
        _interrupt_queue=SimpleQueue(),
    )
    manager._cli_ref = cli

    assert context.inject_message("new input") is True
    assert cli._pending_input.get_nowait() == "new input"
    assert cli._interrupt_queue.empty()


def test_cli_running_injection_keeps_existing_interrupt_behaviour():
    context, manager = _context()
    cli = SimpleNamespace(
        _agent_running=True,
        _pending_input=SimpleQueue(),
        _interrupt_queue=SimpleQueue(),
    )
    manager._cli_ref = cli

    assert context.inject_message("status", "system") is True
    assert cli._interrupt_queue.get_nowait() == "[system] status"
    assert cli._pending_input.empty()


def test_gateway_injection_requires_session_key(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert context.inject_message("wake up") is False
    injector.assert_not_called()


def test_gateway_injection_requires_explicit_permission(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )
    injector.assert_not_called()


def test_gateway_injection_does_not_treat_string_as_permission(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": "false"},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )
    injector.assert_not_called()


def test_gateway_injection_fails_closed_when_config_cannot_be_read():
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    with patch(
        "hermes_cli.plugins.load_config_readonly",
        side_effect=OSError("config unavailable"),
    ):
        assert (
            context.inject_message(
                "wake up",
                session_key="agent:main:telegram:dm:42",
            )
            is False
        )

    injector.assert_not_called()


def test_gateway_injection_requires_live_host(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()

    assert manager.has_gateway_message_injector is False
    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


def test_gateway_injection_passes_host_owned_plugin_identity(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_message_injector(object(), injector)

    result = context.inject_message(
        "wake up",
        role="system",
        session_key="agent:main:telegram:dm:42",
    )

    assert result is True
    injector.assert_called_once_with(
        session_key="agent:main:telegram:dm:42",
        content="[system] wake up",
        plugin_id="notify-plugin",
    )


def test_gateway_injection_returns_host_rejection(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    manager.set_gateway_message_injector(
        object(),
        MagicMock(return_value=False),
    )

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


def test_gateway_injection_fails_closed_on_host_exception(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(side_effect=RuntimeError("gateway unavailable"))
    manager.set_gateway_message_injector(object(), injector)

    assert (
        context.inject_message(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )


# -- gateway_message_injection_available() preflight -------------------------
#
# Plugins load in every Hermes host process (CLI, desktop serve, dashboard,
# gateway), but only the gateway process ever installs a live message
# injector. A plugin delivery worker must be able to check — BEFORE claiming
# durable work — whether THIS process can actually inject, because
# ctx.inject_message existing as a method proves nothing about the host.


def test_injection_available_false_with_permission_but_no_injector(
    tmp_path, monkeypatch
):
    """Permission granted but no live injector (CLI/serve/dashboard host)."""
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()

    assert manager.has_gateway_message_injector is False
    assert context.gateway_message_injection_available() is False


def test_injection_available_false_with_injector_but_no_permission(
    tmp_path, monkeypatch
):
    """Live gateway injector but the plugin lacks the config grant."""
    _write_plugin_config(tmp_path, monkeypatch, {})
    context, manager = _context()
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))

    assert context.gateway_message_injection_available() is False


def test_injection_available_true_with_permission_and_live_injector(
    tmp_path, monkeypatch
):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))

    assert context.gateway_message_injection_available() is True


def test_injection_available_tracks_injector_lifecycle(tmp_path, monkeypatch):
    """Dynamic read, not cached: false -> true -> false across the lifecycle."""
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    owner = object()

    # Before the gateway installs its injector (plugin initialization time).
    assert context.gateway_message_injection_available() is False

    manager.set_gateway_message_injector(owner, MagicMock(return_value=True))
    assert context.gateway_message_injection_available() is True

    # Owner-safe clear (gateway shutdown) drops availability again.
    manager.clear_gateway_message_injector(owner)
    assert context.gateway_message_injection_available() is False


def test_injection_available_fails_closed_when_config_cannot_be_read():
    context, manager = _context()
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))

    with patch(
        "hermes_cli.plugins.load_config_readonly",
        side_effect=OSError("config unavailable"),
    ):
        assert context.gateway_message_injection_available() is False


# -- inject_message_confirmed() ------------------------------------------------
#
# Confirmed injection is a SEPARATELY registered PluginManager contract with
# its own owner-safe lifecycle; the immediate injector stays untouched.


@pytest.mark.parametrize(
    "denial", ["no_session_key", "no_permission", "config_error"]
)
def test_confirmed_injection_precheck_fails_closed(tmp_path, monkeypatch, denial):
    entry = {} if denial == "no_permission" else {"allow_gateway_injection": True}
    _write_plugin_config(tmp_path, monkeypatch, entry)
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_confirmed_message_injector(object(), injector)

    session_key = "" if denial == "no_session_key" else "agent:main:telegram:dm:42"
    if denial == "config_error":
        with patch(
            "hermes_cli.plugins.load_config_readonly",
            side_effect=OSError("config unavailable"),
        ):
            assert (
                context.inject_message_confirmed("wake up", session_key=session_key)
                is False
            )
    else:
        assert (
            context.inject_message_confirmed("wake up", session_key=session_key)
            is False
        )
    injector.assert_not_called()


def test_confirmed_injection_passes_framing_session_and_timeout(
    tmp_path, monkeypatch
):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_gateway_confirmed_message_injector(object(), injector)

    result = context.inject_message_confirmed(
        "wake up",
        role="system",
        session_key="agent:main:telegram:dm:42",
        timeout_s=2.5,
    )

    assert result is True
    injector.assert_called_once_with(
        session_key="agent:main:telegram:dm:42",
        content="[system] wake up",
        plugin_id="notify-plugin",
        timeout_s=2.5,
    )


def test_confirmed_injection_fails_closed_on_injector_signature_mismatch(
    tmp_path, monkeypatch
):
    """No compatibility fallback: a mismatched confirmed injector fails closed."""
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    injector = MagicMock(side_effect=TypeError("unexpected keyword argument"))
    manager.set_gateway_confirmed_message_injector(object(), injector)

    assert (
        context.inject_message_confirmed(
            "wake up",
            session_key="agent:main:telegram:dm:42",
        )
        is False
    )
    injector.assert_called_once()


def test_confirmed_injector_is_separate_and_owner_safe(tmp_path, monkeypatch):
    _write_plugin_config(
        tmp_path,
        monkeypatch,
        {"allow_gateway_injection": True},
    )
    context, manager = _context()
    key = "agent:main:telegram:dm:42"

    # The immediate injector alone must not satisfy the confirmed API.
    manager.set_gateway_message_injector(object(), MagicMock(return_value=True))
    assert manager.has_gateway_confirmed_message_injector is False
    assert context.inject_message_confirmed("wake up", session_key=key) is False

    owner = object()
    manager.set_gateway_confirmed_message_injector(
        owner, MagicMock(return_value=True)
    )
    assert context.inject_message_confirmed("wake up", session_key=key) is True

    # A stranger's clear must not clobber the registered owner.
    manager.clear_gateway_confirmed_message_injector(object())
    assert manager.has_gateway_confirmed_message_injector is True

    manager.clear_gateway_confirmed_message_injector(owner)
    assert context.inject_message_confirmed("wake up", session_key=key) is False
