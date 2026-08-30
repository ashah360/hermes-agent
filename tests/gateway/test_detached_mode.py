"""Focused contracts for the opt-in gateway detached-work experiment."""

from __future__ import annotations

import hashlib
import threading
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key
from hermes_cli.commands import resolve_command


def _source(chat_id: str = "chat-1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str, *, chat_id: str = "chat-1") -> MessageEvent:
    return MessageEvent(text=text, source=_source(chat_id), message_id="message-1")


def _store(tmp_path) -> SessionStore:
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
    store._db = None
    return store


def _runner(store: SessionStore):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._sessions = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._evict_cached_agent = MagicMock()
    return runner


def test_detached_command_is_gateway_only_and_uses_expected_verbs():
    command = resolve_command("detached")

    assert command is not None
    assert command.gateway_only is True
    assert command.subcommands == ("on", "off", "status")


@pytest.mark.asyncio
async def test_default_prompt_path_is_exactly_legacy_and_status_is_off(tmp_path):
    store = _store(tmp_path)
    runner = _runner(store)
    entry = store.get_or_create_session(_source())
    legacy_prompt = "legacy gateway prompt bytes"

    effective = runner._with_detached_mode_prompt(entry.session_key, legacy_prompt)

    assert effective is legacy_prompt
    assert (
        await runner._handle_detached_command(_event("/detached status"))
        == "Detached read-only work is OFF for this conversation."
    )
    assert store.load_transcript(entry.session_id) == []
    runner._evict_cached_agent.assert_not_called()


@pytest.mark.asyncio
async def test_enable_is_conversation_scoped_persistent_and_prompt_stable(tmp_path):
    store = _store(tmp_path)
    runner = _runner(store)
    first = store.get_or_create_session(_source("chat-1"))
    second = store.get_or_create_session(_source("chat-2"))
    base = "gateway prompt"

    reply = await runner._handle_detached_command(
        _event("/detached on", chat_id="chat-1")
    )

    assert reply == (
        "Detached read-only work is ON for this conversation. "
        "It resets on /new."
    )
    assert (
        await runner._handle_detached_command(
            _event("/detached status", chat_id="chat-1")
        )
        == "Detached read-only work is ON for this conversation."
    )
    assert (
        await runner._handle_detached_command(
            _event("/detached on", chat_id="chat-1")
        )
        == "Detached read-only work is already ON for this conversation."
    )
    runner._evict_cached_agent.assert_called_once_with(first.session_key)
    prompted_once = runner._with_detached_mode_prompt(first.session_key, base)
    prompted_twice = runner._with_detached_mode_prompt(first.session_key, base)
    assert prompted_once != base
    assert hashlib.sha256(prompted_once.encode()).digest() == hashlib.sha256(
        prompted_twice.encode()
    ).digest()
    assert "delegate_task" in prompted_once
    assert "background=true" in prompted_once
    assert "read-only" in prompted_once
    assert "Never detach" in prompted_once
    assert runner._with_detached_mode_prompt(second.session_key, base) is base
    assert store.load_transcript(first.session_id) == []

    reloaded = _store(tmp_path)
    restarted_runner = _runner(reloaded)
    assert restarted_runner._with_detached_mode_prompt(first.session_key, base) == prompted_once
    assert restarted_runner._with_detached_mode_prompt(second.session_key, base) is base


@pytest.mark.asyncio
async def test_disable_restores_legacy_prompt_without_transcript_injection(tmp_path):
    store = _store(tmp_path)
    runner = _runner(store)
    entry = store.get_or_create_session(_source())
    base = "unchanged prompt"

    await runner._handle_detached_command(_event("/detached on"))
    runner._evict_cached_agent.reset_mock()
    reply = await runner._handle_detached_command(_event("/detached off"))

    assert reply == "Detached read-only work is OFF for this conversation."
    assert runner._with_detached_mode_prompt(entry.session_key, base) is base
    assert (
        await runner._handle_detached_command(_event("/detached"))
        == "Detached read-only work is OFF for this conversation."
    )
    assert (
        await runner._handle_detached_command(_event("/detached off"))
        == "Detached read-only work is already OFF for this conversation."
    )
    assert store.load_transcript(entry.session_id) == []
    runner._evict_cached_agent.assert_called_once_with(entry.session_key)


@pytest.mark.asyncio
async def test_reset_clears_detached_mode_with_the_conversation(tmp_path):
    store = _store(tmp_path)
    runner = _runner(store)
    old_entry = store.get_or_create_session(_source())
    base = "legacy prompt"

    await runner._handle_detached_command(_event("/detached on"))
    assert runner._with_detached_mode_prompt(old_entry.session_key, base) != base

    new_entry = store.reset_session(old_entry.session_key)

    assert new_entry is not None
    assert new_entry.session_id != old_entry.session_id
    assert runner._with_detached_mode_prompt(new_entry.session_key, base) is base
    assert (
        await runner._handle_detached_command(_event("/detached status"))
        == "Detached read-only work is OFF for this conversation."
    )
