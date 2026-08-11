"""
Tests for Slack inbound mention humanization + bot identity grounding.

Slack delivers user mentions as opaque IDs (``<@U123>``). Passing those to the
agent raw leaves it unable to tell one participant from another — or from
itself — so it can misread a mention of a human as a self-mention and reply to
messages addressed to that person (the reported "bot thinks it's @someone-else"
bug). Two cooperating fixes:

  * ``_humanize_user_mentions`` rewrites ``<@UID>`` → ``@DisplayName`` (the
    Slack equivalent of Discord's ``clean_content``).
  * ``_build_identity_prompt`` returns an ephemeral system-prompt line naming
    the bot's own Slack handle so the agent has a positive "that's me" anchor.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock slack-bolt if not installed (same pattern as test_slack_mention.py)
# ---------------------------------------------------------------------------

def _ensure_slack_mock():
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler",
         slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("aiohttp", MagicMock())


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402
_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


def _make_adapter():
    # object.__new__ skips __init__ (heavy setup) — established slack-test pattern.
    return object.__new__(SlackAdapter)


def _adapter_with_names(names):
    """Adapter whose _resolve_user_name returns from a fixed UID→name map."""
    adapter = _make_adapter()

    async def _resolve(user_id, chat_id="", team_id=""):
        return names.get(user_id, user_id)

    adapter._resolve_user_name = _resolve  # type: ignore[assignment]
    return adapter


# ----- _humanize_user_mentions -------------------------------------------------

@pytest.mark.asyncio
async def test_humanizes_single_mention():
    adapter = _adapter_with_names({"U07ALICE": "Alice Example"})
    out = await adapter._humanize_user_mentions(
        "<@U07ALICE> I think thread is prob the right default", chat_id="C1"
    )
    assert out == "@Alice Example I think thread is prob the right default"
    assert "<@" not in out


@pytest.mark.asyncio
async def test_humanizes_multiple_distinct_mentions():
    adapter = _adapter_with_names(
        {"U07ALICE": "Alice Example", "U07BOB": "Bob Example"}
    )
    out = await adapter._humanize_user_mentions(
        "hey <@U07ALICE> and <@U07BOB>", chat_id="C1"
    )
    assert out == "hey @Alice Example and @Bob Example"


@pytest.mark.asyncio
async def test_handles_labelled_mention_form():
    # Slack sometimes sends <@UID|handle>; only the ID drives resolution.
    adapter = _adapter_with_names({"U07ALICE": "Alice Example"})
    out = await adapter._humanize_user_mentions("<@U07ALICE|alice> hi", chat_id="C1")
    assert out == "@Alice Example hi"


@pytest.mark.asyncio
async def test_repeated_mention_all_replaced():
    adapter = _adapter_with_names({"U07ALICE": "Alice Example"})
    out = await adapter._humanize_user_mentions(
        "<@U07ALICE> ping <@U07ALICE>", chat_id="C1"
    )
    assert out == "@Alice Example ping @Alice Example"


@pytest.mark.asyncio
async def test_unresolvable_mention_falls_back_to_id():
    # Resolution returns the bare ID; keep the message intact, don't empty it.
    adapter = _adapter_with_names({})
    out = await adapter._humanize_user_mentions("<@U07GHOST> hi", chat_id="C1")
    assert out == "@U07GHOST hi"


@pytest.mark.asyncio
async def test_no_mentions_returns_unchanged():
    adapter = _adapter_with_names({"U07ALICE": "Alice Example"})
    out = await adapter._humanize_user_mentions("plain text, no pings", chat_id="C1")
    assert out == "plain text, no pings"


# ----- _build_identity_prompt --------------------------------------------------

def test_identity_prompt_names_the_bot():
    adapter = _make_adapter()
    adapter._bot_display_name = "TestBot"
    adapter._team_bot_names = {}
    prompt = adapter._build_identity_prompt(team_id="T1")
    assert "@TestBot" in prompt
    # Must instruct that another participant's mention is not a self-mention.
    assert "not a mention of you" in prompt


def test_identity_prompt_prefers_per_team_name():
    adapter = _make_adapter()
    adapter._bot_display_name = "PrimaryBot"
    adapter._team_bot_names = {"T2": "WorkspaceTwoBot"}
    prompt = adapter._build_identity_prompt(team_id="T2")
    assert "@WorkspaceTwoBot" in prompt
    assert "PrimaryBot" not in prompt


def test_identity_prompt_empty_when_name_unknown():
    # Before connect (no name resolved) the prompt must be empty, not a
    # half-formed line — callers skip injecting an empty string.
    adapter = _make_adapter()
    adapter._bot_display_name = None
    adapter._team_bot_names = {}
    assert adapter._build_identity_prompt(team_id="T1") == ""


def test_identity_prompt_does_not_re_gate_directedness():
    """Adapter routing is the sole authority for whether an event reaches the
    model. The identity prompt grounds WHO the bot is; it must never instruct
    the model to re-decide WHETHER a routed message is directed at it — that
    second mention gate contradicts routed bare thread follow-ups (which by
    design carry no current-turn @mention) and makes them unstable/silent."""
    adapter = _make_adapter()
    adapter._bot_display_name = "TestBot"
    adapter._team_bot_names = {}
    prompt = adapter._build_identity_prompt(team_id="T1")
    assert "only treat a message as directed" not in prompt.lower()
    # Positive grounding must survive: the handle, and the other-participant
    # disambiguation.
    assert "@TestBot" in prompt
    assert "not a mention of you" in prompt


# ---------------------------------------------------------------------------
# Integration: real _handle_slack_message — routed bare thread follow-ups
# must not be second-guessed by the composed channel_prompt.
# (Harness mirrors tests/gateway/test_slack_ignore_other_user_mentions.py.)
# ---------------------------------------------------------------------------

BOT_USER_ID = "U_BOT_123"
CHANNEL_ID = "C0AQWDLHY9M"


@pytest.fixture
def wired_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT_USER_ID
    a._bot_display_name = "Hermes"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    # Keep gating driven by config.extra, not ambient env.
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("SLACK_STRICT_MENTION", raising=False)


def _event(text, ts, thread_ts=None):
    event = {
        "channel": CHANNEL_ID,
        "channel_type": "channel",
        "user": "U_HUMAN",
        "text": text,
        "ts": ts,
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return event


async def _run(adapter, event):
    with patch.object(
        adapter, "_resolve_user_name", new=AsyncMock(return_value="human")
    ), patch.object(
        adapter, "_fetch_thread_context", new=AsyncMock(return_value=None)
    ), patch.object(
        adapter, "_fetch_thread_parent_text", new=AsyncMock(return_value="")
    ), patch.object(
        adapter, "_has_active_session_for_thread", return_value=False
    ):
        await adapter._handle_slack_message(event)


@pytest.mark.asyncio
async def test_bare_followup_in_mentioned_thread_routes_without_mention_gate(
    wired_adapter,
):
    """Regression: a bare (unmentioned) follow-up in a previously-mentioned
    thread is routed by the adapter, and the composed channel_prompt must not
    instruct the model to require a current-turn @mention before treating the
    message as directed at it."""
    thread_ts = "1700000000.000100"
    wired_adapter._mentioned_threads.add(thread_ts)

    await _run(
        wired_adapter,
        _event("what about next week?", ts="1700000000.000101", thread_ts=thread_ts),
    )

    wired_adapter.handle_message.assert_awaited_once()
    msg_event = wired_adapter.handle_message.await_args.args[0]
    channel_prompt = msg_event.channel_prompt or ""
    # Identity grounding survives...
    assert "@Hermes" in channel_prompt
    assert "not a mention of you" in channel_prompt
    # ...but no second mention gate after routing already accepted the event.
    assert "only treat a message as directed" not in channel_prompt.lower()


@pytest.mark.asyncio
async def test_unrelated_unmentioned_top_level_message_stays_rejected(
    wired_adapter,
):
    """Adapter gates remain the authority: ambient top-level channel chatter
    with no mention, no mentioned thread, and no active session never reaches
    the agent."""
    await _run(wired_adapter, _event("ambient chatter", ts="1700000000.000200"))

    wired_adapter.handle_message.assert_not_awaited()
