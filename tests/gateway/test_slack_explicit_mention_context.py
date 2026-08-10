"""Regression tests: explicit @mention intent must survive mention stripping.

Production symptom (session 20260810_202952_ac9d1fd5): in a multi-user Slack
thread, an allowlisted (non-principal) user explicitly @-mentions the bot. The
adapter routes the event, but ``_handle_slack_message`` strips ``<@bot_uid>``
from the text before building the MessageEvent, while the ephemeral identity
prompt tells the model to "only treat a message as directed at you when it
mentions @<name> specifically". The model therefore sees an unaddressed message
plus an instruction not to answer unaddressed messages — and returns NO_REPLY
to a message that explicitly summoned it.

Fix under test: when routing has already proved ``is_mentioned``, the adapter
appends an ephemeral statement to ``channel_prompt`` (the same per-turn,
never-persisted seam as the identity prompt — prompt caching preserved) saying
the current message explicitly mentioned the bot and is directed at it. The
raw Slack mention markup is still stripped from canonical command/text
processing, and unmentioned accepted messages (free-response channels,
auto-followed threads) must NOT receive that assertion.
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


_ensure_slack_mock()

import plugins.platforms.slack.adapter as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


BOT_USER_ID = "U_BOT_123"
BOT_NAME = "ghost"
CHANNEL_ID = "C0AQWDLHY9M"

# Stable marker phrase the ephemeral directed-context statement must carry.
DIRECTED_MARKER = "directed at you"
# The identity prompt also exists in channel_prompt; its rule line must not be
# confused with the per-turn directed statement, so the directed statement is
# detected by its "explicitly mentioned" claim about the CURRENT message.
EXPLICIT_MENTION_MARKER = "explicitly"


# ---------------------------------------------------------------------------
# Integration fixture: real SlackAdapter + real _handle_slack_message
# (same pattern as tests/gateway/test_slack_ignore_other_user_mentions.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="xoxb-fake-token")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._bot_user_id = BOT_USER_ID
    a._bot_display_name = BOT_NAME
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # Keep gating driven by config.extra, not ambient env.
    monkeypatch.delenv("SLACK_FREE_RESPONSE_CHANNELS", raising=False)
    monkeypatch.delenv("SLACK_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("SLACK_STRICT_MENTION", raising=False)
    monkeypatch.delenv("SLACK_THREAD_REQUIRE_MENTION", raising=False)
    monkeypatch.delenv("SLACK_MENTION_PATTERNS", raising=False)


def _event(text, ts, thread_ts=None):
    event = {
        "channel": CHANNEL_ID,
        "channel_type": "channel",
        "user": "U_HUMAN",
        "client_msg_id": f"cmid-{ts}",
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
        adapter, "_collect_thread_root_images",
        new=AsyncMock(return_value=([], [])),
    ), patch.object(
        adapter, "_has_active_session_for_thread", return_value=False
    ):
        await adapter._handle_slack_message(event)


def _dispatched_event(adapter):
    adapter.handle_message.assert_awaited_once()
    return adapter.handle_message.await_args.args[0]


def _has_directed_statement(channel_prompt):
    prompt = channel_prompt or ""
    return EXPLICIT_MENTION_MARKER in prompt and DIRECTED_MARKER in prompt


# ---------------------------------------------------------------------------
# The regression: explicit @mention → model-visible directed context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_explicit_mention_gets_directed_context_in_channel_prompt(adapter):
    """An explicitly @-mentioned message must carry model-visible per-turn
    context stating the current message mentioned the bot and is directed at
    it — the mention markup itself is stripped from the text, so without this
    the identity prompt's "only when it mentions @ghost" rule yields NO_REPLY."""
    await _run(
        adapter,
        _event(f"<@{BOT_USER_ID}> can you check the deploy?", ts="1700000000.000100"),
    )

    msg_event = _dispatched_event(adapter)
    # Premise: the raw mention was stripped from the canonical text.
    assert f"<@{BOT_USER_ID}>" not in msg_event.text
    # The fix: channel_prompt states this turn explicitly mentioned the bot.
    assert _has_directed_statement(msg_event.channel_prompt), (
        "explicitly mentioned message has no model-visible directed-at-bot "
        f"context; channel_prompt={msg_event.channel_prompt!r}"
    )
    # The bot's handle is named so the statement lines up with the identity
    # prompt's "@ghost" rule.
    assert f"@{BOT_NAME}" in (msg_event.channel_prompt or "")


@pytest.mark.asyncio
async def test_mid_thread_mention_gets_directed_context(adapter):
    """The reported production shape: a non-principal user @-mentions the bot
    mid-thread. The dispatched turn must carry the directed statement."""
    thread_ts = "1700000000.000200"
    await _run(
        adapter,
        _event(
            f"<@{BOT_USER_ID}> what do you think?",
            ts="1700000000.000201",
            thread_ts=thread_ts,
        ),
    )

    msg_event = _dispatched_event(adapter)
    assert f"<@{BOT_USER_ID}>" not in msg_event.text
    assert _has_directed_statement(msg_event.channel_prompt)


# ---------------------------------------------------------------------------
# No weakening: unmentioned accepted messages must NOT look directed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unmentioned_free_response_message_has_no_directed_context(adapter):
    """A free-response-channel message accepted WITHOUT a mention must not be
    dressed up as directed at the bot — ambient chatter stays ambient."""
    adapter.config.extra["free_response_channels"] = CHANNEL_ID

    await _run(adapter, _event("anyone up for lunch?", ts="1700000000.000300"))

    msg_event = _dispatched_event(adapter)
    assert not _has_directed_statement(msg_event.channel_prompt), (
        "unmentioned free-response message must not claim it explicitly "
        f"mentioned the bot; channel_prompt={msg_event.channel_prompt!r}"
    )
    # The stable identity prompt is still present and unchanged.
    assert f"@{BOT_NAME}" in (msg_event.channel_prompt or "")
    assert "not a mention of you" in (msg_event.channel_prompt or "")


@pytest.mark.asyncio
async def test_unmentioned_followup_in_mentioned_thread_has_no_directed_context(adapter):
    """A plain follow-up in an auto-followed (previously mentioned) thread is
    accepted without a mention — it must not receive the directed statement."""
    thread_ts = "1700000000.000400"
    adapter._mentioned_threads.add(thread_ts)

    await _run(
        adapter,
        _event("thanks, that makes sense", ts="1700000000.000401", thread_ts=thread_ts),
    )

    msg_event = _dispatched_event(adapter)
    assert not _has_directed_statement(msg_event.channel_prompt)


# ---------------------------------------------------------------------------
# Composition: identity prompt + per-channel prompt + directed statement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_directed_context_composes_with_channel_prompt_and_identity(adapter):
    """The directed statement must compose with (not overwrite) the identity
    prompt and a configured per-channel prompt."""
    adapter.config.extra["channel_prompts"] = {CHANNEL_ID: "Be terse."}

    await _run(
        adapter,
        _event(f"<@{BOT_USER_ID}> status?", ts="1700000000.000500"),
    )

    msg_event = _dispatched_event(adapter)
    prompt = msg_event.channel_prompt or ""
    assert "not a mention of you" in prompt      # identity prompt intact
    assert "Be terse." in prompt                 # per-channel prompt intact
    assert _has_directed_statement(prompt)       # directed statement added


@pytest.mark.asyncio
async def test_unmentioned_message_keeps_bare_channel_prompt(adapter):
    """Per-channel prompt still resolves on unmentioned accepted messages,
    without any directed statement appended."""
    adapter.config.extra["free_response_channels"] = CHANNEL_ID
    adapter.config.extra["channel_prompts"] = {CHANNEL_ID: "Be terse."}

    await _run(adapter, _event("morning all", ts="1700000000.000600"))

    msg_event = _dispatched_event(adapter)
    prompt = msg_event.channel_prompt or ""
    assert "Be terse." in prompt
    assert not _has_directed_statement(prompt)


# ---------------------------------------------------------------------------
# Unit: the directed-statement builder itself
# ---------------------------------------------------------------------------

def test_mention_directed_prompt_names_the_bot():
    adapter = object.__new__(SlackAdapter)
    adapter._bot_display_name = BOT_NAME
    adapter._team_bot_names = {}
    statement = adapter._build_mention_directed_prompt(team_id="T1")
    assert f"@{BOT_NAME}" in statement
    assert _has_directed_statement(statement)


def test_mention_directed_prompt_prefers_per_team_name():
    adapter = object.__new__(SlackAdapter)
    adapter._bot_display_name = "PrimaryBot"
    adapter._team_bot_names = {"T2": "WorkspaceTwoBot"}
    statement = adapter._build_mention_directed_prompt(team_id="T2")
    assert "@WorkspaceTwoBot" in statement
    assert "PrimaryBot" not in statement


def test_mention_directed_prompt_works_without_a_name():
    """Unlike the identity prompt, the directed statement is meaningful even
    before the bot's display name resolves — routing proved the mention, so
    the turn must still be marked as directed."""
    adapter = object.__new__(SlackAdapter)
    adapter._bot_display_name = None
    adapter._team_bot_names = {}
    statement = adapter._build_mention_directed_prompt(team_id="T1")
    assert _has_directed_statement(statement)
