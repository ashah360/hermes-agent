import asyncio
import json

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key
from plugins.platforms.photon import conversation_actions as actions
from plugins.platforms.photon.tools import CONVERSATION_ACTION_SCHEMA
from toolsets import resolve_toolset


class _Adapter:
    conversation_actions_enabled = True
    is_connected = True

    def __init__(self):
        self.calls = []

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.calls.append(("react", chat_id, message_id, emoji))
        return {"success": True}

    async def remove_reaction(self, chat_id, message_id=None):
        self.calls.append(("unreact", chat_id, message_id))
        return {"success": True}

    async def send(self, chat_id, text, reply_to=None):
        self.calls.append(("reply", chat_id, reply_to, text))
        return SendResult(
            success=True,
            message_id="reply-guid",
            raw_response={"message_ids": ["reply-guid"]},
        )

    async def send_image_group(self, chat_id, paths, caption=None):
        self.calls.append(("images", chat_id, list(paths), caption))
        return SendResult(
            success=True,
            message_id="parent",
            raw_response={
                "parent_message_id": "parent",
                "child_message_ids": ["child-0", "child-1"],
                "part_count": 2,
            },
        )


def test_schema_exposes_no_arbitrary_destination_and_only_photon_bundle():
    properties = CONVERSATION_ACTION_SCHEMA["parameters"]["properties"]
    assert not ({"chat_id", "recipient", "phone", "platform"} & properties.keys())
    assert "photon_conversation_action" in resolve_toolset("hermes-photon")
    assert "photon_conversation_action" not in resolve_toolset("hermes-telegram")


def test_gateway_toolset_resolution_scopes_action_to_photon():
    from hermes_cli.tools_config import _get_platform_tools

    config = {"platform_toolsets": {}}
    assert "hermes-photon" in _get_platform_tools(config, "photon")
    assert "hermes-photon" not in _get_platform_tools(config, "telegram")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {
                "platforms": {
                    "photon": {"extra": {"conversation_actions_enabled": True}}
                }
            },
            True,
        ),
        (
            {
                "gateway": {
                    "platforms": {
                        "photon": {
                            "extra": {"conversation_actions_enabled": True}
                        }
                    }
                }
            },
            True,
        ),
        (
            {
                "platforms": {
                    "photon": {"extra": {"conversation_actions_enabled": False}}
                },
                "gateway": {
                    "platforms": {
                        "photon": {
                            "extra": {"conversation_actions_enabled": True}
                        }
                    }
                },
            },
            False,
        ),
    ],
)
def test_config_gate_matches_gateway_platform_precedence(
    monkeypatch, config, expected
):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config_readonly", lambda: config)
    assert actions.conversation_actions_configured() is expected


@pytest.mark.asyncio
async def test_action_derives_current_chat_and_resolves_earlier_exact_target(
    monkeypatch,
):
    adapter = _Adapter()
    runner = object()
    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (runner, adapter, "current-chat", "current-session-key"),
    )
    monkeypatch.setattr(
        actions,
        "_resolve_session_id",
        lambda *_args: "durable-session",
    )
    monkeypatch.setattr(
        actions,
        "resolve_target_message_id",
        lambda **kwargs: ("earlier-guid", None),
    )
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "trigger-guid" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    receipt = json.loads(
        await actions.conversation_action_tool(
            {
                "action": "react",
                "target": {"messages_back": 2},
                "emoji": "🫡",
            },
            session_id="durable-session",
        )
    )

    assert receipt["success"] is True
    assert receipt["target_message_id"] == "earlier-guid"
    assert adapter.calls == [
        ("react", "current-chat", "earlier-guid", "🫡")
    ]


@pytest.mark.asyncio
async def test_reply_and_images_return_redacted_native_receipts(monkeypatch):
    adapter = _Adapter()
    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "current-chat", "session-key"),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "session-id")
    monkeypatch.setattr(
        actions,
        "resolve_target_message_id",
        lambda **kwargs: ("target-guid", None),
    )
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "trigger-guid" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    reply = json.loads(
        await actions.conversation_action_tool(
            {
                "action": "reply",
                "target": {"trigger": True},
                "text": "in thread",
            }
        )
    )
    images = json.loads(
        await actions.conversation_action_tool(
            {
                "action": "present_images",
                "images": ["/private/one.png", "/private/two.png"],
                "caption": "compare",
            }
        )
    )

    assert reply["presentation"] == "reply"
    assert reply["sent_message_ids"] == ["reply-guid"]
    assert images["presentation"] == "group"
    assert images["sent_message_ids"] == ["parent", "child-0", "child-1"]
    serialized = json.dumps(images)
    assert "current-chat" not in serialized
    assert "/private/" not in serialized


@pytest.mark.asyncio
async def test_content_actions_suppress_final_unless_follow_up_is_explicit(
    monkeypatch,
):
    from tools.approval import (
        reset_current_observability_context,
        set_current_observability_context,
    )

    adapter = _Adapter()
    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "current-chat", "session-key"),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "session-id")
    monkeypatch.setattr(
        actions,
        "resolve_target_message_id",
        lambda **kwargs: ("target-guid", None),
    )
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "trigger-guid" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    tokens = set_current_observability_context(turn_id="turn-content")
    try:
        await actions.conversation_action_tool(
            {
                "action": "reply",
                "target": {"trigger": True},
                "text": "already delivered",
            },
            session_id="session-id",
        )
    finally:
        reset_current_observability_context(tokens)
    assert (
        actions.suppress_redundant_final(
            "ceremonial duplicate",
            platform="photon",
            turn_id="turn-content",
        )
        == "[SILENT]"
    )
    from gateway.response_filters import is_intentional_silence_response

    assert is_intentional_silence_response("[SILENT]") is True
    # The marker is one-turn state and cannot suppress a later response.
    assert (
        actions.suppress_redundant_final(
            "later turn",
            platform="photon",
            turn_id="turn-later",
        )
        is None
    )

    await actions.conversation_action_tool(
        {
            "action": "present_images",
            "images": ["/private/one.png", "/private/two.png"],
            "allow_follow_up": True,
        },
        session_id="session-id",
        turn_id="turn-follow-up",
    )
    assert (
        actions.suppress_redundant_final(
            "useful additional context",
            platform="photon",
            turn_id="turn-follow-up",
        )
        is None
    )


@pytest.mark.asyncio
async def test_reactions_never_suppress_normal_final(monkeypatch):
    adapter = _Adapter()
    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "current-chat", "session-key"),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "reaction-session")
    monkeypatch.setattr(
        actions,
        "resolve_target_message_id",
        lambda **kwargs: ("target-guid", None),
    )
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "trigger-guid" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    await actions.conversation_action_tool(
        {
            "action": "react",
            "target": {"trigger": True},
            "emoji": "❤️",
        },
        session_id="reaction-session",
        turn_id="turn-reaction",
    )
    assert (
        actions.suppress_redundant_final(
            "normal response",
            platform="photon",
            turn_id="turn-reaction",
        )
        is None
    )


class _AnchorAdapter(BasePlatformAdapter):
    """Real base-adapter anchor lifecycle + minimal Photon action surface."""

    conversation_actions_enabled = True
    is_connected = True

    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake", typing_indicator=False),
            Platform("photon"),
        )
        self.reactions = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="out-1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.reactions.append((chat_id, message_id, emoji))
        return {"success": True}


def _anchor_event(message_id, *, chat_id="chat-1", internal=False):
    return MessageEvent(
        text=message_id or "synthetic",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform("photon"),
            chat_id=chat_id,
            chat_type="dm",
            user_id="user-1",
        ),
        message_id=message_id,
        internal=internal,
    )


@pytest.mark.asyncio
async def test_trigger_follows_latest_successful_steer_and_resets_next_turn(
    monkeypatch,
):
    """Live regression: successful mid-turn steers moved final delivery to
    the latest message but ``target={trigger: true}`` kept resolving the
    turn-START binding — successive reactions all landed on the first
    bubble. The trigger must read the same live anchor as final delivery:
    latest successful steer wins; failed/synthetic/cross-session events
    cannot steal it; turn completion clears it so the next turn owns its own
    trigger.
    """
    adapter = _AnchorAdapter()
    event_a = _anchor_event("prov-A")
    session_key = build_session_key(event_a.source)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        entered.set()
        await release.wait()
        return "final"

    adapter.set_message_handler(handler)
    assert adapter._start_session_processing(event_a, session_key)
    await entered.wait()

    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "chat-1", session_key),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "session-id")
    # The turn-START binding never changes for the life of the turn.
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "prov-A" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    async def _react_trigger():
        return json.loads(
            await actions.conversation_action_tool(
                {"action": "react", "target": {"trigger": True}, "emoji": "❤️"}
            )
        )

    # Before any steer the anchor IS the turn-start event.
    assert (await _react_trigger())["target_message_id"] == "prov-A"

    # Latest successful steer owns the trigger: B then C -> C.
    assert adapter.update_active_turn_reply_anchor(session_key, _anchor_event("prov-B"))
    assert adapter.update_active_turn_reply_anchor(session_key, _anchor_event("prov-C"))
    receipt = await _react_trigger()
    assert receipt["target_message_id"] == "prov-C"
    assert adapter.reactions[-1] == ("chat-1", "prov-C", "❤️")

    # Failed (no provider id), synthetic, and cross-session events cannot
    # steal the anchor.
    assert not adapter.update_active_turn_reply_anchor(
        session_key, _anchor_event(None)
    )
    assert not adapter.update_active_turn_reply_anchor(
        session_key, _anchor_event(None, internal=True)
    )
    other_key = build_session_key(_anchor_event("prov-X", chat_id="chat-2").source)
    assert not adapter.update_active_turn_reply_anchor(
        other_key, _anchor_event("prov-X")
    )
    assert (await _react_trigger())["target_message_id"] == "prov-C"

    # Turn completion clears the anchor.
    release.set()
    await adapter._session_tasks[session_key]
    assert adapter.active_turn_reply_anchor_message_id(session_key) is None

    # Next turn D: its own anchor wins even though the mocked env still
    # reports the stale turn-A binding.
    release.clear()
    entered.clear()
    assert adapter._start_session_processing(_anchor_event("prov-D"), session_key)
    await entered.wait()
    assert (await _react_trigger())["target_message_id"] == "prov-D"
    release.set()
    await adapter._session_tasks[session_key]

    # With no active turn at all, the tool falls back to the env binding.
    assert (await _react_trigger())["target_message_id"] == "prov-A"


@pytest.mark.asyncio
async def test_debounced_bubbles_stay_individually_addressable(
    monkeypatch, tmp_path
):
    """Three rapid bubbles A/B/C debounce into ONE combined agent turn, but
    each real provider bubble stays addressable: trigger resolves the latest
    constituent (C), messages_back 1/2 resolve B/A within the live turn, and
    after the turn completes the next turn D resolves trigger D with
    messages_back 1 continuing into C through persisted constituent identity.
    """
    from hermes_state import SessionDB

    adapter = _AnchorAdapter()
    session_key = build_session_key(_anchor_event("spc-A").source)

    # Rapid same-sender burst through the REAL debounce + pending-slot merge.
    await adapter._queue_text_debounce(session_key, _anchor_event("spc-A"))
    await adapter._queue_text_debounce(session_key, _anchor_event("spc-B"))
    await adapter._queue_text_debounce(session_key, _anchor_event("spc-C"))
    assert await adapter._flush_text_debounce_now(session_key)
    merged = adapter._pending_messages.pop(session_key)

    # One coherent model turn, arrival order; exact per-bubble identity kept.
    assert merged.text == "spc-A\nspc-B\nspc-C"
    assert [
        (entry.text, entry.message_id) for entry in merged.constituent_messages
    ] == [("spc-A", "spc-A"), ("spc-B", "spc-B"), ("spc-C", "spc-C")]

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        entered.set()
        await release.wait()
        return "final"

    adapter.set_message_handler(handler)
    assert adapter._start_session_processing(merged, session_key)
    await entered.wait()

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("photon-session", source="photon")
    monkeypatch.setattr(
        actions,
        "_open_session_db",
        lambda: SessionDB(db_path=db_path, read_only=True),
    )
    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "chat-1", session_key),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "photon-session")
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "spc-C" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    async def _target(spec):
        return json.loads(
            await actions.conversation_action_tool(
                {"action": "react", "target": spec, "emoji": "❤️"}
            )
        )["target_message_id"]

    assert await _target({"trigger": True}) == "spc-C"
    assert await _target({"messages_back": 1}) == "spc-B"
    assert await _target({"messages_back": 2}) == "spc-A"

    # Persist the combined turn exactly as the gateway does: ONE user row
    # (role alternation intact), latest constituent as its platform id, the
    # full ordered constituent identity in DB-only display metadata.
    db.append_message(
        "photon-session",
        "user",
        "spc-A\nspc-B\nspc-C",
        platform_message_id="spc-C",
        display_metadata={"constituent_message_ids": ["spc-A", "spc-B", "spc-C"]},
    )
    db.append_message("photon-session", "assistant", "final")

    # Turn completion clears the live anchor.
    release.set()
    await adapter._session_tasks[session_key]
    assert adapter.active_turn_constituent_message_ids(session_key) == []

    # Next turn D: trigger is D; messages_back walks back into the persisted
    # constituents (C, then B, then A) without duplicates.
    release.clear()
    entered.clear()
    assert adapter._start_session_processing(_anchor_event("spc-D"), session_key)
    await entered.wait()
    assert await _target({"trigger": True}) == "spc-D"
    assert await _target({"messages_back": 1}) == "spc-C"
    assert await _target({"messages_back": 2}) == "spc-B"
    assert await _target({"messages_back": 3}) == "spc-A"
    release.set()
    await adapter._session_tasks[session_key]
    db.close()


@pytest.mark.asyncio
async def test_oob_steer_accepted_mid_tool_call_owns_next_trigger(monkeypatch):
    """Live race on 59b5ec5e97: active turn A/B was mid-tool-call when a new
    real inbound bubble C was accepted as an out-of-band steer and appended
    to the running agent context; the immediately following
    target={trigger: true} still resolved B. Hard invariant: the moment the
    gateway accepts an OOB payload into the running turn (REAL busy-steer
    path), the live anchor already names C — before the blocked model/tool
    step is even released — so the next conversation action resolves C.
    """
    from unittest.mock import AsyncMock, MagicMock

    import gateway.run as gateway_run
    from gateway.run import GatewayRunner

    adapter = _AnchorAdapter()

    # Active turn: debounced A/B (two distinct provider bubbles, one turn).
    turn_event = _anchor_event("spc-A")
    turn_event.text = "I’m sending\nA bunch"
    from gateway.platforms.base import ConstituentMessage

    turn_event.constituent_messages = [
        ConstituentMessage(text="I’m sending", message_id="spc-A"),
        ConstituentMessage(text="A bunch", message_id="spc-B"),
    ]
    turn_event.message_id = "spc-B"
    session_key = build_session_key(turn_event.source)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        entered.set()
        await release.wait()  # the blocked model/tool step
        return "final"

    adapter.set_message_handler(handler)
    assert adapter._start_session_processing(turn_event, session_key)
    await entered.wait()
    assert adapter.active_turn_constituent_message_ids(session_key) == [
        "spc-A",
        "spc-B",
    ]

    # Real busy-steer ingress (gateway runner), agent still blocked.
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_input_mode = "steer"
    runner._busy_text_mode = "steer"
    runner.adapters = {}
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._is_user_authorized = lambda _source: True
    runner._adapter_for_source = lambda _source: adapter
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")

    running_agent = MagicMock()
    running_agent.steer = MagicMock(return_value=True)
    runner._running_agents[session_key] = running_agent

    event_c = _anchor_event("spc-C")
    event_c.text = "In succession"
    handled = await runner._handle_active_session_busy_message(
        event_c, session_key
    )
    assert handled is True
    running_agent.steer.assert_called_once_with("In succession")

    # The invariant: BEFORE the blocked step is released, the anchor already
    # names C as the latest addressable bubble.
    assert adapter.active_turn_constituent_message_ids(session_key) == [
        "spc-A",
        "spc-B",
        "spc-C",
    ]

    monkeypatch.setattr(
        actions,
        "_resolve_runtime",
        lambda: (object(), adapter, "chat-1", session_key),
    )
    monkeypatch.setattr(actions, "_resolve_session_id", lambda *_args: "session-id")
    monkeypatch.setattr(
        actions,
        "get_session_env",
        lambda name, default="": (
            "spc-B" if name == "HERMES_SESSION_MESSAGE_ID" else default
        ),
    )

    async def _target(spec):
        return json.loads(
            await actions.conversation_action_tool(
                {"action": "react", "target": spec, "emoji": "3️⃣"}
            )
        )["target_message_id"]

    # The next action after the OOB message resolves it as the trigger; the
    # pre-steer bubbles remain addressable behind it.
    assert await _target({"trigger": True}) == "spc-C"
    assert await _target({"messages_back": 1}) == "spc-B"
    assert await _target({"messages_back": 2}) == "spc-A"

    release.set()
    await adapter._session_tasks[session_key]


@pytest.mark.asyncio
async def test_non_photon_context_fails_without_dispatch(monkeypatch):
    monkeypatch.setattr(actions, "_resolve_runtime", lambda: (None, None, "", ""))

    result = json.loads(
        await actions.conversation_action_tool(
            {
                "action": "react",
                "target": {"trigger": True},
                "emoji": "❤️",
            }
        )
    )

    assert result["success"] is False
    assert result["error"] == "current_conversation_unavailable"
