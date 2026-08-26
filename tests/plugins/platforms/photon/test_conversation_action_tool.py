import json

import pytest

from gateway.platforms.base import SendResult
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
