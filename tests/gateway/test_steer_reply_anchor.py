import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(
            PlatformConfig(enabled=True, token="fake", typing_indicator=False),
            Platform("photon"),
        )
        self.sent = []

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to))
        return SendResult(success=True, message_id=f"out-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _event(message_id, *, chat_id="chat-1", internal=False):
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
async def test_latest_successful_steer_anchors_real_final_and_next_turn_resets():
    adapter = _Adapter()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(_event):
        entered.set()
        await release.wait()
        return "final"

    adapter.set_message_handler(handler)
    event_a = _event("A")
    session_key = build_session_key(event_a.source)
    assert adapter._start_session_processing(event_a, session_key)
    await entered.wait()

    assert adapter.update_active_turn_reply_anchor(session_key, _event("B"))
    assert adapter.update_active_turn_reply_anchor(session_key, _event("C"))
    release.set()
    await adapter._session_tasks[session_key]

    assert adapter.sent[-1] == ("chat-1", "final", "C")
    assert session_key not in adapter._active_turn_reply_anchors

    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="next"))
    await adapter._process_message_background(_event("D"), session_key)
    assert adapter.sent[-1] == ("chat-1", "next", "D")


@pytest.mark.asyncio
async def test_failed_synthetic_and_other_session_steers_cannot_steal_anchor():
    adapter = _Adapter()
    event_a = _event("A")
    session_key = build_session_key(event_a.source)
    other_key = build_session_key(_event("X", chat_id="chat-2").source)
    state = adapter.__dict__.setdefault("_active_turn_reply_anchors", {})
    from gateway.platforms.base import _ActiveTurnReplyAnchor

    anchor = _ActiveTurnReplyAnchor(event_a)
    state[session_key] = anchor

    assert not adapter.update_active_turn_reply_anchor(
        session_key, _event(None, internal=True)
    )
    assert not adapter.update_active_turn_reply_anchor(other_key, _event("X"))
    assert anchor.event.message_id == "A"
