"""Realtime WS transport contracts (ADR D4): session bootstrap payload,
cedar voice-echo verification, reasoning_effort degradation, event-name
mapping (GA + legacy), audio append, and cancel.

Uses an injected fake WebSocket — no network.
"""

import asyncio
import base64
import json

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


class FakeRealtimeWS:
    """Scripted provider socket: echoes session.updated on session.update."""

    def __init__(self, *, voice_echo="cedar", reject_reasoning=False):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False
        self._voice_echo = voice_echo
        self._reject_reasoning = reject_reasoning
        self._rejected_once = False

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "session.update":
            session = msg.get("session") or {}
            if (
                self._reject_reasoning
                and not self._rejected_once
                and "reasoning_effort" in session
            ):
                self._rejected_once = True
                await self.inbox.put(json.dumps({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unknown parameter: 'session.reasoning_effort'.",
                        "param": "session.reasoning_effort",
                    },
                }))
                return
            await self.inbox.put(json.dumps({
                "type": "session.updated",
                "session": {"voice": self._voice_echo, "model": "gpt-realtime-2.1"},
            }))

    async def recv(self):
        item = await self.inbox.get()
        if item is None:
            raise ConnectionError("closed")
        return item

    async def close(self):
        self.closed = True
        await self.inbox.put(None)

    async def push(self, event: dict):
        await self.inbox.put(json.dumps(event))


class FakeLane:
    """Records transport→lane callbacks."""

    def __init__(self):
        self.calls = []
        self.projection = "PROJECTION-BYTES-STABLE"

    def get_projection(self):
        return self.projection

    def session_tool_schemas(self):
        return [{"type": "function", "name": "hermes_dispatch"}]

    def on_user_speech_started(self):
        self.calls.append(("speech_started",))

    def on_user_speech_stopped(self):
        self.calls.append(("speech_stopped",))

    def on_response_created(self, response_id):
        self.calls.append(("response_created", response_id))

    def on_response_audio_delta(self, pcm):
        self.calls.append(("audio_delta", pcm))

    def on_output_transcript_delta(self, text, response_id):
        self.calls.append(("transcript_delta", text, response_id))

    def on_input_transcript(self, text):
        self.calls.append(("input_transcript", text))

    def on_response_done(self):
        self.calls.append(("response_done",))

    def on_function_call(self, name, arguments, call_id):
        self.calls.append(("function_call", name, arguments, call_id))

    def on_transport_closed(self, exc):
        self.calls.append(("closed",))


def _make_transport(ws=None, *, config=None, lane=None):
    from plugins.platforms.discord.realtime.transport import RealtimeTransport

    ws = ws or FakeRealtimeWS()
    lane = lane or FakeLane()
    config = config or load_realtime_voice_config({"enabled": True})

    async def _ws_connect(url, headers):
        _ws_connect.url = url
        _ws_connect.headers = headers
        return ws

    transport = RealtimeTransport(
        config=config, api_key="sk-test", lane=lane, ws_connect=_ws_connect
    )
    return transport, ws, lane, _ws_connect


class TestSessionBootstrap:
    @pytest.mark.asyncio
    async def test_session_update_payload_and_url(self):
        transport, ws, lane, connect = _make_transport()
        await transport.connect()
        assert "model=gpt-realtime-2.1" in connect.url
        auth = dict(connect.headers).get("Authorization")
        assert auth == "Bearer sk-test"

        update = next(m for m in ws.sent if m["type"] == "session.update")
        session = update["session"]
        assert session["voice"] == "cedar"
        assert session["reasoning_effort"] == "low"
        assert session["instructions"] == "PROJECTION-BYTES-STABLE"
        assert session["input_audio_format"] == "pcm16"
        assert session["output_audio_format"] == "pcm16"
        td = session["turn_detection"]
        assert td["type"] == "server_vad"
        assert td["interrupt_response"] is True
        assert session["tools"] == [{"type": "function", "name": "hermes_dispatch"}]
        await transport.close()

    @pytest.mark.asyncio
    async def test_voice_echo_mismatch_fails_connect(self):
        transport, ws, lane, _ = _make_transport(FakeRealtimeWS(voice_echo="alloy"))
        with pytest.raises(Exception) as exc_info:
            await transport.connect()
        assert "cedar" in str(exc_info.value)
        assert ws.closed is True

    @pytest.mark.asyncio
    async def test_reasoning_effort_rejection_degrades_by_omission(self):
        ws = FakeRealtimeWS(reject_reasoning=True)
        transport, ws, lane, _ = _make_transport(ws)
        await transport.connect()
        updates = [m for m in ws.sent if m["type"] == "session.update"]
        assert len(updates) == 2
        assert "reasoning_effort" in updates[0]["session"]
        assert "reasoning_effort" not in updates[1]["session"]
        await transport.close()

    @pytest.mark.asyncio
    async def test_instructions_byte_stable_across_connects(self):
        transport1, ws1, lane, _ = _make_transport()
        await transport1.connect()
        await transport1.close()
        ws2 = FakeRealtimeWS()
        transport2, ws2, _, _ = _make_transport(ws2, lane=lane)
        await transport2.connect()
        await transport2.close()
        i1 = next(m for m in ws1.sent if m["type"] == "session.update")["session"]["instructions"]
        i2 = next(m for m in ws2.sent if m["type"] == "session.update")["session"]["instructions"]
        assert i1 == i2


class TestEventMapping:
    @pytest.mark.asyncio
    async def test_ga_and_legacy_event_names_map_to_lane(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()

        pcm = b"\x01\x02\x03\x04"
        b64 = base64.b64encode(pcm).decode()
        events = [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created", "response": {"id": "resp_1"}},
            # GA name and legacy name must both deliver audio.
            {"type": "response.output_audio.delta", "delta": b64},
            {"type": "response.audio.delta", "delta": b64},
            {"type": "response.output_audio_transcript.delta", "delta": "38 mill", "response_id": "resp_1"},
            {"type": "response.audio_transcript.delta", "delta": "ion", "response_id": "resp_1"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "check deposits"},
            {"type": "response.done", "response": {"id": "resp_1"}},
        ]
        for e in events:
            await ws.push(e)
        await asyncio.sleep(0.05)
        await transport.close()

        assert ("speech_started",) in lane.calls
        assert ("speech_stopped",) in lane.calls
        assert ("response_created", "resp_1") in lane.calls
        assert lane.calls.count(("audio_delta", pcm)) == 2
        assert ("transcript_delta", "38 mill", "resp_1") in lane.calls
        assert ("transcript_delta", "ion", "resp_1") in lane.calls
        assert ("input_transcript", "check deposits") in lane.calls
        assert ("response_done",) in lane.calls

    @pytest.mark.asyncio
    async def test_function_call_events_map_to_lane(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        await ws.push({
            "type": "response.function_call_arguments.done",
            "name": "hermes_dispatch",
            "arguments": json.dumps({"task": "check deposits"}),
            "call_id": "call_9",
        })
        await asyncio.sleep(0.05)
        await transport.close()
        assert ("function_call", "hermes_dispatch", {"task": "check deposits"}, "call_9") in lane.calls


class TestOutbound:
    @pytest.mark.asyncio
    async def test_append_audio_base64(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        await transport.append_audio(b"\x0a\x0b")
        await transport.close()
        append = next(m for m in ws.sent if m["type"] == "input_audio_buffer.append")
        assert base64.b64decode(append["audio"]) == b"\x0a\x0b"

    @pytest.mark.asyncio
    async def test_cancel_response_nowait_sends_cancel(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        transport.cancel_response_nowait()
        await asyncio.sleep(0.05)
        await transport.close()
        assert any(m["type"] == "response.cancel" for m in ws.sent)

    @pytest.mark.asyncio
    async def test_inject_item_and_function_output(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        await transport.inject_item("system", "worker event text")
        await transport.send_function_output("call_9", '{"ok": true}')
        await transport.create_response(instructions="frame only; no figures")
        await transport.close()

        item = next(m for m in ws.sent if m["type"] == "conversation.item.create"
                    and m["item"]["role"] == "system")
        assert item["item"]["content"][0]["text"] == "worker event text"
        fn_out = next(m for m in ws.sent if m["type"] == "conversation.item.create"
                      and m["item"].get("type") == "function_call_output")
        assert fn_out["item"]["call_id"] == "call_9"
        resp = next(m for m in ws.sent if m["type"] == "response.create")
        assert resp["response"]["instructions"] == "frame only; no figures"

    @pytest.mark.asyncio
    async def test_transport_closed_notifies_lane(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        await ws.close()  # peer-side close → reader sees error
        await asyncio.sleep(0.05)
        assert ("closed",) in lane.calls
