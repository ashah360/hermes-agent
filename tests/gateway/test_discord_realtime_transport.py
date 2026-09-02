"""Realtime WS transport contracts against the CURRENT GA API shape.

The live provider rejected the beta payload with
``invalid_request_error.beta_api_shape_disabled`` (WS close 4000) — mocks had
hidden the contract gap. The fake socket here is therefore GA-STRICT: any
beta-era key (root ``voice``/``turn_detection``/``input_audio_format``/
``modalities``/``reasoning_effort``, or a missing ``session.type``) is
rejected exactly like the live API, so a beta-shaped transport can never go
green again.

GA source of truth (developers.openai.com, Realtime guides + API reference):
- no ``OpenAI-Beta`` header;
- ``session.type: "realtime"``, ``model`` inside the session;
- ``output_modalities: ["audio"]``;
- ``audio.input.format = {type: audio/pcm, rate: 24000}``,
  ``audio.input.turn_detection = {type: semantic_vad, ...}``;
- ``audio.output.format = {type: audio/pcm, rate: 24000}``,
  ``audio.output.voice = cedar``;
- reasoning effort as ``reasoning: {effort}`` (object, not a root scalar);
- GA response events (``response.output_audio.delta`` etc.).
"""

import asyncio
import base64
import json

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config

# Beta-era session keys the GA API refuses (the live failure class).
_BETA_SESSION_KEYS = {
    "voice", "turn_detection", "input_audio_format", "output_audio_format",
    "input_audio_transcription", "modalities", "reasoning_effort",
}


class FakeRealtimeWS:
    """GA-strict scripted provider socket."""

    def __init__(self, *, voice_echo="cedar", reject_reasoning=False):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False
        self._voice_echo = voice_echo
        self._reject_reasoning = reject_reasoning
        self._rejected_once = False

    def _validate_ga_session(self, session: dict):
        beta_present = sorted(_BETA_SESSION_KEYS & set(session))
        if beta_present or session.get("type") != "realtime":
            return {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "beta_api_shape_disabled",
                    "message": (
                        "The beta Realtime API shape is disabled; migrate to "
                        f"the GA session schema (offending: {beta_present or 'missing type'})."
                    ),
                },
            }
        td = ((session.get("audio") or {}).get("input") or {}).get("turn_detection")
        return self._validate_turn_detection(td)

    @staticmethod
    def _validate_turn_detection(td):
        """Real-shape rejection: server-only tuning fields are invalid on
        semantic_vad, and unknown VAD types are refused."""
        if td is None:
            return None
        ttype = td.get("type")
        server_only = {"threshold", "prefix_padding_ms", "silence_duration_ms"}
        if ttype == "semantic_vad":
            bad = sorted(server_only & set(td))
            if bad:
                return {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "unknown_parameter",
                        "message": (
                            "Unknown parameter for semantic_vad: "
                            + ", ".join(f"turn_detection.{b}" for b in bad)
                        ),
                    },
                }
            return None
        if ttype == "server_vad":
            return None
        return {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "invalid_value",
                "message": f"Invalid turn_detection.type: {ttype!r}",
            },
        }

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "session.update":
            session = msg.get("session") or {}
            err = self._validate_ga_session(session)
            if err is not None:
                await self.inbox.put(json.dumps(err))
                return
            if (
                self._reject_reasoning
                and not self._rejected_once
                and "reasoning" in session
            ):
                self._rejected_once = True
                await self.inbox.put(json.dumps({
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Unknown parameter: 'session.reasoning'.",
                        "param": "session.reasoning",
                    },
                }))
                return
            await self.inbox.put(json.dumps({
                "type": "session.updated",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "output_modalities": ["audio"],
                    "audio": {
                        "output": {
                            "voice": self._voice_echo,
                            "format": {"type": "audio/pcm", "rate": 24000},
                        }
                    },
                },
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


class TestGASessionBootstrap:
    @pytest.mark.asyncio
    async def test_exact_ga_session_payload_and_headers(self):
        transport, ws, lane, connect = _make_transport()
        await transport.connect()
        assert "model=gpt-realtime-2.1" in connect.url
        headers = dict(connect.headers)
        assert headers.get("Authorization") == "Bearer sk-test"
        # GA interface: the beta header must be GONE.
        assert "OpenAI-Beta" not in headers

        update = next(m for m in ws.sent if m["type"] == "session.update")
        session = update["session"]
        assert session["type"] == "realtime"
        assert session["model"] == "gpt-realtime-2.1"
        assert session["output_modalities"] == ["audio"]
        assert session["audio"]["input"]["format"] == {
            "type": "audio/pcm", "rate": 24000,
        }
        # Production default is server_vad with explicit tuning: live logs
        # showed semantic_vad holding turns open 12-29s; server_vad measured
        # ~590ms speech-end->speech_stopped in the provider canary.
        td = session["audio"]["input"]["turn_detection"]
        assert td["type"] == "server_vad"
        assert td["threshold"] == 0.5
        assert td["prefix_padding_ms"] == 300
        assert td["silence_duration_ms"] == 500
        # Single cancel owner: the LANE owns interruption (local clear +
        # generation-bound provider cancel); provider auto-cancel is OFF or
        # the two race into response_cancel_not_active (live evidence).
        assert td["interrupt_response"] is False
        assert td["create_response"] is True
        assert session["audio"]["output"]["format"] == {
            "type": "audio/pcm", "rate": 24000,
        }
        assert session["audio"]["output"]["voice"] == "cedar"
        # Reasoning effort is the GA nested object, never a root scalar.
        assert session["reasoning"] == {"effort": "low"}
        assert session["instructions"] == "PROJECTION-BYTES-STABLE"
        assert session["tools"] == [{"type": "function", "name": "hermes_dispatch"}]
        # No beta-era keys anywhere in the session payload.
        assert not (_BETA_SESSION_KEYS & set(session))
        await transport.close()

    @pytest.mark.asyncio
    async def test_beta_shape_is_rejected_by_ga_provider(self):
        # Pin the failure class the live canary hit: the GA-strict fake must
        # refuse a beta-shaped session with beta_api_shape_disabled.
        ws = FakeRealtimeWS()
        beta_session = {
            "voice": "cedar",
            "instructions": "x",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "interrupt_response": True},
            "reasoning_effort": "low",
        }
        err = ws._validate_ga_session(beta_session)
        assert err is not None
        assert err["error"]["code"] == "beta_api_shape_disabled"

    @pytest.mark.asyncio
    async def test_voice_echo_mismatch_fails_connect(self):
        transport, ws, lane, _ = _make_transport(FakeRealtimeWS(voice_echo="alloy"))
        with pytest.raises(Exception) as exc_info:
            await transport.connect()
        assert "cedar" in str(exc_info.value)
        assert ws.closed is True

    @pytest.mark.asyncio
    async def test_reasoning_rejection_degrades_by_omission(self):
        ws = FakeRealtimeWS(reject_reasoning=True)
        transport, ws, lane, _ = _make_transport(ws)
        await transport.connect()
        updates = [m for m in ws.sent if m["type"] == "session.update"]
        assert len(updates) == 2
        assert "reasoning" in updates[0]["session"]
        assert "reasoning" not in updates[1]["session"]
        await transport.close()

    @pytest.mark.asyncio
    async def test_server_vad_tuning_knobs_and_bounds(self):
        cfg = load_realtime_voice_config({
            "enabled": True,
            "server_vad_threshold": 0.8,
            "server_vad_prefix_padding_ms": 100,
            "server_vad_silence_duration_ms": 350,
        })
        transport, ws, lane, _ = _make_transport(FakeRealtimeWS(), config=cfg)
        await transport.connect()
        td = next(m for m in ws.sent if m["type"] == "session.update")["session"]["audio"]["input"]["turn_detection"]
        assert td["threshold"] == 0.8
        assert td["prefix_padding_ms"] == 100
        assert td["silence_duration_ms"] == 350
        await transport.close()

        # Out-of-range values clamp to sane bounds, never pass through raw.
        wild = load_realtime_voice_config({
            "enabled": True,
            "server_vad_threshold": 7.5,
            "server_vad_prefix_padding_ms": -50,
            "server_vad_silence_duration_ms": 10_000_000,
        })
        assert 0.0 <= wild.server_vad_threshold <= 1.0
        assert wild.server_vad_prefix_padding_ms >= 0
        assert wild.server_vad_silence_duration_ms <= 10000

    @pytest.mark.asyncio
    async def test_semantic_vad_opt_in_carries_no_server_only_fields(self):
        cfg = load_realtime_voice_config({
            "enabled": True, "turn_detection_type": "semantic_vad",
        })
        transport, ws, lane, _ = _make_transport(FakeRealtimeWS(), config=cfg)
        await transport.connect()
        td = next(m for m in ws.sent if m["type"] == "session.update")["session"]["audio"]["input"]["turn_detection"]
        assert td["type"] == "semantic_vad"
        for server_only in ("threshold", "prefix_padding_ms", "silence_duration_ms"):
            assert server_only not in td
        assert td["interrupt_response"] is False
        await transport.close()

    @pytest.mark.asyncio
    async def test_manual_none_stays_null(self):
        cfg = load_realtime_voice_config({
            "enabled": True, "turn_detection_type": "none",
        })
        transport, ws, lane, _ = _make_transport(FakeRealtimeWS(), config=cfg)
        await transport.connect()
        td = next(m for m in ws.sent if m["type"] == "session.update")["session"]["audio"]["input"]["turn_detection"]
        assert td is None
        await transport.close()

    @pytest.mark.asyncio
    async def test_provider_rejects_server_fields_on_semantic_vad(self):
        # Real-shape rejection mock: the GA API refuses server-only tuning
        # fields on a semantic_vad turn_detection block.
        ws = FakeRealtimeWS()
        err = ws._validate_turn_detection({
            "type": "semantic_vad", "threshold": 0.5,
            "interrupt_response": False, "create_response": True,
        })
        assert err is not None
        assert "threshold" in err["error"]["message"]
        assert ws._validate_turn_detection({
            "type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
            "silence_duration_ms": 500, "interrupt_response": False,
            "create_response": True,
        }) is None

    @pytest.mark.asyncio
    async def test_input_transcription_on_by_default_at_ga_nested_location(self):
        # Transcripts are required for review (live: nobody could see what
        # was said): default model rides the GA nested location; the beta
        # top-level key stays gone; empty string still disables.
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        session = next(m for m in ws.sent if m["type"] == "session.update")["session"]
        assert "input_audio_transcription" not in session
        assert session["audio"]["input"]["transcription"] == {
            "model": "gpt-4o-mini-transcribe"
        }
        await transport.close()

        cfg = load_realtime_voice_config(
            {"enabled": True, "input_transcription_model": ""}
        )
        ws2 = FakeRealtimeWS()
        transport2, ws2, _, _ = _make_transport(ws2, config=cfg)
        await transport2.connect()
        session2 = next(m for m in ws2.sent if m["type"] == "session.update")["session"]
        assert "transcription" not in session2["audio"]["input"]
        await transport2.close()

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
    async def test_ga_event_names_map_to_lane(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()

        pcm = b"\x01\x02\x03\x04"
        b64 = base64.b64encode(pcm).decode()
        events = [
            {"type": "input_audio_buffer.speech_started"},
            {"type": "input_audio_buffer.speech_stopped"},
            {"type": "response.created", "response": {"id": "resp_1"}},
            # GA name and legacy name must both deliver audio (tolerance).
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
    async def test_commit_audio_sends_commit(self):
        # Used by the provider canary to force a turn boundary without VAD.
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        await transport.commit_audio()
        await transport.close()
        assert any(m["type"] == "input_audio_buffer.commit" for m in ws.sent)

    @pytest.mark.asyncio
    async def test_cancel_response_nowait_sends_cancel_only_when_active(self):
        transport, ws, lane, _ = _make_transport()
        await transport.connect()
        # No response in flight: the send-time check suppresses the frame
        # (live response_cancel_not_active fix).
        transport.cancel_response_nowait()
        await asyncio.sleep(0.05)
        assert not any(m["type"] == "response.cancel" for m in ws.sent)
        assert transport.stale_cancels_suppressed == 1
        # Active response: exactly one cancel reaches the wire.
        transport._dispatch_event({"type": "response.created", "response": {"id": "r1"}})
        transport.cancel_response_nowait()
        await asyncio.sleep(0.05)
        await transport.close()
        assert len([m for m in ws.sent if m["type"] == "response.cancel"]) == 1

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
