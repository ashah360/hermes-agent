"""Provider-canary contracts (fixes from the first live run).

Live findings:
1. With semantic VAD + create_response the server auto-commits/responds;
   the canary's manual commit/create after the buffer was consumed produced
   repeated ``input_audio_buffer_commit_empty`` provider errors. The canary
   must exercise a correct protocol per session (manual commits only on a
   manual-VAD session) and treat ANY provider error event as a failure.
2. ``model_echo``/``voice_echo``/``session_type_echo`` were null because
   bootstrap consumed session.updated before the event tap: the transport
   must store the accepted echo itself.
"""

import asyncio
import json

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config

_BETA_SESSION_KEYS = {
    "voice", "turn_detection", "input_audio_format", "output_audio_format",
    "input_audio_transcription", "modalities", "reasoning_effort",
}


class FakeGAWS:
    """Minimal GA-strict fake provider socket (echoes session.updated)."""

    def __init__(self, *, voice_echo="cedar"):
        self.sent = []
        self.inbox = asyncio.Queue()
        self.closed = False
        self._voice_echo = voice_echo

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "session.update":
            session = msg.get("session") or {}
            if (_BETA_SESSION_KEYS & set(session)) or session.get("type") != "realtime":
                await self.inbox.put(json.dumps({
                    "type": "error",
                    "error": {"type": "invalid_request_error",
                              "code": "beta_api_shape_disabled",
                              "message": "beta shape disabled"},
                }))
                return
            await self.inbox.put(json.dumps({
                "type": "session.updated",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "audio": {"output": {"voice": self._voice_echo,
                                         "format": {"type": "audio/pcm", "rate": 24000}}},
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

    async def push(self, event):
        await self.inbox.put(json.dumps(event))


class RecordingLane:
    def __init__(self):
        self.calls = []
        self.errors = []

    def get_projection(self):
        return "P"

    def session_tool_schemas(self):
        return []

    def on_user_speech_started(self):
        self.calls.append("speech_started")

    def on_user_speech_stopped(self):
        self.calls.append("speech_stopped")

    def on_response_created(self, response_id):
        pass

    def on_response_audio_delta(self, pcm):
        pass

    def on_output_transcript_delta(self, text, response_id):
        pass

    def on_input_transcript(self, text):
        pass

    def on_response_done(self):
        pass

    def on_function_call(self, name, args, call_id):
        pass

    def on_provider_error(self, error):
        self.errors.append(error)

    def on_transport_closed(self, exc):
        pass


def _make_transport(ws, config=None, lane=None):
    from plugins.platforms.discord.realtime.transport import RealtimeTransport

    lane = lane or RecordingLane()

    async def _connect(url, headers):
        return ws

    return RealtimeTransport(
        config=config or load_realtime_voice_config({"enabled": True}),
        api_key="sk-test", lane=lane, ws_connect=_connect,
    ), lane


class TestSessionEchoStored:
    @pytest.mark.asyncio
    async def test_transport_stores_accepted_session_echo(self):
        # Bootstrap consumes session.updated itself, so the accepted echo
        # must be exposed on the transport — the canary reads it from there.
        ws = FakeGAWS()
        transport, lane = _make_transport(ws)
        await transport.connect()
        echo = transport.session_echo
        assert echo.get("model") == "gpt-realtime-2.1"
        assert echo.get("type") == "realtime"
        assert transport._echo_voice(echo) == "cedar"
        await transport.close()


class TestProviderErrorPropagation:
    @pytest.mark.asyncio
    async def test_post_connect_error_events_reach_the_lane(self):
        # A provider error event after connect (e.g.
        # input_audio_buffer_commit_empty) must reach the lane/probe — the
        # canary treats any such event as a FAILURE, not a logged warning.
        ws = FakeGAWS()
        transport, lane = _make_transport(ws)
        await transport.connect()
        await ws.push({
            "type": "error",
            "error": {"type": "invalid_request_error",
                      "code": "input_audio_buffer_commit_empty",
                      "message": "buffer too small"},
        })
        await asyncio.sleep(0.05)
        await transport.close()
        assert lane.errors and lane.errors[0]["code"] == "input_audio_buffer_commit_empty"


class TestManualVadSessionShape:
    @pytest.mark.asyncio
    async def test_turn_detection_none_emits_null(self):
        # The canary's manual-commit audio turn runs on a dedicated
        # manual-VAD session: turn_detection must be JSON null there.
        cfg = load_realtime_voice_config(
            {"enabled": True, "turn_detection_type": "none"}
        )
        ws = FakeGAWS()
        transport, lane = _make_transport(ws, config=cfg)
        await transport.connect()
        await transport.close()
        session = next(m for m in ws.sent if m["type"] == "session.update")["session"]
        assert session["audio"]["input"]["turn_detection"] is None


class TestCanaryProtocolPerSession:
    @pytest.mark.asyncio
    async def test_manual_audio_turn_commits_and_semantic_text_turn_does_not(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "discord_realtime_canary",
            Path(__file__).resolve().parents[2] / "scripts" / "discord_realtime_canary.py",
        )
        canary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(canary)

        # Manual-VAD audio turn: append → commit → response.create.
        ws = FakeGAWS()
        cfg = load_realtime_voice_config(
            {"enabled": True, "turn_detection_type": "none"}
        )
        probe = canary._ProbeLane()
        transport, _ = _make_transport(ws, config=cfg, lane=probe)
        await transport.connect()
        result = await canary._manual_audio_turn(transport, probe, timeout=0.3, pace=False)
        await transport.close()
        types = [m["type"] for m in ws.sent]
        assert "input_audio_buffer.append" in types
        assert "input_audio_buffer.commit" in types
        assert "response.create" in types
        assert result["ok"] is False  # fake sends no audio; latency unmeasured

        # Semantic (production) session text turn: NO manual commit ever.
        ws2 = FakeGAWS()
        probe2 = canary._ProbeLane()
        transport2, _ = _make_transport(ws2, lane=probe2)
        await transport2.connect()
        await canary._text_turn(transport2, probe2, timeout=0.3)
        await transport2.close()
        types2 = [m["type"] for m in ws2.sent]
        assert "input_audio_buffer.commit" not in types2
        assert "input_audio_buffer.append" not in types2

    def test_probe_collects_provider_errors_and_evaluation_fails_on_them(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "discord_realtime_canary",
            Path(__file__).resolve().parents[2] / "scripts" / "discord_realtime_canary.py",
        )
        canary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(canary)

        probe = canary._ProbeLane()
        probe.on_provider_error({"code": "input_audio_buffer_commit_empty"})
        assert probe.errors

        failures = canary._echo_failures(
            {"model": "gpt-realtime-2.1", "type": "realtime",
             "audio": {"output": {"voice": "cedar"}}},
            expected_model="gpt-realtime-2.1", expected_voice="cedar",
        )
        assert failures == []

        # Missing echo is a FAILURE, not a pass.
        assert canary._echo_failures(
            {}, expected_model="gpt-realtime-2.1", expected_voice="cedar"
        )
        # Wrong model/voice/type each fail.
        assert canary._echo_failures(
            {"model": "gpt-realtime-2", "type": "realtime",
             "audio": {"output": {"voice": "cedar"}}},
            expected_model="gpt-realtime-2.1", expected_voice="cedar",
        )
