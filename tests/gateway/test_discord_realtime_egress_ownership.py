"""Single-egress ownership: in REALTIME, the lane is the ONLY spoken source.

Live evidence (16:08–16:12): receive path fully healthy, realtime responses
serialized — and the user still heard the old ElevenLabs voice talking over
the lane, because every legacy egress path was still open (play_tts →
play_in_voice_channel, GatewayRunner._send_voice_reply, play_ack_in_voice).
Also a stale-cancel race: speech start scheduled response.cancel, the
response finished 7ms later, and the delayed cancel hit the provider as
response_cancel_not_active.

Contracts are behavioral (synthesis/play observed or suppressed), never
source text. Fallback preserved: suppression applies ONLY when a lane exists
and is REALTIME — absent, cascaded, or demoting lanes keep legacy TTS.

Harness notes (hang repair): every path that can reach the legacy playback
wait loop pins ``vc.is_playing() -> False`` and a sub-second playback
timeout; TTS interception RECORDS calls and returns a failure JSON instead
of raising (broad except blocks swallow raises, which both hangs nothing
and silently un-REDs the test).
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource

# voice_mixer lives inside the discord plugin dir; import by path the same
# way the adapter (and test_discord_voice_mixer.py) do.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)


class _Lane:
    def __init__(self, state_value):
        self.state = SimpleNamespace(value=state_value)


class _TTSRecorder:
    """Patch target for text_to_speech_tool: records, never raises."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs.get("text", ""))
        return json.dumps({"success": False, "error": "intercepted by test"})


def _make_adapter(lane_state=None, guild_id=111, chat_id=222):
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import PlatformConfig

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_mixers = {}
    adapter._voice_receivers = {}
    adapter._voice_text_channels = {guild_id: chat_id}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_fx_cfg = {"enabled": True, "ack_enabled": True,
                             "ack_phrases": ["One moment."]}
    adapter._realtime_lanes = {}
    adapter._reset_voice_timeout = MagicMock()
    adapter._cancel_voice_timeout = MagicMock()
    if lane_state is not None:
        adapter._realtime_lanes[guild_id] = _Lane(lane_state)
    return adapter


def _make_vc(*, playing=False):
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = playing  # never a truthy MagicMock (hang!)
    return vc


def _fast_playback_patch(adapter):
    return patch.object(
        type(adapter), "_playback_timeout_for_audio", AsyncMock(return_value=0.3)
    )


class TestActiveHelper:
    def test_active_only_when_lane_exists_and_realtime(self):
        assert _make_adapter("realtime").is_realtime_voice_active(111) is True
        assert _make_adapter("cascaded").is_realtime_voice_active(111) is False
        assert _make_adapter("demoting").is_realtime_voice_active(111) is False
        assert _make_adapter(None).is_realtime_voice_active(111) is False
        from plugins.platforms.discord.adapter import DiscordAdapter
        bare = object.__new__(DiscordAdapter)
        assert bare.is_realtime_voice_active(111) is False

    def test_chat_scoped_helper_maps_bound_channel(self):
        adapter = _make_adapter("realtime")
        assert adapter.is_realtime_voice_active_for_chat("222") is True
        assert adapter.is_realtime_voice_active_for_chat(222) is True
        assert adapter.is_realtime_voice_active_for_chat("999") is False
        assert _make_adapter("cascaded").is_realtime_voice_active_for_chat("222") is False


class TestAdapterEgressSuppression:
    @pytest.mark.asyncio
    async def test_play_tts_suppressed_in_realtime(self):
        adapter = _make_adapter("realtime")
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter.play_in_voice_channel = AsyncMock()
        adapter.send_voice = AsyncMock()
        result = await adapter.play_tts("222", "/tmp/old_tts.mp3")
        assert result.success is True                 # suppressed, not failed
        adapter.play_in_voice_channel.assert_not_awaited()
        adapter.send_voice.assert_not_awaited()       # no attachment fallback

    @pytest.mark.asyncio
    async def test_play_ack_suppressed_in_realtime_without_synthesis(self):
        adapter = _make_adapter("realtime")
        adapter._voice_mixers[111] = MagicMock()
        tts = _TTSRecorder()
        with patch("tools.tts_tool.text_to_speech_tool", tts):
            assert await adapter.play_ack_in_voice(111) is False
        assert tts.calls == []                        # no ElevenLabs synthesis

    @pytest.mark.asyncio
    async def test_play_in_voice_channel_suppressed_in_realtime(self):
        adapter = _make_adapter("realtime")
        adapter._voice_clients[111] = _make_vc()
        mixer = MagicMock()
        adapter._voice_mixers[111] = mixer
        with _fast_playback_patch(adapter):
            assert await adapter.play_in_voice_channel(111, "/tmp/x.mp3") is False
        mixer.play_speech.assert_not_called()
        adapter._voice_clients[111].play.assert_not_called()

    @pytest.mark.asyncio
    async def test_demoting_lane_keeps_legacy_fallback(self):
        # DEMOTING (and absent/cascaded) lanes must keep the old TTS path.
        import voice_mixer as vm

        adapter = _make_adapter("demoting")
        adapter._voice_clients[111] = _make_vc()

        class _Mixer:
            def __init__(self):
                self._polls = 0
                self.play_speech = MagicMock()

            @property
            def speech_active(self):
                self._polls += 1
                return self._polls <= 1

        mixer = _Mixer()
        adapter._voice_mixers[111] = mixer
        with patch.object(vm, "decode_to_pcm", return_value=b"\x00" * 3840), \
             _fast_playback_patch(adapter):
            ok = await adapter.play_in_voice_channel(111, "/tmp/x.mp3")
        assert ok is True
        mixer.play_speech.assert_called_once()


def _make_runner(adapter, voice_mode="all"):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._voice_mode = {"discord:222": voice_mode}
    runner._adapter_for_source = lambda source: adapter
    runner._get_guild_id = lambda event: 111
    return runner


def _voice_event(chat_id="222", message_type=MessageType.VOICE):
    return MessageEvent(
        source=SessionSource(platform=Platform.DISCORD, chat_id=chat_id,
                             user_id="42", user_name="Shaan", chat_type="channel"),
        text="what's the weather",
        message_type=message_type,
        raw_message=SimpleNamespace(guild_id=111, guild=None),
    )


class TestGatewayEgressSuppression:
    def test_should_send_voice_reply_false_before_synthesis_when_realtime(self):
        from gateway.run import GatewayRunner

        adapter = _make_adapter("realtime")
        runner = _make_runner(adapter)
        # Both event shapes: normal text turn (voice_mode=all) and the old
        # voice path. Realtime ownership gates BOTH before synthesis.
        assert GatewayRunner._should_send_voice_reply(
            runner, _voice_event(message_type=MessageType.TEXT),
            "some response", [],
        ) is False
        assert GatewayRunner._should_send_voice_reply(
            runner, _voice_event(message_type=MessageType.VOICE),
            "some response", [],
        ) is False

    def test_should_send_voice_reply_true_when_lane_inactive(self):
        from gateway.run import GatewayRunner

        adapter = _make_adapter("cascaded")
        runner = _make_runner(adapter)
        # Text turn with voice_mode=all is the runner-owned TTS path (VOICE
        # events defer to the base adapter's auto-TTS — existing dedup).
        assert GatewayRunner._should_send_voice_reply(
            runner, _voice_event(message_type=MessageType.TEXT),
            "some response", [],
        ) is True

    @pytest.mark.asyncio
    async def test_send_voice_reply_defense_skips_synthesis_and_play(self):
        from gateway.run import GatewayRunner

        adapter = _make_adapter("realtime")
        adapter.is_in_voice_channel = MagicMock(return_value=True)
        adapter.play_in_voice_channel = AsyncMock()
        runner = _make_runner(adapter)
        tts = _TTSRecorder()
        with patch("tools.tts_tool.text_to_speech_tool", tts):
            await GatewayRunner._send_voice_reply(runner, _voice_event(), "hello there")
        assert tts.calls == []                        # gated BEFORE synthesis
        adapter.play_in_voice_channel.assert_not_awaited()


class TestOutboxNeverSpeaks:
    @pytest.mark.asyncio
    async def test_outbox_posts_once_and_triggers_no_tts(self):
        from plugins.platforms.discord.realtime.outbox import DiscordTextOutbox

        sends = []

        class _Adapter:
            async def send(self, chat_id, content, **kw):
                sends.append((chat_id, content))
                return SimpleNamespace(success=True, message_id="m1")

        lane = SimpleNamespace(_loop=asyncio.get_running_loop())
        outbox = DiscordTextOutbox(adapter=_Adapter(), lane=lane)
        record = SimpleNamespace(dispatch_id="d1", text_channel_id=222, goal="g")
        tts = _TTSRecorder()
        with patch("tools.tts_tool.text_to_speech_tool", tts):
            ref1 = await asyncio.to_thread(
                outbox.post_result, record, {"final_response": "result text"}
            )
            ref2 = await asyncio.to_thread(
                outbox.post_result, record, {"final_response": "result text"}
            )
        assert ref1 == ref2 == "m1"
        assert len(sends) == 1
        assert tts.calls == []

    @pytest.mark.asyncio
    async def test_post_result_on_event_loop_thread_never_deadlocks(self):
        """run_coroutine_threadsafe(...).result() on its own target loop
        would deadlock forever — post_result must refuse instead."""
        from plugins.platforms.discord.realtime.outbox import DiscordTextOutbox

        class _Adapter:
            async def send(self, chat_id, content, **kw):
                return SimpleNamespace(success=True, message_id="m1")

        lane = SimpleNamespace(_loop=asyncio.get_running_loop())
        outbox = DiscordTextOutbox(adapter=_Adapter(), lane=lane)
        record = SimpleNamespace(dispatch_id="d2", text_channel_id=222, goal="g")
        # Called ON the loop thread (misuse): must return fast, not hang.
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: None), timeout=1
        )  # sanity that the loop is healthy
        assert outbox.post_result(record, {"final_response": "x"}) is None


class TestFullOwnershipScenario:
    @pytest.mark.asyncio
    async def test_realtime_yields_exactly_one_speech_source(self):
        pytest.importorskip("numpy")
        import voice_mixer as vm
        from plugins.platforms.discord.realtime.config import load_realtime_voice_config
        from plugins.platforms.discord.realtime.lane import LaneState, RealtimeVoiceLane

        adapter = _make_adapter(None)
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        adapter._voice_clients[111] = _make_vc()

        class _T:
            async def connect(self):
                return None

            async def close(self):
                return None

            def cancel_response_nowait(self, *a, **k):
                return None

        lane = RealtimeVoiceLane(
            guild_id=111, text_channel_id=222,
            config=load_realtime_voice_config({"enabled": True}),
            api_key="sk-test", adapter=adapter,
            transport_factory=lambda **kw: _T(),
        )
        assert await lane.start() is True
        adapter._realtime_lanes[111] = lane
        assert lane.state is LaneState.REALTIME

        # Every legacy path tries to speak; all suppressed, none synthesize.
        tts = _TTSRecorder()
        with patch("tools.tts_tool.text_to_speech_tool", tts), \
             _fast_playback_patch(adapter):
            await adapter.play_ack_in_voice(111)
            await adapter.play_in_voice_channel(111, "/tmp/x.mp3")
            adapter.is_in_voice_channel = MagicMock(return_value=True)
            adapter.send_voice = AsyncMock()
            await adapter.play_tts("222", "/tmp/y.mp3")
        assert tts.calls == []
        assert mixer._speech == []

        # The realtime lane speaks: exactly ONE speech child appears.
        lane.on_response_audio_delta(b"\x01\x00" * 480)
        lane.on_response_audio_delta(b"\x01\x00" * 480)
        assert len(mixer._speech) == 1
        await lane.stop()


# ── Stale-cancel race (transport, send-time check) ─────────────────────


class _CancelWS:
    """Minimal GA fake socket for cancel tests (local, no cross-imports)."""

    def __init__(self):
        self.sent = []
        self.inbox = asyncio.Queue()

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("type") == "session.update":
            await self.inbox.put(json.dumps({
                "type": "session.updated",
                "session": {"type": "realtime", "model": "gpt-realtime-2.1",
                            "audio": {"output": {"voice": "cedar"}}},
            }))

    async def recv(self):
        item = await self.inbox.get()
        if item is None:
            raise ConnectionError("closed")
        return item

    async def close(self):
        await self.inbox.put(None)


class _CancelLane:
    def get_projection(self):
        return "P"

    def session_tool_schemas(self):
        return []

    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: None
        raise AttributeError(name)


async def _make_cancel_transport():
    from plugins.platforms.discord.realtime.config import load_realtime_voice_config
    from plugins.platforms.discord.realtime.transport import RealtimeTransport

    ws = _CancelWS()

    async def _connect(url, headers):
        return ws

    transport = RealtimeTransport(
        config=load_realtime_voice_config({"enabled": True}),
        api_key="sk-test", lane=_CancelLane(), ws_connect=_connect,
    )
    await transport.connect()
    return transport, ws


class TestStaleCancelRace:
    @pytest.mark.asyncio
    async def test_response_done_racing_cancel_sends_zero_cancels(self):
        # Live race: cancel scheduled at .617, response.done at .624, cancel
        # hit the wire later as response_cancel_not_active. The SEND-time
        # check must suppress it entirely (not just log-filter the error).
        transport, ws = await _make_cancel_transport()
        transport._dispatch_event({"type": "response.created", "response": {"id": "r1"}})
        transport.cancel_response_nowait()            # scheduled…
        transport._dispatch_event({"type": "response.done"})  # …but done wins
        await asyncio.sleep(0.05)
        await transport.close()
        assert not any(m["type"] == "response.cancel" for m in ws.sent)

    @pytest.mark.asyncio
    async def test_active_response_cancel_sends_exactly_once(self):
        transport, ws = await _make_cancel_transport()
        transport._dispatch_event({"type": "response.created", "response": {"id": "r1"}})
        transport.cancel_response_nowait()
        await asyncio.sleep(0.05)
        assert len([m for m in ws.sent if m["type"] == "response.cancel"]) == 1

        # After done, nothing is active: a new cancel never reaches the wire.
        transport._dispatch_event({"type": "response.done"})
        transport.cancel_response_nowait()
        await asyncio.sleep(0.05)
        await transport.close()
        assert len([m for m in ws.sent if m["type"] == "response.cancel"]) == 1

    @pytest.mark.asyncio
    async def test_cancel_token_does_not_cancel_next_generation(self):
        # A cancel scheduled against response r1 must not fire once r2 is
        # the active generation (generation-bound token, not a boolean).
        transport, ws = await _make_cancel_transport()
        transport._dispatch_event({"type": "response.created", "response": {"id": "r1"}})

        # Freeze the send by scheduling, then flip generations BEFORE the
        # task runs (same loop turn — deterministic, no sleeps in between).
        transport.cancel_response_nowait()
        transport._dispatch_event({"type": "response.done"})
        transport._dispatch_event({"type": "response.created", "response": {"id": "r2"}})
        await asyncio.sleep(0.05)
        await transport.close()
        # r2 is active but the stale r1 token must not cancel it.
        assert not any(m["type"] == "response.cancel" for m in ws.sent)
