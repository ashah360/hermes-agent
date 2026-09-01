"""Audio geometry + inbound flow for the realtime lane (ADR D4/D8).

Discord voice is 48 kHz stereo s16le; gpt-realtime speaks 24 kHz mono pcm16.
The resampler must be exact-ratio (2:1), deterministic, and cheap. The lane
pump forwards tapped frames to the provider; barge-in clears playback before
cancelling; first-audio latency is measured from VAD speech_stopped.
"""

import asyncio
import math

import pytest

np = pytest.importorskip("numpy")

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


def _sine_48k_stereo(freq=440.0, seconds=0.1, amp=12000):
    n = int(48000 * seconds)
    t = np.arange(n) / 48000.0
    mono = (amp * np.sin(2 * math.pi * freq * t)).astype(np.int16)
    stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1)
    return stereo.tobytes()


class TestResampling:
    def test_discord_to_realtime_geometry(self):
        from plugins.platforms.discord.realtime.audio import discord_pcm_to_realtime

        src = _sine_48k_stereo()
        out = discord_pcm_to_realtime(src)
        # 48k stereo -> 24k mono: byte count divides by 4.
        assert len(out) == len(src) // 4
        samples = np.frombuffer(out, dtype=np.int16)
        assert int(np.max(np.abs(samples))) > 8000  # energy preserved

    def test_realtime_to_discord_geometry(self):
        from plugins.platforms.discord.realtime.audio import realtime_pcm_to_discord

        n = 2400  # 0.1s at 24kHz mono
        t = np.arange(n) / 24000.0
        mono = (12000 * np.sin(2 * math.pi * 440.0 * t)).astype(np.int16).tobytes()
        out = realtime_pcm_to_discord(mono)
        assert len(out) == len(mono) * 4
        samples = np.frombuffer(out, dtype=np.int16)
        assert int(np.max(np.abs(samples))) > 8000

    def test_silence_maps_to_silence_and_empty_to_empty(self):
        from plugins.platforms.discord.realtime.audio import (
            discord_pcm_to_realtime,
            realtime_pcm_to_discord,
        )

        assert discord_pcm_to_realtime(b"") == b""
        assert realtime_pcm_to_discord(b"") == b""
        silence = b"\x00" * 3840
        assert discord_pcm_to_realtime(silence) == b"\x00" * 960

    def test_frequency_preserved_through_downsample(self):
        from plugins.platforms.discord.realtime.audio import discord_pcm_to_realtime

        src = _sine_48k_stereo(freq=440.0, seconds=0.5)
        out = np.frombuffer(discord_pcm_to_realtime(src), dtype=np.int16).astype(np.float64)
        # Zero-crossing rate ≈ 2*f/sr → crossings over 0.5s ≈ 440.
        crossings = int(np.sum(np.abs(np.diff(np.signbit(out)))))
        assert 400 <= crossings <= 480


class _FakeTransport:
    def __init__(self):
        self.appended = []
        self.cancels = 0

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        self.appended.append(pcm)

    def cancel_response_nowait(self):
        self.cancels += 1

    async def close(self):
        return None


def _make_lane(transport):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

    return RealtimeVoiceLane(
        guild_id=111,
        text_channel_id=222,
        config=load_realtime_voice_config({"enabled": True}),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )


class TestLaneAudioFlow:
    @pytest.mark.asyncio
    async def test_frames_pump_to_provider_resampled(self):
        transport = _FakeTransport()
        lane = _make_lane(transport)
        assert await lane.start() is True
        frame = _sine_48k_stereo(seconds=0.02)
        lane.on_input_frame(42, 100, frame)
        await asyncio.sleep(0.05)
        await lane.stop()
        assert len(transport.appended) == 1
        assert len(transport.appended[0]) == len(frame) // 4

    @pytest.mark.asyncio
    async def test_barge_in_clears_playback_before_cancel_and_records_latency(self):
        transport = _FakeTransport()
        lane = _make_lane(transport)
        assert await lane.start() is True

        order = []

        class _Child:
            finished = False

            def clear(self):
                order.append("clear")

            def end(self):
                order.append("end")

        lane._stream_child = _Child()

        class _OrderedTransport(_FakeTransport):
            def cancel_response_nowait(self):
                order.append("cancel")

        lane.transport = _OrderedTransport()
        # Cancel discipline: provider cancel only fires for an ACTIVE
        # response (idle speech start must not spam response.cancel).
        lane.on_response_created("r1")
        lane.on_user_speech_started()
        await lane.stop()

        assert order[0] == "clear"          # audio silenced first
        assert "cancel" in order            # provider cancelled after
        assert order.index("clear") < order.index("cancel")
        snap = lane.telemetry.snapshot()
        assert snap["histograms"]["barge_in_stop_ms"]["count"] == 1
        assert lane.turn_seq == 1

    @pytest.mark.asyncio
    async def test_first_audio_latency_measured_from_speech_stopped(self):
        transport = _FakeTransport()
        lane = _make_lane(transport)
        assert await lane.start() is True
        lane.on_user_speech_stopped()
        # No mixer available (adapter None) → delta is dropped, but the
        # latency measurement must still be recorded exactly once.
        lane.on_response_audio_delta(b"\x01\x00" * 480)
        lane.on_response_audio_delta(b"\x01\x00" * 480)
        await lane.stop()
        snap = lane.telemetry.snapshot()
        assert snap["histograms"]["speech_end_to_first_audio_ms"]["count"] == 1
