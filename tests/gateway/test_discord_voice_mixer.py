"""Tests for the Discord continuous voice mixer (ambient + ducked speech)
and the verbal-ack-before-tool-calls hook.

The mixer (plugins/platforms/discord/voice_mixer.py) is pure-PCM and has no
discord.py dependency, so its core is tested directly.  The adapter
integration (install on join, play routing, ack) is tested with the standard
``object.__new__(DiscordAdapter)`` helper used elsewhere in the voice suite.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# numpy ships only in the optional "voice" extra (not [all,dev]); the mixer
# math needs it, so skip this whole module when it isn't installed.
np = pytest.importorskip("numpy")

# voice_mixer lives inside the discord plugin package dir; import by path the
# same way the adapter does.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm  # noqa: E402


# =====================================================================
# Pure mixer unit tests
# =====================================================================

class TestVoiceMixerCore:
    def test_frame_geometry_matches_discord(self):
        # 20ms @ 48kHz stereo s16 == 3840 bytes (discord.opus.Encoder.FRAME_SIZE)
        assert vm.FRAME_SIZE == 3840
        assert vm.SAMPLES_PER_FRAME == 960
        assert len(vm.SILENCE_FRAME) == vm.FRAME_SIZE

    def test_empty_mixer_returns_silence_frames(self):
        mx = vm.VoiceMixer()
        for _ in range(5):
            frame = mx.read()
            assert len(frame) == vm.FRAME_SIZE
            assert frame == vm.SILENCE_FRAME

    def test_is_opus_false(self):
        # discord.py sends raw PCM when is_opus() is False.
        assert vm.VoiceMixer().is_opus() is False

    def test_ambient_loops_and_is_quiet(self):
        mx = vm.VoiceMixer(ambient_gain=0.2)
        amb = vm.synth_ambient_pcm(seconds=0.5)
        assert len(amb) % vm.FRAME_SIZE == 0  # frame-aligned for seamless loop
        mx.set_ambient(amb)
        peaks = [int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16))))
                 for _ in range(100)]  # 2s >> 0.5s loop
        # Produces audio after the fade-in and stays under the configured gain.
        assert any(p > 0 for p in peaks[10:])
        assert max(peaks) < int(32767 * 0.5)


# =====================================================================
# Adapter integration
# =====================================================================

def _make_adapter(fx_cfg=None):
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform, PlatformConfig
    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = fx_cfg if fx_cfg is not None else {
        "enabled": True, "ambient_enabled": True, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    return adapter


class TestVoiceMixerActive:


    def test_false_when_attr_missing(self):
        # Defensive getattr path (object.__new__ helper that forgot the attr).
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform
        bare = object.__new__(DiscordAdapter)
        bare.platform = Platform.DISCORD
        assert bare.voice_mixer_active(111) is False


class TestPlayInVoiceChannelMixerPath:
    @pytest.mark.asyncio
    async def test_routes_through_mixer_when_present(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc

        # speech_active returns True once (so play_speech is observed) then
        # False so the wait loop exits promptly.
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
        adapter._reset_voice_timeout = MagicMock()

        fake_pcm = b"\x00" * vm.FRAME_SIZE
        with patch.object(vm, "decode_to_pcm", return_value=fake_pcm):
            ok = await adapter.play_in_voice_channel(111, "/tmp/x.mp3")
        assert ok is True
        mixer.play_speech.assert_called_once()
        adapter._reset_voice_timeout.assert_called_once_with(111)
        # Legacy path must NOT have been used.
        vc.play.assert_not_called()


class TestLeadSilence:
    """Warm-up lead silence prepended to speech so the first word isn't clipped
    (issue #66827)."""

    def test_bytes_empty_when_unset(self):
        adapter = _make_adapter()  # default cfg has no lead_silence_ms
        assert adapter._lead_silence_bytes() == b""


    def test_bytes_length_matches_ms(self):
        adapter = _make_adapter({"lead_silence_ms": 200})
        lead = adapter._lead_silence_bytes()
        assert lead == b"\x00" * (vm.BYTES_PER_MS * 200)
        assert len(lead) == 200 * 192  # 48kHz stereo s16 -> 192 bytes/ms


class TestPlayAckInVoice:
    @pytest.mark.asyncio
    async def test_noop_when_ack_disabled(self):
        adapter = _make_adapter({"ack_enabled": False})
        adapter._voice_mixers[111] = MagicMock()
        assert await adapter.play_ack_in_voice(111) is False



# =====================================================================
# StreamingMixerChild (realtime lane / exact-TTS playback)
# =====================================================================

class TestStreamingMixerChild:
    def _frame(self, value=1000):
        return np.full(vm.SAMPLES_PER_FRAME * vm.CHANNELS, value, dtype=np.int16).tobytes()

    def test_plays_fed_audio_through_mixer(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        child.feed(self._frame() * 2)
        out1 = np.frombuffer(mx.read(), dtype=np.int16)
        out2 = np.frombuffer(mx.read(), dtype=np.int16)
        assert int(np.max(np.abs(out1))) > 0
        assert int(np.max(np.abs(out2))) > 0

    def test_starving_stream_keeps_child_alive_with_silence(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        child.feed(self._frame())
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0
        # Starving (no data, not ended): silence but child stays attached.
        assert mx.read() == vm.SILENCE_FRAME
        child.feed(self._frame())
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0

    def test_clear_drops_queued_audio_within_one_frame(self):
        # Barge-in contract: after clear(), the very next mixer frame is
        # silence — 20ms worst-case in-process latency.
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        child.feed(self._frame() * 50)   # 1s of queued audio
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0
        child.clear()
        assert mx.read() == vm.SILENCE_FRAME

    def test_end_finishes_child_after_drain_and_releases_duck(self):
        mx = vm.VoiceMixer(ambient_gain=0.5, duck_gain=0.0, duck_release_ms=20)
        amb = vm.synth_ambient_pcm(seconds=0.5)
        mx.set_ambient(amb)
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        assert mx.speech_active is True  # duck engaged while stream attached
        child.feed(self._frame())
        child.end()
        mx.read()   # drains the fed frame
        mx.read()   # child finished -> removed, duck release begins
        assert mx.speech_active is False

    def test_partial_frame_feed_is_buffered_not_dropped(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        half = self._frame()[: vm.FRAME_SIZE // 2]
        child.feed(half)
        # Not a full frame yet -> silence, but bytes retained.
        assert mx.read() == vm.SILENCE_FRAME
        child.feed(half)
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0

    def test_pending_ms_reports_queue_depth(self):
        child = vm.StreamingMixerChild("realtime")
        child.feed(self._frame() * 5)
        assert child.pending_ms() == 5 * vm.FRAME_LENGTH_MS


class TestInterruptVoicePlayback:
    def test_stops_mixer_speech_immediately(self):
        # Live-patch regression: playback interruption keeps the mixer alive.
        adapter = _make_adapter()
        mixer = MagicMock()
        adapter._voice_mixers = {111: mixer}
        assert adapter.interrupt_voice_playback(111) is True
        mixer.stop_speech.assert_called_once_with()

    def test_falls_back_to_vc_stop_without_mixer(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_playing.return_value = True
        adapter._voice_clients = {111: vc}
        assert adapter.interrupt_voice_playback(111) is True
        vc.stop.assert_called_once_with()

    def test_false_when_nothing_playing(self):
        adapter = _make_adapter()
        assert adapter.interrupt_voice_playback(111) is False
