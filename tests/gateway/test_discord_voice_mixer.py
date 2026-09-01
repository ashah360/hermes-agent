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

    def test_plays_fed_audio_through_mixer_after_prebuffer(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        frames_to_prime = vm.STREAM_PREBUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * (frames_to_prime + 1))
        out1 = np.frombuffer(mx.read(), dtype=np.int16)
        out2 = np.frombuffer(mx.read(), dtype=np.int16)
        assert int(np.max(np.abs(out1))) > 0
        assert int(np.max(np.abs(out2))) > 0

    def test_starving_stream_keeps_child_alive_with_silence(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("realtime")
        mx.attach_stream(child)
        frames_to_prime = vm.STREAM_PREBUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * frames_to_prime)
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0
        # Drain the rest, then starve: silence but child stays attached.
        for _ in range(frames_to_prime - 1):
            mx.read()
        assert mx.read() == vm.SILENCE_FRAME
        # Rebuffer target (smaller than startup) resumes playback.
        rebuffer_frames = vm.STREAM_REBUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * rebuffer_frames)
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
        # Sub-prebuffer + explicit finish: residual drains, nothing lost.
        child.finish()
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


class TestRealtimeIdleSilence:
    """Live regression: idle echo/noise came from the ambient bed being
    attached even when the mixer exists solely for the realtime lane."""

    @pytest.mark.asyncio
    async def test_realtime_mixer_installs_without_ambient(self):
        adapter = _make_adapter()  # voice_fx enabled with ambient defaults
        vc = MagicMock()
        vc.is_playing.return_value = False
        with patch.object(
            type(adapter), "_realtime_voice_enabled", return_value=True
        ):
            await adapter._install_voice_mixer(111, vc)
        mixer = adapter._voice_mixers[111]
        # No ambient bed: idle output is pure silence, regardless of voice_fx.
        assert mixer._ambient is None
        for _ in range(5):
            assert mixer.read() == vm.SILENCE_FRAME

    @pytest.mark.asyncio
    async def test_legacy_voice_fx_ambient_unchanged_when_realtime_off(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_playing.return_value = False
        with patch.object(
            type(adapter), "_realtime_voice_enabled", return_value=False
        ):
            await adapter._install_voice_mixer(111, vc)
        mixer = adapter._voice_mixers[111]
        assert mixer._ambient is not None


class TestStreamJitterBuffer:
    """Adaptive prebuffer for bursty Realtime deltas (live choppiness fix).

    Deterministic read/feed schedules only — no sleeps, no wall clock."""

    def _frame(self, value=1000):
        return np.full(vm.SAMPLES_PER_FRAME * vm.CHANNELS, value, dtype=np.int16).tobytes()

    def _pattern_frame(self, i):
        # Distinct nonzero content per frame so order/loss is provable.
        return np.full(
            vm.SAMPLES_PER_FRAME * vm.CHANNELS, 100 + (i % 200), dtype=np.int16
        ).tobytes()

    def test_startup_threshold_no_audio_before_target(self):
        child = vm.StreamingMixerChild("rt")
        below = (vm.STREAM_PREBUFFER_MS // vm.FRAME_LENGTH_MS) - 1
        child.feed(self._frame() * below)
        # Still buffering: silence frames, never real audio, never finished.
        for _ in range(3):
            frame = child.read_frame()
            assert frame is not None
            assert int(np.max(np.abs(frame))) == 0
        child.feed(self._frame())  # reaches target
        assert int(np.max(np.abs(child.read_frame()))) > 0

    def test_finish_below_target_drains_tail_and_ends_cleanly(self):
        child = vm.StreamingMixerChild("rt")
        child.feed(self._frame() * 2)  # 40ms < target
        child.finish()
        assert int(np.max(np.abs(child.read_frame()))) > 0
        assert int(np.max(np.abs(child.read_frame()))) > 0
        assert child.read_frame() is None      # ended exactly once, no zombie
        assert child.finished is True

    def _run_schedule(self, feed_plan, total_reads):
        """Drive the real child on a deterministic feed/read schedule."""
        child = vm.StreamingMixerChild("rt")
        fed = []
        frame_index = 0
        outputs = []
        for tick in range(total_reads):
            for _ in range(feed_plan(tick)):
                pcm = self._pattern_frame(frame_index)
                fed.append(pcm)
                frame_index += 1
                child.feed(pcm)
            frame = child.read_frame()
            assert frame is not None  # never finishes mid-stream
            outputs.append(frame)
        child.finish()
        while True:
            frame = child.read_frame()
            if frame is None:
                break
            outputs.append(frame)
        emissions = [
            "audio" if int(np.max(np.abs(f))) > 0 else "silence" for f in outputs
        ]
        transitions = sum(1 for a, b in zip(emissions, emissions[1:]) if a != b)
        audio_bytes = b"".join(
            f.astype(np.int16).tobytes()
            for f in outputs if int(np.max(np.abs(f))) > 0
        )
        return emissions, transitions, audio_bytes, b"".join(fed)

    @staticmethod
    def _old_model_transitions(feed_plan, total_reads):
        """The pre-fix algorithm: emit whenever >=1 frame is buffered."""
        transitions = 0
        depth = 0
        last = None
        for tick in range(total_reads):
            depth += feed_plan(tick)
            emitted = "audio" if depth >= 1 else "silence"
            if depth >= 1:
                depth -= 1
            if last is not None and emitted != last:
                transitions += 1
            last = emitted
        return transitions

    def test_sufficient_delivery_with_jitter_has_zero_gaps_after_start(self):
        """Phase-jittered deltas whose running deficit stays under the
        prebuffer: once speech begins there must be NO hard-zero frames."""
        def feed_plan(tick):
            if tick < 5:
                return 2                      # startup burst primes quickly
            return 2 if (tick % 2) else 0     # then jittered 1.0x average

        emissions, transitions, audio_bytes, fed_bytes = self._run_schedule(
            feed_plan, 120
        )
        first_audio = emissions.index("audio")
        assert "silence" not in emissions[first_audio:]
        assert transitions == 1               # exactly one silence→audio
        assert audio_bytes == fed_bytes       # order preserved, zero loss

    def test_deficit_bursts_consolidate_transitions_at_least_3x(self):
        """The live choppiness shape: stretches where deltas arrive slower
        than playback (0.5x) with periodic catch-up bursts. The old
        algorithm alternated audio/silence on every late frame; the jitter
        buffer must consolidate gaps into >=3x fewer transitions with zero
        sample loss."""
        def feed_plan(tick):
            burst = 5 if tick % 20 == 19 else 0
            slow = 1 if tick % 2 == 0 else 0
            return slow + burst

        total = 160
        old_transitions = self._old_model_transitions(feed_plan, total)
        emissions, new_transitions, audio_bytes, fed_bytes = self._run_schedule(
            feed_plan, total
        )
        assert old_transitions >= 30          # the fixture really is choppy
        assert new_transitions * 3 <= old_transitions
        assert audio_bytes == fed_bytes       # consolidation loses nothing

    def test_bounded_buffer_drops_newest_and_counts(self):
        child = vm.StreamingMixerChild("rt")
        cap_frames = vm.STREAM_MAX_BUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * (cap_frames + 50))
        assert child.pending_ms() <= vm.STREAM_MAX_BUFFER_MS
        assert child.stats["overrun_dropped_bytes"] >= 50 * vm.FRAME_SIZE

    def test_underrun_and_rebuffer_counters(self):
        child = vm.StreamingMixerChild("rt")
        prime = vm.STREAM_PREBUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * prime)
        for _ in range(prime):
            child.read_frame()          # drain fully
        assert int(np.max(np.abs(child.read_frame()))) == 0  # underrun
        assert child.stats["underruns"] == 1
        assert child.stats["max_depth_ms"] >= vm.STREAM_PREBUFFER_MS

    def test_clear_discards_and_ends_within_one_frame(self):
        mx = vm.VoiceMixer()
        child = vm.StreamingMixerChild("rt")
        mx.attach_stream(child)
        prime = vm.STREAM_PREBUFFER_MS // vm.FRAME_LENGTH_MS
        child.feed(self._frame() * (prime * 3))
        assert int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16)))) > 0
        child.clear()
        # Immediate: very next mixer frame is silence and the child is done.
        assert mx.read() == vm.SILENCE_FRAME
        assert child.finished is True

    def test_legacy_fixed_clip_child_unaffected(self):
        # MixerChild (fixed clips: acks, cascaded TTS) has NO prebuffer:
        # first frame plays immediately, exactly as before.
        clip = vm.MixerChild("ack", self._frame() * 2)
        first = clip.read_frame()
        assert int(np.max(np.abs(first))) > 0
