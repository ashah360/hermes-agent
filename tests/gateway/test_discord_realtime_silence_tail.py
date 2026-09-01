"""Discord-input inactivity silence tail (transport completion semantics).

Live root cause (21:23): Discord STOPS sending the speaker's RTP a few Opus
silence frames after they stop talking — the provider's audio timeline
freezes mid-turn, so server VAD never accumulates its 500ms of silence and
`speech_stopped` never fires (counters froze at :52; no turn close for
minutes). The lane must convert Discord packet cessation into explicit
silence on the provider timeline: after ~idle_ms of wall-clock frame
inactivity, append zeros exceeding server_vad_silence_duration_ms, exactly
once per inactivity episode. Scoped to server_vad; never blocks the receive
path; never synthesizes commits or response.create.
"""

import asyncio

import pytest

np = pytest.importorskip("numpy")

from plugins.platforms.discord.realtime.config import load_realtime_voice_config

_BYTES_PER_MS_24K_MONO = 48  # 24 samples/ms * 2 bytes


class _Transport:
    def __init__(self):
        self.appends = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        self.appends.append(pcm)

    def cancel_response_nowait(self, *a, **k):
        return None

    async def close(self):
        return None


def _frame(value=1200):
    return np.full(960 * 2, value, dtype=np.int16).tobytes()  # 20ms 48k stereo


def _zero_tails(transport):
    return [a for a in transport.appends if a and set(a) == {0}]


def _speech_appends(transport):
    return [a for a in transport.appends if a and set(a) != {0}]


async def _make_lane(**cfg_overrides):
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

    cfg = {
        "enabled": True,
        "discord_input_idle_ms": 150,
        "server_vad_silence_duration_ms": 500,
    }
    cfg.update(cfg_overrides)
    transport = _Transport()
    lane = RealtimeVoiceLane(
        guild_id=111, text_channel_id=222,
        config=load_realtime_voice_config(cfg),
        api_key="sk-test",
        transport_factory=lambda **kw: transport,
    )
    assert await lane.start() is True
    return lane, transport


async def _feed(lane, n=1, gap=0.0):
    for _ in range(n):
        lane.on_input_frame(42, 100, _frame())
        if gap:
            await asyncio.sleep(gap)
        else:
            await asyncio.sleep(0)


class TestSilenceTail:
    @pytest.mark.asyncio
    async def test_discord_packet_cessation_regression_one_tail_exceeds_vad_window(self):
        """THE live regression: finite real-speech frames, NO explicit
        silence in the input fixture — the lane's tail alone must give the
        provider enough zero-audio timeline to close the turn."""
        lane, transport = await _make_lane()
        await _feed(lane, n=10)              # Discord speech packets…
        await asyncio.sleep(0.05)
        assert _zero_tails(transport) == []  # not yet idle
        await asyncio.sleep(0.5)             # …then packet cessation

        tails = _zero_tails(transport)
        assert len(tails) == 1               # exactly one tail per episode
        tail_ms = len(tails[0]) / _BYTES_PER_MS_24K_MONO
        # Must EXCEED the configured server VAD silence window.
        assert tail_ms > 500
        # Real speech frames all delivered before it, in order, undropped.
        assert len(_speech_appends(transport)) == 10
        assert set(transport.appends[-1]) == {0}
        assert lane.telemetry.counter("silence_tails_appended") == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_ongoing_frames_postpone_tail(self):
        lane, transport = await _make_lane()
        # Keep talking: inter-frame gaps well under idle_ms.
        for _ in range(8):
            await _feed(lane, n=1, gap=0.05)
        assert _zero_tails(transport) == []  # never idle long enough
        await asyncio.sleep(0.5)
        assert len(_zero_tails(transport)) == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_brief_pause_below_threshold_emits_nothing_extra(self):
        lane, transport = await _make_lane()
        await _feed(lane, n=3)
        await asyncio.sleep(0.08)            # micro-pause < idle_ms
        await _feed(lane, n=3)               # resumes: stale task invalidated
        assert _zero_tails(transport) == []
        await asyncio.sleep(0.5)
        assert len(_zero_tails(transport)) == 1   # only the final episode
        await lane.stop()

    @pytest.mark.asyncio
    async def test_repeated_idle_never_duplicates_tail(self):
        lane, transport = await _make_lane()
        await _feed(lane, n=3)
        await asyncio.sleep(0.6)
        assert len(_zero_tails(transport)) == 1
        await asyncio.sleep(0.6)             # still idle: no frames, no tails
        assert len(_zero_tails(transport)) == 1
        # New speech → new episode → exactly one more tail.
        await _feed(lane, n=3)
        await asyncio.sleep(0.6)
        assert len(_zero_tails(transport)) == 2
        await lane.stop()

    @pytest.mark.asyncio
    async def test_frame_after_tail_is_never_dropped(self):
        lane, transport = await _make_lane()
        await _feed(lane, n=2)
        await asyncio.sleep(0.5)             # tail emitted
        assert len(_zero_tails(transport)) == 1
        await _feed(lane, n=2)               # user talks again immediately
        await asyncio.sleep(0.05)
        assert len(_speech_appends(transport)) == 4
        await lane.stop()

    @pytest.mark.asyncio
    async def test_stop_and_demotion_cancel_pending_tail(self):
        lane, transport = await _make_lane()
        await _feed(lane, n=2)
        await lane.stop()                    # before idle elapses
        await asyncio.sleep(0.4)
        assert _zero_tails(transport) == []

        lane2, transport2 = await _make_lane()
        await _feed(lane2, n=2)
        lane2.on_transport_closed(ConnectionError("drop"))  # demotion
        await asyncio.sleep(0.4)
        assert _zero_tails(transport2) == []
        await lane2.stop()

    @pytest.mark.asyncio
    async def test_scoped_to_server_vad_only(self):
        for vad in ("semantic_vad", "none"):
            lane, transport = await _make_lane(turn_detection_type=vad)
            await _feed(lane, n=3)
            await asyncio.sleep(0.5)
            assert _zero_tails(transport) == [], f"tail leaked for {vad}"
            await lane.stop()

    def test_idle_ms_config_bounds(self):
        cfg = load_realtime_voice_config({})
        assert 100 <= cfg.discord_input_idle_ms <= 2000
        assert load_realtime_voice_config(
            {"discord_input_idle_ms": 5}
        ).discord_input_idle_ms >= 100
        assert load_realtime_voice_config(
            {"discord_input_idle_ms": 60000}
        ).discord_input_idle_ms <= 2000
