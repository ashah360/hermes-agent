"""Worker-event injection must wait for PLAYBACK idle, not just response.done.

Live report: "talks over himself when the worker comes back". Root cause:
provider response.done fires when GENERATION finishes, but the jitter mixer
can still be draining seconds of audio (telemetry saw 8720ms of queue).
Injecting then creates a second response whose audio overlaps the still-
draining child. The safe boundary is response.done AND mixer playback idle
AND no user speech. Worker results never clear/truncate playback — only
genuine user barge-in may.
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

np = pytest.importorskip("numpy")

_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm  # noqa: E402

from plugins.platforms.discord.realtime.config import load_realtime_voice_config  # noqa: E402
from plugins.platforms.discord.realtime.events import WorkerEvent  # noqa: E402


class _Transport:
    def __init__(self):
        self.items = []
        self.creates = []
        self.cancels = 0

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        return None

    def cancel_response_nowait(self, *a, **k):
        self.cancels += 1

    async def inject_item(self, role, text):
        self.items.append((role, text))

    async def create_response(self, *, instructions=None):
        self.creates.append(instructions)

    async def close(self):
        return None


async def _make_lane_with_mixer():
    from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

    mixer = vm.VoiceMixer()
    adapter = SimpleNamespace(
        _voice_mixers={111: mixer},
        interrupt_voice_playback=MagicMock(return_value=True),
    )
    transport = _Transport()
    lane = RealtimeVoiceLane(
        guild_id=111, text_channel_id=222,
        config=load_realtime_voice_config({"enabled": True}),
        api_key="sk-test", adapter=adapter,
        transport_factory=lambda **kw: transport,
    )
    assert await lane.start() is True
    return lane, transport, mixer


def _event(lane, etype="completed", hint="done"):
    record = lane.registry.register(
        goal=f"g-{hint}", guild_id=111, text_channel_id=222, user_id=42
    )
    return WorkerEvent(
        dispatch_id=record.dispatch_id, epoch=record.epoch, type=etype,
        spoken_hint=hint, detail_ref=None, sources=(), unsourced=True,
        ts=0.0, spoken_synthesis=None,
    )


def _queue_audio(lane, frames=30):
    # Provider deltas: ~50 frames prime + queue well past the prebuffer.
    for _ in range(frames):
        lane.on_response_audio_delta(b"\x01\x00" * 480)  # 20ms @24k mono


def _drain_until_idle(mixer, max_reads=600):
    for _ in range(max_reads):
        mixer.read()
        if not mixer._speech:
            return
    raise AssertionError("mixer never drained")


class TestPlaybackBoundary:
    @pytest.mark.asyncio
    async def test_response_done_with_queued_audio_defers_injection(self):
        """(a)+(d): completed event during post-done playback → NO
        response.create and NO mixer clear until the queue drains; exactly
        one response afterward."""
        lane, transport, mixer = await _make_lane_with_mixer()
        lane.on_response_created("r1")
        _queue_audio(lane, frames=40)
        child = lane._stream_child
        assert child is not None and child.pending_ms() > 0
        lane.on_response_done()          # generation done, playback NOT done

        lane.deliver_worker_event(_event(lane))
        await asyncio.sleep(0.05)
        # Deferred: no new response while audio is still draining, and the
        # draining child was never cleared/truncated.
        assert transport.creates == []
        assert child.pending_ms() > 0
        assert lane.telemetry.counter("injections_deferred_playback") == 1

        _drain_until_idle(mixer)
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1   # exactly one, strictly after
        snap = lane.telemetry.snapshot()
        assert snap["histograms"]["injection_playback_defer_ms"]["count"] == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_user_barge_in_during_defer_window_still_clears_immediately(self):
        """(b): only genuine user speech clears playback — and it still wins
        during the defer window."""
        lane, transport, mixer = await _make_lane_with_mixer()
        lane.on_response_created("r1")
        _queue_audio(lane, frames=40)
        lane.on_response_done()
        lane.deliver_worker_event(_event(lane))
        await asyncio.sleep(0.02)
        assert transport.creates == []

        lane.on_user_speech_started()
        # Immediate silence: the very next mixer frame is zero.
        assert mixer.read() == vm.SILENCE_FRAME
        # The queued worker event still delivers, exactly once, after the
        # user's turn completes.
        lane.on_user_speech_stopped()
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1
        await lane.stop()

    @pytest.mark.asyncio
    async def test_two_worker_events_serialize_without_overlap(self):
        """(c): two events queue in order; each waits for the previous
        response's PLAYBACK to finish, not just its response.done."""
        lane, transport, mixer = await _make_lane_with_mixer()
        lane.deliver_worker_event(_event(lane, hint="first"))
        lane.deliver_worker_event(_event(lane, hint="second"))
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1   # first injected (idle mixer)

        # First event's response streams audio, generation ends, audio drains.
        lane.on_response_created("r1")
        _queue_audio(lane, frames=40)
        lane.on_response_done()
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1   # second still deferred (audio!)

        _drain_until_idle(mixer)
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 2   # second strictly after playback
        first_idx = next(i for i, (r, t) in enumerate(transport.items) if "first" in t)
        second_idx = next(i for i, (r, t) in enumerate(transport.items) if "second" in t)
        assert first_idx < second_idx
        await lane.stop()

    @pytest.mark.asyncio
    async def test_worker_paths_never_clear_playback(self):
        """Worker events/continuations must never truncate speech: the
        draining child's queue is untouched by delivery and injection."""
        lane, transport, mixer = await _make_lane_with_mixer()
        lane.on_response_created("r1")
        _queue_audio(lane, frames=40)
        child = lane._stream_child
        depth_before = child.pending_ms()
        lane.on_response_done()
        lane.deliver_worker_event(_event(lane))
        lane.on_function_call("recall_result", {}, "call_1")
        await asyncio.sleep(0.05)
        assert child.pending_ms() == depth_before   # nothing cleared it
        assert transport.cancels == 0               # and nothing cancelled
        await lane.stop()

    @pytest.mark.asyncio
    async def test_continuation_also_waits_for_playback(self):
        """Function-call continuations obey the same playback boundary."""
        lane, transport, mixer = await _make_lane_with_mixer()
        lane.on_response_created("r1")
        _queue_audio(lane, frames=40)
        lane.on_function_call("voice_context", {}, "call_1")  # continuation pends
        lane.on_response_done()
        await asyncio.sleep(0.05)
        assert transport.creates == []       # audio still draining
        _drain_until_idle(mixer)
        await asyncio.sleep(0.05)
        assert len(transport.creates) == 1   # continuation after playback
        await lane.stop()
