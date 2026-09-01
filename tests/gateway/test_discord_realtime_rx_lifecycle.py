"""Receive-path observability + leave→rejoin lifecycle (live second-join bug).

Live evidence: after a second /voice join on the same build, a SPEAKING
event was logged (websocket hook alive) but no provider speech event ever
fired — something between the socket listener and ``append_audio`` dropped.

Contracts:
- per-receiver counters at every seam (raw → RTP → NaCl → DAVE → Opus →
  sink), first-success INFO logs, first-failure WARNINGs regardless of the
  debug packet counter, bounded periodic aggregates;
- ``stop()`` restores the SPEAKING hook it installed (a dead receiver must
  not stay chained on a reused voice connection) and removes only its own
  listener — a fresh receiver on the same connection stays fully active;
- transport crypto params are read LIVE from the connection (a voice-server
  reconnect rotates the secret key; the old cached-at-start copy made every
  packet fail NaCl silently);
- full cross-thread chain: background-thread delivery on the SECOND
  receiver reaches a fake transport append in the event loop.
"""

import asyncio
import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.platforms.discord.realtime.config import load_realtime_voice_config


def _make_conn(*, secret_key=None, ssrc=9999):
    conn = MagicMock()
    conn.secret_key = secret_key if secret_key is not None else [1] * 32
    conn.dave_session = None
    conn.ssrc = ssrc
    conn.hook = None
    listeners = []
    conn.add_socket_listener = lambda fn: listeners.append(fn)
    conn.remove_socket_listener = lambda fn: listeners.remove(fn)
    conn.listeners = listeners
    return conn


def _make_receiver(conn=None, **kwargs):
    from plugins.platforms.discord.adapter import VoiceReceiver

    vc = MagicMock()
    vc._connection = conn if conn is not None else _make_conn()
    return VoiceReceiver(vc, **kwargs), vc._connection


class TestReceiverCounters:
    def test_counters_exist_and_sink_delivery_counts(self):
        receiver, conn = _make_receiver()
        assert receiver.counters["sink_calls"] == 0
        frames = []
        receiver.set_frame_sink(lambda user_id, ssrc, pcm: frames.append(pcm))
        receiver.map_ssrc(100, 42)
        receiver._deliver_pcm(100, b"\x01\x02" * 1920)  # one 3840-byte frame
        assert receiver.counters["sink_calls"] == 1
        assert receiver.counters["sink_bytes"] == 3840
        # Buffered path counts separately.
        receiver.set_frame_sink(None)
        receiver._deliver_pcm(100, b"\x01\x02" * 1920)
        assert receiver.counters["buffered_frames"] == 1

    def test_first_sink_delivery_logs_info_once(self, caplog):
        receiver, conn = _make_receiver()
        receiver.set_frame_sink(lambda *a: None)
        receiver.map_ssrc(100, 42)
        with caplog.at_level(logging.INFO):
            receiver._deliver_pcm(100, b"\x00" * 3840)
            receiver._deliver_pcm(100, b"\x00" * 3840)
        firsts = [r for r in caplog.records if "voice_rx first_sink_delivery" in r.message]
        assert len(firsts) == 1

    def test_first_failure_warns_regardless_of_debug_count(self, caplog):
        receiver, conn = _make_receiver()
        receiver._packet_debug_count = 999  # past the old debug-log window
        with caplog.at_level(logging.WARNING):
            receiver._count_failure("nacl_fail", "boom")
            receiver._count_failure("nacl_fail", "boom2")
            receiver._count_failure("opus_fail", "bad frame")
        warnings = [r for r in caplog.records if "voice_rx first_failure" in r.message]
        assert len(warnings) == 2  # one per failure class
        assert receiver.counters["nacl_fail"] == 2
        assert receiver.counters["opus_fail"] == 1


class TestLiveCryptoParams:
    def test_secret_key_read_live_from_connection(self):
        # A voice-server reconnect rotates conn.secret_key; the receiver must
        # decrypt with the CURRENT key, not the copy cached at start().
        conn = _make_conn(secret_key=[1] * 32)
        receiver, conn = _make_receiver(conn)
        receiver.start()
        assert receiver._current_secret_key() == bytes([1] * 32)
        conn.secret_key = [2] * 32  # rotation
        assert receiver._current_secret_key() == bytes([2] * 32)

    def test_dave_session_read_live_from_connection(self):
        conn = _make_conn()
        receiver, conn = _make_receiver(conn)
        receiver.start()
        assert receiver._current_dave_session() is None
        rotated = object()
        conn.dave_session = rotated
        assert receiver._current_dave_session() is rotated


class TestHookLifecycle:
    def test_stop_restores_prior_hook_and_removes_only_own_listener(self):
        async def original_hook(ws, msg):
            return None

        conn = _make_conn()
        conn.hook = original_hook
        receiver, conn = _make_receiver(conn)
        receiver.start()
        assert conn.hook is not original_hook       # wrapped
        assert len(conn.listeners) == 1
        receiver.stop()
        # The dead receiver must not stay chained on the connection.
        assert conn.hook is original_hook
        assert conn.listeners == []

    def test_stop_does_not_clobber_a_newer_receivers_hook(self):
        conn = _make_conn()
        receiver1, conn = _make_receiver(conn)
        receiver1.start()
        receiver2, _ = _make_receiver(conn)
        receiver2.start()                            # wraps receiver1's hook
        hook_of_r2 = conn.hook
        receiver1.stop()                             # r1 is NOT current owner
        assert conn.hook is hook_of_r2               # r2's hook untouched

    @pytest.mark.asyncio
    async def test_fresh_rejoin_receiver_is_fully_active_after_old_stop(self):
        """leave → fresh rejoin: only the new receiver hears and delivers."""
        old_conn = _make_conn(ssrc=1111)
        old_receiver, old_conn = _make_receiver(old_conn)
        old_receiver.start()
        old_receiver.stop()                          # leave

        new_conn = _make_conn(ssrc=2222)
        new_receiver, new_conn = _make_receiver(new_conn)
        new_receiver.start()                         # rejoin

        # SPEAKING event maps in the NEW receiver only.
        await new_conn.hook(None, {"op": 5, "d": {"ssrc": 572, "user_id": "42", "speaking": 1}})
        assert new_receiver._ssrc_to_user[572] == 42
        assert 572 not in old_receiver._ssrc_to_user

        # Old receiver delivers nothing after stop; new one delivers.
        sink_frames = []
        new_receiver.set_frame_sink(lambda user_id, ssrc, pcm: sink_frames.append((user_id, pcm)))
        old_frames = []
        old_receiver.set_frame_sink(lambda user_id, ssrc, pcm: old_frames.append(pcm))

        new_receiver._deliver_pcm(572, b"\x01\x00" * 1920)
        assert sink_frames and sink_frames[0][0] == 42
        assert old_frames == []
        assert old_conn.listeners == []
        assert len(new_conn.listeners) == 1


class _AppendTransport:
    def __init__(self):
        self.appended = []

    async def connect(self):
        return None

    async def append_audio(self, pcm):
        self.appended.append(pcm)

    def cancel_response_nowait(self):
        return None

    async def close(self):
        return None


class TestThreadBoundaryChain:
    @pytest.mark.asyncio
    async def test_socketreader_thread_delivery_reaches_transport_append(self):
        """The real cross-thread path the same-thread unit tests missed:
        receiver frame callback on a BACKGROUND thread (SocketReader stand-in)
        → lane queue → pump on the gateway loop → transport.append_audio."""
        pytest.importorskip("numpy")
        from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

        transport = _AppendTransport()
        lane = RealtimeVoiceLane(
            guild_id=111, text_channel_id=222,
            config=load_realtime_voice_config({"enabled": True}),
            api_key="sk-test",
            transport_factory=lambda **kw: transport,
        )
        assert await lane.start() is True

        receiver, conn = _make_receiver()
        receiver.start()
        receiver.map_ssrc(572, 42)
        receiver.set_frame_sink(lane.on_input_frame)

        frame = b"\x01\x00" * 1920  # one 20ms 48k stereo frame

        def _socketreader():
            for _ in range(3):
                receiver._deliver_pcm(572, frame)

        thread = threading.Thread(target=_socketreader, name="SocketReader-test")
        thread.start()
        thread.join(timeout=5)

        # Pump runs on this loop — yield until frames arrive.
        for _ in range(50):
            if len(transport.appended) == 3:
                break
            await asyncio.sleep(0.02)
        await lane.stop()

        assert len(transport.appended) == 3
        assert len(transport.appended[0]) == len(frame) // 4  # resampled
        assert lane.telemetry.counter("input_frames") == 3


class TestLaneFirstEventLogs:
    @pytest.mark.asyncio
    async def test_first_input_and_first_append_log_once(self, caplog):
        pytest.importorskip("numpy")
        from plugins.platforms.discord.realtime.lane import RealtimeVoiceLane

        transport = _AppendTransport()
        lane = RealtimeVoiceLane(
            guild_id=111, text_channel_id=222,
            config=load_realtime_voice_config({"enabled": True}),
            api_key="sk-test",
            transport_factory=lambda **kw: transport,
        )
        assert await lane.start() is True
        with caplog.at_level(logging.INFO):
            lane.on_input_frame(42, 572, b"\x01\x00" * 1920)
            lane.on_input_frame(42, 572, b"\x01\x00" * 1920)
            for _ in range(50):
                if len(transport.appended) == 2:
                    break
                await asyncio.sleep(0.02)
        await lane.stop()
        first_input = [r for r in caplog.records
                       if "discord_realtime first_input_frame" in r.message]
        first_append = [r for r in caplog.records
                        if "discord_realtime first_append_audio" in r.message]
        assert len(first_input) == 1
        assert len(first_append) == 1
