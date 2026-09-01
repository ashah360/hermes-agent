"""Per-guild realtime voice lane: single-owner state machine.

One lane per guild, created at voice-join time when the feature flag is on.
The lane owns inbound audio (the receiver's frame tap), the provider
WebSocket, turn sequencing on the audio plane, the dispatch registry
(ownership epochs), rollover, and demotion back to the cascaded lane.

States (ADR D2):
- CASCADED  — lane inactive; the legacy silence-detection path owns audio.
- REALTIME  — lane owns audio end-to-end.
- DEMOTING  — provider connection lost; frames buffer (bounded) while a
  reconnect is attempted inside the budget; on failure the buffer is handed
  to the cascaded path exactly once.
"""

from __future__ import annotations

import asyncio
import collections
import enum
import logging
import threading
import time
from typing import Any, Callable, Deque, Optional, Tuple

from .config import RealtimeVoiceConfig, load_realtime_voice_config, resolve_api_key
from .telemetry import LaneTelemetry

logger = logging.getLogger(__name__)

# Discord-native inbound frame geometry (matches VoiceReceiver output).
_DISCORD_BYTES_PER_SECOND = 48000 * 2 * 2


class LaneState(enum.Enum):
    CASCADED = "cascaded"
    REALTIME = "realtime"
    DEMOTING = "demoting"


class RealtimeVoiceLane:
    """Owns one guild's realtime voice conversation."""

    def __init__(
        self,
        *,
        guild_id: int,
        text_channel_id: Optional[int],
        config: RealtimeVoiceConfig,
        api_key: Optional[str],
        adapter: Any = None,
        voice_client: Any = None,
        receiver: Any = None,
        transport_factory: Optional[Callable[..., Any]] = None,
        worker_bridge: Any = None,
        telemetry: Optional[LaneTelemetry] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.guild_id = guild_id
        self.text_channel_id = text_channel_id
        self.config = config
        self._api_key = api_key
        self._adapter = adapter
        self._voice_client = voice_client
        self._receiver = receiver
        self._transport_factory = transport_factory
        self.worker_bridge = worker_bridge
        self.telemetry = telemetry or LaneTelemetry(guild_id, clock=clock)
        self._clock = clock

        self.state = LaneState.CASCADED
        self.transport: Any = None
        self._session_started_at: Optional[float] = None

        # ── Audio plane (turn sequencing — never touches dispatches) ──
        self.turn_seq = 0
        self._speech_stopped_at: Optional[float] = None
        self._first_audio_recorded = False
        self._stream_child: Any = None  # StreamingMixerChild of current response

        # ── Inbound frames: SocketReader thread → asyncio pump ──
        self._frame_queue: Deque[Tuple[int, int, bytes]] = collections.deque()
        self._frame_lock = threading.Lock()
        self._frame_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pump_task: Optional[asyncio.Task] = None

        # ── Demotion buffer (DEMOTING state), bounded by config seconds ──
        self._demotion_buffer: Deque[Tuple[int, bytes]] = collections.deque()
        self._demotion_buffer_bytes = 0
        self._demoted = False

        # ── Dispatch registry (ownership epochs; ADR D8) ──
        from .events import DispatchRegistry

        self.registry = DispatchRegistry(clock=clock)

        # ── Worker events: safe-boundary reinjection queue (ADR D7) ──
        self.delivered_events: list = []       # claim-passed events, in order
        self._pending_injections: Deque[Any] = collections.deque()
        self._user_speaking = False
        self._response_active = False
        self._result_payloads: dict = {}       # dispatch_id -> result payload
        self.last_speaker_user_id = 0

        # ── Exact transcript window (rollover restore, ADR D10) ──
        self._transcript: Deque[Tuple[str, str]] = collections.deque(
            maxlen=max(4, config.transcript_window_turns)
        )
        self._output_transcript_parts: list = []

        self._stopped = False

    # ------------------------------------------------------------------
    # Transport-facing surface
    # ------------------------------------------------------------------

    def get_projection(self) -> str:
        """Join-time projection, built once and frozen (byte-stable across
        reconnects and rollover — ADR D5/D10)."""
        cached = getattr(self, "_projection", None)
        if cached is not None:
            return cached
        from .projection import build_projection

        guild_name = voice_channel_name = text_channel_name = user_display_name = ""
        vc = self._voice_client
        try:
            channel = getattr(vc, "channel", None)
            if channel is not None:
                voice_channel_name = str(getattr(channel, "name", "") or "")
                guild = getattr(channel, "guild", None)
                guild_name = str(getattr(guild, "name", "") or "")
        except Exception:
            pass
        adapter = self._adapter
        if adapter is not None and self.text_channel_id and getattr(adapter, "_client", None):
            try:
                ch = adapter._client.get_channel(int(self.text_channel_id))
                text_channel_name = str(getattr(ch, "name", "") or "")
            except Exception:
                pass
        self._projection = build_projection(
            config=self.config,
            guild_name=guild_name,
            voice_channel_name=voice_channel_name,
            text_channel_name=text_channel_name,
            user_display_name=user_display_name,
        )
        return self._projection

    def session_tool_schemas(self) -> list:
        """In-lane tool schemas for the provider session (never core tools)."""
        try:
            from .tools import lane_tool_schemas
        except ImportError:
            return []
        return lane_tool_schemas()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        """Connect the provider session. False → caller stays cascaded."""
        if not self._api_key:
            logger.warning(
                "Realtime voice lane: secret %s not resolvable — staying cascaded",
                self.config.api_key_secret,
            )
            self.telemetry.incr("lane_start_failed_no_secret")
            return False
        try:
            transport = self._build_transport()
            await asyncio.wait_for(
                transport.connect(), timeout=self.config.connect_timeout_seconds
            )
        except Exception as e:
            logger.warning("Realtime voice lane connect failed: %s", e)
            self.telemetry.incr("lane_start_failed_connect")
            return False
        self._adopt_transport(transport)
        self.state = LaneState.REALTIME
        self._session_started_at = self._clock()
        self._loop = asyncio.get_running_loop()
        self._frame_event = asyncio.Event()
        self._pump_task = asyncio.ensure_future(self._input_pump())
        self.telemetry.incr("lane_started")
        return True

    def _build_transport(self) -> Any:
        factory = self._transport_factory
        if factory is None:
            from .transport import RealtimeTransport

            factory = RealtimeTransport
        return factory(config=self.config, api_key=self._api_key, lane=self)

    def _adopt_transport(self, transport: Any) -> None:
        self.transport = transport

    async def stop(self, reason: str = "leave") -> None:
        """Tear down the lane (leave/disconnect). Workers keep running."""
        if self._stopped:
            return
        self._stopped = True
        self.state = LaneState.CASCADED
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None
        self._clear_playback()
        # Voice delivery is revoked for every dispatch; outbox text delivery
        # stays live (registry keeps records for exactly-once text posting).
        self.registry.revoke_all(reason=reason)
        transport, self.transport = self.transport, None
        if transport is not None:
            try:
                await transport.close()
            except Exception:
                pass
        self.telemetry.incr("lane_stopped")
        self.telemetry.maybe_flush(force=True)

    # ------------------------------------------------------------------
    # Inbound audio (called from discord.py's SocketReader thread)
    # ------------------------------------------------------------------

    def on_input_frame(self, user_id: int, ssrc: int, pcm: bytes) -> None:
        """Thread-safe frame intake; consumed by the asyncio pump."""
        if self._stopped:
            return
        if user_id:
            self.last_speaker_user_id = user_id
        with self._frame_lock:
            self._frame_queue.append((user_id, ssrc, pcm))
        loop, event = self._loop, self._frame_event
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                pass

    async def _input_pump(self) -> None:
        """Drain inbound frames to the provider (or the demotion buffer)."""
        assert self._frame_event is not None
        try:
            while not self._stopped:
                await self._frame_event.wait()
                self._frame_event.clear()
                while True:
                    with self._frame_lock:
                        if not self._frame_queue:
                            break
                        user_id, ssrc, pcm = self._frame_queue.popleft()
                    await self._handle_frame(user_id, ssrc, pcm)
        except asyncio.CancelledError:
            pass

    async def _handle_frame(self, user_id: int, ssrc: int, pcm: bytes) -> None:
        if self.state is LaneState.REALTIME and self.transport is not None:
            from .audio import discord_pcm_to_realtime

            try:
                await self.transport.append_audio(discord_pcm_to_realtime(pcm))
            except Exception:
                self.telemetry.incr("append_audio_errors")
        elif self.state is LaneState.DEMOTING:
            self._buffer_for_demotion(user_id, pcm)

    def _buffer_for_demotion(self, user_id: int, pcm: bytes) -> None:
        limit = int(self.config.demotion_buffer_seconds * _DISCORD_BYTES_PER_SECOND)
        self._demotion_buffer.append((user_id, pcm))
        self._demotion_buffer_bytes += len(pcm)
        while self._demotion_buffer_bytes > limit and self._demotion_buffer:
            _, dropped = self._demotion_buffer.popleft()
            self._demotion_buffer_bytes -= len(dropped)

    # ------------------------------------------------------------------
    # Audio plane: turn sequencing + barge-in (never touches dispatches)
    # ------------------------------------------------------------------

    def on_user_speech_started(self) -> None:
        """Server-VAD speech start: immediate barge-in on the audio plane.

        Bumps ``turn_seq`` and silences playback — it NEVER touches dispatch
        ownership (ADR D8): follow-up speech must not stale a live worker.
        """
        started = self._clock()
        self.turn_seq += 1
        self._user_speaking = True
        self._clear_playback()
        if self.transport is not None:
            try:
                self.transport.cancel_response_nowait()
            except Exception:
                pass
        self.telemetry.record_ms("barge_in_stop_ms", (self._clock() - started) * 1000.0)
        self.telemetry.incr("barge_ins")

    def on_user_speech_stopped(self) -> None:
        self._speech_stopped_at = self._clock()
        self._first_audio_recorded = False
        self._user_speaking = False
        self._try_flush_injections()

    def on_response_audio_delta(self, pcm_24k_mono: bytes) -> None:
        """Provider audio delta → mixer stream child (Discord-geometry PCM)."""
        if not self._first_audio_recorded and self._speech_stopped_at is not None:
            self.telemetry.record_ms(
                "speech_end_to_first_audio_ms",
                (self._clock() - self._speech_stopped_at) * 1000.0,
            )
            self._first_audio_recorded = True
        child = self._ensure_stream_child()
        if child is None:
            return
        from .audio import realtime_pcm_to_discord

        child.feed(realtime_pcm_to_discord(pcm_24k_mono))
        self.telemetry.gauge("audio_out_queue_depth_ms", child.pending_ms())

    def on_response_created(self, response_id: Optional[str]) -> None:
        self._response_active = True

    def on_response_done(self) -> None:
        child = self._stream_child
        if child is not None:
            child.end()
            self._stream_child = None
        self._response_active = False
        if self._output_transcript_parts:
            self.note_transcript("assistant", "".join(self._output_transcript_parts))
            self._output_transcript_parts = []
        self._try_flush_injections()

    def on_output_transcript_delta(self, text: str, response_id: Optional[str]) -> None:
        if text:
            self._output_transcript_parts.append(text)

    def on_input_transcript(self, text: str) -> None:
        if text:
            self.note_transcript("user", text)

    def on_function_call(self, name: str, args: dict, call_id: str) -> None:
        """Provider function call → in-lane tool handler → function output."""
        from .tools import handle_lane_tool

        output = handle_lane_tool(self, name, args or {})
        transport = self.transport
        if transport is None:
            return

        async def _reply() -> None:
            try:
                await transport.send_function_output(call_id, output)
                await transport.create_response()
            except Exception:
                logger.debug("function output send failed", exc_info=True)

        self._schedule(_reply())

    def on_transport_closed(self, exc: Optional[BaseException]) -> None:
        """Provider socket dropped outside our control.

        Checkpoint behavior (no reconnect machinery yet): demote cleanly and
        hand audio ownership back to the cascaded lane exactly once — the
        frame sink is removed so the silence-detection path owns utterances
        again. A user rejoin (or /voice join) re-attempts REALTIME.
        """
        self.telemetry.incr("transport_drops")
        if self._stopped:
            return
        logger.warning(
            "Realtime voice transport dropped (guild=%d): %s — demoting to "
            "cascaded voice", self.guild_id, exc,
        )
        self._demote_to_cascaded()

    def _demote_to_cascaded(self) -> None:
        self.state = LaneState.CASCADED
        self._clear_playback()
        receiver = self._receiver
        if receiver is not None and hasattr(receiver, "set_frame_sink"):
            try:
                receiver.set_frame_sink(None)
            except Exception:
                pass
        adapter = self._adapter
        if adapter is not None:
            try:
                getattr(adapter, "_realtime_lanes", {}).pop(self.guild_id, None)
            except Exception:
                pass
        transport, self.transport = self.transport, None
        if transport is not None:
            self._schedule(self._close_transport(transport))
        self.telemetry.incr("demotions")
        self.telemetry.maybe_flush(force=True)

    @staticmethod
    async def _close_transport(transport: Any) -> None:
        try:
            await transport.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker events: claim → queue → inject at a safe boundary (ADR D7/D8)
    # ------------------------------------------------------------------

    def deliver_worker_event(self, event: Any) -> bool:
        """Claim + queue one typed worker event. False → dropped as stale."""
        if self._stopped or not self.registry.claim_delivery(event):
            self.telemetry.incr("stale_events_dropped")
            return False
        self.telemetry.record_ms(
            "worker_event_latency_ms", (self._clock() - event.ts) * 1000.0
        )
        self.delivered_events.append(event)
        self._pending_injections.append(event)
        self._try_flush_injections()
        return True

    def deliver_worker_event_threadsafe(self, event: Any) -> None:
        """Executor-thread-safe delivery (bridge workers run off-loop)."""
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self.deliver_worker_event, event)
                return
            except RuntimeError:
                pass
        self.deliver_worker_event(event)

    def _try_flush_injections(self) -> None:
        """Inject queued events when neither the user nor a response is live."""
        if self._user_speaking or self._response_active:
            return
        transport = self.transport
        if transport is None:
            return
        while self._pending_injections:
            event = self._pending_injections.popleft()
            self._schedule(self._inject_event(transport, event))

    async def _inject_event(self, transport: Any, event: Any) -> None:
        try:
            await transport.inject_item("system", self._render_event(event))
            await self._respond_to_event(transport, event)
        except Exception:
            logger.debug("worker event injection failed", exc_info=True)

    async def _respond_to_event(self, transport: Any, event: Any) -> None:
        """Trigger the spoken reaction to an injected event (native path).

        Overridden behavior for figure-bearing sourced completions lands in
        the exact-TTS slice (ADR D13): those never ride a native response.
        """
        await transport.create_response()

    @staticmethod
    def _render_event(event: Any) -> str:
        lines = [
            f"[background worker update] status={event.type} "
            f"dispatch={event.dispatch_id}",
            event.spoken_hint,
        ]
        if event.sources:
            refs = "; ".join(
                f"{s.get('title', 'source')} ({s.get('url', '')}, fetched {s.get('fetched_at', '?')})"
                for s in event.sources
            )
            lines.append(f"Sources: {refs}")
        if event.detail_ref:
            lines.append(
                "Full details, tables, and citations are posted in the bound "
                "text channel."
            )
        lines.append(
            "React naturally and briefly in your own words; never read tables "
            "or URLs aloud."
        )
        return "\n".join(lines)

    def store_result_payload(self, dispatch_id: str, payload: dict) -> None:
        self._result_payloads[dispatch_id] = dict(payload or {})

    def result_payload_for(self, dispatch_id: str) -> Optional[dict]:
        return self._result_payloads.get(dispatch_id)

    # ------------------------------------------------------------------
    # Exact transcript window (ADR D10)
    # ------------------------------------------------------------------

    def note_transcript(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._transcript.append((role, text))

    def transcript_window(self) -> list:
        return list(self._transcript)

    def _schedule(self, coro) -> None:
        loop = self._loop
        try:
            if loop is not None and loop.is_running():
                loop.create_task(coro)
                return
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()

    def _ensure_stream_child(self) -> Any:
        if self._stream_child is not None and not self._stream_child.finished:
            return self._stream_child
        mixer = self._mixer()
        if mixer is None:
            return None
        try:
            from voice_mixer import StreamingMixerChild
        except ImportError:
            try:
                from ..voice_mixer import StreamingMixerChild
            except ImportError:
                from plugins.platforms.discord.voice_mixer import StreamingMixerChild
        child = StreamingMixerChild(f"realtime-{self.guild_id}")
        mixer.attach_stream(child)
        self._stream_child = child
        return child

    def _mixer(self) -> Any:
        adapter = self._adapter
        if adapter is None:
            return None
        return getattr(adapter, "_voice_mixers", {}).get(self.guild_id)

    def _clear_playback(self) -> None:
        child = self._stream_child
        if child is not None:
            child.clear()
            child.end()
            self._stream_child = None
        adapter = self._adapter
        if adapter is not None and hasattr(adapter, "interrupt_voice_playback"):
            try:
                adapter.interrupt_voice_playback(self.guild_id)
            except Exception:
                pass


def create_lane(
    *,
    adapter: Any,
    guild_id: int,
    voice_client: Any,
    receiver: Any,
    raw_config: Optional[dict] = None,
    transport_factory: Optional[Callable[..., Any]] = None,
) -> RealtimeVoiceLane:
    """Build a lane for a guild from the raw config mapping."""
    config = load_realtime_voice_config(raw_config)
    text_channel_id = getattr(adapter, "_voice_text_channels", {}).get(guild_id)
    return RealtimeVoiceLane(
        guild_id=guild_id,
        text_channel_id=text_channel_id,
        config=config,
        api_key=resolve_api_key(config),
        adapter=adapter,
        voice_client=voice_client,
        receiver=receiver,
        transport_factory=transport_factory,
    )
