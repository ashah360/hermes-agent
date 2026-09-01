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
        self._result_payloads: dict = {}       # dispatch_id -> result payload
        self.last_speaker_user_id = 0

        # ── Response coordinator: at most ONE provider response in flight.
        # Ownership is marked SYNCHRONOUSLY before scheduling response.create
        # (waiting for the async response.created echo is the race that
        # produced overlapping replies live). ──
        self._response_owner: Optional[str] = None
        self._provider_response_active = False
        self._cancel_sent = False
        self._pending_continuation = False

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
        self._response_owner = None
        self._provider_response_active = False
        self._pending_continuation = False
        self._pending_injections.clear()
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

        Bumps ``turn_seq`` and ALWAYS silences playback — it NEVER touches
        dispatch ownership (ADR D8). ``response.cancel`` is sent only when a
        response is actually in flight, and exactly once per response (the
        unconditional cancel produced ``response_cancel_not_active`` spam
        live).
        """
        started = self._clock()
        self.turn_seq += 1
        self._user_speaking = True
        self._clear_playback()
        response_live = self._response_owner is not None or self._provider_response_active
        if response_live and not self._cancel_sent and self.transport is not None:
            try:
                self.transport.cancel_response_nowait()
                self._cancel_sent = True
                logger.info(
                    "discord_realtime cancel guild=%s owner=%s turn=%d",
                    self.guild_id, self._response_owner, self.turn_seq,
                )
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
        self._provider_response_active = True
        # A VAD-auto response we didn't schedule still occupies the single
        # in-flight slot.
        if self._response_owner is None:
            self._response_owner = "native"
        logger.info(
            "discord_realtime response_created guild=%s owner=%s rid=%s queue=%d",
            self.guild_id, self._response_owner, response_id,
            len(self._pending_injections),
        )

    def on_response_done(self) -> None:
        child = self._stream_child
        if child is not None:
            child.end()
            self._stream_child = None
        if self._output_transcript_parts:
            self.note_transcript("assistant", "".join(self._output_transcript_parts))
            self._output_transcript_parts = []
        # Release the single in-flight slot, then advance exactly one step:
        # a pending function-call continuation outranks queued worker events.
        prior_owner = self._response_owner
        self._provider_response_active = False
        self._response_owner = None
        self._cancel_sent = False
        logger.info(
            "discord_realtime response_done guild=%s owner=%s queue=%d continuation=%s",
            self.guild_id, prior_owner, len(self._pending_injections),
            self._pending_continuation,
        )
        if self._pending_continuation:
            self._pending_continuation = False
            self._own_and_create("continuation")
            return
        self._try_flush_injections()

    def _own_and_create(self, owner: str, *, instructions: Optional[str] = None) -> None:
        """Synchronously take the in-flight slot, then schedule the create."""
        self._response_owner = owner
        transport = self.transport
        if transport is None:
            self._response_owner = None
            return
        logger.info(
            "discord_realtime response_create guild=%s owner=%s queue=%d",
            self.guild_id, owner, len(self._pending_injections),
        )

        async def _create() -> None:
            try:
                await transport.create_response(instructions=instructions)
            except Exception:
                logger.debug("response.create failed", exc_info=True)
                if self._response_owner == owner:
                    self._response_owner = None

        self._schedule(_create())

    def on_output_transcript_delta(self, text: str, response_id: Optional[str]) -> None:
        if text:
            self._output_transcript_parts.append(text)

    def on_input_transcript(self, text: str) -> None:
        if text:
            self.note_transcript("user", text)

    def on_function_call(self, name: str, args: dict, call_id: str) -> None:
        """Provider function call → in-lane tool → output + ONE continuation.

        The function-call output continuation owns the immediate spoken
        acknowledgement. The calling response is normally still open when
        the arguments arrive, so the continuation is deferred to its
        ``response.done`` — issuing ``response.create`` while a response is
        active is exactly the overlap observed live.
        """
        from .tools import handle_lane_tool

        output = handle_lane_tool(self, name, args or {})
        logger.info(
            "discord_realtime tool_call guild=%s name=%s call_id=%s",
            self.guild_id, name, call_id,
        )
        transport = self.transport
        if transport is None:
            return

        async def _send_output() -> None:
            try:
                await transport.send_function_output(call_id, output)
            except Exception:
                logger.debug("function output send failed", exc_info=True)

        self._schedule(_send_output())
        if self._response_owner is not None or self._provider_response_active:
            self._pending_continuation = True
        else:
            self._own_and_create(f"fn:{call_id}")

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
            logger.info(
                "discord_realtime stale_event_dropped guild=%s dispatch=%s type=%s",
                self.guild_id, event.dispatch_id, event.type,
            )
            return False
        self.telemetry.record_ms(
            "worker_event_latency_ms", (self._clock() - event.ts) * 1000.0
        )
        logger.info(
            "discord_realtime worker_event guild=%s dispatch=%s type=%s queue=%d",
            self.guild_id, event.dispatch_id, event.type,
            len(self._pending_injections) + 1,
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
        """Advance the event queue under the single-response invariant.

        ``started`` events are context-only: injected as conversation items
        without ever creating a response (the function-call continuation owns
        the acknowledgement — a second create here was the live overlap bug).
        Every other event takes the in-flight slot; at most one is advanced
        per response cycle, the rest wait for ``response.done``.
        """
        transport = self.transport
        if transport is None or self._user_speaking:
            return
        while self._pending_injections:
            event = self._pending_injections[0]
            if event.type == "started":
                self._pending_injections.popleft()
                self._schedule(self._inject_item_only(transport, event))
                continue
            if self._response_owner is not None or self._provider_response_active:
                return  # slot busy — this event goes out on response.done
            self._pending_injections.popleft()
            self._response_owner = f"event:{event.dispatch_id}"
            logger.info(
                "discord_realtime response_create guild=%s owner=%s queue=%d",
                self.guild_id, self._response_owner, len(self._pending_injections),
            )
            self._schedule(self._inject_event_owned(transport, event))
            return

    async def _inject_item_only(self, transport: Any, event: Any) -> None:
        try:
            await transport.inject_item("system", self._render_event(event))
        except Exception:
            logger.debug("worker event item injection failed", exc_info=True)

    async def _inject_event_owned(self, transport: Any, event: Any) -> None:
        owner = f"event:{event.dispatch_id}"
        try:
            await transport.inject_item("system", self._render_event(event))
            await transport.create_response()
        except Exception:
            logger.debug("worker event injection failed", exc_info=True)
            if self._response_owner == owner:
                self._response_owner = None

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


def _principal_provider_for(adapter: Any, lane: "RealtimeVoiceLane"):
    """Provider resolving the worker principal from the gateway runner.

    Uses ``GatewayRunner.build_realtime_worker_principal`` — the same
    model/provider/toolset resolution every gateway agent goes through.
    Identity is read at BUILD time: the chat id stays fixed to the lane's
    bound text channel, while the user id follows the lane's current
    speaker (``last_speaker_user_id``) — never a 0 captured at join
    forever. Returns None (fail closed) when the runner or credentials are
    absent; the bridge turns that into one clear failed event.
    """

    def _provider():
        runner = getattr(adapter, "gateway_runner", None)
        build = getattr(runner, "build_realtime_worker_principal", None)
        if not callable(build):
            return None
        speaker = getattr(lane, "last_speaker_user_id", 0) or 0
        try:
            principal = build(
                chat_id=str(lane.text_channel_id or ""),
                user_id=str(speaker) if speaker else None,
            )
        except Exception:
            logger.warning("realtime worker principal build failed", exc_info=True)
            return None
        if principal is None:
            return None
        # Worker performance shaping (config-scoped to realtime voice
        # workers): SAME model/provider — speed via low reasoning effort +
        # priority service tier, merged over the session's own overrides.
        cfg = lane.config
        try:
            principal.reasoning_config = {
                "enabled": True,
                "effort": cfg.worker_reasoning_effort,
            }
            overrides = dict(getattr(principal, "request_overrides", None) or {})
            overrides["service_tier"] = cfg.worker_service_tier
            principal.request_overrides = overrides
        except Exception:
            logger.debug("worker principal perf shaping failed", exc_info=True)
        return principal

    return _provider


def create_lane(
    *,
    adapter: Any,
    guild_id: int,
    voice_client: Any,
    receiver: Any,
    raw_config: Optional[dict] = None,
    transport_factory: Optional[Callable[..., Any]] = None,
) -> RealtimeVoiceLane:
    """Build a PRODUCTION lane: transport + worker bridge + outbox wired."""
    config = load_realtime_voice_config(raw_config)
    text_channel_id = getattr(adapter, "_voice_text_channels", {}).get(guild_id)
    lane = RealtimeVoiceLane(
        guild_id=guild_id,
        text_channel_id=text_channel_id,
        config=config,
        api_key=resolve_api_key(config),
        adapter=adapter,
        voice_client=voice_client,
        receiver=receiver,
        transport_factory=transport_factory,
    )
    # Production wiring — the live "worker dispatch unavailable" defect was
    # this bridge never being instantiated outside tests.
    from .outbox import DiscordTextOutbox
    from .worker_bridge import WorkerBridge

    lane.worker_bridge = WorkerBridge(
        lane=lane,
        adapter=adapter,
        outbox=DiscordTextOutbox(adapter=adapter, lane=lane),
        principal_provider=_principal_provider_for(adapter, lane),
    )
    return lane
