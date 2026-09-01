"""Full-duplex server-side WebSocket client for gpt-realtime-2.1 (ADR D4).

Responsibilities:
- session bootstrap (``session.update``: pcm16 both ways, server VAD with
  interrupt, ``reasoning_effort`` with graceful degradation, cedar voice with
  echo verification, projection instructions, in-lane tool schemas);
- event-name-tolerant dispatch of provider events to lane callbacks (the GA
  API renamed several ``response.audio.*`` events — both spellings map);
- outbound plumbing: audio append, response cancel/create, conversation item
  injection, function-call outputs.

The ``websockets`` dependency is imported lazily; tests inject a fake socket
via ``ws_connect``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from .config import RealtimeVoiceConfig

logger = logging.getLogger(__name__)

REALTIME_URL = "wss://api.openai.com/v1/realtime"


class VoiceMismatchError(RuntimeError):
    """Provider session echo reported a different voice than configured."""


async def _default_ws_connect(url: str, headers: list) -> Any:
    """Open a real provider socket (lazy websockets import).

    GA interface: NO ``OpenAI-Beta`` header — the live API refuses the beta
    shape with ``invalid_request_error.beta_api_shape_disabled`` (WS 4000).
    """
    try:
        from websockets.asyncio.client import connect as _connect
    except ImportError:  # pragma: no cover - legacy websockets
        from websockets.client import connect as _connect  # type: ignore
    try:
        return await _connect(url, additional_headers=headers, max_size=None)
    except TypeError:  # pragma: no cover - older kwarg name
        return await _connect(url, extra_headers=headers, max_size=None)


class RealtimeTransport:
    """One provider WebSocket session. The lane owns lifecycle decisions."""

    def __init__(
        self,
        *,
        config: RealtimeVoiceConfig,
        api_key: str,
        lane: Any,
        ws_connect: Optional[Callable[[str, list], Awaitable[Any]]] = None,
    ) -> None:
        self.config = config
        self._api_key = api_key
        self._lane = lane
        self._ws_connect = ws_connect or _default_ws_connect
        self._ws: Any = None
        self._reader_task: Optional[asyncio.Task] = None
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._reasoning_degraded = False
        # Accepted session.updated echo from bootstrap. Bootstrap consumes
        # that frame itself (it never reaches _dispatch_event), so it is
        # stored here for callers that need it (canary echo assertions).
        self.session_echo: dict = {}
        # Response-generation tracking for stale-cancel suppression: a cancel
        # is bound to the generation it was scheduled against and re-checked
        # at ACTUAL send time — response.done racing a queued cancel must
        # produce zero wire frames (live: response_cancel_not_active spam).
        self._response_in_flight = False
        self._response_gen = 0
        self.stale_cancels_suppressed = 0

    # ------------------------------------------------------------------
    # Connect / bootstrap
    # ------------------------------------------------------------------

    def _session_payload(self, *, include_reasoning: bool) -> dict:
        """GA ``session.update`` payload (developers.openai.com, GA schema).

        - ``type: "realtime"`` + ``model`` inside the session (required);
        - ``output_modalities: ["audio"]`` (``modalities`` is beta-era);
        - audio config nested under ``audio.input`` / ``audio.output`` with
          ``{type: "audio/pcm", rate: 24000}`` formats;
        - semantic VAD with conversation-mode interruption fields;
        - input transcription only at its GA nested location, and only when
          configured (the minimal live vertical omits it);
        - reasoning effort as the nested ``reasoning: {effort}`` object.
        """
        if self.config.turn_detection_type in ("none", "null", "off"):
            # Manual turn taking (canary's manual-commit session): GA spec
            # says pass null to disable VAD entirely.
            turn_detection = None
        else:
            turn_detection = {
                "type": self.config.turn_detection_type,
                "interrupt_response": True,
                "create_response": True,
            }
        audio_input: dict = {
            "format": {"type": "audio/pcm", "rate": 24000},
            "turn_detection": turn_detection,
        }
        if self.config.input_transcription_model:
            audio_input["transcription"] = {
                "model": self.config.input_transcription_model
            }
        session: dict = {
            "type": "realtime",
            "model": self.config.model,
            "output_modalities": ["audio"],
            "instructions": self._lane.get_projection(),
            "audio": {
                "input": audio_input,
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": self.config.voice,
                },
            },
            "tools": list(self._lane.session_tool_schemas()),
            "tool_choice": "auto",
        }
        if include_reasoning and self.config.reasoning_effort:
            session["reasoning"] = {"effort": self.config.reasoning_effort}
        return {"type": "session.update", "session": session}

    @staticmethod
    def _echo_voice(session: dict) -> Optional[str]:
        """Voice from a session echo — GA nested location, root fallback."""
        audio = session.get("audio") or {}
        output = audio.get("output") or {}
        return output.get("voice") or session.get("voice")

    async def connect(self) -> None:
        url = f"{REALTIME_URL}?model={self.config.model}"
        headers = [
            ("Authorization", f"Bearer {self._api_key}"),
        ]
        self._ws = await self._ws_connect(url, headers)
        try:
            await self._bootstrap_session()
        except Exception:
            await self.close()
            raise
        self._reader_task = asyncio.ensure_future(self._reader())

    async def _bootstrap_session(self) -> None:
        await self._send_json(self._session_payload(include_reasoning=True))
        while True:
            frame = await self._recv_json()
            if frame is None:
                raise ConnectionError("provider closed during session bootstrap")
            ftype = frame.get("type")
            if ftype in ("session.updated", "session.created"):
                if ftype == "session.created":
                    # Initial hello; the updated echo follows our update.
                    continue
                session_echo = dict(frame.get("session") or {})
                self.session_echo = session_echo
                echo_voice = self._echo_voice(session_echo)
                if echo_voice and echo_voice != self.config.voice:
                    raise VoiceMismatchError(
                        f"provider voice echo is {echo_voice!r}, required "
                        f"{self.config.voice!r} (cedar). Refusing a silent "
                        f"voice swap; lane falls back to cascaded."
                    )
                return
            if ftype == "error":
                err = frame.get("error") or {}
                message = str(err.get("message", err))
                param = str(err.get("param", ""))
                if (
                    not self._reasoning_degraded
                    and "reasoning" in (message + param)
                ):
                    # Provider rejected the reasoning field specifically:
                    # degrade by omission (ADR D4), log once, keep the lane.
                    self._reasoning_degraded = True
                    logger.warning(
                        "Realtime API rejected session.reasoning; continuing without it"
                    )
                    await self._send_json(self._session_payload(include_reasoning=False))
                    continue
                raise ConnectionError(f"realtime session bootstrap error: {message}")
            # Any other pre-ack event: hand to the dispatcher and keep waiting.
            self._dispatch_event(frame)

    # ------------------------------------------------------------------
    # Reader / event dispatch
    # ------------------------------------------------------------------

    async def _reader(self) -> None:
        exc: Optional[BaseException] = None
        try:
            while not self._closed:
                frame = await self._recv_json()
                if frame is None:
                    break
                self._dispatch_event(frame)
        except asyncio.CancelledError:
            return
        except BaseException as e:  # noqa: BLE001 — reported to the lane
            exc = e
        if not self._closed:
            try:
                self._lane.on_transport_closed(exc)
            except Exception:
                logger.debug("lane.on_transport_closed failed", exc_info=True)

    def _dispatch_event(self, frame: dict) -> None:
        ftype = frame.get("type", "")
        lane = self._lane
        try:
            if ftype == "input_audio_buffer.speech_started":
                lane.on_user_speech_started()
            elif ftype == "input_audio_buffer.speech_stopped":
                lane.on_user_speech_stopped()
            elif ftype == "response.created":
                self._response_in_flight = True
                self._response_gen += 1
                lane.on_response_created((frame.get("response") or {}).get("id"))
            elif ftype in ("response.output_audio.delta", "response.audio.delta"):
                b64 = frame.get("delta") or frame.get("audio") or ""
                if b64:
                    try:
                        pcm = base64.b64decode(b64)
                    except (ValueError, TypeError):
                        pcm = b""
                    if pcm:
                        lane.on_response_audio_delta(pcm)
            elif ftype in (
                "response.output_audio_transcript.delta",
                "response.audio_transcript.delta",
            ):
                lane.on_output_transcript_delta(
                    frame.get("delta") or "", frame.get("response_id")
                )
            elif ftype == "conversation.item.input_audio_transcription.completed":
                lane.on_input_transcript(frame.get("transcript") or "")
            elif ftype in ("response.done", "response.completed", "response.cancelled"):
                self._response_in_flight = False
                lane.on_response_done()
            elif ftype == "response.function_call_arguments.done":
                raw_args = frame.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args)
                except (TypeError, ValueError):
                    args = {}
                lane.on_function_call(
                    frame.get("name") or "", args, frame.get("call_id") or ""
                )
            elif ftype == "error":
                error = frame.get("error") or {}
                logger.warning("Realtime API error event: %s", error)
                on_error = getattr(lane, "on_provider_error", None)
                if callable(on_error):
                    on_error(error)
            # All other event types are intentionally ignored.
        except Exception:
            logger.debug("lane event handler failed for %s", ftype, exc_info=True)

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def append_audio(self, pcm16_24k_mono: bytes) -> None:
        if not pcm16_24k_mono:
            return
        await self._send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm16_24k_mono).decode("ascii"),
        })

    async def commit_audio(self) -> None:
        """Force a turn boundary for the appended audio (canary/manual VAD)."""
        await self._send_json({"type": "input_audio_buffer.commit"})

    def cancel_response_nowait(self) -> None:
        """Barge-in cancel, generation-bound and re-checked at SEND time.

        The token captures the response generation at scheduling; the task
        verifies — immediately before the websocket send — that the SAME
        generation is still in flight. ``response.done`` winning the race
        (or a newer response having started) suppresses the frame entirely,
        so the provider never sees ``response_cancel_not_active``. A truly
        active response is still cancelled within one loop turn (<300ms
        barge-in preserved).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        token = self._response_gen
        loop.create_task(self._send_cancel_if_active(token))

    async def _send_cancel_if_active(self, token: int) -> None:
        if not self._response_in_flight or self._response_gen != token:
            self.stale_cancels_suppressed += 1
            logger.info(
                "realtime cancel suppressed at send time (stale generation "
                "%d, current %d, in_flight=%s)",
                token, self._response_gen, self._response_in_flight,
            )
            return
        await self._send_json({"type": "response.cancel"})

    async def create_response(self, *, instructions: Optional[str] = None) -> None:
        response: dict = {}
        if instructions:
            response["instructions"] = instructions
        await self._send_json({"type": "response.create", "response": response})

    async def inject_item(self, role: str, text: str) -> None:
        content_type = "input_text" if role in ("user", "system") else "text"
        await self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": role,
                "content": [{"type": content_type, "text": text}],
            },
        })

    async def send_function_output(self, call_id: str, output: str) -> None:
        await self._send_json({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    async def _send_json(self, payload: dict) -> None:
        ws = self._ws
        if ws is None or self._closed:
            return
        async with self._send_lock:
            await ws.send(json.dumps(payload))

    async def _recv_json(self) -> Optional[dict]:
        ws = self._ws
        if ws is None:
            return None
        raw = await ws.recv()
        if raw is None:
            return None
        try:
            frame = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return frame if isinstance(frame, dict) else {}

    async def close(self) -> None:
        self._closed = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
