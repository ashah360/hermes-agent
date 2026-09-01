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
    """Open a real provider socket (lazy websockets import)."""
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

    # ------------------------------------------------------------------
    # Connect / bootstrap
    # ------------------------------------------------------------------

    def _session_payload(self, *, include_reasoning: bool) -> dict:
        session: dict = {
            "voice": self.config.voice,
            "instructions": self._lane.get_projection(),
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "interrupt_response": True,
            },
            "input_audio_transcription": {"model": "whisper-1"},
            "tools": list(self._lane.session_tool_schemas()),
            "tool_choice": "auto",
        }
        if include_reasoning and self.config.reasoning_effort:
            session["reasoning_effort"] = self.config.reasoning_effort
        return {"type": "session.update", "session": session}

    async def connect(self) -> None:
        url = f"{REALTIME_URL}?model={self.config.model}"
        headers = [
            ("Authorization", f"Bearer {self._api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
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
                echo_voice = (frame.get("session") or {}).get("voice")
                if ftype == "session.created":
                    # Initial hello; the updated echo follows our update.
                    continue
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
                    and "reasoning_effort" in (message + param)
                ):
                    # Deployed API build predates the field: degrade by
                    # omission (ADR D4), log once, keep the lane alive.
                    self._reasoning_degraded = True
                    logger.warning(
                        "Realtime API rejected reasoning_effort; continuing without it"
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
                logger.warning("Realtime API error event: %s", frame.get("error"))
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

    def cancel_response_nowait(self) -> None:
        """Fire-and-forget response.cancel (barge-in hot path)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._send_json({"type": "response.cancel"}))

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
