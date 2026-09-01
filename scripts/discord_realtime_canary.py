#!/usr/bin/env python3
"""Env-gated live provider canary for the Discord realtime voice lane.

Validates, against the REAL gpt-realtime-2.1 API (no Discord involved):
  1. session bootstrap schema — session.updated echo, MODEL and VOICE echo
     (cedar required; a mismatch is a rollout blocker, never a silent swap),
     reasoning_effort acceptance (or documented degradation);
  2. event-name mapping — every event type the provider actually sends is
     listed, flagged mapped/unmapped against the transport's dispatcher;
  3. a REAL audio turn — synthesized 24 kHz pcm16 audio appended + committed,
     response requested, audio deltas received;
  4. latency — speech-end (commit) → first audio delta, for the audio turn
     and a text-prompted turn (p50 over --turns repetitions);
  5. interruption — response.cancel at first delta → terminal event latency.

Gating: BOTH must be set or the script exits 2 without any network call:
  HERMES_REALTIME_CANARY=1
  OPENAI_API_KEY=<secret>          (or the secret named by --api-key-secret)

The key is read from the environment and NEVER printed. Output is a single
JSON report on stdout. Exit codes: 0 = all checks passed, 1 = a check
failed, 2 = not gated/enabled.

Run:  python scripts/discord_realtime_canary.py [--turns 5] [--model ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from plugins.platforms.discord.realtime.config import load_realtime_voice_config  # noqa: E402
from plugins.platforms.discord.realtime.transport import RealtimeTransport  # noqa: E402

# Event types the transport's dispatcher understands (keep in sync with
# transport._dispatch_event; the canary flags anything the live API sends
# outside these sets so a provider rename is caught before deployment).
_MAPPED_EVENTS = {
    "session.created", "session.updated", "error",
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "response.created", "response.done", "response.completed", "response.cancelled",
    "response.output_audio.delta", "response.audio.delta",
    "response.output_audio_transcript.delta", "response.audio_transcript.delta",
    "conversation.item.input_audio_transcription.completed",
    "response.function_call_arguments.done",
}
# Benign bookkeeping events, enumerated from the GA server-events API
# reference (developers.openai.com/api/reference/resources/realtime/
# server-events) — explicit names, not prefix guesses. Anything the live
# API sends outside _MAPPED_EVENTS ∪ _KNOWN_IGNORED_EVENTS is reported as
# unmapped and fails the canary.
_KNOWN_IGNORED_EVENTS = {
    "conversation.created",
    "conversation.item.added",
    "conversation.item.created",
    "conversation.item.deleted",
    "conversation.item.done",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.failed",
    "conversation.item.input_audio_transcription.segment",
    "conversation.item.retrieved",
    "conversation.item.truncated",
    "input_audio_buffer.cleared",
    "input_audio_buffer.committed",
    "input_audio_buffer.dtmf_event_received",
    "input_audio_buffer.timeout_triggered",
    "item.input_audio_transcription.logprobs",
    "mcp_list_tools.completed",
    "mcp_list_tools.failed",
    "mcp_list_tools.in_progress",
    "output_audio_buffer.cleared",
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "rate_limits.updated",
    "response.content_part.added",
    "response.content_part.done",
    "response.function_call_arguments.delta",
    "response.mcp_call.completed",
    "response.mcp_call.failed",
    "response.mcp_call.in_progress",
    "response.mcp_call_arguments.delta",
    "response.mcp_call_arguments.done",
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.output_item.added",
    "response.output_item.done",
    "response.output_text.delta",
    "response.output_text.done",
}


class _ProbeLane:
    """Minimal lane stand-in: records transport callbacks + timings.

    Any provider ``error`` event is collected in ``errors`` — the canary
    treats a non-empty list as a FAILURE (zero-error contract), never as a
    logged warning.
    """

    def __init__(self) -> None:
        self.first_audio_at: float | None = None
        self.audio_bytes = 0
        self.done_at: float | None = None
        self.done_event = asyncio.Event()
        self.first_audio_event = asyncio.Event()
        self.transcript_parts: list[str] = []
        self.errors: list[dict] = []
        self.speech_stopped_at: float | None = None

    def reset_turn(self) -> None:
        self.first_audio_at = None
        self.done_at = None
        self.transcript_parts = []
        self.speech_stopped_at = None
        self.done_event = asyncio.Event()
        self.first_audio_event = asyncio.Event()

    def get_projection(self) -> str:
        return (
            "You are a latency canary. Reply with ONE short sentence, "
            "then stop."
        )

    def session_tool_schemas(self) -> list:
        return []

    # transport callbacks -------------------------------------------------
    def on_user_speech_started(self):  # server VAD may fire on our audio
        pass

    def on_user_speech_stopped(self):
        self.speech_stopped_at = time.monotonic()

    def on_provider_error(self, error: dict):
        self.errors.append(dict(error or {}))

    def on_response_created(self, response_id):
        pass

    def on_response_audio_delta(self, pcm: bytes):
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()
            self.first_audio_event.set()
        self.audio_bytes += len(pcm)

    def on_output_transcript_delta(self, text, response_id):
        self.transcript_parts.append(text or "")

    def on_input_transcript(self, text):
        pass

    def on_response_done(self):
        self.done_at = time.monotonic()
        self.done_event.set()

    def on_function_call(self, name, args, call_id):
        pass

    def on_transport_closed(self, exc):
        self.done_event.set()


class _EventTapTransport(RealtimeTransport):
    """Records every raw provider event type for the mapping report.

    The accepted ``session.updated`` echo is stored by the base transport
    itself during bootstrap (``session_echo``) — bootstrap consumes that
    frame before dispatch, so a dispatch-side tap can never see it.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_event_types: set[str] = set()

    def _dispatch_event(self, frame: dict) -> None:
        ftype = frame.get("type", "")
        if ftype:
            self.seen_event_types.add(ftype)
        super()._dispatch_event(frame)


def _echo_failures(echo: dict, *, expected_model: str, expected_voice: str) -> list[str]:
    """Exact echo assertions. A MISSING echo is a failure, not a pass."""
    failures: list[str] = []
    if not echo:
        return [f"no session.updated echo captured (expected model "
                f"{expected_model!r}, voice {expected_voice!r})"]
    if echo.get("model") != expected_model:
        failures.append(f"model echo {echo.get('model')!r} != {expected_model!r}")
    if echo.get("type") != "realtime":
        failures.append(f"session type echo {echo.get('type')!r} != 'realtime'")
    voice = _EventTapTransport._echo_voice(echo)
    if voice != expected_voice:
        failures.append(f"voice echo {voice!r} != {expected_voice!r}")
    return failures


def _synth_speech_like_pcm(seconds: float = 1.2, sample_rate: int = 24000) -> bytes:
    """Synthesize speech-band audio (no numpy dependency needed)."""
    frames = bytearray()
    n = int(sample_rate * seconds)
    for i in range(n):
        t = i / sample_rate
        # Two formant-ish tones with a 4 Hz amplitude wobble — enough energy
        # in the speech band for VAD/commit to treat it as an utterance.
        env = 0.55 + 0.45 * math.sin(2 * math.pi * 4.0 * t)
        sample = env * (
            0.6 * math.sin(2 * math.pi * 220.0 * t)
            + 0.4 * math.sin(2 * math.pi * 660.0 * t)
        )
        value = max(-32768, min(32767, int(sample * 20000)))
        frames += value.to_bytes(2, "little", signed=True)
    return bytes(frames)


async def _wait(event: asyncio.Event, timeout: float) -> bool:
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


async def _text_turn(transport, probe, timeout: float) -> dict:
    probe.reset_turn()
    start = time.monotonic()
    await transport.inject_item("user", "Say hello in one short sentence.")
    await transport.create_response()
    ok = await _wait(probe.first_audio_event, timeout)
    first_ms = (probe.first_audio_at - start) * 1000.0 if ok else None
    await _wait(probe.done_event, timeout)
    return {"ok": ok, "first_audio_ms": first_ms}


async def _vad_audio_turn(transport, probe, timeout: float,
                          silence_ms: int = 520, *, pace: bool = True) -> dict:
    """Auto-VAD turn modeling REAL Discord behavior: finite speech packets,
    then packet cessation — no further input except the zero tail the lane's
    silence-tail machinery appends (Discord stops RTP after a speaker goes
    quiet, so server VAD only closes the turn if that tail lands). No manual
    commit, no manual response.create: server VAD owns the turn.
    """
    probe.reset_turn()
    pcm = _synth_speech_like_pcm()
    chunk = 4800  # 100ms of 24kHz mono pcm16
    for i in range(0, len(pcm), chunk):
        await transport.append_audio(pcm[i:i + chunk])
        if pace:
            await asyncio.sleep(0.1)
    speech_end = time.monotonic()
    # The lane's silence tail (discord_input_idle_ms then zeros exceeding
    # server_vad_silence_duration_ms), modeled exactly.
    if pace:
        await asyncio.sleep(0.35)
    await transport.append_audio(b"\x00" * (silence_ms * 48))
    ok = await _wait(probe.first_audio_event, timeout)
    first_ms = (probe.first_audio_at - speech_end) * 1000.0 if ok else None
    stopped_ms = (
        (probe.speech_stopped_at - speech_end) * 1000.0
        if probe.speech_stopped_at else None
    )
    await _wait(probe.done_event, timeout)
    return {"ok": ok, "first_audio_ms": first_ms, "speech_stopped_ms": stopped_ms}


async def _manual_audio_turn(transport, probe, timeout: float, *, pace: bool = True) -> dict:
    """Real audio turn on a MANUAL-VAD session (turn_detection: null).

    With server/semantic VAD + create_response the server auto-commits and
    auto-responds, so a manual commit/create there hits an already-consumed
    buffer (``input_audio_buffer_commit_empty`` — the first live run's
    defect). Manual commits are therefore exercised ONLY on this dedicated
    manual-VAD session, where they are the correct protocol. Chunks are
    paced at real time to mirror production capture.
    """
    probe.reset_turn()
    pcm = _synth_speech_like_pcm()
    chunk = 4800  # 100ms of 24kHz mono pcm16
    for i in range(0, len(pcm), chunk):
        await transport.append_audio(pcm[i:i + chunk])
        if pace:
            await asyncio.sleep(0.1)
    speech_end = time.monotonic()
    await transport.commit_audio()
    await transport.create_response()
    ok = await _wait(probe.first_audio_event, timeout)
    first_ms = (probe.first_audio_at - speech_end) * 1000.0 if ok else None
    await _wait(probe.done_event, timeout)
    return {"ok": ok, "first_audio_ms": first_ms}


async def _interruption_turn(transport, probe, timeout: float) -> dict:
    probe.reset_turn()
    await transport.inject_item(
        "user", "Count slowly from one to thirty, one number per breath."
    )
    await transport.create_response()
    if not await _wait(probe.first_audio_event, timeout):
        return {"ok": False, "cancel_to_done_ms": None}
    cancel_at = time.monotonic()
    transport.cancel_response_nowait()
    ok = await _wait(probe.done_event, timeout)
    ms = (probe.done_at - cancel_at) * 1000.0 if ok and probe.done_at else None
    return {"ok": ok, "cancel_to_done_ms": ms}


def _summ(turns):
    vals = [t["first_audio_ms"] for t in turns if t["ok"] and t["first_audio_ms"]]
    if not vals:
        return {"ok": False, "samples": 0}
    return {
        "ok": True,
        "samples": len(vals),
        "p50_ms": round(statistics.median(vals), 1),
        "min_ms": round(min(vals), 1),
        "max_ms": round(max(vals), 1),
    }


async def run_canary(args) -> int:
    api_key = os.environ.get(args.api_key_secret, "").strip()
    report: dict = {
        "canary": "discord-realtime-provider",
        "model_requested": args.model,
        "voice_requested": args.voice,
        "checks": {},
    }
    failures: list[str] = []
    seen_events: set[str] = set()

    # ── Session 1: PRODUCTION shape (server VAD default). Text turns +
    # interruption only — no manual commits here, ever: the server owns
    # commit/response on this session (first live run's commit_empty bug). ──
    config = load_realtime_voice_config({
        "enabled": True,
        "model": args.model,
        "voice": args.voice,
        "api_key_secret": args.api_key_secret,
    })
    probe = _ProbeLane()
    transport = _EventTapTransport(config=config, api_key=api_key, lane=probe)
    try:
        await asyncio.wait_for(transport.connect(), timeout=args.timeout)
        report["checks"]["connect"] = {"ok": True}
        echo = transport.session_echo
        report["model_echo"] = echo.get("model")
        report["voice_echo"] = _EventTapTransport._echo_voice(echo)
        report["session_type_echo"] = echo.get("type")
        report["reasoning_degraded"] = transport._reasoning_degraded
        failures.extend(_echo_failures(
            echo, expected_model=config.model, expected_voice=config.voice,
        ))
    except Exception as exc:
        report["checks"]["connect"] = {"ok": False, "error": str(exc)}
        report["failures"] = [f"connect failed: {exc}"]
        report["passed"] = False
        print(json.dumps(report, indent=2))
        return 1

    try:
        text_turns = [await _text_turn(transport, probe, args.timeout)
                      for _ in range(args.turns)]
        # Auto-VAD turns: finite speech packets then cessation + lane-style
        # zero tail — the exact live Discord shape.
        vad_turns = [await _vad_audio_turn(transport, probe, args.timeout)
                     for _ in range(max(1, args.turns // 2))]
        interruption = await _interruption_turn(transport, probe, args.timeout)
    finally:
        seen_events |= transport.seen_event_types
        await transport.close()
    semantic_errors = list(probe.errors)
    semantic_audio_bytes = probe.audio_bytes

    # ── Session 2: MANUAL-VAD session (turn_detection: null) for the real
    # audio turn with explicit commit/create — the correct protocol there. ──
    manual_config = load_realtime_voice_config({
        "enabled": True,
        "model": args.model,
        "voice": args.voice,
        "api_key_secret": args.api_key_secret,
        "turn_detection_type": "none",
    })
    manual_probe = _ProbeLane()
    manual_transport = _EventTapTransport(
        config=manual_config, api_key=api_key, lane=manual_probe
    )
    audio_turns: list[dict] = []
    try:
        await asyncio.wait_for(manual_transport.connect(), timeout=args.timeout)
        report["checks"]["manual_session_connect"] = {"ok": True}
        for _ in range(args.turns):
            audio_turns.append(
                await _manual_audio_turn(manual_transport, manual_probe, args.timeout)
            )
    except Exception as exc:
        report["checks"]["manual_session_connect"] = {"ok": False, "error": str(exc)}
        failures.append(f"manual-VAD session failed: {exc}")
    finally:
        seen_events |= manual_transport.seen_event_types
        await manual_transport.close()
    manual_errors = list(manual_probe.errors)

    report["checks"]["text_turn_first_audio"] = _summ(text_turns)
    report["checks"]["vad_turn_first_audio"] = _summ(vad_turns)
    report["checks"]["vad_turn_speech_stopped"] = {
        "samples": len([t for t in vad_turns if t.get("speech_stopped_ms")]),
        "values_ms": [round(t["speech_stopped_ms"], 1) for t in vad_turns
                      if t.get("speech_stopped_ms")],
    }
    report["checks"]["audio_turn_first_audio"] = _summ(audio_turns)
    report["checks"]["interruption"] = interruption
    report["audio_bytes_received"] = semantic_audio_bytes + manual_probe.audio_bytes

    unmapped = sorted(
        t for t in seen_events
        if t not in _MAPPED_EVENTS and t not in _KNOWN_IGNORED_EVENTS
    )
    report["event_types_seen"] = sorted(seen_events)
    report["event_types_unmapped"] = unmapped

    # Zero-provider-error contract: ANY error event on EITHER session fails
    # the canary outright.
    provider_errors = semantic_errors + manual_errors
    report["provider_error_events"] = provider_errors
    if provider_errors:
        failures.append(
            f"provider error events ({len(provider_errors)}): "
            + "; ".join(str(e.get("code") or e.get("message") or e) for e in provider_errors)
        )

    if not report["checks"]["text_turn_first_audio"].get("ok"):
        failures.append("text turn produced no audio")
    if not report["checks"]["vad_turn_first_audio"].get("ok"):
        failures.append("auto-VAD turn (speech + cessation tail) produced no audio")
    if not report["checks"]["audio_turn_first_audio"].get("ok"):
        failures.append("audio turn produced no audio")
    if not interruption.get("ok"):
        failures.append("interruption did not terminate the response")
    if unmapped:
        failures.append(f"unmapped provider events: {unmapped}")

    report["failures"] = failures
    report["passed"] = not failures
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-realtime-2.1")
    parser.add_argument("--voice", default="cedar")
    parser.add_argument("--api-key-secret", default="OPENAI_API_KEY",
                        help="NAME of the env secret to read (value never printed)")
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if os.environ.get("HERMES_REALTIME_CANARY") != "1":
        print(json.dumps({
            "canary": "discord-realtime-provider",
            "skipped": "set HERMES_REALTIME_CANARY=1 to enable (live network + spend)",
        }))
        return 2
    if not os.environ.get(args.api_key_secret, "").strip():
        print(json.dumps({
            "canary": "discord-realtime-provider",
            "skipped": f"secret {args.api_key_secret} not present in the environment",
        }))
        return 2
    return asyncio.run(run_canary(args))


if __name__ == "__main__":
    sys.exit(main())
