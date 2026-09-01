"""Config schema for the Discord realtime voice lane.

All keys live under ``gateway.platforms.discord.voice.realtime`` in
config.yaml (or the equivalent top-level ``platforms`` map that
``gateway/config.py`` also resolves).  Everything here is behavioral —
config.yaml only, never env vars.  The single credential is the ``.env``
secret NAMED by ``api_key_secret`` (default ``OPENAI_API_KEY``), read through
the scope-aware fail-closed path so multiplexed profiles never borrow another
profile's key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    enabled: bool = False
    model: str = "gpt-realtime-2.1"
    # Arbor is ChatGPT-app-only and unavailable in the Realtime API; cedar is
    # the required voice (verified against the session echo at connect).
    voice: str = "cedar"
    reasoning_effort: str = "low"
    api_key_secret: str = "OPENAI_API_KEY"
    connect_timeout_seconds: float = 8.0
    reconnect_budget_seconds: float = 15.0
    rollover_margin_seconds: int = 300
    session_max_seconds: int = 3600  # provider hard cap on realtime sessions
    max_inflight_dispatches: int = 3
    transcript_window_turns: int = 40
    demotion_buffer_seconds: float = 10.0
    # GA nested input transcription (audio.input.transcription.model).
    # Empty (default) omits the block entirely — the minimal live vertical
    # does not require input transcripts.
    input_transcription_model: str = ""
    # audio.input.turn_detection type. Production default is server_vad —
    # live logs showed semantic_vad holding turns open 12-29s, while
    # server_vad measured ~590ms speech-end->speech_stopped in the provider
    # canary. semantic_vad remains opt-in; "none" disables VAD (JSON null)
    # for manual-commit sessions (canary).
    turn_detection_type: str = "server_vad"
    # server_vad tuning (ignored for semantic_vad — server-only fields are
    # rejected by the provider there).
    server_vad_threshold: float = 0.5
    server_vad_prefix_padding_ms: int = 300
    server_vad_silence_duration_ms: int = 500
    # Discord stops sending a speaker's RTP a few Opus silence frames after
    # they stop talking, freezing the provider's audio timeline mid-turn —
    # server VAD then never accumulates its silence window and the turn
    # never closes (live: counters froze, no speech_stopped for minutes).
    # After this much wall-clock input inactivity the lane appends an
    # explicit zero-PCM tail (> server_vad_silence_duration_ms) so the turn
    # completes. 800ms default: the first true Discord E2E showed 350ms
    # split one natural ~9.9s request into three turns at ~500-700ms
    # punctuation pauses — 800ms sits above ordinary sentence pauses while
    # keeping total turn-close latency under the 1.5s product goal
    # (idle 800ms + VAD window). Bounded 100..1500.
    discord_input_idle_ms: int = 800
    # Worker performance knobs: SAME model/provider as the session (never a
    # quality downgrade) — speed comes from low reasoning effort, priority
    # service tier, and a bounded iteration budget. Scoped to realtime voice
    # workers only; normal text agents and global delegation are untouched.
    worker_reasoning_effort: str = "low"
    worker_service_tier: str = "priority"
    worker_max_iterations: int = 30
    # Exact sourced-synthesis TTS path (ADR D13).
    synthesis_tts_model: str = "gpt-4o-mini-tts"
    synthesis_voice: str = "cedar"
    synthesis_framing: bool = True


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _as_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


_VALID_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


def _validated_effort(value: Any, default: str) -> str:
    effort = _as_str(value, default).lower()
    return effort if effort in _VALID_EFFORTS else default


def load_realtime_voice_config(raw: Optional[Mapping[str, Any]]) -> RealtimeVoiceConfig:
    """Parse the raw ``...voice.realtime`` mapping with safe defaults."""
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raw = {}
    synthesis = raw.get("synthesis") or {}
    if not isinstance(synthesis, Mapping):
        synthesis = {}
    defaults = RealtimeVoiceConfig()
    return RealtimeVoiceConfig(
        enabled=_as_bool(raw.get("enabled"), defaults.enabled),
        model=_as_str(raw.get("model"), defaults.model),
        voice=_as_str(raw.get("voice"), defaults.voice),
        reasoning_effort=_as_str(raw.get("reasoning_effort"), defaults.reasoning_effort),
        api_key_secret=_as_str(raw.get("api_key_secret"), defaults.api_key_secret),
        connect_timeout_seconds=_as_float(
            raw.get("connect_timeout_seconds"), defaults.connect_timeout_seconds, 1.0
        ),
        reconnect_budget_seconds=_as_float(
            raw.get("reconnect_budget_seconds"), defaults.reconnect_budget_seconds, 0.0
        ),
        rollover_margin_seconds=_as_int(
            raw.get("rollover_margin_seconds"), defaults.rollover_margin_seconds, 30
        ),
        session_max_seconds=_as_int(
            raw.get("session_max_seconds"), defaults.session_max_seconds, 60
        ),
        max_inflight_dispatches=_as_int(
            raw.get("max_inflight_dispatches"), defaults.max_inflight_dispatches, 1
        ),
        transcript_window_turns=_as_int(
            raw.get("transcript_window_turns"), defaults.transcript_window_turns, 2
        ),
        demotion_buffer_seconds=_as_float(
            raw.get("demotion_buffer_seconds"), defaults.demotion_buffer_seconds, 1.0
        ),
        input_transcription_model=(
            raw.get("input_transcription_model").strip()
            if isinstance(raw.get("input_transcription_model"), str)
            else defaults.input_transcription_model
        ),
        turn_detection_type=_as_str(
            raw.get("turn_detection_type"), defaults.turn_detection_type
        ).lower(),
        server_vad_threshold=min(
            1.0,
            _as_float(raw.get("server_vad_threshold"), defaults.server_vad_threshold, 0.0),
        ),
        server_vad_prefix_padding_ms=min(
            5000,
            _as_int(
                raw.get("server_vad_prefix_padding_ms"),
                defaults.server_vad_prefix_padding_ms, 0,
            ),
        ),
        server_vad_silence_duration_ms=min(
            10000,
            _as_int(
                raw.get("server_vad_silence_duration_ms"),
                defaults.server_vad_silence_duration_ms, 50,
            ),
        ),
        discord_input_idle_ms=min(
            1500,
            _as_int(
                raw.get("discord_input_idle_ms"),
                defaults.discord_input_idle_ms, 100,
            ),
        ),
        worker_reasoning_effort=_validated_effort(
            raw.get("worker_reasoning_effort"), defaults.worker_reasoning_effort
        ),
        worker_service_tier=_as_str(
            raw.get("worker_service_tier"), defaults.worker_service_tier
        ).lower(),
        worker_max_iterations=min(
            500,
            _as_int(raw.get("worker_max_iterations"), defaults.worker_max_iterations, 1),
        ),
        synthesis_tts_model=_as_str(
            synthesis.get("tts_model"), defaults.synthesis_tts_model
        ),
        synthesis_voice=_as_str(synthesis.get("voice"), defaults.synthesis_voice),
        synthesis_framing=_as_bool(synthesis.get("framing"), defaults.synthesis_framing),
    )


def resolve_api_key(config: RealtimeVoiceConfig) -> Optional[str]:
    """Resolve the provider secret named by ``api_key_secret``.

    Scope-aware fail-closed read (same shape as the canonical
    ``_get_scoped_secret`` in the feishu adapter, #86905): under multiplexed
    profiles a scoped miss returns None — NEVER falls through to
    ``os.environ`` (that would leak another profile's key).  Only the
    unscoped single-profile path reads the process env.
    """
    name = config.api_key_secret
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            value = get_secret(name)
        except UnscopedSecretError:
            value = os.getenv(name)
    except Exception:
        value = os.getenv(name)
    value = (value or "").strip()
    return value or None
