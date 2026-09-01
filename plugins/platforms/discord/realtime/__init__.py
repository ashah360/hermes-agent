"""Discord realtime voice lane (gpt-realtime-2.1) — feature-flagged, default off.

This package is imported ONLY after the adapter has confirmed
``gateway.platforms.discord.voice.realtime.enabled`` is true (lazy gate in
``DiscordAdapter._start_realtime_lane_if_enabled``).  Module-level imports here
must stay stdlib-only; optional deps (websockets, numpy, aiohttp) are imported
lazily inside the modules that need them.

Architecture: docs/discord-realtime-voice-architecture.md.
"""

from .config import RealtimeVoiceConfig, load_realtime_voice_config
from .lane import LaneState, RealtimeVoiceLane, create_lane

__all__ = [
    "LaneState",
    "RealtimeVoiceConfig",
    "RealtimeVoiceLane",
    "create_lane",
    "load_realtime_voice_config",
]
