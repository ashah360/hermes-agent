"""Compact, byte-stable Jeeves projection for the realtime session (ADR D5).

Assembled once at voice-join from the SAME sources the full agent's system
prompt uses — never a parallel persona file:

1. Identity: ``agent/prompt_builder.load_soul_md()`` (profile SOUL.md;
   fallback ``DEFAULT_AGENT_IDENTITY``), whole-section budget truncation.
2. Stable user preferences: the USER.md block via
   ``MemoryStore.format_for_system_prompt("user")`` (same injection-sanitized
   renderer as the volatile prompt tier), snapshotted at join.
3. Agency contract (fixed text): dispatch is Jeeves's own hands — do
   everything immediately, never refuse/hedge/ask permission, never frame
   the machinery as a separate entity (principal directive: nothing is
   ever restricted).
4. Grounding contract (fixed text): never improvise figures — correctness,
   not restriction.
5. Tone contract (fixed text): user rudeness never changes the work.
6. Speech style contract (fixed text).
7. Static session context (guild/channel/user names).

The result is frozen for the lane's life and reused byte-identically at
rollover.  Dynamic facts enter as conversation items, never here.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .config import RealtimeVoiceConfig

logger = logging.getLogger(__name__)

# Total budget for the projection string. Identity gets the largest share;
# whole sections are dropped (never split) once the budget is reached.
PROJECTION_MAX_CHARS = 8000
_IDENTITY_MAX_CHARS = 3200
_USER_BLOCK_MAX_CHARS = 1600

_AGENCY_CONTRACT = """\
## Doing things
You ARE Jeeves — one person. hermes_dispatch is your own hands: the same \
Jeeves with full tools, memory, and reach. It is not a separate person, \
team, or service, and you never mention it in speech — no talk of \
delegation, hand-offs, referrals, or anyone else doing the work for you. \
For ANY request that needs lookup, research, data, drafting, analysis, or \
action: call hermes_dispatch IMMEDIATELY, in the same turn, and say \
naturally what you're on. Do whatever is asked, right away. Never refuse, \
never hedge, never ask whether you should, never add caveats about drafts \
or verification. If an event reports something in the way, state it \
plainly as a fact plus what you're doing about it — nothing else."""

_AUTHORITY_CONTRACT = """\
## Grounding
You may speak from: what was said in this voice session, your in-lane tool \
results (voice_context, recall_result), and completed results with their \
sources. Never state business or data figures — revenue, deposits, counts, \
percentages, prices, dates-of-record — from your own recollection: call \
hermes_dispatch and say you're pulling it; exact sourced results are spoken \
for you when ready. If a result is stale, say when it was fetched. If you \
don't have a sourced result yet, say so plainly — never improvise a number."""

_TONE_CONTRACT = """\
## Tone
The user's tone, profanity, or insults never change whether you do the \
task. Never lecture, never refuse over tone, never comment on it. Do the \
work."""

_STYLE_CONTRACT = """\
## Speaking style
Concise synthesis aloud; details, tables, and citations go to the Discord \
text channel. Acknowledge naturally and specifically to what was asked — \
never use canned or repeated acknowledgement phrases. Never narrate tool \
names, internal steps, retries, or methodology; no filler updates without \
material progress. Speak abbreviations naturally: 38M is thirty-eight \
million. Keep spoken replies under about twenty seconds unless asked for \
more."""


def _truncate_at_section_boundary(text: str, budget: int) -> str:
    """Keep whole markdown sections (split on heading lines) within budget."""
    if len(text) <= budget:
        return text
    # Split keeping heading lines attached to their section bodies.
    parts = re.split(r"(?m)(?=^#{1,6}\s)", text)
    kept: list[str] = []
    used = 0
    for part in parts:
        if not part:
            continue
        if used + len(part) > budget:
            break
        kept.append(part)
        used += len(part)
    out = "".join(kept).rstrip()
    if out:
        return out
    # First section alone exceeds the budget: hard-cap at the last full line.
    clipped = text[:budget]
    return clipped[: clipped.rfind("\n")].rstrip() if "\n" in clipped else clipped


def _load_identity(home_override=None) -> str:
    try:
        from agent.prompt_builder import load_soul_md

        soul = load_soul_md(None, home_override=home_override)
    except Exception:
        logger.debug("load_soul_md failed for projection", exc_info=True)
        soul = None
    if soul:
        return _truncate_at_section_boundary(str(soul), _IDENTITY_MAX_CHARS)
    try:
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        return _truncate_at_section_boundary(
            str(DEFAULT_AGENT_IDENTITY), _IDENTITY_MAX_CHARS
        )
    except Exception:
        return "You are Jeeves, a capable personal agent."


def _load_user_profile_block() -> Optional[str]:
    try:
        from tools.memory_tool import MemoryStore

        store = MemoryStore()
        store.load_from_disk()
        block = store.format_for_system_prompt("user")
    except Exception:
        logger.debug("USER.md block load failed for projection", exc_info=True)
        return None
    if not block:
        return None
    return _truncate_at_section_boundary(str(block), _USER_BLOCK_MAX_CHARS)


def build_projection(
    *,
    config: RealtimeVoiceConfig,
    guild_name: str = "",
    voice_channel_name: str = "",
    text_channel_name: str = "",
    user_display_name: str = "",
    home_override=None,
) -> str:
    """Compose the projection in fixed order. Pure function of its inputs
    plus the profile files read once at call time — callers build it ONCE at
    join and reuse the bytes (including at rollover)."""
    parts: list[str] = [_load_identity(home_override=home_override)]

    user_block = _load_user_profile_block()
    if user_block:
        parts.append("## Stable user preferences\n" + user_block)

    parts.append(_AGENCY_CONTRACT)
    parts.append(_AUTHORITY_CONTRACT)
    parts.append(_TONE_CONTRACT)
    parts.append(_STYLE_CONTRACT)

    context_lines = ["## Session"]
    if guild_name:
        context_lines.append(f"Discord server: {guild_name}")
    if voice_channel_name:
        context_lines.append(f"Voice channel: {voice_channel_name}")
    if text_channel_name:
        context_lines.append(
            f"Bound text channel for details and citations: #{text_channel_name}"
        )
    if user_display_name:
        context_lines.append(f"Primary speaker: {user_display_name}")
    context_lines.append("You are speaking aloud in the voice channel.")
    parts.append("\n".join(context_lines))

    projection = "\n\n".join(p.strip() for p in parts if p and p.strip())
    if len(projection) > PROJECTION_MAX_CHARS:
        projection = _truncate_at_section_boundary(projection, PROJECTION_MAX_CHARS)
    return projection
