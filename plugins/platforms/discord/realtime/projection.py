"""Compact, byte-stable Jeeves projection for the realtime session (ADR D5).

Assembled once at voice-join from the SAME sources the full agent's system
prompt uses — never a parallel persona file:

1. Identity: ``agent/prompt_builder.load_soul_md()`` (profile SOUL.md;
   fallback ``DEFAULT_AGENT_IDENTITY``), whole-section budget truncation.
2. Stable user preferences: the USER.md block via
   ``MemoryStore.format_for_system_prompt("user")`` (same injection-sanitized
   renderer as the volatile prompt tier), snapshotted at join.
3. Authority contract (fixed text — backstopped mechanically by the exact
   sourced-synthesis path and authority gate, ADR D13).
4. Approval boundary statement derived from config at join.
5. Speech style contract (fixed text).
6. Static session context (guild/channel/user names).

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

_AUTHORITY_CONTRACT = """\
## Authority
You may speak from: what was said in this voice session, your in-lane tool \
results (voice_context, recall_result), and completed worker results with \
their sources. You must NEVER state business or data figures — revenue, \
deposits, counts, percentages, prices, dates-of-record — from your own \
recollection. For any such answer, call hermes_dispatch and tell the user \
you are checking; exact sourced results are spoken for you when ready. If a \
result is stale, say when it was fetched. If you don't have a sourced \
result, say so plainly — never improvise a number."""

_STYLE_CONTRACT = """\
## Speaking style
Concise synthesis aloud; details, tables, and citations go to the Discord \
text channel. Acknowledge naturally and specifically to what was asked — \
never use canned or repeated acknowledgement phrases. Never narrate tool \
names, internal steps, retries, or methodology; no filler updates without \
material progress. Speak abbreviations naturally: 38M is thirty-eight \
million. Keep spoken replies under about twenty seconds unless asked for \
more."""

_APPROVAL_CONTRACT_AUTO_DENY = """\
## Approvals
Background workers cannot approve dangerous or destructive commands; those \
are auto-denied and reported back as blockers. If work is blocked on an \
approval, say so and route it to the text channel — never claim you can \
grant it by voice."""

_APPROVAL_CONTRACT_AUTO_APPROVE = """\
## Approvals
Background workers run with pre-approved command execution per this \
deployment's configuration. Report anything destructive they did as part of \
the result."""


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


def _approval_contract() -> str:
    auto_approve = False
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
        auto_approve = bool((cfg.get("delegation") or {}).get("subagent_auto_approve"))
    except Exception:
        auto_approve = False
    return _APPROVAL_CONTRACT_AUTO_APPROVE if auto_approve else _APPROVAL_CONTRACT_AUTO_DENY


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

    parts.append(_AUTHORITY_CONTRACT)
    parts.append(_approval_contract())
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
