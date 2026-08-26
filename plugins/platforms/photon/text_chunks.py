"""Deterministic conversational text chunking for Photon/iMessage."""

from __future__ import annotations

from gateway.platforms.helpers import split_text_fence_aware


DEFAULT_SOFT_CHARS = 1200
DEFAULT_HARD_CHARS = 2000


def _bounded_int(value, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, parsed))


def chunk_photon_text(
    text: str,
    *,
    soft_chars: int = DEFAULT_SOFT_CHARS,
    hard_chars: int = DEFAULT_HARD_CHARS,
) -> list[str]:
    """Split one logical answer into ordered, independently renderable bubbles."""
    soft = _bounded_int(soft_chars, DEFAULT_SOFT_CHARS, minimum=200, maximum=8000)
    hard = _bounded_int(hard_chars, DEFAULT_HARD_CHARS, minimum=soft, maximum=8000)
    if not text or len(text) <= soft:
        return [text] if text else []

    semantic = split_text_fence_aware(
        text,
        soft,
        prefer_paragraphs=True,
        balance_fences=False,
    )
    chunks: list[str] = []
    for chunk in semantic:
        if len(chunk) <= hard:
            chunks.append(chunk)
            continue
        chunks.extend(
            split_text_fence_aware(
                chunk,
                hard,
                prefer_paragraphs=False,
                balance_fences=True,
            )
        )
    return [chunk for chunk in chunks if chunk]
