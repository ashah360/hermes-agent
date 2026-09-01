"""Minimal deterministic Discord text outbox for worker results (ADR D9).

Workers never address the user directly — this is the single writer of
worker output to Discord text. Exactly-once per dispatch (idempotent by
dispatch_id), bound text channel only, deterministic rendering, called from
worker executor threads via ``run_coroutine_threadsafe`` onto the lane loop.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BODY_CAP = 3500  # stay under Discord's 4000-char message ceiling


def render_result(record: Any, result: dict) -> str:
    status = str(result.get("status") or "completed")
    body = (result.get("final_response") or result.get("error") or "").strip()
    if len(body) > _BODY_CAP:
        body = body[:_BODY_CAP] + "\n… (truncated)"
    header = f"**Voice worker result** (`{record.dispatch_id}`, {status})"
    return f"{header}\nTask: {record.goal}\n\n{body}".strip()


class DiscordTextOutbox:
    """Exactly-once poster of rendered worker results to the bound channel."""

    def __init__(self, *, adapter: Any, lane: Any) -> None:
        self._adapter = adapter
        self._lane = lane
        self._lock = threading.Lock()
        self._posted: Dict[str, str] = {}  # dispatch_id -> message_id

    def post_result(self, record: Any, result: dict) -> Optional[str]:
        """Post once; repeated calls return the original message id."""
        with self._lock:
            if record.dispatch_id in self._posted:
                return self._posted[record.dispatch_id]
        channel_id = getattr(record, "text_channel_id", None)
        adapter = self._adapter
        loop = getattr(self._lane, "_loop", None)
        if not channel_id or adapter is None or loop is None:
            return None
        import asyncio

        # Deadlock guard: run_coroutine_threadsafe(...).result() on the very
        # loop it targets blocks forever. post_result is an executor-thread
        # API; refuse (loudly) instead of hanging the event loop.
        try:
            if asyncio.get_running_loop() is loop:
                logger.warning(
                    "Realtime outbox post_result called on its own event "
                    "loop (dispatch=%s) — refusing to block; post skipped",
                    record.dispatch_id,
                )
                return None
        except RuntimeError:
            pass  # no running loop on this thread — the normal case
        try:
            future = asyncio.run_coroutine_threadsafe(
                adapter.send(str(channel_id), render_result(record, result)),
                loop,
            )
            send_result = future.result(timeout=20)
            message_id = str(getattr(send_result, "message_id", "") or "") or None
        except Exception:
            logger.warning(
                "Realtime outbox post failed (dispatch=%s)", record.dispatch_id,
                exc_info=True,
            )
            return None
        if message_id is None:
            return None
        with self._lock:
            # First writer wins; a concurrent duplicate returns the original.
            existing = self._posted.setdefault(record.dispatch_id, message_id)
        return existing
