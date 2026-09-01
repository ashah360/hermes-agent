"""Typed worker lifecycle events + dispatch ownership registry (ADR D7/D8).

Two separate mechanisms, never conflated:
- the lane's ``turn_seq`` governs the AUDIO plane (barge-in, stale audio);
- ``DispatchRegistry`` governs WORKER ownership. Records are revoked ONLY by
  explicit semantic actions (cancel, supersession, reset, leave) — never by
  barge-in or follow-up speech.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple

EventType = Literal["started", "finding", "blocker", "completed", "failed", "cancelled"]

# Terminal event types are delivered exactly once per dispatch.
TERMINAL_EVENT_TYPES = ("completed", "failed", "cancelled")


@dataclass(frozen=True)
class WorkerEvent:
    dispatch_id: str
    epoch: int
    type: EventType
    spoken_hint: str                 # concise, user-facing; never CoT/tool names
    detail_ref: Optional[str]        # outbox message id of the detailed post
    sources: Tuple[dict, ...]        # ({title, url, fetched_at}, ...)
    unsourced: bool
    ts: float
    spoken_synthesis: Optional[str] = None  # exact sourced script (ADR D13)


@dataclass
class DispatchRecord:
    dispatch_id: str
    epoch: int
    goal: str
    guild_id: int
    text_channel_id: Optional[int]
    user_id: int
    status: str = "running"          # running|blocked|completed|failed|cancelled
    voice_revoked: bool = False
    revoke_reason: Optional[str] = None
    subagent_id: Optional[str] = None
    created_at: float = 0.0
    interrupt_fn: Optional[Callable[[], None]] = None
    delivered_terminal: bool = False
    delivered_keys: set = field(default_factory=set)
    canonical_goal: str = ""
    # True once at least one duplicate dispatch call was answered with this
    # record instead of spawning a second worker (live-goal dedupe).
    reused: bool = False


class DispatchRegistry:
    """Thread-safe dispatch ownership + exactly-once delivery claims."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._records: Dict[str, DispatchRecord] = {}
        self.epoch = 0

    # ── Registration / lookup ─────────────────────────────────────────

    def register(
        self,
        *,
        goal: str,
        guild_id: int,
        text_channel_id: Optional[int],
        user_id: int,
        canonical_goal: str = "",
    ) -> DispatchRecord:
        record = DispatchRecord(
            dispatch_id=f"disp-{uuid.uuid4().hex[:10]}",
            epoch=self.epoch,
            goal=goal,
            guild_id=guild_id,
            text_channel_id=text_channel_id,
            user_id=user_id,
            created_at=self._clock(),
            canonical_goal=canonical_goal,
        )
        with self._lock:
            self._records[record.dispatch_id] = record
        return record

    def register_or_reuse(
        self,
        *,
        goal: str,
        canonical_goal: str,
        guild_id: int,
        text_channel_id: Optional[int],
        user_id: int,
        max_live: int,
    ) -> Tuple[Optional[DispatchRecord], bool]:
        """Atomic live-goal dedupe + capacity check + registration.

        One lock hold covers the equivalent-live-record scan, the capacity
        check, and the insert, so two simultaneous identical dispatches can
        never both create a worker. Returns ``(record, reused)``;
        ``(None, False)`` means capacity rejection.
        """
        with self._lock:
            if canonical_goal:
                for existing in self._records.values():
                    if (
                        existing.status in ("running", "blocked")
                        and not existing.voice_revoked
                        and existing.canonical_goal == canonical_goal
                    ):
                        existing.reused = True
                        return existing, True
            live = sum(
                1 for r in self._records.values()
                if r.status in ("running", "blocked") and not r.voice_revoked
            )
            if live >= max_live:
                return None, False
            record = DispatchRecord(
                dispatch_id=f"disp-{uuid.uuid4().hex[:10]}",
                epoch=self.epoch,
                goal=goal,
                guild_id=guild_id,
                text_channel_id=text_channel_id,
                user_id=user_id,
                created_at=self._clock(),
                canonical_goal=canonical_goal,
            )
            self._records[record.dispatch_id] = record
            return record, False

    def get(self, dispatch_id: str) -> Optional[DispatchRecord]:
        with self._lock:
            return self._records.get(dispatch_id)

    def live_records(self) -> list:
        with self._lock:
            return [
                r for r in self._records.values()
                if r.status in ("running", "blocked") and not r.voice_revoked
            ]

    def completed_records(self) -> list:
        with self._lock:
            return [r for r in self._records.values() if r.status == "completed"]

    def live_count(self) -> int:
        return len(self.live_records())

    # ── Explicit semantic revocation (the ONLY revocation paths) ─────

    def revoke(self, dispatch_id: str, *, reason: str) -> bool:
        with self._lock:
            record = self._records.get(dispatch_id)
            if record is None:
                return False
            if record.status in ("running", "blocked"):
                record.status = "cancelled"
            record.voice_revoked = True
            record.revoke_reason = reason
            interrupt = record.interrupt_fn
        if interrupt is not None:
            try:
                interrupt()
            except Exception:
                pass
        return True

    def revoke_all(self, *, reason: str) -> None:
        """Reset/leave: revoke voice delivery for every dispatch.

        Deliberately does NOT interrupt running workers — their detailed
        results still post to the text channel through the outbox.
        """
        with self._lock:
            self.epoch += 1
            records = list(self._records.values())
        for record in records:
            record.voice_revoked = True
            if record.revoke_reason is None:
                record.revoke_reason = reason

    # ── Completion + exactly-once delivery claims ─────────────────────

    def mark_terminal(self, dispatch_id: str, status: str) -> None:
        with self._lock:
            record = self._records.get(dispatch_id)
            if record is not None and record.status in ("running", "blocked"):
                record.status = status

    def mark_blocked(self, dispatch_id: str) -> None:
        with self._lock:
            record = self._records.get(dispatch_id)
            if record is not None and record.status == "running":
                record.status = "blocked"

    def claim_delivery(self, event: WorkerEvent) -> bool:
        """Atomically decide whether *event* may be voiced. False → drop.

        Rules (ADR D8): the record must exist and not be voice-revoked;
        terminal events deliver at most once; duplicate non-terminal events
        (same type + hint) deliver once.
        """
        with self._lock:
            record = self._records.get(event.dispatch_id)
            if record is None or record.voice_revoked:
                return False
            if event.type in TERMINAL_EVENT_TYPES:
                if record.delivered_terminal:
                    return False
                record.delivered_terminal = True
                return True
            key = (event.type, event.spoken_hint)
            if key in record.delivered_keys:
                return False
            record.delivered_keys.add(key)
            return True
