"""Typed worker lifecycle events + dispatch ownership registry (ADR D7/D8)."""

from __future__ import annotations

import time
from typing import Callable, Dict


class DispatchRegistry:
    """Durable dispatch ownership, separate from audio-plane turn sequencing.

    Records are revoked ONLY by explicit semantic actions (cancel,
    supersession, reset, leave) — never by barge-in or follow-up speech.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._records: Dict[str, dict] = {}

    def revoke_all(self, *, reason: str) -> None:
        for record in self._records.values():
            if record.get("status") in ("running", "blocked"):
                record["status"] = "cancelled"
                record["revoke_reason"] = reason
                record["voice_revoked"] = True
