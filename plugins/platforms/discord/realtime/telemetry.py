"""Privacy-safe telemetry for the realtime voice lane.

In-memory counters and duration histograms per lane, flushed as structured
single-line log records (agent.log, INFO) periodically and at teardown.
Payloads carry names, counts, and millisecond durations only — never raw
audio, transcripts, or secrets.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from typing import Dict, List

logger = logging.getLogger(__name__)

_FLUSH_INTERVAL_SECONDS = 60.0


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


class LaneTelemetry:
    """Counters + histograms for one lane. Thread-safe; audio/transcript-free."""

    def __init__(self, guild_id: int, *, clock=time.monotonic):
        self.guild_id = guild_id
        self._clock = clock
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = {}
        self._durations: Dict[str, List[float]] = {}
        self._gauges: Dict[str, float] = {}
        self._last_flush = clock()

    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + amount

    def record_ms(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._durations.setdefault(name, []).append(float(value_ms))

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters.get(name, 0)

    def snapshot(self) -> Dict:
        with self._lock:
            out: Dict = {"guild_id": self.guild_id, "counters": dict(self._counters)}
            hists = {}
            for name, values in self._durations.items():
                ordered = sorted(values)
                hists[name] = {
                    "count": len(ordered),
                    "p50_ms": round(_percentile(ordered, 0.50), 1),
                    "p95_ms": round(_percentile(ordered, 0.95), 1),
                    "max_ms": round(ordered[-1], 1) if ordered else 0.0,
                }
            out["histograms"] = hists
            out["gauges"] = dict(self._gauges)
            return out

    def maybe_flush(self, *, force: bool = False) -> None:
        now = self._clock()
        with self._lock:
            due = force or (now - self._last_flush) >= _FLUSH_INTERVAL_SECONDS
            if not due:
                return
            self._last_flush = now
        try:
            logger.info("discord_realtime_telemetry %s", json.dumps(self.snapshot(), sort_keys=True))
        except Exception:
            pass
