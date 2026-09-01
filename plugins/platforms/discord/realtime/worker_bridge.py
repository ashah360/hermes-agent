"""Isolated per-dispatch Hermes child workers + typed event synthesis (ADR D7).

Each ``hermes_dispatch`` runs ONE isolated child worker session on a bounded
plugin-local executor — never a turn injected into the bound text-channel
session (which would run concurrent model loops over one history and break
role alternation and prompt caching).

The default child runner goes through the real delegation machinery
(``tools/delegate_tool._build_child_agent``): same credentials/model/toolsets
as the principal minus the child blocklist (no ``send_message``, no
``clarify`` — workers never address the user), the standard subagent
approval callback (auto-deny by default), ``load_soul_identity=True`` so the
worker carries the same SOUL.md Jeeves identity (the cron pattern), and
parent/child session lineage. Tests inject a fake runner.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .events import DispatchRecord, WorkerEvent

logger = logging.getLogger(__name__)

# Markdown links in the worker's final response become the event's sources.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# Exact spoken script marker the worker is instructed to emit (last line).
_SPOKEN_MARKER_RE = re.compile(r"(?im)^SPOKEN SUMMARY:\s*(.+)$")
# Tool-name shapes scrubbed from spoken hints: `backticked_names`, snake_case
# call syntax — the voice model must never narrate tool names (ADR D7).
_TOOL_NAME_RE = re.compile(r"`[a-z0-9_]+`|\b[a-z0-9]+(?:_[a-z0-9]+)+\b(?:\(\))?")

_WORKER_RESULT_CONTRACT = (
    "\n\nWhen you finish, end your reply with one line starting exactly with "
    "'SPOKEN SUMMARY:' — a one-or-two-sentence spoken-style synthesis of the "
    "result with the key figures, suitable to be read aloud verbatim. Cite "
    "sources as markdown links in the body."
)


def canonicalize_goal(goal: str) -> str:
    """Deterministic goal identity for live-dispatch dedupe.

    Casefold, strip punctuation, collapse whitespace — so "Check the
    weather!!" and "check   the weather" are one live task.
    """
    lowered = (goal or "").casefold()
    stripped = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


@dataclass(frozen=True)
class WorkerSpec:
    """Immutable dispatch-time snapshot handed to the child runner."""

    dispatch_id: str
    goal: str
    context_snapshot: str
    user_id: int
    guild_id: int
    text_channel_id: Optional[int]
    max_iterations: int = 30


class WorkerHooks:
    """Callbacks the child runner uses to surface interim material."""

    def __init__(self, bridge: "WorkerBridge", record: DispatchRecord) -> None:
        self._bridge = bridge
        self._record = record

    def finding(self, text: str) -> None:
        self._bridge._emit(self._record, "finding", text)

    def blocker(self, text: str) -> None:
        self._bridge.lane.registry.mark_blocked(self._record.dispatch_id)
        self._bridge._emit(self._record, "blocker", text)

    def interrupt_fn(self, fn: Callable[[], None]) -> None:
        self._record.interrupt_fn = fn


def _scrub_spoken(text: str) -> str:
    """Remove tool-name shapes and tighten whitespace for spoken hints."""
    text = _TOOL_NAME_RE.sub("", text or "")
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _extract_sources(final_response: str, *, fetched_at: str) -> tuple:
    out: List[dict] = []
    seen = set()
    for title, url in _MD_LINK_RE.findall(final_response or ""):
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": title.strip(), "url": url, "fetched_at": fetched_at})
    return tuple(out)


def _extract_spoken_synthesis(final_response: str) -> Optional[str]:
    match = _SPOKEN_MARKER_RE.search(final_response or "")
    if not match:
        return None
    script = match.group(1).strip()
    if not script:
        return None
    try:
        from tools.tts_text_normalize import prepare_spoken_text

        return prepare_spoken_text(script)
    except Exception:
        return script


def build_worker_child(
    *,
    goal: str,
    context_snapshot: str,
    principal: Any,
    max_iterations: int = 120,
):
    """Construct an isolated child worker through the real delegation path.

    ``_build_child_agent`` gives: principal credentials/model, toolsets minus
    the child blocklist (no send_message/clarify/delegation/memory/cronjob),
    subagent approval callback (``delegation.subagent_auto_approve``,
    auto-deny default), quiet mode, and parent/child session lineage.

    One deliberate divergence from the delegate-leaf default: the worker
    carries the SAME Jeeves identity as the full agent, so
    ``load_soul_identity`` is enabled before the first prompt build (the flag
    cron uses for exactly this need; the prompt is built lazily at run start,
    so this is a pre-run construction step, not a mid-session mutation).
    """
    from tools.delegate_tool import _build_child_agent

    child = _build_child_agent(
        task_index=0,
        goal=goal,
        context=context_snapshot,
        toolsets=None,          # inherit principal's toolsets (minus blocklist)
        model=None,             # inherit principal's model routing
        max_iterations=max_iterations,
        task_count=1,
        parent_agent=principal,
    )
    child.load_soul_identity = True
    return child


def default_child_runner(spec: WorkerSpec, hooks: WorkerHooks, *, principal: Any = None) -> Dict[str, Any]:
    """Run one isolated worker to completion (executor thread).

    Registered in the delegation ``_active_subagents`` registry so the same
    interrupt seam ``/stop`` uses covers `hermes_cancel`; interim
    model-authored content flows through ``interim_assistant_callback`` into
    ``finding`` events (never raw tool starts, never a timer).
    """
    from tools.delegate_tool import _register_subagent, _unregister_subagent

    child = build_worker_child(
        goal=spec.goal + _WORKER_RESULT_CONTRACT,
        context_snapshot=spec.context_snapshot,
        principal=principal,
        max_iterations=getattr(spec, "max_iterations", 30),
    )
    hooks.interrupt_fn(lambda: child.interrupt(reason="realtime lane cancel"))

    def _on_interim(content: str) -> None:
        content = (content or "").strip()
        if content:
            hooks.finding(content)

    try:
        child.interim_assistant_callback = _on_interim
    except Exception:
        pass

    subagent_id = getattr(child, "_subagent_id", None) or f"rt-{spec.dispatch_id}"
    _register_subagent({"subagent_id": subagent_id, "agent": child, "goal": spec.goal})
    try:
        final = child.chat(spec.goal + _WORKER_RESULT_CONTRACT)
        interrupted = bool(getattr(child, "is_interrupted", False))
        status = "cancelled" if interrupted else "completed"
        return {"status": status, "final_response": final or ""}
    except Exception as exc:  # noqa: BLE001 — reported as a failed event
        logger.warning("Realtime worker %s crashed: %s", spec.dispatch_id, exc)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        _unregister_subagent(subagent_id, agent=child)
        close = getattr(child, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class WorkerBridge:
    """Dispatches isolated workers and reduces their lifecycle to typed events."""

    def __init__(
        self,
        *,
        lane: Any,
        adapter: Any = None,
        child_runner: Optional[Callable[..., Dict[str, Any]]] = None,
        outbox: Any = None,
        principal_provider: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.lane = lane
        self.adapter = adapter
        self.outbox = outbox
        self._principal_provider = principal_provider
        self._child_runner = child_runner or self._run_default_child
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, lane.config.max_inflight_dispatches),
            thread_name_prefix=f"rt-worker-{lane.guild_id}",
        )
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Dispatch / cancel
    # ------------------------------------------------------------------

    def dispatch(
        self,
        *,
        task: str,
        user_id: int,
        spoken_ack_hint: Optional[str] = None,
        supersedes_dispatch_id: Optional[str] = None,
    ) -> Optional[DispatchRecord]:
        """Register + spawn one isolated worker. None → capacity rejection."""
        registry = self.lane.registry
        if supersedes_dispatch_id:
            registry.revoke(supersedes_dispatch_id, reason="superseded")
            self.lane.telemetry.incr("dispatch_superseded")

        # Atomic dedupe + capacity + register: an equivalent RUNNING/BLOCKED
        # goal is reused instead of spawning a second worker (live evidence:
        # two identical weather dispatches 35s apart both ran and both spoke).
        record, reused = registry.register_or_reuse(
            goal=task,
            canonical_goal=canonicalize_goal(task),
            guild_id=self.lane.guild_id,
            text_channel_id=self.lane.text_channel_id,
            user_id=user_id,
            max_live=self.lane.config.max_inflight_dispatches,
        )
        if record is None:
            self.lane.telemetry.incr("dispatch_rejected_capacity")
            return None
        if reused:
            self.lane.telemetry.incr("dispatch_deduped")
            logger.info(
                "discord_realtime dispatch_deduped guild=%s dispatch=%s user=%s",
                self.lane.guild_id, record.dispatch_id, user_id,
            )
            return record
        spec = WorkerSpec(
            dispatch_id=record.dispatch_id,
            goal=task,
            context_snapshot=self._build_context_snapshot(task, user_id),
            user_id=user_id,
            guild_id=self.lane.guild_id,
            text_channel_id=self.lane.text_channel_id,
            max_iterations=self.lane.config.worker_max_iterations,
        )
        hooks = WorkerHooks(self, record)
        logger.info(
            "discord_realtime dispatch guild=%s dispatch=%s user=%s goal_len=%d live=%d",
            self.lane.guild_id, record.dispatch_id, user_id, len(task),
            registry.live_count(),
        )
        self._emit(record, "started", spoken_ack_hint or f"Working on: {task}")
        future = self._executor.submit(self._run_worker, spec, hooks, record)
        with self._lock:
            self._futures[record.dispatch_id] = future
        self.lane.telemetry.incr("dispatches")
        return record

    def cancel(self, dispatch_id: str, *, reason: str) -> bool:
        """Explicit semantic cancellation: revoke + interrupt the child."""
        ok = self.lane.registry.revoke(dispatch_id, reason=reason)
        logger.info(
            "discord_realtime dispatch_cancel guild=%s dispatch=%s ok=%s",
            self.lane.guild_id, dispatch_id, ok,
        )
        if ok:
            self.lane.telemetry.incr("dispatch_cancelled")
        return ok

    def drain(self, timeout: float = 10.0) -> None:
        """Test/teardown helper: wait for all spawned workers to finish."""
        with self._lock:
            futures = list(self._futures.values())
        deadline = time.monotonic() + timeout
        for future in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)

    # ------------------------------------------------------------------
    # Worker execution + event synthesis
    # ------------------------------------------------------------------

    def _run_default_child(self, spec: WorkerSpec, hooks: WorkerHooks) -> Dict[str, Any]:
        principal = self._principal_provider() if self._principal_provider else None
        if principal is None:
            return {
                "status": "failed",
                "error": "no worker principal available for this session",
            }
        return default_child_runner(spec, hooks, principal=principal)

    def _run_worker(self, spec: WorkerSpec, hooks: WorkerHooks, record: DispatchRecord) -> None:
        try:
            result = self._child_runner(spec, hooks) or {}
        except Exception as exc:  # noqa: BLE001 — must never kill the executor
            logger.warning("Realtime worker runner crashed: %s", exc, exc_info=True)
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        status = result.get("status") or "completed"
        if status not in ("completed", "failed", "cancelled"):
            status = "completed" if not result.get("error") else "failed"
        self.lane.registry.mark_terminal(spec.dispatch_id, status)
        logger.info(
            "discord_realtime dispatch_terminal guild=%s dispatch=%s status=%s",
            self.lane.guild_id, spec.dispatch_id, status,
        )

        final_response = str(result.get("final_response") or "")
        detail_ref = None
        if self.outbox is not None and status in ("completed", "failed"):
            try:
                detail_ref = self.outbox.post_result(record, result)
            except Exception:
                logger.warning("Realtime outbox post failed", exc_info=True)

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        sources = _extract_sources(final_response, fetched_at=fetched_at)
        try:
            self.lane.store_result_payload(spec.dispatch_id, {
                "final_response": final_response,
                "sources": list(sources),
                "fetched_at": fetched_at,
                "status": status,
            })
        except Exception:
            pass
        spoken_synthesis = (
            _extract_spoken_synthesis(final_response) if status == "completed" else None
        )
        if status == "completed":
            hint = "The result is ready."
        elif status == "cancelled":
            hint = "That work was cancelled."
        else:
            hint = f"The work failed: {result.get('error') or 'unknown error'}"
        self._emit(
            record,
            status,
            hint,
            sources=sources,
            spoken_synthesis=spoken_synthesis,
            detail_ref=detail_ref,
        )

    def _emit(
        self,
        record: DispatchRecord,
        event_type: str,
        spoken_hint: str,
        *,
        sources: tuple = (),
        spoken_synthesis: Optional[str] = None,
        detail_ref: Optional[str] = None,
    ) -> None:
        event = WorkerEvent(
            dispatch_id=record.dispatch_id,
            epoch=record.epoch,
            type=event_type,  # type: ignore[arg-type]
            spoken_hint=_scrub_spoken(spoken_hint),
            detail_ref=detail_ref,
            sources=sources,
            unsourced=not sources,
            ts=time.monotonic(),
            spoken_synthesis=spoken_synthesis,
        )
        try:
            self.lane.deliver_worker_event_threadsafe(event)
        except Exception:
            logger.debug("worker event delivery failed", exc_info=True)

    # ------------------------------------------------------------------
    # Context snapshot (immutable at dispatch — ADR D7)
    # ------------------------------------------------------------------

    def _build_context_snapshot(self, task: str, user_id: int) -> str:
        lines = [
            "You are running as a background worker for Jeeves's live Discord "
            "voice session. You never address the user directly; your result "
            "is spoken for you and your details are posted to the text channel.",
            f"Discord guild {self.lane.guild_id}, bound text channel "
            f"{self.lane.text_channel_id}, requesting user {user_id}.",
            f"Verbatim request: {task}",
        ]
        transcript = list(getattr(self.lane, "transcript_window", lambda: [])())
        if transcript:
            lines.append("Recent voice conversation:")
            lines.extend(f"  {role}: {text}" for role, text in transcript[-10:])
        return "\n".join(lines)
