"""In-lane tools exposed to the realtime model (ADR D6).

These schemas live ONLY in the provider session config — they are never
registered in ``tools/registry.py`` and never appear in any core toolset, so
no other Hermes surface pays their footprint.

Two classes:
- fast in-lane reads (``voice_context``, ``recall_result``) answered by the
  plugin in-process; ``recall_result`` reads the dispatch registry ONLY and
  cannot fabricate;
- asynchronous worker dispatch (``hermes_dispatch``, ``hermes_cancel``).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def lane_tool_schemas() -> list:
    return [
        {
            "type": "function",
            "name": "voice_context",
            "description": (
                "Who is in the voice channel, the bound text channel, and the "
                "current time."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "type": "function",
            "name": "recall_result",
            "description": (
                "Look up a completed background-worker result (with sources and "
                "freshness). Returns no_result if there is none — results can "
                "only come from workers, never from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dispatch_id": {
                        "type": "string",
                        "description": "Specific dispatch to recall; omit for the most recent.",
                    }
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "hermes_dispatch",
            "description": (
                "Start a background Hermes worker for anything needing research, "
                "tools, or authoritative business/data figures. Returns "
                "immediately; progress and the result are delivered to you as "
                "events. Acknowledge naturally in your own words."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The full task, self-contained."},
                    "spoken_ack_hint": {
                        "type": "string",
                        "description": "Optional short phrase describing what you're checking.",
                    },
                    "supersedes_dispatch_id": {
                        "type": "string",
                        "description": "Dispatch this task replaces (user corrected/redirected).",
                    },
                },
                "required": ["task"],
            },
        },
        {
            "type": "function",
            "name": "hermes_cancel",
            "description": "Cancel a background worker that is no longer wanted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dispatch_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["dispatch_id"],
            },
        },
    ]


def handle_lane_tool(lane: Any, name: str, args: dict) -> str:
    """Execute one in-lane tool call. Always returns a JSON string."""
    args = args or {}
    try:
        if name == "voice_context":
            return json.dumps(_voice_context(lane))
        if name == "recall_result":
            return json.dumps(_recall_result(lane, args.get("dispatch_id")))
        if name == "hermes_dispatch":
            return json.dumps(_dispatch(lane, args))
        if name == "hermes_cancel":
            return json.dumps(_cancel(lane, args))
        return json.dumps({"status": "error", "error": f"unknown tool {name}"})
    except Exception as exc:  # noqa: BLE001 — tool results must not raise
        logger.warning("lane tool %s failed: %s", name, exc, exc_info=True)
        return json.dumps({"status": "error", "error": str(exc)})


def _voice_context(lane: Any) -> dict:
    out: dict = {
        "guild_id": lane.guild_id,
        "text_channel_id": lane.text_channel_id,
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    adapter = getattr(lane, "_adapter", None)
    if adapter is not None and hasattr(adapter, "get_voice_channel_info"):
        try:
            info = adapter.get_voice_channel_info(lane.guild_id)
            if info:
                out["channel_name"] = info.get("channel_name")
                out["members"] = [
                    {"name": m.get("display_name"), "speaking": m.get("is_speaking")}
                    for m in info.get("members", [])
                ]
        except Exception:
            pass
    return out


def _recall_result(lane: Any, dispatch_id: Any) -> dict:
    completed = lane.registry.completed_records()
    record = None
    if dispatch_id:
        record = lane.registry.get(str(dispatch_id))
        if record is None or record.status != "completed":
            record = None
    elif completed:
        record = max(completed, key=lambda r: r.created_at)
    if record is None:
        return {
            "status": "no_result",
            "note": (
                "No completed worker result matches. Results only exist after "
                "a hermes_dispatch completes — do not answer from memory."
            ),
        }
    payload = getattr(lane, "result_payload_for", lambda _rid: None)(record.dispatch_id)
    return {
        "status": "completed",
        "dispatch_id": record.dispatch_id,
        "task": record.goal,
        "result": (payload or {}).get("final_response", ""),
        "sources": (payload or {}).get("sources", []),
        "fetched_at": (payload or {}).get("fetched_at"),
    }


def _dispatch(lane: Any, args: dict) -> dict:
    task = str(args.get("task") or "").strip()
    if not task:
        return {"status": "error", "error": "task is required"}
    bridge = lane.worker_bridge
    if bridge is None:
        return {"status": "error", "error": "worker dispatch unavailable"}
    record = bridge.dispatch(
        task=task,
        user_id=getattr(lane, "last_speaker_user_id", 0) or 0,
        spoken_ack_hint=args.get("spoken_ack_hint"),
        supersedes_dispatch_id=args.get("supersedes_dispatch_id"),
    )
    if record is None:
        return {
            "status": "rejected",
            "error": (
                "Too many background tasks are already running. Tell the user "
                "and offer to queue or replace one."
            ),
        }
    return {"status": "dispatched", "dispatch_id": record.dispatch_id}


def _cancel(lane: Any, args: dict) -> dict:
    dispatch_id = str(args.get("dispatch_id") or "").strip()
    if not dispatch_id:
        return {"status": "error", "error": "dispatch_id is required"}
    bridge = lane.worker_bridge
    ok = bool(bridge and bridge.cancel(dispatch_id, reason=str(args.get("reason") or "cancelled")))
    return {"status": "cancelled" if ok else "not_found", "dispatch_id": dispatch_id}
