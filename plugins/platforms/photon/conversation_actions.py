"""Current-conversation-only native actions for Photon sessions."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from gateway.session_context import get_session_env


TOOL_NAME = "photon_conversation_action"


def conversation_actions_configured() -> bool:
    """Return the profile-level rollout flag used when schemas are assembled."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        gateway = config.get("gateway") or {}
        platforms = gateway.get("platforms") or {}
        photon = platforms.get("photon") or {}
        extra = photon.get("extra") or {}
        return extra.get("conversation_actions_enabled") is True
    except Exception:
        return False


def _error(code: str, detail: str) -> str:
    return json.dumps({"success": False, "error": code, "detail": detail})


def _receipt(action: str, **fields: Any) -> str:
    result = {"success": True, "action": action}
    result.update(fields)
    return json.dumps(result, ensure_ascii=False)


def _validate_request(args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
    action = str(args.get("action") or "").strip()
    if action not in {"react", "unreact", "reply", "present_images"}:
        return None, "action must be react, unreact, reply, or present_images", 0

    target = args.get("target")
    if action == "present_images":
        if target is not None:
            return None, "present_images does not accept a message target", 0
        images = args.get("images")
        if not isinstance(images, list) or not 2 <= len(images) <= 5:
            return None, "present_images requires 2 to 5 image paths", 0
        if any(not isinstance(path, str) or not path.strip() for path in images):
            return None, "every image path must be a non-empty string", 0
        if args.get("emoji") or args.get("text"):
            return None, "present_images accepts only images and an optional caption", 0
        return action, None, 0

    if not isinstance(target, dict):
        return None, f"{action} requires a target", 0
    has_trigger = target.get("trigger") is True
    has_back = "messages_back" in target
    if has_trigger == has_back:
        return None, "target must set exactly one of trigger or messages_back", 0
    if has_back:
        back = target.get("messages_back")
        if isinstance(back, bool) or not isinstance(back, int) or back < 0:
            return None, "messages_back must be a non-negative integer", 0
    else:
        back = 0

    if action == "react":
        emoji = args.get("emoji")
        if not isinstance(emoji, str) or not emoji.strip():
            return None, "react requires a non-empty emoji", 0
        if args.get("text") or args.get("images") or args.get("caption"):
            return None, "react accepts only target and emoji", 0
    elif action == "unreact":
        if any(args.get(key) for key in ("emoji", "text", "images", "caption")):
            return None, "unreact accepts only a target", 0
    elif action == "reply":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return None, "reply requires non-empty text", 0
        if args.get("emoji") or args.get("images") or args.get("caption"):
            return None, "reply accepts only target and text", 0
    return action, None, back


def _open_session_db():
    from hermes_state import SessionDB

    return SessionDB(read_only=True)


def _resolve_session_id(runner: Any, explicit: str, session_key: str) -> str:
    if explicit:
        return explicit
    store = getattr(runner, "session_store", None)
    if store is not None and session_key:
        try:
            return store.peek_session_id(session_key) or ""
        except Exception:
            pass
    return get_session_env("HERMES_SESSION_ID", "")


def resolve_target_message_id(
    *,
    session_id: str,
    trigger_message_id: str,
    messages_back: int,
    db: Any = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a user bubble relative to this turn's exact persisted trigger."""
    if not trigger_message_id:
        return None, "the triggering Photon message has no platform identifier"
    if messages_back == 0:
        return trigger_message_id, None
    if not session_id:
        return None, "the active conversation has no durable session"

    owns_db = db is None
    try:
        db = db or _open_session_db()
        messages = db.get_messages_as_conversation(
            session_id,
            include_ancestors=True,
            include_row_ids=True,
        )
    except Exception as exc:
        return None, f"conversation history is unavailable: {exc}"
    finally:
        if owns_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    user_messages = [
        message
        for message in messages
        if message.get("role") == "user"
        and message.get("display_kind") != "internal_notification"
    ]
    # Compression can clone a protected tail into a child session. Collapse
    # those copies by durable platform id while retaining the newest row.
    seen = set()
    unique_reversed = []
    for message in reversed(user_messages):
        platform_id = message.get("message_id")
        if platform_id and platform_id in seen:
            continue
        if platform_id:
            seen.add(platform_id)
        unique_reversed.append(message)
    user_messages = list(reversed(unique_reversed))

    trigger_index = next(
        (
            index
            for index in range(len(user_messages) - 1, -1, -1)
            if user_messages[index].get("message_id") == trigger_message_id
        ),
        -1,
    )
    target_index = trigger_index - messages_back
    if trigger_index < 0:
        return None, "the triggering message is not present in this conversation"
    if target_index < 0:
        return None, f"no user message exists {messages_back} back"
    target_id = user_messages[target_index].get("message_id")
    if not target_id:
        return None, "the selected historical message predates platform-id persistence"
    return str(target_id), None


def _resolve_runtime() -> Tuple[Any, Any, str, str]:
    platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    if platform != "photon" or not chat_id:
        return None, None, chat_id, session_key
    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = (
            getattr(runner, "adapters", {}).get(Platform("photon"))
            if runner
            else None
        )
    except Exception:
        runner = adapter = None
    return runner, adapter, chat_id, session_key


async def conversation_action_tool(args: Dict[str, Any], **kwargs: Any) -> str:
    action, validation_error, messages_back = _validate_request(args)
    if validation_error:
        return _error("invalid_request", validation_error)

    runner, adapter, chat_id, session_key = _resolve_runtime()
    if runner is None or adapter is None:
        return _error(
            "current_conversation_unavailable",
            "this action is available only in a live Photon gateway conversation",
        )
    if not bool(getattr(adapter, "conversation_actions_enabled", False)):
        return _error("feature_disabled", "Photon conversation actions are disabled")
    if not bool(getattr(adapter, "is_connected", False)):
        return _error("adapter_disconnected", "the Photon adapter is not connected")

    if action == "present_images":
        result = await adapter.send_image_group(
            chat_id,
            [str(path) for path in args["images"]],
            caption=str(args.get("caption") or "").strip() or None,
        )
        if not result.success:
            return _error("action_failed", result.error or "image group failed")
        raw = result.raw_response or {}
        return _receipt(
            action,
            sent_message_ids=[
                raw.get("parent_message_id"),
                *(raw.get("child_message_ids") or []),
            ],
            part_count=raw.get("part_count"),
            presentation="group",
        )

    trigger_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "")
    session_id = _resolve_session_id(
        runner,
        str(kwargs.get("session_id") or ""),
        session_key,
    )
    target_id, target_error = resolve_target_message_id(
        session_id=session_id,
        trigger_message_id=trigger_id,
        messages_back=messages_back,
    )
    if target_error or not target_id:
        return _error("target_unavailable", target_error or "message target unavailable")

    if action == "react":
        result = await adapter.add_reaction(
            chat_id, str(args.get("emoji") or "").strip(), message_id=target_id
        )
        if not result.get("success"):
            return _error("action_failed", result.get("error") or "reaction failed")
        return _receipt(
            action,
            target_message_id=target_id,
            sent_message_ids=[],
            part_count=1,
            presentation="reaction",
        )
    if action == "unreact":
        result = await adapter.remove_reaction(chat_id, message_id=target_id)
        if not result.get("success"):
            return _error("action_failed", result.get("error") or "unreact failed")
        return _receipt(
            action,
            target_message_id=target_id,
            sent_message_ids=[],
            part_count=0,
            presentation="reaction_removed",
        )

    result = await adapter.send(
        chat_id,
        str(args.get("text") or "").strip(),
        reply_to=target_id,
    )
    if not result.success:
        raw = result.raw_response or {}
        return _error(
            str(raw.get("error_class") or "action_failed"),
            result.error or "native reply failed",
        )
    raw = result.raw_response or {}
    message_ids = list(raw.get("message_ids") or [result.message_id])
    return _receipt(
        action,
        target_message_id=target_id,
        sent_message_ids=message_ids,
        part_count=len(message_ids),
        presentation="reply",
    )
