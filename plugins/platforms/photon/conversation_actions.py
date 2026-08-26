"""Current-conversation-only native actions for Photon sessions."""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional, Tuple

from gateway.session_context import get_session_env


TOOL_NAME = "photon_conversation_action"
_delivered_content_turns: dict[str, bool] = {}
_delivered_content_lock = threading.Lock()


def conversation_actions_configured() -> bool:
    """Return the profile-level rollout flag used when schemas are assembled."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
        gateway = config.get("gateway") or {}
        nested_platforms = gateway.get("platforms") or {}
        top_level_platforms = config.get("platforms") or {}
        nested_photon = nested_platforms.get("photon") or {}
        top_level_photon = top_level_platforms.get("photon") or {}
        nested_extra = (
            nested_photon.get("extra", {})
            if isinstance(nested_photon, dict)
            else {}
        )
        top_level_extra = (
            top_level_photon.get("extra", {})
            if isinstance(top_level_photon, dict)
            else {}
        )
        extra = {
            **(nested_extra if isinstance(nested_extra, dict) else {}),
            **(top_level_extra if isinstance(top_level_extra, dict) else {}),
        }
        return extra.get("conversation_actions_enabled") is True
    except Exception:
        return False


def _error(code: str, detail: str) -> str:
    return json.dumps({"success": False, "error": code, "detail": detail})


def _receipt(action: str, **fields: Any) -> str:
    result = {"success": True, "action": action}
    result.update(fields)
    return json.dumps(result, ensure_ascii=False)


def _set_content_delivery_policy(turn_id: str, allow_follow_up: bool) -> None:
    if not turn_id:
        return
    with _delivered_content_lock:
        if allow_follow_up:
            _delivered_content_turns.pop(turn_id, None)
        else:
            _delivered_content_turns.pop(turn_id, None)
            _delivered_content_turns[turn_id] = True
            while len(_delivered_content_turns) > 1024:
                _delivered_content_turns.pop(next(iter(_delivered_content_turns)))


def suppress_redundant_final(
    response_text: str,
    platform: str = "",
    turn_id: str = "",
    **_: Any,
) -> Optional[str]:
    """Consume a content-action delivery marker at the generic output seam."""
    if str(platform).strip().lower() != "photon" or not turn_id:
        return None
    with _delivered_content_lock:
        suppress = _delivered_content_turns.pop(turn_id, False)
        if not suppress:
            return None
    return "[SILENT]"


def _validate_request(args: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
    action = str(args.get("action") or "").strip()
    if action not in {"react", "unreact", "reply", "present_images"}:
        return None, "action must be react, unreact, reply, or present_images", 0
    if action in {"react", "unreact"} and "allow_follow_up" in args:
        return None, f"{action} does not accept allow_follow_up", 0

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
        if not isinstance(args.get("allow_follow_up", False), bool):
            return None, "allow_follow_up must be a boolean", 0
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
        if not isinstance(args.get("allow_follow_up", False), bool):
            return None, "allow_follow_up must be a boolean", 0
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


#: display_metadata key persisting a debounced turn's ordered bubble ids so
#: messages_back can address individual constituents after restart without
#: exposing extra user turns to the model.
CONSTITUENT_METADATA_KEY = "constituent_message_ids"


def _expand_user_bubbles(message: Dict[str, Any]) -> list:
    """Ordered addressable ids one persisted user row represents.

    A debounced turn persists ONE user row (role alternation intact) whose
    display_metadata carries the ordered constituent provider ids; expand it
    so messages_back counts real bubbles. Ordinary rows are one bubble; rows
    without any identity contribute one non-addressable placeholder so
    counting still spaces correctly.
    """
    metadata = message.get("display_metadata")
    constituent_ids = (
        metadata.get(CONSTITUENT_METADATA_KEY) if isinstance(metadata, dict) else None
    )
    if isinstance(constituent_ids, list) and constituent_ids:
        return [str(entry) if entry else None for entry in constituent_ids]
    platform_id = message.get("message_id")
    return [str(platform_id) if platform_id else None]


def resolve_target_message_id(
    *,
    session_id: str,
    trigger_message_id: str,
    messages_back: int,
    db: Any = None,
    turn_constituents: Optional[list] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a user bubble relative to this turn's exact trigger.

    ``turn_constituents`` is the ACTIVE turn's ordered addressable bubble
    sequence (oldest → newest — debounced constituents plus successful
    steers). ``messages_back`` consumes it newest-first, then continues into
    prior persisted user bubbles, skipping any persisted copy of the current
    turn so a mid-turn crash persist can never double-count.
    """
    constituents = [str(entry) for entry in (turn_constituents or []) if entry]
    if not constituents and trigger_message_id:
        constituents = [str(trigger_message_id)]
    if not constituents:
        return None, "the triggering Photon message has no platform identifier"
    if not trigger_message_id:
        trigger_message_id = constituents[-1]
    if messages_back < len(constituents):
        return constituents[-1 - messages_back], None
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

    # Flatten persisted user rows into real bubbles (oldest → newest).
    persisted_bubbles: list = []
    for message in user_messages:
        persisted_bubbles.extend(_expand_user_bubbles(message))

    if turn_constituents:
        # Live-anchored turn: it belongs to this session by construction.
        # Skip persisted copies of the current turn's bubbles (dedupe), then
        # keep walking back from the newest prior bubble.
        constituent_set = set(constituents)
        prior = [
            bubble
            for bubble in persisted_bubbles
            if bubble is None or bubble not in constituent_set
        ]
        sequence = prior + constituents
    else:
        # Turn-env fallback: the trigger must belong to this conversation;
        # anchor the walk at its persisted position.
        trigger_index = next(
            (
                index
                for index in range(len(persisted_bubbles) - 1, -1, -1)
                if persisted_bubbles[index] == trigger_message_id
            ),
            -1,
        )
        if trigger_index < 0:
            return None, "the triggering message is not present in this conversation"
        sequence = persisted_bubbles[: trigger_index + 1]

    target_index = len(sequence) - 1 - messages_back
    if target_index < 0:
        return None, f"no user message exists {messages_back} back"
    target_id = sequence[target_index]
    if not target_id:
        return None, "the selected historical message predates platform-id persistence"
    return str(target_id), None


def _resolve_trigger_context(
    adapter: Any, session_key: str
) -> Tuple[str, list]:
    """Resolve this turn's trigger id and ordered addressable bubble ids.

    The gateway retargets a running turn's reply anchor on every SUCCESSFUL
    mid-turn steer (``BasePlatformAdapter.update_active_turn_reply_anchor``),
    while ``HERMES_SESSION_MESSAGE_ID`` stays pinned to the event that
    STARTED the turn. Exact conversation actions must follow the same
    ownership rule as final delivery — the latest successful steer owns the
    turn — so consult the adapter's live anchor first: its constituent
    sequence expands a debounced/steered turn into real bubbles (trigger =
    newest). Failed, queued, synthetic, and cross-session events never move
    that anchor, and turn completion clears it, so the fallback to the
    turn-start binding only engages when no anchored active turn exists
    (inline dispatch paths).
    """
    if session_key:
        ids_fn = getattr(adapter, "active_turn_constituent_message_ids", None)
        if callable(ids_fn):
            try:
                anchor_ids = [str(i) for i in (ids_fn(session_key) or []) if i]
            except Exception:
                anchor_ids = []
            if anchor_ids:
                return anchor_ids[-1], anchor_ids
        anchor_fn = getattr(adapter, "active_turn_reply_anchor_message_id", None)
        if callable(anchor_fn):
            try:
                anchored = anchor_fn(session_key)
            except Exception:
                anchored = None
            if anchored:
                return str(anchored), [str(anchored)]
    trigger = get_session_env("HERMES_SESSION_MESSAGE_ID", "")
    return trigger, []


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
        adapter = None
        if runner:
            platform_key = Platform("photon")
            profile = get_session_env("HERMES_SESSION_PROFILE", "")
            if profile:
                adapter = (
                    getattr(runner, "_profile_adapters", {}).get(profile, {})
                ).get(platform_key)
            if adapter is None:
                adapter = getattr(runner, "adapters", {}).get(platform_key)
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

    session_id = _resolve_session_id(
        runner,
        str(kwargs.get("session_id") or ""),
        session_key,
    )
    turn_id = str(kwargs.get("turn_id") or "")
    if not turn_id:
        try:
            from tools.approval import get_current_turn_id

            turn_id = get_current_turn_id()
        except Exception:
            pass
    if action == "present_images":
        result = await adapter.send_image_group(
            chat_id,
            [str(path) for path in args["images"]],
            caption=str(args.get("caption") or "").strip() or None,
        )
        if not result.success:
            return _error("action_failed", result.error or "image group failed")
        raw = result.raw_response or {}
        _set_content_delivery_policy(
            turn_id, bool(args.get("allow_follow_up", False))
        )
        return _receipt(
            action,
            sent_message_ids=[
                raw.get("parent_message_id"),
                *(raw.get("child_message_ids") or []),
            ],
            part_count=raw.get("part_count"),
            presentation="group",
        )

    trigger_id, turn_constituents = _resolve_trigger_context(adapter, session_key)
    target_id, target_error = resolve_target_message_id(
        session_id=session_id,
        trigger_message_id=trigger_id,
        messages_back=messages_back,
        turn_constituents=turn_constituents,
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
    _set_content_delivery_policy(
        turn_id, bool(args.get("allow_follow_up", False))
    )
    return _receipt(
        action,
        target_message_id=target_id,
        sent_message_ids=message_ids,
        part_count=len(message_ids),
        presentation="reply",
    )
