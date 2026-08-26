"""Agent-facing tools supplied by the Photon platform plugin."""

from __future__ import annotations

from .conversation_actions import (
    TOOL_NAME,
    conversation_action_tool,
    conversation_actions_configured,
)


CONVERSATION_ACTION_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Use a native iMessage affordance in the current Photon conversation only. "
        "React to the exact user bubble that earned it, remove your reaction, anchor "
        "a true threaded reply when context matters, or present 2–5 useful images as "
        "one ordered stack. Reactions are occasional social punctuation, never status "
        "signals; they may be the whole response when sufficient. Do not narrate an "
        "action or decorate every turn. Reply/image content is the final response by "
        "default; set allow_follow_up only when a separate top-level message adds "
        "genuinely useful content. This tool cannot address another chat."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["react", "unreact", "reply", "present_images"],
            },
            "target": {
                "description": (
                    "Exact user bubble in this conversation. trigger means the bubble "
                    "that started this turn; messages_back counts earlier user bubbles "
                    "relative to that trigger (0 is the trigger)."
                ),
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"trigger": {"const": True}},
                        "required": ["trigger"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "messages_back": {"type": "integer", "minimum": 0}
                        },
                        "required": ["messages_back"],
                        "additionalProperties": False,
                    },
                ],
            },
            "emoji": {
                "type": "string",
                "description": "Emoji for react. Classic Tapbacks and custom emoji work.",
            },
            "text": {
                "type": "string",
                "description": "Text for a native threaded reply.",
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 5,
                "description": "Ordered approved local image paths for one native stack.",
            },
            "caption": {
                "type": "string",
                "description": "Optional short caption placed after the image stack.",
            },
            "allow_follow_up": {
                "type": "boolean",
                "default": False,
                "description": (
                    "For reply or present_images only. Set true only when a separate "
                    "top-level final message will add useful, non-duplicate content."
                ),
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def register_tools(ctx) -> None:
    ctx.register_tool(
        name=TOOL_NAME,
        toolset="hermes-photon",
        schema=CONVERSATION_ACTION_SCHEMA,
        handler=conversation_action_tool,
        check_fn=conversation_actions_configured,
        is_async=True,
        description=CONVERSATION_ACTION_SCHEMA["description"],
        emoji="💬",
    )
