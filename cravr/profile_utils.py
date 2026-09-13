"""Helpers shared by the extraction agents (intent/context) for accumulating
structured facts across turns. The LLM only ever describes what's in the
*current* message -- it's inconsistent about restating the whole conversation
faithfully -- so persistence and "what's still missing" are handled here in
plain code instead, and an earlier turn's answer is never lost.
"""

from __future__ import annotations

from typing import Any

# Bookkeeping keys the agents recompute themselves every call; a fresh
# extraction should never blindly overwrite these via the generic merge.
MANAGED_KEYS = {
    "missing_information",
    "important_missing_information",
    "clarification_questions",
    "raw_intent",
    "raw_context",
}


def latest_user_text(messages: Any) -> str:
    """Return the most recent human/user message content as plain text.
    Handles both raw {"role", "content"} dicts (from the CLI) and LangChain
    message objects (from `add_messages`), since nodes can see either."""
    if not messages:
        return ""

    for message in reversed(list(messages)):
        if isinstance(message, dict):
            if message.get("role") in {"user", "human"} and message.get("content"):
                return str(message["content"])
            continue
        if getattr(message, "type", None) in {"human", "user"}:
            content = getattr(message, "content", "")
            if content:
                return str(content)
        if getattr(message, "role", None) in {"human", "user"}:
            content = getattr(message, "content", "")
            if content:
                return str(content)

    last_message = messages[-1]
    return str(getattr(last_message, "content", last_message))


def merge_profile(previous: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Layer freshly-extracted fields onto the running profile. A field only
    overwrites when it actually carries information (non-null/non-empty);
    list fields are unioned rather than replaced, so e.g. dietary preferences
    from different turns both stick around."""
    merged: dict[str, Any] = dict(previous or {})
    for key, value in (incoming or {}).items():
        if key in MANAGED_KEYS:
            continue
        if value in (None, "", [], {}):
            continue
        existing = merged.get(key)
        if isinstance(value, list) and isinstance(existing, list):
            combined = list(existing)
            for item in value:
                if item not in combined:
                    combined.append(item)
            merged[key] = combined
        else:
            merged[key] = value
    return merged
