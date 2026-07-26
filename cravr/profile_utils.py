"""Helpers shared by the extraction agents (intent/context) for accumulating
structured facts across turns.

The local model (`llama3.2:1b`) only ever has to describe what's in the
*current* message — it is small and inconsistent about restating the whole
conversation faithfully. Persistence and "what's still missing" are handled
here in plain code instead of being trusted to the model, so an answer given
in an earlier turn is never lost just because a later extraction pass forgot
to repeat it.
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

    Handles both raw ``{"role": ..., "content": ...}`` dicts (what the CLI
    feeds in) and LangChain message objects (what `add_messages` stores),
    since nodes can see either depending on where in the turn they run.
    """
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
    """Layer freshly-extracted fields onto the running profile.

    A field from `incoming` only overwrites `previous` when it actually
    carries information (non-null/non-empty). List fields are unioned rather
    than replaced, so e.g. dietary preferences mentioned in different turns
    both stick around.
    """
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
