from __future__ import annotations

from typing import Any

MANAGED_KEYS = {
    "missing_information",
    "important_missing_information",
    "clarification_questions",
    "raw_intent",
    "raw_context",
}


def latest_user_text(messages: Any) -> str:
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
