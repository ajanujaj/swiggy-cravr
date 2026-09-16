"""Low-level helpers shared across the food_mcp submodules: unwrapping a raw
MCP tool result, best-effort JSON parsing, and a couple of small dict
lookups. Nothing here calls a Swiggy tool directly -- see __init__.py's
`_call_tool` for the one place that does.
"""

from __future__ import annotations

import json
from typing import Any


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """First key in `keys` present in `d` with a non-empty value, else
    `default` -- Swiggy's JSON responses use inconsistent key casing/naming
    across tools (e.g. "restaurantId" vs "id"), so call sites list every
    spelling they've seen rather than assuming one."""
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _unwrap(result: Any) -> Any:
    """MCP tool results arrive as an object with a `.content` list of
    `{text: ...}` parts, not a plain value. Pulls the text out and parses a
    JSON prefix from it (see `_parse_leading_json`); passes dicts/lists
    through unchanged (e.g. the mock server's already-decoded responses)."""
    if isinstance(result, (dict, list)):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                return _parse_leading_json(text)
    return result


_JSON_DECODER = json.JSONDecoder()


def _parse_leading_json(text: str) -> Any:
    """Parse a JSON prefix, ignoring trailing text Swiggy appends after it
    (e.g. a UI-widget disclaimer) -- plain json.loads chokes on that with
    "Extra data". Falls back to the raw text if it isn't JSON at all
    (get_addresses' plain-language response)."""
    try:
        value, _end = _JSON_DECODER.raw_decode(text.lstrip())
        return value
    except json.JSONDecodeError:
        return text
