from __future__ import annotations

import json
from typing import Any


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] not in (None, ""):
            return d[key]
    return default


def _unwrap(result: Any) -> Any:
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
    try:
        value, _end = _JSON_DECODER.raw_decode(text.lstrip())
        return value
    except json.JSONDecodeError:
        return text
