from __future__ import annotations

import json
import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool

_ADDRESS_LINE_RE = re.compile(
    r"^\d+\.\s*\[(?P<label>[^\]]+)\]\s*[^:]*:\s*(?P<address>.+?)\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)
_SUGGESTED_ID_RE = re.compile(r"Suggested/preselected address ID:\s*([^\s.,]+)")


def get_saved_address() -> dict[str, Any]:
    try:
        raw = _call_tool("get_addresses", {})
    except Exception as exc:
        debug("food_mcp", f"get_addresses FAILED ({type(exc).__name__}: {exc})")
        return {"status": "error", "message": f"Unable to retrieve saved address: {exc}"}

    text = raw if isinstance(raw, str) else json.dumps(raw)

    addresses = {
        m.group("id"): (m.group("label"), m.group("address"))
        for m in _ADDRESS_LINE_RE.finditer(text)
    }
    if not addresses:
        return {"status": "no_results", "message": "No saved addresses found"}

    suggested = _SUGGESTED_ID_RE.search(text)
    address_id = suggested.group(1) if suggested and suggested.group(1) in addresses else next(iter(addresses))
    label, address_line = addresses[address_id]

    return {
        "status": "success",
        "location": label,
        "delivery_area": address_line,
        "address_id": address_id,
    }
