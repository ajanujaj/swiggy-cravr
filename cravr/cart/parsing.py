from __future__ import annotations

import re
from typing import Any

_CART_LINE_RE = re.compile(
    r"^\s*-\s*.+?(?:\s+x(?P<qty>\d+))?\s*[—-]\s*₹[\d,.]+\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)


def _parse_existing_cart_items(cart_message: str) -> list[dict[str, Any]]:
    items = []
    for match in _CART_LINE_RE.finditer(cart_message or ""):
        items.append({
            "menu_item_id": match.group("id").strip(),
            "quantity": int(match.group("qty")) if match.group("qty") else 1,
        })
    return items


def _existing_cart_items(cart: dict[str, Any]) -> list[dict[str, Any]]:
    items = cart.get("items")
    if items:
        parsed = [
            {"menu_item_id": i.get("menu_item_id"), "quantity": i.get("quantity", 1)}
            for i in items if i.get("menu_item_id") is not None
        ]
        if parsed:
            return parsed
    if cart.get("status") == "text":
        return _parse_existing_cart_items(cart.get("message") or "")
    return []


