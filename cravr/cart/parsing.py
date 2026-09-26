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


_DRINK_CUES = ("beverage", "drink", "juice", "shake", "lassi", "soda", "mocktail", "smoothie")
_DESSERT_CUES = ("dessert", "sweet", "ice cream", "cake", "pastry", "pudding")
_SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "add-on", "addon")

_COMPLEMENT_ORDER = {
    "main": ["drink", "dessert", "side"],
    "side": ["main", "drink"],
    "drink": ["main", "dessert"],
    "dessert": ["main", "drink"],
}


def _menu_bucket(category_text: str) -> str:
    haystack = (category_text or "").lower()
    if any(cue in haystack for cue in _DRINK_CUES):
        return "drink"
    if any(cue in haystack for cue in _DESSERT_CUES):
        return "dessert"
    if any(cue in haystack for cue in _SIDE_CUES):
        return "side"
    return "main"


def _format_cart(cart: dict[str, Any]) -> str:
    items = cart.get("items") or []
    if items:
        lines = [
            f"  - {i.get('item_name')} x{i.get('quantity', 1)} — ₹{i.get('total', i.get('final_price', 0))}"
            for i in items
        ]
        body = "Here's your cart:\n" + "\n".join(lines)
        pricing = cart.get("pricing") or {}
        to_pay = pricing.get("to_pay")
        if to_pay is not None:
            body += f"\nTotal to pay: ₹{to_pay}"
        coupon = cart.get("coupon_applied")
        if coupon:
            body += f"\nCoupon applied: {coupon} (-₹{cart.get('coupon_discount', 0)})"
        return body
    return cart.get("message") or "Your cart is empty."
