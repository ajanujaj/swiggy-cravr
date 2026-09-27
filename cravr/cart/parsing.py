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
    """The current cart's items as {"menu_item_id", "quantity"} dicts, for
    building a merged cartItems payload before an update_cart call, or for
    verifying an item actually landed after one. Prefers food_mcp.get_cart's
    own structured `items` list -- reliable now that get_cart reads
    Swiggy's structured_content directly rather than only its
    human-readable text (see food_mcp/cart.py) -- and falls back to
    regex-parsing `message` only for the residual case where get_cart
    genuinely had nothing but text to work with (status == "text").
    get_cart's `message` is None whenever it already parsed real items, so
    parsing it unconditionally (the old code at every call site here) saw
    an empty cart any time get_cart succeeded via structured data instead
    of text -- silently dropping whatever was already in the cart on the
    next merged update_cart call, or wrongly refusing an add-on attempt
    that had actually landed (live regression, 2026-09-27: an add-on going
    out of stock made update_cart's own response json-mode-parseable, and
    the "verify it landed" check here then saw an empty cart)."""
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
