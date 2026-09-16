"""Rendering the cart back to the user: plain-text formatting (Swiggy's
get_food_cart returns no itemized breakdown of its own) plus the reusable
`_cart_reply` node-return shape and the "what next" button attachment.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..debug_utils import debug
from .routing import _NEXT_STEP_OPTIONS


def _cart_reply(content: str, **cart_state: Any) -> dict:
    return {
        "cart_state": cart_state,
        "messages": [AIMessage(content=content, name="cart_agent")],
    }


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
    # No structured items -- the real get_food_cart returns plain text with
    # no itemized breakdown (Swiggy expects a widget to show that), so its
    # own status message ("Cart is empty.", etc.) is all there is to show.
    return cart.get("message") or "Your cart is empty."


def _show_cart(address_id: str, lead_in: str = "") -> dict:
    cart = food_mcp.get_cart(address_id)
    debug("cart_agent", f"get_cart -> status={cart.get('status')!r} item_count={len(cart.get('items') or [])}")
    if cart.get("status") == "error":
        return _cart_reply(f"Couldn't load your cart: {cart.get('message', 'unknown error')}")
    body = _format_cart(cart)
    content = f"{lead_in}\n{body}".strip() if lead_in else body
    return _cart_reply(content, **cart)


def _show_cart_with_next_steps(address_id: str, lead_in: str = "") -> dict:
    """`_show_cart`, plus the "what next" buttons (add more / proceed /
    browse other restaurants) whenever the cart actually has something in
    it -- attached after the fact rather than threading an `options` param
    through every `_show_cart` caller, so the buttons show up everywhere
    the cart does (every add, "show my cart", after applying a coupon),
    per the design: offer them any time the cart is displayed, not just
    right after the first add."""
    result = _show_cart(address_id, lead_in=lead_in)
    cart_state = result.get("cart_state") or {}
    if cart_state.get("status") in ("text", "success"):
        result["cart_state"] = {**cart_state, "options": _NEXT_STEP_OPTIONS}
    return result
