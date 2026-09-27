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


def _item_lines(items: list[dict[str, Any]], item_details: dict[str, Any] | None = None) -> list[str]:
    """One "- name xN — ₹price" bullet per cart item, with any chosen
    variant inline and each add-on as its OWN indented sub-bullet directly
    under it (never comma-joined onto one line -- that reads as a wall of
    text) -- sourced from get_cart_item_details, since get_cart's own items
    list carries no variant/add-on info. Shared by the plain cart view and
    the checkout order summary so both look the same and both render as a
    real nested list rather than a run-on paragraph.

    Always "-" -- never "+" (markdown-inconsistent with the item bullet
    above it) and never a numbered "1."/"2." marker: UI/components.py's
    _clean_text strips any line starting with a digit whenever the message
    carries option chips (e.g. the payment-method question always does),
    which would silently delete a numbered item line."""
    item_details = item_details or {}
    lines = []
    for i in items:
        lines.append(
            f"- {i.get('item_name')} x{i.get('quantity', 1)} — ₹{i.get('total', i.get('final_price', 0))}"
        )
        detail = item_details.get(str(i.get("menu_item_id"))) or {}
        variant_names = [v.get("name") for v in (detail.get("variants") or []) if v.get("name")]
        if variant_names:
            lines[-1] += f" ({', '.join(variant_names)})"
        addon_names = [a.get("name") for a in (detail.get("addons") or []) if a.get("name")]
        for name in addon_names:
            lines.append(f"  - {name}")
    return lines


def _format_cart(cart: dict[str, Any], item_details: dict[str, Any] | None = None) -> str:
    """Built as separate markdown sections (items / total / coupon) joined
    by blank lines -- never packed onto one or two lines with no blank-line
    separation, which markdown reads as one run-on paragraph (a list needs
    a blank line to end, or whatever text follows gets swallowed into the
    last bullet)."""
    items = cart.get("items") or []
    if not items:
        return cart.get("message") or "Your cart is empty."
    sections = ["**Your cart:**\n\n" + "\n".join(_item_lines(items, item_details))]
    pricing = cart.get("pricing") or {}
    to_pay = pricing.get("to_pay")
    if to_pay is not None:
        sections.append(f"**Total to pay:** ₹{to_pay}")
    coupon = cart.get("coupon_applied")
    if coupon:
        sections.append(f"**Coupon applied:** {coupon} (-₹{cart.get('coupon_discount', 0)})")
    return "\n\n".join(sections)


def _show_cart(address_id: str, lead_in: str = "") -> dict:
    cart = food_mcp.get_cart(address_id)
    debug("cart_agent", f"get_cart -> status={cart.get('status')!r} item_count={len(cart.get('items') or [])}")
    if cart.get("status") == "error":
        return _cart_reply(f"Couldn't load your cart: {cart.get('message', 'unknown error')}")
    item_details = food_mcp.get_cart_item_details(address_id) if cart.get("items") else {}
    body = _format_cart(cart, item_details)
    content = f"{lead_in}\n\n{body}".strip() if lead_in else body
    return _cart_reply(content, **cart)


def _show_cart_with_next_steps(address_id: str, lead_in: str = "") -> dict:
    result = _show_cart(address_id, lead_in=lead_in)
    cart_state = result.get("cart_state") or {}
    if cart_state.get("status") in ("text", "success"):
        result["cart_state"] = {**cart_state, "options": _NEXT_STEP_OPTIONS}
    return result
