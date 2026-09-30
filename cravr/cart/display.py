from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..debug_utils import debug
from .routing import _NEXT_STEP_OPTIONS, CART_REMOVE_OOS_VALUE


def _cart_reply(content: str, **cart_state: Any) -> dict:
    return {
        "cart_state": cart_state,
        "messages": [AIMessage(content=content, name="cart_agent")],
    }


def _item_lines(items: list[dict[str, Any]], item_details: dict[str, Any] | None = None) -> list[str]:
    item_details = item_details or {}
    lines = []
    for i in items:
        lines.append(
            f"- {i.get('item_name')} x{i.get('quantity', 1)} — ₹{i.get('total', i.get('final_price', 0))}"
            + ("  ⚠️ out of stock" if i.get("in_stock") is False else "")
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
    items = cart.get("items") or []
    if not items:
        return cart.get("message") or "Your cart is empty."
    sections = ["**Your cart:**\n\n" + "\n".join(_item_lines(items, item_details))]
    pricing = cart.get("pricing") or {}
    to_pay = pricing.get("to_pay")
    if to_pay is not None:
        sections.append(f"**Total to pay:** ₹{to_pay}")
    out_of_stock = cart.get("out_of_stock") or []
    if out_of_stock:
        sections.append(
            f"⚠️ **{', '.join(out_of_stock)} can't be ordered right now** (out of stock as it is -- often one "
            "of its add-ons). Remove it or pick something else before you pay."
        )
    coupon = cart.get("coupon_applied")
    if coupon:
        sections.append(f"**Coupon applied:** {coupon} (-₹{cart.get('coupon_discount', 0)})")
    return "\n\n".join(sections)


def _item_details(cart: dict[str, Any], address_id: str) -> dict[str, Any]:
    items = cart.get("items") or []
    if not items:
        return {}
    if all("addons" in i for i in items):  # structured cart: details already in hand
        return {str(i.get("menu_item_id")): {"addons": i.get("addons") or [], "variants": i.get("variants") or []}
                for i in items}
    return food_mcp.get_cart_item_details(address_id)


def _show_cart(address_id: str, lead_in: str = "", cart: dict[str, Any] | None = None) -> dict:
    """`cart` = a cart already fetched (e.g. from update_food_cart's response), to skip another call."""
    if not (isinstance(cart, dict) and cart.get("status") in ("success", "empty", "text")):
        cart = food_mcp.get_cart(address_id)
    debug("cart_agent", f"get_cart -> status={cart.get('status')!r} item_count={len(cart.get('items') or [])} "
          f"out_of_stock={cart.get('out_of_stock') or []}")
    if cart.get("status") == "error":
        return _cart_reply(f"Couldn't load your cart: {cart.get('message', 'unknown error')}")
    body = _format_cart(cart, _item_details(cart, address_id))
    if cart.get("status") == "text" and cart.get("out_of_stock"):
        body += (f"\n\n⚠️ **{', '.join(cart['out_of_stock'])} can't be ordered right now.** "
                 "Remove it or pick something else before you pay.")
    content = f"{lead_in}\n\n{body}".strip() if lead_in else body
    return _cart_reply(content, **cart)


REMOVE_OOS_OPTION = {"value": CART_REMOVE_OOS_VALUE, "label": "Remove out-of-stock items"}


def _show_cart_with_next_steps(address_id: str, lead_in: str = "", cart: dict[str, Any] | None = None) -> dict:
    result = _show_cart(address_id, lead_in=lead_in, cart=cart)
    cart_state = result.get("cart_state") or {}
    if cart_state.get("status") in ("text", "success"):
        options = list(_NEXT_STEP_OPTIONS)
        if cart_state.get("out_of_stock"):
            options = [REMOVE_OOS_OPTION] + options
        result["cart_state"] = {**cart_state, "options": options}
    return result
