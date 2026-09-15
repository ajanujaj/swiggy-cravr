"""Cart agent: view/add/clear the Swiggy cart and manage coupons. Routed to
directly from graph.py's `_route_entry` (see `is_cart_request`) whenever a
message is clearly cart-topic, bypassing intent/history/decision the same
way `is_pure_feedback` short-circuits reactions to an offered dish list.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..debug_utils import debug
from ..profile_utils import latest_user_text
from ..state import CravrState

# Deterministic keyword routing -- same lightweight-heuristic style as
# RecommendAgent.is_pure_feedback, no extra LLM call just to tell "add to
# cart" from "clear my cart".
_CART_TOPIC_RE = re.compile(
    r"\bcart\b|\bcoupons?\b|\bpromo\s*codes?\b|\bdiscounts?\b|"
    r"\border it\b|\border this\b|\border that\b|\badd it\b|\badd this\b|\badd that\b",
    re.IGNORECASE,
)
_COUPON_APPLY_RE = re.compile(
    r"\b(?:apply|use|redeem)\s+(?:coupon\s+|code\s+|promo\s+)?([A-Za-z0-9][A-Za-z0-9-]{2,19})\b",
    re.IGNORECASE,
)
_CLEAR_RE = re.compile(r"\b(clear|empty|flush)\b.*\bcart\b", re.IGNORECASE)
_ADD_RE = re.compile(
    r"\badd\b.*\bcart\b|\border (it|this|that)\b|\bput (it|this|that) in (my |the )?cart\b|"
    r"\badd (it|this|that)\b",
    re.IGNORECASE,
)
_VIEW_RE = re.compile(
    r"\b(view|show|see|check)\b.*\bcart\b|\bwhat'?s in (my |the )?cart\b|^\s*(my )?cart\s*$",
    re.IGNORECASE,
)
_COUPON_LIST_RE = re.compile(r"\bcoupons?\b|\boffers?\b|\bdiscounts?\b|\bdeals?\b", re.IGNORECASE)
_QTY_RE = re.compile(r"\b(\d+)\b")


def is_cart_request(text: str) -> bool:
    """Entry-router gate (graph.py) -- only messages with a strong cart/
    coupon signal get routed here at all. Also catches a bare "apply
    SAVE20"/"use FREESHIP" with no "coupon"/"cart" word in it, since
    `_classify_cart_action` supports that phrasing too."""
    t = (text or "").strip()
    return bool(_CART_TOPIC_RE.search(t) or _COUPON_APPLY_RE.search(t))


def _classify_cart_action(text: str) -> tuple[str, Any]:
    t = (text or "").strip()
    if _CLEAR_RE.search(t):
        return "clear", None
    coupon_match = _COUPON_APPLY_RE.search(t)
    if coupon_match:
        return "apply_coupon", coupon_match.group(1)
    if _ADD_RE.search(t):
        qty_match = _QTY_RE.search(t)
        return "add", int(qty_match.group(1)) if qty_match else 1
    if _COUPON_LIST_RE.search(t):
        return "coupons", None
    if _VIEW_RE.search(t):
        return "view", None
    # is_cart_request already confirmed cart-topic vocabulary; default to
    # showing the cart rather than silently doing nothing.
    return "view", None


def _log_action(action: str, arg: Any) -> None:
    debug("cart_agent", f"classified action={action!r} arg={arg!r}")


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


def _current_restaurant_id(state: CravrState) -> str | None:
    """Cart is single-restaurant on Swiggy -- prefer whatever's already in
    the cart, else fall back to a just-picked dish's restaurant."""
    cart_state = state.get("cart_state") or {}
    restaurant = cart_state.get("restaurant") or {}
    if isinstance(restaurant, dict) and restaurant.get("id"):
        return restaurant.get("id")
    chosen = (state.get("recommendation_state") or {}).get("last_recommendations") or []
    if len(chosen) == 1:
        return chosen[0].get("restaurant_id")
    return None


def add_dish_to_cart(dish: dict[str, Any], address_id: str, quantity: int = 1, lead_in: str = "") -> dict:
    """Add one dish via update_cart, then show the result via get_cart.
    Single call site for "how a dish gets added" -- used by both this
    node's own "add" action and RecommendAgent's pick-to-cart handoff, so
    the two can't drift apart. `lead_in` prefixes whatever message comes
    back (success or failure)."""
    menu_item_id = dish.get("menu_item_id")
    restaurant_id = dish.get("restaurant_id")
    debug(
        "cart_agent",
        f"add_dish_to_cart dish={dish.get('item_name')!r} menu_item_id={menu_item_id!r} "
        f"restaurant_id={restaurant_id!r} quantity={quantity} address_id={address_id!r}",
    )
    if not menu_item_id or not restaurant_id:
        content = "I don't have enough details on that dish to add it to your cart -- try searching again."
        return _cart_reply(f"{lead_in} {content}".strip())

    result = food_mcp.update_cart(
        restaurant_id, address_id,
        [{"id": menu_item_id, "quantity": quantity}],
        restaurant_name=dish.get("restaurant_name"),
    )
    debug("cart_agent", f"update_cart -> {result!r}")
    if result.get("status") != "success":
        content = f"Couldn't add that to your cart: {result.get('message', 'unknown error')}"
        return _cart_reply(f"{lead_in} {content}".strip())

    # update_cart succeeding doesn't guarantee the item actually landed --
    # a real response was 'Cart updated.\nCart is empty.' -- so its own
    # message is relayed rather than an independent "Added X" claim, unless
    # it's just going to repeat get_cart's own "Cart is empty." next (avoid
    # the duplicate line); get_cart right after is the real source of truth.
    update_message = result.get("message")
    if update_message and "cart is empty" not in update_message.lower():
        prefix = f"{lead_in} {update_message}".strip()
    else:
        prefix = lead_in
    return _show_cart(address_id, lead_in=prefix)


def cart_agent_node(state: CravrState) -> dict:
    user_input = latest_user_text(state.get("messages") or [])
    context_profile = state.get("context_profile") or {}
    address_id = context_profile.get("address_id") if isinstance(context_profile, dict) else None
    debug("cart_agent", f"user_input={user_input!r} address_id={address_id!r}")

    if not address_id:
        debug("cart_agent", "no address_id in context_profile -- asking the user to order first")
        return _cart_reply(
            "I don't have a delivery address saved yet -- tell me what you'd like to eat first "
            "so I can look one up, then come back to your cart."
        )

    action, arg = _classify_cart_action(user_input)
    _log_action(action, arg)

    if action == "clear":
        result = food_mcp.flush_cart()
        debug("cart_agent", f"flush_cart -> {result!r}")
        if result.get("status") != "success":
            return _cart_reply(f"Couldn't clear the cart: {result.get('message', 'unknown error')}")
        return _cart_reply(result.get("message") or "Your cart is now empty.", items=[], pricing={})

    if action == "apply_coupon":
        result = food_mcp.apply_coupon(str(arg), address_id)
        debug("cart_agent", f"apply_coupon({arg!r}) -> {result!r}")
        if result.get("status") != "success":
            return _cart_reply(f"Couldn't apply that coupon: {result.get('message', 'unknown error')}")
        return _show_cart(address_id, lead_in=result.get("message") or "Coupon applied.")

    if action == "coupons":
        restaurant_id = _current_restaurant_id(state)
        debug("cart_agent", f"coupons requested, restaurant_id={restaurant_id!r}")
        if not restaurant_id:
            return _cart_reply("Add something to your cart first, then I can check available coupons.")
        result = food_mcp.fetch_coupons(restaurant_id, address_id)
        debug("cart_agent", f"fetch_coupons -> status={result.get('status')!r} count={len(result.get('coupons') or [])}")
        if result.get("status") == "error":
            return _cart_reply(f"Couldn't fetch coupons: {result.get('message', 'unknown error')}")
        coupons = result.get("coupons") or []
        if not coupons:
            return _cart_reply(result.get("message") or "No coupons available for this restaurant right now.")
        lines = [f"  - {c.get('title')}: {c.get('subtitle') or c.get('description') or ''}" for c in coupons]
        return _cart_reply("Available coupons:\n" + "\n".join(lines))

    if action == "add":
        chosen = (state.get("recommendation_state") or {}).get("last_recommendations") or []
        debug("cart_agent", f"add requested, {len(chosen)} dish(es) in last_recommendations")
        if len(chosen) != 1:
            return _cart_reply("Which dish should I add? Pick one from the options first.")
        return add_dish_to_cart(chosen[0], address_id, quantity=arg or 1)

    return _show_cart(address_id)
