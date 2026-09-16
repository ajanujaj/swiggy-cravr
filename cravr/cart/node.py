"""The cart agent's graph entry point: classify the user's cart-topic
message into an action (routing.py's `_classify_cart_action`) and dispatch
to the right flow.
"""

from __future__ import annotations

from .. import food_mcp
from ..debug_utils import debug
from ..profile_utils import latest_user_text
from ..state import CravrState
from .add_flow import add_dish_to_cart
from .display import _cart_reply, _show_cart_with_next_steps
from .flows import _start_over_new_restaurant, _suggest_more_from_restaurant
from .restaurant_context import _current_restaurant_id, _current_restaurant_name
from .routing import _classify_cart_action, _log_action


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
        return _show_cart_with_next_steps(address_id, lead_in=result.get("message") or "Coupon applied.")

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
        return add_dish_to_cart(
            chosen[0], address_id, quantity=arg or 1,
            known_restaurant_id=_current_restaurant_id(state),
            known_restaurant_name=_current_restaurant_name(state),
        )

    if action == "suggest_more":
        return _suggest_more_from_restaurant(state, address_id)

    if action == "proceed_placeholder":
        # Sub-phase 2 (place_food_order) isn't built yet -- this is
        # intentionally just a stub reply for now.
        return _show_cart_with_next_steps(
            address_id,
            lead_in="Checkout isn't wired up yet -- that's next on the list. "
                    "For now you can keep adding items or browse other restaurants.",
        )

    if action == "start_over":
        return _start_over_new_restaurant(state, address_id)

    return _show_cart_with_next_steps(address_id)
