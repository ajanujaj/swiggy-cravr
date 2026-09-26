from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..debug_utils import debug
from ..state import CravrState
from .display import _cart_reply
from .guards import _suggest_complementary_items
from .parsing import _parse_existing_cart_items
from .restaurant_context import _current_restaurant_id, _current_restaurant_name


def _offer_dishes(
    lead_in: str, batch: list[dict[str, Any]], filters: dict[str, Any],
    cart_state: dict[str, Any] | None = None,
) -> dict:
    from ..recommend.RecommendAgent import ONLY_MATCH_OPTIONS, _build_recommendation_text, _dish_pick_options

    body = _build_recommendation_text(batch)
    content = f"{lead_in}\n{body}".strip() if lead_in else body
    options = ONLY_MATCH_OPTIONS if len(batch) == 1 else _dish_pick_options(batch)
    outcome: dict[str, Any] = {
        "recommendation_state": {
            "last_recommendations": batch,
            "alternates": [],
            "filters": filters,
            "awaiting_feedback": True,
            "satisfied": False,
            "options": options,
        },
        "messages": [AIMessage(content=content, name="recommendation_agent")],
    }
    if cart_state is not None:
        outcome["cart_state"] = cart_state
    return outcome


def _suggest_more_from_restaurant(state: CravrState, address_id: str) -> dict:
    restaurant_id = _current_restaurant_id(state)
    if not restaurant_id:
        return _cart_reply("I've lost track of which restaurant your cart is from -- try viewing your cart first.")

    cart = food_mcp.get_cart(address_id)
    existing_items = _parse_existing_cart_items(cart.get("message") or "")
    existing_ids = {str(item["menu_item_id"]) for item in existing_items}
    cart_item_id = existing_items[0]["menu_item_id"] if existing_items else None

    restaurant_name = _current_restaurant_name(state) or ""
    menu_items = food_mcp.get_restaurant_menu_items(restaurant_id, address_id, restaurant_name)
    debug(
        "cart_agent",
        f"suggest_more: restaurant_id={restaurant_id!r} menu_items={len(menu_items)} "
        f"existing_ids={existing_ids!r} cart_item_id={cart_item_id!r}",
    )
    if not menu_items:
        return _cart_reply("Couldn't load that restaurant's menu right now -- try again in a moment.")

    suggestions = _suggest_complementary_items(menu_items, cart_item_id, existing_ids, limit=5)
    if not suggestions:
        return _cart_reply("Looks like you've already got everything worth adding from this restaurant!")

    filters = dict((state.get("recommendation_state") or {}).get("filters") or {})
    return _offer_dishes("Goes well with what's in your cart:", suggestions, filters)


def _start_over_new_restaurant(state: CravrState, address_id: str) -> dict:
    restaurant_id = _current_restaurant_id(state)

    flush_result = food_mcp.flush_cart()
    debug("cart_agent", f"start_over: flush_cart -> {flush_result!r}")
    if flush_result.get("status") != "success":
        return _cart_reply(f"Couldn't clear the cart: {flush_result.get('message', 'unknown error')}")

    cleared_cart_state = {"restaurant": None}

    brief = state.get("recommendation_brief") or {}
    if not brief:
        return _cart_reply(
            "Cart's cleared. Tell me what you're craving and I'll find some other restaurants.",
            **cleared_cart_state,
        )

    from ..recommend.RecommendAgent import fetch_recommendations

    exclude = {str(restaurant_id)} if restaurant_id else set()
    filters = dict(brief.get("filters") or {})
    ranked = fetch_recommendations(brief, address_id=address_id, exclude_restaurant_ids=exclude)
    if not ranked:
        return _cart_reply(
            "Cart's cleared, but I couldn't find another restaurant with the same kind of food "
            "right now -- tell me what to change and I'll look again.",
            **cleared_cart_state,
        )
    return _offer_dishes(
        "Cart's cleared. Other restaurants with what you were after:", ranked, filters,
        cart_state=cleared_cart_state,
    )


def _clear_cart_and_add_pending(state: CravrState, address_id: str) -> dict:
    from .add_flow import add_dish_to_cart

    cart_state = state.get("cart_state") or {}
    pending = cart_state.get("pending_dish")
    if not isinstance(pending, dict):
        return _cart_reply("I don't have a dish waiting to be added -- pick one from the options first.")

    flush_result = food_mcp.flush_cart()
    debug("cart_agent", f"clear_and_add: flush_cart -> {flush_result!r}; adding {pending.get('item_name')!r}")
    if flush_result.get("status") != "success":
        return _cart_reply(f"Couldn't clear the cart: {flush_result.get('message', 'unknown error')}")

    return add_dish_to_cart(
        pending, address_id, quantity=cart_state.get("pending_quantity") or 1,
        lead_in="Cart cleared.",
        preselected_addons=cart_state.get("pending_addons"),
    )
