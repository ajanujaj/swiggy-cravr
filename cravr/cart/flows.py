"""The three "what next" branches out of the cart's option buttons: offer
more from the same restaurant, or clear the cart and start over elsewhere.
Both funnel back through `_offer_dishes`, which hands off to
RecommendAgent's own pick machinery rather than duplicating it.
"""

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
    """Hand a dish batch off through the SAME "offer -> pick -> add to
    cart" machinery RecommendAgent already uses, instead of building a
    parallel one here: `name="recommendation_agent"` on the reply message
    is what makes graph_bridge.py render it exactly like a normal
    recommendation (card view or the consolidated list+buttons, depending
    on count -- see UI/components.py), and setting `recommendation_state`
    the same way `_reply(...)` does is what makes next turn's bare-digit
    reply resolve as a pick via `is_pure_feedback` -> `_resolve_choice` ->
    `_reply_with_pick` -> `add_dish_to_cart`, with zero new state-tracking
    code. Deferred import: RecommendAgent imports `add_dish_to_cart` from
    this module at load time, so importing back at module scope would be
    circular -- safe here since both modules are fully loaded by the time
    any node actually runs. `cart_state`, when passed, is included in the
    return so a caller can overwrite state the graph would otherwise
    leave untouched -- LangGraph only updates keys a node's return dict
    actually includes, so a caller with nothing to say about `cart_state`
    (e.g. "add more") must simply omit it here, while one that changed
    the cart (e.g. "start over") must say so explicitly or a STALE value
    survives into the next turn. Live regression: after "start over"
    flushed the cart, the next pick was refused with "Your cart already
    has items from Social" -- the restaurant _current_restaurant_id
    recorded before the flush, never cleared, because this function used
    to omit `cart_state` unconditionally."""
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
    """Option (a): pull more items from the SAME restaurant, preferring
    ones that pair with what's already in the cart (see
    `_suggest_complementary_items`)."""
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
    """Option (c): clear the cart and re-search the same craving/category,
    excluding the restaurant just cleared."""
    restaurant_id = _current_restaurant_id(state)

    flush_result = food_mcp.flush_cart()
    debug("cart_agent", f"start_over: flush_cart -> {flush_result!r}")
    if flush_result.get("status") != "success":
        return _cart_reply(f"Couldn't clear the cart: {flush_result.get('message', 'unknown error')}")

    # The cart is genuinely empty now -- clear the restaurant we recorded
    # for it, or the NEXT add wrongly compares against the one just
    # flushed (live regression: "Your cart already has items from Social"
    # right after "start over" successfully cleared it).
    cleared_cart_state = {"restaurant": None}

    brief = state.get("recommendation_brief") or {}
    if not brief:
        return _cart_reply(
            "Cart's cleared. Tell me what you're craving and I'll find some other restaurants.",
            **cleared_cart_state,
        )

    from ..recommend.RecommendAgent import fetch_recommendations  # deferred -- see _offer_dishes

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
