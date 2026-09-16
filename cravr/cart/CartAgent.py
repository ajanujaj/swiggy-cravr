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

# Best-effort extraction of {menu_item_id, quantity} pairs from a real
# get_food_cart/update_food_cart plain-text item list, e.g.
# "  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)". Swiggy's cart
# responses are plain text with no JSON items array for the real tool (see
# food_mcp._strip_widget_disclaimer's docstring), so this is the only way
# to know what's already in the cart before merging in a newly-added item.
# Only the single-item, quantity-omitted format has been observed live; an
# "x<N>" quantity suffix is supported defensively but unverified against a
# real multi-quantity response.
_CART_LINE_RE = re.compile(
    r"^\s*-\s*.+?(?:\s+x(?P<qty>\d+))?\s*[—-]\s*₹[\d,.]+\s*\(ID:\s*(?P<id>[^)]+)\)\s*$",
    re.MULTILINE,
)

# Sentinels for the post-add "what next" buttons -- same convention as
# cravr.options.OTHER_VALUE (a fixed, unmistakable value a tap always
# sends, independent of the button's display label).
CART_ADD_MORE_VALUE = "__cart_add_more__"
CART_PROCEED_VALUE = "__cart_proceed__"
CART_START_OVER_VALUE = "__cart_start_over__"
_NEXT_STEP_SENTINELS = {CART_ADD_MORE_VALUE, CART_PROCEED_VALUE, CART_START_OVER_VALUE}

_NEXT_STEP_OPTIONS = [
    {"value": CART_ADD_MORE_VALUE, "label": "Add more from this restaurant"},
    {"value": CART_PROCEED_VALUE, "label": "Proceed"},
    {"value": CART_START_OVER_VALUE, "label": "Browse other restaurants"},
]

# Menu category header text -> a coarse bucket, for pairing "add more"
# suggestions with what's already in the cart. Matched against
# get_restaurant_menu's "## Category" headers (see food_mcp._parse_menu_items'
# `tags` field), not free text -- a smaller, more literal vocabulary than
# RecommendAgent's intent-parsing cues is enough for menu section names.
_DRINK_CUES = ("beverage", "drink", "juice", "shake", "lassi", "soda", "mocktail", "smoothie")
_DESSERT_CUES = ("dessert", "sweet", "ice cream", "cake", "pastry", "pudding")
_SIDE_CUES = ("side", "starter", "appetizer", "appetiser", "snack", "add-on", "addon")

# Which buckets to prefer suggesting, given what's already in the cart --
# a main course pairs with a drink/dessert/side, not five more mains.
_COMPLEMENT_ORDER = {
    "main": ["drink", "dessert", "side"],
    "side": ["main", "drink"],
    "drink": ["main", "dessert"],
    "dessert": ["main", "drink"],
}


def is_cart_request(text: str) -> bool:
    """Entry-router gate (graph.py) -- only messages with a strong cart/
    coupon signal get routed here at all. Also catches a bare "apply
    SAVE20"/"use FREESHIP" with no "coupon"/"cart" word in it, since
    `_classify_cart_action` supports that phrasing too. The post-add "what
    next" buttons send an exact sentinel (see `_NEXT_STEP_SENTINELS`) that
    wouldn't otherwise match any of the natural-language patterns below."""
    t = (text or "").strip()
    if t in _NEXT_STEP_SENTINELS:
        return True
    return bool(_CART_TOPIC_RE.search(t) or _COUPON_APPLY_RE.search(t))


def _classify_cart_action(text: str) -> tuple[str, Any]:
    t = (text or "").strip()
    if t == CART_ADD_MORE_VALUE:
        return "suggest_more", None
    if t == CART_PROCEED_VALUE:
        return "proceed_placeholder", None
    if t == CART_START_OVER_VALUE:
        return "start_over", None
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


def _parse_existing_cart_items(cart_message: str) -> list[dict[str, Any]]:
    """See _CART_LINE_RE. Returns [{"menu_item_id": ..., "quantity": ...}]."""
    items = []
    for match in _CART_LINE_RE.finditer(cart_message or ""):
        items.append({
            "menu_item_id": match.group("id").strip(),
            "quantity": int(match.group("qty")) if match.group("qty") else 1,
        })
    return items


def _menu_bucket(category_text: str) -> str:
    """Coarse bucket (main/side/drink/dessert) for one menu item's category
    text, via _DRINK_CUES/_DESSERT_CUES/_SIDE_CUES. Defaults to "main" --
    that's the safe default for an uncategorised or genuinely-main item,
    and also what makes an empty/unknown bucket fall back to suggesting
    drinks/desserts/sides rather than more unclassified items."""
    haystack = (category_text or "").lower()
    if any(cue in haystack for cue in _DRINK_CUES):
        return "drink"
    if any(cue in haystack for cue in _DESSERT_CUES):
        return "dessert"
    if any(cue in haystack for cue in _SIDE_CUES):
        return "side"
    return "main"


def _suggest_complementary_items(
    menu_items: list[dict[str, Any]],
    cart_item_id: str | None,
    exclude_ids: set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Up to `limit` items from `menu_items` that pair with what's already
    in the cart: look up the cart item's own bucket from the freshly
    fetched menu (its category header is the only reliable source -- the
    dish record captured when it was originally picked has no category
    info, see food_mcp._parse_search_menu_items), then prefer OTHER
    buckets per `_COMPLEMENT_ORDER`. Tops up with whatever's left (any
    bucket) if the preferred ones don't reach `limit` -- a preference, not
    a hard filter, so "add more" never comes back emptier than it has to."""
    by_id = {str(i.get("menu_item_id")): i for i in menu_items if i.get("menu_item_id")}
    cart_item = by_id.get(str(cart_item_id)) if cart_item_id else None
    cart_bucket = _menu_bucket((cart_item or {}).get("tags") or "")
    preferred_buckets = _COMPLEMENT_ORDER.get(cart_bucket, ["drink", "dessert", "side"])

    candidates = [i for i in menu_items if str(i.get("menu_item_id")) not in exclude_ids]
    bucketed: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        bucketed.setdefault(_menu_bucket(item.get("tags") or ""), []).append(item)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _take(items: list[dict[str, Any]]) -> None:
        for item in items:
            item_id = str(item.get("menu_item_id"))
            if item_id in seen:
                continue
            seen.add(item_id)
            ordered.append(item)

    for bucket in preferred_buckets:
        _take(bucketed.get(bucket, []))
    _take(candidates)
    return ordered[:limit]


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


def _current_restaurant_name(state: CravrState) -> str | None:
    """Same fallback chain as `_current_restaurant_id`, for the name."""
    cart_state = state.get("cart_state") or {}
    restaurant = cart_state.get("restaurant") or {}
    if isinstance(restaurant, dict) and restaurant.get("name"):
        return restaurant.get("name")
    chosen = (state.get("recommendation_state") or {}).get("last_recommendations") or []
    if len(chosen) == 1:
        return chosen[0].get("restaurant_name")
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

    # Swiggy's update_food_cart is documented as additive ("Add items to
    # cart"), but a live run showed the opposite: two consecutive calls
    # (same restaurant/address, one new item each) each came back reporting
    # only the item just sent -- "Items (1): ... " both times, with item
    # total unchanged (~₹400, a single item's price) -- meaning the first
    # item was dropped rather than accumulated. There's no cartId param to
    # target an existing cart (confirmed against the docs: cart identity is
    # resolved server-side from restaurantId/addressId, not a client-held
    # id), so the safest fix from our side is to never rely on server-side
    # accumulation: always fetch what's already in the cart, merge the new
    # item into it (bumping quantity if it's already there), and send the
    # FULL desired cart contents every time.
    existing_cart = food_mcp.get_cart(address_id)
    existing_items = (
        _parse_existing_cart_items(existing_cart.get("message") or "")
        if existing_cart.get("status") in ("text", "success", "empty")
        else []
    )
    debug("cart_agent", f"existing cart before add: status={existing_cart.get('status')!r} "
          f"parsed_items={existing_items!r}")

    merged: dict[str, int] = {str(item["menu_item_id"]): item.get("quantity", 1) for item in existing_items}
    merged[str(menu_item_id)] = merged.get(str(menu_item_id), 0) + quantity
    cart_items_payload = [{"menu_item_id": mid, "quantity": qty} for mid, qty in merged.items()]
    debug("cart_agent", f"sending merged cartItems={cart_items_payload!r}")

    result = food_mcp.update_cart(
        restaurant_id, address_id,
        cart_items_payload,
        restaurant_name=dish.get("restaurant_name"),
    )
    debug("cart_agent", f"update_cart -> {result!r}")
    if result.get("status") != "success":
        content = f"Couldn't add that to your cart: {result.get('message', 'unknown error')}"
        return _cart_reply(f"{lead_in} {content}".strip())

    # update_cart succeeding doesn't guarantee the item actually landed --
    # a real response was 'Cart updated.\nCart is empty.' -- so its own
    # message is relayed rather than an independent "Added X" claim, UNLESS
    # it's just going to repeat what get_cart is about to show next (avoid
    # duplicate lines); get_cart right after is the real source of truth.
    # Two shapes look like a redundant full dump: "Cart is empty." (empty
    # cart) and an "Items (N): ... TO PAY: ..." itemized+pricing dump
    # (non-empty cart) -- a real observed update_cart response was 'Cart
    # updated.\nRestaurant: X\nItems (1):\n  - Dish — ₹260 (ID: 123)\n\n
    # Item total: ₹400\n...\nTO PAY: ₹398', which get_cart repeats verbatim
    # right after. Only relay update_cart's own message when it's neither --
    # e.g. a genuinely different short status line.
    update_message = result.get("message")
    looks_like_full_dump = bool(update_message) and (
        "cart is empty" in update_message.lower()
        or "items (" in update_message.lower()
        or "to pay" in update_message.lower()
    )
    if update_message and not looks_like_full_dump:
        prefix = f"{lead_in} {update_message}".strip()
    else:
        prefix = lead_in
    return _show_cart_with_next_steps(address_id, lead_in=prefix)


def _offer_dishes(lead_in: str, batch: list[dict[str, Any]], filters: dict[str, Any]) -> dict:
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
    any node actually runs."""
    from ..recommend.RecommendAgent import ONLY_MATCH_OPTIONS, _build_recommendation_text, _dish_pick_options

    body = _build_recommendation_text(batch)
    content = f"{lead_in}\n{body}".strip() if lead_in else body
    options = ONLY_MATCH_OPTIONS if len(batch) == 1 else _dish_pick_options(batch)
    return {
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

    brief = state.get("recommendation_brief") or {}
    if not brief:
        return _cart_reply("Cart's cleared. Tell me what you're craving and I'll find some other restaurants.")

    from ..recommend.RecommendAgent import fetch_recommendations  # deferred -- see _offer_dishes

    exclude = {str(restaurant_id)} if restaurant_id else set()
    filters = dict(brief.get("filters") or {})
    ranked = fetch_recommendations(brief, address_id=address_id, exclude_restaurant_ids=exclude)
    if not ranked:
        return _cart_reply(
            "Cart's cleared, but I couldn't find another restaurant with the same kind of food "
            "right now -- tell me what to change and I'll look again."
        )
    return _offer_dishes("Cart's cleared. Other restaurants with what you were after:", ranked, filters)


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
        return add_dish_to_cart(chosen[0], address_id, quantity=arg or 1)

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
