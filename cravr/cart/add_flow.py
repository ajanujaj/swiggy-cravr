"""The single call site for "how a dish gets added to the cart" -- used by
both CartAgent's own "add" action and RecommendAgent's pick-to-cart
handoff, so the two can't drift apart. See guards.py for why some items
get refused outright and why others get an "attempt, then verify" path
instead.
"""

from __future__ import annotations

from typing import Any

from .. import food_mcp
from ..debug_utils import debug
from .display import _cart_reply, _show_cart_with_next_steps
from .guards import _customization_refusal, _requires_customization
from .parsing import _parse_existing_cart_items


def add_dish_to_cart(
    dish: dict[str, Any], address_id: str, quantity: int = 1, lead_in: str = "",
    known_restaurant_id: str | None = None, known_restaurant_name: str | None = None,
) -> dict:
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

    # Live evidence (3 separate real CRAVR_DEBUG occurrences, 2 restaurants:
    # Jamie Oliver Kitchen's "Mediterranean Veg Pizza [11 inches]", Mister
    # Edamame's "Manchow Soup" and "Sichuan Hot & Sour Soup" -- all tagged
    # "has addons"/"has variants" in search_menu/get_restaurant_menu's own
    # `tags` text): merging a has-addons/has-variants item into a
    # non-empty cart made the real update_food_cart respond
    # 'Cart updated.\nCart is empty.' -- the WHOLE update rejected, not
    # just that item -- and Cravr then falsely told the user it worked.
    # Cravr has no UI to collect the required variant/add-on choice, so
    # refuse before sending rather than attempt-and-lie. (Only checked for
    # the item being ADDED here -- an item already sitting in the real
    # cart already went through Swiggy's own validation once and doesn't
    # need re-checking.)
    # Live evidence (Domino's Pizza item added while the cart already held
    # a Mister Edamame item, addon-free on both sides): merging an item
    # from a DIFFERENT restaurant into a non-empty cart made the real
    # update_food_cart respond 'Cart updated.\nCart is empty.' -- and this
    # time it didn't just fail to add the new item, it wiped the
    # PRE-EXISTING item too (get_food_cart afterward showed status='empty'
    # item_count=0). Swiggy carts are single-restaurant; refuse before
    # sending rather than risk destroying what's already there.
    if (
        known_restaurant_id
        and str(known_restaurant_id) != str(restaurant_id)
    ):
        existing_name = known_restaurant_name or "your current restaurant"
        content = (
            f"Your cart already has items from {existing_name}, and Swiggy carts can only hold one "
            f"restaurant at a time -- so I haven't added {dish.get('item_name', 'that item')} from "
            f"{dish.get('restaurant_name') or 'this restaurant'}. Use \"Browse other restaurants\" from "
            "your cart to clear it and start fresh here, or pick something from your current restaurant instead."
        )
        # Same lesson as the addon guard below: lead_in assumes success
        # ("Great -- X it is."), so it is dropped rather than prefixed.
        return _cart_reply(content)

    # Set below when this item is addon-only (no variant/size choice) --
    # threads through to the merged cart_items_payload build and gets a
    # dedicated post-add verification, instead of an unconditional refusal.
    attempt_addon_only = False

    if _requires_customization(dish.get("tags") or ""):
        tags_haystack = (dish.get("tags") or "").lower()
        needs_variant_choice = "has variants" in tags_haystack

        # Diagnostic-only probe (does not change the refusal below either
        # way, never blocks it). CONFIRMED live (logger.txt, "Falafel
        # Hummus Pita Pocket", restaurant 300897): a targeted
        # search_menu(item name, restaurantIdOfAddedItem=restaurant) call
        # DOES return the item's real addon groups with group/choice ids --
        # Swiggy's own docs were right about that. food_mcp.search_menu_items()
        # parses that shape itself (see _parse_detailed_search_menu_items)
        # and surfaces it under result["detailed_items"] when the
        # broad-format parser finds nothing; log the parsed group/choice
        # names here instead of the raw dict so this stays readable as more
        # items get tested. No min/maxAddons-style "mandatory" flag is
        # exposed in this response shape -- only name/price/ids -- so
        # whether any of these groups are actually required is still
        # unknown from this data alone.
        try:
            probe = food_mcp.search_menu_items(
                str(dish.get("item_name") or ""), address_id, restaurant_id=str(restaurant_id),
            )
            detailed = probe.get("detailed_items") or {}
            match = detailed.get(str(dish.get("menu_item_id") or ""))
            if match:
                group_summary = [
                    f"{g['group_name']} ({len(g['choices'])} choices, group_id={g['group_id']})"
                    for g in match.get("addon_groups", [])
                ]
                debug(
                    "cart_agent",
                    f"customization probe for {dish.get('item_name')!r} (id={dish.get('menu_item_id')!r}) "
                    f"-> matched item with addon groups: {group_summary!r}",
                )
            else:
                debug(
                    "cart_agent",
                    f"customization probe for {dish.get('item_name')!r} (id={dish.get('menu_item_id')!r}) "
                    f"-> no matching item id in response; {len(detailed)} detailed item(s), "
                    f"{len(probe.get('items') or [])} broad-format item(s) found instead",
                )
        except Exception as exc:  # noqa: BLE001 -- diagnostic only, must never break the refusal path
            debug("cart_agent", f"customization probe FAILED: {exc}")

        # Deliberately drop `lead_in` here, not prefix it -- callers build
        # it assuming a successful add (RecommendAgent._reply_with_pick:
        # "Great -- X it is."), and prefixing that onto a refusal produced
        # a self-contradicting reply confirmed live: "Great -- The Veggie
        # Melt Pizza from Captain Grub (Rs675.0) it is. The Veggie Melt
        # Pizza needs a variant/add-on choice I can't set for you yet, so
        # I haven't added it..." -- say plainly it wasn't added instead.
        if needs_variant_choice:
            # A variant (size/base choice, e.g. "Choice of Cheese: With/
            # Without") changes what the item actually IS -- there's no safe
            # default to guess (the live response shape confirmed this
            # segment carries no "default" flag either), so this case still
            # refuses outright, unlike the addon-only case below.
            return _cart_reply(_customization_refusal(dish.get("item_name")))

        # Addon-only item (e.g. live-confirmed "Korean Honey Gochujang
        # Chicken Wings", tagged has-addons with one optional-looking
        # "Add on Sauce" group, no variants): every addon group actually
        # seen live so far (sauces, dressings, drinks, desserts, sides) reads
        # as an optional upsell, not a mandatory choice -- and the repeated
        # real failure pattern (update_food_cart silently replying 'Cart
        # updated.\nCart is empty.') is consistent with Swiggy's validation
        # wanting an explicit answer for these groups, not with every group
        # actually being required. UNTESTED hypothesis, first attempted here
        # live: try the add with an explicit empty addon selection
        # ("addons": []) instead of refusing outright. `attempt_addon_only`
        # threads this into the normal merge-and-send flow below, which
        # already re-fetches the cart afterward to verify -- if that
        # verification shows the item didn't actually land, this same
        # refusal is returned instead of a broken/misleading cart display.
        attempt_addon_only = True

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
    target_quantity = merged.get(str(menu_item_id), 0) + quantity
    merged[str(menu_item_id)] = target_quantity
    cart_items_payload = []
    for mid, qty in merged.items():
        entry: dict[str, Any] = {"menu_item_id": mid, "quantity": qty}
        if attempt_addon_only and mid == str(menu_item_id):
            # See the attempt_addon_only comment above -- explicit empty
            # selection instead of omitting the field entirely.
            entry["addons"] = []
        cart_items_payload.append(entry)
    debug("cart_agent", f"sending merged cartItems={cart_items_payload!r}")

    result = food_mcp.update_cart(
        restaurant_id, address_id,
        cart_items_payload,
        restaurant_name=dish.get("restaurant_name"),
    )
    debug("cart_agent", f"update_cart -> {result!r}")
    if result.get("status") != "success":
        if attempt_addon_only:
            debug("cart_agent", "addon-only attempt: update_cart call itself failed -- falling back to refusal")
            return _cart_reply(_customization_refusal(dish.get("item_name")))
        content = f"Couldn't add that to your cart: {result.get('message', 'unknown error')}"
        return _cart_reply(f"{lead_in} {content}".strip())

    if attempt_addon_only:
        # update_cart reporting "success" doesn't mean the item actually
        # landed -- the whole reason this path exists is that has-addons
        # items previously came back 'Cart updated.\nCart is empty.' while
        # still reporting success. Re-fetch and confirm the item is really
        # there (at the expected quantity) before trusting this attempt;
        # if not, fall back to the same honest refusal rather than show the
        # user a cart that silently dropped their item.
        verify_cart = food_mcp.get_cart(address_id)
        verify_items = (
            _parse_existing_cart_items(verify_cart.get("message") or "")
            if verify_cart.get("status") in ("text", "success", "empty")
            else []
        )
        landed = next(
            (i for i in verify_items if str(i.get("menu_item_id")) == str(menu_item_id)), None,
        )
        debug(
            "cart_agent",
            f"addon-only attempt verify -> landed={landed!r} target_quantity={target_quantity} "
            f"all_items={verify_items!r}",
        )
        if not landed or landed.get("quantity", 0) < target_quantity:
            return _cart_reply(_customization_refusal(dish.get("item_name")))
        debug("cart_agent", "addon-only attempt VERIFIED -- explicit empty addons selection worked live")

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
    outcome = _show_cart_with_next_steps(address_id, lead_in=prefix)
    # get_cart's real (text-mode) response never fills in `restaurant`
    # (food_mcp.get_cart hardcodes it to None off that path), so the
    # cross-restaurant guard above would never have anything to compare
    # against on the NEXT add unless we record it ourselves here, on our
    # own successful add.
    cart_state = outcome.get("cart_state") or {}
    if cart_state.get("status") in ("text", "success"):
        outcome["cart_state"] = {
            **cart_state,
            "restaurant": {"id": restaurant_id, "name": dish.get("restaurant_name")},
        }
    return outcome
