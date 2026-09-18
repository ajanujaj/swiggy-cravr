"""The single call site for "how a dish gets added to the cart" -- used by
both CartAgent's own "add" action and RecommendAgent's pick-to-cart
handoff, so the two can't drift apart. See guards.py for why some items
get refused outright and why others get an "attempt, then verify" path
instead.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from .. import food_mcp
from ..debug_utils import debug
from ..options import resolve_option_choice
from .display import _cart_reply, _show_cart_with_next_steps
from .guards import _customization_refusal, _requires_customization
from .parsing import _parse_existing_cart_items

# Sentinel for "none of these" on an addon-group question -- same
# fixed-value-independent-of-label convention as options.OTHER_VALUE and
# routing.CART_ADD_MORE_VALUE.
_NONE_ADDON_VALUE = "__no_addon__"

# An item can carry several addon groups (the live "Falafel Hummus Pita
# Pocket" fixture has four: Extra Toppings, Dressing Add On's, Make Your
# Own Meal, Desserts) -- capped so "want any add-ons?" doesn't turn into an
# interrogation before the item lands in the cart. Addons are optional
# upsells (see _requires_customization), never a required choice, so
# silently leaving the remaining groups unselected is the safe default,
# not a broken flow.
_MAX_ADDON_GROUPS_TO_ASK = 3
# Tappable-button budget per group, same ballpark as options.suggest_options'
# own default of 5 -- some groups run to 8-9 real choices.
_MAX_CHOICES_PER_GROUP = 6


def _fetch_addon_groups(
    item_name: str, menu_item_id: str, address_id: str, restaurant_id: str,
) -> list[dict[str, Any]]:
    """Real addon groups/choices for one dish, via the targeted search_menu
    lookup Swiggy's own docs point at ("use search_menu with the item name
    and restaurantId to get full customization details"). CONFIRMED live
    (logger.txt, "Falafel Hummus Pita Pocket", restaurant 300897) to return
    real group/choice ids in a third text shape food_mcp.search_menu_items
    parses separately and surfaces under `detailed_items` (see
    food_mcp.search._parse_detailed_search_menu_items).

    Returns [] whenever the lookup doesn't yield a usable match for this
    exact item id -- no result, an unfamiliar response shape, or a
    transient failure -- never raises. Callers must treat [] as "nothing
    usable" and fall back to the old empty-addons attempt, not block the
    add on it: this fetch is a best-effort upgrade over that fallback, not
    a new required step.
    """
    try:
        probe = food_mcp.search_menu_items(item_name, address_id, restaurant_id=str(restaurant_id))
    except Exception as exc:  # noqa: BLE001 -- must never break the add path
        debug("cart_agent", f"addon-group fetch FAILED for {item_name!r}: {exc}")
        return []
    detailed = probe.get("detailed_items") if isinstance(probe, dict) else None
    if not isinstance(detailed, dict):
        return []
    match = detailed.get(str(menu_item_id))
    if not isinstance(match, dict):
        return []
    groups = match.get("addon_groups")
    return groups if isinstance(groups, list) else []


def _ask_for_addon_choices(item_name: str, addon_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask the user, one addon group at a time, which optional extra (if
    any) they want -- via `interrupt()`/`Command(resume=...)`, the same
    pause-mid-graph mechanism intent_discovery_node uses for its own
    clarification loop (see graph.py, intentAgent.py). cart_agent is now
    the second (and, as of this change, only other) node that ever calls
    `interrupt()`.

    Options are built directly from the real group/choice ids
    `_fetch_addon_groups` returned -- deliberately NOT run through
    options.suggest_options()'s LLM paraphrasing. A paraphrased label could
    stop round-tripping to the exact choice id Swiggy expects, and these
    ids are the one thing in this flow that must never be guessed at.

    Every group is treated as single-choice-or-none: no minAddons/
    maxAddons-style "how many can I pick" field has been observed in any
    response shape seen so far (see _parse_detailed_search_menu_items), so
    this is the safe simplification -- worst case, a user who wanted two
    items from one group only gets to pick one, never a rejected/garbled
    request. Capped at `_MAX_ADDON_GROUPS_TO_ASK` groups /
    `_MAX_CHOICES_PER_GROUP` choices each; remaining groups/choices are
    silently left unselected rather than asked about, since an addon group
    is always optional (see _requires_customization) -- never worth
    refusing the whole add over.

    Returns FoodCartAddon-shaped entries ready to send: [{"id", "name",
    "group_id", "quantity", "price"}].
    """
    selected: list[dict[str, Any]] = []
    for group in addon_groups[:_MAX_ADDON_GROUPS_TO_ASK]:
        choices = [c for c in (group.get("choices") or []) if c.get("id") and c.get("name")]
        choices = choices[:_MAX_CHOICES_PER_GROUP]
        if not choices:
            continue
        group_name = group.get("group_name") or "add-on"

        options = [
            {
                "value": str(c["id"]),
                "label": f"{c['name']} (+₹{c['price_inr']:.0f})" if c.get("price_inr") else str(c["name"]),
            }
            for c in choices
        ]
        options.append({"value": _NONE_ADDON_VALUE, "label": "None of these"})

        raw_answer = interrupt({
            "question": f"{item_name} has a {group_name} add-on -- want any of these, or none?",
            "options": options,
            "agent": "cart_agent",
            "field": f"addon_group:{group.get('group_id')}",
        })
        debug("cart_agent", f"addon group {group_name!r} raw_answer={raw_answer!r}")

        # Try an EXACT id match first, before resolve_option_choice --
        # unlike intent_discovery_node's options (short text labels),
        # these option "value"s are themselves real numeric Swiggy ids.
        # resolve_option_choice treats an all-digit reply as a 1-based
        # *position* in the list, which happens to still fall through to
        # the raw value unchanged here (a real choice id is always far
        # larger than this short option list), but that's a coincidence of
        # id size, not a guarantee -- matching the id directly first
        # avoids depending on it.
        answer_text = str(raw_answer).strip()
        if answer_text == _NONE_ADDON_VALUE:
            continue
        chosen = next((c for c in choices if str(c.get("id")) == answer_text), None)
        if not chosen:
            resolved = resolve_option_choice(answer_text, options)
            if resolved == _NONE_ADDON_VALUE:
                continue
            chosen = next((c for c in choices if str(c.get("id")) == str(resolved)), None)
        if not chosen:
            # Free text the option menu didn't anticipate (e.g. the user
            # typed something instead of tapping) -- no safe id to send
            # Swiggy for it, so this group is skipped rather than guessed at.
            debug("cart_agent", f"addon group {group_name!r}: unresolved answer {raw_answer!r}, skipping")
            continue
        selected.append({
            "id": chosen["id"],
            "name": chosen["name"],
            "group_id": chosen.get("group_id") or group.get("group_id"),
            "quantity": 1,
            "price": chosen.get("price_inr", 0),
        })
    return selected


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
    # The addon entries actually sent for the item being added -- real
    # user choices when _fetch_addon_groups found usable groups and
    # _ask_for_addon_choices got answers for them, otherwise [] (the
    # original explicit-empty-selection guess, kept as the fallback for
    # whenever the targeted lookup comes up empty).
    selected_addons: list[dict[str, Any]] = []

    if _requires_customization(dish.get("tags") or ""):
        tags_haystack = (dish.get("tags") or "").lower()

        # A variant (size/base choice, e.g. "Choice of Cheese: With/
        # Without") changes what the item actually IS -- there's no safe
        # default to guess (the live detailed-lookup response shape
        # confirmed this segment carries no "default" flag either), so
        # this case still refuses outright, unlike the addon-only case
        # below. Checked and returned FIRST, before any addon-group
        # fetch/ask, so a variant-only item never triggers a pointless
        # (and confusing -- "want any add-ons?" right before a refusal)
        # round of interrupts for an add that's about to be declined anyway.
        if "has variants" in tags_haystack:
            # Deliberately drop `lead_in` here, not prefix it -- callers build
            # it assuming a successful add (RecommendAgent._reply_with_pick:
            # "Great -- X it is."), and prefixing that onto a refusal produced
            # a self-contradicting reply confirmed live: "Great -- The Veggie
            # Melt Pizza from Captain Grub (Rs675.0) it is. The Veggie Melt
            # Pizza needs a variant/add-on choice I can't set for you yet, so
            # I haven't added it..." -- say plainly it wasn't added instead.
            return _cart_reply(_customization_refusal(dish.get("item_name")))

        # Addon-only item (e.g. live-confirmed "Korean Honey Gochujang
        # Chicken Wings", tagged has-addons with one optional-looking
        # "Add on Sauce" group, no variants): every addon group actually
        # seen live so far (sauces, dressings, drinks, desserts, sides) reads
        # as an optional upsell, not a mandatory choice, and the repeated
        # real failure pattern (update_food_cart silently replying 'Cart
        # updated.\nCart is empty.') is consistent with Swiggy's validation
        # wanting an explicit answer for these groups, not with every group
        # actually being required.
        #
        # First try for the user's REAL preference: a targeted search_menu
        # lookup (_fetch_addon_groups) returns real group/choice ids when it
        # matches this item, and _ask_for_addon_choices asks about each one
        # via interrupt() -- so "extra cheese" actually means extra cheese,
        # not a silently-dropped preference. If that lookup comes up empty
        # (unfamiliar response shape, no match, a transient failure),
        # `selected_addons` stays [] and the item is still attempted with
        # an explicit empty selection, same as before this existed -- the
        # merge-and-send flow below always re-fetches the cart afterward to
        # verify either way; if verification shows the item didn't actually
        # land, the same honest refusal is returned rather than a broken/
        # misleading cart display.
        addon_groups = _fetch_addon_groups(
            str(dish.get("item_name") or ""), str(dish.get("menu_item_id") or ""),
            address_id, str(restaurant_id),
        )
        debug(
            "cart_agent",
            f"addon groups for {dish.get('item_name')!r}: "
            f"{[g.get('group_name') for g in addon_groups]!r}",
        )
        if addon_groups:
            selected_addons = _ask_for_addon_choices(str(dish.get("item_name") or ""), addon_groups)
            debug("cart_agent", f"user-selected addons for {dish.get('item_name')!r}: {selected_addons!r}")
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
            # Real user selections when _ask_for_addon_choices got any,
            # else the explicit-empty-selection fallback -- see the
            # attempt_addon_only comment above.
            entry["addons"] = selected_addons
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
