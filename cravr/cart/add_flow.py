"""The single call site for "how a dish gets added to the cart" -- used by
both CartAgent's own "add" action and RecommendAgent's pick-to-cart
handoff, so the two can't drift apart. See guards.py for why some items
get refused outright and why others get an "attempt, then verify" path
instead.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from .. import food_mcp
from ..debug_utils import debug
from .display import _cart_reply, _show_cart_with_next_steps
from .guards import _customization_refusal, _requires_customization
from .parsing import _parse_existing_cart_items

# Sentinel for "none of these" on an addon question -- same
# fixed-value-independent-of-label convention as options.OTHER_VALUE and
# routing.CART_ADD_MORE_VALUE.
_NONE_ADDON_VALUE = "__no_addon__"

# Sanity ceiling only -- NOT a UX cap. Showing every group/choice at once
# is the whole point of _ask_for_all_addons (see its docstring); this just
# stops a pathological response (a parser bug, a menu with hundreds of
# choices) from producing an unusably huge interrupt payload.
_MAX_TOTAL_ADDON_CHOICES = 40

# Whole-reply phrases that mean "no add-ons at all" -- checked against the
# ENTIRE reply before any per-token matching, so "none" or "no thanks"
# never gets misread as a (nonexistent) choice named "none".
_NONE_REPLY_PHRASES = {
    "none", "no", "nothing", "no thanks", "no thank you", "skip",
    "nope", "no addons", "no addon", "not now", "none of these",
}


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


def _addon_question_and_options(
    item_name: str, choices: list[dict[str, Any]], is_followup: bool,
) -> tuple[str, list[dict[str, str]]]:
    """The question text + tap options for one round of
    `_ask_for_all_addons` -- `choices` is whatever's still unanswered
    going into this round (every choice on round 1, fewer on a followup).
    Numbering restarts at 1 each round; that's fine because
    `_match_addon_reply` is always handed exactly THIS round's `choices`,
    so a position always means "this round's Nth listed item", never a
    stale one from an earlier round."""
    if is_followup:
        lines = [
            f'Anything else for {item_name}? (numbers, names, or comma-separated), or say "done":',
            "",
        ]
        none_label = "Nothing else"
    else:
        lines = [
            f'{item_name} has these add-ons -- reply with any you want '
            '(numbers, names, or comma-separated), or say "none":',
            "",
        ]
        none_label = "None of these"

    options: list[dict[str, str]] = []
    current_group_name: str | None = None
    for i, choice in enumerate(choices, start=1):
        if choice["group_name"] != current_group_name:
            current_group_name = choice["group_name"]
            lines.append(f"{current_group_name}:")
        price_bit = f" (+₹{choice['price_inr']:.0f})" if choice.get("price_inr") else ""
        lines.append(f"  {i}. {choice['name']}{price_bit}")
        options.append({"value": str(choice["id"]), "label": f"{choice['name']}{price_bit}"})
    options.append({"value": _NONE_ADDON_VALUE, "label": none_label})
    return "\n".join(lines), options


def _ask_for_all_addons(item_name: str, addon_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask about every addon group, starting from ONE consolidated
    `interrupt()` that lists all of them at once -- not one per group.

    Live evidence this replaces (logger.txt, "Spaghetti Aglio Olio [Veg]",
    4 real addon groups): the original group-by-group loop took 3 separate
    conversation turns to answer 3 groups, AND -- because it capped at a
    fixed group count -- never even asked about the 4th group ("Choose
    Your Non Veg Toppings") before sending the add, which still came back
    'Cart updated.\nCart is empty.' from the real update_food_cart.

    The FIRST version of this "all at once" question (a single
    interrupt(), no loop) fixed the missing-4th-group problem but broke
    something else, also confirmed live: a real user's natural move was to
    TAP one button, same as every other options menu in this app -- and
    chainlit's Action buttons only ever return ONE value per click, so a
    single tap could only ever pick from ONE of the groups shown, with the
    rest implicitly left unanswered ("can use only 1" -- accurate
    complaint, not a misunderstanding). Free-text multi-select ("1, 3" or
    comma-separated names) still worked, but nothing about the UI
    suggested that was necessary over just tapping.

    This version keeps the single big question up front (every group
    still shown at once, so nothing requires several turns just to be
    SEEN), but treats one round's reply as "at least this," not "only
    this": whatever groups the reply didn't resolve go into a shorter
    FOLLOWUP question (only the still-open groups, with a "done"/"nothing
    else" way out), repeating until either every group has a real answer
    or the user says they're done. A single free-text reply naming
    several groups at once (still fully supported) finishes in one round,
    same as before; a user who taps is walked through the remaining groups
    a tap at a time instead of being capped at whichever one group they
    happened to hit first. Bounded by the number of distinct groups --
    each round that matches anything resolves at least one whole group
    (single-choice-or-none per group, same simplification as before; see
    _match_addon_reply), so `remaining` strictly shrinks every successful
    round, and an unproductive reply (unmatched text, or "none"/"done")
    ends the loop immediately rather than re-asking.

    Returns FoodCartAddon-shaped entries: [{"id", "name", "group_id",
    "quantity", "price"}].
    """
    indexed_choices: list[dict[str, Any]] = []
    for group in addon_groups:
        group_name = group.get("group_name") or "add-on"
        for choice in group.get("choices") or []:
            if not (choice.get("id") and choice.get("name")):
                continue
            indexed_choices.append({**choice, "group_name": group_name})
            if len(indexed_choices) >= _MAX_TOTAL_ADDON_CHOICES:
                break
        if len(indexed_choices) >= _MAX_TOTAL_ADDON_CHOICES:
            break
    if not indexed_choices:
        return []

    selected: list[dict[str, Any]] = []
    picked_ids: set[str] = set()
    remaining = indexed_choices
    group_count = len({c.get("group_id") for c in indexed_choices})
    round_number = 0

    # Belt-and-suspenders bound -- the break-on-no-match below already
    # guarantees termination, this just caps it defensively at the number
    # of groups (each successful round removes at least one whole group).
    while remaining and round_number < max(group_count, 1):
        round_number += 1
        question, options = _addon_question_and_options(item_name, remaining, round_number > 1)
        raw_answer = interrupt({
            "question": question,
            "options": options,
            "agent": "cart_agent",
            "field": f"addon_groups:round{round_number}",
        })
        debug("cart_agent", f"addon selection round {round_number} raw_answer={raw_answer!r}")

        reply_matches = _match_addon_reply(str(raw_answer), remaining)
        if not reply_matches:
            # A genuine "none"/"done", or free text that matched nothing
            # in THIS round's list -- either way, stop asking rather than
            # loop on an unproductive reply.
            break

        newly_answered_group_ids: set[Any] = set()
        for choice in reply_matches:
            if choice["id"] not in picked_ids:
                picked_ids.add(choice["id"])
                selected.append(choice)
            newly_answered_group_ids.add(choice.get("group_id"))
        remaining = [c for c in remaining if c.get("group_id") not in newly_answered_group_ids]

    return selected

def _match_addon_reply(raw_answer: str, indexed_choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve one free-text (or tapped) reply against every choice
    `_ask_for_all_addons` listed, split on commas/"and"/"&" so one reply
    can name several choices across different groups at once.

    Per token, most-certain match first: an exact choice id (a button tap
    always sends one, verbatim) -- then a 1-based position in the numbered
    list the question showed -- then a case-insensitive exact name match --
    then a loose substring match (so "mozzarella" still finds "Extra
    Mozzarella Cheese"). A token matching nothing is dropped rather than
    guessed at -- never forwarded to Swiggy as if it were a real id. A
    whole reply that's a bare "none"/"no"/"skip" (see _NONE_REPLY_PHRASES)
    short-circuits to [] before any per-token matching, so it can't be
    misread as a search for a choice literally named "none". Dedupes by
    id, keeping first-mention order.
    """
    text = (raw_answer or "").strip()
    if not text:
        return []
    normalized_whole = text.lower().rstrip(".!")
    if normalized_whole == _NONE_ADDON_VALUE or normalized_whole in _NONE_REPLY_PHRASES:
        return []

    tokens = [t.strip() for t in re.split(r",|\band\b|&", text, flags=re.IGNORECASE) if t.strip()]
    matched: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for token in tokens:
        token_norm = token.lower().rstrip(".!")
        if not token_norm or token_norm in _NONE_REPLY_PHRASES:
            continue

        choice = next((c for c in indexed_choices if str(c["id"]) == token), None)
        if not choice and token_norm.isdigit():
            position = int(token_norm)
            if 1 <= position <= len(indexed_choices):
                choice = indexed_choices[position - 1]
        if not choice:
            choice = next((c for c in indexed_choices if c["name"].strip().lower() == token_norm), None)
        if not choice:
            choice = next(
                (c for c in indexed_choices
                 if token_norm in c["name"].lower() or c["name"].lower() in token_norm),
                None,
            )
        if not choice or str(choice["id"]) in seen_ids:
            continue
        seen_ids.add(str(choice["id"]))
        matched.append({
            "id": choice["id"],
            "name": choice["name"],
            "group_id": choice.get("group_id"),
            "quantity": 1,
            "price": choice.get("price_inr", 0),
        })
    return matched


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
            selected_addons = _ask_for_all_addons(str(dish.get("item_name") or ""), addon_groups)
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
            # `error_codes` only shows up when Swiggy's own structured_content
            # said `successful: False` (see food_mcp.cart.update_cart) -- a
            # real, specific rejection, not the old "reported success but the
            # item silently never landed" mystery. Say what Swiggy actually
            # said instead of the generic "can't set that yet" refusal,
            # which is actively misleading here: Cravr DID try, and DID send
            # the add-ons the user picked -- Swiggy turned them down.
            error_codes = result.get("error_codes")
            if error_codes:
                reason = result.get("message") or "one or more add-ons weren't accepted"
                debug("cart_agent", f"addon-only attempt: explicit rejection {error_codes!r} -- {reason!r}")
                return _cart_reply(
                    f"{dish.get('item_name') or 'That item'} couldn't be added -- Swiggy rejected the "
                    f"add-on(s) you picked ({', '.join(error_codes)}): {reason} Try again without those "
                    f"specific add-ons, or add it with add-ons directly in the Swiggy app."
                )
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
