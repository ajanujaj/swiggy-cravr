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

    Logs (tag `addon-diag`) exactly what Swiggy said about this item's
    customisation -- every group with its min/max ("REQUIRED" when min >= 1),
    and whether the item also has variants -- so a later INVALID_ADDON can be
    traced to a cause from logger.txt alone.
    """
    try:
        probe = food_mcp.search_menu_items(item_name, address_id, restaurant_id=str(restaurant_id))
    except Exception as exc:  # noqa: BLE001 -- must never break the add path
        debug("cart_agent", f"addon-diag lookup FAILED for {item_name!r}: {exc}")
        return []
    detailed = probe.get("detailed_items") if isinstance(probe, dict) else None
    if not isinstance(detailed, dict):
        debug("cart_agent", f"addon-diag lookup for {item_name!r}: no detailed_items in response "
              f"(items={len((probe or {}).get('items') or [])})")
        return []
    match = detailed.get(str(menu_item_id))
    if not isinstance(match, dict):
        debug("cart_agent", f"addon-diag lookup for {item_name!r}: id {menu_item_id!r} not among "
              f"detailed ids {list(detailed)!r}")
        return []
    groups = match.get("addon_groups")
    groups = groups if isinstance(groups, list) else []
    debug(
        "cart_agent",
        f"addon-diag item {item_name!r} id={menu_item_id} restaurant={restaurant_id} "
        f"has_variants={match.get('has_variants')!r} variant_groups="
        f"{[(v.get('group_id'), v.get('group_name'), [c.get('name') for c in v.get('choices') or []]) for v in match.get('variant_groups') or []]!r}",
    )
    for g in groups:
        required = isinstance(g.get("min_addons"), int) and g["min_addons"] >= 1
        debug(
            "cart_agent",
            f"addon-diag   group id={g.get('group_id')} name={g.get('group_name')!r} "
            f"min={g.get('min_addons')!r} max={g.get('max_addons')!r} choices={len(g.get('choices') or [])}"
            f"{'  <-- REQUIRED' if required else ''}",
        )
    return groups


def _required_groups(indexed_choices: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """{group_id: {"name", "min", "max"}} for every group Swiggy marks
    mandatory (`minAddons` >= 1). Live (logger.txt 2026-09-26, 'Gourmet Veg
    Caesar Salad'): 'Dressing & Accompaniments' and 'Accompaniments' both
    had minAddons=1, and update_food_cart answered INVALID_ADDON to a pick
    that skipped them -- and even to the item added alone, because a
    mandatory group can't be left unanswered."""
    required: dict[Any, dict[str, Any]] = {}
    for c in indexed_choices:
        lo = c.get("min_addons")
        if isinstance(lo, int) and lo >= 1:
            required[c.get("group_id")] = {"name": c["group_name"], "min": lo, "max": c.get("max_addons")}
    return required


def _unmet_required_groups(
    selected: list[dict[str, Any]], indexed_choices: list[dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
    """Required groups (see `_required_groups`) with fewer picks than their
    `minAddons`."""
    counts: dict[Any, int] = {}
    for entry in selected:
        counts[entry.get("group_id")] = counts.get(entry.get("group_id"), 0) + 1
    return {
        gid: info for gid, info in _required_groups(indexed_choices).items()
        if counts.get(gid, 0) < info["min"]
    }


def _addon_question_and_options(
    item_name: str, choices: list[dict[str, Any]], required_only: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """The question text + selectable options for the ONE add-on question.
    Every choice of every group is listed and numbered continuously, so a
    typed reply ("1, 4, 9" / names) and a UI selection (choice ids) both
    resolve against the same list in `_match_addon_reply`. Each option
    carries its `group` so the chat UI can render group headings, and a
    group with a selection cap says so ("choose up to N")."""
    if required_only:
        lines = [
            f"{item_name} needs you to choose from these before it can be added "
            "(numbers, names, or comma-separated):",
            "",
        ]
    else:
        lines = [
            f"{item_name} has these add-ons -- select as many as you like "
            '(numbers, names, or comma-separated), or say "none". '
            "Groups marked REQUIRED must have a choice:",
            "",
        ]
    options: list[dict[str, str]] = []
    current_group_id: Any = object()
    for i, choice in enumerate(choices, start=1):
        if choice.get("group_id") != current_group_id:
            current_group_id = choice.get("group_id")
            cap = choice.get("max_addons")
            lo = choice.get("min_addons")
            if isinstance(lo, int) and lo >= 1:
                cap_bit = f" (REQUIRED -- choose {lo}" + (f" to {cap}" if isinstance(cap, int) and cap > lo else "") + ")"
            else:
                cap_bit = f" (choose up to {cap})" if isinstance(cap, int) and cap > 0 else ""
            lines.append(f"{choice['group_name']}{cap_bit}:")
        price_bit = f" (+₹{choice['price_inr']:.0f})" if choice.get("price_inr") else ""
        lines.append(f"  {i}. {choice['name']}{price_bit}")
        options.append({
            "value": str(choice["id"]),
            "label": f"{choice['name']}{price_bit}",
            "group": str(choice["group_name"]),
            "required": "1" if isinstance(choice.get("min_addons"), int) and choice["min_addons"] >= 1 else "",
        })
    if not required_only:
        options.append({"value": _NONE_ADDON_VALUE, "label": "None of these"})
    return "\n".join(lines), options


def _enforce_group_limits(
    selected: list[dict[str, Any]], indexed_choices: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Honour each group's `maxAddons` (search_menu reference: per-group
    `minAddons` / `maxAddons` / `maxFreeAddons`). Extra picks beyond a
    group's max are dropped -- first-mentioned wins -- rather than sent to
    Swiggy to be rejected. A missing/non-positive max means "no cap known"."""
    caps: dict[Any, int] = {}
    for c in indexed_choices:
        cap = c.get("max_addons")
        if isinstance(cap, int) and cap > 0:
            caps[c.get("group_id")] = cap
    kept: list[dict[str, Any]] = []
    counts: dict[Any, int] = {}
    for entry in selected:
        gid = entry.get("group_id")
        if gid in caps and counts.get(gid, 0) >= caps[gid]:
            debug("cart_agent", f"addon {entry.get('name')!r} dropped: group {gid} already at max {caps[gid]}")
            continue
        counts[gid] = counts.get(gid, 0) + 1
        kept.append(entry)
    return kept


def _ask_for_all_addons(item_name: str, addon_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask about every addon group in ONE multi-select question.

    History: the first version asked group-by-group, the second listed
    every group but let a button tap answer only ONE group (a tap returns
    a single value), so users could only ever pick one add-on ("can use
    only 1"). This version is a real multi-select: the interrupt payload
    carries `multi_select=True` and the chat UI (UI/components.py
    `send_multiselect_message`) renders toggle buttons plus a Done button,
    then resumes with ALL chosen ids comma-joined. Typed replies ("1, 4,
    9", names) still work through the same `_match_addon_reply` -- and any
    number of choices, across any groups, can be picked in one reply.
    Per-group `maxAddons` is enforced afterwards by `_enforce_group_limits`.

    Returns FoodCartAddon-shaped entries: [{"id", "name", "group_id",
    "quantity", "price"}].
    """
    indexed_choices: list[dict[str, Any]] = []
    # Mandatory groups first: the _MAX_TOTAL_ADDON_CHOICES sanity cap below
    # must never be what cuts off a group Swiggy insists on.
    addon_groups = sorted(
        addon_groups,
        key=lambda g: 0 if isinstance(g.get("min_addons"), int) and g["min_addons"] >= 1 else 1,
    )
    for group in addon_groups:
        group_name = group.get("group_name") or "add-on"
        for choice in group.get("choices") or []:
            if not (choice.get("id") and choice.get("name")):
                continue
            indexed_choices.append({
                **choice, "group_name": group_name,
                "min_addons": group.get("min_addons"), "max_addons": group.get("max_addons"),
            })
            if len(indexed_choices) >= _MAX_TOTAL_ADDON_CHOICES:
                break
        if len(indexed_choices) >= _MAX_TOTAL_ADDON_CHOICES:
            break
    if not indexed_choices:
        return []

    question, options = _addon_question_and_options(item_name, indexed_choices)
    raw_answer = interrupt({
        "question": question,
        "options": options,
        "agent": "cart_agent",
        "field": "addon_groups",
        "multi_select": True,
    })
    debug("cart_agent", f"addon-diag selection raw_answer={raw_answer!r}")
    matches = _match_addon_reply(str(raw_answer), indexed_choices)
    selected = _enforce_group_limits(matches, indexed_choices)

    # Mandatory groups (minAddons >= 1) can't be skipped: Swiggy rejects the
    # whole add with INVALID_ADDON otherwise. Ask again about JUST those, up
    # to twice, instead of sending a request known to fail.
    for attempt in range(1, 3):
        unmet = _unmet_required_groups(selected, indexed_choices)
        if not unmet:
            break
        debug("cart_agent", f"addon-diag REQUIRED group(s) unanswered (round {attempt}): "
              f"{[(gid, info['name'], 'min=' + str(info['min'])) for gid, info in unmet.items()]!r}")
        pending = [c for c in indexed_choices if c.get("group_id") in unmet]
        question, options = _addon_question_and_options(item_name, pending, required_only=True)
        raw_answer = interrupt({
            "question": question, "options": options, "agent": "cart_agent",
            "field": f"addon_groups:required{attempt}", "multi_select": True,
        })
        debug("cart_agent", f"addon-diag required-group answer raw_answer={raw_answer!r}")
        more = _match_addon_reply(str(raw_answer), pending)
        selected = _enforce_group_limits(selected + [m for m in more if m["id"] not in {x["id"] for x in selected}],
                                         indexed_choices)
    else:
        unmet = _unmet_required_groups(selected, indexed_choices)
        if unmet:
            debug("cart_agent", f"addon-diag REQUIRED group(s) STILL unanswered -- add will likely be "
                  f"rejected: {[info['name'] for info in unmet.values()]!r}")

    # Per-group summary of what is about to be sent.
    for gid in dict.fromkeys(c.get("group_id") for c in indexed_choices):
        picked = [e["name"] for e in selected if e.get("group_id") == gid]
        sample = next(c for c in indexed_choices if c.get("group_id") == gid)
        debug("cart_agent", f"addon-diag pick group={gid} {sample['group_name']!r} "
              f"min={sample.get('min_addons')!r} max={sample.get('max_addons')!r} picked={picked!r}")
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


def _map_to_valid_addons(
    selected: list[dict[str, Any]], valid_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each pick with Swiggy's own `valid_addons` entry for it: returns
    `[{"id", "name", "group_id", "price_inr", "raw"}]` where `raw` is the
    exact choice dict from the cart. Picks Swiggy doesn't list as valid for
    this cart item are dropped; each group's `max_addons` is honoured."""
    by_choice: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for group in valid_groups:
        for choice in group.get("choices") or []:
            by_choice.setdefault(str(choice["id"]), (group, choice))
    mapped: list[dict[str, Any]] = []
    counts: dict[Any, int] = {}
    for pick in selected:
        hit = by_choice.get(str(pick.get("id")))
        if not hit:
            debug("cart_agent", f"addon-diag pick {pick.get('name')!r} (id {pick.get('id')}) is NOT in the "
                  "cart's valid_addons for this item")
            continue
        group, choice = hit
        cap = group.get("max_addons")
        if isinstance(cap, int) and cap > 0 and counts.get(group["group_id"], 0) >= cap:
            continue
        counts[group["group_id"]] = counts.get(group["group_id"], 0) + 1
        mapped.append({
            "id": choice["id"], "name": choice["name"],
            "group_id": choice.get("group_id") or group["group_id"],
            "price_inr": choice["price_inr"], "raw": choice.get("raw") or {},
        })
    return mapped


def _addon_shapes(mapped: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Candidate request shapes for the `addons` array, most-likely first.
    Swiggy's docs never spell the request shape out -- they say cart addons
    "extend the variant structure" (`id, name, group_id, variation_id, price`
    + `quantity`), and a cart variant is keyed `variation_id` -- and every
    shape used so far (`id` only) has been answered INVALID_ADDON, so each
    is tried in turn and the winner is logged. All numeric ids are sent as
    ints (that's what the cart's own `valid_addons` uses); `price` is left
    out where the cart holds it in paise, to avoid a rupee/paise mismatch."""
    def gid(m):  # ints, like the cart's own valid_addons
        return int(m["group_id"]) if str(m["group_id"]).isdigit() else m["group_id"]

    def cid(m):
        return int(m["id"]) if str(m["id"]).isdigit() else m["id"]

    return [
        ("A: variation_id+group_id+quantity",
         [{"variation_id": cid(m), "group_id": gid(m), "quantity": 1} for m in mapped]),
        ("B: id+group_id+quantity",
         [{"id": cid(m), "group_id": gid(m), "quantity": 1} for m in mapped]),
        ("C: variation_id+id+group_id+name+price(paise)+quantity",
         [{"variation_id": cid(m), "id": cid(m), "group_id": gid(m), "name": m["name"],
           "price": m["raw"].get("price", int(round(m["price_inr"] * 100))), "quantity": 1}
          for m in mapped]),
        ("D: cart's own valid_addons choice verbatim + group_id + quantity",
         [{**m["raw"], "group_id": gid(m), "quantity": 1} for m in mapped]),
    ]


def _retry_addons_via_valid_addons(
    build_payload, restaurant_id: str, address_id: str, restaurant_name: str | None,
    menu_item_id: str, selected: list[dict[str, Any]], failed_result: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Swiggy's documented add-on flow, used after a one-shot add with
    add-ons was rejected (INVALID_ADDON): (1) add the item alone, (2) read
    `valid_addons` for it off the cart, (3) resend with add-ons taken from
    that list -- trying each candidate request shape in `_addon_shapes` until
    one is accepted. Every step is logged (`addon-diag`). Returns
    `(result, note)`. If step 1 fails, the original failure is returned
    untouched. If steps 2/3 can't get the add-ons in, the plain item stays in
    the cart (result = step 1's success) and `note` says honestly that the
    add-ons were not added -- better than losing the dish."""
    base = food_mcp.update_cart(restaurant_id, address_id, build_payload([]), restaurant_name=restaurant_name)
    debug("cart_agent", f"addon-diag retry step 1 (item alone) -> status={base.get('status')!r} "
          f"error_codes={base.get('error_codes')!r}")
    if base.get("status") != "success":
        debug("cart_agent", "addon-diag the item is rejected EVEN WITHOUT add-ons -- points at a mandatory "
              "(minAddons>=1) group, a variant that must be chosen, or an unavailable item/restaurant, not at "
              "the picked add-ons")
        return failed_result, ""

    detail = food_mcp.get_cart_item_details(address_id).get(str(menu_item_id)) or {}
    valid = detail.get("valid_addons") or []
    mapped = _map_to_valid_addons(selected, valid)
    debug("cart_agent", f"addon-diag retry step 2: {len(selected)} picked, {len(mapped)} valid per cart; "
          f"cart variants={[(v.get('name'), v.get('variation_id')) for v in detail.get('variants') or []]!r}")
    if not mapped:
        return base, (
            "Swiggy didn't list those add-ons as valid for this dish, so I added it without them."
        )

    last_codes = None
    for label, addons in _addon_shapes(mapped):
        debug("cart_agent", f"addon-diag retry step 3 trying shape {label}: {addons!r}")
        retry = food_mcp.update_cart(restaurant_id, address_id, build_payload(addons), restaurant_name=restaurant_name)
        if retry.get("status") == "success":
            debug("cart_agent", f"addon-diag SUCCESS -- Swiggy accepted shape {label}")
            return retry, ""
        last_codes = retry.get("error_codes")
        debug("cart_agent", f"addon-diag shape {label} rejected: error_codes={last_codes!r} "
              f"message={retry.get('message')!r}")
    debug("cart_agent", "addon-diag NO request shape was accepted -- see the attempts above")
    codes = ", ".join(last_codes or []) or "rejected"
    return base, (
        f"Swiggy rejected the add-ons ({codes}), so I added the dish without them -- "
        "you can add them in the Swiggy app."
    )


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
    def _build_payload(target_addons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items = []
        for mid, qty in merged.items():
            entry: dict[str, Any] = {"menu_item_id": mid, "quantity": qty}
            if attempt_addon_only and mid == str(menu_item_id):
                # Real user selections when the addon question got any,
                # else the explicit-empty-selection fallback -- see the
                # attempt_addon_only comment above.
                entry["addons"] = target_addons
            items.append(entry)
        return items

    cart_items_payload = _build_payload(selected_addons)
    debug("cart_agent", f"sending merged cartItems={cart_items_payload!r}")
    if attempt_addon_only:
        import json as _json
        debug(
            "cart_agent",
            "addon-diag first attempt: wire JSON=" + _json.dumps(cart_items_payload, default=str)
            + " | id types=" + repr(sorted({type(a.get("id")).__name__ for e in cart_items_payload
                                             for a in e.get("addons", [])}))
            + " | price units=rupees (search_menu), the cart's valid_addons uses paise",
        )

    result = food_mcp.update_cart(
        restaurant_id, address_id,
        cart_items_payload,
        restaurant_name=dish.get("restaurant_name"),
    )
    debug("cart_agent", f"update_cart -> {result!r}")

    # Live evidence (logger.txt 2026-09-26, 'Greek Falafel Salad', addon
    # 'Sun Dried Tomatoes Topping'): sending the picked add-ons in the same
    # call that adds the item came back INVALID_ADDON ("Restaurant may have
    # removed the item(s) from their menu") with the cart left empty --
    # even though every id/group pair was exactly what search_menu returned.
    # Swiggy's docs describe a two-step flow instead: add the item first,
    # read `valid_addons` off the cart, then add add-ons FROM that list with
    # the group/choice ids paired exactly as returned. So on that specific
    # rejection, fall back to the documented flow before giving up.
    addon_note = ""
    if (
        result.get("status") != "success" and attempt_addon_only and selected_addons
        and "INVALID_ADDON" in (result.get("error_codes") or [])
    ):
        result, addon_note = _retry_addons_via_valid_addons(
            _build_payload, restaurant_id, address_id, dish.get("restaurant_name"),
            str(menu_item_id), selected_addons, result,
        )
        if addon_note:
            lead_in = f"{lead_in} {addon_note}".strip()
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
