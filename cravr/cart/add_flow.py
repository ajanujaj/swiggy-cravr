from __future__ import annotations

import re
from typing import Any

from langgraph.types import interrupt

from .. import food_mcp
from ..debug_utils import debug
from .display import _cart_reply, _show_cart_with_next_steps
from .guards import _customization_refusal, _requires_customization
from .parsing import _existing_cart_items
from .routing import _CONFLICT_OPTIONS

_NONE_ADDON_VALUE = "__no_addon__"

_MAX_TOTAL_ADDON_CHOICES = 40

_NONE_REPLY_PHRASES = {
    "none", "no", "nothing", "no thanks", "no thank you", "skip",
    "nope", "no addons", "no addon", "not now", "none of these",
}


def _fetch_customization(
    item_name: str, menu_item_id: str, address_id: str, restaurant_id: str,
) -> dict[str, list[dict[str, Any]]]:
    empty: dict[str, list[dict[str, Any]]] = {"addon_groups": [], "variant_groups": []}
    try:
        probe = food_mcp.search_menu_items(item_name, address_id, restaurant_id=str(restaurant_id))
    except Exception as exc:  # noqa: BLE001 -- must never break the add path
        debug("cart_agent", f"addon-diag lookup FAILED for {item_name!r}: {exc}")
        return empty
    detailed = probe.get("detailed_items") if isinstance(probe, dict) else None
    if not isinstance(detailed, dict):
        debug("cart_agent", f"addon-diag lookup for {item_name!r}: no detailed_items in response "
              f"(items={len((probe or {}).get('items') or [])})")
        return empty
    match = detailed.get(str(menu_item_id))
    if not isinstance(match, dict):
        debug("cart_agent", f"addon-diag lookup for {item_name!r}: id {menu_item_id!r} not among "
              f"detailed ids {list(detailed)!r}")
        return empty
    groups = match.get("addon_groups")
    groups = groups if isinstance(groups, list) else []
    variant_groups = match.get("variant_groups")
    variant_groups = variant_groups if isinstance(variant_groups, list) else []
    debug(
        "cart_agent",
        f"addon-diag item {item_name!r} id={menu_item_id} restaurant={restaurant_id} "
        f"has_variants={match.get('has_variants')!r} variant_groups="
        f"{[(v.get('group_id'), v.get('group_name'), [c.get('name') for c in v.get('choices') or []]) for v in variant_groups]!r}",
    )
    for g in groups:
        required = isinstance(g.get("min_addons"), int) and g["min_addons"] >= 1
        debug(
            "cart_agent",
            f"addon-diag   group id={g.get('group_id')} name={g.get('group_name')!r} "
            f"min={g.get('min_addons')!r} max={g.get('max_addons')!r} choices={len(g.get('choices') or [])}"
            f"{'  <-- REQUIRED' if required else ''}",
        )
    return {"addon_groups": groups, "variant_groups": variant_groups}


def _required_groups(indexed_choices: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    required: dict[Any, dict[str, Any]] = {}
    for c in indexed_choices:
        lo = c.get("min_addons")
        if isinstance(lo, int) and lo >= 1:
            required[c.get("group_id")] = {"name": c["group_name"], "min": lo, "max": c.get("max_addons")}
    return required


def _unmet_required_groups(
    selected: list[dict[str, Any]], indexed_choices: list[dict[str, Any]],
) -> dict[Any, dict[str, Any]]:
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


def _ask_for_all_addons(
    item_name: str, addon_groups: list[dict[str, Any]], field_prefix: str = "addon_groups",
) -> list[dict[str, Any]]:
    indexed_choices: list[dict[str, Any]] = []
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

    required_only = field_prefix != "addon_groups"
    question, options = _addon_question_and_options(item_name, indexed_choices, required_only=required_only)
    raw_answer = interrupt({
        "question": question,
        "options": options,
        "agent": "cart_agent",
        "field": field_prefix,
        "multi_select": True,
    })
    debug("cart_agent", f"addon-diag selection raw_answer={raw_answer!r}")
    matches = _match_addon_reply(str(raw_answer), indexed_choices)
    selected = _enforce_group_limits(matches, indexed_choices)

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
            "field": f"{field_prefix}:required{attempt}", "multi_select": True,
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

    for gid in dict.fromkeys(c.get("group_id") for c in indexed_choices):
        picked = [e["name"] for e in selected if e.get("group_id") == gid]
        sample = next(c for c in indexed_choices if c.get("group_id") == gid)
        debug("cart_agent", f"addon-diag pick group={gid} {sample['group_name']!r} "
              f"min={sample.get('min_addons')!r} max={sample.get('max_addons')!r} picked={picked!r}")
    return selected


def _match_addon_reply(raw_answer: str, indexed_choices: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _wire_addon(addon: dict[str, Any]) -> dict[str, Any]:
    def as_num(v):
        return int(v) if str(v).isdigit() else v

    entry: dict[str, Any] = {
        "choice_id": as_num(addon.get("choice_id", addon.get("id"))),
        "group_id": as_num(addon.get("group_id")),
    }
    if addon.get("name"):
        entry["name"] = addon["name"]
    return entry


def _carry_forward_customizations(address_id: str, keep_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not keep_ids:
        return {}
    details = food_mcp.get_cart_item_details(address_id)
    carry: dict[str, dict[str, Any]] = {}
    for mid in keep_ids:
        detail = details.get(str(mid)) or {}
        extra: dict[str, Any] = {}
        addons = [
            a for a in (detail.get("addons") or [])
            if isinstance(a, dict) and (a.get("id") is not None or a.get("choice_id") is not None)
        ]
        if addons:
            extra["addons"] = [_wire_addon(a) for a in addons]
        variants = [
            v for v in (detail.get("variants") or [])
            if isinstance(v, dict) and v.get("variation_id") is not None
        ]
        if variants:
            extra["variants"] = [
                {"variation_id": v.get("variation_id"), "group_id": v.get("group_id"), "name": v.get("name")}
                for v in variants
            ]
        if extra:
            carry[str(mid)] = extra
            debug("cart_agent", f"carrying forward existing customization for cart item {mid}: {extra!r}")
    return carry


def _map_to_valid_addons(
    selected: list[dict[str, Any]], valid_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
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
    def gid(m):
        return int(m["group_id"]) if str(m["group_id"]).isdigit() else m["group_id"]

    def cid(m):
        return int(m["id"]) if str(m["id"]).isdigit() else m["id"]

    return [
        ("S: schema shape choice_id+group_id+name",
         [_wire_addon(m) for m in mapped]),
        ("S2: schema camelCase choiceId+groupId",
         [{"choiceId": cid(m), "groupId": gid(m)} for m in mapped]),
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


def _variant_shapes(variant_picks: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    def gid(v):
        return int(v["group_id"]) if str(v["group_id"]).isdigit() else v["group_id"]

    def vid(v):
        return int(v["id"]) if str(v["id"]).isdigit() else v["id"]

    return [
        ("V1: variantsV2 group_id+variation_id (schema's dedicated new format)",
         "variantsV2", [{"group_id": gid(v), "variation_id": vid(v)} for v in variant_picks]),
        ("V2: variants variation_id+group_id+name (legacy snake_case)",
         "variants", [{"variation_id": vid(v), "group_id": gid(v), "name": v["name"]} for v in variant_picks]),
        ("V3: variants camelCase variationId+groupId",
         "variants", [{"variationId": vid(v), "groupId": gid(v)} for v in variant_picks]),
    ]


def _addon_groups_matching_variant(
    addon_groups: list[dict[str, Any]], variant_groups: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chosen_names = {str(c.get("name") or "").strip().lower() for c in chosen}
    sibling_names: set[str] = set()
    for vg in variant_groups:
        for c in vg.get("choices") or []:
            name = str(c.get("name") or "").strip().lower()
            if name and name not in chosen_names:
                sibling_names.add(name)
    if not sibling_names:
        return addon_groups

    def names_a_sibling(group_name: str) -> bool:
        gn = (group_name or "").lower()
        return any(sib in gn for sib in sibling_names)

    kept = [g for g in addon_groups if not names_a_sibling(g.get("group_name") or "")]
    dropped = [g.get("group_name") for g in addon_groups if names_a_sibling(g.get("group_name") or "")]
    if dropped:
        debug("cart_agent", f"addon-diag filtered out addon groups belonging to a different variant: {dropped!r}")
    return kept


def _add_item_with_variant(
    restaurant_id: str, address_id: str, restaurant_name: str | None,
    item_name: str, menu_item_id: str, quantity: int, variant_groups: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str, list[dict[str, Any]]]:
    variant_choices = [{**g, "min_addons": 1, "max_addons": 1} for g in variant_groups]
    selected_variants = _ask_for_all_addons(item_name, variant_choices, field_prefix="variant_groups")
    if not selected_variants:
        debug("cart_agent", f"addon-diag no variant selected for {item_name!r} -- can't add")
        return {"status": "error", "message": "No size/variant was chosen.", "error_codes": []}, [], "", []

    existing_cart = food_mcp.get_cart(address_id)
    existing_items = _existing_cart_items(existing_cart)
    debug("cart_agent", f"variant flow -- existing cart before add: status={existing_cart.get('status')!r} "
          f"parsed_items={existing_items!r}")
    merged: dict[str, int] = {str(item["menu_item_id"]): item.get("quantity", 1) for item in existing_items}
    merged[str(menu_item_id)] = merged.get(str(menu_item_id), 0) + quantity
    carry_forward = _carry_forward_customizations(
        address_id, {mid for mid in merged if mid != str(menu_item_id)},
    )

    def build_payload(
        target_addons: list[dict[str, Any]], variant_entry: tuple[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        items = []
        for mid, qty in merged.items():
            entry: dict[str, Any] = {"menu_item_id": mid, "quantity": qty}
            if mid == str(menu_item_id):
                entry["addons"] = target_addons
                if variant_entry:
                    entry[variant_entry[0]] = variant_entry[1]
            elif mid in carry_forward:
                entry.update(carry_forward[mid])
            items.append(entry)
        return items

    result: dict[str, Any] | None = None
    last_codes = None
    for label, field, payload in _variant_shapes(selected_variants):
        debug("cart_agent", f"addon-diag variant retry trying shape {label}: {payload!r}")
        result = food_mcp.update_cart(
            restaurant_id, address_id, build_payload([], variant_entry=(field, payload)),
            restaurant_name=restaurant_name,
        )
        if result.get("status") == "success":
            debug("cart_agent", f"addon-diag variant SUCCESS -- Swiggy accepted shape {label} (field={field!r})")
            break
        last_codes = result.get("error_codes")
        debug("cart_agent", f"addon-diag variant shape {label} rejected: error_codes={last_codes!r} "
              f"message={result.get('message')!r}")
        if "INVALID_ITEM_IDS_IN_REQUEST" in (last_codes or []) and any(
            str(i.get("menu_item_id")) != str(menu_item_id) for i in existing_items
        ):
            debug("cart_agent", "addon-diag variant retry: cart-identity conflict, not a shape problem -- stopping")
            break

    if not result or result.get("status") != "success":
        debug("cart_agent", "addon-diag NO variant request shape was accepted -- see the attempts above")
        return (result or {"status": "error", "message": "variant rejected", "error_codes": last_codes}), \
            [], "", existing_items

    detail = food_mcp.get_cart_item_details(address_id).get(str(menu_item_id)) or {}
    valid = detail.get("valid_addons") or []
    debug("cart_agent", f"addon-diag variant landed -- valid_addons for THIS variant: "
          f"{[(g.get('group_id'), g.get('group_name'), len(g.get('choices') or [])) for g in valid]!r}")
    valid = _addon_groups_matching_variant(valid, variant_groups, selected_variants)
    debug("cart_agent", f"addon-diag valid_addons after filtering to the chosen variant: "
          f"{[(g.get('group_id'), g.get('group_name')) for g in valid]!r}")
    if not valid:
        return result, [], "", existing_items

    selected_addons = _ask_for_all_addons(item_name, valid)
    if not selected_addons:
        return result, [], "", existing_items

    with_addons = food_mcp.update_cart(
        restaurant_id, address_id,
        build_payload([_wire_addon(a) for a in selected_addons], variant_entry=(field, payload)),
        restaurant_name=restaurant_name,
    )
    if with_addons.get("status") == "success":
        debug("cart_agent", "addon-diag variant+addons SUCCESS")
        return with_addons, selected_addons, "", existing_items
    debug("cart_agent", f"addon-diag variant landed but add-ons rejected: "
          f"{with_addons.get('error_codes')!r} -- keeping the item with its variant, without the add-ons")
    return result, [], (
        "I added it with your size/variant choice, but Swiggy didn't accept the add-ons on top of it, "
        "so I left those off -- you can add them in the Swiggy app."
    ), existing_items


def _retry_addons_via_valid_addons(
    build_payload, restaurant_id: str, address_id: str, restaurant_name: str | None,
    menu_item_id: str, selected: list[dict[str, Any]], failed_result: dict[str, Any],
) -> tuple[dict[str, Any], str]:
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


def _cart_conflict_reply(
    dish: dict[str, Any], quantity: int, existing_name: str | None,
    addons: list[dict[str, Any]] | None,
) -> dict:
    where = f" from {existing_name}" if existing_name else " from another restaurant"
    content = (
        f"Your Swiggy cart already has items{where}, and a cart can only hold one restaurant at a time, "
        f"so I couldn't add {dish.get('item_name') or 'that dish'}"
        f"{' from ' + str(dish.get('restaurant_name')) if dish.get('restaurant_name') else ''}. "
        "Want me to clear your cart and add it instead?"
    )
    return _cart_reply(
        content,
        pending_dish=dish, pending_quantity=quantity, pending_addons=addons,
        options=_CONFLICT_OPTIONS,
    )


def add_dish_to_cart(
    dish: dict[str, Any], address_id: str, quantity: int = 1, lead_in: str = "",
    known_restaurant_id: str | None = None, known_restaurant_name: str | None = None,
    preselected_addons: list[dict[str, Any]] | None = None,
) -> dict:
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

    if (
        known_restaurant_id
        and str(known_restaurant_id) != str(restaurant_id)
    ):
        return _cart_conflict_reply(dish, quantity, known_restaurant_name, preselected_addons)

    attempt_addon_only = False
    selected_addons: list[dict[str, Any]] = []

    variant_handled = False

    if _requires_customization(dish.get("tags") or ""):
        tags_haystack = (dish.get("tags") or "").lower()

        if preselected_addons is not None:
            selected_addons = preselected_addons
            debug("cart_agent", f"re-using previously picked addons for {dish.get('item_name')!r}: {selected_addons!r}")
        else:
            customization = _fetch_customization(
                str(dish.get("item_name") or ""), str(dish.get("menu_item_id") or ""),
                address_id, str(restaurant_id),
            )
            addon_groups = customization["addon_groups"]
            variant_groups = customization["variant_groups"]
            debug(
                "cart_agent",
                f"customization for {dish.get('item_name')!r}: "
                f"addon_groups={[g.get('group_name') for g in addon_groups]!r} "
                f"variant_groups={[g.get('group_name') for g in variant_groups]!r}",
            )

            if not variant_groups and not addon_groups and "has variants" in tags_haystack:
                return _cart_reply(_customization_refusal(dish.get("item_name")))

            if variant_groups:
                variant_result, selected_addons, variant_note, existing_items = _add_item_with_variant(
                    restaurant_id, address_id, dish.get("restaurant_name"),
                    str(dish.get("item_name") or ""), str(menu_item_id), quantity, variant_groups,
                )
                if variant_note:
                    lead_in = f"{lead_in} {variant_note}".strip()
                result = variant_result
                variant_handled = True
            elif addon_groups:
                selected_addons = _ask_for_all_addons(str(dish.get("item_name") or ""), addon_groups)
                debug("cart_agent", f"user-selected addons for {dish.get('item_name')!r}: {selected_addons!r}")
        attempt_addon_only = True

    if not variant_handled:
        existing_cart = food_mcp.get_cart(address_id)
        existing_items = _existing_cart_items(existing_cart)
        debug("cart_agent", f"existing cart before add: status={existing_cart.get('status')!r} "
              f"parsed_items={existing_items!r}")

        merged: dict[str, int] = {str(item["menu_item_id"]): item.get("quantity", 1) for item in existing_items}
        target_quantity = merged.get(str(menu_item_id), 0) + quantity
        merged[str(menu_item_id)] = target_quantity
        carry_forward = _carry_forward_customizations(
            address_id, {mid for mid in merged if mid != str(menu_item_id)},
        )
        def _build_payload(target_addons: list[dict[str, Any]]) -> list[dict[str, Any]]:
            items = []
            for mid, qty in merged.items():
                entry: dict[str, Any] = {"menu_item_id": mid, "quantity": qty}
                if attempt_addon_only and mid == str(menu_item_id):
                    entry["addons"] = target_addons
                elif mid in carry_forward:
                    entry.update(carry_forward[mid])
                items.append(entry)
            return items
    else:
        target_quantity = next(
            (i.get("quantity", 1) for i in existing_items if str(i.get("menu_item_id")) == str(menu_item_id)), 0,
        ) + quantity

    if not variant_handled:
        cart_items_payload = _build_payload([_wire_addon(a) for a in selected_addons])
        debug("cart_agent", f"sending merged cartItems={cart_items_payload!r}")
        if attempt_addon_only:
            import json as _json
            debug(
                "cart_agent",
                "addon-diag first attempt: wire JSON=" + _json.dumps(cart_items_payload, default=str)
                + " | id types=" + repr(sorted({type(a.get("choice_id")).__name__ for e in cart_items_payload
                                                 for a in e.get("addons", [])}))
                + " | price units=rupees (search_menu), the cart's valid_addons uses paise",
            )

        result = food_mcp.update_cart(
            restaurant_id, address_id,
            cart_items_payload,
            restaurant_name=dish.get("restaurant_name"),
        )
        debug("cart_agent", f"update_cart -> {result!r}")

    if (
        result.get("status") != "success"
        and "INVALID_ITEM_IDS_IN_REQUEST" in (result.get("error_codes") or [])
        and any(str(i.get("menu_item_id")) != str(menu_item_id) for i in existing_items)
    ):
        debug("cart_agent", "cart conflict: existing cart items don't belong to this restaurant -- "
              "offering clear-and-add")
        return _cart_conflict_reply(dish, quantity, known_restaurant_name, selected_addons if attempt_addon_only else None)

    addon_note = ""
    if (
        not variant_handled and result.get("status") != "success" and attempt_addon_only and selected_addons
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
            error_codes = result.get("error_codes")
            if error_codes and "INVALID_ADDON" not in error_codes:
                debug("cart_agent", f"addon-only attempt: non-addon rejection {error_codes!r}")
                content = (
                    f"Couldn't add {dish.get('item_name') or 'that item'} to your cart "
                    f"({', '.join(error_codes)}): {result.get('message') or 'Swiggy rejected the request'}"
                )
                return _cart_reply(content)
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
        verify_cart = food_mcp.get_cart(address_id)
        verify_items = _existing_cart_items(verify_cart)
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
    cart_state = outcome.get("cart_state") or {}
    if cart_state.get("status") in ("text", "success"):
        outcome["cart_state"] = {
            **cart_state,
            "restaurant": {"id": restaurant_id, "name": dish.get("restaurant_name")},
        }
    return outcome
