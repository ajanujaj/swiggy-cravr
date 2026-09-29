from __future__ import annotations

import re
from typing import Any

from ..debug_utils import debug
from . import _call_tool
from ._shared import _first
from .menu import _IMAGE_REF_RE, _normalize_item, _parse_menu_items

_EATRIGHT_TAGS = {"low-calorie", "high-protein", "low-carb", "gluten-free", "sugar-free-option", "organic"}


def _infer_collection(filters: dict[str, Any]) -> str | None:
    tags = {t.strip() for t in str(filters.get("tags") or "").split(",") if t.strip()}
    if tags & _EATRIGHT_TAGS:
        return "EATRIGHT"
    max_price = filters.get("max_price_inr")
    if isinstance(max_price, (int, float)) and max_price <= 99:
        return "STORE_99"
    return None


_SEARCH_MENU_ITEM_RE = re.compile(
    r"^\d+\.\s*(?P<name>.+?)\s*—\s*₹(?P<price>[\d,]+(?:\.\d+)?)\s*\|\s*(?P<rest>.+?)\s*"
    r"\(restaurantId:\s*(?P<restaurant_id>[^)]+)\)\s*\(ID:\s*(?P<id>[^)]+)\)"
    r"(?:\s*\[(?P<tags>[^\]]+)\])?\s*$",
    re.MULTILINE,
)
_RATING_RE = re.compile(r"^\d+(?:\.\d+)?★$")


def _parse_search_menu_items(text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in _SEARCH_MENU_ITEM_RE.finditer(text):
        rest_parts = [p.strip() for p in m.group("rest").split("|") if p.strip()]
        restaurant_name = rest_parts[-1] if rest_parts else ""
        rating: str | None = None
        dietary_type = ""
        for part in rest_parts[:-1]:
            if _RATING_RE.match(part):
                rating = part.rstrip("★")
            elif part.lower() in ("veg", "non-veg", "nonveg", "non veg"):
                dietary_type = "Veg" if part.lower() == "veg" else "Non-Veg"

        try:
            price_inr: float | int = float(m.group("price").replace(",", ""))
        except ValueError:
            price_inr = 0

        items.append({
            "item_name": m.group("name").strip(),
            "menu_item_id": m.group("id").strip(),
            "restaurant_name": restaurant_name,
            "restaurant_id": m.group("restaurant_id"),
            "cuisine": "",
            "price_inr": price_inr,
            "rating": rating,
            "tags": m.group("tags") or "",
            "dietary_type": dietary_type,
        })
    return items


_DETAILED_ITEM_HEADER_RE = re.compile(
    r"^\d+\.[ \t]*(?P<name>.+?)[ \t]*—[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?)[ \t]*\|[ \t]*"
    r"(?P<rest>(?:(?!\(restaurantId:).)+?)[ \t]*"
    r"\(ID:[ \t]*(?P<id>[^)]+)\)[ \t]*$",
    re.MULTILINE,
)
_DETAILED_ADDON_GROUP_RE = re.compile(
    r"^[ \t]*(?P<kind>Addons|Variants) \((?P<group_name>.*?)\):[ \t]*\[(?P<body>.*)\][ \t]*$",
    re.MULTILINE,
)
_DETAILED_ADDON_CHOICE_RE = re.compile(
    r"(?P<name>.+?)[ \t]*₹(?P<price>[\d,]+(?:\.\d+)?)[ \t]*"
    r"\(group:(?P<group_id>\d+),[ \t]*(?:choice|var):(?P<choice_id>\d+)\)"
)


def _parse_detailed_search_menu_items(text: str) -> dict[str, dict[str, Any]]:
    headers = list(_DETAILED_ITEM_HEADER_RE.finditer(text))
    items: dict[str, dict[str, Any]] = {}
    for idx, m in enumerate(headers):
        item_id = m.group("id").strip()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[m.end():block_end]

        try:
            price_inr: float | int = float(m.group("price").replace(",", ""))
        except ValueError:
            price_inr = 0

        addon_groups = []
        variant_groups = []
        for gm in _DETAILED_ADDON_GROUP_RE.finditer(block):
            choices = []
            for cm in _DETAILED_ADDON_CHOICE_RE.finditer(gm.group("body")):
                try:
                    choice_price: float | int = float(cm.group("price").replace(",", ""))
                except ValueError:
                    choice_price = 0
                choices.append({
                    "id": cm.group("choice_id"),
                    "name": cm.group("name").strip().strip(", ").strip(),
                    "price_inr": choice_price,
                    "group_id": cm.group("group_id"),
                })
            group_record = {
                "group_id": choices[0]["group_id"] if choices else None,
                "group_name": gm.group("group_name").strip(),
                "choices": choices,
            }
            (variant_groups if gm.group("kind") == "Variants" else addon_groups).append(group_record)
        items[item_id] = {
            "item_name": m.group("name").strip(),
            "price_inr": price_inr,
            "addon_groups": addon_groups,
            "variant_groups": variant_groups,
        }
    return items


def _attach_addon_limits(detailed: dict[str, dict[str, Any]], structured: Any) -> None:
    if not isinstance(structured, dict):
        return
    for entry in structured.get("items") or []:
        if not isinstance(entry, dict):
            continue
        parsed = detailed.get(str(entry.get("menu_item_id")))
        if not parsed:
            continue
        parsed["has_variants"] = bool(entry.get("hasVariants"))
        parsed["has_addons"] = bool(entry.get("hasAddons"))
        limits = {
            str(g.get("groupId")): g for g in entry.get("addons") or [] if isinstance(g, dict)
        }
        for group in parsed.get("addon_groups") or []:
            raw_group = limits.get(str(group.get("group_id")))
            if raw_group:
                group["min_addons"] = raw_group.get("minAddons")
                group["max_addons"] = raw_group.get("maxAddons")


def search_menu_items(
    query: str, address_id: str | None, veg_only: bool = False, offset: int = 0,
    restaurant_id: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"addressId": address_id, "query": query, "offset": offset}
    if veg_only:
        arguments["vegFilter"] = 1
    if restaurant_id:
        arguments["restaurantIdOfAddedItem"] = restaurant_id

    try:
        raw, meta = _call_tool("search_menu", arguments, return_meta=True)
    except Exception as exc:
        debug("food_mcp", f"search_menu FAILED: {exc}")
        return {"items": [], "has_more": False, "next_offset": None}

    if isinstance(raw, str):
        parsed_items = _parse_search_menu_items(raw)
        result: dict[str, Any] = {"items": parsed_items, "has_more": False, "next_offset": None}
        if not parsed_items:
            detailed = _parse_detailed_search_menu_items(raw)
            if detailed:
                _attach_addon_limits(detailed, meta.get("structured_content") if isinstance(meta, dict) else None)
                result["detailed_items"] = detailed
        return result

    if not isinstance(raw, dict):
        return {"items": [], "has_more": False, "next_offset": None}

    items: list[dict[str, Any]] = []
    for entry in raw.get("items") or []:
        if not isinstance(entry, dict):
            continue
        restaurant = entry.get("restaurant") if isinstance(entry.get("restaurant"), dict) else {}
        restaurant_name = _first(entry, "restaurantName") or _first(restaurant, "name", "restaurantName", default="")
        cuisine = _first(entry, "cuisine") or _first(restaurant, "cuisine", "cuisines")
        if isinstance(cuisine, list):
            cuisine = ", ".join(str(c) for c in cuisine)
        name = _first(entry, "name", "itemName", "dishName", default="")
        if not name:
            continue
        items.append({
            "item_name": name,
            "menu_item_id": _first(entry, "id", "itemId", "menuItemId"),
            "restaurant_name": restaurant_name or "",
            "restaurant_id": _first(entry, "restaurantId") or _first(restaurant, "id", "restaurantId"),
            "cuisine": cuisine or "",
            "price_inr": _first(entry, "price", "priceInr", "price_inr", default=0),
            "rating": _first(entry, "rating", "restaurantRating"),
            "tags": _first(entry, "tags", default="") or "",
            "dietary_type": "Veg" if _first(entry, "isVeg", "veg") else "",
        })

    return {
        "items": items,
        "has_more": bool(raw.get("hasMore")),
        "next_offset": raw.get("nextOffset"),
    }


def _search_menu_relevant(
    query: str, address_id: str | None, veg_only: bool, local_filters: dict[str, Any],
) -> list[dict[str, Any]]:
    """search_menu_items for `query`, cut down to items that pass
    _apply_filters (dietary/price/cuisine) AND actually relate to `query`
    by name (_matches_craving) -- search_menu is Swiggy's own free-text
    search, not an exact-name lookup, so it can return dishes that don't
    contain the craving at all, and _apply_filters alone never checks
    relevance."""
    result = search_menu_items(query, address_id, veg_only=veg_only)
    if not result["items"]:
        return []
    filtered = _apply_filters(result["items"], local_filters)
    return [d for d in filtered if _matches_craving(str(d.get("item_name") or ""), query)]


def search_dishes(filters: dict[str, Any], address_id: str | None, limit_restaurants: int = 5) -> list[dict[str, Any]]:
    craving = filters.get("craving")
    if craving:
        veg_only = filters.get("dietary_type") == "Veg"
        local_filters = {k: v for k, v in filters.items() if k != "dietary_type"}
        filtered = _search_menu_relevant(str(craving), address_id, veg_only, local_filters)

        # craving_tag can independently name a real, different dish from
        # craving (round 1 "Palak Paneer" + round 2 "Naan" -- a side, not a
        # narrower version of it). Searching craving alone never looks for
        # round 1's dish at all. A single combined search string doesn't
        # work either -- confirmed live: "Palak Paneer Naan" ranked
        # "Paneer Naan" (an unrelated dish sharing two of three words)
        # above genuine Palak Paneer options. So both are searched
        # separately and merged here; _rank_candidates scores the merged
        # pool by how well each dish matches either phrase (see
        # RecommendAgent's _craving_match_score), so the genuinely-named
        # dish wins regardless of which search turned it up.
        craving_secondary = filters.get("craving_secondary")
        if craving_secondary:
            secondary_filtered = _search_menu_relevant(
                str(craving_secondary), address_id, veg_only, local_filters
            )
            seen_ids = {str(d.get("menu_item_id")) for d in filtered}
            filtered = filtered + [
                d for d in secondary_filtered if str(d.get("menu_item_id")) not in seen_ids
            ]

        if filtered:
            return filtered
        debug("food_mcp", f"search_menu found nothing usable for {craving!r} -- falling back to search_restaurants")

    query = filters.get("cuisine") or filters.get("tags") or filters.get("dietary_type") or "restaurants"
    arguments: dict[str, Any] = {"addressId": address_id, "query": query}
    collection = _infer_collection(filters)
    if collection:
        arguments["collection"] = collection

    try:
        raw = _call_tool("search_restaurants", arguments)
    except Exception as exc:
        debug("food_mcp", f"search_restaurants FAILED: {exc}")
        return []

    restaurants = raw.get("restaurants") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    if not restaurants:
        return []

    open_restaurants = [
        r for r in restaurants
        if isinstance(r, dict) and _first(r, "availabilityStatus", default="OPEN") == "OPEN"
    ][:limit_restaurants]

    dishes: list[dict[str, Any]] = []
    for restaurant in open_restaurants:
        restaurant_id = _first(restaurant, "restaurantId", "id")
        restaurant_name = _first(restaurant, "name", "restaurantName", default="")
        if not restaurant_id:
            continue
        try:
            menu_raw = _call_tool(
                "get_restaurant_menu", {"restaurantId": restaurant_id, "addressId": address_id},
                context=f"for {restaurant_id}",
            )
        except Exception as exc:
            debug("food_mcp", f"get_restaurant_menu FAILED for {restaurant_id}: {exc}")
            continue
        if isinstance(menu_raw, str):
            menu_raw = _IMAGE_REF_RE.sub("", menu_raw)
        restaurant_cuisine = _first(restaurant, "cuisine", "cuisines")
        if isinstance(restaurant_cuisine, list):
            restaurant_cuisine = ", ".join(str(c) for c in restaurant_cuisine)

        if isinstance(menu_raw, str):
            dishes.extend(_parse_menu_items(menu_raw, restaurant_name, restaurant_cuisine, restaurant_id))
        else:
            raw_items = menu_raw.get("items") if isinstance(menu_raw, dict) else (menu_raw if isinstance(menu_raw, list) else [])
            for item in raw_items or []:
                dishes.append(_normalize_item(item, restaurant_name, restaurant_cuisine, restaurant_id))

    dishes = _apply_filters(dishes, filters)
    if craving:
        dishes = [d for d in dishes if _matches_craving(str(d.get("item_name") or ""), str(craving))]
    return dishes


def _apply_filters(dishes: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    dietary_type = filters.get("dietary_type")
    max_price = filters.get("max_price_inr")
    cuisine = (filters.get("cuisine") or "").lower()

    def keep(d: dict[str, Any]) -> bool:
        if dietary_type and str(d.get("dietary_type") or "").lower() != dietary_type.lower():
            return False
        if max_price is not None and isinstance(d.get("price_inr"), (int, float)) and d["price_inr"] > max_price:
            return False
        if cuisine and cuisine not in str(d.get("cuisine") or "").lower():
            return False
        return True

    return [d for d in dishes if keep(d)]


_CRAVING_STOPWORDS = {"a", "an", "and", "the", "or", "with", "some", "of", "some", "please"}


def _matches_craving(item_name: str, craving: str) -> bool:
    name = (item_name or "").lower()
    craving_l = (craving or "").lower().strip()
    if not craving_l:
        return True
    if craving_l in name or name in craving_l:
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", craving_l) if w not in _CRAVING_STOPWORDS and len(w) > 2]
    if not words:
        return True
    return any(w in name for w in words)
