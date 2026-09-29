from __future__ import annotations

from typing import Any


def _requires_customization(tags: str) -> bool:
    haystack = (tags or "").lower()
    return "has addons" in haystack or "has variants" in haystack


def _customization_refusal(item_name: str) -> str:
    return (
        f"{item_name or 'That item'} needs a variant/add-on choice I can't set for you yet, "
        "so I haven't added it to your cart -- pick something else, or add it with add-ons directly in the Swiggy app."
    )


def _menu_categories(
    menu_items: list[dict[str, Any]], exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    exclude = {str(i) for i in (exclude_ids or ())}
    order: list[str] = []
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in menu_items:
        if str(item.get("menu_item_id")) in exclude:
            continue
        if _requires_customization(item.get("tags") or ""):
            continue
        category = str(item.get("tags") or "").split(";")[0].strip()
        if not category:
            continue
        if category not in buckets:
            order.append(category)
            buckets[category] = []
        buckets[category].append(item)
    return [{"category": name, "items": buckets[name]} for name in order]
