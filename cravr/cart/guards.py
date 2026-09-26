from __future__ import annotations

from typing import Any

from .parsing import _COMPLEMENT_ORDER, _menu_bucket


def _requires_customization(tags: str) -> bool:
    haystack = (tags or "").lower()
    return "has addons" in haystack or "has variants" in haystack


def _customization_refusal(item_name: str) -> str:
    return (
        f"{item_name or 'That item'} needs a variant/add-on choice I can't set for you yet, "
        "so I haven't added it to your cart -- pick something else, or add it with add-ons directly in the Swiggy app."
    )


def _suggest_complementary_items(
    menu_items: list[dict[str, Any]],
    cart_item_id: str | None,
    exclude_ids: set[str],
    limit: int = 5,
) -> list[dict[str, Any]]:
    by_id = {str(i.get("menu_item_id")): i for i in menu_items if i.get("menu_item_id")}
    cart_item = by_id.get(str(cart_item_id)) if cart_item_id else None
    cart_bucket = _menu_bucket((cart_item or {}).get("tags") or "")
    preferred_buckets = _COMPLEMENT_ORDER.get(cart_bucket, ["drink", "dessert", "side"])

    candidates = [
        i for i in menu_items
        if str(i.get("menu_item_id")) not in exclude_ids
        and not _requires_customization(i.get("tags") or "")
    ]
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
