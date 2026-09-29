"""Group mode (feature 6): "order for 4 flatmates".

1. Start  -> create a poll, reply with its link (share on WhatsApp).
2. Results -> combine everyone's answers (tightest budget, veg if anyone
   is veg-only, mild if anyone wants mild, cuisines liked by most and
   vetoed by nobody) and show the three restaurants that work for all.
3. Pick   -> open that restaurant's menu as a multi-select, so dishes for
   everyone go into one cart through the normal add flow.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import AIMessage

from .. import food_mcp
from ..client import using_real_food_mcp
from ..debug_utils import debug
from ..profile_utils import latest_user_text
from ..state import CravrState
from . import combine as combiner
from .store import create_group, get_group

GROUP_RESULTS_VALUE = "__group_results__"
GROUP_CANCEL_VALUE = "__group_cancel__"
GROUP_PICK_PREFIX = "__group_pick__:"

_NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "couple": 2}
_GROUP_RE = re.compile(
    r"\bgroup order\b|\border(?:ing)? for (?:the |my )?(?:team|office|group|gang|flatmates|roommates|friends|family|all of us)\b|"
    r"\bfor (?P<n>\d{1,2}|two|three|four|five|six|seven|eight|nine|ten)\s+(?:of us|friends|flatmates|roommates|colleagues)\b|"
    r"\b(?:me and|with) my (?:flatmates|roommates|friends|team)\b",
    re.IGNORECASE,
)


def _public_url() -> str:
    explicit = os.getenv("CRAVR_PUBLIC_URL")
    if explicit:
        return explicit.rstrip("/")
    redirect = os.getenv("SWIGGY_REDIRECT_URI", "")
    if redirect.endswith("/oauth/callback"):
        return redirect[: -len("/oauth/callback")]
    return "http://localhost:8001"


def is_group_request(text: str, group_state: dict[str, Any] | None = None) -> bool:
    t = (text or "").strip()
    if t in (GROUP_RESULTS_VALUE, GROUP_CANCEL_VALUE) or t.startswith(GROUP_PICK_PREFIX):
        return True
    return bool(_GROUP_RE.search(t))


def _group_size(text: str) -> int:
    match = _GROUP_RE.search(text or "")
    n = (match.group("n") if match else None) or ""
    if n.isdigit():
        return int(n)
    if n.lower() in _NUMBER_WORDS:
        return _NUMBER_WORDS[n.lower()]
    digits = re.search(r"\b(\d{1,2})\b", text or "")
    return int(digits.group(1)) if digits else 3


def _reply(content: str, group_state: dict[str, Any], **extra: Any) -> dict:
    return {
        "group_state": group_state,
        "messages": [AIMessage(content=content, name="group_agent")],
        **extra,
    }


_COLLECT_OPTIONS = [
    {"value": GROUP_RESULTS_VALUE, "label": "Everyone's in — find a place"},
    {"value": GROUP_CANCEL_VALUE, "label": "Cancel group order"},
]


def _start(text: str) -> dict:
    size = _group_size(text)
    group = create_group(size)
    link = f"{_public_url()}/group/{group['id']}"
    debug("group_agent", f"created group {group['id']} for {size}")
    return _reply(
        f"Group order for {size}! Share this link with everyone (you can fill it in too):\n\n{link}\n\n"
        "Each person picks veg/non-veg, budget, spice and what they're in the mood for — 30 seconds. "
        "Tap below once they've answered and I'll find one restaurant that works for all of you.",
        {"group_id": group["id"], "size": size, "status": "collecting", "link": link, "options": _COLLECT_OPTIONS},
    )


def _candidate_restaurants(combined: dict[str, Any], address_id: str | None) -> list[dict[str, Any]]:
    cuisines = combined.get("top_cuisines") or []
    queries = [combiner.SEARCH_QUERIES.get(c, c) for c in cuisines[:2]] or ["restaurants"]
    if using_real_food_mcp():
        found: list[dict[str, Any]] = []
        for query in queries + (["restaurants"] if "restaurants" not in queries else []):
            result = food_mcp.search_restaurants(query, address_id)
            food_mcp.restaurants.remember(address_id, result["restaurants"])
            found += result["restaurants"]
            if len(found) >= 12:
                break
        return found

    # Mock: build restaurant records from the menu.
    from ..recommend.RecommendAgent import _call_recommendation_tool

    dishes = _call_recommendation_tool({}, address_id=address_id)
    by_restaurant: dict[str, list[dict[str, Any]]] = {}
    for dish in dishes:
        by_restaurant.setdefault(str(dish.get("restaurant_name") or ""), []).append(dish)
    restaurants = []
    for name, items in by_restaurant.items():
        prices = sorted(float(i.get("price_inr") or 0) for i in items)
        ratings = [float(i.get("rating") or 0) for i in items if i.get("rating")]
        restaurants.append({
            "restaurant_id": str(items[0].get("restaurant_id") or name),
            "restaurant_name": name,
            "cuisines": sorted({str(i.get("cuisine") or "") for i in items} - {""}),
            "rating": round(sum(ratings) / len(ratings), 1) if ratings else None,
            "cost_for_two": f"₹{int(prices[len(prices) // 2] * 2)} for two" if prices else None,
            "veg": all(str(i.get("dietary_type")) == "Veg" for i in items),
            "is_open": True,
        })
    return restaurants


def _results(group_state: dict[str, Any], address_id: str | None) -> dict:
    group = get_group(group_state.get("group_id") or "")
    if not group:
        return _reply("That group poll has expired — say \"group order for 4\" to start a new one.", {})
    responses = group.get("responses") or []
    if not responses:
        return _reply(
            f"Nobody has answered yet. Share the link again:\n\n{group_state.get('link')}",
            {**group_state, "options": _COLLECT_OPTIONS},
        )

    combined = combiner.combine(responses)
    summary = combiner.summary_text(combined, group["size"])
    ranked = combiner.rank_restaurants(_candidate_restaurants(combined, address_id), combined)
    debug("group_agent", f"combined={combined} ranked={[r.get('restaurant_name') for r in ranked]}")
    if not ranked:
        return _reply(
            f"{summary}\n\nI couldn't find a restaurant that works for everyone right now. "
            "Try relaxing a veto or the budget and tap again.",
            {**group_state, "combined": combined, "options": _COLLECT_OPTIONS},
        )

    lines = [f"  {i}. {r['restaurant_name']} — {combiner.restaurant_reason(r, combined)}"
             for i, r in enumerate(ranked, start=1)]
    options = [{"value": f"{GROUP_PICK_PREFIX}{r['restaurant_id']}", "label": r["restaurant_name"]} for r in ranked]
    return _reply(
        f"{summary}\n\nPlaces that work for everyone:\n" + "\n".join(lines) + "\nPick one and I'll open its menu.",
        {**group_state, "status": "choosing", "combined": combined, "candidates": ranked, "options": options},
    )


def _pick(value: str, group_state: dict[str, Any], state: CravrState, address_id: str | None) -> dict:
    restaurant_id = value[len(GROUP_PICK_PREFIX):]
    restaurant = next((r for r in group_state.get("candidates") or [] if str(r.get("restaurant_id")) == restaurant_id), None)
    if not restaurant:
        return _reply("I lost track of that restaurant — tap \"Everyone's in\" again.", {**group_state, "options": _COLLECT_OPTIONS})
    combined = group_state.get("combined") or {}
    name = restaurant.get("restaurant_name") or ""
    size = int(group_state.get("size") or 2)
    budget_total = (combined.get("budget_per_person") or 0) * size or None

    if using_real_food_mcp():
        menu = food_mcp.get_restaurant_menu_items(restaurant_id, address_id, name)
    else:
        from ..recommend.RecommendAgent import _call_recommendation_tool

        menu = _call_recommendation_tool({"restaurant_name": name}, address_id=address_id)
    if combined.get("all_veg"):
        menu = [m for m in menu if str(m.get("dietary_type") or "Veg") == "Veg"]
    if combined.get("spice") == "mild":
        from ..recommend.feedback import SPICY_WORDS, _has_word

        menu = [m for m in menu if not _has_word(str(m.get("item_name") or "").lower(), SPICY_WORDS)]

    new_group_state = {**group_state, "status": "ordering", "restaurant_id": restaurant_id, "options": []}
    brief = {**(state.get("recommendation_brief") or {}), "filters": {"group_budget_inr": budget_total}}
    if not menu:
        return _reply(f"I couldn't load {name}'s menu right now — pick another place.", {**group_state})

    scope = {"restaurant_id": restaurant_id, "restaurant_name": name, "existing_ids": []}
    budget_note = f" Aim for about ₹{budget_total} in total." if budget_total else ""
    from ..cart.guards import _menu_categories
    from ..recommend.RecommendAgent import _offer_category_batch, _offer_dish_batch

    categories = _menu_categories(menu)
    if using_real_food_mcp() and categories:
        result = _offer_category_batch(
            f"{name} it is! Pick the sections to order from for {size} people.{budget_note}", categories, scope,
        )
    else:
        result = _offer_dish_batch(f"{name} it is! Pick dishes for {size} people.{budget_note}", menu[:12], scope)
    result["group_state"] = new_group_state
    result["recommendation_brief"] = brief
    return result


def group_agent_node(state: CravrState) -> dict:
    text = latest_user_text(state.get("messages") or []).strip()
    group_state = dict(state.get("group_state") or {})
    address_id = (state.get("context_profile") or {}).get("address_id")
    if address_id is None and using_real_food_mcp():
        address_id = food_mcp.get_saved_address().get("address_id")

    if text == GROUP_CANCEL_VALUE:
        return _reply("Group order cancelled. Tell me what you'd like and I'll help you pick.", {})
    if text == GROUP_RESULTS_VALUE:
        return _results(group_state, address_id)
    if text.startswith(GROUP_PICK_PREFIX):
        return _pick(text, group_state, state, address_id)
    return _start(text)
