"""Combine everyone's poll answers into one set of limits, then find the
restaurant that works for the whole group (feature 6)."""

from __future__ import annotations

import re
from collections import Counter
from statistics import median
from typing import Any

CUISINE_CHOICES = [
    "North Indian", "South Indian", "Chinese", "Biryani", "Pizza", "Burgers",
    "Rolls & Wraps", "Healthy", "Street Food", "Desserts",
]
# How each poll choice shows up in restaurant cuisine lists / names.
CUISINE_ALIASES = {
    "North Indian": ("north indian", "punjabi", "mughlai", "tandoor"),
    "South Indian": ("south indian", "dosa", "idli", "kerala", "andhra", "chettinad"),
    "Chinese": ("chinese", "asian", "momos", "indo-chinese", "tibetan"),
    "Biryani": ("biryani", "hyderabadi"),
    "Pizza": ("pizza", "pizzas", "italian"),
    "Burgers": ("burger", "burgers", "american", "fast food"),
    "Rolls & Wraps": ("rolls", "wraps", "shawarma", "kathi", "lebanese", "arabian"),
    "Healthy": ("healthy", "salad", "salads", "bowl", "keto"),
    "Street Food": ("street food", "chaat", "snacks", "vada pav", "pav bhaji"),
    "Desserts": ("desserts", "ice cream", "bakery", "sweets", "waffle"),
}
SEARCH_QUERIES = {
    "North Indian": "north indian", "South Indian": "south indian", "Chinese": "chinese",
    "Biryani": "biryani", "Pizza": "pizza", "Burgers": "burger", "Rolls & Wraps": "rolls",
    "Healthy": "healthy", "Street Food": "street food", "Desserts": "desserts",
}


def combine(responses: list[dict[str, Any]]) -> dict[str, Any]:
    diets = [str(r.get("diet") or "either").lower() for r in responses]
    budgets = [int(r["budget"]) for r in responses if str(r.get("budget") or "").isdigit()]
    spices = [str(r.get("spice") or "medium").lower() for r in responses]
    likes: Counter = Counter()
    vetoes: Counter = Counter()
    allergies: list[str] = []
    for r in responses:
        likes.update(r.get("likes") or [])
        vetoes.update(r.get("dislikes") or [])
        if str(r.get("allergies") or "").strip():
            allergies.append(f"{r.get('name') or 'Someone'}: {str(r['allergies']).strip()}")

    veg_count = diets.count("veg")
    return {
        "answered": len(responses),
        "all_veg": bool(responses) and veg_count == len(responses),
        "veg_count": veg_count,
        # The group eats together, so the tightest budget sets the per-person cap.
        "budget_per_person": min(budgets) if budgets else None,
        "median_budget": int(median(budgets)) if budgets else None,
        "spice": "mild" if "mild" in spices else ("spicy" if spices and all(s == "spicy" for s in spices) else "medium"),
        "likes": dict(likes),
        "vetoes": sorted(v for v in vetoes if v),
        "allergies": allergies,
        # Liked by most, vetoed by nobody.
        "top_cuisines": [c for c, _n in likes.most_common() if c not in vetoes][:3],
    }


def summary_text(combined: dict[str, Any], size: int) -> str:
    parts = [f"{combined['answered']} of {size} answered."]
    if combined["all_veg"]:
        parts.append("Everyone's veg.")
    elif combined["veg_count"]:
        parts.append(f"{combined['veg_count']} veg, so the place needs good veg options.")
    if combined["budget_per_person"]:
        parts.append(f"Budget ₹{combined['budget_per_person']} each.")
    if combined["spice"] == "mild":
        parts.append("Keep it mild.")
    if combined["top_cuisines"]:
        parts.append("Most want " + " or ".join(combined["top_cuisines"][:2]) + ".")
    if combined["vetoes"]:
        parts.append("No " + ", ".join(combined["vetoes"]) + ".")
    if combined["allergies"]:
        parts.append("Allergies — " + "; ".join(combined["allergies"]) + ".")
    return " ".join(parts)


def _matches(cuisine_choice: str, text: str) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text) for a in CUISINE_ALIASES.get(cuisine_choice, ()))


def _per_person_cost(restaurant: dict[str, Any]) -> float | None:
    raw = str(restaurant.get("cost_for_two") or "")
    digits = re.findall(r"\d+", raw.replace(",", ""))
    return int(digits[0]) / 2 if digits else None


def score_restaurant(restaurant: dict[str, Any], combined: dict[str, Any]) -> float | None:
    """Higher is better; None = doesn't work for this group (closed, vetoed, not veg enough)."""
    if restaurant.get("is_open") is False:
        return None
    text = " ".join([str(restaurant.get("restaurant_name") or ""), *map(str, restaurant.get("cuisines") or [])]).lower()
    if any(_matches(v, text) for v in combined.get("vetoes") or []):
        return None
    if combined.get("all_veg") and restaurant.get("veg") is False:
        return None

    score = 0.0
    for cuisine, votes in (combined.get("likes") or {}).items():
        if _matches(cuisine, text):
            score += 1.5 * votes
    if combined.get("all_veg") and restaurant.get("veg"):
        score += 1.0
    rating = restaurant.get("rating") or 0
    try:
        score += max(0.0, float(rating) - 3.5) * 1.2
    except (TypeError, ValueError):
        pass
    eta = restaurant.get("eta_min")
    if isinstance(eta, (int, float)) and eta > 0:
        score += max(0.0, (60 - eta) / 40) * 0.6
    budget = combined.get("budget_per_person")
    cost = _per_person_cost(restaurant)
    if budget and cost:
        score += 0.8 if cost <= budget else -1.5 * min(2.0, (cost - budget) / budget * 4)
    if restaurant.get("offer"):
        score += 0.3
    return round(score, 3)


def rank_restaurants(restaurants: list[dict[str, Any]], combined: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    seen: set[str] = set()
    scored = []
    for r in restaurants:
        key = str(r.get("restaurant_id") or r.get("restaurant_name") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        s = score_restaurant(r, combined)
        if s is not None:
            scored.append((s, r))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**r, "group_score": s} for s, r in scored[:limit]]


def restaurant_reason(restaurant: dict[str, Any], combined: dict[str, Any]) -> str:
    text = " ".join([str(restaurant.get("restaurant_name") or ""), *map(str, restaurant.get("cuisines") or [])]).lower()
    fans = sum(v for c, v in (combined.get("likes") or {}).items() if _matches(c, text))
    bits = []
    if fans:
        bits.append(f"{fans} vote{'s' if fans != 1 else ''} for its food")
    cost = _per_person_cost(restaurant)
    if cost and combined.get("budget_per_person") and cost <= combined["budget_per_person"]:
        bits.append(f"~₹{int(cost)} each")
    if restaurant.get("eta_min"):
        bits.append(f"{restaurant['eta_min']} min")
    if restaurant.get("rating"):
        bits.append(f"★ {restaurant['rating']}")
    return " · ".join(bits)
