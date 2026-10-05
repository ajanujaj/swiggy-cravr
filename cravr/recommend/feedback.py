"""Learning from "no" (feature 2).

A rejection used to just show the next three. Now Cravr works out *why*
-- from the words ("too expensive", "something lighter", "takes too
long") or by asking with four tappable reasons -- then changes the search
to fix exactly that, and remembers the reason (userPreference/taste.py).

Adjustments accumulate over one set of results: "too pricey" then "too
slow" keeps both the lower price cap and the delivery-time cap.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Any

TOO_PRICEY = "too_pricey"
TOO_HEAVY = "too_heavy"
TOO_SLOW = "too_slow"
TOO_SPICY = "too_spicy"
NOT_SPICY_ENOUGH = "not_spicy_enough"
NOT_FOR_ME = "not_for_me"

# Asked when someone just says "no" -- one tap tells us what to fix.
REASON_QUESTION = "No problem — what's off about these?"
REASON_OPTIONS = [
    {"value": "too pricey", "label": "Too pricey"},
    {"value": "too heavy", "label": "Too heavy"},
    {"value": "takes too long", "label": "Takes too long"},
    {"value": "not in the mood for these", "label": "Not in the mood for these"},
]

# Checked in order. Spice: only the phrases the taste parser misreads live here
# ("too spicy" reads as *spicy*, "not spicy enough"/"bland" as *mild*); plain
# "milder" or "spicier" already work as a changed request through intent.
_REASON_PATTERNS: list[tuple[str, re.Pattern]] = [
    (NOT_SPICY_ENOUGH, re.compile(r"\b(not spicy enough|too bland|bland)\b")),
    (TOO_SPICY, re.compile(r"\b(too spicy|too hot)\b")),
    (TOO_PRICEY, re.compile(r"\b(too )?(expensive|pricey|costly|pricy)\b|\bcheaper\b|\btoo much money\b|"
                            r"\bover (my )?budget\b|\bless expensive\b|\bprice\b")),
    (TOO_HEAVY, re.compile(r"\b(too )?(heavy|oily|greasy|rich)\b|\blighter\b|\bhealthier\b|"
                           r"\bsomething light\b")),
    (TOO_SLOW, re.compile(r"\b(too )?slow\b|\btakes? (too )?long\b|\btoo late\b|\bfaster\b|\bquicker\b|"
                          r"\bdelivery time\b")),
    (NOT_FOR_ME, re.compile(r"\bnot in the mood\b|\b(don'?t|do not) (like|want)\b|\bnot feeling\b|"
                            r"\bhate\b|\bsomething different\b|\bbored of\b|\bnot for me\b")),
]

LEAD_INS = {
    TOO_PRICEY: "Got it — lighter on the wallet:",
    TOO_HEAVY: "Got it — something lighter:",
    TOO_SLOW: "Got it — these arrive faster:",
    TOO_SPICY: "Got it — milder options:",
    NOT_SPICY_ENOUGH: "Got it — turning up the heat:",
    NOT_FOR_ME: "Fair enough — something different:",
}
NOTHING_FOUND = {
    TOO_PRICEY: "I couldn't find anything cheaper that still fits.",
    TOO_HEAVY: "I couldn't find a lighter option that still fits.",
    TOO_SLOW: "I couldn't find anything that arrives faster.",
    TOO_SPICY: "I couldn't find a milder option that still fits.",
    NOT_SPICY_ENOUGH: "I couldn't find anything spicier that still fits.",
    NOT_FOR_ME: "That's everything that fits what you asked for.",
}

HEAVY_WORDS = ("butter", "cheese", "cheesy", "fried", "fry", "makhani", "creamy", "biryani",
               "bhature", "pizza", "burger", "loaded", "double", "korma", "malai", "naan", "paratha")
LIGHT_WORDS = ("salad", "soup", "grilled", "steamed", "idli", "poha", "sprouts", "bowl", "tikka",
               "wrap", "khichdi", "dal", "upma", "fruit", "curd")
SPICY_WORDS = ("spicy", "chilli", "chili", "peri peri", "schezwan", "szechuan", "mirchi", "hot",
               "kolhapuri", "chettinad", "vindaloo", "andhra", "65")

_MIN_PRICE_CAP = 60
_DEFAULT_ETA_CAP = 30


def infer_reason(text: str) -> str | None:
    normalized = f" {(text or '').strip().lower()} "
    for reason, pattern in _REASON_PATTERNS:
        if pattern.search(normalized):
            return reason
    return None


def _prices(dishes: list[dict[str, Any]]) -> list[float]:
    out = []
    for d in dishes:
        try:
            p = float(d.get("price_inr") or 0)
        except (TypeError, ValueError):
            continue
        if p > 0:
            out.append(p)
    return out


def adjustments_for(reason: str, offered: list[dict[str, Any]], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """What to change so the next batch fixes `reason`, on top of earlier fixes."""
    adj = dict(previous or {})
    if reason == TOO_PRICEY:
        prices = _prices(offered)
        if prices:
            cap = max(_MIN_PRICE_CAP, int(median(prices) * 0.8))
            adj["max_price_inr"] = min(cap, adj.get("max_price_inr") or cap)
    elif reason == TOO_HEAVY:
        adj["avoid_heavy"] = True
    elif reason == TOO_SLOW:
        etas = [d["eta_min"] for d in offered if isinstance(d.get("eta_min"), (int, float))]
        cap = max(15, int(min(etas)) - 5) if etas else _DEFAULT_ETA_CAP
        adj["max_eta_min"] = min(cap, adj.get("max_eta_min") or cap)
    elif reason == TOO_SPICY:
        adj["spice"] = "mild"
    elif reason == NOT_SPICY_ENOUGH:
        adj["spice"] = "spicy"
    return adj


def adjusted_brief(brief: dict[str, Any], adj: dict[str, Any]) -> dict[str, Any]:
    """The search brief with the adjustments the search itself can apply."""
    filters = dict(brief.get("filters") or {})
    if adj.get("max_price_inr"):
        current = filters.get("max_price_inr")
        filters["max_price_inr"] = min(adj["max_price_inr"], current) if current else adj["max_price_inr"]
    tags = [t for t in str(filters.get("tags") or "").split(",") if t and t not in ("spicy", "mild")]
    if adj.get("spice") in ("mild", "spicy"):
        tags.append(adj["spice"])
    if adj.get("avoid_heavy") and "lightweight" not in tags and not filters.get("craving"):
        tags.append("lightweight")
    if tags:
        filters["tags"] = ",".join(tags)
    return {**brief, "filters": filters}


def _has_word(name: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", name) for w in words)


def keeps(dish: dict[str, Any], adj: dict[str, Any]) -> bool:
    """Checks the search can't do itself (Swiggy has no price/heat/ETA filters)."""
    name = str(dish.get("item_name") or "").lower()
    cap = adj.get("max_price_inr")
    if cap:
        try:
            if float(dish.get("price_inr") or 0) > cap:
                return False
        except (TypeError, ValueError):
            pass
    eta_cap = adj.get("max_eta_min")
    eta = dish.get("eta_min")
    if eta_cap and isinstance(eta, (int, float)) and eta > eta_cap:
        return False
    if adj.get("avoid_heavy") and _has_word(name, HEAVY_WORDS):
        return False
    if adj.get("spice") == "mild" and _has_word(name, SPICY_WORDS):
        return False
    return True


def preference_key(dish: dict[str, Any], adj: dict[str, Any]) -> int:
    """Nudge within the kept dishes toward what was asked (lighter / spicier)."""
    name = str(dish.get("item_name") or "").lower()
    score = 0
    if adj.get("avoid_heavy") and _has_word(name, LIGHT_WORDS):
        score += 1
    if adj.get("spice") == "spicy" and _has_word(name, SPICY_WORDS):
        score += 1
    return score
