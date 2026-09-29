"""Context signals (feature 5): weather, mood and occasion.

Each turns into a *theme* -- a label and a few dishes that suit it -- used
by the dishes-first opening in place of the plain time-of-day card.
Priority: what the user said about themselves (mood) beats the weather,
which beats the calendar (occasion).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

MOODS: dict[str, dict[str, Any]] = {
    "hungover": {
        "cues": ("hungover", "hangover", "hung over", "too many drinks", "drank too much"),
        "label": "Hangover helper", "queries": ["aloo paratha", "chole bhature", "burger", "nimbu soda"],
    },
    "post_workout": {
        "cues": ("gym", "workout", "work out", "worked out", "post workout", "after my run", "protein"),
        "label": "Post-workout protein", "queries": ["grilled chicken", "paneer tikka", "egg bhurji", "protein bowl"],
    },
    "unwell": {
        "cues": ("sick", "unwell", "fever", "cold and cough", "upset stomach", "not feeling well"),
        "label": "Easy on the stomach", "queries": ["khichdi", "soup", "idli", "curd rice"],
    },
    "low": {
        "cues": ("sad", "stressed", "bad day", "rough day", "feeling low", "feeling down", "exhausted"),
        "label": "Comfort food", "queries": ["pizza", "maggi", "butter chicken", "brownie"],
    },
    "celebrating": {
        "cues": ("celebrate", "celebrating", "celebration", "party", "birthday", "promotion", "anniversary"),
        "label": "Something to celebrate with", "queries": ["biryani", "pizza", "cake", "kebab platter"],
    },
    "late_study": {
        "cues": ("studying", "exam", "deadline", "working late", "all nighter", "all-nighter"),
        "label": "Fuel for a long night", "queries": ["maggi", "cold coffee", "sandwich", "rolls"],
    },
}

WEATHER_THEMES = {
    "rain": {"label": "Rainy day pick", "queries": ["pakoda", "samosa", "masala chai", "maggi"]},
    "hot": {"label": "Beat the heat", "queries": ["lassi", "salad", "curd rice", "ice cream"]},
    "cold": {"label": "Something warm", "queries": ["soup", "momos", "masala chai", "dal makhani"]},
}

OCCASIONS = {
    "match_night": {"label": "Match-night snacks", "queries": ["pizza", "chicken wings", "nachos", "fries"]},
    "weekend_night": {"label": "Weekend treat", "queries": ["pizza", "burger", "biryani", "shawarma"]},
    "sunday_brunch": {"label": "Sunday brunch", "queries": ["chole bhature", "aloo paratha", "pancakes", "poori bhaji"]},
}
_MATCH_RE = re.compile(r"\b(match|ipl|cricket|world cup|game night|football)\b", re.IGNORECASE)


def detect_mood(text: str | None) -> str | None:
    normalized = f" {(text or '').lower()} "
    for key, mood in MOODS.items():
        if any(re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", normalized) for cue in mood["cues"]):
            return key
    return None


def detect_occasion(now: datetime, text: str | None = None) -> str | None:
    if text and _MATCH_RE.search(text):
        return "match_night"
    weekday, hour = now.weekday(), now.hour  # Mon=0
    if weekday in (4, 5) and hour >= 19:     # Friday / Saturday evening
        return "weekend_night"
    if weekday == 6 and 9 <= hour < 13:
        return "sunday_brunch"
    return None


def theme(mood: str | None = None, weather: dict[str, Any] | None = None,
          occasion: str | None = None) -> dict[str, Any] | None:
    """The one context theme to use, or None when nothing notable is going on."""
    if mood and mood in MOODS:
        return {"kind": "mood", "key": mood, "label": MOODS[mood]["label"], "queries": list(MOODS[mood]["queries"])}
    condition = (weather or {}).get("condition")
    if condition in WEATHER_THEMES:
        t = WEATHER_THEMES[condition]
        return {"kind": "weather", "key": condition, "label": t["label"], "queries": list(t["queries"])}
    if occasion in OCCASIONS:
        t = OCCASIONS[occasion]
        return {"kind": "occasion", "key": occasion, "label": t["label"], "queries": list(t["queries"])}
    return None
