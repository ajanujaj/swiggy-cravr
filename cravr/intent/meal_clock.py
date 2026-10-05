from __future__ import annotations
from datetime import datetime
from ..clock import now_ist
from .meal_utils import MealWindow, MEAL_WINDOWS


def current_meal_window(now: datetime | None = None) -> MealWindow:
    moment = (now or now_ist()).time()

    for window in MEAL_WINDOWS:
        if window.spans_midnight():
            if moment >= window.start or moment <= window.end:
                return window
        elif window.start <= moment <= window.end:
            return window

    raise ValueError(f"No meal window covers {moment!r}; MEAL_WINDOWS has a gap.")


def meal_intent_options(now: datetime | None = None) -> list[dict[str, str]]:
    return [
        {"value": value, "label": label}
        for value, label in current_meal_window(now).options
    ]


def meal_intent_question(now: datetime | None = None) -> str:
    moment = now or now_ist()
    window = current_meal_window(moment)
    labels = [label for _value, label in window.options]
    clock = moment.strftime("%I:%M %p").lstrip("0")

    if len(labels) == 1:
        return f"It's {clock} — {labels[0].lower()}?"

    head = ", ".join(labels[:-1])
    return f"It's {clock} — {head.lower()}, or {labels[-1].lower()}?"


def resolve_meal_intent(answer: str, now: datetime | None = None) -> str | None:
    normalized = (answer or "").strip().lower().rstrip(".!")
    if not normalized:
        return None

    options = current_meal_window(now).options

    if normalized.isdigit():
        position = int(normalized)
        if 1 <= position <= len(options):
            return options[position - 1][0]
        return None

    for value, label in options:
        if normalized in {value, value.replace("_", " "), label.lower()}:
            return value

    for value, label in options:
        words = {w for w in label.lower().replace("-", " ").split() if len(w) > 3}
        if any(word in normalized for word in words):
            return value

    aliases = {
        "dinner": ("dinner", "supper"),
        "lunch": ("lunch",),
        "breakfast": ("breakfast", "brekkie"),
        "snack": ("snack", "snacky", "nibble"),
        "quick_bite": ("quick", "light", "small", "something small"),
        "late_night_craving": ("late night", "midnight", "craving"),
    }
    available = {value for value, _label in options}
    for value, cues in aliases.items():
        if value in available and any(cue in normalized for cue in cues):
            return value

    return None

def menu_meal_type(meal_intent: str | None, now: datetime | None = None) -> str | None:
    if (meal_intent or "").strip().lower() == "breakfast":
        return "Breakfast"
    return None
