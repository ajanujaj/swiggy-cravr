"""Clock-driven meal windows.

`meal_intent` is the one required field the user should almost never have to
*type*. The wall clock already narrows it to two or three plausible answers --
at 10pm nobody is ordering breakfast -- so this module turns the current time
into a short menu of choices instead of an open question.

Deliberately pure and dependency-free: it takes a `datetime` (defaulting to
now) and returns data. No LLM, no state, no I/O -- which is what makes the
window boundaries testable at 05:00:00 and 04:59:59 rather than "whenever the
suite happens to run".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time


@dataclass(frozen=True)
class MealWindow:
    """One slice of the day and the meal intents plausible inside it."""

    name: str
    start: time
    end: time
    # First option is the "obvious" answer for this window and is used as the
    # fallback when the user's reply can't be matched to anything.
    options: tuple[tuple[str, str], ...]  # (value, human label)

    @property
    def default_intent(self) -> str:
        return self.options[0][0]

    def spans_midnight(self) -> bool:
        return self.start > self.end


# Ordered windows covering all 24 hours. The late-night window wraps midnight,
# which is why containment is checked with an explicit `spans_midnight` branch
# rather than a naive `start <= t <= end`.
#
# The option sets encode a real asymmetry: at lunch and dinner the user might
# want a full meal OR something small, but at 3pm "lunch" is no longer on the
# table. Offering an impossible option is worse than offering fewer.
MEAL_WINDOWS: tuple[MealWindow, ...] = (
    MealWindow(
        name="breakfast",
        start=time(5, 0),
        end=time(10, 29),
        options=(("breakfast", "Breakfast"), ("quick_bite", "Just a quick bite")),
    ),
    MealWindow(
        name="lunch",
        start=time(10, 30),
        end=time(15, 29),
        options=(
            ("lunch", "Lunch"),
            ("quick_lunch", "A quick lunch"),
            ("snack", "Something snacky"),
        ),
    ),
    MealWindow(
        name="evening",
        start=time(15, 30),
        end=time(18, 59),
        options=(("snack", "An evening snack"), ("quick_bite", "Just a quick bite")),
    ),
    MealWindow(
        name="dinner",
        start=time(19, 0),
        end=time(22, 59),
        options=(
            ("dinner", "Dinner"),
            ("quick_bite", "Something quick"),
            ("snack", "Just a snack"),
        ),
    ),
    MealWindow(
        name="late_night",
        start=time(23, 0),
        end=time(4, 59),
        options=(
            ("late_night_craving", "A late-night craving"),
            ("snack", "Just a snack"),
            ("quick_bite", "Something quick"),
        ),
    ),
)


def current_meal_window(now: datetime | None = None) -> MealWindow:
    """Return the window the given moment falls in (defaults to right now)."""
    moment = (now or datetime.now()).time()

    for window in MEAL_WINDOWS:
        if window.spans_midnight():
            if moment >= window.start or moment <= window.end:
                return window
        elif window.start <= moment <= window.end:
            return window

    # Unreachable while the windows tile the full 24 hours, but a silent
    # KeyError at 3am is a bad way to find out someone edited the table.
    raise ValueError(f"No meal window covers {moment!r}; MEAL_WINDOWS has a gap.")


def meal_intent_options(now: datetime | None = None) -> list[dict[str, str]]:
    """The choices to put in front of the user, as interrupt-payload dicts."""
    return [
        {"value": value, "label": label}
        for value, label in current_meal_window(now).options
    ]


def meal_intent_question(now: datetime | None = None) -> str:
    """Phrase the question with the time in it, so the options make sense.

    Naming the hour is what makes a three-way choice feel considered rather
    than arbitrary -- "It's 10:04 PM" explains why breakfast isn't on offer.
    """
    moment = now or datetime.now()
    window = current_meal_window(moment)
    labels = [label for _value, label in window.options]
    clock = moment.strftime("%I:%M %p").lstrip("0")

    if len(labels) == 1:
        return f"It's {clock} — {labels[0].lower()}?"

    head = ", ".join(labels[:-1])
    return f"It's {clock} — {head.lower()}, or {labels[-1].lower()}?"


def resolve_meal_intent(answer: str, now: datetime | None = None) -> str | None:
    """Map a free-text reply onto one of the window's option values.

    Accepts a position ("2"), the value ("quick_bite"), the label ("just a
    snack"), or any distinctive word from the label ("snack"). Returns None
    when nothing matches, so the caller can decide whether to re-ask or fall
    back -- silently defaulting here would make a typo look like a choice.
    """
    normalized = (answer or "").strip().lower().rstrip(".!")
    if not normalized:
        return None

    options = current_meal_window(now).options

    if normalized.isdigit():
        position = int(normalized)
        if 1 <= position <= len(options):
            return options[position - 1][0]
        return None

    # Exact value or label first -- "snack" should pick the option whose value
    # is literally "snack", not merely one whose label happens to contain it.
    for value, label in options:
        if normalized in {value, value.replace("_", " "), label.lower()}:
            return value

    for value, label in options:
        words = {w for w in label.lower().replace("-", " ").split() if len(w) > 3}
        if any(word in normalized for word in words):
            return value

    # Vocabulary the user might reach for that isn't in this window's labels.
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


# What the menu can actually filter on. `menu.csv`'s `meal_type` column holds
# only "Breakfast" (32 rows) and "All Day" (48), so breakfast is the single
# meal_type worth constraining on -- and only in the morning. Every other
# window would either match everything or wrongly exclude the All Day items.
def menu_meal_type(meal_intent: str | None, now: datetime | None = None) -> str | None:
    """Translate a meal_intent into the menu's `meal_type`, or None for no filter."""
    if (meal_intent or "").strip().lower() == "breakfast":
        return "Breakfast"
    return None
