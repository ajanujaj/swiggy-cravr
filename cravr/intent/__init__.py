"""Intent discovery: turns the user's message into a structured intent profile.

Re-exported here so callers import from the *package* (`from .intent import
intent_discovery_node`) rather than reaching into the module. The vocabulary
constants are part of the public surface too: `recommend` reuses them to tell a
reply that changes the request ("make it non-veg") from one that merely reacts
to it ("3"), and duplicating those word lists there would let the two drift.
"""

from .intentAgent import (
    KNOWN_CUISINE_WORDS,
    MAX_INTENT_CLARIFICATION_ROUNDS,
    NON_VEG_ANSWERS,
    REQUIRED_FIELDS,
    VEG_ANSWERS,
    extract_tastes,
    intent_discovery_node,
)
from .meal_clock import (
    MEAL_WINDOWS,
    current_meal_window,
    meal_intent_options,
    meal_intent_question,
    menu_meal_type,
    resolve_meal_intent,
)

__all__ = [
    "KNOWN_CUISINE_WORDS",
    "MAX_INTENT_CLARIFICATION_ROUNDS",
    "MEAL_WINDOWS",
    "NON_VEG_ANSWERS",
    "REQUIRED_FIELDS",
    "VEG_ANSWERS",
    "current_meal_window",
    "extract_tastes",
    "intent_discovery_node",
    "meal_intent_options",
    "meal_intent_question",
    "menu_meal_type",
    "resolve_meal_intent",
]
