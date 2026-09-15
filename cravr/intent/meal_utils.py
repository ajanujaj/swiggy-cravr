import re
from dataclasses import dataclass
from datetime import time

MAX_INTENT_CLARIFICATION_ROUNDS = 3

KNOWN_CUISINE_WORDS = (
    "north indian", "south indian", "indian", "chinese", "italian", "mexican",
    "japanese", "thai", "continental", "punjabi", "mughlai", "asian",
)

NO_PREFERENCE_ANSWERS = (
    "anything", "any", "no preference", "nothing", "none", "doesn't matter",
    "does not matter", "dont matter", "don't matter", "you decide", "you choose",
    "surprise me", "whatever", "up to you", "no idea", "not sure", "idk",
)

TASTE_CUES = {
    "spicy": ("spicy", "hot", "fiery", "chilli", "chili", "masala"),
    "sweet": ("sweet", "sugary", "sweet tooth"),
    "savory": ("savory", "savoury", "salty", "umami", "hearty"),
    "tangy": ("tangy", "tart", "sour", "citrusy", "zesty"),
}

MILD_PATTERN = re.compile(
    r"\b(?:not|no|nothing|less|avoid|without)\s+(?:too\s+|very\s+)?(?:spicy|hot|chill?i)\b"
)
MILD_WORDS = ("mild", "bland")

VEG_ANSWERS = ("vegetarian", "veg", "pure veg", "no meat", "veggie")
NON_VEG_ANSWERS = ("non-veg", "non veg", "nonveg", "non-vegetarian", "non vegetarian",
                   "no-veg", "no veg", "noveg",
                   "meat", "chicken", "mutton", "fish", "egg", "eggetarian")

CRAVING_TAG_CUES = {
    "comfort-food": ("comfort", "homely", "home food", "cosy", "cozy"),
    "lightweight": ("light", "lightweight"),
    "high-protein": ("protein", "high protein"),
    "low-calorie": ("low calorie", "low-cal", "diet", "healthy"),
    "dessert": ("dessert", "pudding", "cake", "ice cream", "brownie", "pastry"),
}

HUNGER_BY_INTENT = {
    "breakfast": "medium",
    "lunch": "high",
    "quick_lunch": "medium",
    "dinner": "high",
    "celebration": "high",
    "snack": "low",
    "quick_bite": "low",
    "late_night_craving": "low",
}
DEFAULT_HUNGER = "medium"

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


# Ordered windows covering all 24 hours. The late-night window wraps midnight, hence the
# explicit `spans_midnight` branch instead of a naive `start <= t <= end`. Option sets encode
# a real asymmetry -- at 3pm "lunch" is no longer on offer, so fewer options beat an impossible one.
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