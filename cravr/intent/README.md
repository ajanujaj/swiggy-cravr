# `cravr/intent/` — understand what the user wants

Turns the user's message into a structured **intent profile** and owns the clarification loop: it pauses the graph with `interrupt()` to ask a question (with tappable options) whenever it can't yet tell what to search for.

## `intentAgent.py`

**Graph node**

| Function | Usage |
|---|---|
| `intent_discovery_node(state)` | **Graph node `intent_discovery`.** (1) LLM extraction with `IntentExtraction` + `INTENT_DISCOVERY_PROMPT`, merged into the running `intent_profile`. (2) Deterministic derivations (`_derive`). (3) Up to `MAX_INTENT_CLARIFICATION_ROUNDS` questions via `interrupt()` for each missing field, applying each answer directly. If the user picks "something else", asks one open follow-up. Returns `{"intent_profile": ...}` plus the Q&A transcript. |

**Helpers**

| Function | Usage |
|---|---|
| `extract_tastes(text)` | Maps free wording ("fiery", "sweet tooth") onto the menu's taste tags. Public, reused by `recommend`. |
| `_derive(profile, now=None)` | Fills what code can work out: `meal_time` from the clock, a default `meal_intent`, `flavor`, and `hunger_level` (unless the user stated it). |
| `_derive_flavor(profile)` | Normalises craving/cuisine text into one menu flavour tag ("spicy Indian" → `spicy`). |
| `_derive_dietary_preference(text)` | Explicit veg / non-veg signal from the latest message (longest phrase first, so "non-veg" beats "veg"). |
| `_mentions_dietary_word(text, phrases)` | Whole-word match, so "veg" doesn't fire inside "vegetable". |
| `_has_flavour_signal(profile)` | `True` once there's anything to search with (a dish, cuisine or taste). |
| `_missing_fields(profile)` | What still needs asking, in order: `category` (only when there's no signal at all), then `craving_or_cuisine`. |
| `_question_for(field, now, profile)` | The question text plus LLM-suggested options (`options.suggest_options`) with static fallbacks per category. |
| `_apply_answer(profile, field, answer, now)` | Writes a reply straight into the field that was asked (no extra LLM round-trip), handling "no preference" answers. When the answer to `craving_or_cuisine` names a cuisine and no craving is set yet, a free-typed category (one that isn't a known `CRAVING_TAG_CUES` tag) is carried over as the `craving`, so it isn't lost. |
| `_is_no_preference(answer)` | Detects "anything", "no preference", etc. |

## `meal_clock.py`
Time-of-day logic (always in IST via `clock.now_ist`).

| Function | Usage |
|---|---|
| `current_meal_window(now=None)` | The `MealWindow` (breakfast / lunch / evening / dinner / late_night) containing `now`. |
| `meal_intent_options(now=None)` | That window's choices as `[{"value","label"}]`. |
| `meal_intent_question(now=None)` | A question that names the time, e.g. "It's 10:04 PM — ...?". |
| `resolve_meal_intent(answer, now=None)` | Maps a reply (position, value, label or a distinctive word) to a meal intent value, or `None`. |
| `menu_meal_type(meal_intent, now=None)` | Translates a meal intent into the menu's `meal_type` filter (or `None`). |

## `meal_utils.py`
Vocabulary and constants shared by the intent code (and re-exported for `recommend`).

| Name | Usage |
|---|---|
| `MealWindow` | Dataclass: `name`, `start`, `end`, `options`; `default_intent` (first option) and `spans_midnight()` for the late-night window. |
| `MEAL_WINDOWS` | The ordered windows covering all 24 hours. |
| `MAX_INTENT_CLARIFICATION_ROUNDS` | Cap on questions per turn (3). |
| `KNOWN_CUISINE_WORDS`, `TASTE_CUES`, `CRAVING_TAG_CUES`, `MILD_PATTERN`, `MILD_WORDS` | Word lists used to spot cuisines, tastes and craving categories. |
| `VEG_ANSWERS`, `NON_VEG_ANSWERS`, `NO_PREFERENCE_ANSWERS` | Recognised replies. |
| `HUNGER_BY_INTENT`, `DEFAULT_HUNGER` | Default hunger level per meal intent. |

## `prompt.py`
`INTENT_DISCOVERY_PROMPT`: the system prompt for extraction. It tells the model to extract only what was said and never recommend or invent values.

## `__init__.py`
Re-exports the node, `extract_tastes`, the meal-clock functions and the vocabulary constants.
