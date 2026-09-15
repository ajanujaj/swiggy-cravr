"""Pydantic schemas for each agent's LLM output. Passed to
`.with_structured_output(...)` so LangChain enforces the shape via the
model's native tool-calling/JSON-schema support -- a model either returns a
schema-valid object or raises loudly, never silent malformed JSON.

Computed fields (missing_information, important_missing_*,
clarification_questions) are deliberately not part of these schemas -- the
model only describes what's in the message; intentAgent derives what's
missing deterministically (see profile_utils.merge_profile).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class IntentExtraction(BaseModel):
    """Structured facts extracted from the user's food-ordering message(s)."""

    hunger_level: Optional[str] = Field(None, description="The user's hunger level, e.g. low, medium, high.")
    meal_intent: Optional[str] = Field(None, description="The user's intent for the meal, e.g. breakfast, lunch, dinner, snack,late_night_craving, quick_bite, quick_lunch, celebration etc.")
    meal_time: Optional[str] = None
    craving: Optional[str] = None
    craving_tag: Optional[str] = Field(None, description="A normalized tag for the craving, e.g. comfort-food, lightweight, high-protein, low-calorie, dessert. Leave null if none of them fit.")
    flavor: Optional[str] = Field(None, description="A normalized tag for the craving, e.g. spicy, sweet, savory, etc.")
    cuisine: Optional[str] = None

class SuggestedOptions(BaseModel):
    """Short, tappable answer choices for a clarification question, turning
    free text into a numbered menu the user can pick from."""

    options: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 5 short (1-4 word), concrete, mutually exclusive answer "
            "choices the user could pick instead of typing -- use all 5 "
            "when the context genuinely supports that many distinct, "
            "plausible answers, fewer only when it doesn't. Never include a "
            "catch-all like 'other' or 'something else' -- that is added "
            "separately."
        ),
    )
