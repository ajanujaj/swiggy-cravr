"""Pydantic schemas for each agent's LLM output.

Passed to `.with_structured_output(...)` so LangChain enforces the shape via
the model's native tool-calling/JSON-schema support, instead of us asking the
model to print JSON text and hoping `json.loads` doesn't blow up on it. This
is what actually eliminates the "predefined fallback response" failure mode:
a model that supports structured output either returns a schema-valid object
or the call raises loudly, so we're never silently working with malformed
free-text pretending to be JSON.

Fields the code computes itself (missing_information, important_missing_*,
clarification_questions) are deliberately NOT part of these schemas -- the
model only has to describe what's in the current message, and decisionAgent
/ intentAgent / ContextAgent figure out what's still missing deterministically
from the merged profile (see profile_utils.merge_profile).
"""

from __future__ import annotations

from typing import Literal, Optional

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

class ContextExtraction(BaseModel):
    """Structured delivery-context facts extracted from the conversation."""

    location: Optional[str] = None
    weather: Optional[str] = None
    delivery_area: Optional[str] = None
    max_delivery_minutes: Optional[int] = "under 30 minutes"

class DecisionOutput(BaseModel):
    """The decision agent's verdict on whether to recommend now."""

    ready_to_recommend: bool
    decision: Literal["recommend", "clarify"]
    clarification_message: Optional[str] = None

class SuggestedOptions(BaseModel):
    """Short, tappable answer choices for a clarification question.

    Used by `cravr.options.suggest_options` so any agent can turn a
    free-text question into a numbered menu -- the whole point of a
    recommender for someone who "doesn't know what to eat" is that they
    should have to type as little as possible.
    """

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

class UserProfile(BaseModel):
    """Structured facts about the user, extracted from the conversation and/or
    fetched from the MCP.
    """
    
    dietary_preference: Optional[str] = None
    allergies: Optional[str] = None
    default_budget_inr: Optional[str] = None
