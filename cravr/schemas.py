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

    mood: Optional[str] = None
    hunger_level: Optional[Literal["low", "medium", "high"]] = None
    occasion: Optional[str] = None
    meal_time: Optional[str] = None
    craving: Optional[str] = None
    cuisine: Optional[str] = None
    taste_preference: list[Literal["spicy", "mild", "sweet", "savory", "tangy"]] = Field(
        default_factory=list
    )
    dietary_preferences: list[str] = Field(default_factory=list)
    allergies_or_avoidances: list[str] = Field(default_factory=list)
    budget_inr: Optional[int] = None
    minimum_rating: Optional[float] = None
    delivery_preference: Optional[Literal["delivery", "dine_in"]] = None
    location: Optional[str] = None
    other_constraints: list[str] = Field(default_factory=list)


class ContextExtraction(BaseModel):
    """Structured delivery-context facts extracted from the conversation."""

    location: Optional[str] = None
    weather: Optional[str] = None


class DecisionOutput(BaseModel):
    """The decision agent's verdict on whether to recommend now."""

    ready_to_recommend: bool
    decision: Literal["recommend", "clarify"]
    clarification_message: Optional[str] = None
