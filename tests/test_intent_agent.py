import unittest
from unittest.mock import patch

from cravr.intentAgent import _important_missing_fields, _question_for, intent_discovery_node
from cravr.schemas import IntentExtraction


class StubStructured:
    def __init__(self, value):
        self._value = value

    def invoke(self, messages):
        return self._value


class StubLLM:
    def __init__(self, extraction: IntentExtraction):
        self._extraction = extraction

    def with_structured_output(self, schema):
        return StubStructured(self._extraction)


class TastePreferenceTests(unittest.TestCase):
    """taste_preference is deliberately optional (per product decision): it
    should be captured and merged whenever the model detects it, but must
    never show up in important_missing_information / trigger a clarification
    question the way craving, dietary_preferences, and budget_inr do."""

    def test_taste_preference_is_never_flagged_as_missing(self):
        missing = _important_missing_fields({"craving": "biryani", "dietary_preferences": ["veg"], "budget_inr": 300})
        self.assertNotIn("taste_preference", missing)

    def test_merges_taste_preference_from_structured_extraction(self):
        state = {
            "messages": [{"role": "user", "content": "I want something spicy and tangy"}],
            "intent_profile": {},
        }
        stub = StubLLM(IntentExtraction(taste_preference=["spicy", "tangy"]))
        with patch("cravr.intentAgent.get_llm", return_value=stub):
            result = intent_discovery_node(state)

        self.assertEqual(sorted(result["intent_profile"]["taste_preference"]), ["spicy", "tangy"])
        self.assertNotIn("taste_preference", result["intent_profile"]["important_missing_information"])

    def test_taste_preference_persists_across_turns_without_being_restated(self):
        state = {
            "messages": [{"role": "user", "content": "500 rupees please"}],
            "intent_profile": {"taste_preference": ["sweet"]},
        }
        # This turn's extraction doesn't mention taste at all -- the earlier
        # answer must survive via merge_profile, not get wiped.
        stub = StubLLM(IntentExtraction(budget_inr=500))
        with patch("cravr.intentAgent.get_llm", return_value=stub):
            result = intent_discovery_node(state)

        self.assertEqual(result["intent_profile"]["taste_preference"], ["sweet"])
        self.assertEqual(result["intent_profile"]["budget_inr"], 500)


class BlockingGapTests(unittest.TestCase):
    """Every entry in important_missing_information costs the user a question
    before they see any food, so only genuine blockers belong there."""

    def test_a_taste_counts_as_a_craving(self):
        """"I want something spicy" IS a craving. Asking "do you have a
        specific craving in mind?" straight afterwards ignores the user."""
        missing = _important_missing_fields({"taste_preference": ["spicy"]})
        self.assertNotIn("craving_or_cuisine", missing)

    def test_no_signal_at_all_still_asks(self):
        self.assertIn("craving_or_cuisine", _important_missing_fields({}))

    def test_budget_never_blocks(self):
        """An optional refinement: used when volunteered, never demanded."""
        missing = _important_missing_fields({"taste_preference": ["spicy"]})
        self.assertNotIn("budget_inr", missing)

    def test_spicy_request_asks_exactly_one_question(self):
        self.assertEqual(
            _important_missing_fields({"taste_preference": ["spicy"]}), ["dietary_preferences"]
        )

    def test_the_question_acknowledges_what_was_already_said(self):
        self.assertEqual(
            _question_for("dietary_preferences", {"taste_preference": ["spicy"]}),
            "Spicy it is — veg or non-veg?",
        )
        self.assertEqual(
            _question_for("dietary_preferences", {"taste_preference": ["spicy"], "cuisine": "Indian"}),
            "Spicy Indian it is — veg or non-veg?",
        )

    def test_a_duplicated_taste_is_not_echoed_twice(self):
        """The extractor sometimes puts "spicy" in both craving and taste."""
        self.assertEqual(
            _question_for("dietary_preferences", {"taste_preference": ["spicy"], "craving": "spicy"}),
            "Spicy it is — veg or non-veg?",
        )

    def test_falls_back_to_a_plain_question_when_nothing_is_known(self):
        self.assertEqual(_question_for("dietary_preferences", {}), "Veg or non-veg?")


class ClarificationAnswerCaptureTests(unittest.TestCase):
    """The root cause of the repeated-question loop.

    `decision_agent` knows exactly which field it just asked about, but the
    answer used to reach the profile only via `llama3.2:1b`'s extraction --
    which routinely dropped a bare "Indian" / "vegetarian" / "1000", so the
    field stayed missing and the same question came back round after round.
    `pending_clarification_field` closes that loop deterministically.
    """

    def _run(self, answer: str, field: str, extraction: IntentExtraction, profile=None):
        state = {
            "messages": [
                {"role": "user", "content": "I want to eat something spicy"},
                {"role": "assistant", "content": "Do you have a specific craving or cuisine in mind?"},
                {"role": "user", "content": answer},
            ],
            "intent_profile": profile if profile is not None else {"taste_preference": ["spicy"]},
            "pending_clarification_field": field,
        }
        with patch("cravr.intentAgent.get_llm", return_value=StubLLM(extraction)):
            return intent_discovery_node(state)

    def test_cuisine_answer_lands_even_when_extraction_misses_it(self):
        result = self._run("Indian", "craving_or_cuisine", IntentExtraction(taste_preference=["spicy"]))
        self.assertEqual(result["intent_profile"]["cuisine"], "Indian")
        self.assertNotIn(
            "craving_or_cuisine", result["intent_profile"]["important_missing_information"]
        )

    def test_dietary_answer_lands_even_when_extraction_misses_it(self):
        result = self._run("vegetarian", "dietary_preferences", IntentExtraction())
        self.assertEqual(result["intent_profile"]["dietary_preferences"], ["vegetarian"])

    def test_non_veg_answer_is_not_read_as_vegetarian(self):
        result = self._run("non-veg", "dietary_preferences", IntentExtraction())
        self.assertEqual(result["intent_profile"]["dietary_preferences"], ["non-vegetarian"])

    def test_budget_answer_is_parsed_from_free_text(self):
        result = self._run("around 1,000 rupees", "budget_inr", IntentExtraction())
        self.assertEqual(result["intent_profile"]["budget_inr"], 1000)

    def test_no_preference_answer_is_not_stored_as_a_literal_value(self):
        """"anything" as a cuisine would be filtered on literally and match
        nothing; it means the constraint is absent, not that it is "anything"."""
        result = self._run("anything", "craving_or_cuisine", IntentExtraction())
        self.assertIsNone(result["intent_profile"].get("cuisine"))
        self.assertIsNone(result["intent_profile"].get("craving"))

    def test_unfilterable_answer_is_kept_as_a_free_text_craving(self):
        result = self._run("biryani", "craving_or_cuisine", IntentExtraction(), profile={})
        self.assertEqual(result["intent_profile"]["craving"], "biryani")

    def test_a_taste_answer_becomes_a_filterable_tag_not_a_craving(self):
        """craving="something spicy" is discarded when menu filters are built,
        so the user's whole stated preference would silently evaporate."""
        result = self._run("something spicy", "craving_or_cuisine", IntentExtraction(), profile={})
        self.assertEqual(result["intent_profile"]["taste_preference"], ["spicy"])
        self.assertIsNone(result["intent_profile"].get("craving"))

    def test_a_taste_and_a_cuisine_in_one_answer_both_land(self):
        result = self._run("spicy Thai", "craving_or_cuisine", IntentExtraction(), profile={})
        self.assertEqual(result["intent_profile"]["taste_preference"], ["spicy"])
        self.assertEqual(result["intent_profile"]["cuisine"], "spicy Thai")

    def test_a_negated_taste_reads_as_mild_not_spicy(self):
        result = self._run("nothing too spicy", "craving_or_cuisine", IntentExtraction(), profile={})
        self.assertEqual(result["intent_profile"]["taste_preference"], ["mild"])

    def test_extraction_wins_when_it_succeeds(self):
        result = self._run(
            "South Indian", "craving_or_cuisine", IntentExtraction(cuisine="South Indian")
        )
        self.assertEqual(result["intent_profile"]["cuisine"], "South Indian")

    def test_pending_field_is_cleared_after_being_consumed(self):
        result = self._run("Indian", "craving_or_cuisine", IntentExtraction())
        self.assertIsNone(result["pending_clarification_field"])
        # A clarify-and-retry loop must NOT reset the streak counters mid-flight.
        self.assertNotIn("clarification_rounds", result)

    def test_a_fresh_turn_resets_the_clarification_streak(self):
        """Previously the counter only went back to 0 on give-up, so an
        unanswerable question was re-asked in streak after streak."""
        state = {
            "messages": [{"role": "user", "content": "I want pizza"}],
            "intent_profile": {},
        }
        with patch("cravr.intentAgent.get_llm", return_value=StubLLM(IntentExtraction(craving="pizza"))):
            result = intent_discovery_node(state)

        self.assertEqual(result["clarification_rounds"], 0)
        self.assertEqual(result["asked_clarification_fields"], [])


if __name__ == "__main__":
    unittest.main()
