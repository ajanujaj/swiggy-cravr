import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from cravr.intent.intentAgent import (
    MAX_INTENT_CLARIFICATION_ROUNDS,
    _apply_answer,
    _derive,
    _missing_fields,
    extract_tastes,
    intent_discovery_node,
)
from cravr.schemas import IntentExtraction


TEN_PM = datetime(2026, 8, 9, 22, 0)


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


class ExplodingLLM:
    def with_structured_output(self, schema):
        raise RuntimeError("model unavailable")


def run_node(state, extraction=None, answers=(), now=TEN_PM):
    """Run the node with the clock frozen and interrupt() scripted."""
    llm = StubLLM(extraction) if extraction is not None else ExplodingLLM()
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = now

    with patch("cravr.intent.intentAgent.get_llm", return_value=llm), \
         patch("cravr.intent.intentAgent.datetime", clock), \
         patch("cravr.intent.intentAgent.interrupt", side_effect=list(answers)) as spy:
        result = intent_discovery_node(state)
    return result, spy


class DerivedFieldTests(unittest.TestCase):
    """Fields the code can work out are never worth an interrupt."""

    def test_meal_time_comes_from_the_clock(self):
        profile = _derive({}, TEN_PM)
        self.assertEqual(profile["meal_time"], "dinner")

    def test_craving_tag_is_derived_from_free_text_craving(self):
        profile = _derive({"craving": "something really spicy"}, TEN_PM)
        self.assertEqual(profile["craving_tag"], "spicy")

    def test_craving_tag_is_derived_from_cuisine_wording_too(self):
        profile = _derive({"cuisine": "spicy Indian"}, TEN_PM)
        self.assertEqual(profile["craving_tag"], "spicy")

    def test_negated_heat_derives_mild_not_spicy(self):
        profile = _derive({"craving": "nothing too spicy"}, TEN_PM)
        self.assertEqual(profile["craving_tag"], "mild")

    def test_non_taste_craving_maps_to_a_real_menu_tag(self):
        profile = _derive({"craving": "comfort food"}, TEN_PM)
        self.assertEqual(profile["craving_tag"], "comfort-food")

    def test_hunger_level_defaults_from_meal_intent(self):
        self.assertEqual(_derive({"meal_intent": "snack"}, TEN_PM)["hunger_level"], "low")
        self.assertEqual(_derive({"meal_intent": "dinner"}, TEN_PM)["hunger_level"], "high")

    def test_stated_hunger_level_beats_the_default(self):
        profile = _derive(
            {"meal_intent": "snack", "hunger_level": "high", "hunger_level_stated": True},
            TEN_PM,
        )
        self.assertEqual(profile["hunger_level"], "high")

    def test_an_earlier_default_does_not_freeze_the_later_real_value(self):
        """The first derive runs before meal_intent is known and writes the
        placeholder; answering "dinner" must still upgrade it to high."""
        placeholder = _derive({}, TEN_PM)
        self.assertEqual(placeholder["hunger_level"], "medium")

        answered = _derive({**placeholder, "meal_intent": "dinner"}, TEN_PM)
        self.assertEqual(answered["hunger_level"], "high")


class MissingFieldTests(unittest.TestCase):
    def test_craving_alone_satisfies_the_flavour_requirement(self):
        self.assertNotIn("craving_or_cuisine", _missing_fields({"craving": "biryani"}))

    def test_cuisine_alone_satisfies_the_flavour_requirement(self):
        self.assertNotIn("craving_or_cuisine", _missing_fields({"cuisine": "Thai"}))

    def test_craving_tag_alone_satisfies_it(self):
        """'something spicy' is a real request, not an empty one."""
        self.assertNotIn("craving_or_cuisine", _missing_fields({"craving_tag": "spicy"}))

    def test_empty_profile_is_missing_both_requirements(self):
        self.assertEqual(_missing_fields({}), ["meal_intent", "craving_or_cuisine"])

    def test_hunger_level_is_never_a_blocker(self):
        self.assertNotIn("hunger_level", _missing_fields({}))

    def test_meal_time_is_never_a_blocker(self):
        self.assertNotIn("meal_time", _missing_fields({}))


class ApplyAnswerTests(unittest.TestCase):
    def test_positional_reply_resolves_against_the_current_window(self):
        profile = _apply_answer({}, "meal_intent", "2", TEN_PM)
        self.assertEqual(profile["meal_intent"], "quick_bite")

    def test_unmatched_meal_reply_falls_back_to_the_window_default(self):
        profile = _apply_answer({}, "meal_intent", "zzzz", TEN_PM)
        self.assertEqual(profile["meal_intent"], "dinner")

    def test_cuisine_word_lands_in_cuisine(self):
        profile = _apply_answer({}, "craving_or_cuisine", "Thai", TEN_PM)
        self.assertEqual(profile["cuisine"], "Thai")
        self.assertIsNone(profile.get("craving"))

    def test_dish_name_lands_in_craving(self):
        profile = _apply_answer({}, "craving_or_cuisine", "biryani", TEN_PM)
        self.assertEqual(profile["craving"], "biryani")
        self.assertIsNone(profile.get("cuisine"))

    def test_bare_digit_is_not_stored_as_a_craving(self):
        """This question has no numbered options; "1" is a stray keystroke,
        and `craving="1"` would go straight into the menu filters."""
        profile = _apply_answer({}, "craving_or_cuisine", "1", TEN_PM)

        self.assertIsNone(profile.get("craving"))
        self.assertIsNone(profile.get("cuisine"))

    def test_no_preference_resolves_the_field_instead_of_storing_it(self):
        """'anything' must not become a literal cuisine that matches zero rows."""
        profile = _apply_answer({}, "craving_or_cuisine", "anything", TEN_PM)

        self.assertIn("craving_or_cuisine", profile["no_preference"])
        self.assertIsNone(profile.get("cuisine"))
        self.assertIsNone(profile.get("craving"))
        # Resolved, so the loop stops asking about it -- unlike meal_intent,
        # which this answer said nothing about and is still outstanding.
        self.assertNotIn("craving_or_cuisine", _missing_fields(profile))


class IntentFillLoopTests(unittest.TestCase):
    def test_complete_request_never_interrupts(self):
        state = {
            "messages": [{"role": "user", "content": "spicy biryani for dinner"}],
            "intent_profile": {},
        }
        extraction = IntentExtraction(meal_intent="dinner", craving="biryani")

        result, spy = run_node(state, extraction)

        spy.assert_not_called()
        profile = result["intent_profile"]
        self.assertEqual(profile["meal_intent"], "dinner")
        self.assertEqual(profile["craving"], "biryani")
        self.assertEqual(profile["important_missing_information"], [])

    def test_bare_request_asks_for_both_required_fields_in_order(self):
        state = {
            "messages": [{"role": "user", "content": "i'm hungry"}],
            "intent_profile": {},
        }

        result, spy = run_node(
            state, IntentExtraction(), answers=["1", "biryani"]
        )

        self.assertEqual(spy.call_count, 2)
        asked_fields = [call.args[0]["field"] for call in spy.call_args_list]
        self.assertEqual(asked_fields, ["meal_intent", "craving_or_cuisine"])

        profile = result["intent_profile"]
        self.assertEqual(profile["meal_intent"], "dinner")
        self.assertEqual(profile["craving"], "biryani")
        self.assertEqual(profile["important_missing_information"], [])

    def test_meal_question_carries_clock_options(self):
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        _result, spy = run_node(state, IntentExtraction(), answers=["1", "pizza"])

        payload = spy.call_args_list[0].args[0]
        self.assertIn("10:00 PM", payload["question"])
        self.assertEqual(
            [o["value"] for o in payload["options"]], ["dinner", "quick_bite", "snack"]
        )

    def test_only_the_missing_field_is_asked(self):
        state = {
            "messages": [{"role": "user", "content": "biryani"}],
            "intent_profile": {},
        }

        _result, spy = run_node(state, IntentExtraction(craving="biryani"), answers=["3"])

        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args_list[0].args[0]["field"], "meal_intent")

    def test_answers_are_recorded_with_their_questions(self):
        """A bare '1' in the transcript is meaningless without the question."""
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        result, _spy = run_node(state, IntentExtraction(), answers=["1", "pizza"])

        contents = [m.content for m in result["messages"]]
        self.assertEqual(len(contents), 4)
        self.assertIn("10:00 PM", contents[0])
        self.assertEqual(contents[1], "1")
        self.assertEqual(contents[3], "pizza")

    def test_a_field_is_never_asked_twice(self):
        """An unusable reply moves the loop on rather than repeating itself."""
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        # "" leaves craving_or_cuisine unfilled; the loop must not re-ask it.
        _result, spy = run_node(state, IntentExtraction(), answers=["1", ""])

        asked_fields = [call.args[0]["field"] for call in spy.call_args_list]
        self.assertEqual(len(asked_fields), len(set(asked_fields)))

    def test_loop_is_bounded(self):
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        _result, spy = run_node(
            state, IntentExtraction(), answers=[""] * MAX_INTENT_CLARIFICATION_ROUNDS
        )

        self.assertLessEqual(spy.call_count, MAX_INTENT_CLARIFICATION_ROUNDS)

    def test_unfilled_gaps_are_reported_not_hidden(self):
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        result, _spy = run_node(state, IntentExtraction(), answers=["1", ""])

        self.assertEqual(
            result["intent_profile"]["important_missing_information"],
            ["craving_or_cuisine"],
        )
        self.assertTrue(result["intent_profile"]["clarification_questions"])

    def test_earlier_turns_survive_an_extraction_that_forgets_them(self):
        state = {
            "messages": [{"role": "user", "content": "actually make it Thai"}],
            "intent_profile": {"meal_intent": "dinner", "craving": "biryani"},
        }

        result, spy = run_node(state, IntentExtraction(cuisine="Thai"))

        spy.assert_not_called()
        profile = result["intent_profile"]
        self.assertEqual(profile["cuisine"], "Thai")
        self.assertEqual(profile["craving"], "biryani")
        self.assertEqual(profile["meal_intent"], "dinner")

    def test_extraction_failure_still_asks_rather_than_crashing(self):
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        result, spy = run_node(state, extraction=None, answers=["1", "pizza"])

        self.assertEqual(spy.call_count, 2)
        self.assertEqual(result["intent_profile"]["craving"], "pizza")

    def test_no_messages_key_added_when_nothing_was_asked(self):
        state = {
            "messages": [{"role": "user", "content": "dinner, biryani"}],
            "intent_profile": {},
        }

        result, _spy = run_node(
            state, IntentExtraction(meal_intent="dinner", craving="biryani")
        )

        self.assertNotIn("messages", result)


class PendingDecisionQuestionTests(unittest.TestCase):
    """decision_agent_node never calls `interrupt()` itself (see
    tests/test_decision_agent.py) -- it hands a question off via
    `state["pending_decision_question"]`, and this node is the one that
    actually asks it, with the same tappable-option treatment as its own
    fields, before folding the answer into the profile the normal way."""

    def test_asks_the_handed_off_question_first(self):
        state = {
            "messages": [{"role": "user", "content": "chinese or japanese"}],
            "intent_profile": {
                "cuisine": "Chinese or Japanese",
                "meal_intent": "late_night_craving",
                "flavor": "sweet",
            },
            "pending_decision_question": "Vegetarian or non-vegetarian today?",
        }

        result, spy = run_node(state, IntentExtraction(), answers=["non-veg"])

        self.assertEqual(spy.call_count, 1)
        payload = spy.call_args_list[0].args[0]
        self.assertEqual(payload["question"], "Vegetarian or non-vegetarian today?")
        self.assertEqual(payload["agent"], "decision_agent")
        # important_missing_information/clarification_questions are already
        # empty in this fixture (mirrors the real trigger: decision only
        # hands off once intent has nothing left of its own to ask), so this
        # is the only interrupt raised.
        self.assertEqual(result["intent_profile"]["important_missing_information"], [])

    def test_answer_is_folded_into_the_profile_and_the_handoff_is_cleared(self):
        state = {
            "messages": [{"role": "user", "content": "chinese or japanese"}],
            "intent_profile": {
                "cuisine": "Chinese or Japanese",
                "meal_intent": "late_night_craving",
                "flavor": "sweet",
            },
            "pending_decision_question": "Vegetarian or non-vegetarian today?",
        }

        result, _spy = run_node(state, IntentExtraction(), answers=["non-veg"])

        self.assertEqual(result["intent_profile"]["dietary_preferences"], ["Non-Veg"])
        self.assertIsNone(result["pending_decision_question"])
        self.assertIsNone(result["pending_clarification_field"])

    def test_question_and_answer_are_recorded_in_messages(self):
        state = {
            "messages": [{"role": "user", "content": "chinese or japanese"}],
            "intent_profile": {
                "cuisine": "Chinese or Japanese",
                "meal_intent": "late_night_craving",
                "flavor": "sweet",
            },
            "pending_decision_question": "Vegetarian or non-vegetarian today?",
        }

        result, _spy = run_node(state, IntentExtraction(), answers=["non-veg"])

        contents = [m.content for m in result["messages"]]
        self.assertEqual(contents[0], "Vegetarian or non-vegetarian today?")
        self.assertEqual(contents[1], "non-veg")

    def test_no_handoff_means_no_extra_interrupt(self):
        """Without `pending_decision_question` set, behavior is unchanged --
        this node's own loop still runs exactly as before."""
        state = {
            "messages": [{"role": "user", "content": "spicy biryani for dinner"}],
            "intent_profile": {},
        }

        result, spy = run_node(state, IntentExtraction(meal_intent="dinner", craving="biryani"))

        spy.assert_not_called()
        self.assertIsNone(result.get("pending_decision_question"))


class ExtractTastesTests(unittest.TestCase):
    def test_maps_synonyms_onto_menu_tags(self):
        self.assertIn("spicy", extract_tastes("make it fiery"))

    def test_negation_wins_over_the_literal_word(self):
        self.assertEqual(extract_tastes("nothing too spicy"), ["mild"])

    def test_multiple_tastes(self):
        self.assertEqual(sorted(extract_tastes("sweet and tangy")), ["sweet", "tangy"])

    def test_no_taste_mentioned(self):
        self.assertEqual(extract_tastes("i want biryani"), [])


if __name__ == "__main__":
    unittest.main()
