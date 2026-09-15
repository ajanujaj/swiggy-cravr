import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from cravr.intent.intentAgent import (
    MAX_INTENT_CLARIFICATION_ROUNDS,
    _CATEGORY_OPTION_FALLBACK,
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
    """Run the node with the clock frozen and interrupt() scripted. Also
    forces cravr.options.suggest_options onto its static fallback lists --
    without this, option suggestion would try a real (network) LLM call and
    make option-dependent assertions flaky/environment-dependent."""
    llm = StubLLM(extraction) if extraction is not None else ExplodingLLM()
    clock = MagicMock(wraps=datetime)
    clock.now.return_value = now

    with patch("cravr.intent.intentAgent.get_llm", return_value=llm), \
         patch("cravr.options.get_llm", side_effect=RuntimeError("no llm in tests")), \
         patch("cravr.intent.intentAgent.datetime", clock), \
         patch("cravr.intent.intentAgent.interrupt", side_effect=list(answers)) as spy:
        result = intent_discovery_node(state)
    return result, spy


class DerivedFieldTests(unittest.TestCase):
    """Fields the code can work out are never worth an interrupt."""

    def test_meal_time_comes_from_the_clock(self):
        profile = _derive({}, TEN_PM)
        self.assertEqual(profile["meal_time"], "dinner")

    def test_meal_intent_is_silently_inferred_too(self):
        """Never asked -- filled from the clock like meal_time."""
        profile = _derive({}, TEN_PM)
        self.assertEqual(profile["meal_intent"], "dinner")

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
    """Category, then a specific dish/cuisine -- meal_intent is never asked
    (silently inferred, see DerivedFieldTests)."""

    def test_empty_profile_asks_category_then_a_specific_type(self):
        self.assertEqual(_missing_fields({}), ["category", "craving_or_cuisine"])

    def test_craving_alone_satisfies_the_whole_funnel(self):
        self.assertEqual(_missing_fields({"craving": "biryani"}), [])

    def test_cuisine_alone_satisfies_the_whole_funnel(self):
        self.assertEqual(_missing_fields({"cuisine": "Thai"}), [])

    def test_flavor_alone_satisfies_the_whole_funnel(self):
        """'something spicy' is a real request, not an empty one."""
        self.assertEqual(_missing_fields({"flavor": "spicy"}), [])

    def test_category_already_picked_skips_straight_to_the_type_question(self):
        self.assertEqual(_missing_fields({"craving_tag": "comfort-food"}), ["craving_or_cuisine"])

    def test_no_preference_on_category_skips_it_but_still_asks_the_type(self):
        self.assertEqual(_missing_fields({"no_preference": ["category"]}), ["craving_or_cuisine"])

    def test_no_preference_on_the_type_question_satisfies_everything(self):
        """'surprise me' resolves the whole funnel, category included."""
        self.assertEqual(_missing_fields({"no_preference": ["craving_or_cuisine"]}), [])

    def test_hunger_level_is_never_a_blocker(self):
        self.assertNotIn("hunger_level", _missing_fields({}))

    def test_meal_time_is_never_a_blocker(self):
        self.assertNotIn("meal_time", _missing_fields({}))

    def test_meal_intent_is_never_a_blocker(self):
        self.assertNotIn("meal_intent", _missing_fields({}))


class ApplyAnswerTests(unittest.TestCase):
    def test_category_answer_maps_onto_a_known_tag(self):
        profile = _apply_answer({}, "category", "Healthy", TEN_PM)
        self.assertEqual(profile["craving_tag"], "low-calorie")

    def test_unrecognised_category_label_is_kept_raw(self):
        """Still scopes the next question even without a known-tag match --
        just doesn't also narrow the search filters."""
        profile = _apply_answer({}, "category", "Keto", TEN_PM)
        self.assertEqual(profile["craving_tag"], "Keto")

    def test_bare_digit_category_answer_is_not_stored(self):
        """The category question's options aren't numbered positionally in
        a way this branch resolves on its own -- a stray "1" is dropped."""
        profile = _apply_answer({}, "category", "1", TEN_PM)
        self.assertIsNone(profile.get("craving_tag"))

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

    def test_bare_request_asks_category_then_a_specific_type(self):
        state = {
            "messages": [{"role": "user", "content": "i'm hungry"}],
            "intent_profile": {},
        }

        result, spy = run_node(
            state, IntentExtraction(), answers=["Healthy", "Salad"]
        )

        self.assertEqual(spy.call_count, 2)
        asked_fields = [call.args[0]["field"] for call in spy.call_args_list]
        self.assertEqual(asked_fields, ["category", "craving_or_cuisine"])

        profile = result["intent_profile"]
        self.assertEqual(profile["craving_tag"], "low-calorie")  # "Healthy" -> low-calorie
        self.assertEqual(profile["craving"], "Salad")
        self.assertEqual(profile["important_missing_information"], [])

    def test_category_question_offers_fallback_options_when_llm_unavailable(self):
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        _result, spy = run_node(state, IntentExtraction(), answers=["Healthy", "Salad"])

        payload = spy.call_args_list[0].args[0]
        self.assertEqual(payload["field"], "category")
        self.assertEqual(
            [o["value"] for o in payload["options"]],
            [*_CATEGORY_OPTION_FALLBACK, "__other__"],
        )

    def test_flavour_already_known_skips_the_whole_funnel(self):
        """A craving from extraction satisfies category AND the type
        question at once -- neither is asked."""
        state = {
            "messages": [{"role": "user", "content": "biryani"}],
            "intent_profile": {},
        }

        _result, spy = run_node(state, IntentExtraction(craving="biryani"))

        spy.assert_not_called()

    def test_answers_are_recorded_with_their_questions(self):
        """A bare answer in the transcript is meaningless without the
        question that prompted it."""
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        result, _spy = run_node(state, IntentExtraction(), answers=["Healthy", "Salad"])

        contents = [m.content for m in result["messages"]]
        self.assertEqual(len(contents), 4)
        self.assertEqual(contents[0], "What are you in the mood for?")
        self.assertEqual(contents[1], "Healthy")
        self.assertEqual(contents[3], "Salad")

    def test_a_field_is_never_asked_twice(self):
        """An unusable reply moves the loop on rather than repeating itself."""
        state = {"messages": [{"role": "user", "content": "hi"}], "intent_profile": {}}

        # "" leaves craving_or_cuisine unfilled; the loop must not re-ask it.
        _result, spy = run_node(state, IntentExtraction(), answers=["Healthy", ""])

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

        result, _spy = run_node(state, IntentExtraction(), answers=["Healthy", ""])

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

        result, spy = run_node(state, extraction=None, answers=["1", "biryani"])

        self.assertEqual(spy.call_count, 2)
        self.assertEqual(result["intent_profile"]["craving"], "biryani")

    def test_no_messages_key_added_when_nothing_was_asked(self):
        state = {
            "messages": [{"role": "user", "content": "dinner, biryani"}],
            "intent_profile": {},
        }

        result, _spy = run_node(
            state, IntentExtraction(meal_intent="dinner", craving="biryani")
        )

        self.assertNotIn("messages", result)


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
