import unittest
from unittest.mock import patch

from cravr.decisionAgent import CLARIFY_AND_RETRY, MAX_CLARIFICATION_ROUNDS, decision_agent_node
from cravr.schemas import DecisionOutput


class StubStructuredLLM:
    """Stands in for `get_llm().with_structured_output(DecisionOutput)`."""

    def __init__(self, result: DecisionOutput):
        self._result = result

    def invoke(self, messages):
        return self._result


class StubLLM:
    """Stands in for `get_llm()`; `.with_structured_output(...)` returns a
    stub that always yields the given `DecisionOutput`, mirroring how a real
    tool-calling model would be used."""

    def __init__(self, decision_output: DecisionOutput):
        self._decision_output = decision_output

    def with_structured_output(self, schema):
        return StubStructuredLLM(self._decision_output)


class DecisionAgentTests(unittest.TestCase):
    def test_asks_specific_question_via_interrupt_and_routes_back_for_retry(self):
        """The clarification question must come from what's actually missing,
        not a generic canned line, and this node must ask only once per call
        (looping happens via the graph, not inside this node) so that a
        resumed interrupt can't silently drop info learned in a prior round.
        """
        state = {
            "messages": [{"role": "user", "content": "Suggest something for dinner"}],
            "intent_profile": {
                "missing_information": ["craving_or_cuisine"],
                "important_missing_information": ["craving_or_cuisine"],
                "clarification_questions": ["Do you have a specific craving or cuisine in mind?"],
            },
            "context_profile": {
                "missing_information": ["location"],
                "important_missing_information": ["location"],
                "clarification_questions": ["What's the delivery location (city/area)?"],
            },
            "history_profile": {"status": "success", "count": 0, "orders": []},
            "clarification_rounds": 0,
        }

        with patch(
            "cravr.decisionAgent.interrupt", return_value="Spicy Indian food, in Bangalore"
        ) as mock_interrupt:
            result = decision_agent_node(state)

        mock_interrupt.assert_called_once()
        asked = mock_interrupt.call_args[0][0]["question"]
        self.assertEqual(asked, "Do you have a specific craving or cuisine in mind?")

        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)
        self.assertEqual(result["clarification_rounds"], 1)
        # Both the question AND the human's answer must land in `messages`.
        # A bare answer like "Indian" or "500" is meaningless to the
        # extraction model on the next round without the question that
        # prompted it -- that's what caused cuisine/budget to never get
        # extracted from short clarification replies.
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0].content, "Do you have a specific craving or cuisine in mind?")
        self.assertEqual(result["messages"][1].content, "Spicy Indian food, in Bangalore")

    def test_falls_back_to_generic_message_when_no_specific_question_exists(self):
        state = {
            "messages": [{"role": "user", "content": "i want to eat light food"}],
            "intent_profile": {"important_missing_information": ["craving_or_cuisine"]},
            "context_profile": {},
            "history_profile": {"status": "success", "count": 0, "orders": []},
            "clarification_rounds": 0,
        }

        with patch("cravr.decisionAgent.interrupt", return_value="something light") as mock_interrupt:
            result = decision_agent_node(state)

        mock_interrupt.assert_called_once()
        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)
        # No specific clarification_questions were provided on the profile,
        # so the generic fallback question is what gets asked.
        asked = mock_interrupt.call_args[0][0]["question"]
        self.assertIn("craving", asked.lower())

    def test_never_asks_the_same_field_twice_in_a_streak(self):
        """The loop that had users typing "vegetarian" six times: the node
        always re-served `clarification_questions[0]`, so one field whose
        answer failed to extract blocked every other question forever."""
        state = {
            "messages": [{"role": "user", "content": "something spicy"}],
            "intent_profile": {
                "important_missing_information": ["craving_or_cuisine", "dietary_preferences", "budget_inr"],
                "clarification_questions": [
                    "Do you have a specific craving or cuisine in mind?",
                    "Should the meal be strictly vegetarian, non-veg, or something else?",
                    "What's your budget for this order in INR?",
                ],
                "taste_preference": ["spicy"],
            },
            "context_profile": {},
            "history_profile": {"status": "success", "count": 0, "orders": []},
            "clarification_rounds": 1,
            # craving_or_cuisine was asked last round; the answer didn't stick.
            "asked_clarification_fields": ["craving_or_cuisine"],
        }

        with patch("cravr.decisionAgent.interrupt", return_value="vegetarian") as mock_interrupt:
            result = decision_agent_node(state)

        asked = mock_interrupt.call_args[0][0]
        self.assertEqual(asked["field"], "dietary_preferences")
        self.assertIn("vegetarian", asked["question"].lower())
        # The field is recorded so intent_discovery can apply the answer
        # directly rather than hoping the extraction model infers it.
        self.assertEqual(result["pending_clarification_field"], "dietary_preferences")
        self.assertEqual(
            result["asked_clarification_fields"], ["craving_or_cuisine", "dietary_preferences"]
        )

    def test_stops_asking_once_every_gap_has_been_raised_once(self):
        state = {
            "messages": [{"role": "user", "content": "something spicy"}],
            "intent_profile": {
                "important_missing_information": ["craving_or_cuisine"],
                "clarification_questions": ["Do you have a specific craving or cuisine in mind?"],
                "taste_preference": ["spicy"],
            },
            "context_profile": {},
            "history_profile": {"status": "success", "count": 0, "orders": []},
            "clarification_rounds": 1,
            "asked_clarification_fields": ["craving_or_cuisine"],
        }

        with patch("cravr.decisionAgent.interrupt") as mock_interrupt, patch(
            "cravr.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(ready_to_recommend=False, decision="clarify", clarification_message=None)
            ),
        ):
            result = decision_agent_node(state)

        mock_interrupt.assert_not_called()
        # A taste preference is enough to recommend loosely — far better than
        # ending the turn on a question the user already answered.
        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")

    def test_gives_up_after_max_rounds_without_re_asking_the_question(self):
        """Re-emitting the abandoned question is what restarted the loop: the
        user's reply arrived as a fresh turn, resetting the round counter."""
        state = {
            "messages": [{"role": "user", "content": "I want something for dinner"}],
            "intent_profile": {
                "missing_information": ["craving_or_cuisine"],
                "important_missing_information": ["craving_or_cuisine"],
                "clarification_questions": ["Do you have a specific craving or cuisine in mind?"],
            },
            "context_profile": {},
            "history_profile": {"status": "success", "count": 0, "orders": []},
            "clarification_rounds": MAX_CLARIFICATION_ROUNDS,
        }

        with patch("cravr.decisionAgent.interrupt") as mock_interrupt, patch(
            "cravr.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(ready_to_recommend=False, decision="clarify", clarification_message=None)
            ),
        ):
            result = decision_agent_node(state)

        mock_interrupt.assert_not_called()
        # Nothing usable in the profile, so this legitimately ends the turn --
        # but with a give-up line, not the question we just abandoned.
        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "clarify")
        self.assertNotIn("craving or cuisine in mind", result["messages"][-1].content)
        self.assertIn("summary", result)
        # The streak resets so the next request gets a fresh budget.
        self.assertEqual(result["clarification_rounds"], 0)
        self.assertEqual(result["asked_clarification_fields"], [])

    def test_ready_when_core_preference_is_present_without_clarification(self):
        state = {
            "messages": [{"role": "user", "content": "Suggest a spicy vegetarian dinner"}],
            "intent_profile": {"craving": "spicy food", "cuisine": "Indian", "dietary_preferences": ["vegetarian"]},
            "context_profile": {"location": "Bangalore"},
            "history_profile": {"status": "success", "count": 2, "orders": [{"restaurant_name": "Test Cafe"}]},
        }

        with patch("cravr.decisionAgent.interrupt") as mock_interrupt, patch(
            "cravr.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(ready_to_recommend=True, decision="recommend", clarification_message=None)
            ),
        ):
            result = decision_agent_node(state)

        mock_interrupt.assert_not_called()
        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertEqual(result["messages"][-1].content, "Ready to recommend.")
        self.assertEqual(result["clarification_rounds"], 0)

    def test_falls_back_to_clarify_when_structured_output_call_fails(self):
        """If the model/tool-calling call itself raises (e.g. the chosen
        Ollama model doesn't actually support tool calling), the node must
        degrade to a clarify response instead of propagating the exception."""
        state = {
            "messages": [{"role": "user", "content": "Suggest a spicy vegetarian dinner"}],
            "intent_profile": {"craving": "spicy food", "cuisine": "Indian", "dietary_preferences": ["vegetarian"]},
            "context_profile": {"location": "Bangalore"},
            "history_profile": {"status": "success", "count": 2, "orders": [{"restaurant_name": "Test Cafe"}]},
        }

        class ExplodingLLM:
            def with_structured_output(self, schema):
                raise RuntimeError("model does not support tool calling")

        with patch("cravr.decisionAgent.interrupt") as mock_interrupt, patch(
            "cravr.decisionAgent.get_llm", return_value=ExplodingLLM()
        ):
            result = decision_agent_node(state)

        mock_interrupt.assert_not_called()
        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "clarify")

    def test_clarifies_vague_requests_even_when_no_important_fields_are_flagged(self):
        state = {
            "messages": [{"role": "user", "content": "i want to eat light food"}],
            "intent_profile": {"missing_information": ["craving_or_cuisine"]},
            "context_profile": {"missing_information": ["location"]},
            "history_profile": {"status": "success", "count": 0, "orders": []},
        }

        with patch("cravr.decisionAgent.interrupt", return_value="still not sure") as mock_interrupt:
            result = decision_agent_node(state)

        mock_interrupt.assert_called_once()
        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)


if __name__ == "__main__":
    unittest.main()
