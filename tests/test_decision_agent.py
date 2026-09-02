import unittest
from unittest.mock import patch

from cravr.decision.decisionAgent import decision_agent_node
from cravr.decision.utils import CLARIFY_AND_RETRY, MAX_CLARIFICATION_ROUNDS
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


class ExplodingLLM:
    def with_structured_output(self, schema):
        raise RuntimeError("structured output unavailable")


def state_with(intent_profile, user_text="Suggest something for dinner", clarification_rounds=0):
    return {
        "messages": [{"role": "user", "content": user_text}],
        "intent_profile": intent_profile,
        "history_profile": {"status": "success", "count": 0, "orders": []},
        "clarification_rounds": clarification_rounds,
    }


class DecisionAgentNeverAsksTests(unittest.TestCase):
    """`intent_discovery_node` is the only node that ever calls `interrupt()`.
    Two interrupt sites would mean two round budgets, neither aware of the
    other, and the user answering the same gap twice -- so when
    decision_agent_node wants something asked, it hands the question off via
    `pending_decision_question` + `CLARIFY_AND_RETRY` instead of asking it
    directly."""

    def test_does_not_import_or_use_interrupt(self):
        import cravr.decision.decisionAgent as module

        self.assertFalse(hasattr(module, "interrupt"))

    def test_gaps_left_by_intent_do_not_trigger_a_question(self):
        state = state_with({
            "craving": "biryani",
            "important_missing_information": ["meal_intent"],
        })

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(DecisionOutput(ready_to_recommend=True, decision="recommend")),
        ):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")


class DecisionAgentVerdictTests(unittest.TestCase):
    def test_ready_verdict_freezes_a_recommendation_brief(self):
        state = state_with({"craving": "biryani", "cuisine": "North Indian"})

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(DecisionOutput(ready_to_recommend=True, decision="recommend")),
        ):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertIn("recommendation_brief", result)
        self.assertIn("filters", result["recommendation_brief"])

    def test_not_ready_verdict_hands_the_question_to_intent_discovery(self):
        """A fresh "clarify" verdict (budget not spent) is handed to
        intent_discovery_node to ask with tappable options -- decision_agent
        itself never puts the question straight into `messages` anymore."""
        state = state_with({})

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(
                    ready_to_recommend=False,
                    decision="clarify",
                    clarification_message="What are you in the mood for?",
                )
            ),
        ):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertNotIn("recommendation_brief", result)
        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)
        self.assertEqual(result["pending_decision_question"], "What are you in the mood for?")
        self.assertEqual(result["clarification_rounds"], 1)
        # Not asked directly -- intent_discovery_node asks it via interrupt().
        self.assertNotIn("messages", result)

    def test_structured_output_failure_falls_back_to_clarify(self):
        """A model error must not read as 'ready to recommend'."""
        state = state_with({})

        with patch("cravr.decision.decisionAgent.get_llm", return_value=ExplodingLLM()):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)
        self.assertTrue(result["pending_decision_question"])

    def test_recommends_anyway_when_intent_gave_up_but_left_usable_signal(self):
        """Ending a turn with another question is the worst outcome. If the
        user told us anything filterable, search loosely and let them steer."""
        state = state_with({
            "craving_tag": "spicy",
            "important_missing_information": ["meal_intent"],
        })

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(DecisionOutput(ready_to_recommend=False, decision="clarify")),
        ):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertIn("recommendation_brief", result)

    def test_does_not_recommend_when_nothing_is_filterable(self):
        """meal_intent alone narrows nothing: 48 of 80 menu rows are All Day."""
        state = state_with({
            "meal_intent": "dinner",
            "important_missing_information": ["craving_or_cuisine"],
        })

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(DecisionOutput(ready_to_recommend=False, decision="clarify")),
        ):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])

    def test_summary_is_always_published(self):
        state = state_with({"craving": "biryani"})

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(DecisionOutput(ready_to_recommend=False, decision="clarify")),
        ):
            result = decision_agent_node(state)

        self.assertTrue(result["summary"])


class DecisionLevelRoundBudgetTests(unittest.TestCase):
    """`clarification_rounds` caps decision_agent's own ad hoc questions
    (the ones intent never asked about) separately from intent's own
    budget, so a decision LLM that keeps inventing new questions can't loop
    forever -- same failure mode `MAX_CLARIFICATION_ROUNDS` already guards
    against on intent's side."""

    def test_asks_again_below_the_round_cap(self):
        state = state_with({"craving": "biryani"}, clarification_rounds=MAX_CLARIFICATION_ROUNDS - 1)

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(
                    ready_to_recommend=False,
                    decision="clarify",
                    clarification_message="Veg or non-veg today?",
                )
            ),
        ):
            result = decision_agent_node(state)

        self.assertEqual(result["decision"], CLARIFY_AND_RETRY)
        self.assertEqual(result["pending_decision_question"], "Veg or non-veg today?")
        self.assertEqual(result["clarification_rounds"], MAX_CLARIFICATION_ROUNDS)

    def test_recommends_anyway_once_the_round_cap_is_spent_with_usable_signal(self):
        state = state_with({"craving": "biryani"}, clarification_rounds=MAX_CLARIFICATION_ROUNDS)

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(
                    ready_to_recommend=False,
                    decision="clarify",
                    clarification_message="Veg or non-veg today?",
                )
            ),
        ):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertIn("recommendation_brief", result)
        self.assertEqual(result["clarification_rounds"], 0)

    def test_gives_up_gracefully_once_the_round_cap_is_spent_with_nothing_usable(self):
        state = state_with({}, clarification_rounds=MAX_CLARIFICATION_ROUNDS)

        with patch(
            "cravr.decision.decisionAgent.get_llm",
            return_value=StubLLM(
                DecisionOutput(
                    ready_to_recommend=False,
                    decision="clarify",
                    clarification_message="Veg or non-veg today?",
                )
            ),
        ):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertNotIn("recommendation_brief", result)
        self.assertNotEqual(result["decision"], CLARIFY_AND_RETRY)
        self.assertIsNone(result.get("pending_decision_question"))
        self.assertEqual(result["clarification_rounds"], 0)
        # Never echoes the question the budget just ran out on.
        self.assertNotEqual(result["messages"][0].content, "Veg or non-veg today?")


if __name__ == "__main__":
    unittest.main()
