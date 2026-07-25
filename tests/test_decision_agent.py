import unittest
from unittest.mock import patch

from cravr.decisionAgent import decision_agent_node


class StubResponse:
    def __init__(self, content):
        self.content = content


class StubLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, messages):
        return StubResponse(self._content)


class DecisionAgentTests(unittest.TestCase):
    def test_clarifies_when_required_context_is_missing(self):
        state = {
            "messages": [{"role": "user", "content": "Suggest something for dinner"}],
            "intent_profile": {"missing_information": ["craving_or_cuisine"], "important_missing_information": ["craving_or_cuisine"]},
            "context_profile": {"missing_information": ["location"], "important_missing_information": ["location"]},
            "history_profile": {"status": "success", "count": 0, "orders": []},
        }

        with patch("cravr.decisionAgent.get_llm", return_value=StubLLM('{"ready_to_recommend": false, "decision": "clarify", "clarification_message": "Can you share your craving or cuisine?"}')):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "clarify")
        self.assertIn("craving", result["messages"][0].content.lower())
        self.assertIn("summary", result)
        self.assertIn("intent", result["summary"].lower())

    def test_ready_when_core_preference_is_present(self):
        state = {
            "messages": [{"role": "user", "content": "Suggest a spicy vegetarian dinner"}],
            "intent_profile": {"craving": "spicy food", "cuisine": "Indian", "dietary_preferences": ["vegetarian"]},
            "context_profile": {"location": "Bangalore"},
            "history_profile": {"status": "success", "count": 2, "orders": [{"restaurant_name": "Test Cafe"}]},
        }

        with patch("cravr.decisionAgent.get_llm", return_value=StubLLM('{"ready_to_recommend": true, "decision": "recommend", "clarification_message": null}')):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertEqual(result["messages"][0].content, "Ready to recommend.")

    def test_uses_llm_to_decide_when_to_recommend(self):
        state = {
            "messages": [{"role": "user", "content": "Suggest something for dinner"}],
            "intent_profile": {"missing_information": ["craving_or_cuisine"]},
            "context_profile": {"missing_information": ["location"]},
            "history_profile": {"status": "success", "count": 1, "orders": []},
        }

        with patch("cravr.decisionAgent.get_llm", return_value=StubLLM('{"ready_to_recommend": true, "decision": "recommend", "clarification_message": null}')):
            result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")

    def test_clarifies_vague_requests_even_if_llm_says_recommend(self):
        state = {
            "messages": [{"role": "user", "content": "i want to eat light food"}],
            "intent_profile": {"missing_information": ["craving_or_cuisine"]},
            "context_profile": {"missing_information": ["location"]},
            "history_profile": {"status": "success", "count": 0, "orders": []},
        }

        with patch("cravr.decisionAgent.get_llm", return_value=StubLLM('{"ready_to_recommend": true, "decision": "recommend", "clarification_message": null}')):
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "clarify")

    def test_clarifies_immediately_when_critical_details_are_missing(self):
        state = {
            "messages": [{"role": "user", "content": "I want something for dinner"}],
            "intent_profile": {"missing_information": ["craving_or_cuisine"], "important_missing_information": ["craving_or_cuisine"]},
            "context_profile": {},
            "history_profile": {"status": "success", "count": 0, "orders": []},
        }

        with patch("cravr.decisionAgent.get_llm") as mock_get_llm:
            result = decision_agent_node(state)

        self.assertFalse(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "clarify")
        self.assertIn("craving", result["messages"][0].content.lower())
        self.assertIn("summary", result)
        mock_get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
