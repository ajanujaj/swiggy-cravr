import unittest

from cravr.decision.decisionAgent import decision_agent_node


def state_with(intent_profile, user_text="Suggest something for dinner", **extra):
    return {
        "messages": [{"role": "user", "content": user_text}],
        "intent_profile": intent_profile,
        "history_profile": {"status": "success", "count": 0, "orders": []},
        **extra,
    }


class DecisionAgentNeverAsksTests(unittest.TestCase):
    def test_does_not_import_or_use_interrupt(self):
        import cravr.decision.decisionAgent as module

        self.assertFalse(hasattr(module, "interrupt"))


class DecisionAgentAlwaysRecommendsTests(unittest.TestCase):
    def test_freezes_a_recommendation_brief(self):
        state = state_with({"craving": "biryani", "cuisine": "North Indian"})

        result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertIn("recommendation_brief", result)
        self.assertIn("filters", result["recommendation_brief"])

    def test_recommends_even_with_an_empty_profile(self):
        state = state_with({})

        result = decision_agent_node(state)

        self.assertTrue(result["ready_to_recommend"])
        self.assertEqual(result["decision"], "recommend")
        self.assertIn("recommendation_brief", result)

    def test_summary_is_always_published(self):
        state = state_with({"craving": "biryani"})

        result = decision_agent_node(state)

        self.assertTrue(result["summary"])


if __name__ == "__main__":
    unittest.main()
