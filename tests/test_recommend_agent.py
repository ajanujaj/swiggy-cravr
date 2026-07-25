import unittest
from unittest.mock import patch

from cravr.RecommendAgent import recommendation_agent_node


class RecommendationAgentTests(unittest.TestCase):
    def test_returns_initial_recommendations_and_waits_for_feedback(self):
        state = {
            "messages": [{"role": "user", "content": "I want light food"}],
            "intent_profile": {"dietary_preferences": ["vegetarian"], "craving": "light food"},
            "context_profile": {"location": "Bangalore"},
            "history_profile": {"status": "success", "count": 1, "orders": []},
            "user_id": "USR-001",
            "summary": "User asked: I want light food | intent: craving=light food; dietary_preferences=vegetarian | context: location=Bangalore",
        }

        recommendations = [
            {
                "restaurant_name": "Dakshin Delights",
                "item_name": "Plain Idli",
                "cuisine": "South Indian",
                "price_inr": 100,
                "rating": 4.4,
                "tags": "vegetarian;breakfast;lightweight",
            }
        ]

        with patch("cravr.RecommendAgent.fetch_recommendations", return_value=recommendations):
            result = recommendation_agent_node(state)

        self.assertIn("Plain Idli", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["awaiting_feedback"])
        self.assertEqual(result["recommendation_state"]["last_recommendations"], recommendations)

    def test_confirms_when_user_accepts_recommendation(self):
        state = {
            "messages": [
                {"role": "assistant", "content": "I recommend Plain Idli from Dakshin Delights."},
                {"role": "user", "content": "Yes, that sounds good"},
            ],
            "intent_profile": {"dietary_preferences": ["vegetarian"]},
            "context_profile": {"location": "Bangalore"},
            "history_profile": {"status": "success", "count": 1, "orders": []},
            "user_id": "USR-001",
            "recommendation_state": {
                "awaiting_feedback": True,
                "last_recommendations": [
                    {
                        "restaurant_name": "Dakshin Delights",
                        "item_name": "Plain Idli",
                        "cuisine": "South Indian",
                        "price_inr": 100,
                        "rating": 4.4,
                        "tags": "vegetarian;breakfast;lightweight",
                    }
                ],
            },
        }

        with patch("cravr.RecommendAgent.fetch_recommendations", return_value=[]):
            result = recommendation_agent_node(state)

        self.assertIn("perfect", result["messages"][0].content.lower())
        self.assertTrue(result["recommendation_state"]["satisfied"])


if __name__ == "__main__":
    unittest.main()
