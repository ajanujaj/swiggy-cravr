import os
import unittest
from unittest.mock import patch

from cravr import client as client_module
from cravr.context.ContextAgent import context_discovery_node
from cravr.userPreference.preference import user_preference_node
from mcp_server.run_mcp import order_history


class ClientTests(unittest.TestCase):
    def test_default_mcp_url_points_to_http_endpoint(self):
        self.assertEqual(client_module.DEFAULT_MCP_URL, "http://127.0.0.1:8000/mcp")

    def test_env_override_is_used_for_mcp_url(self):
        with patch.dict(os.environ, {"CRAVR_MCP_URL": "http://example.test/mcp"}, clear=False):
            import importlib

            module = importlib.reload(client_module)
            self.assertEqual(module.DEFAULT_MCP_URL, "http://example.test/mcp")

    def test_order_history_returns_results_for_known_user(self):
        result = order_history("USR-001")

        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["count"], 1)
        self.assertGreaterEqual(len(result["orders"]), 1)

    def test_user_preference_node_loads_preferences_and_order_history(self):
        fake_history = {"status": "success", "count": 1, "orders": [{"restaurant_name": "Test Cafe"}]}
        fake_preferences = {
            "status": "success",
            "preferences": {"dietary_preference": "Veg", "default_budget_inr": 300},
        }

        with patch("cravr.userPreference.preference.fetch_order_history", return_value=fake_history) as mock_history, \
             patch("cravr.userPreference.preference.fetch_user_preferences", return_value=fake_preferences) as mock_preferences:
            result = user_preference_node({
                "messages": [{"role": "user", "content": "show me my last orders"}],
                "intent_profile": {},
                "context_profile": {},
                "user_id": "USR-001",
            })

        self.assertEqual(result["history_profile"], fake_history)
        self.assertEqual(result["preference_profile"], fake_preferences)
        self.assertEqual(result["user_id"], "USR-001")
        mock_history.assert_called_once_with("USR-001")
        mock_preferences.assert_called_once_with("USR-001")

    def test_context_agent_loads_saved_location_without_questions(self):
        saved_location = {
            "status": "success",
            "location": "Bengaluru",
            "delivery_area": "Koramangala",
        }
        with patch("cravr.context.ContextAgent.fetch_user_location", return_value=saved_location) as mock_fetch:
            result = context_discovery_node({"context_profile": {}, "user_id": "USR-001"})

        self.assertEqual(result["context_profile"]["location"], "Bengaluru")
        self.assertEqual(result["context_profile"]["delivery_area"], "Koramangala")
        self.assertEqual(result["context_profile"]["clarification_questions"], [])
        mock_fetch.assert_called_once_with("USR-001")
