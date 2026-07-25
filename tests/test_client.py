import os
import unittest
from unittest.mock import patch

from cravr import client as client_module
from cravr.historyAgent import history_discovery_node
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

    def test_history_discovery_node_uses_order_history_tool(self):
        fake_history = {"status": "success", "count": 1, "orders": [{"restaurant_name": "Test Cafe"}]}

        with patch("cravr.historyAgent.fetch_order_history", return_value=fake_history) as mock_fetch:
            result = history_discovery_node({
                "messages": [{"role": "user", "content": "show me my last orders"}],
                "intent_profile": {},
                "context_profile": {},
                "user_id": "USR-001",
            })

        self.assertEqual(result["history_profile"], fake_history)
        self.assertEqual(result["user_id"], "USR-001")
        mock_fetch.assert_called_once_with("USR-001", None)
