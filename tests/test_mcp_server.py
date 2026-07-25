import unittest

import pandas as pd

from mcp_server import run_mcp


class OrderHistoryTests(unittest.TestCase):
    def setUp(self):
        self.original_menu = run_mcp.recommendations_data
        self.original_history = run_mcp.order_history_data

    def tearDown(self):
        run_mcp.recommendations_data = self.original_menu
        run_mcp.order_history_data = self.original_history

    def test_order_history_handles_multiindex_menu_data(self):
        run_mcp.recommendations_data = pd.DataFrame(
            [
                {
                    "restaurant_name": "Punjab Grill",
                    "cuisine": "North Indian",
                    "item_name": "Butter Chicken",
                    "price_inr": 420,
                    "rating": 4.5,
                    "tags": "high-protein",
                }
            ],
            index=pd.MultiIndex.from_tuples([("Punjab Grill", "North Indian", "Butter Chicken")], names=["restaurant_name", "cuisine", "item_name"]),
        )
        run_mcp.order_history_data = pd.DataFrame(
            [
                {
                    "user_id": "USR-001",
                    "restaurant_name": "Punjab Grill",
                    "cuisine": "North Indian",
                    "item_name": "Butter Chicken",
                    "quantity": 1,
                    "unit_price_inr": 420,
                    "line_total_inr": 420,
                    "order_status": "Delivered",
                }
            ]
        )

        result = run_mcp.order_history("USR-001")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["orders"]), 1)
        self.assertEqual(result["orders"][0]["restaurant_name"], "Punjab Grill")
        self.assertEqual(result["orders"][0]["cuisine"], "North Indian")
        self.assertEqual(result["orders"][0]["item_name"], "Butter Chicken")
        self.assertEqual(result["orders"][0]["meal_type"], "All Day")
        self.assertEqual(result["orders"][0]["tags"], "high-protein")
        self.assertNotIn("price_inr", result["orders"][0])
        self.assertNotIn("rating", result["orders"][0])


if __name__ == "__main__":
    unittest.main()
