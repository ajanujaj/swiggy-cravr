import unittest

import pandas as pd

from mcp_server import run_mcp


class RecommendationTagFilterTests(unittest.TestCase):
    """Multiple tags (comma-separated) must be matched with AND semantics --
    e.g. tags='spicy,vegetarian' should only return dishes carrying both,
    not either. A single tag with no comma must keep behaving exactly like
    the original substring-contains filter."""

    def setUp(self):
        self.original_menu = run_mcp.recommendations_data
        run_mcp.recommendations_data = pd.DataFrame(
            [
                {"restaurant_name": "R1", "cuisine": "Indian", "item_name": "A", "dietary_type": "Veg",
                 "meal_type": "All Day", "price_inr": 200, "rating": 4.5, "tags": "spicy;vegetarian"},
                {"restaurant_name": "R2", "cuisine": "Indian", "item_name": "B", "dietary_type": "Non-Veg",
                 "meal_type": "All Day", "price_inr": 250, "rating": 4.2, "tags": "spicy;high-protein"},
                {"restaurant_name": "R3", "cuisine": "Indian", "item_name": "C", "dietary_type": "Veg",
                 "meal_type": "All Day", "price_inr": 150, "rating": 4.0, "tags": "sweet;vegetarian"},
            ]
        )

    def tearDown(self):
        run_mcp.recommendations_data = self.original_menu

    def test_single_tag_matches_by_substring_as_before(self):
        result = run_mcp.recommendation(tags="spicy")
        names = {r["item_name"] for r in result["recommendations"]}
        self.assertEqual(names, {"A", "B"})

    def test_comma_separated_tags_require_all_of_them(self):
        result = run_mcp.recommendation(tags="spicy,vegetarian")
        names = {r["item_name"] for r in result["recommendations"]}
        self.assertEqual(names, {"A"})

    def test_comma_separated_tags_with_no_match_returns_no_results(self):
        result = run_mcp.recommendation(tags="spicy,sweet")
        self.assertEqual(result["status"], "no_results")

    def test_veg_filter_excludes_non_veg_dishes(self):
        """Regression test: dietary_type='Veg' must not also match 'Non-Veg'
        rows via substring containment (.str.contains('veg') matches both
        'veg' and 'non-veg')."""
        result = run_mcp.recommendation(dietary_type="Veg")
        dietary_types = {r["dietary_type"] for r in result["recommendations"]}
        self.assertEqual(dietary_types, {"Veg"})

    def test_non_veg_filter_excludes_veg_dishes(self):
        result = run_mcp.recommendation(dietary_type="Non-Veg")
        dietary_types = {r["dietary_type"] for r in result["recommendations"]}
        self.assertEqual(dietary_types, {"Non-Veg"})


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
                    "meal_type": "All Day",
                    "order_date": "2026-01-12",
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

    def test_order_history_returns_most_recent_ten_orders(self):
        run_mcp.recommendations_data = pd.DataFrame(
            [
                {
                    "restaurant_name": "Punjab Grill",
                    "cuisine": "North Indian",
                    "item_name": f"Item {i}",
                    "price_inr": 300 + i,
                    "rating": 4.0 + (i % 3) * 0.1,
                    "tags": "high-protein",
                }
                for i in range(1, 13)
            ]
        )
        run_mcp.order_history_data = pd.DataFrame(
            [
                {
                    "user_id": "USR-001",
                    "restaurant_name": "Punjab Grill",
                    "cuisine": "North Indian",
                    "item_name": f"Item {i}",
                    "quantity": 1,
                    "unit_price_inr": 300 + i,
                    "line_total_inr": 300 + i,
                    "order_status": "Delivered",
                    "meal_type": "Dinner",
                    "order_date": f"2024-01-{i:02d}",
                }
                for i in range(1, 13)
            ]
        )

        result = run_mcp.order_history("USR-001")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["orders"]), 10)
        self.assertEqual(result["orders"][0]["item_name"], "Item 12")
        self.assertEqual(result["orders"][-1]["item_name"], "Item 3")


if __name__ == "__main__":
    unittest.main()
