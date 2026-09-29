"""Regression tests for the bugs found in logger.txt on real Swiggy data."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from cravr.food_mcp import orders as orders_mod
from cravr.intent import intentAgent
from cravr.recommend.RecommendAgent import _build_recommendation_filters, _summarise_history
from cravr.state import CravrState
from cravr.userPreference import preference, store

STRUCTURED_ORDERS = {
    "orders": [
        {
            "orderId": "237377750766536", "restaurantId": "950281",
            "restaurantName": "Jay Bhavani Vadapav", "orderTotal": "₹338",
            "orderStatus": "Delivered",
            "orderedItems": "JB Vadapav (2),Bread Butter (1),Jb Dabeli (1),Ctc (1)",
            "orderedTime": "May 10, 3:45 PM",
            "actions": [{"type": "PAST_ORDER_CTA_ENUM_REORDER", "reorderMeta": {"orderItems": [
                {"itemId": "149960228", "name": "JB Vadapav", "isVeg": "1", "quantity": "2",
                 "categoryDetails": {"category": "Vadapav"}},
            ]}}],
        },
        {
            "orderId": "237187444315551", "restaurantId": "1",
            "restaurantName": "Chai Point", "orderedItems": "Poha (1),Ginger Chai (1)",
            "orderedTime": "May 8, 10:54 AM",
        },
        {
            "orderId": "2", "restaurantId": "950281", "restaurantName": "Jay Bhavani Vadapav",
            "orderedItems": "JB Vadapav (3)", "orderedTime": "May 1, 4:00 PM",
        },
    ]
}
ORDERS_TEXT = (
    "Found 2 orders:\n"
    "1. Order 237377750766536 — Jay Bhavani Vadapav | May 10, 3:45 PM | Delivered | ₹338 "
    "[reorderable] — JB Vadapav (2),Bread Butter (1)\n"
    "2. Order 237187444315551 — Chai Point | May 8, 10:54 AM | Delivered | ₹299 "
    "[reorderable] — Poha (1),Ginger Chai - Uniflask (Serves 1-2) (1)\n"
)


class OrderHistoryTests(unittest.TestCase):
    def _run(self, raw, structured):
        with patch.object(orders_mod, "_call_tool", return_value=(raw, {"structured_content": structured})):
            return orders_mod.get_order_history(address_id="addr-1")

    def test_structured_orders_are_parsed_even_when_text_is_not_json(self):
        result = self._run(ORDERS_TEXT, STRUCTURED_ORDERS)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 3)
        first = result["orders"][0]
        self.assertEqual(first["restaurant_name"], "Jay Bhavani Vadapav")
        self.assertEqual(first["restaurant_id"], "950281")
        self.assertEqual(first["item_name"], "JB Vadapav")
        self.assertEqual(first["order_time"], "May 10, 3:45 PM")
        self.assertEqual(first["items"][0], {"name": "JB Vadapav", "quantity": 2})
        self.assertEqual(first["reorder_items"][0]["menu_item_id"], "149960228")
        self.assertTrue(first["reorder_items"][0]["is_veg"])

    def test_text_only_response_falls_back_to_parsing_lines(self):
        result = self._run(ORDERS_TEXT, None)
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["orders"][1]["restaurant_name"], "Chai Point")
        self.assertEqual(result["orders"][1]["items"][1]["name"], "Ginger Chai - Uniflask (Serves 1-2)")

    def test_history_summary_now_knows_favourite_restaurants_and_repeats(self):
        result = self._run(ORDERS_TEXT, STRUCTURED_ORDERS)
        summary = _summarise_history(result)
        self.assertEqual(summary["top_restaurants"][0], "Jay Bhavani Vadapav")
        self.assertIn("JB Vadapav", summary["repeat_items"])


class PreferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"CRAVR_PREFS_PATH": os.path.join(self.tmp.name, "p.json")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_empty_store_is_no_results_not_error(self):
        self.assertEqual(store.load_preferences(None)["status"], "no_results")

    def test_save_then_load_round_trips(self):
        store.save_preferences(None, dietary_preference="Veg", default_budget_inr=300)
        store.save_preferences(None, default_budget_inr=None)
        self.assertEqual(store.load_preferences(None)["preferences"], {"dietary_preference": "Veg"})

    def test_real_swiggy_reads_the_store_and_never_calls_the_mock(self):
        store.save_preferences(None, dietary_preference="Veg")
        with patch.object(preference, "using_real_food_mcp", return_value=True), \
             patch.object(preference, "_call_mcp_tool") as mock_call:
            result = preference.fetch_user_preferences(None)
        mock_call.assert_not_called()
        self.assertEqual(result["preferences"]["dietary_preference"], "Veg")

    def test_unreachable_mock_falls_back_to_the_store(self):
        store.save_preferences("USR-9", dietary_preference="Non-Veg")
        with patch.object(preference, "using_real_food_mcp", return_value=False), \
             patch.object(preference, "_call_mcp_tool", return_value={"status": "error", "message": "down"}):
            result = preference.fetch_user_preferences("USR-9")
        self.assertEqual(result["preferences"]["dietary_preference"], "Non-Veg")


class GroundedOptionsTests(unittest.TestCase):
    def test_options_context_carries_real_order_history(self):
        state = {
            "context_profile": {"delivery_area": "DLF Phase 3, Haryana"},
            "history_profile": orders_mod_result(),
        }
        grounding = intentAgent._user_grounding(state)
        self.assertEqual(grounding["delivery_area"], "DLF Phase 3, Haryana")
        self.assertEqual(grounding["favourite_restaurants"][0], "Jay Bhavani Vadapav")
        self.assertIn("JB Vadapav", grounding["recent_orders"][0]["items"])

    def test_history_is_passed_to_the_option_llm(self):
        captured = {}

        def fake_suggest(question, context=None, fallback=None, **_):
            captured.update(context or {})
            return []

        with patch.object(intentAgent, "suggest_options", side_effect=fake_suggest):
            intentAgent._question_for("category", None, {}, {"history_profile": orders_mod_result()})
        self.assertIn("recent_orders", captured)


def orders_mod_result():
    with patch.object(orders_mod, "_call_tool", return_value=("", {"structured_content": STRUCTURED_ORDERS})):
        return orders_mod.get_order_history(address_id="addr-1")


class StableOptionsTests(unittest.TestCase):
    """Options must not change between showing a question and resuming with the answer."""

    def test_a_typed_number_resolves_against_the_options_actually_shown(self):
        calls = {"n": 0}

        def changing_options(question, context=None, fallback=None, **_):
            calls["n"] += 1
            return [{"value": f"Choice{calls['n']}x{i}", "label": f"Choice{calls['n']}x{i}"} for i in range(1, 4)]

        graph = StateGraph(CravrState)
        graph.add_node("intent", intentAgent.intent_discovery_node)
        graph.add_edge(START, "intent")
        graph.add_edge("intent", END)
        app = graph.compile(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "stable-options-test"}}

        with patch.object(intentAgent, "suggest_options", side_effect=changing_options):
            first = app.invoke({"messages": [], "intent_profile": {}}, config)
            shown = first["__interrupt__"][0].value["options"]
            second = app.invoke(Command(resume="3"), config)

        self.assertEqual(second["__interrupt__"][0].value["field"], "craving_or_cuisine")
        # The follow-up question names the pick. Before the fix the replay asked the
        # LLM again, so "3" resolved against a fresh list ("Choice2x3"), not what was shown.
        self.assertEqual(shown[2]["value"], "Choice1x3")
        self.assertIn("Choice1x3", second["__interrupt__"][0].value["question"])
        self.assertEqual(calls["n"], 2)  # one call per question, none on replay


class FreeTextCategoryTests(unittest.TestCase):
    def test_a_free_text_first_answer_is_searched_when_the_second_is_skipped(self):
        filters = _build_recommendation_filters(
            {"craving_tag": "Small wrap", "no_preference": ["craving_or_cuisine"]}
        )
        self.assertEqual(filters["craving"], "Small wrap")

    def test_a_mood_tag_is_not_searched_as_a_dish(self):
        self.assertNotIn("craving", _build_recommendation_filters({"craving_tag": "comfort-food"}))

    def test_a_cuisine_first_answer_becomes_the_cuisine_filter(self):
        filters = _build_recommendation_filters({"craving_tag": "chinese"})
        self.assertEqual(filters["cuisine"], "chinese")
        self.assertNotIn("craving", filters)

    def test_budget_friendly_sets_a_price_cap_instead_of_a_dish_search(self):
        filters = _build_recommendation_filters({"craving_tag": "Budget-friendly"})
        self.assertNotIn("craving", filters)
        self.assertEqual(filters["max_price_inr"], 200)


if __name__ == "__main__":
    unittest.main()
