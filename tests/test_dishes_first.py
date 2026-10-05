"""Section 2, items 1-3: dishes-first opening, history-led ranking, and
ranking on delivery time / offers / open status."""

import os
import unittest
from datetime import datetime
from unittest.mock import patch

from cravr.clock import IST
from cravr.decision.decisionAgent import decision_agent_node
from cravr.food_mcp import restaurants as rest_mod
from cravr.intent import intentAgent
from cravr.recommend import opening
from cravr.recommend import RecommendAgent as ra
from cravr.recommend.signals import (
    attach_restaurant_info,
    history_signals,
    parse_order_time,
    practical_score,
)

NOW = datetime(2026, 9, 30, 16, 5, tzinfo=IST)  # 4:05 PM -> evening window

HISTORY = {"status": "success", "orders": [
    {"restaurant_name": "Jay Bhavani Vadapav", "restaurant_id": "950281", "order_time": "Sep 22, 3:45 PM",
     "items": [{"name": "JB Vadapav", "quantity": 2}, {"name": "Ctc", "quantity": 1}],
     "reorder_items": [
         {"menu_item_id": "149960228", "name": "JB Vadapav", "quantity": 2, "price_inr": 79.0,
          "is_veg": True, "category": "Vadapav", "has_variants": True, "has_addons": False},
         {"menu_item_id": "149960300", "name": "Ctc", "quantity": 1, "price_inr": 40.0,
          "is_veg": True, "category": "Tea", "has_variants": False, "has_addons": False},
     ]},
    {"restaurant_name": "Chai Point", "restaurant_id": "111", "order_time": "Sep 20, 10:54 AM",
     "items": [{"name": "Poha", "quantity": 1}],
     "reorder_items": [{"menu_item_id": "5", "name": "Poha", "quantity": 1, "is_veg": True,
                        "category": "Breakfast"}]},
    {"restaurant_name": "Jay Bhavani Vadapav", "restaurant_id": "950281", "order_time": "Sep 10, 4:00 PM",
     "items": [{"name": "JB Vadapav", "quantity": 3}],
     "reorder_items": [{"menu_item_id": "149960228", "name": "JB Vadapav", "quantity": 3,
                        "is_veg": True, "category": "Vadapav", "has_variants": True}]},
]}


def dish(name, restaurant, rid, rating=4.2, **extra):
    return {"item_name": name, "restaurant_name": restaurant, "restaurant_id": rid,
            "menu_item_id": f"m-{name}", "rating": rating, "course": "main", "price_inr": 150, **extra}


class SignalTests(unittest.TestCase):
    def test_swiggy_and_mock_order_times_parse(self):
        self.assertEqual(parse_order_time({"order_time": "May 10, 3:45 PM"}, NOW),
                         datetime(2026, 5, 10, 15, 45, tzinfo=IST))
        # No year and "in the future" -> last year.
        self.assertEqual(parse_order_time({"order_time": "Dec 25, 8:00 PM"}, NOW).year, 2025)
        self.assertEqual(parse_order_time({"order_date": "2026-01-12", "order_time": "20:14"}, NOW),
                         datetime(2026, 1, 12, 20, 14, tzinfo=IST))

    def test_orders_in_the_last_day_are_marked_recent(self):
        history = {"orders": [{"restaurant_name": "Kaku Pizza", "order_time": "Sep 29, 9:47 PM",
                               "items": [{"name": "Tandoori Roti"}]}]}
        signals = history_signals(history, NOW)
        self.assertEqual(signals["recent_items"], ["tandoori roti"])
        self.assertEqual(signals["recent_restaurants"], ["kaku pizza"])

    def test_usual_restaurant_outscores_a_better_rated_stranger(self):
        signals = history_signals(HISTORY, NOW)
        usual = practical_score(dish("Dabeli", "Jay Bhavani Vadapav", "950281", rating=4.0), signals)
        stranger = practical_score(dish("Dabeli", "Somewhere New", "9", rating=4.6), signals)
        self.assertGreater(usual, stranger)

    def test_yesterdays_dish_is_demoted(self):
        history = {"orders": [{"restaurant_name": "Kaku Pizza", "order_time": "Sep 29, 9:47 PM",
                               "items": [{"name": "Paneer Pizza"}]}] * 3}
        signals = history_signals(history, NOW)
        again = practical_score(dish("Paneer Pizza", "Kaku Pizza", "1", rating=4.5), signals)
        other = practical_score(dish("Veg Burger", "Burger Place", "2", rating=4.0), signals)
        self.assertLess(again, other)

    def test_faster_delivery_and_an_offer_win_between_equals(self):
        slow = practical_score(dish("Thali", "A", "1", eta_min=50), {})
        fast = practical_score(dish("Thali", "B", "2", eta_min=20), {})
        offer = practical_score(dish("Thali", "C", "3", eta_min=50, offer="50% off"), {})
        self.assertGreater(fast, slow)
        self.assertGreater(offer, slow)

    def test_closed_restaurants_are_dropped_and_facts_attached(self):
        info = {"1": {"is_open": False}, "2": {"is_open": True, "eta_min": 22, "offer": "Flat ₹50 off"}}
        kept = attach_restaurant_info([dish("A", "Closed", "1"), dish("B", "Open", "2"), dish("C", "Unknown", "3")], info)
        self.assertEqual([d["item_name"] for d in kept], ["B", "C"])
        self.assertEqual(kept[0]["eta_min"], 22)
        self.assertEqual(kept[0]["offer"], "Flat ₹50 off")


class RankingTests(unittest.TestCase):
    def test_relevance_still_comes_first(self):
        brief = ra.build_recommendation_brief({"craving": "Biryani"}, HISTORY)
        ranked = ra._rank_candidates(
            [dish("Vada Pav", "Jay Bhavani Vadapav", "950281", eta_min=10, offer="50% off"),
             dish("Biryani", "Somewhere", "9")],
            [], None, "main", brief["history"], desired_craving="Biryani",
        )
        self.assertEqual(ranked[0]["item_name"], "Biryani")

    def test_between_equally_relevant_dishes_history_then_speed_decides(self):
        brief = ra.build_recommendation_brief({"craving": "Sandwich"}, HISTORY)
        ranked = ra._rank_candidates(
            [dish("Sandwich", "Far Away Cafe", "7", rating=4.4, eta_min=45),
             dish("Sandwich", "Quick Cafe", "8", rating=4.4, eta_min=18),
             dish("Sandwich", "Chai Point", "111", rating=4.1, eta_min=30)],
            [], None, "main", brief["history"], desired_craving="Sandwich",
        )
        self.assertEqual([d["restaurant_name"] for d in ranked],
                         ["Chai Point", "Quick Cafe", "Far Away Cafe"])

    def test_real_search_is_enriched_and_closed_places_dropped(self):
        found = [dish("Sandwich", "Closed Cafe", "1"), dish("Sandwich", "Open Cafe", "2")]
        info = {"1": {"is_open": False}, "2": {"is_open": True, "eta_min": 20}}
        with patch.object(ra, "using_real_food_mcp", return_value=True), \
             patch.object(ra.food_mcp, "restaurant_info", return_value=info), \
             patch.object(ra, "_resolve_address_id", return_value="addr"):
            kept = ra._enrich_with_restaurant_info(found, "addr")
        self.assertEqual([d["restaurant_name"] for d in kept], ["Open Cafe"])


class RestaurantLookupTests(unittest.TestCase):
    STRUCTURED = {"success": True, "data": {
        "restaurants": [
            {"id": "950281", "name": "Jay Bhavani Vadapav", "avgRating": 4.3, "deliveryTimeMinutes": 24,
             "distanceKm": 1.8, "offer": "20% OFF", "availabilityStatus": "OPEN", "cuisines": ["Street Food"]},
            {"id": "42", "name": "Night Owl", "availabilityStatus": "CLOSED"},
        ],
        "dishes": [{"id": "d1", "name": "Vada Pav", "price": 49, "isVeg": True, "restaurantId": "950281",
                    "restaurantName": "Jay Bhavani Vadapav", "restaurantRating": 4.3, "deliveryTimeMinutes": 24}],
    }}

    def setUp(self):
        rest_mod._cache.clear()

    def test_structured_response_is_normalized(self):
        with patch.object(rest_mod, "_call_tool", return_value=("text", {"structured_content": self.STRUCTURED})):
            result = rest_mod.search_restaurants("vada pav", "addr")
        first = result["restaurants"][0]
        self.assertEqual((first["eta_min"], first["offer"], first["is_open"]), (24, "20% OFF", True))
        self.assertFalse(result["restaurants"][1]["is_open"])
        self.assertEqual(result["dishes"][0]["eta_min"], 24)
        self.assertEqual(result["dishes"][0]["dietary_type"], "Veg")

    def test_restaurant_info_matches_by_id_and_caches(self):
        with patch.object(rest_mod, "_call_tool", return_value=("text", {"structured_content": self.STRUCTURED})) as call:
            info = rest_mod.restaurant_info([("950281", "Jay Bhavani Vadapav")], "addr")
            again = rest_mod.restaurant_info([("950281", "Jay Bhavani Vadapav")], "addr")
        self.assertEqual(info["950281"]["eta_min"], 24)
        self.assertEqual(again, info)
        self.assertEqual(call.call_count, 1)


class OpeningPicksTests(unittest.TestCase):
    def _state(self, **extra):
        return {"context_profile": {"address_id": "addr"}, "history_profile": HISTORY,
                "intent_profile": {}, "preference_profile": {}, **extra}

    def _run(self, state, searches):
        def fake_search(query, address_id):
            return {"restaurants": [], "dishes": searches.get(query, [])}

        with patch.object(opening, "using_real_food_mcp", return_value=True), \
             patch.object(opening.food_mcp, "search_restaurants", side_effect=fake_search), \
             patch.object(opening.food_mcp, "restaurant_info", return_value={}), \
             patch.object(opening.food_mcp, "search_menu_items", return_value={"items": []}), \
             patch.object(ra, "using_real_food_mcp", return_value=True), \
             patch.object(ra, "_resolve_address_id", return_value="addr"):
            return opening.build_opening_picks(state, NOW)

    def test_usual_new_and_right_now_cards(self):
        searches = {
            "Vadapav": [dish("JB Vadapav", "Jay Bhavani Vadapav", "950281"),   # already ordered
                        dish("Cheese Vadapav", "Jay Bhavani Vadapav", "950281", rating=4.0),
                        dish("Vada Pav", "Goli Vada Pav", "77", rating=4.5)],
            **{q: [dish("Samosa", "Haldiram's", "88", rating=4.4, eta_min=20)]
               for q in opening.TIME_OF_DAY_QUERIES["evening"]},
        }
        picks = self._run(self._state(), searches)
        self.assertEqual([p["badge"] for p in picks], ["Your usual", "Something new", "Good for an evening snack"])
        usual, new, now = picks
        self.assertEqual((usual["item_name"], usual["menu_item_id"], usual["restaurant_id"]),
                         ("JB Vadapav", "149960228", "950281"))
        self.assertIn("has variants", usual["tags"])  # cart flow will ask for the variant
        self.assertEqual(new["item_name"], "Cheese Vadapav")  # never ordered, at a place they like
        self.assertEqual(now["item_name"], "Samosa")

    def test_usual_skips_what_they_had_in_the_last_day(self):
        history = {"orders": [dict(HISTORY["orders"][0], order_time="Sep 30, 1:00 PM"), *HISTORY["orders"][1:]]}
        picks = self._run(self._state(history_profile=history), {})
        usual = next(p for p in picks if p["badge"] == "Your usual")
        self.assertNotEqual(usual["item_name"], "JB Vadapav")

    def test_no_history_still_opens_from_time_of_day(self):
        searches = {q: [dish("Samosa", "A", "1", rating=4.4), dish("Pav Bhaji", "B", "2", rating=4.3),
                        dish("Momos", "C", "3", rating=4.1)]
                    for q in opening.TIME_OF_DAY_QUERIES["evening"]}
        picks = self._run(self._state(history_profile={}), searches)
        self.assertEqual(len(picks), 3)
        self.assertTrue(picks[0]["badge"].startswith("Good for"))

    def test_veg_preference_filters_cards(self):
        searches = {q: [dish("Chicken Roll", "A", "1", rating=4.9, dietary_type="Non-Veg"),
                        dish("Samosa", "B", "2", rating=4.0, dietary_type="Veg")]
                    for q in opening.TIME_OF_DAY_QUERIES["evening"]}
        state = self._state(history_profile={}, intent_profile={"dietary_preferences": ["Veg"]})
        picks = self._run(state, searches)
        self.assertEqual([p["item_name"] for p in picks], ["Samosa"])

    def test_search_failure_returns_no_cards_instead_of_raising(self):
        with patch.object(opening, "_build", side_effect=RuntimeError("boom")):
            self.assertEqual(opening.build_opening_picks(self._state(), NOW), [])


class DishesFirstFlowTests(unittest.TestCase):
    PICKS = [dict(dish("JB Vadapav", "Jay Bhavani Vadapav", "950281"), badge="Your usual", reason="x"),
             dict(dish("Cheese Vadapav", "Jay Bhavani Vadapav", "950281"), badge="Something new"),
             dict(dish("Samosa", "Haldiram's", "88"), badge="Good for an evening snack")]

    def setUp(self):
        self.env = patch.dict(os.environ, {"CRAVR_DISHES_FIRST": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_a_vague_request_gets_cards_not_questions(self):
        state = {"messages": [], "intent_profile": {}, "history_profile": HISTORY}
        with patch("cravr.recommend.opening.build_opening_picks", return_value=self.PICKS), \
             patch.object(intentAgent, "interrupt", side_effect=AssertionError("asked a question")):
            update = intentAgent.intent_discovery_node(state)
        self.assertEqual(update["intent_profile"]["opening_picks"], self.PICKS)

        decided = decision_agent_node({**state, **update})
        brief = decided["recommendation_brief"]
        self.assertTrue(brief["opening_token"])

        reply = ra.recommendation_agent_node({**state, **update, **decided, "recommendation_state": {}})
        rec = reply["recommendation_state"]
        self.assertEqual(rec["offer_kind"], "opening")
        self.assertEqual(rec["last_recommendations"], self.PICKS)
        self.assertEqual(rec["options"][-1]["label"], "Something else")
        self.assertIn("Your usual: JB Vadapav", reply["messages"][0].content)

        # A pick is feedback (goes to the cart); "Something else" is a new request.
        self.assertTrue(ra.is_pure_feedback("2", rec))
        self.assertFalse(ra.is_pure_feedback("something else", rec))
        self.assertFalse(ra.is_pure_feedback("no", rec))

        # Picking card 1 adds it; the same brief doesn't re-offer the cards.
        with patch.object(ra, "_reply_with_pick", return_value={"messages": [], "recommendation_state": {}}) as pick:
            ra.recommendation_agent_node({**state, **update, **decided, "recommendation_state": rec,
                                          "context_profile": {"address_id": "addr"},
                                          "messages": [{"role": "user", "content": "1"}]})
        self.assertEqual(pick.call_args[0][0]["item_name"], "JB Vadapav")

    def test_something_else_falls_back_to_the_questions(self):
        declined = {"offer_kind": "opening", "awaiting_feedback": True, "satisfied": False,
                    "last_recommendations": self.PICKS}
        state = {"messages": [{"role": "user", "content": "something else"}], "intent_profile": {},
                 "recommendation_state": declined}
        with patch("cravr.recommend.opening.build_opening_picks", side_effect=AssertionError("re-opened")), \
             patch.object(intentAgent, "get_llm", side_effect=RuntimeError("no llm")), \
             patch.object(intentAgent, "suggest_options", return_value=[]), \
             patch.object(intentAgent, "interrupt", side_effect=RuntimeError("asked")) as ask:
            with self.assertRaisesRegex(RuntimeError, "asked"):
                intentAgent.intent_discovery_node(state)
        self.assertEqual(ask.call_args[0][0]["field"], "category")

    def test_a_request_with_a_craving_skips_the_opening(self):
        profile = intentAgent._derive({"craving": "biryani"}, NOW)
        self.assertFalse(intentAgent._should_open_with_dishes({}, profile))

    def test_no_cards_means_the_old_question_flow(self):
        state = {"messages": [], "intent_profile": {}}
        with patch("cravr.recommend.opening.build_opening_picks", return_value=[]), \
             patch.object(intentAgent, "suggest_options", return_value=[]), \
             patch.object(intentAgent, "interrupt", side_effect=RuntimeError("asked")):
            with self.assertRaisesRegex(RuntimeError, "asked"):
                intentAgent.intent_discovery_node(state)


if __name__ == "__main__":
    unittest.main()
