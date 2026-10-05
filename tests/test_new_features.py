"""Section 3 features 1-6: taste profile, learning from "no", card reasons,
complete the meal, context signals, group mode."""

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage

from cravr.cart import pairing
from cravr.cart.routing import CART_COMPLETE_MEAL_VALUE, _classify_cart_action, is_cart_request
from cravr.clock import IST
from cravr.context import signals as ctx
from cravr.context import weather
from cravr.graph import GROUP, _route_entry
from cravr.group import GROUP_PICK_PREFIX, GROUP_RESULTS_VALUE, group_agent_node, is_group_request
from cravr.group import combine as combiner
from cravr.group import node as group_node
from cravr.group import store as group_store
from cravr.group.web import router as group_router
from cravr.recommend import RecommendAgent as ra
from cravr.recommend import feedback
from cravr.recommend import opening
from cravr.recommend.reasons import card_reason
from cravr.recommend.signals import practical_score
from cravr.userPreference import store, taste


def dish(name, restaurant="Cafe", price=200, rating=4.2, **extra):
    return {"item_name": name, "restaurant_name": restaurant, "restaurant_id": f"r-{restaurant}",
            "menu_item_id": f"m-{name}", "price_inr": price, "rating": rating, "course": "main", **extra}


class TempStores(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "CRAVR_PREFS_PATH": os.path.join(self.tmp.name, "prefs.json"),
            "CRAVR_GROUPS_PATH": os.path.join(self.tmp.name, "groups.json"),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()


# ------------------------------------------------------------------ 1. taste profile
class TasteProfileTests(TempStores):
    def test_picks_build_likes_typical_price_and_an_inferred_diet(self):
        for price in (150, 200, 250, 180):
            taste.record_pick(None, dish("Paneer Wrap", "Wrap House", price, dietary_type="Veg"))
        prefs = store.load_preferences(None)["preferences"]
        self.assertEqual(prefs["liked_items"]["paneer wrap"], 4)
        self.assertEqual(prefs["liked_restaurants"]["wrap house"], 4)
        self.assertEqual(prefs["typical_price_inr"], 190)
        self.assertEqual(prefs["dietary_preference"], "Veg")
        self.assertTrue(prefs["dietary_preference_inferred"])

    def test_an_explicit_diet_is_never_overridden(self):
        store.save_preferences(None, dietary_preference="Non-Veg")
        for _ in range(4):
            taste.record_pick(None, dish("Dal", dietary_type="Veg"))
        self.assertEqual(store.load_preferences(None)["preferences"]["dietary_preference"], "Non-Veg")

    def test_liked_dishes_rank_up_and_disliked_rank_down(self):
        taste.record_pick(None, dish("Paneer Wrap", "Wrap House"))
        taste.record_rejection(None, [dish("Veg Burger", "Burger Co")], feedback.NOT_FOR_ME)
        signals = taste.taste_signals(store.load_preferences(None))
        liked = practical_score(dish("Paneer Wrap", "Wrap House"), signals)
        neutral = practical_score(dish("Masala Dosa", "Dosa Co"), signals)
        disliked = practical_score(dish("Veg Burger", "Burger Co"), signals)
        self.assertGreater(liked, neutral)
        self.assertLess(disliked, neutral)

    def test_price_reasons_do_not_mark_the_food_as_disliked(self):
        taste.record_rejection(None, [dish("Biryani")], feedback.TOO_PRICEY)
        prefs = store.load_preferences(None)["preferences"]
        self.assertEqual(prefs["rejection_reasons"], {"too_pricey": 1})
        self.assertNotIn("disliked_items", prefs)

    def test_brief_carries_taste_signals(self):
        taste.record_pick(None, dish("Paneer Wrap", "Wrap House"))
        brief = ra.build_recommendation_brief({"craving": "wrap"}, {}, store.load_preferences(None))
        self.assertEqual(brief["history"]["liked_items"], {"paneer wrap": 1})


# ------------------------------------------------------------------ 2. learning from "no"
class RejectionReasonTests(TempStores):
    OFFERED = [dish("Butter Chicken", price=400, eta_min=45), dish("Cheese Pizza", price=350, eta_min=40),
               dish("Paneer Makhani", price=380, eta_min=50)]

    def test_reasons_are_read_from_plain_words(self):
        cases = {"too expensive": feedback.TOO_PRICEY, "something cheaper": feedback.TOO_PRICEY,
                 "too heavy": feedback.TOO_HEAVY, "something lighter": feedback.TOO_HEAVY,
                 "takes too long": feedback.TOO_SLOW, "too spicy": feedback.TOO_SPICY,
                 "not spicy enough": feedback.NOT_SPICY_ENOUGH,
                 "not in the mood for these": feedback.NOT_FOR_ME, "biryani": None,
                 "I want something filling": None}
        for text, reason in cases.items():
            with self.subTest(text=text):
                self.assertEqual(feedback.infer_reason(text), reason)

    def _state(self, **rec):
        return {"messages": [HumanMessage(content=rec.pop("text"))], "recommendation_brief": {
                    "filters": {"craving": "dinner", "course": "main"}, "history": {}},
                "recommendation_state": {"awaiting_feedback": True, "satisfied": False,
                                         "last_recommendations": self.OFFERED, **rec},
                "context_profile": {"address_id": "addr"}}

    def test_a_bare_no_asks_why_with_four_taps(self):
        reply = ra.recommendation_agent_node(self._state(text="no", alternates=[]))
        self.assertEqual(reply["messages"][0].content, feedback.REASON_QUESTION)
        self.assertTrue(reply["recommendation_state"]["awaiting_reason"])
        self.assertEqual(len(reply["recommendation_state"]["options"]), 4)
        self.assertTrue(ra.is_pure_feedback("too pricey", reply["recommendation_state"]))

    def test_too_pricey_caps_price_and_searches_again(self):
        fresh = [dish("Veg Thali", price=180), dish("Butter Chicken", price=400), dish("Dal Rice", price=150)]
        with patch.object(ra, "fetch_recommendations", return_value=fresh) as search:
            reply = ra.recommendation_agent_node(self._state(text="too pricey", alternates=[]))
        cap = search.call_args[0][0]["filters"]["max_price_inr"]
        self.assertEqual(cap, 304)  # 80% of the median offered price
        names = [d["item_name"] for d in reply["recommendation_state"]["last_recommendations"]]
        self.assertEqual(sorted(names), ["Dal Rice", "Veg Thali"])  # shown & over-cap dishes gone
        self.assertEqual(store.load_preferences("addr")["preferences"]["rejection_reasons"], {"too_pricey": 1})

    def test_too_heavy_drops_heavy_dishes_and_prefers_light_ones(self):
        held = [dish("Double Cheese Burger"), dish("Grilled Chicken Salad"), dish("Veg Biryani"),
                dish("Tomato Soup"), dish("Dal Khichdi")]
        with patch.object(ra, "fetch_recommendations", return_value=[]):
            reply = ra.recommendation_agent_node(self._state(text="too heavy", alternates=held))
        names = [d["item_name"] for d in reply["recommendation_state"]["last_recommendations"]]
        self.assertEqual(set(names), {"Grilled Chicken Salad", "Tomato Soup", "Dal Khichdi"})

    def test_too_slow_caps_delivery_time(self):
        held = [dish("Thali A", eta_min=20), dish("Thali B", eta_min=50), dish("Thali C", eta_min=30)]
        with patch.object(ra, "fetch_recommendations", return_value=[]):
            reply = ra.recommendation_agent_node(self._state(text="takes too long", alternates=held))
        self.assertEqual(reply["recommendation_state"]["adjustments"]["max_eta_min"], 35)
        names = [d["item_name"] for d in reply["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Thali A", "Thali C"])

    def test_answering_the_why_question_applies_it(self):
        with patch.object(ra, "fetch_recommendations", return_value=[dish("Idli", price=90)]):
            reply = ra.recommendation_agent_node(self._state(text="too pricey", alternates=[], awaiting_reason=True))
        self.assertEqual(reply["recommendation_state"]["last_recommendations"][0]["item_name"], "Idli")


# ------------------------------------------------------------------ 3. reasons on every card
class CardReasonTests(unittest.TestCase):
    def test_history_beats_everything(self):
        self.assertEqual(card_reason(dish("Paneer Wrap", eta_min=20), {"item_counts": {"paneer wrap": 3}}),
                         "You've had this 3 times · Arrives in ~20 min")

    def test_regular_restaurant_speed_offer_rating_budget(self):
        signals = {"restaurant_counts": {"chai point": 2}}
        self.assertEqual(card_reason(dish("Poha", "Chai Point", eta_min=18), signals),
                         "From Chai Point, a regular of yours · Arrives in ~18 min")
        self.assertEqual(card_reason(dish("Thali", offer="50% OFF", rating=4.7), {}), "50% OFF · Rated 4.7")
        self.assertEqual(card_reason(dish("Thali", price=180, rating=4.1), {}, {"max_price_inr": 200}),
                         "Within your ₹200 budget")
        self.assertEqual(card_reason(dish("Biryani", rating=4.1), {}, {"craving": "biryani"}), "Matches “biryani”")

    def test_search_results_come_back_with_reasons(self):
        with patch.object(ra, "_call_recommendation_tool", return_value=[dish("Biryani", rating=4.6)]):
            ranked = ra.fetch_recommendations({"filters": {"craving": "biryani", "course": "main"}, "history": {}})
        self.assertIn("reason", ranked[0])
        self.assertIn("Biryani", ra._numbered(ranked))


# ------------------------------------------------------------------ 4. complete the meal
class CompleteTheMealTests(unittest.TestCase):
    MENU = [
        dish("Chicken Biryani", price=320, tags="Biryani"),
        dish("Boondi Raita", price=60, tags="Sides", rating=4.5),
        dish("Mirchi Salan", price=80, tags="Sides", rating=4.3),
        dish("Garlic Naan Basket", price=400, tags="Breads"),  # over the side cap
        dish("Masala Chaas", price=50, tags="Beverages", rating=4.4),
        dish("Coke 500ml", price=60, tags="Beverages"),
        dish("Mango Lassi", price=90, tags="Beverages; has addons"),  # needs customisation
    ]

    def test_classifies_and_suggests_two_sides_and_two_drinks(self):
        picks = pairing.suggest_pairings(self.MENU, main_price=320, exclude_ids={"m-Chicken Biryani"})
        self.assertEqual([p["item_name"] for p in picks], ["Boondi Raita", "Mirchi Salan", "Masala Chaas", "Coke 500ml"])
        self.assertTrue(all(p["reason"] for p in picks))

    def test_budget_left_is_respected(self):
        picks = pairing.suggest_pairings(self.MENU, 320, set(), budget_left=55)
        self.assertEqual([p["item_name"] for p in picks], ["Masala Chaas"])

    def test_button_and_words_route_to_the_cart_agent(self):
        self.assertTrue(is_cart_request(CART_COMPLETE_MEAL_VALUE))
        self.assertEqual(_classify_cart_action(CART_COMPLETE_MEAL_VALUE)[0], "complete_meal")
        self.assertEqual(_classify_cart_action("add a drink")[0], "complete_meal")

    def test_offers_a_multiselect_from_the_same_restaurant(self):
        state = {"cart_state": {"restaurant_id": "r1", "restaurant": "Biryani Blues"},
                 "recommendation_state": {"last_recommendations": [dish("Chicken Biryani", price=320)]}}
        with patch.object(pairing, "_current_restaurant_id", return_value="r1"), \
             patch.object(pairing, "_current_restaurant_name", return_value="Biryani Blues"), \
             patch.object(pairing, "using_real_food_mcp", return_value=False), \
             patch.object(pairing, "_menu_for", return_value=self.MENU):
            reply = pairing.complete_the_meal(state, "addr")
        rec = reply["recommendation_state"]
        self.assertEqual(rec["offer_kind"], "dish")
        self.assertEqual(rec["restaurant_scope"]["restaurant_id"], "r1")
        self.assertIn("Boondi Raita", [d["item_name"] for d in rec["last_recommendations"]])


# ------------------------------------------------------------------ 5. context signals
class ContextSignalTests(unittest.TestCase):
    def test_moods(self):
        self.assertEqual(ctx.detect_mood("so hungover today"), "hungover")
        self.assertEqual(ctx.detect_mood("just back from the gym"), "post_workout")
        self.assertIsNone(ctx.detect_mood("something low calorie"))

    def test_occasions(self):
        friday_night = datetime(2026, 10, 2, 21, 0, tzinfo=IST)
        self.assertEqual(ctx.detect_occasion(friday_night), "weekend_night")
        self.assertEqual(ctx.detect_occasion(datetime(2026, 9, 30, 21, 0, tzinfo=IST), "watching the match"), "match_night")
        self.assertIsNone(ctx.detect_occasion(datetime(2026, 9, 30, 15, 0, tzinfo=IST)))

    def test_mood_beats_weather_beats_occasion(self):
        rain = {"condition": "rain", "label": "Rainy", "temp_c": 24}
        self.assertEqual(ctx.theme("hungover", rain, "weekend_night")["label"], "Hangover helper")
        self.assertEqual(ctx.theme(None, rain, "weekend_night")["label"], "Rainy day pick")
        self.assertEqual(ctx.theme(None, {"condition": "pleasant"}, "weekend_night")["label"], "Weekend treat")

    def test_weather_classification_and_address_parsing(self):
        self.assertEqual(weather.classify(26, 61, 1.2)["condition"], "rain")
        self.assertEqual(weather.classify(38, 0, 0)["condition"], "hot")
        self.assertEqual(weather._candidate_places("N-803, Magarpatta, Hadapsar, Pune, Maharashtra, India")[:2],
                         ["Maharashtra", "Pune"])

    def test_weather_off_means_no_network(self):
        with patch.dict(os.environ, {"CRAVR_WEATHER": "0"}), patch.object(weather.httpx, "Client") as client:
            self.assertIsNone(weather.current_weather("Pune, Maharashtra, India"))
        client.assert_not_called()

    def test_rainy_evening_changes_the_right_now_card(self):
        now = datetime(2026, 9, 30, 17, 0, tzinfo=IST)
        state = {"context_profile": {"address_id": "addr", "weather": {"condition": "rain", "label": "Rainy", "temp_c": 24}},
                 "history_profile": {}, "intent_profile": {}, "preference_profile": {}}
        searched = []

        def fake_search(query, mock_filters, address_id, dietary_type):
            searched.append(query)
            return [dish("Onion Pakoda", rating=4.4)] if query == "pakoda" else []

        with patch.object(opening, "_search", side_effect=fake_search), \
             patch.object(opening, "using_real_food_mcp", return_value=True):
            picks = opening.build_opening_picks(state, now)
        self.assertIn("pakoda", searched)
        self.assertEqual(picks[0]["badge"], "Rainy day pick")
        self.assertIn("24°C", picks[0]["reason"])


# ------------------------------------------------------------------ 6. group mode
class GroupModeTests(TempStores):
    RESPONSES = [
        {"name": "Riya", "diet": "veg", "budget": "250", "spice": "mild", "likes": ["North Indian", "Chinese"], "dislikes": ["Pizza"]},
        {"name": "Aman", "diet": "nonveg", "budget": "400", "spice": "spicy", "likes": ["North Indian", "Biryani"], "dislikes": []},
        {"name": "Kabir", "diet": "either", "budget": "400", "spice": "medium", "likes": ["Chinese"], "dislikes": [], "allergies": "peanuts"},
    ]

    def test_detection_and_routing(self):
        self.assertTrue(is_group_request("group order for 4"))
        self.assertTrue(is_group_request("order for my flatmates"))
        self.assertTrue(is_group_request("dinner for 4 of us"))
        self.assertFalse(is_group_request("biryani for 2 people"))
        self.assertEqual(_route_entry({"messages": [HumanMessage(content="order for four flatmates")]}), GROUP)
        self.assertEqual(group_node._group_size("dinner for four of us"), 4)

    def test_combining_answers(self):
        c = combiner.combine(self.RESPONSES)
        self.assertEqual(c["budget_per_person"], 250)
        self.assertEqual(c["spice"], "mild")
        self.assertEqual(c["veg_count"], 1)
        self.assertEqual(c["top_cuisines"][:2], ["North Indian", "Chinese"])
        self.assertEqual(c["vetoes"], ["Pizza"])
        self.assertIn("Kabir: peanuts", c["allergies"])

    def test_ranking_respects_vetoes_budget_and_votes(self):
        c = combiner.combine(self.RESPONSES)
        restaurants = [
            {"restaurant_id": "1", "restaurant_name": "Pizza Hub", "cuisines": ["Pizzas"], "rating": 4.8, "cost_for_two": "₹300 for two"},
            {"restaurant_id": "2", "restaurant_name": "Punjab Grill", "cuisines": ["North Indian", "Chinese"], "rating": 4.3,
             "cost_for_two": "₹400 for two", "eta_min": 25},
            {"restaurant_id": "3", "restaurant_name": "Fancy Place", "cuisines": ["North Indian"], "rating": 4.5, "cost_for_two": "₹1400 for two"},
            {"restaurant_id": "4", "restaurant_name": "Closed Wok", "cuisines": ["Chinese"], "is_open": False},
        ]
        ranked = combiner.rank_restaurants(restaurants, c)
        self.assertEqual([r["restaurant_name"] for r in ranked], ["Punjab Grill", "Fancy Place"])

    def test_poll_page_round_trip(self):
        group = group_store.create_group(3)
        app = FastAPI()
        app.include_router(group_router)
        client = TestClient(app)
        page = client.get(f"/group/{group['id']}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("What should we order?", page.text)
        done = client.post(f"/group/{group['id']}", data={
            "name": "Riya", "diet": "veg", "budget": "250", "spice": "mild",
            "likes": ["North Indian", "<script>"], "dislikes": ["Pizza"]})
        self.assertIn("1 of 3", done.text)
        saved = group_store.get_group(group["id"])["responses"][0]
        self.assertEqual(saved["likes"], ["North Indian"])  # unknown choices dropped
        self.assertEqual(client.get("/group/nope").status_code, 404)

    def test_full_chat_flow(self):
        with patch.object(group_node, "using_real_food_mcp", return_value=False):
            start = group_agent_node({"messages": [HumanMessage(content="group order for 3 flatmates")]})
            gs = start["group_state"]
            self.assertIn("/group/", start["messages"][0].content)
            for r in self.RESPONSES:
                group_store.add_response(gs["group_id"], r)

            restaurants = [{"restaurant_id": "2", "restaurant_name": "Punjab Grill", "cuisines": ["North Indian"],
                            "rating": 4.3, "cost_for_two": "₹400 for two", "is_open": True}]
            with patch.object(group_node, "_candidate_restaurants", return_value=restaurants):
                results = group_agent_node({"messages": [HumanMessage(content=GROUP_RESULTS_VALUE)], "group_state": gs})
            gs = results["group_state"]
            self.assertIn("Budget ₹250 each", results["messages"][0].content)
            self.assertEqual(gs["options"][0]["value"], f"{GROUP_PICK_PREFIX}2")

            menu = [dish("Paneer Tikka", "Punjab Grill", dietary_type="Veg"),
                    dish("Chilli Chicken", "Punjab Grill", dietary_type="Non-Veg")]
            with patch.object(ra, "_call_recommendation_tool", return_value=menu):
                picked = group_agent_node({"messages": [HumanMessage(content=f"{GROUP_PICK_PREFIX}2")], "group_state": gs})
        rec = picked["recommendation_state"]
        self.assertEqual(rec["offer_kind"], "dish")
        # Mild group: "Chilli Chicken" is filtered out.
        self.assertEqual([d["item_name"] for d in rec["last_recommendations"]], ["Paneer Tikka"])
        self.assertEqual(picked["recommendation_brief"]["filters"]["group_budget_inr"], 750)


if __name__ == "__main__":
    unittest.main()
