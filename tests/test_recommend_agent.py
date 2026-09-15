import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from cravr.graph import FEEDBACK, NEW_REQUEST, _route_entry, app
from cravr.recommend.RecommendAgent import (
    ONLY_MATCH_OPTIONS,
    _build_recommendation_filters,
    _dish_pick_options,
    build_recommendation_brief,
    _filter_variants,
    _looks_like_acceptance,
    _looks_like_rejection,
    _resolve_choice,
    fetch_recommendations,
    is_pure_feedback,
    recommendation_agent_node,
)


class RecommendationFilterTests(unittest.TestCase):
    """`summary` must never leak into the MCP tool's arguments (see
    _build_recommendation_filters docstring) -- that was a real bug where
    every recommendation call failed argument validation and silently
    returned no results, regardless of how complete the profile was."""

    def test_summary_is_never_forwarded_as_a_filter(self):
        filters = _build_recommendation_filters({"craving": "spicy food"})
        self.assertNotIn("summary", filters)

    def test_free_text_craving_is_not_sent_as_a_tag(self):
        """`tags` is AND-matched against a closed vocabulary in menu.csv;
        "noodles" is not in it, so forwarding it guarantees zero results."""
        filters = _build_recommendation_filters(
            {"craving": "noodles", "taste_preference": ["spicy", "tangy"]}
        )
        self.assertEqual(filters["tags"], "spicy,tangy")

    def test_cuisine_is_never_injected_into_tags(self):
        """The original bug: cuisine fell through to `tags` via
        `craving or cuisine`, so every dish had to carry an "Indian" tag.
        No menu row does -- cuisine is its own column -- so the AND-match
        collapsed the result set to zero on every call."""
        filters = _build_recommendation_filters(
            {"cuisine": "Indian", "taste_preference": ["spicy"]}
        )
        self.assertEqual(filters["tags"], "spicy")
        self.assertEqual(filters["cuisine"], "Indian")

    def test_taste_preference_alone_populates_tags(self):
        filters = _build_recommendation_filters({"taste_preference": ["sweet"]})
        self.assertEqual(filters["tags"], "sweet")

    def test_duplicate_tags_are_not_repeated(self):
        filters = _build_recommendation_filters(
            {"craving": "spicy", "taste_preference": ["spicy"]}
        )
        self.assertEqual(filters["tags"], "spicy")

    def test_dietary_preference_is_translated_to_the_menu_vocabulary(self):
        """menu.csv stores 'Veg'/'Non-Veg' and the MCP tool compares with `==`,
        so the profile's own word 'vegetarian' matched nothing at all."""
        self.assertEqual(
            _build_recommendation_filters({"dietary_preferences": ["vegetarian"]})["dietary_type"],
            "Veg",
        )
        self.assertEqual(
            _build_recommendation_filters({"dietary_preferences": ["non-veg"]})["dietary_type"],
            "Non-Veg",
        )

    def test_unrecognised_dietary_preference_is_dropped_rather_than_zeroing_results(self):
        filters = _build_recommendation_filters({"dietary_preferences": ["no preference"]})
        self.assertNotIn("dietary_type", filters)

    def test_budget_and_dietary_and_cuisine_are_mapped(self):
        filters = _build_recommendation_filters(
            {
                "cuisine": "Thai",
                "dietary_preferences": ["vegetarian", "vegan"],
                "budget_inr": 300,
            }
        )
        self.assertEqual(filters["cuisine"], "Thai")
        self.assertEqual(filters["dietary_type"], "Veg")
        self.assertEqual(filters["max_price_inr"], 300)

    def test_an_empty_profile_still_asks_for_a_meal(self):
        """No preferences is not the same as no constraints: someone with an
        empty profile still wants food, not a coffee or a side of naan."""
        self.assertEqual(_build_recommendation_filters({}), {"course": "main"})


class FilterRelaxationTests(unittest.TestCase):
    def test_diet_and_budget_are_never_relaxed(self):
        variants = _filter_variants(
            {"tags": "spicy", "cuisine": "Indian", "dietary_type": "Veg", "max_price_inr": 1000}
        )
        # A vegetarian must never be shown meat, and a budget is a budget.
        for variant in variants:
            self.assertEqual(variant["dietary_type"], "Veg")
            self.assertEqual(variant["max_price_inr"], 1000)
        self.assertEqual(variants[-1], {"dietary_type": "Veg", "max_price_inr": 1000})

    def test_cuisine_is_relaxed_before_the_stated_taste(self):
        """Taste is usually what the user actually said ("something spicy");
        cuisine is often our inference, so it should give way first."""
        variants = _filter_variants(
            {"tags": "spicy", "cuisine": "Indian", "dietary_type": "Veg"}
        )
        self.assertEqual(variants[1], {"tags": "spicy", "dietary_type": "Veg"})

    def test_fetch_falls_back_to_a_looser_filter_set(self):
        calls = []

        def fake_call(filters):
            calls.append(filters)
            if "cuisine" in filters:
                return []
            return [{"item_name": "Pad Thai", "cuisine": "Thai", "rating": 4.3, "tags": "spicy"}]

        with patch("cravr.recommend.RecommendAgent._call_recommendation_tool", side_effect=fake_call):
            results = fetch_recommendations(build_recommendation_brief({"cuisine": "Indian", "taste_preference": ["spicy"]}, {}))

        # main+cuisine, then main,side+cuisine, then cuisine dropped.
        self.assertEqual([c.get("course") for c in calls], ["main", "main,side", "main"])
        self.assertEqual(results[0]["item_name"], "Pad Thai")

    def test_relaxed_results_are_still_ranked_by_the_original_preference(self):
        """The Filter Coffee bug: once `tags` was relaxed the tool returned
        the whole menu and rating alone decided, so a 4.6 coffee beat every
        spicy dish for a user who asked for something spicy."""
        whole_menu = [
            {"item_name": "Filter Coffee", "cuisine": "South Indian", "rating": 4.6,
             "tags": "vegetarian;breakfast;sugar-free-option;sweet"},
            {"item_name": "Chole Bhature", "cuisine": "North Indian", "rating": 4.3,
             "tags": "vegetarian;breakfast;high-protein;spicy"},
        ]

        def fake_call(filters):
            # Nothing carries the tag (e.g. a stale menu), forcing relaxation.
            return [] if "tags" in filters or "cuisine" in filters else whole_menu

        with patch("cravr.recommend.RecommendAgent._call_recommendation_tool", side_effect=fake_call):
            results = fetch_recommendations(build_recommendation_brief({"cuisine": "Indian", "taste_preference": ["spicy"], "dietary_preferences": ["vegetarian"]}, {}))

        self.assertEqual(results[0]["item_name"], "Chole Bhature")

    def test_history_breaks_ties_toward_familiar_cuisines(self):
        menu = [
            {"item_name": "Pad Thai", "cuisine": "Thai", "rating": 4.4, "tags": "spicy"},
            {"item_name": "Masala Dosa", "cuisine": "South Indian", "rating": 4.4, "tags": "spicy"},
        ]
        history = {"orders": [{"cuisine": "South Indian", "item_name": "Idli"}]}

        with patch("cravr.recommend.RecommendAgent._call_recommendation_tool", return_value=menu):
            results = fetch_recommendations(build_recommendation_brief({"taste_preference": ["spicy"]}, history))

        self.assertEqual(results[0]["item_name"], "Masala Dosa")


class CourseTests(unittest.TestCase):
    """menu.csv's `course` column (main/side/dessert/beverage) exists so a
    ₹70 Filter Coffee can't be the answer to "I want to eat something spicy"."""

    def test_a_meal_request_defaults_to_mains(self):
        filters = _build_recommendation_filters({"taste_preference": ["spicy"]})
        self.assertEqual(filters["course"], "main")

    def test_a_drink_request_asks_for_beverages(self):
        self.assertEqual(
            _build_recommendation_filters({"craving": "cold coffee"})["course"], "beverage"
        )

    def test_an_explicit_dessert_request_asks_for_desserts(self):
        self.assertEqual(
            _build_recommendation_filters({"craving": "some dessert"})["course"], "dessert"
        )

    def test_a_sweet_tooth_allows_dessert_without_excluding_mains(self):
        """"sweet" as a taste is a weaker signal than the word "dessert"."""
        self.assertEqual(
            _build_recommendation_filters({"taste_preference": ["sweet"]})["course"],
            "dessert,main",
        )

    def test_a_snack_request_asks_for_sides(self):
        self.assertEqual(
            _build_recommendation_filters({"craving": "a light snack"})["course"], "side"
        )

    def test_course_is_widened_to_sides_but_never_dropped(self):
        variants = _filter_variants({"tags": "spicy", "cuisine": "Indian", "course": "main"})
        self.assertTrue(all("course" in v for v in variants))
        self.assertEqual({v["course"] for v in variants}, {"main", "main,side"})
        # Widening the course is tried before sacrificing the stated taste.
        self.assertEqual(variants[1], {"tags": "spicy", "cuisine": "Indian", "course": "main,side"})

    def test_a_beverage_request_is_never_widened_into_food(self):
        variants = _filter_variants({"course": "beverage", "cuisine": "Indian"})
        self.assertEqual({v["course"] for v in variants}, {"beverage"})

    def test_mains_outrank_sides_when_both_come_back(self):
        results = [
            {"item_name": "Garlic Naan", "course": "side", "rating": 4.9, "tags": "savory", "cuisine": "North Indian"},
            {"item_name": "Dal Makhani", "course": "main", "rating": 4.2, "tags": "savory", "cuisine": "North Indian"},
        ]
        with patch("cravr.recommend.RecommendAgent._call_recommendation_tool", return_value=results):
            ranked = fetch_recommendations(build_recommendation_brief({"taste_preference": ["savory"]}, {}))
        self.assertEqual(ranked[0]["item_name"], "Dal Makhani")


class FeedbackParsingTests(unittest.TestCase):
    def test_noodles_is_not_read_as_a_rejection(self):
        """Substring matching made any word containing "no" a rejection."""
        self.assertFalse(_looks_like_rejection("I want noodles"))
        self.assertTrue(_looks_like_rejection("no, something else"))

    def test_negated_praise_is_not_read_as_acceptance(self):
        self.assertFalse(_looks_like_acceptance("no, that's not good"))
        self.assertTrue(_looks_like_acceptance("yes, sounds good"))


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

        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=recommendations):
            result = recommendation_agent_node(state)

        self.assertIn("Plain Idli", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["awaiting_feedback"])
        self.assertEqual(result["recommendation_state"]["last_recommendations"], recommendations)
        # A single match offers yes/widen buttons, not a dish-pick list --
        # there's only one dish, nothing to pick between.
        self.assertEqual(result["recommendation_state"]["options"], ONLY_MATCH_OPTIONS)

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

        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=[]):
            result = recommendation_agent_node(state)

        self.assertIn("Plain Idli", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["satisfied"])


class BriefHandoffTests(unittest.TestCase):
    """recommendation_agent reads the frozen brief and nothing else, so the
    profile-to-menu-filter translation happens in exactly one place."""

    def test_brief_carries_filters_history_and_a_concise_summary(self):
        brief = build_recommendation_brief(
            {"cuisine": "Indian", "taste_preference": ["spicy"],
             "dietary_preferences": ["vegetarian"], "budget_inr": 1000},
            {"orders": [
                {"cuisine": "North Indian", "item_name": "Butter Chicken"},
                {"cuisine": "North Indian", "item_name": "Butter Chicken"},
                {"cuisine": "Thai", "item_name": "Pad Thai Noodles"},
            ]},
        )
        self.assertEqual(
            brief["filters"],
            {"tags": "spicy", "cuisine": "Indian", "course": "main",
             "dietary_type": "Veg", "max_price_inr": 1000},
        )
        self.assertEqual(brief["history"]["top_cuisines"], ["North Indian", "Thai"])
        self.assertEqual(brief["history"]["repeat_items"], ["Butter Chicken"])
        self.assertEqual(
            brief["summary"],
            "spicy vegetarian Indian main under ₹1000; usually orders North Indian and Thai; "
            "re-orders Butter Chicken.",
        )

    def test_the_node_uses_the_brief_in_state_over_the_raw_profiles(self):
        seen = {}

        def fake_fetch(brief):
            seen.update(brief)
            return []

        state = {
            "messages": [{"role": "user", "content": "anything"}],
            # Deliberately contradictory: if the node reads these, we'll see it.
            "intent_profile": {"cuisine": "Mexican"},
            "history_profile": {"orders": []},
            "recommendation_brief": {"filters": {"cuisine": "Thai"}, "history": {}, "summary": "thai"},
        }
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", side_effect=fake_fetch):
            recommendation_agent_node(state)

        self.assertEqual(seen["filters"], {"cuisine": "Thai"})


class ThreeSuggestionTests(unittest.TestCase):
    MENU = [
        {"item_name": "Hyderabadi Veg Biryani", "restaurant_name": "Coastal Kitchen",
         "cuisine": "South Indian", "price_inr": 290, "rating": 4.5, "course": "main"},
        {"item_name": "Schezwan Paneer", "restaurant_name": "Dragon Bowl",
         "cuisine": "Chinese", "price_inr": 280, "rating": 4.5, "course": "main"},
        {"item_name": "Green Curry with Rice", "restaurant_name": "Thai Basil",
         "cuisine": "Thai", "price_inr": 360, "rating": 4.5, "course": "main"},
        {"item_name": "Chole Bhature", "restaurant_name": "Punjab Grill",
         "cuisine": "North Indian", "price_inr": 180, "rating": 4.3, "course": "main"},
    ]

    def _offer(self, menu=None):
        state = {"messages": [{"role": "user", "content": "something spicy"}],
                 "recommendation_brief": {"filters": {}, "history": {}, "summary": ""}}
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=menu or self.MENU):
            return recommendation_agent_node(state)

    def _follow_up(self, offered, reply, alternates=()):
        state = {
            "messages": [{"role": "user", "content": reply}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": {"awaiting_feedback": True, "satisfied": False,
                                     "last_recommendations": offered, "alternates": list(alternates)},
        }
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=[]):
            return recommendation_agent_node(state)

    def test_offers_three_and_keeps_the_rest_as_alternates(self):
        result = self._offer()
        content = result["messages"][0].content
        for name in ("Hyderabadi Veg Biryani", "Schezwan Paneer", "Green Curry with Rice"):
            self.assertIn(name, content)
        self.assertNotIn("Chole Bhature", content)
        self.assertEqual(len(result["recommendation_state"]["last_recommendations"]), 3)
        self.assertEqual(len(result["recommendation_state"]["alternates"]), 1)

    def test_shows_fewer_than_three_without_padding(self):
        result = self._offer(self.MENU[:2])
        self.assertEqual(len(result["recommendation_state"]["last_recommendations"]), 2)
        self.assertIn("Here are 2 options", result["messages"][0].content)

    def test_selection_by_number(self):
        result = self._follow_up(self.MENU[:3], "2")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Schezwan Paneer", result["messages"][0].content)

    def test_selection_by_ordinal_word(self):
        result = self._follow_up(self.MENU[:3], "the third one please")
        self.assertIn("Green Curry", result["messages"][0].content)

    def test_selection_by_dish_name(self):
        result = self._follow_up(self.MENU[:3], "I'll take the biryani")
        self.assertIn("Hyderabadi Veg Biryani", result["messages"][0].content)

    def test_a_correction_is_a_selection_not_a_rejection(self):
        """"no, the second one" must pick, not restart the search."""
        result = self._follow_up(self.MENU[:3], "no, the second one")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Schezwan Paneer", result["messages"][0].content)

    def test_bare_yes_against_three_options_asks_which(self):
        result = self._follow_up(self.MENU[:3], "yes")
        self.assertFalse(result["recommendation_state"]["satisfied"])
        self.assertIn("Which one", result["messages"][0].content)

    def test_bare_yes_against_a_single_option_confirms_it(self):
        result = self._follow_up(self.MENU[:1], "yes")
        self.assertTrue(result["recommendation_state"]["satisfied"])

    def test_a_cardinal_inside_a_sentence_is_not_a_pick(self):
        """"the thai one" must not silently select option 1 -- "one" is filler."""
        self.assertIsNone(_resolve_choice("the thai one", self.MENU[:3]))
        self.assertIsNotNone(_resolve_choice("two", self.MENU[:3]))
        self.assertIsNotNone(_resolve_choice("number two", self.MENU[:3]))

    def test_rejection_offers_the_next_batch_before_giving_up(self):
        result = self._follow_up(self.MENU[:3], "something else", alternates=self.MENU[3:])
        self.assertIn("Chole Bhature", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["awaiting_feedback"])

    def test_asking_for_more_serves_the_held_back_dishes(self):
        """"recommend more" used to re-run the same query and reprint the same
        three dishes, discarding the alternates already in hand."""
        for reply in ["recommend more.", "more", "any others?", "what else do you have"]:
            with self.subTest(reply=reply):
                result = self._follow_up(self.MENU[:3], reply, alternates=self.MENU[3:])
                content = result["messages"][0].content
                self.assertIn("Chole Bhature", content)
                self.assertNotIn("Hyderabadi Veg Biryani", content)

    def test_asking_for_more_when_there_are_none_says_so(self):
        result = self._follow_up(self.MENU[:3], "more", alternates=())
        content = result["messages"][0].content
        self.assertIn("That's everything", content)
        self.assertIn("widen", content)
        # Never silently reprint the same three dishes.
        self.assertNotIn("1. Hyderabadi", content)

    def test_an_unrecognised_reply_never_reprints_the_same_list(self):
        """Whatever the phrasing, an unchanged request must not produce the
        identical three dishes again."""
        result = self._follow_up(self.MENU[:3], "hmm, surprise me", alternates=self.MENU[3:])
        self.assertIn("Chole Bhature", result["messages"][0].content)

    def test_a_changed_request_does_re_run_the_search(self):
        state = {
            "messages": [{"role": "user", "content": "actually make it non-veg"}],
            "recommendation_brief": {"filters": {"dietary_type": "Non-Veg"}, "history": {}, "summary": ""},
            "recommendation_state": {
                "awaiting_feedback": True, "satisfied": False,
                "last_recommendations": self.MENU[:3], "alternates": self.MENU[3:],
                "filters": {"dietary_type": "Veg"},
            },
        }
        fresh = [{"item_name": "Butter Chicken", "restaurant_name": "Punjab Grill",
                  "cuisine": "North Indian", "price_inr": 420, "rating": 4.5, "course": "main"}]
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=fresh):
            result = recommendation_agent_node(state)
        self.assertIn("Butter Chicken", result["messages"][0].content)


class PickToCartHandoffTests(unittest.TestCase):
    """Picking a dish auto-adds it to the cart (via
    cart.CartAgent.add_dish_to_cart) when an address is known -- no
    confirmation gate here, since a cart isn't a placed order."""

    DISH = {"item_name": "Schezwan Paneer", "restaurant_name": "Dragon Bowl",
            "menu_item_id": "ITEM1", "restaurant_id": "REST1", "cuisine": "Chinese"}
    MENU = [
        {"item_name": "Hyderabadi Veg Biryani", "restaurant_name": "Coastal Kitchen",
         "cuisine": "South Indian", "price_inr": 290, "rating": 4.5, "course": "main"},
        DISH,
        {"item_name": "Green Curry with Rice", "restaurant_name": "Thai Basil",
         "cuisine": "Thai", "price_inr": 360, "rating": 4.5, "course": "main"},
    ]

    def _follow_up(self, reply, address_id="addr1"):
        state = {
            "messages": [{"role": "user", "content": reply}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "context_profile": {"address_id": address_id} if address_id else {},
            "recommendation_state": {"awaiting_feedback": True, "satisfied": False,
                                     "last_recommendations": self.MENU, "alternates": []},
        }
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=[]):
            return recommendation_agent_node(state)

    def test_pick_with_an_address_calls_add_dish_to_cart(self):
        cart_reply = {"cart_state": {"items": [{"item_name": "Schezwan Paneer"}]},
                      "messages": [AIMessage(content="Great — Schezwan Paneer from Dragon Bowl it is. "
                                              "Added Schezwan Paneer to your cart.\nHere's your cart:\n"
                                              "  - Schezwan Paneer x1 — ₹280", name="cart_agent")]}
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart", return_value=cart_reply) as mock_add:
            result = self._follow_up("2")
        mock_add.assert_called_once()
        called_dish, called_address = mock_add.call_args.args[0], mock_add.call_args.args[1]
        self.assertEqual(called_dish["item_name"], "Schezwan Paneer")
        self.assertEqual(called_address, "addr1")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertEqual(result["recommendation_state"]["last_recommendations"], [self.DISH])
        self.assertIn("Added Schezwan Paneer to your cart", result["messages"][0].content)
        self.assertEqual(result["cart_state"], cart_reply["cart_state"])

    def test_pick_without_an_address_skips_the_cart_call_gracefully(self):
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart") as mock_add:
            result = self._follow_up("2", address_id=None)
        mock_add.assert_not_called()
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Great — Schezwan Paneer", result["messages"][0].content)
        self.assertIn("couldn't add it to your cart", result["messages"][0].content)

    def test_a_cart_failure_still_confirms_the_pick(self):
        """add_dish_to_cart's own failure message (missing IDs, MCP error,
        etc.) is what gets shown -- the pick itself must still register as
        satisfied, not get stuck re-offering the same list."""
        failure_reply = {"cart_state": {}, "messages": [AIMessage(
            content="Couldn't add that to your cart: timeout", name="cart_agent")]}
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart", return_value=failure_reply):
            result = self._follow_up("2")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Couldn't add that to your cart", result["messages"][0].content)


class RecommendationOptionsTests(unittest.TestCase):
    """UI/components.py renders `recommendation_state["options"]` as tappable
    buttons (see UI/graph_bridge.py's process_result), same treatment as a
    clarification question's options -- these check RecommendAgent builds
    the right buttons for each branch, kept in sync with the numbered text
    by construction (`_dish_pick_options`/`ONLY_MATCH_OPTIONS` are the single
    source of truth for both)."""

    MENU = ThreeSuggestionTests.MENU

    def _offer(self, menu=None):
        state = {"messages": [{"role": "user", "content": "something spicy"}],
                 "recommendation_brief": {"filters": {}, "history": {}, "summary": ""}}
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=menu or self.MENU):
            return recommendation_agent_node(state)

    def _follow_up(self, offered, reply, alternates=()):
        state = {
            "messages": [{"role": "user", "content": reply}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": {"awaiting_feedback": True, "satisfied": False,
                                     "last_recommendations": offered, "alternates": list(alternates)},
        }
        with patch("cravr.recommend.RecommendAgent.fetch_recommendations", return_value=[]):
            return recommendation_agent_node(state)

    def test_a_fresh_batch_of_three_gets_one_button_per_dish(self):
        result = self._offer()
        offered = result["recommendation_state"]["last_recommendations"]
        self.assertEqual(result["recommendation_state"]["options"], _dish_pick_options(offered))
        self.assertEqual(
            [o["value"] for o in result["recommendation_state"]["options"]], ["1", "2", "3"],
        )
        self.assertEqual(
            [o["label"] for o in result["recommendation_state"]["options"]],
            ["Hyderabadi Veg Biryani", "Schezwan Paneer", "Green Curry with Rice"],
        )

    def test_fewer_than_three_still_gets_a_button_per_dish(self):
        result = self._offer(self.MENU[:2])
        self.assertEqual(len(result["recommendation_state"]["options"]), 2)

    def test_more_request_carries_options_for_the_new_batch(self):
        result = self._follow_up(self.MENU[:3], "more", alternates=self.MENU[3:])
        self.assertEqual(
            [o["label"] for o in result["recommendation_state"]["options"]], ["Chole Bhature"],
        )

    def test_ambiguous_yes_against_three_offers_the_same_three_as_buttons(self):
        result = self._follow_up(self.MENU[:3], "yes")
        self.assertEqual(result["recommendation_state"]["options"], _dish_pick_options(self.MENU[:3]))

    def test_confirmed_pick_has_no_more_options(self):
        """Once a dish is chosen there's nothing left to pick -- no stale
        buttons should linger under the confirmation message."""
        result = self._follow_up(self.MENU[:3], "2")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertNotIn("options", result["recommendation_state"])

    def test_declining_the_only_match_offers_the_widened_batch_as_buttons(self):
        result = self._offer(self.MENU[:1])
        self.assertEqual(result["recommendation_state"]["options"], ONLY_MATCH_OPTIONS)

        state = {
            "messages": [{"role": "user", "content": "widen the search"}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": {
                "awaiting_feedback": True, "satisfied": False,
                "last_recommendations": self.MENU[:1], "alternates": [],
            },
        }
        with patch("cravr.recommend.RecommendAgent.widen_recommendations", return_value=self.MENU[1:4]):
            result = recommendation_agent_node(state)

        self.assertEqual(
            result["recommendation_state"]["options"], _dish_pick_options(self.MENU[1:4]),
        )


class FeedbackShortcutTests(unittest.TestCase):
    """The graph skips intent/history/decision for pure reactions, so "3"
    is never handed to the extraction model (which could read it as a budget
    of ₹3 and zero out every later search)."""

    OFFERED = [{"item_name": "Hyderabadi Veg Biryani"}, {"item_name": "Schezwan Paneer"},
               {"item_name": "Green Curry with Rice"}]
    STATE = {"awaiting_feedback": True, "satisfied": False, "last_recommendations": OFFERED}

    def test_reactions_shortcut_to_the_recommender(self):
        for reply in ["3", "the second one", "yes", "something else", "I'll take the biryani"]:
            with self.subTest(reply=reply):
                self.assertTrue(is_pure_feedback(reply, self.STATE))

    def test_a_changed_request_takes_the_full_path(self):
        for reply in ["make it non-veg", "something under 200", "actually Thai", "no, milder please"]:
            with self.subTest(reply=reply):
                self.assertFalse(is_pure_feedback(reply, self.STATE))

    def test_nothing_shortcuts_when_no_dishes_are_on_the_table(self):
        self.assertFalse(is_pure_feedback("yes", {}))
        self.assertFalse(is_pure_feedback("yes", {"awaiting_feedback": False, "last_recommendations": self.OFFERED}))
        self.assertFalse(
            is_pure_feedback("yes", {"awaiting_feedback": True, "satisfied": True,
                                     "last_recommendations": self.OFFERED})
        )

    def test_graph_routes_the_entry_edge_both_ways(self):
        base = {"messages": [], "recommendation_state": self.STATE}
        self.assertEqual(_route_entry({**base, "messages": [HumanMessage(content="3")]}), FEEDBACK)
        self.assertEqual(
            _route_entry({**base, "messages": [HumanMessage(content="make it non-veg")]}),
            NEW_REQUEST,
        )

    def test_entry_labels_are_wired_to_the_right_nodes(self):
        """The router returns a reason, not a node name -- so the path map is
        what actually decides where each reason goes. Check they line up."""
        edges = {(e.source, e.target, getattr(e, "data", None)) for e in app.get_graph().edges}
        self.assertIn(("__start__", "recommendation_agent", FEEDBACK), edges)
        self.assertIn(("__start__", "intent_discovery", NEW_REQUEST), edges)


if __name__ == "__main__":
    unittest.main()
