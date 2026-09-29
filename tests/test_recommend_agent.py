import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from cravr.graph import FEEDBACK, NEW_REQUEST, _route_entry, app
from cravr.recommend.RecommendAgent import (
    ONLY_MATCH_OPTIONS,
    _NO_SELECTION_VALUE,
    _build_recommendation_filters,
    _craving_match_score,
    _dish_multiselect_options,
    _dish_pick_options,
    _handle_dish_multiselect,
    _offer_category_batch,
    _offer_dish_batch,
    _rank_candidates,
    _resolve_multiple_choices,
    _widen_within_restaurant,
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
    def test_summary_is_never_forwarded_as_a_filter(self):
        filters = _build_recommendation_filters({"craving": "spicy food"})
        self.assertNotIn("summary", filters)

    def test_free_text_craving_is_not_sent_as_a_tag(self):
        filters = _build_recommendation_filters(
            {"craving": "noodles", "taste_preference": ["spicy", "tangy"]}
        )
        self.assertEqual(filters["tags"], "spicy,tangy")

    def test_cuisine_is_never_injected_into_tags(self):
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
        self.assertEqual(_build_recommendation_filters({}), {"course": "main"})

    def test_a_specific_round1_dish_becomes_a_secondary_search_term(self):
        # Live regression, 2026-09-28: round 1 ("category") had already
        # named a specific, complete dish ("Palak Paneer"), and round 2
        # ("craving_or_cuisine") picked a pairing/side ("Naan"), not a
        # narrower version of it. craving alone ("Naan") searched for Naan
        # only, silently dropping Palak Paneer -- "giving only Naan". A
        # single combined search string ("Palak Paneer Naan") was tried
        # live and made things worse: it ranked "Paneer Naan" -- a
        # different, unrelated dish -- above genuine Palak Paneer options.
        # So craving stays a single clean term and craving_tag becomes a
        # SEPARATE search (craving_secondary, see search.py), not glued on.
        filters = _build_recommendation_filters(
            {"craving_tag": "Palak Paneer", "craving": "Naan"}
        )
        self.assertEqual(filters["craving"], "Naan")
        self.assertEqual(filters["craving_secondary"], "Palak Paneer")

    def test_a_single_word_category_is_not_promoted_to_a_secondary_search(self):
        # "sandwich" is a vague category, not a specific dish -- round 2
        # narrowing it down to "Grilled Cheese" must win outright, not get
        # diluted by a secondary search re-admitting every other kind of
        # sandwich (this is exactly the over-broad-options bug already
        # fixed once in search.py's _matches_craving).
        filters = _build_recommendation_filters(
            {"craving_tag": "sandwich", "craving": "Grilled Cheese"}
        )
        self.assertEqual(filters["craving"], "Grilled Cheese")
        self.assertNotIn("craving_secondary", filters)

    def test_a_cuisine_name_is_not_promoted_to_a_secondary_search(self):
        # A cuisine name isn't a dish -- search_menu returns poor results
        # for a bare cuisine query (this is the "North Indian" -> 272
        # unfiltered options bug), so it must not become craving_secondary
        # even though it's two words.
        filters = _build_recommendation_filters(
            {"craving_tag": "North Indian", "craving": "Palak Paneer"}
        )
        self.assertEqual(filters["craving"], "Palak Paneer")
        self.assertNotIn("craving_secondary", filters)

    def test_a_generic_mood_tag_is_not_promoted_to_a_secondary_search(self):
        filters = _build_recommendation_filters(
            {"craving_tag": "comfort-food", "craving": "Grilled Cheese"}
        )
        self.assertEqual(filters["craving"], "Grilled Cheese")
        self.assertNotIn("craving_secondary", filters)

    def test_a_craving_tag_repeating_the_craving_is_not_duplicated(self):
        filters = _build_recommendation_filters(
            {"craving_tag": "Palak Paneer", "craving": "Palak Paneer"}
        )
        self.assertEqual(filters["craving"], "Palak Paneer")
        self.assertNotIn("craving_secondary", filters)


class FilterRelaxationTests(unittest.TestCase):
    def test_diet_and_budget_are_never_relaxed(self):
        variants = _filter_variants(
            {"tags": "spicy", "cuisine": "Indian", "dietary_type": "Veg", "max_price_inr": 1000}
        )
        for variant in variants:
            self.assertEqual(variant["dietary_type"], "Veg")
            self.assertEqual(variant["max_price_inr"], 1000)
        self.assertEqual(variants[-1], {"dietary_type": "Veg", "max_price_inr": 1000})

    def test_cuisine_is_relaxed_before_the_stated_taste(self):
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

        self.assertEqual([c.get("course") for c in calls], ["main", "main,side", "main"])
        self.assertEqual(results[0]["item_name"], "Pad Thai")

    def test_relaxed_results_are_still_ranked_by_the_original_preference(self):
        whole_menu = [
            {"item_name": "Filter Coffee", "cuisine": "South Indian", "rating": 4.6,
             "tags": "vegetarian;breakfast;sugar-free-option;sweet"},
            {"item_name": "Chole Bhature", "cuisine": "North Indian", "rating": 4.3,
             "tags": "vegetarian;breakfast;high-protein;spicy"},
        ]

        def fake_call(filters):
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


class CravingMatchScoreTests(unittest.TestCase):
    def test_an_exact_match_outscores_a_partial_one(self):
        self.assertGreater(
            _craving_match_score("Palak Paneer", "Palak Paneer", "Naan"),
            _craving_match_score("Paneer Naan", "Palak Paneer", "Naan"),
        )

    def test_matching_both_phrases_outscores_matching_one(self):
        self.assertGreater(
            _craving_match_score("Grilled Cheese Sandwich", "Grilled Cheese", "sandwich"),
            _craving_match_score("Veg Achaari Sandwich", "Grilled Cheese", "sandwich"),
        )

    def test_no_match_scores_zero(self):
        self.assertEqual(_craving_match_score("Butter Chicken", "Palak Paneer", "Naan"), 0)

    def test_a_missing_phrase_is_skipped_without_erroring(self):
        self.assertEqual(_craving_match_score("Naan", "Naan", None), 2)


class RankCandidatesCravingRelevanceTests(unittest.TestCase):
    """Live regression, 2026-09-28: round 1 gave a specific dish ("Palak
    Paneer") and round 2 picked a side ("Naan"), so both get searched
    separately (search.py) and merged into one pool. Without a relevance
    score to rank that merged pool by, "Paneer Naan" (only related to the
    side) tied with "Palak Paneer" (the actual dish asked for) on every
    other ranking criterion and could out-rank it on rating alone."""

    def test_the_dish_that_actually_matches_wins_over_a_look_alike(self):
        candidates = [
            {"item_name": "Paneer Naan", "rating": 4.9, "course": "main"},
            {"item_name": "Palak Paneer", "rating": 4.2, "course": "main"},
        ]
        ranked = _rank_candidates(
            candidates, [], None, "main", {},
            desired_craving="Naan", desired_craving_secondary="Palak Paneer",
        )
        self.assertEqual(ranked[0]["item_name"], "Palak Paneer")

    def test_a_generic_side_match_does_not_dilute_an_exact_narrowing_pick(self):
        # No secondary term here (single-word categories like "sandwich"
        # never become craving_secondary -- see _build_recommendation_filters),
        # so this just confirms plain craving relevance still sorts first.
        candidates = [
            {"item_name": "Veg Achaari Sandwich", "rating": 4.9, "course": "main"},
            {"item_name": "Grilled Cheese", "rating": 4.2, "course": "main"},
        ]
        ranked = _rank_candidates(
            candidates, [], None, "main", {}, desired_craving="Grilled Cheese",
        )
        self.assertEqual(ranked[0]["item_name"], "Grilled Cheese")

    def test_no_craving_terms_falls_back_to_existing_ranking_unchanged(self):
        candidates = [
            {"item_name": "A", "rating": 4.0, "course": "main"},
            {"item_name": "B", "rating": 4.9, "course": "main"},
        ]
        ranked = _rank_candidates(candidates, [], None, "main", {})
        self.assertEqual(ranked[0]["item_name"], "B")


class CourseTests(unittest.TestCase):
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
        self.assertIsNone(_resolve_choice("the thai one", self.MENU[:3]))
        self.assertIsNotNone(_resolve_choice("two", self.MENU[:3]))
        self.assertIsNotNone(_resolve_choice("number two", self.MENU[:3]))

    def test_rejection_offers_the_next_batch_before_giving_up(self):
        result = self._follow_up(self.MENU[:3], "something else", alternates=self.MENU[3:])
        self.assertIn("Chole Bhature", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["awaiting_feedback"])

    def test_asking_for_more_serves_the_held_back_dishes(self):
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
        self.assertNotIn("1. Hyderabadi", content)

    def test_an_unrecognised_reply_never_reprints_the_same_list(self):
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
        failure_reply = {"cart_state": {}, "messages": [AIMessage(
            content="Couldn't add that to your cart: timeout", name="cart_agent")]}
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart", return_value=failure_reply):
            result = self._follow_up("2")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Couldn't add that to your cart", result["messages"][0].content)


class RecommendationOptionsTests(unittest.TestCase):
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


class RestaurantScopedWidenTests(unittest.TestCase):
    """Regression for the 'add more from this restaurant' -> decline -> widen
    flow (live bug, 2026-09-29): after adding Palak Paneer to the cart from
    Punjabi Angithi and asking for more items from that restaurant, declining
    the single suggestion used to fall back to a global, stale-brief widen
    that surfaced "Palak Paneer from Punjab Grill" -- a different restaurant
    entirely. recommendation_state.restaurant_scope (stamped by
    cart/flows.py's _suggest_more_from_restaurant) is how
    recommendation_agent_node now tells that case apart from an ordinary
    catalog-wide "only match, widen" and keeps the widen within the same
    restaurant instead."""

    MENU = ThreeSuggestionTests.MENU

    def _restaurant_scope(self):
        return {
            "restaurant_id": "336078",
            "restaurant_name": "Punjabi Angithi (Vegorama Group)",
            "cart_item_id": "63356710",
            "existing_ids": ["63356710"],
        }

    def test_declining_a_restaurant_scoped_only_match_widens_within_the_same_restaurant(self):
        offered = [{"item_name": "Soya Angithi Meal", "menu_item_id": "999", "restaurant_id": "336078"}]
        restaurant_scope = self._restaurant_scope()
        state = {
            "messages": [{"role": "user", "content": "2"}],
            "recommendation_brief": {
                "filters": {"craving": "Naan", "craving_secondary": "Palak Paneer", "course": "main,side"},
                "history": {}, "summary": "",
            },
            "recommendation_state": {
                "awaiting_feedback": True, "satisfied": False,
                "last_recommendations": offered, "alternates": [],
                "restaurant_scope": restaurant_scope,
            },
        }
        canned_reply = {"recommendation_state": {"awaiting_feedback": True}, "messages": []}
        with patch(
            "cravr.recommend.RecommendAgent._widen_within_restaurant", return_value=canned_reply
        ) as mock_restaurant_widen, patch(
            "cravr.recommend.RecommendAgent.widen_recommendations"
        ) as mock_global_widen:
            result = recommendation_agent_node(state)

        mock_restaurant_widen.assert_called_once()
        mock_global_widen.assert_not_called()
        # _widen_within_restaurant now builds the whole reply itself (a category
        # offer, see CategoryBrowsingTests below) -- the node just forwards it.
        self.assertIs(result, canned_reply)

    def test_declining_within_a_restaurant_falls_back_to_a_browse_elsewhere_message_when_nothing_is_left(self):
        offered = [{"item_name": "Soya Angithi Meal", "menu_item_id": "999", "restaurant_id": "336078"}]
        restaurant_scope = self._restaurant_scope()
        state = {
            "messages": [{"role": "user", "content": "2"}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": {
                "awaiting_feedback": True, "satisfied": False,
                "last_recommendations": offered, "alternates": [],
                "restaurant_scope": restaurant_scope,
            },
        }
        with patch(
            "cravr.recommend.RecommendAgent._widen_within_restaurant", return_value=None
        ), patch("cravr.recommend.RecommendAgent.widen_recommendations") as mock_global_widen:
            result = recommendation_agent_node(state)

        mock_global_widen.assert_not_called()
        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])
        self.assertIn("other restaurants", result["messages"][0].content.lower())

    def test_an_ordinary_catalog_wide_decline_still_uses_the_global_widen(self):
        offered = self.MENU[:1]
        state = {
            "messages": [{"role": "user", "content": "2"}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": {
                "awaiting_feedback": True, "satisfied": False,
                "last_recommendations": offered, "alternates": [],
            },
        }
        with patch(
            "cravr.recommend.RecommendAgent._widen_within_restaurant"
        ) as mock_restaurant_widen, patch(
            "cravr.recommend.RecommendAgent.widen_recommendations", return_value=self.MENU[1:4]
        ) as mock_global_widen:
            result = recommendation_agent_node(state)

        mock_restaurant_widen.assert_not_called()
        mock_global_widen.assert_called_once()
        self.assertEqual(result["recommendation_state"]["last_recommendations"], self.MENU[1:4])

    def test_widen_within_restaurant_pulls_the_same_restaurants_menu_and_offers_its_other_categories(self):
        restaurant_scope = self._restaurant_scope()
        restaurant_scope["current_category"] = ["Curries"]
        offered = [{"item_name": "Soya Angithi Meal", "menu_item_id": "999"}]
        menu_items = [
            {"item_name": "Dal Makhani", "menu_item_id": "1", "tags": "Curries"},
            {"item_name": "Garlic Naan", "menu_item_id": "2", "tags": "Breads"},
        ]

        with patch(
            "cravr.recommend.RecommendAgent.food_mcp.get_restaurant_menu_items",
            return_value=menu_items,
        ) as mock_get_menu:
            result = _widen_within_restaurant(restaurant_scope, offered, "addr1")

        mock_get_menu.assert_called_once_with("336078", "addr1", "Punjabi Angithi (Vegorama Group)")
        self.assertIsNotNone(result)
        self.assertEqual(result["recommendation_state"]["offer_kind"], "category")
        category_names = [c["category"] for c in result["recommendation_state"]["last_recommendations"]]
        # "Curries" was the category just declined -- it's not immediately re-offered.
        self.assertEqual(category_names, ["Breads"])
        # the scope carries forward for a further decline, with current_category cleared
        self.assertNotIn("current_category", result["recommendation_state"]["restaurant_scope"])

    def test_widen_within_restaurant_with_no_restaurant_id_returns_nothing(self):
        self.assertIsNone(_widen_within_restaurant({}, [], "addr1"))

    def test_widen_within_restaurant_with_nothing_left_returns_none(self):
        restaurant_scope = self._restaurant_scope()
        with patch(
            "cravr.recommend.RecommendAgent.food_mcp.get_restaurant_menu_items", return_value=[],
        ):
            self.assertIsNone(_widen_within_restaurant(restaurant_scope, [], "addr1"))


class CategoryBrowsingTests(unittest.TestCase):
    """The 'add more from this restaurant' flow now offers the restaurant's
    own menu categories (Bread, Rice, Chinese, ...) to browse instead of
    guessing a single complementary dish -- see cart/flows.py's
    _suggest_more_from_restaurant and RecommendAgent's
    _offer_category_batch / _handle_category_pick."""

    RESTAURANT_SCOPE = {
        "restaurant_id": "336078", "restaurant_name": "Punjabi Angithi (Vegorama Group)",
        "cart_item_id": "63356710", "existing_ids": ["63356710"],
    }
    BREADS = [
        {"item_name": "Butter Naan", "menu_item_id": "10", "price_inr": 60},
        {"item_name": "Garlic Naan", "menu_item_id": "11", "price_inr": 70},
    ]
    CATEGORIES = [
        {"category": "Breads", "items": BREADS},
        {"category": "Rice", "items": [{"item_name": "Jeera Rice", "menu_item_id": "12", "price_inr": 180}]},
    ]

    def _category_offer_state(self, user_text, categories=None):
        offer = _offer_category_batch(
            "Pick a category:", categories if categories is not None else self.CATEGORIES,
            dict(self.RESTAURANT_SCOPE),
        )
        return {
            "messages": [{"role": "user", "content": user_text}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": offer["recommendation_state"],
            "context_profile": {"address_id": "addr1"},
        }

    def test_picking_a_category_shows_its_own_dishes(self):
        state = self._category_offer_state("Breads")
        result = recommendation_agent_node(state)

        self.assertNotEqual(result["recommendation_state"].get("satisfied"), True)
        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Butter Naan", "Garlic Naan"])
        self.assertEqual(result["recommendation_state"]["restaurant_scope"]["current_category"], ["Breads"])

    def test_a_single_remaining_category_is_still_browsed_not_auto_added(self):
        state = self._category_offer_state("1", categories=self.CATEGORIES[:1])
        result = recommendation_agent_node(state)

        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Butter Naan", "Garlic Naan"])
        self.assertNotEqual(result["recommendation_state"].get("satisfied"), True)

    def test_saying_yes_to_a_single_category_offer_browses_it_not_adds_it(self):
        state = self._category_offer_state("yes", categories=self.CATEGORIES[:1])
        result = recommendation_agent_node(state)

        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Butter Naan", "Garlic Naan"])
        self.assertNotEqual(result["recommendation_state"].get("satisfied"), True)

    def test_a_category_with_a_single_dish_still_offers_it_as_a_multiselect_pick(self):
        state = self._category_offer_state("Rice")
        result = recommendation_agent_node(state)

        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Jeera Rice"])
        self.assertEqual(result["recommendation_state"]["offer_kind"], "dish")
        self.assertEqual(
            result["recommendation_state"]["options"], _dish_multiselect_options([self.CATEGORIES[1]["items"][0]]),
        )

    def test_picking_a_dish_within_a_category_adds_it_to_the_cart_normally(self):
        state = self._category_offer_state("Breads")
        dish_offer = recommendation_agent_node(state)

        pick_state = {
            "messages": [{"role": "user", "content": "1"}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": dish_offer["recommendation_state"],
            "context_profile": {"address_id": "addr1"},
        }
        mock_result = {
            "messages": [AIMessage(content="**Your cart:**\n\n- Butter Naan x1 \u2014 \u20b960", name="cart_agent")],
            "cart_state": {"restaurant": {"id": "336078", "name": "Punjabi Angithi (Vegorama Group)"}},
        }
        with patch(
            "cravr.recommend.RecommendAgent.add_dish_to_cart", return_value=mock_result,
        ) as mock_add:
            result = recommendation_agent_node(pick_state)

        mock_add.assert_called_once()
        self.assertEqual(mock_add.call_args[0][0]["item_name"], "Butter Naan")
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertIn("Added Butter Naan to your cart.", result["messages"][0].content)


class DishMultiSelectTests(unittest.TestCase):
    """The dishes shown after picking one or more categories are ALSO a
    multi-select (OptionChips.jsx's `multi` mode -- see UI/graph_bridge.py
    routing offer_kind == "dish" to components.send_multiselect_message):
    the user can check several dishes and add them all to the cart in one
    go, instead of one at a time across several separate round trips (live
    regression, 2026-09-29: picking 5 categories in one go still took 4
    more round trips to get everything from them into the cart). See
    RecommendAgent._offer_dish_batch and _handle_dish_multiselect."""

    RESTAURANT_SCOPE = {
        "restaurant_id": "336078", "restaurant_name": "Punjabi Angithi (Vegorama Group)",
        "cart_item_id": "63356710", "existing_ids": ["63356710"],
    }
    DISHES = [
        {"item_name": "Butter Naan", "menu_item_id": "10", "price_inr": 60, "restaurant_id": "336078"},
        {"item_name": "Garlic Naan", "menu_item_id": "11", "price_inr": 70, "restaurant_id": "336078"},
        {"item_name": "Jeera Rice", "menu_item_id": "12", "price_inr": 180, "restaurant_id": "336078"},
    ]

    def _dish_offer_state(self, user_text, dishes=None, restaurant_scope=None):
        scope = self.RESTAURANT_SCOPE if restaurant_scope is None else restaurant_scope
        offer = _offer_dish_batch(
            "Here's what's under Breads:", dishes if dishes is not None else self.DISHES,
            dict(scope) if scope else scope,
        )
        return {
            "messages": [{"role": "user", "content": user_text}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": offer["recommendation_state"],
            "context_profile": {"address_id": "addr1"},
        }

    def _success(self, item_name, price):
        return {
            "messages": [AIMessage(
                content=f"**Your cart:**\n\n- {item_name} x1 \u2014 \u20b9{price}", name="cart_agent",
            )],
            "cart_state": {"restaurant": {"id": "336078", "name": "Punjabi Angithi (Vegorama Group)"}},
        }

    def test_offer_dish_batch_shows_every_item_uncapped_as_a_multiselect(self):
        offer = _offer_dish_batch("Here's what's under Breads and Rice:", self.DISHES, self.RESTAURANT_SCOPE)
        rec_state = offer["recommendation_state"]

        self.assertEqual(rec_state["offer_kind"], "dish")
        self.assertEqual(len(rec_state["last_recommendations"]), 3)
        self.assertEqual(rec_state["alternates"], [])
        self.assertEqual(rec_state["options"], _dish_multiselect_options(self.DISHES))

    def test_dish_multiselect_options_fold_price_into_the_label(self):
        options = _dish_multiselect_options(self.DISHES)
        self.assertEqual(options[0]["label"], "Butter Naan \u2014 \u20b960")

    def test_picking_two_dishes_adds_both_in_one_go(self):
        state = self._dish_offer_state("1,3")
        with patch(
            "cravr.recommend.RecommendAgent.add_dish_to_cart",
            side_effect=[self._success("Butter Naan", 60), self._success("Jeera Rice", 180)],
        ) as mock_add:
            result = recommendation_agent_node(state)

        self.assertEqual(mock_add.call_count, 2)
        picked_names = [call.args[0]["item_name"] for call in mock_add.call_args_list]
        self.assertEqual(picked_names, ["Butter Naan", "Jeera Rice"])
        self.assertIn("Added Butter Naan and Jeera Rice to your cart.", result["messages"][0].content)
        self.assertTrue(result["recommendation_state"]["satisfied"])
        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])

    def test_a_dish_that_fails_partway_through_stops_the_batch_and_notes_what_already_landed(self):
        state = self._dish_offer_state("1,2,3")
        failure = {"messages": [AIMessage(content="Couldn't add that to your cart: rejected", name="cart_agent")],
                   "cart_state": {}}
        with patch(
            "cravr.recommend.RecommendAgent.add_dish_to_cart",
            side_effect=[self._success("Butter Naan", 60), failure],
        ) as mock_add:
            result = recommendation_agent_node(state)

        self.assertEqual(mock_add.call_count, 2)  # stopped before the third dish
        self.assertIn("Added Butter Naan first.", result["messages"][0].content)
        self.assertIn("Couldn't add that to your cart", result["messages"][0].content)

    def test_skipping_every_dish_within_a_restaurant_declines_gracefully(self):
        state = self._dish_offer_state(_NO_SELECTION_VALUE)
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart") as mock_add:
            result = recommendation_agent_node(state)

        mock_add.assert_not_called()
        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])
        self.assertIn("this restaurant", result["messages"][0].content.lower())

    def test_saying_yes_to_a_single_dish_offer_adds_it(self):
        state = self._dish_offer_state("yes", dishes=self.DISHES[:1])
        with patch(
            "cravr.recommend.RecommendAgent.add_dish_to_cart",
            return_value=self._success("Butter Naan", 60),
        ) as mock_add:
            result = recommendation_agent_node(state)

        mock_add.assert_called_once()
        self.assertTrue(result["recommendation_state"]["satisfied"])

    def test_handle_dish_multiselect_without_an_address_does_not_touch_the_cart(self):
        with patch("cravr.recommend.RecommendAgent.add_dish_to_cart") as mock_add:
            result = _handle_dish_multiselect(self.DISHES[:2], None, None)

        mock_add.assert_not_called()
        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])
        self.assertIn("Butter Naan", result["messages"][0].content)
        self.assertIn("Garlic Naan", result["messages"][0].content)


class MultiSelectCategoryTests(unittest.TestCase):
    """The category offer is a multi-select (OptionChips.jsx's `multi`
    mode -- see UI/graph_bridge.py routing offer_kind == "category" to
    components.send_multiselect_message): the user can check several
    categories in one submission, which arrives here either as a
    comma-joined value string (UI/components.py's multiselect_answer,
    fired by the chips' submit button) or free-typed text naming more than
    one ("Breads and Rice"). See RecommendAgent._resolve_multiple_choices
    and _handle_category_pick."""

    RESTAURANT_SCOPE = {
        "restaurant_id": "336078", "restaurant_name": "Punjabi Angithi (Vegorama Group)",
        "cart_item_id": "63356710", "existing_ids": ["63356710"],
    }
    BREADS = [
        {"item_name": "Butter Naan", "menu_item_id": "10", "price_inr": 60},
        {"item_name": "Garlic Naan", "menu_item_id": "11", "price_inr": 70},
    ]
    RICE = [{"item_name": "Jeera Rice", "menu_item_id": "12", "price_inr": 180}]
    CHINESE = [{"item_name": "Veg Manchurian", "menu_item_id": "13", "price_inr": 220}]
    CATEGORIES = [
        {"category": "Breads", "items": BREADS},
        {"category": "Rice", "items": RICE},
        {"category": "Chinese", "items": CHINESE},
    ]

    _UNSET = object()

    def _category_offer_state(self, user_text, categories=None, restaurant_scope=_UNSET):
        scope = self.RESTAURANT_SCOPE if restaurant_scope is self._UNSET else restaurant_scope
        offer = _offer_category_batch(
            "Pick some categories:", categories if categories is not None else self.CATEGORIES,
            dict(scope) if scope else scope,
        )
        return {
            "messages": [{"role": "user", "content": user_text}],
            "recommendation_brief": {"filters": {}, "history": {}, "summary": ""},
            "recommendation_state": offer["recommendation_state"],
            "context_profile": {"address_id": "addr1"},
        }

    def test_resolve_multiple_choices_splits_comma_joined_values(self):
        options = [{"item_name": "Breads"}, {"item_name": "Rice"}, {"item_name": "Chinese"}]
        chosen = _resolve_multiple_choices("1,3", options)
        self.assertEqual([o["item_name"] for o in chosen], ["Breads", "Chinese"])

    def test_resolve_multiple_choices_splits_free_typed_and(self):
        options = [{"item_name": "Breads"}, {"item_name": "Rice"}, {"item_name": "Chinese"}]
        chosen = _resolve_multiple_choices("Breads and Chinese", options)
        self.assertEqual([o["item_name"] for o in chosen], ["Breads", "Chinese"])

    def test_resolve_multiple_choices_ignores_unresolvable_tokens_and_dedupes(self):
        options = [{"item_name": "Breads"}, {"item_name": "Rice"}]
        chosen = _resolve_multiple_choices("1, nonsense, 1, Rice", options)
        self.assertEqual([o["item_name"] for o in chosen], ["Breads", "Rice"])

    def test_resolve_multiple_choices_with_no_input_returns_empty(self):
        self.assertEqual(_resolve_multiple_choices("", [{"item_name": "Breads"}]), [])
        self.assertEqual(_resolve_multiple_choices("1", []), [])

    def test_picking_two_categories_merges_their_dishes(self):
        state = self._category_offer_state("1,2")
        result = recommendation_agent_node(state)

        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["Butter Naan", "Garlic Naan", "Jeera Rice"])
        self.assertEqual(
            result["recommendation_state"]["restaurant_scope"]["current_category"],
            ["Breads", "Rice"],
        )
        self.assertIn("Breads and Rice", result["messages"][0].content)

    def test_picking_all_three_categories_joins_labels_with_oxford_comma(self):
        state = self._category_offer_state("Breads, Rice and Chinese")
        result = recommendation_agent_node(state)

        rec_state = result["recommendation_state"]
        names = (
            [d["item_name"] for d in rec_state["last_recommendations"]]
            + [d["item_name"] for d in rec_state.get("alternates") or []]
        )
        self.assertEqual(names, ["Butter Naan", "Garlic Naan", "Jeera Rice", "Veg Manchurian"])
        self.assertIn("Breads, Rice, and Chinese", result["messages"][0].content)

    def test_merged_items_are_deduped_by_menu_item_id(self):
        shared = {"item_name": "House Special Combo", "menu_item_id": "99", "price_inr": 300}
        categories = [
            {"category": "Combos", "items": [shared]},
            {"category": "Specials", "items": [shared]},
        ]
        state = self._category_offer_state("1,2", categories=categories)
        result = recommendation_agent_node(state)

        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(names, ["House Special Combo"])

    def test_skipping_every_category_within_a_restaurant_declines_gracefully(self):
        state = self._category_offer_state(_NO_SELECTION_VALUE)
        result = recommendation_agent_node(state)

        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])
        self.assertIn("this restaurant", result["messages"][0].content.lower())

    def test_skipping_every_category_without_a_restaurant_scope_gives_the_generic_decline(self):
        state = self._category_offer_state(
            _NO_SELECTION_VALUE, categories=self.CATEGORIES[:1], restaurant_scope=None,
        )
        result = recommendation_agent_node(state)

        self.assertFalse(result["recommendation_state"]["awaiting_feedback"])
        self.assertIn("tell me what you want to change", result["messages"][0].content)

    def test_widen_within_restaurant_excludes_every_previously_picked_category(self):
        restaurant_scope = dict(self.RESTAURANT_SCOPE)
        restaurant_scope["current_category"] = ["Breads", "Rice"]
        offered = [{"item_name": "Butter Naan", "menu_item_id": "10"}]
        menu_items = [
            {"item_name": "Butter Naan", "menu_item_id": "10", "tags": "Breads"},
            {"item_name": "Jeera Rice", "menu_item_id": "12", "tags": "Rice"},
            {"item_name": "Veg Manchurian", "menu_item_id": "13", "tags": "Chinese"},
        ]
        with patch(
            "cravr.recommend.RecommendAgent.food_mcp.get_restaurant_menu_items",
            return_value=menu_items,
        ):
            result = _widen_within_restaurant(restaurant_scope, offered, "addr1")

        category_names = [c["category"] for c in result["recommendation_state"]["last_recommendations"]]
        self.assertEqual(category_names, ["Chinese"])


class FeedbackShortcutTests(unittest.TestCase):
    OFFERED = [{"item_name": "Hyderabadi Veg Biryani"}, {"item_name": "Schezwan Paneer"},
               {"item_name": "Green Curry with Rice"}]
    STATE = {"awaiting_feedback": True, "satisfied": False, "last_recommendations": OFFERED}

    def test_reactions_shortcut_to_the_recommender(self):
        for reply in ["3", "the second one", "yes", "something else", "I'll take the biryani"]:
            with self.subTest(reply=reply):
                self.assertTrue(is_pure_feedback(reply, self.STATE))

    def test_the_multiselect_skip_sentinel_shortcuts_to_the_recommender(self):
        self.assertTrue(is_pure_feedback(_NO_SELECTION_VALUE, self.STATE))

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
        edges = {(e.source, e.target, getattr(e, "data", None)) for e in app.get_graph().edges}
        self.assertIn(("__start__", "recommendation_agent", FEEDBACK), edges)
        self.assertIn(("__start__", "intent_discovery", NEW_REQUEST), edges)


if __name__ == "__main__":
    unittest.main()
