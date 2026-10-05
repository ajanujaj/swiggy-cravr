"""Regressions found in real use (logger.txt, 29-30 Sep)."""

import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from cravr.recommend import RecommendAgent as ra
from cravr.userPreference import store

PICKS = [
    {"item_name": "Herbivores Burger", "restaurant_name": "Louis Burger", "restaurant_id": "417218",
     "menu_item_id": "76582145", "price_inr": 235, "dietary_type": "Veg", "badge": "Your usual"},
    {"item_name": "Samosa", "restaurant_name": "Haldiram's", "restaurant_id": "88", "menu_item_id": "s1",
     "price_inr": 60, "badge": "Good for an evening snack"},
]
BRIEF = {"filters": {"course": "main"}, "history": {}, "opening_picks": PICKS, "opening_token": "tok1"}


class OpeningIsShownOnceTests(unittest.TestCase):
    """Tapping 'Add more from this restaurant' / 'Complete the meal' and then picking
    used to show the opening cards again: the brief still held them."""

    def _after_opening_offered(self):
        state = {"recommendation_brief": BRIEF, "recommendation_state": {}, "messages": []}
        return ra.recommendation_agent_node(state)

    def test_offering_the_opening_consumes_it(self):
        reply = self._after_opening_offered()
        self.assertEqual(reply["recommendation_state"]["offer_kind"], "opening")
        self.assertNotIn("opening_picks", reply["recommendation_brief"])
        self.assertEqual(reply["recommendation_brief"]["opening_token"], "tok1")

    def test_picking_from_the_opening_still_works(self):
        reply = self._after_opening_offered()
        state = {"recommendation_brief": reply["recommendation_brief"],
                 "recommendation_state": reply["recommendation_state"],
                 "context_profile": {"address_id": "addr"},
                 "messages": [HumanMessage(content="1")]}
        with patch.object(ra, "_reply_with_pick", return_value={"messages": [], "recommendation_state": {}}) as pick:
            ra.recommendation_agent_node(state)
        self.assertEqual(pick.call_args[0][0]["item_name"], "Herbivores Burger")

    def test_a_later_restaurant_menu_pick_does_not_restart_the_conversation(self):
        reply = self._after_opening_offered()
        # "Add more from this restaurant" -> category multi-select, set by the cart agent.
        categories = [{"category": "Sides", "items": [PICKS[1]]}, {"category": "Drinks", "items": []}]
        rec_state = {"awaiting_feedback": True, "satisfied": False, "offer_kind": "category",
                     "last_recommendations": categories, "restaurant_scope": {"restaurant_id": "417218"}}
        state = {"recommendation_brief": reply["recommendation_brief"], "recommendation_state": rec_state,
                 "context_profile": {"address_id": "addr"}, "messages": [HumanMessage(content="1")]}
        with patch.object(ra, "_handle_category_pick", return_value={"messages": [], "recommendation_state": {}}) as handle:
            result = ra.recommendation_agent_node(state)
        handle.assert_called_once()
        self.assertNotEqual((result.get("recommendation_state") or {}).get("offer_kind"), "opening")

    def test_old_briefs_that_still_hold_the_cards_do_not_restart_either(self):
        # A brief saved before this fix (cards still in it) + a dish multi-select on screen.
        rec_state = {"awaiting_feedback": True, "satisfied": False, "offer_kind": "dish",
                     "last_recommendations": [PICKS[1]], "restaurant_scope": {"restaurant_id": "88"}}
        state = {"recommendation_brief": BRIEF, "recommendation_state": rec_state,
                 "context_profile": {"address_id": "addr"}, "messages": [HumanMessage(content="1")]}
        with patch.object(ra, "_handle_dish_multiselect", return_value={"messages": [], "recommendation_state": {}}) as handle:
            ra.recommendation_agent_node(state)
        handle.assert_called_once()


class PickIsLearnedOnceTests(unittest.TestCase):
    """The add-on question re-runs the node on resume; the pick used to be saved each time,
    so one burger became 4 picks and an inferred 'Veg' diet."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"CRAVR_PREFS_PATH": os.path.join(self.tmp.name, "p.json")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_a_pick_that_pauses_for_add_ons_is_saved_once(self):
        from langgraph.errors import GraphInterrupt

        calls = {"n": 0}

        def add_with_one_question(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < 3:           # two re-runs pause on questions...
                raise GraphInterrupt(())
            return {"messages": [], "cart_state": {}}  # ...the last one finishes

        with patch.object(ra, "add_dish_to_cart", side_effect=add_with_one_question):
            for _ in range(3):
                try:
                    ra._reply_with_pick(PICKS[0], "addr", state={})
                except GraphInterrupt:
                    pass
        prefs = store.load_preferences("addr")["preferences"]  # keyed by address_id
        self.assertEqual(prefs["liked_items"], {"herbivores burger": 1})
        self.assertNotIn("dietary_preference", prefs)


if __name__ == "__main__":
    unittest.main()


class EndToEndNoRestartTests(unittest.TestCase):
    """Opening -> pick -> 'Add more from this restaurant' -> pick a section -> dishes,
    through the real graph (Swiggy calls faked)."""

    def test_add_more_goes_to_the_menu_not_back_to_the_start(self):
        import uuid

        from langchain_core.messages import AIMessage

        from cravr.cart import flows
        from cravr.cart.routing import CART_ADD_MORE_VALUE
        from cravr.graph import build_graph

        menu = [{"item_name": "Peri Peri Fries", "menu_item_id": "f1", "restaurant_id": "417218",
                 "restaurant_name": "Louis Burger", "price_inr": 129, "tags": "Sides"},
                {"item_name": "Coke", "menu_item_id": "c1", "restaurant_id": "417218",
                 "restaurant_name": "Louis Burger", "price_inr": 60, "tags": "Beverages"}]

        def fake_add(dish, address_id, **_k):
            return {"messages": [AIMessage(content=f"Added {dish['item_name']}", name="cart_agent")],
                    "cart_state": {"restaurant_id": dish["restaurant_id"], "restaurant": dish["restaurant_name"]}}

        app = build_graph()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        with patch.dict(os.environ, {"CRAVR_DISHES_FIRST": "1"}), \
             patch("cravr.recommend.opening.build_opening_picks", return_value=PICKS), \
             patch("cravr.userPreference.preference.fetch_user_preferences", return_value={}), \
             patch("cravr.userPreference.preference.fetch_order_history", return_value={}), \
             patch("cravr.context.ContextAgent.fetch_user_location", return_value={"status": "success", "address_id": "addr"}), \
             patch("cravr.context.ContextAgent.current_weather", return_value=None), \
             patch.object(ra, "add_dish_to_cart", side_effect=fake_add), \
             patch.object(ra, "record_pick"), \
             patch.object(flows.food_mcp, "get_cart", return_value={"items": [{"menu_item_id": "76582145", "quantity": 1}]}), \
             patch.object(flows.food_mcp, "get_restaurant_menu_items", return_value=menu), \
             patch.object(flows, "_current_restaurant_id", return_value="417218"), \
             patch.object(flows, "_current_restaurant_name", return_value="Louis Burger"):
            first = app.invoke({"messages": []}, cfg)
            self.assertEqual(first["recommendation_state"]["offer_kind"], "opening")
            app.invoke({"messages": [HumanMessage(content="1")]}, cfg)
            more = app.invoke({"messages": [HumanMessage(content=CART_ADD_MORE_VALUE)]}, cfg)
            self.assertEqual(more["recommendation_state"]["offer_kind"], "category")
            dishes = app.invoke({"messages": [HumanMessage(content="1")]}, cfg)
        self.assertEqual(dishes["recommendation_state"]["offer_kind"], "dish")
        self.assertIn("Peri Peri Fries", [d["item_name"] for d in dishes["recommendation_state"]["last_recommendations"]])


# ---------------------------------------------------------------------------
# The logger.txt cart run (29 Sep): Herbivores Burger already in the cart with
# five add-ons and out of stock; the user picks it again with Caramelized Onion.
# ---------------------------------------------------------------------------
from cravr.cart import add_flow, checkout, display, flows  # noqa: E402
from cravr.cart.routing import CART_REMOVE_OOS_VALUE, _classify_cart_action  # noqa: E402
from cravr.food_mcp.cart import parse_cart_data  # noqa: E402

BURGER = {"item_name": "Herbivores Burger", "menu_item_id": "76582145", "restaurant_id": "417218",
          "restaurant_name": "Louis Burger", "price_inr": 235, "tags": "Burgers; has addons"}
OLD_ADDONS = [{"choice_id": "138030778", "group_id": "229800503", "name": "Cheese Sauce"},
              {"choice_id": "138030771", "group_id": "229800497", "name": "English Cheddar Slice"},
              {"choice_id": "152343309", "group_id": "229800501", "name": "Peri Peri Fries Regular"}]
ONION = {"choice_id": "152353833", "group_id": "229800497", "name": "Caramelized Onion"}
CUSTOMIZATION = {"addon_groups": [{"group_id": "229800497", "group_name": "Choose Your Addons", "max_addons": 8,
                                   "choices": [{"id": "152353833", "name": "Caramelized Onion", "price_inr": 39,
                                                "group_id": "229800497"}]}],
                 "variant_groups": []}


def swiggy_cart(items, status_message="CART_UPDATED_SUCCESSFULLY"):
    return parse_cart_data({"statusCode": 0, "statusMessage": status_message, "data": {
        "cart_id": 1, "restaurant": {"name": "Louis Burger"}, "items": items}}, "addr")


def burger_line(quantity=1, addons=None, in_stock=1):
    return {"menu_item_id": 76582145, "name": "Herbivores Burger", "quantity": quantity, "total": 648,
            "in_stock": in_stock, "variants": [], "addons": addons if addons is not None else OLD_ADDONS}


class DuplicateItemTests(unittest.TestCase):
    def _add(self, answers, update_cart=None):
        existing = swiggy_cart([burger_line(in_stock=0)], "Sorry! few items are now out of stock.")
        update_cart = update_cart or swiggy_cart([burger_line(addons=[ONION])])
        answers = list(answers)
        questions = []

        def ask(payload):
            questions.append(payload)
            return answers.pop(0)

        with patch.object(add_flow.food_mcp, "get_cart", return_value=existing) as get_cart, \
             patch.object(add_flow.food_mcp, "update_cart",
                          return_value={"status": "success", "message": "Cart updated.", "cart": update_cart}) as update, \
             patch.object(add_flow, "_lookup_customization", return_value=CUSTOMIZATION), \
             patch.object(add_flow.food_mcp, "get_cart_item_details", side_effect=AssertionError("extra cart call")), \
             patch.object(add_flow, "interrupt", side_effect=ask):
            result = add_flow.add_dish_to_cart(dict(BURGER), "addr", quantity=1, lead_in="Great.")
        return result, questions, update, get_cart

    def test_it_asks_instead_of_silently_doubling(self):
        _result, questions, _update, _get = self._add(["__dup_keep__"])
        q = questions[0]
        self.assertEqual(q["field"], "duplicate_item")
        self.assertIn("Cheese Sauce", q["question"])
        self.assertIn("out of stock", q["question"])
        self.assertEqual([o["value"] for o in q["options"]], ["__dup_replace__", "__dup_one_more__", "__dup_keep__"])

    def test_replace_sends_one_burger_with_only_the_new_add_on(self):
        _result, questions, update, get_cart = self._add(["__dup_replace__", "152353833"])
        self.assertEqual([q["field"] for q in questions], ["duplicate_item", "addon_groups"])
        payload = update.call_args[0][2]
        self.assertEqual(payload, [{"menu_item_id": "76582145", "quantity": 1,
                                    "addons": [{"choice_id": 152353833, "group_id": 229800497, "name": "Caramelized Onion"}]}])
        # One cart read before, none after: the update's own response is the verify + display.
        self.assertEqual(get_cart.call_count, 1)

    def test_one_more_keeps_the_same_add_ons_for_both(self):
        _result, questions, update, _get = self._add(["__dup_one_more__"],
                                                     update_cart=swiggy_cart([burger_line(quantity=2)]))
        self.assertEqual([q["field"] for q in questions], ["duplicate_item"])  # no add-on question
        entry = update.call_args[0][2][0]
        self.assertEqual(entry["quantity"], 2)
        self.assertEqual({a["name"] for a in entry["addons"]}, {a["name"] for a in OLD_ADDONS})

    def test_keep_changes_nothing(self):
        result, _q, update, _get = self._add(["__dup_keep__"])
        update.assert_not_called()
        self.assertIn("left your cart as it is", result["messages"][0].content)


class AddOnVerificationTests(unittest.TestCase):
    def test_a_dropped_add_on_is_reported_not_called_verified(self):
        dropped = swiggy_cart([burger_line(addons=[])])  # quantity fine, onion missing
        with patch.object(add_flow.food_mcp, "get_cart", return_value=swiggy_cart([])), \
             patch.object(add_flow.food_mcp, "update_cart",
                          return_value={"status": "success", "message": "Cart updated.", "cart": dropped}), \
             patch.object(add_flow, "_lookup_customization", return_value=CUSTOMIZATION), \
             patch.object(add_flow, "interrupt", return_value="152353833"):
            result = add_flow.add_dish_to_cart(dict(BURGER), "addr", lead_in="Great.")
        self.assertIn("Swiggy didn't keep Caramelized Onion", result["messages"][0].content)


class OutOfStockTests(unittest.TestCase):
    OOS_CART = swiggy_cart([burger_line(in_stock=0),
                            {"menu_item_id": 5, "name": "Coke", "quantity": 1, "total": 60, "in_stock": 1,
                             "addons": [], "variants": []}], "Sorry! few items are now out of stock.")

    def test_parser_flags_it(self):
        self.assertEqual(self.OOS_CART["out_of_stock"], ["Herbivores Burger"])

    def test_the_cart_says_so_and_offers_removal_first(self):
        with patch.object(display.food_mcp, "get_cart", return_value=self.OOS_CART):
            reply = display._show_cart_with_next_steps("addr")
        content = reply["messages"][0].content
        self.assertIn("out of stock", content)
        self.assertIn("Herbivores Burger can't be ordered right now", content)
        self.assertEqual(reply["cart_state"]["options"][0]["value"], CART_REMOVE_OOS_VALUE)

    def test_checkout_stops_before_payment(self):
        with patch.object(checkout.food_mcp, "get_cart", return_value=self.OOS_CART), \
             patch.object(checkout, "interrupt", side_effect=AssertionError("asked to pay")):
            reply = checkout.run_checkout("addr")
        self.assertIn("won't accept this order yet", reply["messages"][0].content)

    def test_remove_keeps_the_rest_with_their_add_ons(self):
        self.assertEqual(_classify_cart_action(CART_REMOVE_OOS_VALUE)[0], "remove_out_of_stock")
        with patch.object(flows.food_mcp, "get_cart", return_value=self.OOS_CART), \
             patch.object(flows, "_current_restaurant_id", return_value="417218"), \
             patch.object(flows, "_current_restaurant_name", return_value="Louis Burger"), \
             patch.object(flows.food_mcp, "update_cart",
                          return_value={"status": "success", "cart": swiggy_cart([])}) as update:
            reply = flows._remove_out_of_stock({}, "addr")
        self.assertEqual(update.call_args[0][2], [{"menu_item_id": "5", "quantity": 1}])
        self.assertIn("Removed Herbivores Burger", reply["messages"][0].content)

    def test_text_only_carts_are_flagged_too(self):
        text = ("Items (1):\n  - Herbivores Burger — ₹648 (ID: 76582145) ⚠️ OUT OF STOCK\n\n⚠️ 1 item in the cart is "
                "OUT OF STOCK: Herbivores Burger.\n\nTO PAY: ₹776")
        with patch("cravr.food_mcp.call_tool_sync", return_value=text):
            from cravr import food_mcp
            cart = food_mcp.get_cart("addr")
        self.assertEqual(cart["out_of_stock"], ["Herbivores Burger"])


class ResumeDoesNotRepeatLookupsTests(unittest.TestCase):
    """Through a real LangGraph node: the add-on question pauses the flow; resuming
    must not repeat the add-on lookup (~3 s) or the first cart read (~2 s)."""

    def test_one_lookup_and_one_cart_read_across_the_pause(self):
        import uuid

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command

        from cravr.state import CravrState

        def node(state):
            return add_flow.add_dish_to_cart(dict(BURGER), "addr", lead_in="Great.")

        graph = StateGraph(CravrState)
        graph.add_node("add", node)
        graph.add_edge(START, "add")
        graph.add_edge("add", END)
        app = graph.compile(checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        with patch.object(add_flow.food_mcp, "get_cart", return_value=swiggy_cart([])) as get_cart, \
             patch.object(add_flow, "_lookup_customization", return_value=CUSTOMIZATION) as lookup, \
             patch.object(add_flow.food_mcp, "update_cart",
                          return_value={"status": "success", "message": "Cart updated.",
                                        "cart": swiggy_cart([burger_line(addons=[ONION])])}):
            first = app.invoke({"messages": []}, cfg)
            self.assertEqual(first["__interrupt__"][0].value["field"], "addon_groups")
            app.invoke(Command(resume="152353833"), cfg)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(get_cart.call_count, 1)


class DatabaseStorageTests(unittest.TestCase):
    """DATABASE_URL moves taste profiles and group polls off Render's wiped disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{self.tmp.name}/cravr.db"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_backend_is_chosen_from_the_url(self):
        from cravr import storage

        self.assertEqual(storage.backend_name(), "sqlite")
        with patch.dict(os.environ, {"DATABASE_URL": "postgres://u:p@host/db"}):
            self.assertEqual(storage.backend_name(), "postgres")
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            self.assertEqual(storage.backend_name(), "json")

    def test_preferences_and_groups_round_trip_through_the_database(self):
        from cravr.group import store as group_store

        store.save_preferences(None, dietary_preference="Veg", liked_items={"poha": 2})
        store.save_preferences(None, dietary_preference=None)
        self.assertEqual(store.load_preferences(None)["preferences"], {"liked_items": {"poha": 2}})
        store.clear_preferences(None)
        self.assertEqual(store.load_preferences(None)["status"], "no_results")

        group = group_store.create_group(3)
        group_store.add_response(group["id"], {"name": "Riya", "diet": "veg"})
        group_store.add_response(group["id"], {"name": "riya", "diet": "nonveg"})  # same person again
        saved = group_store.get_group(group["id"])
        self.assertEqual([r["diet"] for r in saved["responses"]], ["nonveg"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "groups.json")))

    def test_a_broken_database_never_blocks_ordering(self):
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:////nonexistent-dir/x.db"}):
            self.assertEqual(store.load_preferences(None)["status"], "no_results")


class ProfileKeyIsAddressIdTests(unittest.TestCase):
    """The whole flow runs on Swiggy's address_id, so taste profiles are keyed by it."""

    ADDR = "d3n3lphv42g4rgmuvu70__AbKqwgSYhkE5YttayIYiXs"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"CRAVR_PREFS_PATH": os.path.join(self.tmp.name, "p.json"),
                                           "DATABASE_URL": ""})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_owner_comes_from_the_address(self):
        self.assertEqual(store.profile_owner({"context_profile": {"address_id": self.ADDR}, "user_id": "USR-1"}), self.ADDR)
        self.assertEqual(store.profile_owner({"user_id": "USR-1"}), "USR-1")

    def test_a_pick_is_saved_under_the_address_id(self):
        with patch.object(ra, "add_dish_to_cart", return_value={"messages": [], "cart_state": {}}):
            ra._reply_with_pick(PICKS[0], self.ADDR, state={"user_id": None})
        self.assertEqual(store.load_preferences(self.ADDR)["preferences"]["liked_items"], {"herbivores burger": 1})
        self.assertEqual(store.load_preferences(None)["status"], "no_results")  # nothing under "default"

    def test_real_swiggy_loads_the_profile_for_the_resolved_address(self):
        from cravr.userPreference import preference

        store.save_preferences(self.ADDR, dietary_preference="Non-Veg")
        with patch.object(preference, "using_real_food_mcp", return_value=True), \
             patch.object(preference.food_mcp, "get_saved_address", return_value={"status": "success", "address_id": self.ADDR}), \
             patch.object(preference, "fetch_order_history", return_value={}):
            update = preference.user_preference_node({"messages": []})
        self.assertEqual(update["preference_profile"]["preferences"]["dietary_preference"], "Non-Veg")
