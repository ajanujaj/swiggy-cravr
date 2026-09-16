import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from cravr import food_mcp
from cravr.cart import CartAgent
from cravr.cart.CartAgent import _classify_cart_action, cart_agent_node, is_cart_request
from cravr.graph import CART, FEEDBACK, NEW_REQUEST, _route_entry


class CartRoutingTests(unittest.TestCase):
    """is_cart_request gates graph.py's entry router -- must catch cart/
    coupon phrasing without hijacking ordinary recommendation requests."""

    def test_catches_add_view_clear_and_coupon_phrasing(self):
        for text in (
            "add it to cart", "add this to my cart", "order it",
            "what's in my cart", "show my cart", "clear my cart",
            "empty my cart", "apply coupon SAVE20", "apply SAVE20",
            "use FREESHIP", "any discount codes?", "what coupons do you have",
        ):
            self.assertTrue(is_cart_request(text), text)

    def test_leaves_ordinary_requests_and_feedback_alone(self):
        for text in (
            "I want something spicy", "order me a pizza", "show me Chinese food",
            "yes", "2", "make it non-veg", "SAVE20 please",
        ):
            self.assertFalse(is_cart_request(text), text)

    def test_route_entry_prefers_cart_over_feedback_and_new_request(self):
        # Even mid-feedback-loop (awaiting_feedback=True), a cart message wins.
        state = {
            "messages": [HumanMessage(content="add it to my cart")],
            "recommendation_state": {
                "last_recommendations": [{"item_name": "Dosa"}],
                "awaiting_feedback": True, "satisfied": False,
            },
        }
        self.assertEqual(_route_entry(state), CART)

    def test_route_entry_still_falls_back_to_feedback_and_new_request(self):
        feedback_state = {
            "messages": [HumanMessage(content="2")],
            "recommendation_state": {
                "last_recommendations": [{"item_name": "A"}, {"item_name": "B"}],
                "awaiting_feedback": True, "satisfied": False,
            },
        }
        self.assertEqual(_route_entry(feedback_state), FEEDBACK)
        new_state = {"messages": [HumanMessage(content="I want biryani")], "recommendation_state": {}}
        self.assertEqual(_route_entry(new_state), NEW_REQUEST)


class ClassifyActionTests(unittest.TestCase):
    def test_add_defaults_to_quantity_one(self):
        self.assertEqual(_classify_cart_action("add it to my cart"), ("add", 1))

    def test_add_picks_up_an_explicit_quantity(self):
        self.assertEqual(_classify_cart_action("add 3 to my cart"), ("add", 3))

    def test_apply_coupon_extracts_the_code(self):
        self.assertEqual(_classify_cart_action("apply coupon SAVE20"), ("apply_coupon", "SAVE20"))
        self.assertEqual(_classify_cart_action("use FREESHIP"), ("apply_coupon", "FREESHIP"))

    def test_clear_wins_over_view(self):
        self.assertEqual(_classify_cart_action("clear my cart"), ("clear", None))

    def test_bare_cart_word_defaults_to_view(self):
        self.assertEqual(_classify_cart_action("my cart"), ("view", None))


def _state(text, *, address_id="addr1", chosen=None, cart_state=None, recommendation_state=None):
    return {
        "messages": [HumanMessage(content=text)],
        "context_profile": {"address_id": address_id} if address_id else {},
        "recommendation_state": recommendation_state or {"last_recommendations": chosen or []},
        "cart_state": cart_state or {},
    }


class CartAgentNodeTests(unittest.TestCase):
    """cart_agent_node against a mocked food_mcp -- the MCP-call seam,
    mirroring how test_recommend_agent.py mocks _call_recommendation_tool
    rather than the raw MCP client."""

    DISH = {"item_name": "Masala Dosa", "menu_item_id": "ITEM0011", "restaurant_id": "REST003",
            "restaurant_name": "Dakshin Delights"}

    def test_no_address_yet_asks_for_one_instead_of_calling_mcp(self):
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get:
            result = cart_agent_node(_state("show my cart", address_id=None))
        mock_get.assert_not_called()
        self.assertIn("address", result["messages"][0].content.lower())

    def test_add_with_no_dish_picked_asks_the_user_to_choose(self):
        with patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update:
            result = cart_agent_node(_state("add it to my cart", chosen=[]))
        mock_update.assert_not_called()
        self.assertIn("Which dish", result["messages"][0].content)

    def test_add_with_a_dish_missing_ids_fails_gracefully(self):
        """A dish sourced from a path that never got the menu_item_id/
        restaurant_id fix (or the mock recommendation tool before this
        change) must not crash update_cart with None ids."""
        with patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update:
            result = cart_agent_node(_state("add it to my cart", chosen=[{"item_name": "Mystery Dish"}]))
        mock_update.assert_not_called()
        self.assertIn("enough details", result["messages"][0].content)

    def test_add_calls_update_cart_then_shows_the_cart(self):
        cart_after = {
            "status": "success", "items": [{"item_name": "Masala Dosa", "quantity": 1, "total": 180}],
            "pricing": {"to_pay": 210}, "coupon_applied": None, "coupon_discount": 0,
        }
        with patch("cravr.cart.CartAgent.food_mcp.update_cart", return_value={"status": "success"}) as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value=cart_after):
            result = cart_agent_node(_state("add it to my cart", chosen=[self.DISH]))
        mock_update.assert_called_once_with(
            "REST003", "addr1", [{"menu_item_id": "ITEM0011", "quantity": 1}], restaurant_name="Dakshin Delights",
        )
        self.assertIn("Masala Dosa", result["messages"][0].content)
        self.assertIn("Total to pay: ₹210", result["messages"][0].content)

    def test_add_relays_update_carts_own_message_when_it_has_signal(self):
        """A real update_cart message that isn't just "cart is empty" noise
        (e.g. a genuine confirmation) is shown as a lead-in to the cart."""
        cart_after = {"status": "empty", "items": [], "pricing": {}, "message": "Cart is empty."}
        with patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Item added, size required."}), \
             patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value=cart_after):
            result = cart_agent_node(_state("add it to my cart", chosen=[self.DISH]))
        self.assertIn("Item added, size required.", result["messages"][0].content)

    def test_add_honours_an_explicit_quantity(self):
        with patch("cravr.cart.CartAgent.food_mcp.update_cart", return_value={"status": "success"}) as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value={"status": "success", "items": []}):
            cart_agent_node(_state("add 2 to my cart", chosen=[self.DISH]))
        mock_update.assert_called_once_with(
            "REST003", "addr1", [{"menu_item_id": "ITEM0011", "quantity": 2}], restaurant_name="Dakshin Delights",
        )

    def test_clear_reports_success_without_a_get_cart_round_trip(self):
        with patch("cravr.cart.CartAgent.food_mcp.flush_cart", return_value={"status": "success"}), \
             patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get:
            result = cart_agent_node(_state("clear my cart"))
        mock_get.assert_not_called()
        self.assertIn("empty", result["messages"][0].content.lower())

    def test_coupons_without_a_restaurant_in_context_asks_to_add_something_first(self):
        with patch("cravr.cart.CartAgent.food_mcp.fetch_coupons") as mock_fetch:
            result = cart_agent_node(_state("what coupons are available", chosen=[]))
        mock_fetch.assert_not_called()
        self.assertIn("Add something", result["messages"][0].content)

    def test_coupons_uses_the_restaurant_already_in_the_cart(self):
        cart_state = {"restaurant": {"id": "REST003", "name": "Dakshin Delights"}}
        coupons = {"status": "success", "coupons": [{"title": "SAVE20", "subtitle": "20% off"}]}
        with patch("cravr.cart.CartAgent.food_mcp.fetch_coupons", return_value=coupons) as mock_fetch:
            result = cart_agent_node(_state("any coupons?", cart_state=cart_state))
        mock_fetch.assert_called_once_with("REST003", "addr1")
        self.assertIn("SAVE20", result["messages"][0].content)

    def test_apply_coupon_surfaces_a_rejection_without_crashing(self):
        with patch("cravr.cart.CartAgent.food_mcp.apply_coupon", return_value={"status": "error", "message": "not valid"}):
            result = cart_agent_node(_state("apply BOGUS"))
        self.assertIn("not valid", result["messages"][0].content)

    def test_get_cart_error_is_surfaced_not_raised(self):
        with patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value={"status": "error", "message": "timeout"}):
            result = cart_agent_node(_state("show my cart"))
        self.assertIn("timeout", result["messages"][0].content)


class FoodMcpCartFunctionTests(unittest.TestCase):
    """food_mcp.py's cart wrappers against the {"success","data"} envelope
    the docs describe -- and the {"success": false} error shape."""

    def test_get_cart_parses_items_pricing_and_offers(self):
        raw = {
            "success": True, "addressId": "addr1",
            "data": {
                "cart_id": "C1", "restaurant": {"id": "R1", "name": "X"},
                "items": [{"menu_item_id": "I1", "name": "Dosa", "quantity": 2, "total": 360, "final_price": 180}],
                "item_count": 1,
                "pricing": {"item_total": 360, "delivery_charge": 30, "taxes_and_charges": 18, "to_pay": 408},
                "offers": {"coupon_applied": "SAVE20", "coupon_discount": 50, "free_delivery_applied": False},
            },
        }
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw):
            result = food_mcp.get_cart("addr1")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["items"][0]["menu_item_id"], "I1")
        self.assertEqual(result["coupon_applied"], "SAVE20")
        self.assertEqual(result["pricing"]["to_pay"], 408)

    def test_get_cart_treats_zero_discount_coupon_as_not_applied(self):
        """Docs: coupon_applied with coupon_discount==0 means "suggested",
        not actually active -- must not be reported as applied."""
        raw = {"success": True, "data": {"items": [], "offers": {"coupon_applied": "SAVE20", "coupon_discount": 0}}}
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw):
            result = food_mcp.get_cart("addr1")
        self.assertIsNone(result["coupon_applied"])

    def test_get_cart_surfaces_a_success_false_error(self):
        raw = {"success": False, "error": {"message": "invalid address"}}
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw):
            result = food_mcp.get_cart("bad-addr")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "invalid address")

    def test_get_cart_survives_an_mcp_exception(self):
        with patch("cravr.food_mcp.call_tool_sync", side_effect=RuntimeError("boom")):
            result = food_mcp.get_cart("addr1")
        self.assertEqual(result["status"], "error")

    def test_update_cart_sends_the_documented_argument_names(self):
        with patch("cravr.food_mcp.call_tool_sync", return_value={"success": True, "data": {}}) as mock_call:
            food_mcp.update_cart("R1", "addr1", [{"menu_item_id": "I1", "quantity": 2}], restaurant_name="X")
        mock_call.assert_called_once_with(
            "update_food_cart",
            {"restaurantId": "R1", "addressId": "addr1", "cartItems": [{"menu_item_id": "I1", "quantity": 2}], "restaurantName": "X"},
        )

    def test_apply_coupon_sends_the_documented_argument_names(self):
        with patch("cravr.food_mcp.call_tool_sync", return_value={"success": True, "data": {}}) as mock_call:
            food_mcp.apply_coupon("SAVE20", "addr1")
        mock_call.assert_called_once_with("apply_food_coupon", {"couponCode": "SAVE20", "addressId": "addr1"})

    def test_fetch_coupons_flattens_sections_into_one_list(self):
        raw = {
            "success": True,
            "data": {"coupon_sections": [
                {"coupons": [{"id": "A", "applicabilityStatus": "APPLICABLE"}]},
                {"coupons": [{"id": "B", "applicabilityStatus": "NOT_APPLICABLE"}]},
            ]},
        }
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw):
            result = food_mcp.fetch_coupons("R1", "addr1")
        self.assertEqual(result["count"], 2)
        self.assertEqual({c["id"] for c in result["coupons"]}, {"A", "B"})

    def test_fetch_coupons_no_results_when_empty(self):
        raw = {"success": True, "data": {"coupon_sections": [], "status_message": "none today"}}
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw):
            result = food_mcp.fetch_coupons("R1", "addr1")
        self.assertEqual(result["status"], "no_results")

    def test_flush_cart_reports_success(self):
        with patch("cravr.food_mcp.call_tool_sync", return_value={"success": True, "data": {}}):
            result = food_mcp.flush_cart()
        self.assertEqual(result["status"], "success")


class RealTextResponseRegressionTests(unittest.TestCase):
    """Regression coverage for the exact plain-text payloads Swiggy returned
    in a live run (CRAVR_DEBUG=1 log, Sep 2026): a non-empty cart's raw text
    has THREE blank-line-separated paragraphs (items, pricing, disclaimer),
    not two -- a naive "split on first blank line" silently threw away the
    whole pricing breakdown (Bug B), and add_dish_to_cart relayed
    update_cart's own full item+pricing dump as a lead-in right before
    get_cart printed the identical dump again (Bug A, the user's reported
    "duplicate .. i guess.")."""

    REAL_UPDATE_CART_RAW = (
        "Cart updated.\n"
        "Restaurant: Timpano - The Pasta Shop\n"
        "Items (1):\n"
        "  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)\n\n"
        "Item total: ₹400\n"
        "Delivery: ₹80\n"
        "Taxes & charges: ₹57.98\n"
        "Coupon (FLAT140): -₹140\n"
        "TO PAY: ₹398\n\n"
        "Cart widget is displayed. Wait for the user to review and confirm."
    )

    REAL_GET_CART_RAW = (
        "Items (1):\n"
        "  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)\n\n"
        "Item total: ₹400\n"
        "Delivery: ₹80\n"
        "Taxes & charges: ₹57.98\n"
        "Coupon (FLAT140): -₹140\n"
        "TO PAY: ₹398\n\n"
        "Cart widget is displayed — do not repeat the cart items, prices, or totals in text.\n"
        "NOTE: The cart widget has quantity stepper controls. The user can adjust quantities "
        "and click \"Confirm and Review Cart\" — the widget will send you the changes to "
        "process via update_food_cart.\n"
        "Some items have add-on options available (check valid_addons). Briefly mention this "
        "to the user, e.g. \"Your Veg Burger has add-on options (Fries, Cheese) — want to add any?\"\n\n"
        "⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything "
        "the widget already displays — instead, provide a brief recommendation or ask what "
        "they'd like to do next."
    )

    def test_update_cart_text_mode_keeps_the_full_pricing_breakdown(self):
        with patch("cravr.food_mcp.call_tool_sync", return_value=self.REAL_UPDATE_CART_RAW):
            result = food_mcp.update_cart("351682", "addr1", [{"menu_item_id": "179082816", "quantity": 1}])
        self.assertEqual(result["status"], "success")
        for expected in ("Item total: ₹400", "Delivery: ₹80", "Taxes & charges: ₹57.98",
                          "Coupon (FLAT140): -₹140", "TO PAY: ₹398"):
            self.assertIn(expected, result["message"])
        self.assertNotIn("Cart widget is displayed", result["message"])

    def test_get_cart_text_mode_keeps_the_full_pricing_breakdown(self):
        with patch("cravr.food_mcp.call_tool_sync", return_value=self.REAL_GET_CART_RAW):
            result = food_mcp.get_cart("addr1")
        self.assertEqual(result["status"], "text")
        for expected in ("Item total: ₹400", "Delivery: ₹80", "Taxes & charges: ₹57.98",
                          "Coupon (FLAT140): -₹140", "TO PAY: ₹398"):
            self.assertIn(expected, result["message"])
        self.assertNotIn("Cart widget is displayed", result["message"])
        self.assertNotIn("NOTE:", result["message"])
        self.assertNotIn("⚠️", result["message"])

    def test_add_dish_to_cart_does_not_duplicate_the_real_pizza_response(self):
        """End-to-end replay of the exact log that triggered the user's
        "duplicate .. i guess." report: the item line must appear once, and
        the pricing must survive intact."""
        dish = {"item_name": "Italian Meat Lover Pizza", "menu_item_id": "179082816",
                "restaurant_id": "351682", "restaurant_name": "Timpano - The Pasta Shop"}

        def fake_call_tool_sync(tool_name, arguments):
            if tool_name == "update_food_cart":
                return self.REAL_UPDATE_CART_RAW
            if tool_name == "get_food_cart":
                return self.REAL_GET_CART_RAW
            raise AssertionError(f"unexpected tool {tool_name}")

        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_call_tool_sync):
            result = CartAgent.add_dish_to_cart(
                dish, "d3n3lphv42g4rgmuvu70__AbKqwgSYhkE5YttayIYiXs", quantity=1,
                lead_in="Great — Italian Meat Lover Pizza from Timpano - The Pasta Shop (₹400.0) it is.",
            )

        content = result["messages"][0].content
        # The lead_in naturally names the dish once ("... it is."), and the
        # cart body names it again in its item line -- that's two DIFFERENT
        # sentences, not a duplicate. The actual bug was the cart body
        # itself (the "Items (1): ..." block) appearing twice.
        self.assertEqual(content.count("Items (1):"), 1, content)
        self.assertEqual(content.count("TO PAY: ₹398"), 1, content)
        self.assertIn("Coupon (FLAT140): -₹140", content)

    def test_add_dish_to_cart_accumulates_a_second_item_from_the_same_restaurant(self):
        """The user's actual follow-up report: picking a second dish from
        the SAME restaurant ("add more") wasn't landing in the cart. Root
        cause found in a real log: update_food_cart is documented as
        additive, but two consecutive real calls (one new item each) both
        came back with "Items (1): ..." and an unchanged item total -- the
        first item was silently dropped, not accumulated, and there's no
        cartId param to target an existing cart. Fix: add_dish_to_cart now
        fetches the current cart first and always sends the FULL merged
        item list, so it no longer depends on the server merging for us.
        This replays that exact real single-item response text as "what's
        already in the cart" and checks the second call's cartItems
        includes both the old and the new item."""
        first_item_cart_text = (
            "Items (1):\n  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)\n\n"
            "Item total: ₹400\nDelivery: ₹80\nTaxes & charges: ₹57.98\n"
            "Coupon (FLAT140): -₹140\nTO PAY: ₹398\n\n"
            "Cart widget is displayed — do not repeat the cart items, prices, or totals in text."
        )
        second_dish = {"item_name": "All Fungi Pizza", "menu_item_id": "179082813",
                        "restaurant_id": "351682", "restaurant_name": "Timpano - The Pasta Shop"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "text", "message": first_item_cart_text}), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}) as mock_update:
            CartAgent.add_dish_to_cart(second_dish, "addr1", quantity=1, lead_in="")

        sent_cart_items = mock_update.call_args.args[2]
        sent_ids = {entry["menu_item_id"] for entry in sent_cart_items}
        self.assertEqual(sent_ids, {"179082816", "179082813"}, sent_cart_items)

    def test_add_dish_to_cart_bumps_quantity_when_the_same_item_is_added_again(self):
        existing_cart_text = (
            "Items (1):\n  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)\n\n"
            "TO PAY: ₹398"
        )
        dish = {"item_name": "Italian Meat Lover Pizza", "menu_item_id": "179082816",
                "restaurant_id": "351682", "restaurant_name": "Timpano - The Pasta Shop"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "text", "message": existing_cart_text}), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}) as mock_update:
            CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="")

        sent_cart_items = mock_update.call_args.args[2]
        self.assertEqual(sent_cart_items, [{"menu_item_id": "179082816", "quantity": 2}])


    def test_add_dish_to_cart_refuses_items_that_need_addon_or_variant_choices(self):
        """Real-world regression: THREE separate live occurrences (Jamie
        Oliver Kitchen's "Mediterranean Veg Pizza [11 inches]" via a normal
        search+pick, plus Mister Edamame's "Manchow Soup" and "Sichuan Hot
        & Sour Soup" via "add more") all merged a has-addons/has-variants
        item into a non-empty cart and got back
        'Cart updated.\nCart is empty.' from the real update_food_cart --
        the whole update silently rejected, with Cravr then falsely
        telling the user it worked. Cravr can't collect the required
        choice, so it must refuse before ever calling update_cart/get_cart
        -- this covers the normal pick path, not just "add more"."""
        dish = {"item_name": "Mediterranean Veg Pizza [11 inches]", "menu_item_id": "136714282",
                "restaurant_id": "855114", "restaurant_name": "Jamie Oliver Kitchen",
                "tags": "has addons"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update:
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="")

        mock_get_cart.assert_not_called()
        mock_update.assert_not_called()
        content = result["messages"][0].content
        self.assertIn("Mediterranean Veg Pizza [11 inches]", content)
        self.assertIn("add-on", content)

    def test_add_dish_to_cart_refuses_items_with_variant_choices_too(self):
        dish = {"item_name": "Fiery Avocado Roll", "menu_item_id": "157423529",
                "restaurant_id": "996967", "restaurant_name": "Mister Edamame",
                "tags": "Veg, has variants"}

        with patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update:
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="")

        mock_update.assert_not_called()
        self.assertIn("Fiery Avocado Roll", result["messages"][0].content)


class CartNextStepTests(unittest.TestCase):
    """The post-add "what next" buttons: add more from this restaurant /
    proceed (placeholder) / browse other restaurants -- offered any time a
    non-empty cart is shown (Anuj's answer), and the sentinel values they
    send must round-trip through is_cart_request/_classify_cart_action
    without colliding with a normal pick reply."""

    NON_EMPTY_CART = {"status": "text", "message": "Items (1):\n  - Bowl — ₹200 (ID: 111)\n\nTO PAY: ₹250"}

    def _state_with_brief(self, text, **kwargs):
        state = _state(text, **kwargs)
        state["recommendation_brief"] = {"filters": {"tags": "high-protein", "craving": "Protein bowl", "course": "main"}}
        return state

    def test_sentinels_route_to_cart_and_classify_distinctly(self):
        for value, expected_action in (
            (CartAgent.CART_ADD_MORE_VALUE, "suggest_more"),
            (CartAgent.CART_PROCEED_VALUE, "proceed_placeholder"),
            (CartAgent.CART_START_OVER_VALUE, "start_over"),
        ):
            self.assertTrue(is_cart_request(value), value)
            action, _ = _classify_cart_action(value)
            self.assertEqual(action, expected_action)

    def test_a_normal_pick_reply_is_not_swallowed_by_cart_routing(self):
        """A bare digit or dish name must keep going to recommendation_agent's
        feedback branch -- the sentinels are exact-match only."""
        for text in ("2", "9", "the pizza", "yes"):
            self.assertFalse(is_cart_request(text), text)

    def test_viewing_a_non_empty_cart_offers_the_three_options(self):
        with patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value=self.NON_EMPTY_CART):
            result = cart_agent_node(_state("show my cart", cart_state=self.NON_EMPTY_CART))
        self.assertEqual(result["cart_state"].get("options"), CartAgent._NEXT_STEP_OPTIONS)

    def test_an_empty_cart_does_not_offer_the_three_options(self):
        with patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value={"status": "empty", "message": "Cart is empty."}):
            result = cart_agent_node(_state("show my cart"))
        self.assertNotIn("options", result["cart_state"])

    def test_proceed_is_a_stub_that_still_shows_the_cart_and_buttons(self):
        with patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value=self.NON_EMPTY_CART):
            result = cart_agent_node(_state(CartAgent.CART_PROCEED_VALUE, cart_state=self.NON_EMPTY_CART))
        self.assertIn("isn't wired up yet", result["messages"][0].content)
        self.assertEqual(result["cart_state"].get("options"), CartAgent._NEXT_STEP_OPTIONS)

    def test_add_more_suggests_complementary_items_and_excludes_the_cart_item(self):
        full_menu = (
            "## Mains\n- Bowl — ₹200 | Veg (ID: 111)\n- Big Bowl — ₹250 | Veg (ID: 222)\n"
            "## Beverages\n- Lemonade — ₹80 | Veg (ID: 333)\n"
            "## Desserts\n- Brownie — ₹90 | Veg (ID: 444)\n"
        )

        def fake_call_tool_sync(tool_name, arguments):
            if tool_name == "get_food_cart":
                return "Items (1):\n  - Bowl — ₹200 (ID: 111)\n\nTO PAY: ₹250\n\nCart widget is displayed."
            if tool_name == "get_restaurant_menu":
                return full_menu
            raise AssertionError(tool_name)

        chosen = [{"item_name": "Bowl", "menu_item_id": "111", "restaurant_id": "R1", "restaurant_name": "Healthy Eats"}]
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_call_tool_sync):
            result = cart_agent_node(self._state_with_brief(
                CartAgent.CART_ADD_MORE_VALUE, chosen=chosen,
                cart_state={"status": "text", "message": "Items (1):\n  - Bowl — ₹200 (ID: 111)"},
            ))

        self.assertEqual(result["messages"][0].name, "recommendation_agent")
        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertNotIn("Bowl", names, "the item already in the cart must never be re-suggested")
        self.assertIn("Lemonade", names[:2])
        self.assertIn("Brownie", names[:2])

    def test_add_more_excludes_items_requiring_customization_even_as_fallback(self):
        """Real-world regression (Mister Edamame, live CRAVR_DEBUG log):
        suggesting "Manchow Soup" / "Sichuan Hot & Sour Soup" -- both
        tagged "has addons" -- and picking one made the real
        update_food_cart respond 'Cart updated.\nCart is empty.', silently
        rejecting the WHOLE merged cart update. Only one addon-free main
        (besides the cart item) is on this menu, so without the hard
        filter the old "top up with whatever's left" fallback would have
        pulled in a has-addons item to fill the limit=5 quota."""
        full_menu = (
            "## Gourmet Soups\n"
            "- Manchow Soup — ₹399 | Veg, has addons (ID: 156291316)\n"
            "- Sichuan Hot & Sour Soup — ₹399 | Veg, has addons (ID: 156291303)\n"
            "## Recommended\n"
            "- Burmese Khao Suey — ₹499 | Veg, has addons (ID: 182049278)\n"
            "- Fiery Avocado Roll — ₹499 | Veg, has variants (ID: 157423529)\n"
            "- Stuffed Mushroom In Thai Sauce — ₹509 | Veg, Bestseller (ID: 157253913)\n"
            "## Mains\n"
            "- Veg Stir Fried Udon Noodles — ₹409 | Veg (ID: 157423586)\n"
        )

        def fake_call_tool_sync(tool_name, arguments):
            if tool_name == "get_food_cart":
                return ("Items (1):\n  - Veg Stir Fried Udon Noodles — ₹284 (ID: 157423586)"
                        "\n\nTO PAY: ₹426\n\nCart widget is displayed.")
            if tool_name == "get_restaurant_menu":
                return full_menu
            raise AssertionError(tool_name)

        chosen = [{"item_name": "Veg Stir Fried Udon Noodles", "menu_item_id": "157423586",
                   "restaurant_id": "996967", "restaurant_name": "Mister Edamame"}]
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_call_tool_sync):
            result = cart_agent_node(self._state_with_brief(
                CartAgent.CART_ADD_MORE_VALUE, chosen=chosen,
                cart_state={"status": "text",
                            "message": "Items (1):\n  - Veg Stir Fried Udon Noodles — ₹284 (ID: 157423586)"},
            ))

        self.assertEqual(result["messages"][0].name, "recommendation_agent")
        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        for addon_item in ("Manchow Soup", "Sichuan Hot & Sour Soup", "Burmese Khao Suey", "Fiery Avocado Roll"):
            self.assertNotIn(addon_item, names, f"{addon_item!r} needs a variant/add-on choice Cravr can't supply")
        self.assertEqual(names, ["Stuffed Mushroom In Thai Sauce"])

    def test_add_more_reports_nothing_left_when_every_remaining_item_needs_customization(self):
        full_menu = (
            "## Gourmet Soups\n"
            "- Manchow Soup — ₹399 | Veg, has addons (ID: 156291316)\n"
            "## Mains\n"
            "- Veg Stir Fried Udon Noodles — ₹409 | Veg (ID: 157423586)\n"
        )

        def fake_call_tool_sync(tool_name, arguments):
            if tool_name == "get_food_cart":
                return ("Items (1):\n  - Veg Stir Fried Udon Noodles — ₹284 (ID: 157423586)"
                        "\n\nTO PAY: ₹426\n\nCart widget is displayed.")
            if tool_name == "get_restaurant_menu":
                return full_menu
            raise AssertionError(tool_name)

        chosen = [{"item_name": "Veg Stir Fried Udon Noodles", "menu_item_id": "157423586",
                   "restaurant_id": "996967", "restaurant_name": "Mister Edamame"}]
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_call_tool_sync):
            result = cart_agent_node(self._state_with_brief(
                CartAgent.CART_ADD_MORE_VALUE, chosen=chosen,
                cart_state={"status": "text",
                            "message": "Items (1):\n  - Veg Stir Fried Udon Noodles — ₹284 (ID: 157423586)"},
            ))

        self.assertNotEqual(result["messages"][0].name, "recommendation_agent")
        self.assertIn("everything worth adding", result["messages"][0].content)

    def test_add_more_with_no_restaurant_in_context_fails_gracefully(self):
        with patch("cravr.food_mcp.get_cart") as mock_get_cart:
            result = cart_agent_node(_state(CartAgent.CART_ADD_MORE_VALUE, chosen=[]))
        mock_get_cart.assert_not_called()
        self.assertIn("lost track", result["messages"][0].content)

    def test_start_over_flushes_the_cart_and_excludes_that_restaurant_from_the_new_search(self):
        raw_results = [
            {"item_name": "Protein Bowl v2", "menu_item_id": "555", "restaurant_id": "R1",
             "restaurant_name": "Healthy Eats", "price_inr": 300, "course": "main", "tags": "high-protein"},
            {"item_name": "Grain Bowl", "menu_item_id": "666", "restaurant_id": "R2",
             "restaurant_name": "Green Table", "price_inr": 320, "course": "main", "tags": "high-protein"},
        ]

        def fake_flush(tool_name, arguments):
            if tool_name == "flush_food_cart":
                return "Cart cleared.\n\nCart widget is displayed."
            raise AssertionError(tool_name)

        chosen = [{"item_name": "Bowl", "menu_item_id": "111", "restaurant_id": "R1", "restaurant_name": "Healthy Eats"}]
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_flush), \
             patch("cravr.recommend.RecommendAgent._call_recommendation_tool", return_value=raw_results):
            result = cart_agent_node(self._state_with_brief(CartAgent.CART_START_OVER_VALUE, chosen=chosen))

        self.assertEqual(result["messages"][0].name, "recommendation_agent")
        names = [d["item_name"] for d in result["recommendation_state"]["last_recommendations"]]
        self.assertNotIn("Protein Bowl v2", names, "must exclude the restaurant just cleared")
        self.assertIn("Grain Bowl", names)

    def test_picking_a_suggested_item_lands_in_the_cart_via_the_normal_pick_machinery(self):
        """End-to-end: "add more" hands off to recommendation_agent's own
        pick-resolution, so a follow-up "2" must resolve and add to cart --
        no separate pick-tracking code to maintain."""
        from cravr.recommend.RecommendAgent import recommendation_agent_node

        suggestions_state = {
            "last_recommendations": [
                {"item_name": "Lemonade", "menu_item_id": "333", "restaurant_id": "R1", "restaurant_name": "Healthy Eats"},
                {"item_name": "Brownie", "menu_item_id": "444", "restaurant_id": "R1", "restaurant_name": "Healthy Eats"},
            ],
            "alternates": [], "filters": {}, "awaiting_feedback": True, "satisfied": False,
        }

        def fake_call_tool_sync(tool_name, arguments):
            if tool_name == "get_food_cart":
                return ("Items (2):\n  - Bowl — ₹200 (ID: 111)\n  - Brownie — ₹90 (ID: 444)\n\n"
                        "TO PAY: ₹340\n\nCart widget is displayed — do not repeat.")
            if tool_name == "update_food_cart":
                return ("Cart updated.\nRestaurant: Healthy Eats\nItems (2):\n"
                        "  - Bowl — ₹200 (ID: 111)\n  - Brownie — ₹90 (ID: 444)\n\nTO PAY: ₹340\n\n"
                        "Cart widget is displayed. Wait for the user to review and confirm.")
            raise AssertionError(tool_name)

        state = _state("2", address_id="addr1", recommendation_state=suggestions_state)
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_call_tool_sync):
            result = recommendation_agent_node(state)

        content = result["messages"][0].content
        self.assertIn("Brownie", content)
        self.assertEqual(content.count("Items (2):"), 1)
        self.assertIn("TO PAY: ₹340", content)
        self.assertEqual(result["cart_state"].get("options"), CartAgent._NEXT_STEP_OPTIONS)


if __name__ == "__main__":
    unittest.main()
