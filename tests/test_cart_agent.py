import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from cravr import food_mcp
from cravr.cart import CartAgent
from cravr.cart.CartAgent import _classify_cart_action, cart_agent_node, is_cart_request
from cravr.graph import CART, FEEDBACK, NEW_REQUEST, _route_entry

import uuid

from langgraph.types import Command

from cravr.graph import app as cravr_app


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


    def test_add_dish_to_cart_falls_back_to_refusal_when_the_addon_only_attempt_does_not_land(self):
        """Real-world regression: THREE separate live occurrences (Jamie
        Oliver Kitchen's "Mediterranean Veg Pizza [11 inches]" via a normal
        search+pick, plus Mister Edamame's "Manchow Soup" and "Sichuan Hot
        & Sour Soup" via "add more") all merged a has-addons/has-variants
        item into a non-empty cart and got back
        'Cart updated.\nCart is empty.' from the real update_food_cart --
        the whole update silently rejected, with Cravr then falsely
        telling the user it worked.

        Cravr now attempts addon-only items (no "has variants" tag) with an
        explicit empty addon selection instead of refusing outright (see
        CartAgent's attempt_addon_only comment) -- but it must still verify
        the item actually landed before trusting that, and fall back to
        this same honest refusal if it didn't. This fixture simulates that
        exact real 'Cart updated.\nCart is empty.' outcome to prove the
        fallback still holds even though update_cart itself now gets called."""
        dish = {"item_name": "Mediterranean Veg Pizza [11 inches]", "menu_item_id": "136714282",
                "restaurant_id": "855114", "restaurant_name": "Jamie Oliver Kitchen",
                "tags": "has addons"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items") as mock_probe:
            mock_get_cart.side_effect = [
                {"status": "empty", "message": "Cart is empty."},  # cart fetched before the merge
                {"status": "empty", "message": "Cart is empty."},  # verify after the attempt -- item never landed
            ]
            mock_update.return_value = {"status": "success", "message": "Cart updated.\nCart is empty."}
            # A non-empty lead_in mirrors RecommendAgent._reply_with_pick's
            # "Great -- X it is." -- it must be dropped, not prefixed, or
            # the reply contradicts itself (confirmed live: see CartAgent).
            result = CartAgent.add_dish_to_cart(
                dish, "addr1", quantity=1,
                lead_in="Great — Mediterranean Veg Pizza [11 inches] from Jamie Oliver Kitchen (₹699.0) it is.",
            )

        self.assertEqual(mock_get_cart.call_count, 2)
        mock_update.assert_called_once()
        sent_items = mock_update.call_args[0][2]
        target_entry = next(e for e in sent_items if e["menu_item_id"] == "136714282")
        self.assertEqual(target_entry.get("addons"), [], "addon-only attempt must send an explicit empty selection")
        # Diagnostic probe -- see CartAgent's comment. Must target the
        # SPECIFIC dish+restaurant the docs point at, not the broad
        # unscoped query CartAgent otherwise never makes here.
        mock_probe.assert_called_once_with(
            "Mediterranean Veg Pizza [11 inches]", "addr1", restaurant_id="855114",
        )
        content = result["messages"][0].content
        self.assertIn("Mediterranean Veg Pizza [11 inches]", content)
        self.assertIn("add-on", content)
        self.assertNotIn("Great —", content, "a success lead-in must not precede a refusal")

    def test_add_dish_to_cart_accepts_an_addon_only_item_when_the_empty_selection_verifiably_lands(self):
        """The new, opposite case from the fallback test above: real
        evidence (logger.txt, "Korean Honey Gochujang Chicken Wings (4 Pcs)
        (Its extra Spicy)", tagged has-addons only with one "Add on Sauce"
        group -- no variants) is what motivated trying an explicit empty
        addon selection instead of refusing every has-addons item outright.
        When get_cart afterward actually shows the item present, Cravr must
        accept it as a real success (cart shown, restaurant recorded) --
        not the old unconditional refusal."""
        dish = {
            "item_name": "Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy)",
            "menu_item_id": "182602816", "restaurant_id": "300123",
            "restaurant_name": "Some Korean Place", "tags": "has addons",
        }

        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items"):
            landed_message = (
                "Items (1):\n  - Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy) "
                "— ₹279 (ID: 182602816)\n\nTO PAY: ₹279"
            )
            mock_get_cart.side_effect = [
                {"status": "empty", "message": "Cart is empty."},  # before the merge
                {"status": "text", "message": landed_message},  # verify after the attempt -- item landed this time
                {"status": "text", "message": landed_message},  # _show_cart_with_next_steps' own get_cart call
            ]
            mock_update.return_value = {"status": "success", "message": "Cart updated."}
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="Great — X it is.")

        mock_update.assert_called_once()
        sent_items = mock_update.call_args[0][2]
        target_entry = next(e for e in sent_items if e["menu_item_id"] == "182602816")
        self.assertEqual(target_entry.get("addons"), [])
        content = result["messages"][0].content
        self.assertNotIn("needs a variant/add-on choice", content)
        self.assertEqual(result["cart_state"]["restaurant"], {"id": "300123", "name": "Some Korean Place"})

    def test_add_dish_to_cart_refuses_items_with_variant_choices_too(self):
        dish = {"item_name": "Fiery Avocado Roll", "menu_item_id": "157423529",
                "restaurant_id": "996967", "restaurant_name": "Mister Edamame",
                "tags": "Veg, has variants"}

        with patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items"):
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="")

        mock_update.assert_not_called()
        self.assertIn("Fiery Avocado Roll", result["messages"][0].content)

    def test_add_dish_to_cart_refuses_even_when_the_verification_probe_itself_fails(self):
        """The probe is diagnostic only -- a failure there (e.g. the real
        MCP call errors out) must never block or alter the refusal."""
        dish = {"item_name": "Fiery Avocado Roll", "menu_item_id": "157423529",
                "restaurant_id": "996967", "restaurant_name": "Mister Edamame",
                "tags": "Veg, has variants"}

        with patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", side_effect=RuntimeError("boom")):
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1, lead_in="")

        mock_update.assert_not_called()
        self.assertIn("Fiery Avocado Roll", result["messages"][0].content)


    def test_add_dish_to_cart_refuses_a_different_restaurant_than_the_known_cart(self):
        """Real-world regression: adding a Domino's Pizza item while the
        cart already held a Mister Edamame item (neither tagged
        has-addons/has-variants) made the real update_food_cart respond
        'Cart updated.\nCart is empty.' -- and unlike the addon failures,
        the follow-up get_food_cart showed the cart fully wiped
        (status='empty' item_count=0), not just the new item missing.
        Swiggy carts are single-restaurant; must refuse before sending."""
        dish = {"item_name": "Rich molten cheddar & cheesy melt", "menu_item_id": "208239838",
                "restaurant_id": "879984", "restaurant_name": "Domino's Pizza"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update:
            result = CartAgent.add_dish_to_cart(
                dish, "addr1", quantity=1,
                lead_in="Great — Rich molten cheddar & cheesy melt from Domino's Pizza (₹699.0) it is.",
                known_restaurant_id="996967", known_restaurant_name="Mister Edamame",
            )

        mock_get_cart.assert_not_called()
        mock_update.assert_not_called()
        content = result["messages"][0].content
        self.assertIn("Mister Edamame", content)
        self.assertIn("Domino's Pizza", content)
        self.assertNotIn("Great —", content, "a success lead-in must not precede a refusal")

    def test_add_dish_to_cart_proceeds_when_known_restaurant_matches(self):
        """Same-restaurant adds must not trip the new guard -- this is the
        ordinary "add a second item" path the guard must not break."""
        dish = {"item_name": "All Fungi Pizza", "menu_item_id": "179082813",
                "restaurant_id": "351682", "restaurant_name": "Timpano - The Pasta Shop"}
        existing_cart_text = (
            "Items (1):\n  - Italian Meat Lover Pizza — ₹260 (ID: 179082816)\n\nTO PAY: ₹398"
        )

        with patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "text", "message": existing_cart_text}), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}) as mock_update:
            result = CartAgent.add_dish_to_cart(
                dish, "addr1", quantity=1,
                known_restaurant_id="351682", known_restaurant_name="Timpano - The Pasta Shop",
            )

        mock_update.assert_called_once()
        self.assertNotIn("Swiggy carts can only hold", result["messages"][0].content)

    def test_add_dish_to_cart_records_the_restaurant_it_just_used(self):
        """get_food_cart's real (text-mode) response never fills in
        `restaurant` (food_mcp.get_cart hardcodes it to None off that
        path), so without this the cross-restaurant guard would have
        nothing to compare against on the NEXT add. A successful add must
        record what it used so _current_restaurant_id/_name can see it."""
        dish = {"item_name": "Bowl", "menu_item_id": "111", "restaurant_id": "R1",
                "restaurant_name": "Healthy Eats"}

        with patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "text", "message": "Items (1):\n  - Bowl — ₹200 (ID: 111)\n\nTO PAY: ₹200"}), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}):
            result = CartAgent.add_dish_to_cart(dish, "addr1", quantity=1)

        self.assertEqual(
            result["cart_state"]["restaurant"], {"id": "R1", "name": "Healthy Eats"},
        )

    def test_current_restaurant_helpers_read_the_recorded_restaurant(self):
        state = _state("view", cart_state={"status": "text", "restaurant": {"id": "R1", "name": "Healthy Eats"}})
        self.assertEqual(CartAgent._current_restaurant_id(state), "R1")
        self.assertEqual(CartAgent._current_restaurant_name(state), "Healthy Eats")


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
        # Regression: the cart is genuinely empty after a successful flush,
        # so the restaurant recorded for it must be cleared here -- or the
        # NEXT add wrongly compares against the one just cleared. Live bug:
        # "Your cart already has items from Social" right after "start
        # over" had just flushed the cart clean.
        self.assertIsNone(result["cart_state"]["restaurant"])

    def test_start_over_then_picking_a_different_restaurant_is_not_refused(self):
        """End-to-end replay of the live regression: cart had items from
        "Social", user hit "start over", flush succeeded, then picking a
        dish from an entirely different restaurant ("Blue Tokai Coffee
        Roasters") must NOT trip the cross-restaurant guard -- the cart is
        empty now, there's nothing to conflict with."""
        from cravr.recommend.RecommendAgent import recommendation_agent_node

        raw_results = [
            {"item_name": "Chicken Mortadella And Cheese Sando", "menu_item_id": "183585989",
             "restaurant_id": "32955", "restaurant_name": "Blue Tokai Coffee Roasters",
             "price_inr": 545, "course": "main"},
        ]

        def fake_flush(tool_name, arguments):
            if tool_name == "flush_food_cart":
                return "Cart cleared.\n\nCart widget is displayed."
            raise AssertionError(tool_name)

        chosen = [{"item_name": "Mile High Club Veg", "menu_item_id": "63432924",
                   "restaurant_id": "53859", "restaurant_name": "Social"}]
        state = self._state_with_brief(
            CartAgent.CART_START_OVER_VALUE, chosen=chosen,
            cart_state={"status": "text", "restaurant": {"id": "53859", "name": "Social"}},
        )
        with patch("cravr.food_mcp.call_tool_sync", side_effect=fake_flush),              patch("cravr.recommend.RecommendAgent._call_recommendation_tool", return_value=raw_results):
            start_over_result = cart_agent_node(state)

        state["cart_state"] = start_over_result["cart_state"]
        state["recommendation_state"] = start_over_result["recommendation_state"]
        state["messages"] = state["messages"] + [HumanMessage(content="1")]

        with patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "empty", "message": "Cart is empty."}),              patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}):
            pick_result = recommendation_agent_node(state)

        content = pick_result["messages"][0].content
        self.assertNotIn("Swiggy carts can only hold", content, content)
        self.assertNotIn("needs a variant/add-on choice", content, content)

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



class AddonSelectionInterruptTests(unittest.TestCase):
    """add_flow._ask_for_all_addons asks about every real addon group in
    ONE consolidated question (via interrupt()/Command(resume=...), the
    same pause-and-resume mechanism intent_discovery_node uses -- see
    add_flow.py and graph.py) instead of always attempting an item with an
    unconditional empty addons selection. This fixture has just one group,
    so it also covers the single-group case; MultiGroupAddonSelectionTests
    below covers several groups answered in one reply.

    Exercised through the compiled graph (cravr.graph.app), not
    add_dish_to_cart directly -- interrupt() only actually pauses/resumes
    inside a real graph run backed by a checkpointer, same reason
    test_graph.py's ClarificationContextTests goes through `app` rather
    than calling intent_discovery_node bare.

    The node function re-runs from the top on resume (replaying the
    already-answered interrupt instead of pausing again), so
    search_menu_items legitimately gets called once per invoke -- twice
    total here -- while get_cart/update_cart (which only run AFTER the
    addon question is resolved) run exactly once each, all during the
    resume. Same idempotent-replay shape intentAgent.py documents for its
    own extraction call.
    """

    _DISH = {
        "item_name": "Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy)",
        "menu_item_id": "182602816", "restaurant_id": "300123",
        "restaurant_name": "Some Korean Place", "tags": "has addons",
    }
    # Real live shape (see test_food_mcp.py's KOREAN_WINGS_RESPONSE_TEXT),
    # already parsed into what _fetch_addon_groups expects back.
    _DETAILED_PROBE = {
        "items": {}, "has_more": False, "next_offset": None,
        "detailed_items": {
            "182602816": {
                "item_name": _DISH["item_name"], "price_inr": 279,
                "addon_groups": [{
                    "group_id": "306232169", "group_name": "Add on Sauce",
                    "choices": [
                        {"id": "115016421", "name": "Korean Buffalo Sauce",
                         "price_inr": 49, "group_id": "306232169"},
                        {"id": "115016420", "name": "Korean BBQ Sauce",
                         "price_inr": 49, "group_id": "306232169"},
                    ],
                }],
                "variant_groups": [],
            },
        },
    }
    _LANDED_MESSAGE = (
        "Items (1):\n  - Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy) "
        "— ₹328 (ID: 182602816)\n\nTO PAY: ₹328"
    )

    def _run_addon_flow(self, resume_value):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "messages": [HumanMessage(content="add it to cart")],
            "recommendation_state": {"last_recommendations": [self._DISH]},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {},
        }

        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items",
                   return_value=self._DETAILED_PROBE) as mock_probe:
            mock_get_cart.side_effect = [
                {"status": "empty", "message": "Cart is empty."},  # before the merge
                {"status": "text", "message": self._LANDED_MESSAGE},  # verify after the attempt
                {"status": "text", "message": self._LANDED_MESSAGE},  # _show_cart_with_next_steps' own call
            ]
            mock_update.return_value = {"status": "success", "message": "Cart updated."}

            first = cravr_app.invoke(state, config=config)
            self.assertTrue(first.get("__interrupt__"),
                             "expected cart_agent to pause and ask about the addon group")
            pending = first["__interrupt__"][0].value
            self.assertEqual(pending["agent"], "cart_agent")
            self.assertIn("Add on Sauce", pending["question"])
            option_values = {o["value"] for o in pending["options"]}
            self.assertEqual(option_values, {"115016421", "115016420", "__no_addon__"})

            second = cravr_app.invoke(Command(resume=resume_value), config=config)

        mock_probe.assert_any_call(self._DISH["item_name"], "addr1", restaurant_id="300123")
        mock_update.assert_called_once()
        sent_items = mock_update.call_args[0][2]
        target_entry = next(e for e in sent_items if e["menu_item_id"] == "182602816")
        return target_entry, second

    def test_picking_a_real_addon_sends_its_actual_id_not_an_empty_selection(self):
        target_entry, result = self._run_addon_flow("115016421")
        self.assertEqual(
            target_entry.get("addons"),
            [{"choice_id": 115016421, "group_id": 306232169, "name": "Korean Buffalo Sauce"}],
        )
        content = result["messages"][0].content
        self.assertNotIn("needs a variant/add-on choice", content)

    def test_choosing_none_still_sends_an_explicit_empty_selection(self):
        target_entry, result = self._run_addon_flow("__no_addon__")
        self.assertEqual(target_entry.get("addons"), [])
        content = result["messages"][0].content
        self.assertNotIn("needs a variant/add-on choice", content)

    def test_a_reply_the_menu_never_offered_is_skipped_not_guessed_at(self):
        """Free text the option list didn't anticipate (typed instead of
        tapped) must never be forwarded to Swiggy as if it were a real
        choice id -- see _match_addon_reply."""
        target_entry, result = self._run_addon_flow("surprise me")
        self.assertEqual(target_entry.get("addons"), [])
        content = result["messages"][0].content
        self.assertNotIn("needs a variant/add-on choice", content)




class UpdateCartStructuredRejectionTests(unittest.TestCase):
    """Live regression (logger.txt, 'Roasted Veggie Pita Pocket', 2026-09-
    18): update_food_cart's TEXT content said 'Cart updated.\nCart is
    empty.' -- which update_cart used to report as plain success -- while
    fastmcp's `structured_content` (never inspected before this) carried
    the real, explicit verdict the whole time: `{'successful': False,
    'errorCodes': ['INVALID_ADDON'], 'titleMessage': 'Apologies! one or
    more items in your cart are no longer available', ...}`. Two things
    must hold now: food_mcp.cart.update_cart treats that as an error (not
    text-mode "success"), and add_flow's addon-only attempt tells the user
    Swiggy rejected the add-on(s) they picked, not the generic "needs a
    variant/add-on choice I can't set for you yet" refusal -- which is
    actively misleading here, since Cravr DID try and DID send the
    add-ons; Swiggy turned them down.
    """

    def _fake_call_tool_result(self, text, structured_content):
        # Minimal stand-in for fastmcp's own CallToolResult dataclass --
        # only the attributes _unwrap()/update_cart() actually read
        # (content[i].text, .structured_content, .data, .is_error).
        content_item = SimpleNamespace(type="text", text=text)
        return SimpleNamespace(
            content=[content_item],
            structured_content=structured_content,
            data=structured_content,
            is_error=False,
        )

    def test_food_mcp_update_cart_treats_an_explicit_rejection_as_an_error(self):
        raw_result = self._fake_call_tool_result(
            "Cart updated.\nCart is empty.\n\nCart widget is displayed. "
            "Wait for the user to review and confirm.",
            {
                "successful": False,
                "errorCodes": ["INVALID_ADDON"],
                "titleMessage": "Apologies! one or more items in your cart are no longer available",
                "statusMessage": "Restaurant may have removed the item(s) from their menu.",
            },
        )
        with patch("cravr.food_mcp.call_tool_sync", return_value=raw_result):
            result = food_mcp.update_cart(
                "58024", "addr1",
                [{"menu_item_id": "188424395", "quantity": 1, "addons": [
                    {"id": "133264885", "name": "Hummus Cup", "group_id": "259942536",
                     "quantity": 1, "price": 80.0},
                ]}],
                restaurant_name="Saladspoint",
            )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_codes"], ["INVALID_ADDON"])
        self.assertIn("no longer available", result["message"])

    def test_addon_only_attempt_relays_the_real_rejection_not_the_generic_refusal(self):
        dish = {
            "item_name": "Roasted Veggie Pita Pocket",
            "menu_item_id": "188424395", "restaurant_id": "58024",
            "restaurant_name": "Saladspoint", "tags": "has addons",
        }
        probe = {
            "items": {}, "has_more": False, "next_offset": None,
            "detailed_items": {
                "188424395": {
                    "item_name": dish["item_name"], "price_inr": 259,
                    "addon_groups": [{
                        "group_id": "259942536", "group_name": "Dressing Add On's",
                        "choices": [
                            {"id": "133264885", "name": "Hummus Cup", "price_inr": 80, "group_id": "259942536"},
                        ],
                    }],
                    "variant_groups": [],
                },
            },
        }
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "messages": [HumanMessage(content="add it to cart")],
            "recommendation_state": {"last_recommendations": [dish]},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {},
        }
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", return_value=probe):
            mock_get_cart.return_value = {"status": "empty", "message": "Cart is empty."}
            mock_update.return_value = {
                "status": "error",
                "message": "Apologies! one or more items in your cart are no longer available",
                "error_codes": ["INVALID_ADDON"],
            }

            cravr_app.invoke(state, config=config)
            second = cravr_app.invoke(Command(resume="133264885"), config=config)

        content = second["messages"][-1].content
        self.assertIn("Swiggy rejected the add-on(s) you picked", content)
        self.assertIn("INVALID_ADDON", content)
        self.assertNotIn("needs a variant/add-on choice I can't set for you yet", content)


class MultiGroupAddonSelectionTests(unittest.TestCase):
    """Real-world regressions (logger.txt, 'Spaghetti Aglio Olio [Veg]' /
    'Spaghetti Carbonara Pasta', 4 real addon groups: Choose Your Gourmet
    Topping, Choose Your Veg Toppings, Choose Your Cheese, Choose Your Non
    Veg Toppings). Two separate live bugs this class guards against:

    1. The original group-by-group loop capped at 3 groups and never even
       asked about the 4th before sending the add. _ask_for_all_addons
       shows every group in one consolidated first question -- no cap.

    2. The first "all at once" version fixed (1) but broke something else,
       also confirmed live: a real user naturally TAPPED one button, and
       since a tap only ever returns ONE value, that silently finalized
       the whole add with just one group answered and the other three
       left as an implicit (unintended) "none" -- "can use only 1, WTF".
       _ask_for_all_addons now treats one round's reply as "at least
       this": unresolved groups get a shorter FOLLOWUP question instead of
       being dropped, repeating until the user says done/none or every
       group's had a real answer. See
       test_tapping_one_choice_gets_asked_again_about_the_rest for the
       direct regression test.

    Ids/names/prices for the first three groups are the real ones from
    the log; the 4th group's choice is a fixture (the real runs never got
    asked about it, since they either capped at 3 groups or stopped after
    the first tap)."""

    _DISH = {
        "item_name": "Spaghetti Aglio Olio [Veg]", "menu_item_id": "32257515",
        "restaurant_id": "30449", "restaurant_name": "Some Italian Place",
        "tags": "has addons",
    }
    _ADDON_GROUPS = [
        {"group_id": "37840179", "group_name": "Choose Your Gourmet Topping",
         "choices": [{"id": "25698072", "name": "Sundried Tomato", "price_inr": 95.0, "group_id": "37840179"}]},
        {"group_id": "37840188", "group_name": "Choose Your Veg Toppings",
         "choices": [{"id": "25698096", "name": "Mushroom", "price_inr": 75.0, "group_id": "37840188"}]},
        {"group_id": "37840199", "group_name": "Choose Your Cheese",
         "choices": [{"id": "25698108", "name": "Extra Mozzarella Cheese", "price_inr": 95.0, "group_id": "37840199"}]},
        {"group_id": "37840200", "group_name": "Choose Your Non Veg Toppings",
         "choices": [{"id": "25698200", "name": "Chicken Sausage", "price_inr": 60.0, "group_id": "37840200"}]},
    ]
    _DETAILED_PROBE = {
        "items": {}, "has_more": False, "next_offset": None,
        "detailed_items": {
            "32257515": {
                "item_name": _DISH["item_name"], "price_inr": 399,
                "addon_groups": _ADDON_GROUPS, "variant_groups": [],
            },
        },
    }
    _LANDED_MESSAGE = "Items (1):\n  - Spaghetti Aglio Olio [Veg] — ₹664 (ID: 32257515)\n\nTO PAY: ₹664"

    def _run(self, *resume_sequence):
        """Drives the graph through as many interrupt rounds as it takes
        to finish. Supplied resume values are used in order; once
        exhausted, any further round is closed out with "__no_addon__"
        ("nothing else") so a test that doesn't care about the later
        rounds doesn't have to spell them all out."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "messages": [HumanMessage(content="add it to cart")],
            "recommendation_state": {"last_recommendations": [self._DISH]},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {},
        }
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", return_value=self._DETAILED_PROBE):
            mock_get_cart.side_effect = [
                {"status": "empty", "message": "Cart is empty."},
                {"status": "text", "message": self._LANDED_MESSAGE},
                {"status": "text", "message": self._LANDED_MESSAGE},
            ]
            mock_update.return_value = {"status": "success", "message": "Cart updated."}

            result = cravr_app.invoke(state, config=config)
            self.assertTrue(result.get("__interrupt__"))
            first_question = result["__interrupt__"][0].value["question"]
            for group_name in (
                "Choose Your Gourmet Topping", "Choose Your Veg Toppings",
                "Choose Your Cheese", "Choose Your Non Veg Toppings",
            ):
                self.assertIn(group_name, first_question)

            pending = list(resume_sequence)
            rounds = 0
            while result.get("__interrupt__"):
                rounds += 1
                self.assertLessEqual(rounds, 4, "should never loop more rounds than there are groups")
                value = pending.pop(0) if pending else "__no_addon__"
                result = cravr_app.invoke(Command(resume=value), config=config)

        sent_items = mock_update.call_args[0][2]
        target_entry = next(e for e in sent_items if e["menu_item_id"] == "32257515")
        return target_entry, result

    def test_all_four_groups_are_surfaced_in_one_question_not_capped_at_three(self):
        target_entry, result = self._run("Sundried Tomato, Mushroom, and Extra Mozzarella Cheese")
        self.assertEqual(
            {a["name"] for a in target_entry.get("addons", [])},
            {"Sundried Tomato", "Mushroom", "Extra Mozzarella Cheese"},
        )
        content = result["messages"][0].content
        self.assertNotIn("needs a variant/add-on choice", content)

    def test_numbered_positions_across_groups_pick_the_right_choices(self):
        # 1=Sundried Tomato, 2=Mushroom, 3=Extra Mozzarella Cheese, 4=Chicken Sausage
        target_entry, _ = self._run("1, 4")
        self.assertEqual(
            {a["name"] for a in target_entry.get("addons", [])},
            {"Sundried Tomato", "Chicken Sausage"},
        )

    def test_none_clears_every_group_at_once(self):
        target_entry, _ = self._run("no thanks")
        self.assertEqual(target_entry.get("addons"), [])

    def _invoke_once(self, resume_value, update_side_effect=None, valid_addons=None):
        """Start the flow, assert it pauses ONCE with a multi-select
        question, resume with `resume_value`, and return (first_interrupt,
        result, mock_update)."""
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "messages": [HumanMessage(content="add it to cart")],
            "recommendation_state": {"last_recommendations": [self._DISH]},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {},
        }
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart_item_details",
                   return_value={k: {"variants": [], "addons": [], "valid_addons": v}
                                 for k, v in (valid_addons or {}).items()}), \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", return_value=self._DETAILED_PROBE):
            mock_get_cart.return_value = {"status": "text", "message": self._LANDED_MESSAGE}
            mock_get_cart.side_effect = None
            first_get = [{"status": "empty", "message": "Cart is empty."}]
            mock_get_cart.side_effect = first_get + [
                {"status": "text", "message": self._LANDED_MESSAGE}] * 6
            if update_side_effect is not None:
                mock_update.side_effect = update_side_effect
            else:
                mock_update.return_value = {"status": "success", "message": "Cart updated."}
            first = cravr_app.invoke(state, config=config)
            self.assertTrue(first.get("__interrupt__"))
            pending = first["__interrupt__"][0].value
            result = cravr_app.invoke(Command(resume=resume_value), config=config)
        return pending, result, mock_update

    def test_the_addon_question_is_one_multi_select_interrupt(self):
        pending, result, mock_update = self._invoke_once("25698072,25698096,25698108,25698200")
        self.assertTrue(pending.get("multi_select"))
        self.assertEqual(
            {o["group"] for o in pending["options"] if o.get("group")},
            {"Choose Your Gourmet Topping", "Choose Your Veg Toppings",
             "Choose Your Cheese", "Choose Your Non Veg Toppings"},
        )
        # ONE resume finishes the flow -- no per-group follow-up questions.
        self.assertFalse(result.get("__interrupt__"))
        sent = next(e for e in mock_update.call_args[0][2] if e["menu_item_id"] == "32257515")
        self.assertEqual(
            [a["name"] for a in sent["addons"]],
            ["Sundried Tomato", "Mushroom", "Extra Mozzarella Cheese", "Chicken Sausage"],
        )

    def test_several_picks_in_the_same_group_are_all_kept(self):
        # Regression for "can select only one": two choices from ONE group.
        probe = {"items": {}, "has_more": False, "next_offset": None, "detailed_items": {
            "32257515": {"item_name": "Salad", "price_inr": 399, "variant_groups": [],
                         "addon_groups": [{"group_id": "g1", "group_name": "Salad Add-ons",
                                           "max_addons": 8, "choices": [
                {"id": "1", "name": "Feta Cheese", "price_inr": 69, "group_id": "g1"},
                {"id": "2", "name": "Black Olives", "price_inr": 49, "group_id": "g1"},
                {"id": "3", "name": "Lavash", "price_inr": 49, "group_id": "g1"}]}]}}}
        with patch.object(type(self), "_DETAILED_PROBE", probe):
            _, result, mock_update = self._invoke_once("1,3")
        sent = next(e for e in mock_update.call_args[0][2] if e["menu_item_id"] == "32257515")
        self.assertEqual([a["name"] for a in sent["addons"]], ["Feta Cheese", "Lavash"])

    def test_group_max_addons_is_enforced(self):
        from cravr.cart.add_flow import _enforce_group_limits
        choices = [{"id": "1", "group_id": "g", "max_addons": 1},
                   {"id": "2", "group_id": "g", "max_addons": 1}]
        picked = [{"id": "1", "group_id": "g", "name": "a"}, {"id": "2", "group_id": "g", "name": "b"}]
        self.assertEqual([p["id"] for p in _enforce_group_limits(picked, choices)], ["1"])

    def test_invalid_addon_falls_back_to_item_first_then_valid_addons(self):
        rejection = {"status": "error", "message": "Apologies!", "error_codes": ["INVALID_ADDON"]}
        ok = {"status": "success", "message": "Cart updated."}
        valid = {"32257515": [{
            "group_id": "37840188", "group_name": "Choose Your Veg Toppings",
            "min_addons": None, "max_addons": 3,
            "choices": [{"id": "25698096", "name": "Mushroom", "price_inr": 75,
                         "group_id": "37840188",
                         "raw": {"id": "25698096", "name": "Mushroom", "price": 75, "group_id": "37840188"}}],
        }]}
        _, result, mock_update = self._invoke_once(
            "25698096", update_side_effect=[rejection, ok, ok], valid_addons=valid)
        self.assertEqual(mock_update.call_count, 3)
        step1 = next(e for e in mock_update.call_args_list[1][0][2] if e["menu_item_id"] == "32257515")
        step3 = next(e for e in mock_update.call_args_list[2][0][2] if e["menu_item_id"] == "32257515")
        self.assertEqual(step1["addons"], [])
        # first candidate shape follows Swiggy's real schema: choice_id + group_id (+ name)
        self.assertEqual(step3["addons"], [{"choice_id": 25698096, "group_id": 37840188, "name": "Mushroom"}])
        self.assertNotIn("couldn't be added", result["messages"][-1].content)

    _VALID_ONE = {"32257515": [{
        "group_id": "37840188", "group_name": "Choose Your Veg Toppings",
        "min_addons": None, "max_addons": 3,
        "choices": [{"id": "25698096", "name": "Mushroom", "price_inr": 75, "group_id": "37840188",
                     "raw": {"id": "25698096", "name": "Mushroom", "price": 7500}}],
    }]}

    def test_request_shapes_are_tried_in_turn_until_one_is_accepted(self):
        rej = {"status": "error", "message": "x", "error_codes": ["INVALID_ADDON"]}
        ok = {"status": "success", "message": "Cart updated."}
        # first attempt rejected, item alone ok, shape S rejected, shape S2 accepted
        _, result, mock_update = self._invoke_once(
            "25698096", update_side_effect=[rej, ok, rej, ok], valid_addons=self._VALID_ONE)
        self.assertEqual(mock_update.call_count, 4)
        shape_b = next(e for e in mock_update.call_args_list[3][0][2] if e["menu_item_id"] == "32257515")
        self.assertEqual(shape_b["addons"], [{"choiceId": 25698096, "groupId": 37840188}])

    def test_when_no_shape_is_accepted_the_plain_dish_is_kept(self):
        rej = {"status": "error", "message": "x", "error_codes": ["INVALID_ADDON"]}
        ok = {"status": "success", "message": "Cart updated."}
        _, result, mock_update = self._invoke_once(
            "25698096", update_side_effect=[rej, ok] + [rej] * 6, valid_addons=self._VALID_ONE)
        self.assertEqual(mock_update.call_count, 8)  # 1 first + 1 alone + 6 shapes
        self.assertIn("without them", result["messages"][-1].content)

    def test_item_rejected_even_alone_reports_the_rejection(self):
        rej = {"status": "error", "message": "x", "error_codes": ["INVALID_ADDON"]}
        _, result, mock_update = self._invoke_once("25698096", update_side_effect=[rej, rej])
        self.assertEqual(mock_update.call_count, 2)
        self.assertIn("couldn't be added", result["messages"][-1].content)

    def test_invalid_addon_with_no_valid_addons_keeps_the_plain_dish(self):
        rejection = {"status": "error", "message": "Apologies!", "error_codes": ["INVALID_ADDON"]}
        ok = {"status": "success", "message": "Cart updated."}
        _, result, mock_update = self._invoke_once(
            "25698096", update_side_effect=[rejection, ok], valid_addons={})
        self.assertEqual(mock_update.call_count, 2)
        content = result["messages"][-1].content
        self.assertIn("without them", content)
        self.assertNotIn("couldn't be added", content)

    def test_required_group_is_asked_again_when_the_reply_skips_it(self):
        """Live regression (Gourmet Veg Caesar Salad): 'Dressing &
        Accompaniments' had minAddons=1; a reply that skipped it was sent
        anyway and rejected with INVALID_ADDON -- even the plain item was."""
        probe = {"items": {}, "has_more": False, "next_offset": None, "detailed_items": {
            "32257515": {"item_name": "Salad", "price_inr": 399, "variant_groups": [],
                         "addon_groups": [
                {"group_id": "gd", "group_name": "Dressing", "min_addons": 1, "max_addons": 1, "choices": [
                    {"id": "10", "name": "Caesar Yoghurt", "price_inr": 0, "group_id": "gd"},
                    {"id": "11", "name": "Olive Oil", "price_inr": 0, "group_id": "gd"}]},
                {"group_id": "gf", "group_name": "Fruit", "max_addons": 5, "choices": [
                    {"id": "20", "name": "Plum", "price_inr": 40, "group_id": "gf"}]}]}}}
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {"messages": [HumanMessage(content="add it to cart")],
                 "recommendation_state": {"last_recommendations": [self._DISH]},
                 "context_profile": {"address_id": "addr1"}, "intent_profile": {}, "history_profile": {}}
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as gc, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart") as up, \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", return_value=probe):
            gc.side_effect = [{"status": "empty", "message": "Cart is empty."}] + \
                [{"status": "text", "message": self._LANDED_MESSAGE}] * 5
            up.return_value = {"status": "success", "message": "Cart updated."}
            first = cravr_app.invoke(state, config=config)
            q1 = first["__interrupt__"][0].value
            self.assertIn("REQUIRED", q1["question"])
            # required group listed first
            self.assertEqual(q1["options"][0]["group"], "Dressing")
            second = cravr_app.invoke(Command(resume="20"), config=config)  # only the optional fruit
            self.assertTrue(second.get("__interrupt__"), "must re-ask about the required group")
            q2 = second["__interrupt__"][0].value
            self.assertEqual({o["value"] for o in q2["options"]}, {"10", "11"})
            self.assertNotIn("__no_addon__", {o["value"] for o in q2["options"]})
            third = cravr_app.invoke(Command(resume="10"), config=config)
            self.assertFalse(third.get("__interrupt__"))
        sent = next(e for e in up.call_args[0][2] if e["menu_item_id"] == "32257515")
        self.assertEqual({a["name"] for a in sent["addons"]}, {"Plum", "Caesar Yoghurt"})


class MatchAddonReplyTests(unittest.TestCase):
    """Direct unit tests of add_flow._match_addon_reply -- the free-text
    matcher _ask_for_all_addons hands a resumed reply to. Exercised
    without the graph since it's a pure function of (reply, choices)."""

    _CHOICES = [
        {"id": "1001", "name": "Sundried Tomato", "price_inr": 95.0,
         "group_id": "g1", "group_name": "Toppings"},
        {"id": "1002", "name": "Mushroom", "price_inr": 75.0,
         "group_id": "g1", "group_name": "Toppings"},
        {"id": "1003", "name": "Extra Mozzarella Cheese", "price_inr": 95.0,
         "group_id": "g2", "group_name": "Cheese"},
    ]

    def _names(self, reply):
        from cravr.cart.add_flow import _match_addon_reply
        return [c["name"] for c in _match_addon_reply(reply, self._CHOICES)]

    def test_bare_none_phrase_returns_nothing(self):
        for phrase in ("none", "No", "nothing", "no thanks", "skip", "  Nope.  "):
            self.assertEqual(self._names(phrase), [], phrase)

    def test_exact_id_matches_regardless_of_list_position(self):
        self.assertEqual(self._names("1002"), ["Mushroom"])

    def test_1_based_position_in_the_shown_list(self):
        self.assertEqual(self._names("3"), ["Extra Mozzarella Cheese"])

    def test_out_of_range_position_matches_nothing(self):
        self.assertEqual(self._names("99"), [])

    def test_exact_name_is_case_insensitive(self):
        self.assertEqual(self._names("sundried tomato"), ["Sundried Tomato"])

    def test_partial_name_still_matches(self):
        self.assertEqual(self._names("mozzarella"), ["Extra Mozzarella Cheese"])

    def test_comma_and_ampersand_separated_multi_select(self):
        self.assertEqual(
            set(self._names("Sundried Tomato, mushroom & mozzarella")),
            {"Sundried Tomato", "Mushroom", "Extra Mozzarella Cheese"},
        )

    def test_the_word_and_also_splits_tokens(self):
        self.assertEqual(
            set(self._names("mushroom and mozzarella")),
            {"Mushroom", "Extra Mozzarella Cheese"},
        )

    def test_an_unmatched_token_is_dropped_not_guessed_at(self):
        self.assertEqual(self._names("extra spicy sauce"), [])

    def test_a_matched_token_survives_alongside_an_unmatched_one(self):
        self.assertEqual(self._names("mushroom, extra spicy sauce"), ["Mushroom"])

    def test_repeating_the_same_choice_twice_is_deduped(self):
        self.assertEqual(self._names("mushroom, mushroom"), ["Mushroom"])

    def test_empty_reply_matches_nothing(self):
        self.assertEqual(self._names(""), [])
        self.assertEqual(self._names("   "), [])



if __name__ == "__main__":
    unittest.main()



class CartItemDetailsParsingTests(unittest.TestCase):
    """food_mcp.get_cart_item_details reads valid_addons/variants off
    get_food_cart's structured_content -- live shape from logger.txt
    (2026-09-26, Harvest Salad Co): ids are INTs and valid_addons prices are
    PAISE, group names live under `group_name`."""

    def test_parses_the_live_shape(self):
        structured = {"statusCode": 6, "data": {"items": [{
            "menu_item_id": 205083764, "quantity": 1,
            "variants": [{"variation_id": 226498675, "group_id": 77918327, "name": "Small"}],
            "addons": [],
            "valid_addons": [{
                "group_id": 308598175, "group_name": "Add Grain", "minAddons": 0, "maxAddons": 1,
                "choices": [{"id": 149928102, "name": "Brown Rice", "price": 4000, "inStock": 1}],
            }],
        }]}}
        result_obj = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Items (1): ...")],
            structured_content=structured, data=None, is_error=False, meta=None,
        )
        from cravr import food_mcp
        with patch("cravr.food_mcp.call_tool_sync", return_value=result_obj):
            details = food_mcp.get_cart_item_details("addr1")
        item = details["205083764"]
        self.assertEqual(item["variants"][0]["variation_id"], 226498675)
        group = item["valid_addons"][0]
        self.assertEqual((group["group_id"], group["group_name"], group["max_addons"]),
                         ("308598175", "Add Grain", 1))
        choice = group["choices"][0]
        self.assertEqual((choice["id"], choice["price_inr"], choice["raw"]["price"]), ("149928102", 40.0, 4000))


class CartConflictTests(unittest.TestCase):
    """A dish from another restaurant while the real Swiggy cart still holds
    leftovers -> INVALID_ITEM_IDS_IN_REQUEST. Must be reported as a cart
    conflict with clear/keep buttons, not as an add-on rejection."""

    _DISH = {"item_name": "Paneer Wrap", "menu_item_id": "555", "restaurant_id": "9",
             "restaurant_name": "Wrap Place"}
    _LEFTOVER_CART = {"status": "text", "message": "Items (1):\n- Greek Falafel Salad — ₹300 (ID: 111)"}

    def test_leftover_items_rejected_offers_clear_and_add(self):
        from cravr.cart.routing import CART_CLEAR_AND_ADD_VALUE, CART_KEEP_VALUE
        rej = {"status": "error", "message": "Cart updated.\nCart is empty.",
               "error_codes": ["INVALID_ITEM_IDS_IN_REQUEST"]}
        with patch("cravr.cart.CartAgent.food_mcp.get_cart", return_value=self._LEFTOVER_CART), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart", return_value=rej):
            result = CartAgent.add_dish_to_cart(self._DISH, "addr1", quantity=1, lead_in="")
        cs = result["cart_state"]
        self.assertEqual([o["value"] for o in cs["options"]], [CART_CLEAR_AND_ADD_VALUE, CART_KEEP_VALUE])
        self.assertEqual(cs["pending_dish"], self._DISH)
        text = result["messages"][-1].content
        self.assertIn("one restaurant", text)
        self.assertNotIn("add-on", text.lower())

    def test_clear_and_add_button_flushes_then_adds(self):
        from cravr.cart.routing import CART_CLEAR_AND_ADD_VALUE
        state = {
            "messages": [HumanMessage(content=CART_CLEAR_AND_ADD_VALUE)],
            "cart_state": {"pending_dish": self._DISH, "pending_quantity": 1},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {}, "recommendation_state": {},
        }
        landed = {"status": "text", "message": "Items (1):\n- Paneer Wrap — ₹200 (ID: 555)\nTO PAY: ₹200"}
        with patch("cravr.cart.CartAgent.food_mcp.flush_cart",
                   return_value={"status": "success", "message": "flushed"}) as flush, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   side_effect=[{"status": "empty", "message": "Cart is empty."}, landed, landed]), \
             patch("cravr.cart.CartAgent.food_mcp.update_cart",
                   return_value={"status": "success", "message": "Cart updated."}) as upd:
            CartAgent.cart_agent_node(state)
        flush.assert_called_once()
        sent = upd.call_args[0][2]
        self.assertEqual([i["menu_item_id"] for i in sent], ["555"])

    def test_keep_button_leaves_cart_alone(self):
        from cravr.cart.routing import CART_KEEP_VALUE
        state = {
            "messages": [HumanMessage(content=CART_KEEP_VALUE)],
            "cart_state": {"pending_dish": self._DISH},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {}, "recommendation_state": {},
        }
        with patch("cravr.cart.CartAgent.food_mcp.flush_cart") as flush, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart",
                   return_value={"status": "text", "message": self._LEFTOVER_CART["message"]}):
            CartAgent.cart_agent_node(state)
        flush.assert_not_called()


class ISTClockTests(unittest.TestCase):
    def test_now_ist_is_utc_plus_530(self):
        from datetime import datetime, timezone
        from cravr.clock import now_ist
        delta = now_ist().utcoffset().total_seconds()
        self.assertEqual(delta, 5.5 * 3600)
        self.assertLess(abs((now_ist() - datetime.now(timezone.utc)).total_seconds()), 5)

    def test_meal_window_default_uses_ist_not_host_time(self):
        from datetime import datetime, timezone
        from unittest.mock import patch
        from cravr.intent import meal_clock
        # 06:30 UTC == 12:00 IST -> a lunch-time window, whatever the host TZ is.
        fake = datetime(2026, 9, 26, 6, 30, tzinfo=timezone.utc).astimezone(__import__("cravr.clock", fromlist=["IST"]).IST)
        with patch("cravr.intent.meal_clock.now_ist", return_value=fake):
            window = meal_clock.current_meal_window()
        self.assertTrue(window.start <= fake.time() <= window.end or window.spans_midnight())


class AddonWireFormatTests(unittest.TestCase):
    """Swiggy's real update_food_cart schema (dumped via local_dev/probe.py):
    addons[] entries are {choice_id|choiceId, group_id|groupId, name?, price?}.
    Sending `id` made every add-on INVALID_ADDON on the live service."""

    def test_wire_addon_uses_choice_id_and_group_id_and_drops_price(self):
        from cravr.cart.add_flow import _wire_addon
        wire = _wire_addon({"id": "131955669", "name": "Mirinda", "group_id": "195081163",
                            "quantity": 1, "price": 66.67})
        self.assertEqual(wire, {"choice_id": 131955669, "group_id": 195081163, "name": "Mirinda"})
        self.assertNotIn("id", wire)
        self.assertNotIn("price", wire)
        self.assertNotIn("variation_id", wire)


class VariantThenAddonFlowTests(unittest.TestCase):
    """Real-world regression (logger.txt 2026-09-27, 'Classic Margherita
    Pizza', Caro Napoli Pizza -- restaurantId 964454): the item has a real
    Size variant (6/8 slices) with SIZE-SPECIFIC addon groups (a 6-slice
    'Napoli Veg'/'Booster' vs an 8-slice one), and the dish's own `tags`
    string ("has addons") never said "has variants" -- so the old
    tag-only refusal check never caught it. Asking one combined addon
    question up front let a user pick 'Extra Cheese' (6-slice group) AND
    'Hot Honey' (8-slice-only group) with no variant ever chosen --
    update_food_cart accepted the write, then get_food_cart came back
    'Cart is empty' with errorCodes=['INVALID_ADDON'] a beat later.

    Fix: ask which variant FIRST (a required single-select, like any
    other mandatory group), add the item with JUST that variant, then ask
    add-ons using ONLY what Swiggy's own valid_addons says is valid for
    THAT variant -- never the static, size-blind groups from search_menu.
    """

    _DISH = {
        "item_name": "Classic Margherita Pizza", "menu_item_id": "152977469",
        "restaurant_id": "964454", "restaurant_name": "Caro Napoli Pizza",
        "tags": "has addons",  # matches the live case -- NOT tagged "has variants"
    }
    _DETAILED_PROBE = {
        "items": {}, "has_more": False, "next_offset": None,
        "detailed_items": {
            "152977469": {
                "item_name": _DISH["item_name"], "price_inr": 1085,
                "addon_groups": [
                    {"group_id": "254031236", "group_name": "Napoli Veg [6 slices]",
                     "choices": [{"id": "141009291", "name": "Extra Cheese",
                                  "price_inr": 95.0, "group_id": "254031236"}]},
                    {"group_id": "285951665", "group_name": "Booster [8 Slices]",
                     "choices": [{"id": "141009301", "name": "Hot Honey",
                                  "price_inr": 35.0, "group_id": "285951665"}]},
                ],
                "variant_groups": [
                    {"group_id": "68366736", "group_name": "Size",
                     "choices": [
                         {"id": "68366800", "name": "6 Slices", "price_inr": 0, "group_id": "68366736"},
                         {"id": "68366801", "name": "8 Slices", "price_inr": 150, "group_id": "68366736"},
                     ]},
                ],
            },
        },
    }
    # Swiggy's OWN filtered view once the 6-slice variant is in the cart --
    # deliberately excludes the 8-slice "Booster" group entirely.
    _VALID_ADDONS_FOR_6_SLICE = {
        "152977469": {"variants": [], "addons": [], "valid_addons": [
            {"group_id": "254031236", "group_name": "Napoli Veg [6 slices]",
             "min_addons": 0, "max_addons": 2,
             "choices": [{"id": "141009291", "name": "Extra Cheese", "price_inr": 95.0,
                          "group_id": "254031236", "raw": {}}]},
        ]},
    }

    @staticmethod
    def _fake_update_cart(restaurant_id, address_id, cart_items, restaurant_name=None):
        target = next((i for i in cart_items if i["menu_item_id"] == "152977469"), None)
        assert target is not None
        if target.get("addons"):
            return {"status": "success", "message": "Cart updated. (variant+addons)"}
        return {"status": "success", "message": "Cart updated. (variant only)"}

    def _run(self):
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = {
            "messages": [HumanMessage(content="add it to cart")],
            "recommendation_state": {"last_recommendations": [self._DISH]},
            "context_profile": {"address_id": "addr1"},
            "intent_profile": {}, "history_profile": {},
        }
        with patch("cravr.cart.CartAgent.food_mcp.get_cart") as mock_get_cart, \
             patch("cravr.cart.CartAgent.food_mcp.update_cart", side_effect=self._fake_update_cart) as mock_update, \
             patch("cravr.cart.CartAgent.food_mcp.get_cart_item_details",
                   return_value=self._VALID_ADDONS_FOR_6_SLICE), \
             patch("cravr.cart.CartAgent.food_mcp.search_menu_items", return_value=self._DETAILED_PROBE):
            landed_message = (
                "Items (1):\n- Classic Margherita Pizza — ₹1085 (ID: 152977469)\n\nTO PAY: ₹1245"
            )
            mock_get_cart.side_effect = [
                {"status": "empty", "message": "Cart is empty."},  # _add_item_with_variant's own merge fetch
                {"status": "empty", "message": "Cart is empty."},  # same fetch again, replayed on the final resume
                {"status": "text", "message": landed_message},      # add_dish_to_cart's own post-add verify
                {"status": "text", "message": landed_message},      # _show_cart_with_next_steps' own get_cart call
            ]
            first = cravr_app.invoke(state, config=config)
            self.assertTrue(first.get("__interrupt__"))
            first_payload = first["__interrupt__"][0].value

            second = cravr_app.invoke(Command(resume="68366800"), config=config)  # "6 Slices"
            self.assertTrue(second.get("__interrupt__"), "expected a SECOND interrupt asking about add-ons")
            second_payload = second["__interrupt__"][0].value

            final = cravr_app.invoke(Command(resume="141009291"), config=config)  # "Extra Cheese"
        return first_payload, second_payload, final, mock_update

    def test_variant_is_asked_first_as_a_required_single_choice(self):
        first_payload, _second_payload, _final, _mock_update = self._run()
        self.assertEqual(first_payload["field"], "variant_groups")
        self.assertTrue(first_payload.get("multi_select"))
        labels = {o["label"] for o in first_payload["options"]}
        self.assertIn("6 Slices", labels)
        self.assertIn("8 Slices (+₹150)", labels)
        # required_only phrasing -- no "None of these" escape hatch for a variant.
        self.assertNotIn("None of these", labels)

    def test_addon_question_only_offers_groups_valid_for_the_chosen_variant(self):
        _first, second_payload, _final, _mock_update = self._run()
        labels = {o["label"] for o in second_payload["options"]}
        self.assertIn("Extra Cheese (+₹95)", labels)
        # The 8-slice-only 'Hot Honey' must NEVER be offered once "6 Slices" was chosen --
        # this is the actual live bug (mixed-variant addons accepted then invalidated).
        self.assertNotIn("Hot Honey (+₹35)", labels)

    def test_variant_sent_alone_first_then_addons_on_top(self):
        _first, _second, _final, mock_update = self._run()
        calls = mock_update.call_args_list
        variant_only_calls = [c for c in calls if not next(
            i for i in c[0][2] if i["menu_item_id"] == "152977469")["addons"]]
        with_addons_calls = [c for c in calls if next(
            i for i in c[0][2] if i["menu_item_id"] == "152977469")["addons"]]
        self.assertTrue(variant_only_calls, "the variant must be sent by itself at least once")
        self.assertTrue(with_addons_calls, "add-ons must be sent on a LATER call, on top of the variant")
        last_addon_entry = next(i for i in with_addons_calls[-1][0][2] if i["menu_item_id"] == "152977469")
        self.assertEqual(last_addon_entry["addons"], [
            {"choice_id": 141009291, "group_id": 254031236, "name": "Extra Cheese"},
        ])
        self.assertIn("variantsV2", last_addon_entry)
        self.assertEqual(last_addon_entry["variantsV2"], [{"group_id": 68366736, "variation_id": 68366800}])

    def test_final_reply_is_not_the_misleading_customization_refusal(self):
        _first, _second, final, _mock_update = self._run()
        content = final["messages"][-1].content
        self.assertNotIn("needs a variant/add-on choice", content)


class VariantWireShapeTests(unittest.TestCase):
    def test_variantsV2_shape_tried_first_with_int_ids(self):
        from cravr.cart.add_flow import _variant_shapes
        shapes = _variant_shapes([{"id": "68366800", "name": "6 Slices", "group_id": "68366736"}])
        label, field, payload = shapes[0]
        self.assertEqual(field, "variantsV2")
        self.assertEqual(payload, [{"group_id": 68366736, "variation_id": 68366800}])

    def test_legacy_variants_shapes_are_fallbacks(self):
        from cravr.cart.add_flow import _variant_shapes
        fields = [field for _label, field, _payload in _variant_shapes(
            [{"id": "1", "name": "Small", "group_id": "9"}])]
        self.assertEqual(fields, ["variantsV2", "variants", "variants"])
