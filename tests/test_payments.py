import unittest
from unittest.mock import patch

from cravr.food_mcp import payments


class GetPaymentOptionsTests(unittest.TestCase):
    RESPONSE = {
        "success": True,
        "data": {
            "cod": {"available": True, "id": "cod"},
            "allMethods": [
                {"id": "phonepe", "displayName": "PhonePe", "groupName": "UPI Apps", "kind": "intent", "enabled": True},
                {"id": "gpay", "displayName": "Google Pay", "groupName": "UPI Apps", "kind": "intent", "enabled": False},
                {"id": "upi_qr", "displayName": "Scan to pay", "groupName": "UPI QR", "kind": "qr", "enabled": True},
            ],
            "paymentAmount": 557,
            "placeOrderToolName": "place_food_order",
        },
    }

    def test_parses_cod_and_upi_availability(self):
        with patch("cravr.food_mcp.payments._call_tool", return_value=self.RESPONSE):
            result = payments.get_payment_options("addr1")
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["cod_available"])
        self.assertTrue(result["upi_available"])
        self.assertEqual(result["payment_amount"], 557)

    def test_disabled_methods_do_not_count_as_available(self):
        response = {
            "success": True,
            "data": {
                "cod": {"available": False},
                "allMethods": [
                    {"id": "gpay", "displayName": "Google Pay", "groupName": "UPI Apps", "kind": "intent", "enabled": False},
                ],
            },
        }
        with patch("cravr.food_mcp.payments._call_tool", return_value=response):
            result = payments.get_payment_options("addr1")
        self.assertFalse(result["cod_available"])
        self.assertFalse(result["upi_available"])

    def test_explicit_failure_is_surfaced_as_an_error(self):
        with patch("cravr.food_mcp.payments._call_tool",
                   return_value={"success": False, "error": {"message": "no address"}}):
            result = payments.get_payment_options("addr1")
        self.assertEqual(result, {"status": "error", "message": "no address"})

    def test_mcp_exception_is_caught_not_raised(self):
        with patch("cravr.food_mcp.payments._call_tool", side_effect=RuntimeError("boom")):
            result = payments.get_payment_options("addr1")
        self.assertEqual(result["status"], "error")
        self.assertIn("boom", result["message"])


class PlaceOrderTests(unittest.TestCase):
    def test_cash_response_is_immediate_success(self):
        response = {
            "success": True,
            "data": {
                "orderId": "ORD1", "status": "CONFIRMED", "normalizedStatus": "success",
                "totalAmount": 557, "estimatedDelivery": "35-40 mins",
                "restaurantName": "Dakshin Delights",
            },
        }
        with patch("cravr.food_mcp.payments._call_tool", return_value=response) as mock_call:
            result = payments.place_order("addr1", "Cash")
        mock_call.assert_called_once_with(
            "place_food_order", {"addressId": "addr1", "paymentMethod": "Cash"}, log_request=True,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["order_id"], "ORD1")
        self.assertEqual(result["total_amount"], 557)
        self.assertNotIn("paas_id", result)

    def test_upi_response_is_pending_with_polling_fields(self):
        response = {
            "success": True,
            "data": {
                "orderId": "ORD2", "paasId": "PAAS1", "transactionId": "TXN1",
                "status": "PENDING_PAYMENT", "normalizedStatus": "pending",
                "upiIntentUrl": "upi://pay", "bridgeUrl": "https://pay.example/x",
                "isQrFlow": True, "pollingIntervalInMs": 19000, "maxTimeToPollForInMs": 120000,
                "lat": 12.9, "lng": 77.6, "addressId": "addr1", "cartId": "cart1",
            },
        }
        with patch("cravr.food_mcp.payments._call_tool", return_value=response) as mock_call:
            result = payments.place_order("addr1", "UPI", generate_upi_qr=True)
        mock_call.assert_called_once_with(
            "place_food_order",
            {"addressId": "addr1", "paymentMethod": "UPI", "generateUPIQR": True},
            log_request=True,
        )
        self.assertEqual(result["status"], "pending_payment")
        self.assertEqual(result["paas_id"], "PAAS1")
        self.assertEqual(result["bridge_url"], "https://pay.example/x")
        self.assertEqual(result["polling_interval_ms"], 19000)
        self.assertEqual(result["max_poll_time_ms"], 120000)
        self.assertEqual(result["lat"], 12.9)
        self.assertEqual(result["lng"], 77.6)

    def test_explicit_failure_is_surfaced_as_an_error(self):
        with patch("cravr.food_mcp.payments._call_tool",
                   return_value={"success": False, "error": {"message": "restaurant closed"}}):
            result = payments.place_order("addr1", "Cash")
        self.assertEqual(result, {"status": "error", "message": "restaurant closed"})

    def test_intent_app_and_note_are_passed_through_when_given(self):
        with patch("cravr.food_mcp.payments._call_tool",
                   return_value={"success": True, "data": {"orderId": "O", "status": "CONFIRMED"}}) as mock_call:
            payments.place_order("addr1", "UPI", intent_app="phonepe", note_to_restaurant="no onions")
        args = mock_call.call_args[0][1]
        self.assertEqual(args["intentApp"], "phonepe")
        self.assertEqual(args["noteToRestaurant"], "no onions")


class CheckPaymentStatusTests(unittest.TestCase):
    def test_terminal_success_is_parsed(self):
        response = {
            "success": True,
            "data": {
                "paasId": "PAAS1", "status": "success", "terminal": True,
                "isTerminalSuccess": True, "isTerminalFailure": False,
                "confirmed": True, "orderStatus": "CONFIRMED",
            },
        }
        with patch("cravr.food_mcp.payments._call_tool", return_value=response) as mock_call:
            result = payments.check_payment_status("PAAS1", order_id="ORD1", address_id="addr1", lat=1.0, lng=2.0)
        mock_call.assert_called_once_with(
            "check_payment_status",
            {"paasId": "PAAS1", "orderId": "ORD1", "addressId": "addr1", "lat": 1.0, "lng": 2.0},
            log_request=True,
        )
        self.assertTrue(result["terminal"])
        self.assertTrue(result["is_terminal_success"])
        self.assertEqual(result["order_status"], "CONFIRMED")

    def test_pending_is_not_terminal(self):
        response = {"success": True, "data": {"status": "pending", "terminal": False}}
        with patch("cravr.food_mcp.payments._call_tool", return_value=response):
            result = payments.check_payment_status("PAAS1")
        self.assertFalse(result["terminal"])
        self.assertEqual(result["status"], "pending")

    def test_mcp_exception_is_treated_as_a_terminal_failure_so_polling_stops(self):
        with patch("cravr.food_mcp.payments._call_tool", side_effect=RuntimeError("timeout")):
            result = payments.check_payment_status("PAAS1")
        self.assertTrue(result["terminal"])
        self.assertTrue(result["is_terminal_failure"])


class ConfirmOrderTests(unittest.TestCase):
    def test_success_result_field_is_parsed(self):
        response = {"success": True, "data": {"orderId": "ORD1", "orderStatus": "CONFIRMED", "result": "success"}}
        with patch("cravr.food_mcp.payments._call_tool", return_value=response) as mock_call:
            result = payments.confirm_order("ORD1", "addr1", 12.9, 77.6, cart_id="cart1")
        mock_call.assert_called_once_with(
            "confirm_order",
            {"orderId": "ORD1", "addressId": "addr1", "lat": 12.9, "lng": 77.6, "cartId": "cart1"},
            log_request=True,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["order_status"], "CONFIRMED")

    def test_food_orders_never_send_a_paas_id(self):
        response = {"success": True, "data": {"orderId": "ORD1", "result": "success"}}
        with patch("cravr.food_mcp.payments._call_tool", return_value=response) as mock_call:
            payments.confirm_order("ORD1", "addr1", 12.9, 77.6)
        self.assertNotIn("paasId", mock_call.call_args[0][1])

    def test_explicit_failure_is_surfaced_as_an_error(self):
        with patch("cravr.food_mcp.payments._call_tool",
                   return_value={"success": False, "error": {"message": "order not found"}}):
            result = payments.confirm_order("ORD1", "addr1", 12.9, 77.6)
        self.assertEqual(result, {"status": "error", "message": "order not found"})


if __name__ == "__main__":
    unittest.main()
