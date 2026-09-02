"""Tests for mcp_server/oauth.py -- the Swiggy OAuth callback that has to
exist (and behave sanely) before Swiggy will whitelist its URL, well before
real Swiggy credentials are configured.

`login`/`callback` take a Starlette `Request` and don't need FastMCP or the
HTTP server actually running -- a bare `Request` built from an ASGI-style
scope is enough to exercise them directly, the same way FastMCP's routing
would call them.
"""

import time
import unittest
from unittest.mock import AsyncMock, patch

from starlette.requests import Request

from mcp_server import oauth


def make_request(query_string: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/oauth/callback",
        "query_string": query_string.encode(),
        "headers": [],
    }
    return Request(scope)


class LoginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        oauth._PENDING.clear()

    async def test_not_configured_returns_503_without_touching_swiggy(self):
        with patch.dict("os.environ", {}, clear=True):
            response = await oauth.login(make_request())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(oauth._PENDING, {})

    async def test_configured_redirects_to_the_authorize_url_with_pkce_params(self):
        env = {
            "SWIGGY_AUTHORIZE_URL": "https://mcp.swiggy.com/oauth/authorize",
            "SWIGGY_CLIENT_ID": "cravr-client",
        }
        with patch.dict("os.environ", env, clear=True):
            response = await oauth.login(make_request())

        self.assertEqual(response.status_code, 307)
        location = response.headers["location"]
        self.assertTrue(location.startswith("https://mcp.swiggy.com/oauth/authorize?"))
        self.assertIn("client_id=cravr-client", location)
        self.assertIn("code_challenge_method=S256", location)
        self.assertIn("response_type=code", location)
        # Exactly one pending PKCE verifier was stashed, keyed by the state
        # just put on the redirect -- callback() needs to find it later.
        self.assertEqual(len(oauth._PENDING), 1)
        (state, entry) = next(iter(oauth._PENDING.items()))
        self.assertIn(f"state={state}", location)
        self.assertTrue(entry["verifier"])


class CallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        oauth._PENDING.clear()
        patcher = patch.object(oauth, "_save_tokens")
        self.save_tokens = patcher.start()
        self.addCleanup(patcher.stop)

    async def test_swiggy_reported_error_is_surfaced_without_a_token_exchange(self):
        response = await oauth.callback(make_request("error=access_denied"))

        self.assertEqual(response.status_code, 400)
        self.save_tokens.assert_not_called()

    async def test_missing_code_or_state_is_rejected(self):
        response = await oauth.callback(make_request("code=abc123"))  # no state

        self.assertEqual(response.status_code, 400)
        self.save_tokens.assert_not_called()

    async def test_unknown_or_expired_state_is_rejected(self):
        """Guards against a forged/replayed callback -- state must match a
        login this server actually started."""
        response = await oauth.callback(make_request("code=abc123&state=never-issued"))

        self.assertEqual(response.status_code, 400)
        self.save_tokens.assert_not_called()

    async def test_not_configured_leaves_the_code_unexchanged(self):
        """A pending state exists (login happened) but SWIGGY_TOKEN_URL /
        SWIGGY_CLIENT_ID aren't set yet -- must not crash, and must not
        silently drop the pending PKCE entry as if it had been used."""
        oauth._PENDING["state-1"] = {"verifier": "v", "created_at": time.time()}

        with patch.dict("os.environ", {}, clear=True):
            response = await oauth.callback(make_request("code=abc123&state=state-1"))

        self.assertEqual(response.status_code, 503)
        self.save_tokens.assert_not_called()

    async def test_successful_exchange_saves_the_token_response(self):
        oauth._PENDING["state-1"] = {"verifier": "the-verifier", "created_at": time.time()}
        env = {
            "SWIGGY_TOKEN_URL": "https://mcp.swiggy.com/oauth/token",
            "SWIGGY_CLIENT_ID": "cravr-client",
        }
        token_payload = {"access_token": "tok", "refresh_token": "ref", "expires_in": 3600}

        fake_response = unittest.mock.Mock(status_code=200)
        fake_response.json.return_value = token_payload
        mock_post = AsyncMock(return_value=fake_response)

        with patch.dict("os.environ", env, clear=True), \
             patch("httpx.AsyncClient.post", mock_post):
            response = await oauth.callback(make_request("code=abc123&state=state-1"))

        self.assertEqual(response.status_code, 200)
        self.save_tokens.assert_called_once_with(token_payload)
        # The verifier that came from *this* login, not a fresh one, is what
        # must be sent back -- that's the whole point of PKCE.
        sent_data = mock_post.call_args.kwargs["data"]
        self.assertEqual(sent_data["code_verifier"], "the-verifier")
        self.assertEqual(sent_data["code"], "abc123")
        # Consumed -- a replayed callback with the same state must fail.
        self.assertNotIn("state-1", oauth._PENDING)

    async def test_upstream_rejection_is_surfaced_and_nothing_is_saved(self):
        oauth._PENDING["state-1"] = {"verifier": "v", "created_at": time.time()}
        env = {
            "SWIGGY_TOKEN_URL": "https://mcp.swiggy.com/oauth/token",
            "SWIGGY_CLIENT_ID": "cravr-client",
        }
        fake_response = unittest.mock.Mock(status_code=400, text="invalid_grant")
        mock_post = AsyncMock(return_value=fake_response)

        with patch.dict("os.environ", env, clear=True), \
             patch("httpx.AsyncClient.post", mock_post):
            response = await oauth.callback(make_request("code=abc123&state=state-1"))

        self.assertEqual(response.status_code, 502)
        self.save_tokens.assert_not_called()


class RedirectUriTests(unittest.TestCase):
    def test_defaults_to_the_render_url(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                oauth.redirect_uri(), "https://cravr-swiggy-mcp.onrender.com/oauth/callback"
            )

    def test_env_override_wins(self):
        with patch.dict("os.environ", {"SWIGGY_REDIRECT_URI": "https://example.com/cb"}, clear=True):
            self.assertEqual(oauth.redirect_uri(), "https://example.com/cb")


class TokenStorageRoundTripTests(unittest.TestCase):
    def test_save_then_load_round_trips(self):
        with patch.object(oauth, "TOKENS_PATH", oauth.TOKENS_PATH.parent / "swiggy_tokens.test.json") as path:
            self.addCleanup(lambda: path.unlink(missing_ok=True))
            oauth._save_tokens({"access_token": "tok"})
            loaded = oauth.load_tokens()

        self.assertEqual(loaded["access_token"], "tok")
        self.assertIn("obtained_at", loaded)


if __name__ == "__main__":
    unittest.main()
