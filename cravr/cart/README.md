# `cravr/cart/` — cart, add-ons, coupons, checkout

Everything after "I want this dish": adding it to the Swiggy cart (including required variants and add-ons), showing the cart, coupons, "what next" flows, and checkout with Cash on Delivery or UPI.

Swiggy carts hold items from **one restaurant only**, which drives several of the guards below.

## Entry points

### `node.py`
| Function | Usage |
|---|---|
| `cart_agent_node(state)` | **Graph node `cart_agent`.** Needs `context_profile.address_id` (otherwise asks the user to order something first). Classifies the message with `_classify_cart_action` and dispatches: `clear`, `apply_coupon`, `coupons`, `add`, `suggest_more`, `checkout`, `upi_paid`, `upi_not_paid`, `start_over`, `clear_and_add`, `keep_cart`, or (default) `view`. |

### `CartAgent.py`
A facade that re-exports the public names (`cart_agent_node`, `is_cart_request`, `add_dish_to_cart`, the `CART_*_VALUE` button values, and so on) so other code and tests can import from `cravr.cart.CartAgent`.

### `__init__.py`
Re-exports `cart_agent_node` and `is_cart_request`.

## `routing.py` — which cart action?
Keyword/regex routing, no LLM call.

| Name | Usage |
|---|---|
| `is_cart_request(text)` | Used by `graph._route_entry`. `True` for messages with a clear cart/coupon/checkout signal, and for the cart's own button values. |
| `_classify_cart_action(text)` | Returns `(action, arg)`. Button values map directly (e.g. `CART_PROCEED_VALUE` → `checkout`). Otherwise regexes decide: clear, apply coupon (arg = code), add (arg = quantity), list coupons, view. |
| `_log_action(action, arg)` | Debug log of the chosen action. |
| `CART_ADD_MORE_VALUE`, `CART_PROCEED_VALUE`, `CART_START_OVER_VALUE`, `CART_CLEAR_AND_ADD_VALUE`, `CART_KEEP_VALUE`, `CART_UPI_PAID_VALUE`, `CART_UPI_NOT_PAID_VALUE` | Hidden values behind the cart's buttons. |
| `_NEXT_STEP_OPTIONS`, `_CONFLICT_OPTIONS`, `_UPI_CONFIRM_OPTIONS` | Button sets: "Add more / Proceed / Browse other restaurants", "Clear cart and add / Keep cart", "I've paid / Not yet". |

## `add_flow.py` — adding a dish
The **single** place a dish gets added, used by both the cart agent's `add` action and the recommender's pick.

| Function | Usage |
|---|---|
| `add_dish_to_cart(dish, address_id, quantity=1, lead_in="", known_restaurant_id=None, known_restaurant_name=None, ...)` | Main entry. Checks the restaurant conflict (→ `_cart_conflict_reply`). If the dish needs customisation, looks up its variants/add-ons and asks the user. Merges it with what's already in the cart, calls `food_mcp.update_cart` (retrying other request shapes if Swiggy rejects them), verifies the add, then shows the cart with next-step buttons. |
| `_fetch_customization(item_name, menu_item_id, address_id, restaurant_id)` | Looks up the item's `addon_groups` and `variant_groups` via a targeted `search_menu_items`. |
| `_ask_for_all_addons(item_name, addon_groups, field_prefix="addon_groups")` | Asks **one** multi-select question covering every group (`interrupt(..., multi_select=True)`), then re-asks only for required groups still unanswered (up to twice). Returns the selected choices. |
| `_addon_question_and_options(item_name, choices, required_only=False)` | Builds that question text and the grouped options (group name, required flag, price). |
| `_match_addon_reply(raw_answer, indexed_choices)` | Resolves a reply (choice ids from buttons, numbers, or names; comma/"and" separated) to choices. |
| `_required_groups` / `_unmet_required_groups` | Groups Swiggy marks mandatory (`min_addons >= 1`), and which still lack enough picks. |
| `_enforce_group_limits(selected, indexed_choices)` | Drops picks beyond each group's `max_addons` (first mentioned wins). |
| `_add_item_with_variant(...)` | For items with sizes/variants: asks for the variant, then tries each `_variant_shapes` payload until one is accepted. |
| `_variant_shapes(variant_picks)` | Candidate request formats for variants (`variantsV2`, legacy `variants`, camelCase). |
| `_addon_groups_matching_variant(addon_groups, variant_groups, chosen)` | Hides add-on groups that belong to a variant the user didn't pick. |
| `_retry_addons_via_valid_addons(...)` | Swiggy's documented fallback when add-ons are rejected: add the item alone, read its `valid_addons` from the cart, resend using those. |
| `_map_to_valid_addons(selected, valid_groups)` | Pairs each pick with Swiggy's own `valid_addons` entry. |
| `_addon_shapes(mapped)` / `_wire_addon(addon)` | Build the add-on payload in the formats Swiggy accepts (`choice_id` + `group_id`, ...). |
| `_cart_conflict_reply(dish, quantity, existing_name, addons)` | "Your cart has items from another restaurant" message with Clear-and-add / Keep buttons. The pending dish is parked in `cart_state`. |

## `guards.py`
| Function | Usage |
|---|---|
| `_requires_customization(tags)` | `True` if the menu flagged the item "has addons" / "has variants". |
| `_customization_refusal(item_name)` | The standard message when an item's required choices can't be set. |
| `_suggest_complementary_items(menu_items, cart_item_id, exclude_ids, limit=5)` | Items from the same menu that pair with what's in the cart (e.g. drinks, desserts and sides after a main, following `_COMPLEMENT_ORDER`), skipping items that need customisation. |

## `flows.py` — "what next" branches
| Function | Usage |
|---|---|
| `_suggest_more_from_restaurant(state, address_id)` | "Add more from this restaurant": fetches that restaurant's menu and offers complementary items. |
| `_start_over_new_restaurant(state, address_id)` | "Browse other restaurants": clears the cart and re-runs the search excluding the current restaurant. |
| `_clear_cart_and_add_pending(state, address_id)` | "Clear my cart and add this": flushes the cart, then adds the parked dish with its already chosen add-ons. |
| `_offer_dishes(lead_in, batch, filters, cart_state=None)` | Presents dishes through the recommender's own pick flow (same buttons and handling). |

## `checkout.py` — ordering and payment
| Function | Usage |
|---|---|
| `run_checkout(address_id)` | Loads the cart, address, item details and payment options; shows an order summary and asks how to pay (COD / UPI / Clear cart / Cancel) via `interrupt()`. Then runs a test-mode or real order. |
| `_order_summary(cart, address, item_details=None)` | Markdown summary: items, total, delivery address. |
| `_payment_method_options(payment_options)` | Payment buttons based on what Swiggy offers. |
| `_run_real_order(choice, address_id)` | `food_mcp.place_order`. COD completes immediately; UPI returns a pay link plus "I've paid" buttons and parks the order in `cart_state.pending_upi_order`. |
| `_run_test_mode_order(choice, cart)` | With `CRAVR_TEST_MODE` on, fakes a confirmed order (nothing sent to Swiggy). |
| `resume_upi_payment(cart_state, address_id)` | "I've paid": polls payment status, then finishes the order. |
| `cancel_upi_payment(cart_state)` | "Not yet": leaves the order unconfirmed. |
| `_poll_payment_status(order, address_id)` | Polls `check_payment_status` until terminal or the time limit (max 3 min). |
| `_finish_upi_order(order, status, address_id)` | Confirms the order on success; explains cancelled / failed / refund / cart-changed / still-pending cases. |
| `_order_placed_reply(order, method_label, confirm=None)` | "Order placed!" with ID, total, ETA and status. |
| `_clear_cart_at_checkout()` | The "Clear my cart" payment option. |

## `display.py` — showing the cart
| Function | Usage |
|---|---|
| `_cart_reply(content, **cart_state)` | Node return shape: one `cart_agent` message plus `cart_state`. |
| `_item_lines(items, item_details=None)` | One line per cart item (name × qty — ₹total, plus chosen variants/add-ons). |
| `_format_cart(cart, item_details=None)` | Full cart text: items, total to pay, coupon. |
| `_show_cart(address_id, lead_in="")` | Fetches the cart and replies with it. |
| `_show_cart_with_next_steps(address_id, lead_in="")` | Same, plus the "Add more / Proceed / Browse other restaurants" buttons when the cart isn't empty. |

## `parsing.py`
| Function | Usage |
|---|---|
| `_parse_existing_cart_items(cart_message)` | Pulls `{menu_item_id, quantity}` pairs out of Swiggy's plain-text cart response. |
| `_existing_cart_items(cart)` | Current cart items (structured data first, text fallback), used to merge a new add with what's there. |
| `_menu_bucket(category_text)` | Coarse category: `main` / `side` / `drink` / `dessert`. |
| `_format_cart(cart)` | Older plain cart formatter. Not called anywhere (`display._format_cart` is used instead), so it can be removed. |

## `restaurant_context.py`
| Function | Usage |
|---|---|
| `_current_restaurant_id(state)` | The restaurant the cart is pinned to (cart first, then the just-picked dish). |
| `_current_restaurant_name(state)` | The same, for the name. |
