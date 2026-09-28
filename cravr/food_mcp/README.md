# `cravr/food_mcp/` — Swiggy Food MCP adapter

Maps Cravr's needs onto Swiggy's real Food MCP tools. All calls are account-scoped through the OAuth token (no `user_id`). Agents use these functions only when `client.using_real_food_mcp()` is `True`; with the mock server they call mock tools instead.

Swiggy's docs describe most tools as returning JSON, but many actually return **plain text** (sometimes with a UI disclaimer appended). So each function parses defensively and returns a stable dict with a `status` key (`"success"`, `"empty"`, `"text"` or `"error"`) instead of raising.

| Swiggy tool | Wrapped by |
|---|---|
| `get_addresses` | `addresses.get_saved_address` |
| `search_menu`, `search_restaurants` | `search.search_menu_items`, `search.search_dishes` |
| `get_restaurant_menu` | `menu.get_restaurant_menu_items` |
| `get_food_orders` | `orders.get_order_history` |
| `get_food_cart`, `update_food_cart`, `flush_food_cart` | `cart.get_cart`, `cart.get_cart_item_details`, `cart.update_cart`, `cart.flush_cart` |
| `apply_food_coupon`, `fetch_food_coupons` | `cart.apply_coupon`, `cart.fetch_coupons` |
| `get_payment_options`, `place_food_order`, `check_payment_status`, `confirm_order` | `payments.*` |

## `__init__.py`
| Name | Usage |
|---|---|
| `_call_tool(name, arguments, *, log_request=False, context="", return_meta=False)` | The one place every submodule calls a Swiggy tool. Uses `client.call_tool_sync`, unwraps the result with `_shared._unwrap` and logs the raw response (`CRAVR_DEBUG`). With `return_meta=True` also returns `structured_content` / `data` / `is_error`. |
| Re-exports | All public functions below (import as `from cravr import food_mcp; food_mcp.get_cart(...)`). |

## `_shared.py`
| Function | Usage |
|---|---|
| `_first(d, *keys, default=None)` | First non-empty value among several possible key spellings (`"restaurantId"`, `"id"`, ...). |
| `_unwrap(result)` | Extracts text from an MCP result's `content` parts and parses a leading JSON value; dicts/lists pass through. |
| `_parse_leading_json(text)` | Parses JSON at the start of `text`, ignoring trailing text. Returns the raw text if it isn't JSON. |

## `addresses.py`
| Function | Usage |
|---|---|
| `get_saved_address()` | Parses the `get_addresses` text block and picks the suggested/pre-selected address. Returns `{"status", "location", "delivery_area", "address_id"}`. |

## `search.py`
| Function | Usage |
|---|---|
| `search_dishes(filters, address_id, limit_restaurants=5)` | Main dish search used by the recommender. If a specific craving is known → `search_menu_items`. Otherwise `search_restaurants` → `get_restaurant_menu` for the top restaurants → flatten → `_apply_filters` / `_matches_craving`. Returns a list of dish dicts (`item_name`, `menu_item_id`, `restaurant_name`, `restaurant_id`, `cuisine`, `price_inr`, `rating`, `tags`, `dietary_type`, ...). |
| `search_menu_items(query, address_id, veg_only=False, offset=0, restaurant_id=None)` | `search_menu` by dish name. With `restaurant_id` it's a targeted lookup that also returns `detailed_items` (per-item add-on and variant groups), used by the cart's add-on flow. Returns `{"items", "has_more", "next_offset"}`, plus `detailed_items` on a targeted lookup. |
| `_parse_search_menu_items(text)` | Parses the numbered plain-text search results. |
| `_parse_detailed_search_menu_items(text)` | Parses the targeted-lookup text into `{menu_item_id: {"item_name", "price_inr", "addon_groups", "variant_groups"}}`. |
| `_attach_addon_limits(detailed, structured)` | Copies `minAddons` / `maxAddons` from `structured_content` onto those groups (the text has no selection rules). |
| `_infer_collection(filters)` | Optional `collection` for `search_restaurants`: `EATRIGHT` for healthy tags, `STORE_99` for budgets of ₹99 or less. |
| `_apply_filters(dishes, filters)` | Local filtering by dietary type, budget, tags, etc. (Swiggy doesn't filter by these). |
| `_matches_craving(item_name, craving)` | Loose name match so a restaurant's whole menu isn't returned for a specific craving. |

## `menu.py`
| Function | Usage |
|---|---|
| `get_restaurant_menu_items(restaurant_id, address_id, restaurant_name="")` | Full menu of one known restaurant (used by "add more from this restaurant"). |
| `_parse_menu_items(text, restaurant_name, restaurant_cuisine, restaurant_id=None)` | Parses menu text into dish records; each item gets its category from the nearest `##` / `###` header, and "has addons/variants" flags go into `tags`. |
| `_normalize_item(item, ...)` | Same, for a JSON-shaped menu item (fallback path). |

## `orders.py`
| Function | Usage |
|---|---|
| `get_order_history(address_id=None)` | `get_food_orders` → `{"status", "count", "orders"}`. Resolves the address itself if none is given. |

## `cart.py`
| Function | Usage |
|---|---|
| `get_cart(address_id, restaurant_name=None)` | `get_food_cart` → `{"status", "cart_id", "restaurant", "items", "item_count", "pricing", "coupon_applied", "coupon_discount", "address_id", "message"}`. |
| `get_cart_item_details(address_id)` | Per item in the cart: `{menu_item_id: {"variants", "addons", "valid_addons"}}`, read from `structured_content`. |
| `get_cart_valid_addons(address_id)` | Just the `valid_addons` part of the above. |
| `update_cart(restaurant_id, address_id, cart_items, restaurant_name=None)` | `update_food_cart` with the **full** item list (`menu_item_id`, `quantity`, optional `addons` / variants). Returns `{"status", "message"}` (plus `error_codes` on failure). Follow with `get_cart` to verify. |
| `apply_coupon(coupon_code, address_id, cart_id=None)` | `apply_food_coupon` → `{"status", "message"}`. |
| `fetch_coupons(restaurant_id, address_id, coupon_code=None)` | `fetch_food_coupons` → `{"status", "coupons", "count", "message"}`. |
| `flush_cart()` | `flush_food_cart`: empties the cart. |
| `_check_success(raw)` | Error message if `raw` is the `{"success": false, "error": ...}` shape, else `None`. |
| `_strip_widget_disclaimer(text)` | Removes the UI-widget disclaimer Swiggy appends to cart text. |
| `_collect_cart_items(node, out)` | Finds cart-item dicts anywhere in a `structured_content` tree. |

## `payments.py`
| Function | Usage |
|---|---|
| `get_payment_options(address_id)` | → `{"status", "cod_available", "upi_available", "payment_amount", ...}`. |
| `place_order(address_id, payment_method, *, generate_upi_qr=False, intent_app=None, note_to_restaurant=None)` | `place_food_order`. COD returns a placed order; UPI returns the order plus a `bridge_url` pay link, `paas_id` and polling hints. |
| `check_payment_status(paas_id, order_id=None, address_id=None, lat=None, lng=None, cart_id=None)` | → `{"status", "terminal", "is_terminal_success", "is_terminal_failure", "order_status", ...}`. |
| `confirm_order(order_id, address_id, lat, lng, cart_id=None)` | `confirm_order` after a successful UPI payment. |
| `_check_success(raw)` | Same helper as in `cart.py`. |
