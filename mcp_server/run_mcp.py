import os
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

# Mock-data tool server only -- no OAuth/LLM calls. Real Swiggy OAuth lives in
# mcp_server/oauth.py/server.py, so switching to real MCP access is just
# deleting this dir and pointing CRAVR_MCP_URL at Swiggy's endpoint.
app = FastMCP("recommendation-server")

DATA_DIR = Path(__file__).resolve().parent / "data"
recommendations_data = pd.read_csv(DATA_DIR / "menu.csv")
order_history_data = pd.read_csv(DATA_DIR / "order_history.csv")
user_data = pd.read_csv(DATA_DIR / "user.csv")

# menu.csv has no restaurant/item ID columns -- synthesize stable ones so mock
# recommendations carry the same "menu_item_id"/"restaurant_id" fields real
# Swiggy dish records do (needed by the cart tools below). Row-index-based,
# so they're stable across calls within one server run but NOT across a CSV
# edit/reorder -- fine for local mock testing, never used against real money.
recommendations_data["menu_item_id"] = "ITEM" + (recommendations_data.index + 1).astype(str).str.zfill(4)
_restaurant_codes, _ = pd.factorize(recommendations_data["restaurant_name"])
recommendations_data["restaurant_id"] = "REST" + pd.Series(_restaurant_codes + 1, index=recommendations_data.index).astype(str).str.zfill(3)

# In-memory mock cart -- single global cart (this test server has no
# per-user auth), mirroring real Swiggy's "one active cart, single
# restaurant" behavior. Reset by flush_food_cart or a server restart.
_cart_state: dict = {
    "restaurant_id": None,
    "restaurant_name": None,
    "items": [],  # [{"menu_item_id","name","quantity","final_price","total","in_stock","is_veg"}]
    "coupon_code": None,
    "coupon_discount": 0,
}
_MOCK_COUPONS = {
    "SAVE20": {"kind": "percent", "value": 0.2, "cap": 100},
    "FREESHIP": {"kind": "delivery", "value": 30},
}

@app.tool()
def recommendation(
    cuisine: str = None,
    restaurant_name: str = None,
    tags: str = None,
    dietary_type: str = None,
    meal_type: str = None,
    course: str = None,
    max_price_inr: int = None,
) -> dict:
    """
    Get restaurant recommendations based on user inputs.
    Can accept cuisine, restaurant_name, or tags individually or in combination.

    Args:
        cuisine: Type of cuisine (e.g., 'Italian', 'Chinese', 'Indian')
        restaurant_name: Name of the restaurant
        tags: Food tags (e.g., 'spicy', 'vegetarian;dessert'). Pass multiple
            tags comma-separated (e.g., 'spicy,gluten-free') to require a dish
            to carry all of them.
        dietary_type: Dietary type (for example, 'Veg' or 'Non-Veg')
        meal_type: Meal time (for example, 'Breakfast' or 'All Day')
        course: What role the item plays in a meal -- 'main' (stands alone),
            'side' (accompaniment, small plate, clear soup), 'dessert', or
            'beverage'. Pass multiple comma-separated (e.g. 'main,side') to
            allow any of them. Filter on 'main' when the user wants a meal,
            so a coffee or a portion of naan is never offered as the answer.
        max_price_inr: Maximum item price in INR

    Returns:
        dict: Filtered recommendations with restaurant details
    """
    
    filtered_df = recommendations_data.copy()
    
    # Apply filters based on provided inputs
    if cuisine:
        filtered_df = filtered_df[filtered_df['cuisine'].str.lower().str.contains(cuisine.lower(), na=False)]
    
    if restaurant_name:
        filtered_df = filtered_df[filtered_df['restaurant_name'].str.lower().str.contains(restaurant_name.lower(), na=False)]
    
    if tags:
        # Comma-separated tags use AND semantics (must carry every tag); a single free-text tag still works.
        for tag in [t.strip() for t in tags.split(",") if t.strip()]:
            filtered_df = filtered_df[filtered_df['tags'].str.lower().str.contains(tag.lower(), na=False)]

    if dietary_type:
        # Exact match, not substring -- .str.contains('veg') would also match 'non-veg'.
        filtered_df = filtered_df[filtered_df['dietary_type'].str.lower() == dietary_type.lower()]

    if meal_type:
        filtered_df = filtered_df[
            filtered_df['meal_type'].str.lower().str.contains(meal_type.lower(), na=False)
        ]

    # `course` is OR semantics (unlike tags): "main,side" means either is acceptable (widening, not narrowing).
    # Guarded on column presence so older menu files without it don't raise KeyError.
    if course and 'course' in filtered_df.columns:
        wanted_courses = {c.strip().lower() for c in course.split(",") if c.strip()}
        if wanted_courses:
            filtered_df = filtered_df[filtered_df['course'].str.lower().isin(wanted_courses)]

    if max_price_inr is not None:
        filtered_df = filtered_df[filtered_df['price_inr'] <= max_price_inr]
    
    # Sort by rating (highest first)
    filtered_df = filtered_df.sort_values('rating', ascending=False)
    
    if filtered_df.empty:
        return {"status": "no_results", "message": "No recommendations found for the given criteria"}
    
    recommendations = filtered_df.to_dict('records')
    
    return {
        "status": "success",
        "count": len(recommendations),
        "recommendations": recommendations
    }

@app.tool()
def order_history(user_id: str) -> dict:
    """
    Get the order history for a specific user.
    
    Args:
        user_id: Unique identifier for the user"""


    user_orders = order_history_data[order_history_data['user_id'] == user_id]

    if user_orders.empty:
        return {"status": "no_results", "message": "No order history found for the given user ID"}

    menu_details = recommendations_data.reset_index(drop=True).copy()
    menu_details = menu_details[
        ["restaurant_name", "cuisine", "item_name", "price_inr", "rating", "tags"]
    ].drop_duplicates(subset=["restaurant_name", "cuisine", "item_name", "tags"])

    user_orders = user_orders.merge(
        menu_details,
        on=["restaurant_name", "cuisine", "item_name"],
        how="left",
    )

    user_orders["ordered_at"] = pd.to_datetime(
        user_orders["order_date"] + " " + user_orders["order_time"],
        errors="coerce",
    )
    user_orders = user_orders.sort_values("ordered_at", ascending=False, na_position="last").head(10)

    orders = user_orders[
        ["restaurant_name", "cuisine", "item_name", "meal_type", "order_date", "order_time", "tags"]
    ].to_dict("records")

    return {
        "status": "success",
        "count": len(orders),
        "orders": orders
    }

@app.tool()
def user_preferences(user_id: str) -> dict:
    """
    Get the user preferences for a specific user.
    
    Args:
        user_id: Unique identifier for the user
    """

    user_prefs = user_data[user_data['user_id'] == user_id]

    if user_prefs.empty:
        return {"status": "no_results", "message": "No user preferences found for the given user ID"}

    preferences = user_prefs[
        [
            "default_budget_inr",
            "dietary_preference",
            "allergies",
            "preferred_cuisines",
            "preferred_flavors",
        ]
    ].iloc[0].to_dict()

    return {
        "preferences": preferences
    }


@app.tool()
def user_location(user_id: str) -> dict:
    """Get the saved city and delivery area for a user.

    This is intentionally separate from ``user_preferences`` so the context
    agent can obtain delivery context without handling dietary or budget data.
    """
    user = user_data[user_data["user_id"] == user_id]
    if user.empty:
        return {"status": "no_results", "message": "No saved location found for the given user ID"}

    location = user[["default_city", "default_delivery_area"]].iloc[0].to_dict()
    return {
        "location": location["default_city"],
        "delivery_area": location["default_delivery_area"],
    }


# ---------------------------------------------------------------------------
# Mock Cart tools -- mirror Swiggy's real get_food_cart/update_food_cart/
# apply_food_coupon/fetch_food_coupons/flush_food_cart tool names and
# {"success", "data", "message"} response envelope (see cravr/food_mcp.py's
# cart functions) so the same code path can be exercised safely here before
# ever touching a real Swiggy account.
# ---------------------------------------------------------------------------

def _cart_pricing() -> dict:
    item_total = sum(i["final_price"] * i["quantity"] for i in _cart_state["items"])
    delivery_charge = 0 if not _cart_state["items"] else 30
    taxes = round(item_total * 0.05, 2) if _cart_state["items"] else 0
    discount = _cart_state.get("coupon_discount", 0)
    to_pay = max(round(item_total + delivery_charge + taxes - discount, 2), 0)
    return {
        "item_total": round(item_total, 2),
        "delivery_charge": delivery_charge,
        "taxes_and_charges": taxes,
        "to_pay": to_pay,
    }


@app.tool()
def get_food_cart(addressId: str, restaurantName: str = None) -> dict:
    """Mock get_food_cart -- returns the current in-memory cart."""
    restaurant = (
        {"id": _cart_state["restaurant_id"], "name": _cart_state["restaurant_name"]}
        if _cart_state["restaurant_id"] else None
    )
    return {
        "success": True,
        "data": {
            "cart_id": "MOCK-CART-1",
            "restaurant": restaurant,
            "items": _cart_state["items"],
            "item_count": len(_cart_state["items"]),
            "pricing": _cart_pricing(),
            "offers": {
                "coupon_applied": _cart_state.get("coupon_code"),
                "coupon_discount": _cart_state.get("coupon_discount", 0),
                "free_delivery_applied": False,
            },
        },
        "addressId": addressId,
    }


@app.tool()
def update_food_cart(restaurantId: str, addressId: str, cartItems: list, restaurantName: str = None) -> dict:
    """Mock update_food_cart. cartItems entries: {"menu_item_id": <id>,
    "quantity": <int>} -- matches real Swiggy's field name (confirmed via
    its docs.md: '"menu_item_id": Food menu item identifier used by
    update_food_cart.'); an earlier version of this mock and of
    cart.CartAgent both used "id" instead, which the real tool silently
    ignores (accepts the call, adds nothing). Silently ignoring an
    unrecognised key here too, rather than falling back to it, so a
    regression back to "id" fails the same way against the mock as it does
    for real. quantity<=0 removes the item. Switching restaurants replaces
    the cart -- real Swiggy carts are single-restaurant too."""
    if _cart_state["restaurant_id"] and _cart_state["restaurant_id"] != restaurantId:
        _cart_state["items"] = []
        _cart_state["coupon_code"] = None
        _cart_state["coupon_discount"] = 0
    _cart_state["restaurant_id"] = restaurantId
    _cart_state["restaurant_name"] = restaurantName or _cart_state["restaurant_name"]

    for entry in cartItems or []:
        item_id = (entry or {}).get("menu_item_id")
        if not item_id:
            continue
        quantity = (entry or {}).get("quantity", 1)
        existing = next((i for i in _cart_state["items"] if i["menu_item_id"] == item_id), None)

        if quantity <= 0:
            if existing:
                _cart_state["items"].remove(existing)
            continue

        if existing:
            existing["quantity"] = quantity
            existing["total"] = round(existing["final_price"] * quantity, 2)
            continue

        row = recommendations_data[recommendations_data["menu_item_id"] == item_id]
        if row.empty:
            return {"success": False, "error": {"message": f"Unknown menu_item_id {item_id!r}"}}
        row0 = row.iloc[0]
        price = float(row0["price_inr"])
        _cart_state["items"].append({
            "menu_item_id": item_id,
            "name": row0["item_name"],
            "quantity": quantity,
            "final_price": price,
            "total": round(price * quantity, 2),
            "in_stock": True,
            "is_veg": str(row0.get("dietary_type", "")).lower() == "veg",
        })

    return {"success": True, "data": {}, "message": "Cart updated"}


@app.tool()
def apply_food_coupon(couponCode: str, addressId: str, cartId: str = None) -> dict:
    """Mock apply_food_coupon -- static SAVE20 (20% off, capped ₹100) /
    FREESHIP (waives delivery) catalog; anything else is rejected."""
    item_total = sum(i["final_price"] * i["quantity"] for i in _cart_state["items"])
    if item_total <= 0:
        return {"success": False, "error": {"message": "Cart is empty -- nothing to apply a coupon to"}}

    code = (couponCode or "").strip().upper()
    coupon = _MOCK_COUPONS.get(code)
    if not coupon:
        return {"success": False, "error": {"message": f"Coupon {couponCode!r} is not valid"}}

    discount = min(round(item_total * coupon["value"], 2), coupon.get("cap", item_total))         if coupon["kind"] == "percent" else coupon["value"]
    _cart_state["coupon_code"] = code
    _cart_state["coupon_discount"] = discount
    return {"success": True, "data": {"coupon_discount": discount}, "message": f"{code} applied: ₹{discount} off"}


@app.tool()
def fetch_food_coupons(restaurantId: str, addressId: str, couponCode: str = None) -> dict:
    """Mock fetch_food_coupons -- static SAVE20/FREESHIP catalog, COD-only
    per the real tool's stated scope."""
    catalog = [
        {
            "id": "SAVE20", "title": "SAVE20", "subtitle": "20% off, up to ₹100",
            "description": "Get 20% off your order", "applicable": True,
            "applicabilityStatus": "APPLICABLE", "terms": "Max discount ₹100. Valid for Cash on Delivery.",
        },
        {
            "id": "FREESHIP", "title": "FREESHIP", "subtitle": "Free delivery",
            "description": "Waives the delivery charge", "applicable": True,
            "applicabilityStatus": "APPLICABLE", "terms": "Valid for Cash on Delivery.",
        },
    ]
    if couponCode:
        catalog = [c for c in catalog if c["id"] == couponCode.strip().upper()]

    return {
        "success": True,
        "data": {
            "status_message": None if catalog else "No matching coupon",
            "coupon_sections": [{"title": "Available offers", "type": "COD", "coupons": catalog}] if catalog else [],
            "summary": {"total": len(catalog), "applicable": len(catalog), "sections": 1 if catalog else 0},
        },
    }


@app.tool()
def flush_food_cart() -> dict:
    """Mock flush_food_cart -- clears the in-memory cart. No parameters."""
    _cart_state["restaurant_id"] = None
    _cart_state["restaurant_name"] = None
    _cart_state["items"] = []
    _cart_state["coupon_code"] = None
    _cart_state["coupon_discount"] = 0
    return {"success": True, "data": {}, "message": "Cart cleared"}



if __name__ == "__main__":
    # $PORT so Render (or any host assigning its own port) can run this unchanged;
    # locally $PORT is unset, so this falls back to 8000 (matches CRAVR_MCP_URL's default).
    app.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        path="/mcp",
    )
