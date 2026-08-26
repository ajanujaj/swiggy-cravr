from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

app = FastMCP("recommendation-server")

DATA_DIR = Path(__file__).resolve().parent / "data"
recommendations_data = pd.read_csv(DATA_DIR / "menu.csv")
order_history_data = pd.read_csv(DATA_DIR / "order_history.csv")
user_preferences_data = pd.read_csv(DATA_DIR / "user_preferences.csv")

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
        # Support comma-separated tags with AND semantics (a dish must carry
        # every requested tag), while staying backward-compatible with a
        # single free-text tag string (e.g. a craving phrase).
        for tag in [t.strip() for t in tags.split(",") if t.strip()]:
            filtered_df = filtered_df[filtered_df['tags'].str.lower().str.contains(tag.lower(), na=False)]

    if dietary_type:
        # Exact match, not substring: dietary_type is a closed set ('Veg' /
        # 'Non-Veg'), and .str.contains('veg') would also match 'non-veg'
        # (it contains the substring 'veg'), silently letting a vegetarian
        # request return meat dishes.
        filtered_df = filtered_df[filtered_df['dietary_type'].str.lower() == dietary_type.lower()]

    if meal_type:
        filtered_df = filtered_df[
            filtered_df['meal_type'].str.lower().str.contains(meal_type.lower(), na=False)
        ]

    # `course` is OR semantics (unlike tags): "main,side" means either is
    # acceptable, because a caller widening the course is loosening a
    # constraint, not adding one. Guarded on column presence so older menu
    # files without the column keep working instead of raising KeyError.
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

    user_prefs = user_preferences_data[user_preferences_data['user_id'] == user_id]

    if user_prefs.empty:
        return {"status": "no_results", "message": "No user preferences found for the given user ID"}

    preferences = user_prefs[
        ["default_budget_inr", "dietary_preference","allergies"]
    ].to_dict("records")

    return {
        "preferences": preferences
    }


if __name__ == "__main__":
    app.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
    )
