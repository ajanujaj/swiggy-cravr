from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

app = FastMCP("recommendation-server")

DATA_DIR = Path(__file__).resolve().parent / "data"
recommendations_data = pd.read_csv(DATA_DIR / "menu.csv")
order_history_data = pd.read_csv(DATA_DIR / "order_history.csv")


@app.tool()
def recommendation(
    cuisine: str = None,
    restaurant_name: str = None,
    tags: str = None,
    dietary_type: str = None,
    meal_type: str = None,
    max_price_inr: int = None,
) -> dict:
    """
    Get restaurant recommendations based on user inputs.
    Can accept cuisine, restaurant_name, or tags individually or in combination.
    
    Args:
        cuisine: Type of cuisine (e.g., 'Italian', 'Chinese', 'Indian')
        restaurant_name: Name of the restaurant
        tags: Food tags (e.g., 'pasta', 'pizza', 'noodles')
        dietary_type: Dietary type (for example, 'Veg' or 'Non-Veg')
        meal_type: Meal time (for example, 'Breakfast' or 'All Day')
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
        filtered_df = filtered_df[filtered_df['tags'].str.lower().str.contains(tags.lower(), na=False)]

    if dietary_type:
        filtered_df = filtered_df[
            filtered_df['dietary_type'].str.lower().str.contains(dietary_type.lower(), na=False)
        ]

    if meal_type:
        filtered_df = filtered_df[
            filtered_df['meal_type'].str.lower().str.contains(meal_type.lower(), na=False)
        ]

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

    orders = user_orders[
        ["restaurant_name", "cuisine", "item_name", "meal_type", "tags"]
    ].to_dict("records")

    return {
        "status": "success",
        "count": len(orders),
        "orders": orders
    }

if __name__ == "__main__":
    app.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/mcp",
    )
