"""Regression tests for cravr/food_mcp.py's search_menu response parsers.

Both fixtures below are REAL captured text from a live Swiggy MCP
search_menu call (see logger.txt), not hand-written samples -- per this
project's standing rule of grounding fixes in real evidence rather than
assumptions about the API shape.
"""
import unittest

from cravr.food_mcp import _parse_detailed_search_menu_items, _parse_search_menu_items, search_menu_items


# Captured live: targeted query (item name + restaurantIdOfAddedItem=300897,
# Caterspoint) for "Falafel Hummus Pita Pocket". This is the shape docs.md
# calls out as how to "get full customization details" for one dish --
# confirmed live to actually carry real addon group/choice ids, in a text
# shape neither the broad-search parser nor the JSON-shape fallback handles.
DETAILED_RESPONSE_TEXT = 'Found 10 menu items for "Falafel Hummus Pita Pocket":\n1. Falafel Hummus Pita Pocket — ₹249 | Veg | 4.8★ (53) (ID: 186053967)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911647, choice:151484365), Jalapenos ₹30 (group:318911647, choice:151484364), Gherkins ₹30 (group:318911647, choice:151484361), Feta Cheese ₹40 (group:318911647, choice:151484360), Pasmesan Cheese ₹70 (group:318911647, choice:151484363), Sour Cream Cheese ₹50 (group:318911647, choice:151484362), Hass Avocado ₹100 (group:318911647, choice:151484359), Dried Cranberries ₹40 (group:318911647, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378011, choice:133264959), Olive Tapenade Cup ₹80 (group:252378011, choice:133264952), Chipotle Yoghurt ₹50 (group:252378011, choice:133264953), Ranch Yoghurt ₹50 (group:252378011, choice:133264954), Mexican Salsa ₹50 (group:252378011, choice:133264955), House Caesar Yoghurt ₹50 (group:252378011, choice:133264956), Tzatziki Dip ₹50 (group:252378011, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911648, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911648, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911648, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911649, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911649, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911649, choice:144849004)]\n2. Falafel Hummus Power Bowl — ₹329 | Veg | 4.3★ (166) (ID: 145353528)\n   Addons (Veg Extra Protein): [Hass Avocado ₹100 (group:189055155, choice:129846401), Grilled Tofu | Protein (4g) ₹80 (group:189055155, choice:129846400), Falafel (4pc) | Protein (7g) ₹80 (group:189055155, choice:129846399), Smoked Chickpeas | Protein (6g) ₹50 (group:189055155, choice:129846397), Boiled Chana | Protein (5g) ₹30 (group:189055155, choice:135876049), Mexican Beans | Protein (5g) ₹50 (group:189055155, choice:129846396), Sprouts | Protein (3g) ₹30 (group:189055155, choice:135876050), Superfood Quinoa | Protein (2g) ₹50 (group:189055155, choice:129846395), Grilled Cottage Cheese | Protein (9g) ₹80 (group:189055155, choice:129846392)]\n   Addons (Non-veg Extra Protein): [Tossed Chicken | Protein (15g) ₹100 (group:215648565, choice:135876046), Grilled Chicken | Protein (15g) ₹120 (group:215648565, choice:135876047), Chicken Ham Slice | Protein (8g) ₹80 (group:215648565, choice:135876044), Hard-boiled Egg (2pc) | Protein (12g) ₹50 (group:215648565, choice:135876045), Hard-boiled Egg (1pc) | Protein (6g) ₹30 (group:215648565, choice:135876048)]\n   Addons (Extra Nuts & Seeds): [Golden Raisins ₹30 (group:188194625, choice:129031788), Black Raisins ₹30 (group:188194625, choice:129031789), Pumpkin Seeds ₹30 (group:188194625, choice:129031794), Roasted Almond Flakes ₹40 (group:188194625, choice:129031784), Dried Apricots ₹40 (group:188194625, choice:129031782), Dried Cranberries ₹40 (group:188194625, choice:129031783), Candied Walnuts ₹40 (group:188194625, choice:129031785), Flax Seeds ₹30 (group:188194625, choice:134514446), Chia Seeds ₹30 (group:188194625, choice:134514447)]\n   Addons (Extra Topping): [Jalapenos ₹30 (group:188194627, choice:129031790), Black Olives ₹30 (group:188194627, choice:129031781), Sundried Tomatoes ₹70 (group:188194627, choice:129031808), Cherry Tomatoes ₹30 (group:188194627, choice:129031807), Crispy Nachos ₹30 (group:188194627, choice:129031806), Roasted Peanuts ₹30 (group:188194627, choice:129031810), Kale Leaf ₹20 (group:188194627, choice:143600217), Multigrain Crotons ₹30 (group:188194627, choice:129031809)]\n   Addons (Extra Breads): [Brown Buns (2pc) ₹30 (group:188194628, choice:129031811), Beetroot Buns (2pc) ₹30 (group:188194628, choice:129031812), Spinach Buns (2pc) ₹30 (group:188194628, choice:129031813)]\n   Addons (Extra Cheese): [Fresh Cottage Cheese ₹50 (group:188194626, choice:129031793), Parmesan Cheese ₹70 (group:188194626, choice:129031792), Sour Cream Cheese ₹50 (group:188194626, choice:129031791), Feta Cheese ₹40 (group:188194626, choice:135519278)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:224249307, choice:133264959), Olive Tapenade Cup ₹80 (group:224249307, choice:133264952), Chipotle Yoghurt ₹50 (group:224249307, choice:133264953), Ranch Yoghurt ₹50 (group:224249307, choice:133264954), Mexican Salsa ₹50 (group:224249307, choice:133264955), House Caesar Yoghurt ₹50 (group:224249307, choice:133264956), Tzatziki Dip ₹50 (group:224249307, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056428, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056428, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056428, choice:134675974)]\n3. Hummus With Pita Triangles — ₹299 | Veg | 4.2★ (169) (ID: 63799487)\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:198872380, choice:133264959), Olive Tapenade Cup ₹80 (group:198872380, choice:133264952), Chipotle Yoghurt ₹50 (group:198872380, choice:133264953), Ranch Yoghurt ₹50 (group:198872380, choice:133264954), Mexican Salsa ₹50 (group:198872380, choice:133264955), House Caesar Yoghurt ₹50 (group:198872380, choice:133264956), Tzatziki Dip ₹50 (group:198872380, choice:140815073)]\n   Addons (Side Salad): [Greek Fattoush Salad ₹159 (group:198872379, choice:133264957), Garden Falafel Salad ₹159 (group:198872379, choice:133264960), Lebanese Chicken Salad ₹159 (group:198872379, choice:133264961), Chicken Shawarma Salad ₹159 (group:198872379, choice:133264958)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056589, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056589, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056589, choice:134675974)]\n4. Healthy Falafel Hummus Meal — ₹399 | Veg | 4.4★ (64) (ID: 134398009)\n   Addons (Sides): [Classic Hummus ₹0 (group:138043205, choice:100876181), Garlic Yoghurt ₹0 (group:138043205, choice:100876182)]\n   Addons (Choice Of Pita Breads): [White + Brown Pita Bread ₹0 (group:138043204, choice:110570916)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056516, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056516, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056516, choice:134675974)]\n5. Chicken Shawarma Pita Pocket — ₹279 | 4.5★ (15) (ID: 186053971)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911619, choice:151484365), Jalapenos ₹30 (group:318911619, choice:151484364), Gherkins ₹30 (group:318911619, choice:151484361), Feta Cheese ₹40 (group:318911619, choice:151484360), Pasmesan Cheese ₹70 (group:318911619, choice:151484363), Sour Cream Cheese ₹50 (group:318911619, choice:151484362), Hass Avocado ₹100 (group:318911619, choice:151484359), Dried Cranberries ₹40 (group:318911619, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378024, choice:133264959), Olive Tapenade Cup ₹80 (group:252378024, choice:133264952), Chipotle Yoghurt ₹50 (group:252378024, choice:133264953), Ranch Yoghurt ₹50 (group:252378024, choice:133264954), Mexican Salsa ₹50 (group:252378024, choice:133264955), House Caesar Yoghurt ₹50 (group:252378024, choice:133264956), Tzatziki Dip ₹50 (group:252378024, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911620, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911620, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911620, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911621, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911621, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911621, choice:144849004)]\n6. Farm Veggies Pita Pocket — ₹259 | Veg | 4.7★ (6) (ID: 186053968)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911616, choice:151484365), Jalapenos ₹30 (group:318911616, choice:151484364), Gherkins ₹30 (group:318911616, choice:151484361), Feta Cheese ₹40 (group:318911616, choice:151484360), Pasmesan Cheese ₹70 (group:318911616, choice:151484363), Sour Cream Cheese ₹50 (group:318911616, choice:151484362), Hass Avocado ₹100 (group:318911616, choice:151484359), Dried Cranberries ₹40 (group:318911616, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378015, choice:133264959), Olive Tapenade Cup ₹80 (group:252378015, choice:133264952), Chipotle Yoghurt ₹50 (group:252378015, choice:133264953), Ranch Yoghurt ₹50 (group:252378015, choice:133264954), Mexican Salsa ₹50 (group:252378015, choice:133264955), House Caesar Yoghurt ₹50 (group:252378015, choice:133264956), Tzatziki Dip ₹50 (group:252378015, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911617, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911617, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911617, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911618, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911618, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911618, choice:144849004)]\n7. Pita Hummus Veg Meal — ₹379 | Veg | 4.8★ (198) (ID: 60960461)\n   Addons (Sides): [Classic Hummus ₹0 (group:113199429, choice:100876181), Garlic Yoghurt ₹0 (group:113199429, choice:100876182)]\n   Addons (Choice Of Pita Breads): [White + Brown Pita Bread ₹0 (group:130047267, choice:110570916)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056512, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056512, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056512, choice:134675974)]\n8. Hummus Falafel Sandwich — ₹329 | Veg | 3.9★ (13) (ID: 181452593)\n   Addons (Extra Toppings): [Hass Avocado ₹100 (group:250059702, choice:107648884), Cheese Slice ₹30 (group:250059702, choice:100876141), Black Olives ₹30 (group:250059702, choice:100876142), Jalapenos ₹30 (group:250059702, choice:100876143), Gherkins ₹30 (group:250059702, choice:100876144), Kale Leaf ₹20 (group:250059702, choice:143600218)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:250059703, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:250059703, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:250059703, choice:134675974)]\n9. Loaded Chicken Pita Pocket — ₹279 | 5.0★ (4) (ID: 191685302)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911626, choice:151484365), Jalapenos ₹30 (group:318911626, choice:151484364), Gherkins ₹30 (group:318911626, choice:151484361), Feta Cheese ₹40 (group:318911626, choice:151484360), Pasmesan Cheese ₹70 (group:318911626, choice:151484363), Sour Cream Cheese ₹50 (group:318911626, choice:151484362), Hass Avocado ₹100 (group:318911626, choice:151484359), Dried Cranberries ₹40 (group:318911626, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:268066792, choice:133264959), Olive Tapenade Cup ₹80 (group:268066792, choice:133264952), Chipotle Yoghurt ₹50 (group:268066792, choice:133264953), Ranch Yoghurt ₹50 (group:268066792, choice:133264954), Mexican Salsa ₹50 (group:268066792, choice:133264955), House Caesar Yoghurt ₹50 (group:268066792, choice:133264956), Tzatziki Dip ₹50 (group:268066792, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911627, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911627, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911627, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911628, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911628, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911628, choice:144849004)]\n10. Avocado Egg Breakfast Pita Pocket — ₹279 | 4.7★ (3) (ID: 191685301)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911629, choice:151484365), Jalapenos ₹30 (group:318911629, choice:151484364), Gherkins ₹30 (group:318911629, choice:151484361), Feta Cheese ₹40 (group:318911629, choice:151484360), Pasmesan Cheese ₹70 (group:318911629, choice:151484363), Sour Cream Cheese ₹50 (group:318911629, choice:151484362), Hass Avocado ₹100 (group:318911629, choice:151484359), Dried Cranberries ₹40 (group:318911629, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:268066791, choice:133264959), Olive Tapenade Cup ₹80 (group:268066791, choice:133264952), Chipotle Yoghurt ₹50 (group:268066791, choice:133264953), Ranch Yoghurt ₹50 (group:268066791, choice:133264954), Mexican Salsa ₹50 (group:268066791, choice:133264955), House Caesar Yoghurt ₹50 (group:268066791, choice:133264956), Tzatziki Dip ₹50 (group:268066791, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911630, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911630, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911630, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911631, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911631, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911631, choice:144849004)]\n\nSome items have customisation options (variants/add-ons). Briefly ask the user which variant/size they prefer or if they want any add-ons. Keep it concise, e.g. "Chicken Biryani has size options (Reg/Large) — which do you prefer? Any add-ons?"\n\n⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget already displays — instead, provide a brief recommendation or ask what they\'d like to do next.'

# Captured live: broad, unscoped query ("Hummus & pita") -- the OLD/original
# search_menu shape, one line per item, "[has addons]" tag instead of
# per-item addon breakdown. Used to prove the new detailed parser does NOT
# false-positive on this shape.
BROAD_RESPONSE_TEXT = 'Found 10 menu items for "Hummus & pita":\n1. Falafel Hummus Pita Pocket — ₹249 | Veg | 4.8★ | Caterspoint (restaurantId: 300897) (ID: 186053967) [has addons]\n2. Hummus With Pita Bread — ₹210 | Veg | 4.0★ | Shawarmao Na (restaurantId: 1374628) (ID: 200616039) [has addons]\n3. Mini Hummus And Pita — ₹295 | Veg | 4.8★ | Layr\'d (restaurantId: 1096649) (ID: 182906973) [has addons]\n4. Hummus Falafel Pita — ₹399 | Veg | 3.8★ | Nutriobox (restaurantId: 979965) (ID: 155354051) [has addons]\n5. Avocado & Chili Oil Hummus Veg Pita Pocket — ₹369 | Veg | Witchery-The Sandwich Co. (restaurantId: 592168) (ID: 207555927) [has addons]\n6. Hummus With Pita Triangles — ₹299 | Veg | 4.2★ | Caterspoint (restaurantId: 300897) (ID: 63799487) [has addons]\n7. The Hummus & Pita Plate — ₹439 | Veg | Coffee Island (restaurantId: 1024381) (ID: 193115134)\n8. Hummus And Pita Platter — ₹315 | Veg | 4.1★ | Third Wave Coffee (restaurantId: 450915) (ID: 127378953)\n9. Falafel Pita Hummus — ₹379 | Veg | The Big Tree (restaurantId: 1074936) (ID: 169539818)\n10. Hummus, Pita Strips & PERi-PERi Drizzle — ₹399 | Veg | 4.3★ | Nando\'s (restaurantId: 17464) (ID: 90359506)\n\nSome items have customisation options (variants/add-ons). Briefly ask the user which variant/size they prefer or if they want any add-ons. Keep it concise, e.g. "Chicken Biryani has size options (Reg/Large) — which do you prefer? Any add-ons?"\n\n⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget already displays — instead, provide a brief recommendation or ask what they\'d like to do next.'


class ParseDetailedSearchMenuItemsTests(unittest.TestCase):
    def test_parses_every_item_in_the_real_response(self):
        parsed = _parse_detailed_search_menu_items(DETAILED_RESPONSE_TEXT)
        self.assertEqual(
            list(parsed.keys()),
            [
                "186053967", "145353528", "63799487", "134398009", "186053971",
                "186053968", "60960461", "181452593", "191685302", "191685301",
            ],
        )

    def test_target_item_has_its_four_real_addon_groups_with_ids(self):
        parsed = _parse_detailed_search_menu_items(DETAILED_RESPONSE_TEXT)
        item = parsed["186053967"]
        self.assertEqual(item["item_name"], "Falafel Hummus Pita Pocket")
        self.assertEqual(item["price_inr"], 249.0)
        group_names = [g["group_name"] for g in item["addon_groups"]]
        self.assertEqual(
            group_names, ["Extra Toppings", "Dressing Add On's", "Make Your Own Meal", "Desserts"],
        )
        toppings = item["addon_groups"][0]
        self.assertEqual(toppings["group_id"], "318911647")
        self.assertEqual(len(toppings["choices"]), 8)
        black_olives = toppings["choices"][0]
        self.assertEqual(black_olives, {
            "id": "151484365", "name": "Black Olives", "price_inr": 30.0, "group_id": "318911647",
        })

    def test_choice_names_are_not_corrupted_by_the_comma_separator(self):
        # Regression: an earlier version left the leading ", " from
        # finditer's separator stuck onto every choice name after the first
        # in a group (e.g. ", Jalapenos" instead of "Jalapenos").
        parsed = _parse_detailed_search_menu_items(DETAILED_RESPONSE_TEXT)
        toppings = parsed["186053967"]["addon_groups"][0]
        names = [c["name"] for c in toppings["choices"]]
        self.assertEqual(
            names,
            [
                "Black Olives", "Jalapenos", "Gherkins", "Feta Cheese", "Pasmesan Cheese",
                "Sour Cream Cheese", "Hass Avocado", "Dried Cranberries",
            ],
        )
        for name in names:
            self.assertFalse(name.startswith(","))
            self.assertFalse(name.startswith(" "))

    def test_choice_names_containing_brackets_parse_correctly(self):
        # "Tropical Fruit Juice [215ml]" -- the choice name itself contains
        # "[...]", which must not be confused with the addon list's own
        # closing "]".
        parsed = _parse_detailed_search_menu_items(DETAILED_RESPONSE_TEXT)
        meal_group = next(
            g for g in parsed["186053967"]["addon_groups"] if g["group_name"] == "Make Your Own Meal"
        )
        names = [c["name"] for c in meal_group["choices"]]
        self.assertIn("Tropical Fruit Juice [215ml]", names)

    def test_every_choice_in_a_group_shares_that_groups_id(self):
        parsed = _parse_detailed_search_menu_items(DETAILED_RESPONSE_TEXT)
        for item in parsed.values():
            for group in item["addon_groups"]:
                if not group["choices"]:
                    continue
                self.assertEqual(group["group_id"], group["choices"][0]["group_id"])
                self.assertTrue(all(c["group_id"] == group["group_id"] for c in group["choices"]))

    def test_does_not_false_positive_on_the_old_broad_search_shape(self):
        self.assertEqual(_parse_detailed_search_menu_items(BROAD_RESPONSE_TEXT), {})

    def test_broad_shape_still_parses_normally_via_the_original_parser(self):
        items = _parse_search_menu_items(BROAD_RESPONSE_TEXT)
        self.assertTrue(items)
        self.assertTrue(any(i.get("menu_item_id") == "186053967" for i in items))


class SearchMenuItemsDetailedFallbackTests(unittest.TestCase):
    """search_menu_items() itself should surface the detailed shape under
    result["detailed_items"] only when the broad-format parser found
    nothing -- never both, never neither when one of the two actually
    matches."""

    def test_targeted_call_surfaces_detailed_items_when_broad_parse_is_empty(self):
        from unittest.mock import patch

        with patch("cravr.food_mcp.call_tool_sync", return_value=DETAILED_RESPONSE_TEXT):
            result = search_menu_items("Falafel Hummus Pita Pocket", "addr1", restaurant_id="300897")

        self.assertEqual(result["items"], [])
        self.assertIn("detailed_items", result)
        self.assertIn("186053967", result["detailed_items"])

    def test_broad_call_does_not_populate_detailed_items(self):
        from unittest.mock import patch

        with patch("cravr.food_mcp.call_tool_sync", return_value=BROAD_RESPONSE_TEXT):
            result = search_menu_items("Hummus & pita", "addr1")

        self.assertTrue(result["items"])
        self.assertNotIn("detailed_items", result)


if __name__ == "__main__":
    unittest.main()
