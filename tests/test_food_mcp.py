import unittest

from cravr.food_mcp import (
    _parse_detailed_search_menu_items,
    _parse_search_menu_items,
    search_dishes,
    search_menu_items,
)
from cravr.food_mcp.search import _matches_craving


DETAILED_RESPONSE_TEXT = 'Found 10 menu items for "Falafel Hummus Pita Pocket":\n1. Falafel Hummus Pita Pocket — ₹249 | Veg | 4.8★ (53) (ID: 186053967)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911647, choice:151484365), Jalapenos ₹30 (group:318911647, choice:151484364), Gherkins ₹30 (group:318911647, choice:151484361), Feta Cheese ₹40 (group:318911647, choice:151484360), Pasmesan Cheese ₹70 (group:318911647, choice:151484363), Sour Cream Cheese ₹50 (group:318911647, choice:151484362), Hass Avocado ₹100 (group:318911647, choice:151484359), Dried Cranberries ₹40 (group:318911647, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378011, choice:133264959), Olive Tapenade Cup ₹80 (group:252378011, choice:133264952), Chipotle Yoghurt ₹50 (group:252378011, choice:133264953), Ranch Yoghurt ₹50 (group:252378011, choice:133264954), Mexican Salsa ₹50 (group:252378011, choice:133264955), House Caesar Yoghurt ₹50 (group:252378011, choice:133264956), Tzatziki Dip ₹50 (group:252378011, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911648, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911648, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911648, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911649, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911649, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911649, choice:144849004)]\n2. Falafel Hummus Power Bowl — ₹329 | Veg | 4.3★ (166) (ID: 145353528)\n   Addons (Veg Extra Protein): [Hass Avocado ₹100 (group:189055155, choice:129846401), Grilled Tofu | Protein (4g) ₹80 (group:189055155, choice:129846400), Falafel (4pc) | Protein (7g) ₹80 (group:189055155, choice:129846399), Smoked Chickpeas | Protein (6g) ₹50 (group:189055155, choice:129846397), Boiled Chana | Protein (5g) ₹30 (group:189055155, choice:135876049), Mexican Beans | Protein (5g) ₹50 (group:189055155, choice:129846396), Sprouts | Protein (3g) ₹30 (group:189055155, choice:135876050), Superfood Quinoa | Protein (2g) ₹50 (group:189055155, choice:129846395), Grilled Cottage Cheese | Protein (9g) ₹80 (group:189055155, choice:129846392)]\n   Addons (Non-veg Extra Protein): [Tossed Chicken | Protein (15g) ₹100 (group:215648565, choice:135876046), Grilled Chicken | Protein (15g) ₹120 (group:215648565, choice:135876047), Chicken Ham Slice | Protein (8g) ₹80 (group:215648565, choice:135876044), Hard-boiled Egg (2pc) | Protein (12g) ₹50 (group:215648565, choice:135876045), Hard-boiled Egg (1pc) | Protein (6g) ₹30 (group:215648565, choice:135876048)]\n   Addons (Extra Nuts & Seeds): [Golden Raisins ₹30 (group:188194625, choice:129031788), Black Raisins ₹30 (group:188194625, choice:129031789), Pumpkin Seeds ₹30 (group:188194625, choice:129031794), Roasted Almond Flakes ₹40 (group:188194625, choice:129031784), Dried Apricots ₹40 (group:188194625, choice:129031782), Dried Cranberries ₹40 (group:188194625, choice:129031783), Candied Walnuts ₹40 (group:188194625, choice:129031785), Flax Seeds ₹30 (group:188194625, choice:134514446), Chia Seeds ₹30 (group:188194625, choice:134514447)]\n   Addons (Extra Topping): [Jalapenos ₹30 (group:188194627, choice:129031790), Black Olives ₹30 (group:188194627, choice:129031781), Sundried Tomatoes ₹70 (group:188194627, choice:129031808), Cherry Tomatoes ₹30 (group:188194627, choice:129031807), Crispy Nachos ₹30 (group:188194627, choice:129031806), Roasted Peanuts ₹30 (group:188194627, choice:129031810), Kale Leaf ₹20 (group:188194627, choice:143600217), Multigrain Crotons ₹30 (group:188194627, choice:129031809)]\n   Addons (Extra Breads): [Brown Buns (2pc) ₹30 (group:188194628, choice:129031811), Beetroot Buns (2pc) ₹30 (group:188194628, choice:129031812), Spinach Buns (2pc) ₹30 (group:188194628, choice:129031813)]\n   Addons (Extra Cheese): [Fresh Cottage Cheese ₹50 (group:188194626, choice:129031793), Parmesan Cheese ₹70 (group:188194626, choice:129031792), Sour Cream Cheese ₹50 (group:188194626, choice:129031791), Feta Cheese ₹40 (group:188194626, choice:135519278)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:224249307, choice:133264959), Olive Tapenade Cup ₹80 (group:224249307, choice:133264952), Chipotle Yoghurt ₹50 (group:224249307, choice:133264953), Ranch Yoghurt ₹50 (group:224249307, choice:133264954), Mexican Salsa ₹50 (group:224249307, choice:133264955), House Caesar Yoghurt ₹50 (group:224249307, choice:133264956), Tzatziki Dip ₹50 (group:224249307, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056428, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056428, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056428, choice:134675974)]\n3. Hummus With Pita Triangles — ₹299 | Veg | 4.2★ (169) (ID: 63799487)\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:198872380, choice:133264959), Olive Tapenade Cup ₹80 (group:198872380, choice:133264952), Chipotle Yoghurt ₹50 (group:198872380, choice:133264953), Ranch Yoghurt ₹50 (group:198872380, choice:133264954), Mexican Salsa ₹50 (group:198872380, choice:133264955), House Caesar Yoghurt ₹50 (group:198872380, choice:133264956), Tzatziki Dip ₹50 (group:198872380, choice:140815073)]\n   Addons (Side Salad): [Greek Fattoush Salad ₹159 (group:198872379, choice:133264957), Garden Falafel Salad ₹159 (group:198872379, choice:133264960), Lebanese Chicken Salad ₹159 (group:198872379, choice:133264961), Chicken Shawarma Salad ₹159 (group:198872379, choice:133264958)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056589, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056589, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056589, choice:134675974)]\n4. Healthy Falafel Hummus Meal — ₹399 | Veg | 4.4★ (64) (ID: 134398009)\n   Addons (Sides): [Classic Hummus ₹0 (group:138043205, choice:100876181), Garlic Yoghurt ₹0 (group:138043205, choice:100876182)]\n   Addons (Choice Of Pita Breads): [White + Brown Pita Bread ₹0 (group:138043204, choice:110570916)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056516, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056516, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056516, choice:134675974)]\n5. Chicken Shawarma Pita Pocket — ₹279 | 4.5★ (15) (ID: 186053971)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911619, choice:151484365), Jalapenos ₹30 (group:318911619, choice:151484364), Gherkins ₹30 (group:318911619, choice:151484361), Feta Cheese ₹40 (group:318911619, choice:151484360), Pasmesan Cheese ₹70 (group:318911619, choice:151484363), Sour Cream Cheese ₹50 (group:318911619, choice:151484362), Hass Avocado ₹100 (group:318911619, choice:151484359), Dried Cranberries ₹40 (group:318911619, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378024, choice:133264959), Olive Tapenade Cup ₹80 (group:252378024, choice:133264952), Chipotle Yoghurt ₹50 (group:252378024, choice:133264953), Ranch Yoghurt ₹50 (group:252378024, choice:133264954), Mexican Salsa ₹50 (group:252378024, choice:133264955), House Caesar Yoghurt ₹50 (group:252378024, choice:133264956), Tzatziki Dip ₹50 (group:252378024, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911620, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911620, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911620, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911621, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911621, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911621, choice:144849004)]\n6. Farm Veggies Pita Pocket — ₹259 | Veg | 4.7★ (6) (ID: 186053968)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911616, choice:151484365), Jalapenos ₹30 (group:318911616, choice:151484364), Gherkins ₹30 (group:318911616, choice:151484361), Feta Cheese ₹40 (group:318911616, choice:151484360), Pasmesan Cheese ₹70 (group:318911616, choice:151484363), Sour Cream Cheese ₹50 (group:318911616, choice:151484362), Hass Avocado ₹100 (group:318911616, choice:151484359), Dried Cranberries ₹40 (group:318911616, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:252378015, choice:133264959), Olive Tapenade Cup ₹80 (group:252378015, choice:133264952), Chipotle Yoghurt ₹50 (group:252378015, choice:133264953), Ranch Yoghurt ₹50 (group:252378015, choice:133264954), Mexican Salsa ₹50 (group:252378015, choice:133264955), House Caesar Yoghurt ₹50 (group:252378015, choice:133264956), Tzatziki Dip ₹50 (group:252378015, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911617, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911617, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911617, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911618, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911618, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911618, choice:144849004)]\n7. Pita Hummus Veg Meal — ₹379 | Veg | 4.8★ (198) (ID: 60960461)\n   Addons (Sides): [Classic Hummus ₹0 (group:113199429, choice:100876181), Garlic Yoghurt ₹0 (group:113199429, choice:100876182)]\n   Addons (Choice Of Pita Breads): [White + Brown Pita Bread ₹0 (group:130047267, choice:110570916)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:207056512, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:207056512, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:207056512, choice:134675974)]\n8. Hummus Falafel Sandwich — ₹329 | Veg | 3.9★ (13) (ID: 181452593)\n   Addons (Extra Toppings): [Hass Avocado ₹100 (group:250059702, choice:107648884), Cheese Slice ₹30 (group:250059702, choice:100876141), Black Olives ₹30 (group:250059702, choice:100876142), Jalapenos ₹30 (group:250059702, choice:100876143), Gherkins ₹30 (group:250059702, choice:100876144), Kale Leaf ₹20 (group:250059702, choice:143600218)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:250059703, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:250059703, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:250059703, choice:134675974)]\n9. Loaded Chicken Pita Pocket — ₹279 | 5.0★ (4) (ID: 191685302)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911626, choice:151484365), Jalapenos ₹30 (group:318911626, choice:151484364), Gherkins ₹30 (group:318911626, choice:151484361), Feta Cheese ₹40 (group:318911626, choice:151484360), Pasmesan Cheese ₹70 (group:318911626, choice:151484363), Sour Cream Cheese ₹50 (group:318911626, choice:151484362), Hass Avocado ₹100 (group:318911626, choice:151484359), Dried Cranberries ₹40 (group:318911626, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:268066792, choice:133264959), Olive Tapenade Cup ₹80 (group:268066792, choice:133264952), Chipotle Yoghurt ₹50 (group:268066792, choice:133264953), Ranch Yoghurt ₹50 (group:268066792, choice:133264954), Mexican Salsa ₹50 (group:268066792, choice:133264955), House Caesar Yoghurt ₹50 (group:268066792, choice:133264956), Tzatziki Dip ₹50 (group:268066792, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911627, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911627, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911627, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911628, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911628, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911628, choice:144849004)]\n10. Avocado Egg Breakfast Pita Pocket — ₹279 | 4.7★ (3) (ID: 191685301)\n   Addons (Extra Toppings): [Black Olives ₹30 (group:318911629, choice:151484365), Jalapenos ₹30 (group:318911629, choice:151484364), Gherkins ₹30 (group:318911629, choice:151484361), Feta Cheese ₹40 (group:318911629, choice:151484360), Pasmesan Cheese ₹70 (group:318911629, choice:151484363), Sour Cream Cheese ₹50 (group:318911629, choice:151484362), Hass Avocado ₹100 (group:318911629, choice:151484359), Dried Cranberries ₹40 (group:318911629, choice:151484358)]\n   Addons (Dressing Add On\'s): [Hummus Cup ₹80 (group:268066791, choice:133264959), Olive Tapenade Cup ₹80 (group:268066791, choice:133264952), Chipotle Yoghurt ₹50 (group:268066791, choice:133264953), Ranch Yoghurt ₹50 (group:268066791, choice:133264954), Mexican Salsa ₹50 (group:268066791, choice:133264955), House Caesar Yoghurt ₹50 (group:268066791, choice:133264956), Tzatziki Dip ₹50 (group:268066791, choice:140815073)]\n   Addons (Make Your Own Meal): [Tropical Fruit Juice [215ml] ₹199 (group:318911630, choice:134675976), Cold Brew Coffee [215ml] ₹179 (group:318911630, choice:134675975), Very Berry Breakfast Smoothie [215ml] ₹199 (group:318911630, choice:134675974)]\n   Addons (Desserts): [Exotic Fruit Custard (200ml Jar) ₹249 (group:318911631, choice:144849002), Banana Blueberry Custard (200ml Jar) ₹259 (group:318911631, choice:144849003), Badami Pista Sabudana Kheer (200ml Jar) ₹259 (group:318911631, choice:144849004)]\n\nSome items have customisation options (variants/add-ons). Briefly ask the user which variant/size they prefer or if they want any add-ons. Keep it concise, e.g. "Chicken Biryani has size options (Reg/Large) — which do you prefer? Any add-ons?"\n\n⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget already displays — instead, provide a brief recommendation or ask what they\'d like to do next.'

KOREAN_WINGS_RESPONSE_TEXT = 'Found 10 menu items for "Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy)":\n1. Korean Honey Gochujang Chicken Wings (4 Pcs) (Its extra Spicy) — ₹279 | 5.0★ (1) (ID: 182602816)\n   Addons (Add on Sauce): [Korean Buffalo Sauce ₹49 (group:306232169, choice:115016421), Korean BBQ Sauce ₹49 (group:306232169, choice:115016420), Korean Honey Sriracha Sauce ₹49 (group:306232169, choice:115016442), Korean Hot Sauce ₹59 (group:306232169, choice:115016431), Honey Gochujang ₹49 (group:306232169, choice:128918259)]\n2. Korean Honey Gochujang Fried Chicken Burger (Its extra Spicy) — ₹299 | 4.1★ (4) (ID: 187075710)\n   Variants (Choice of Cheese): [With Cheese ₹318 (group:68602032, var:205649464), Without Cheese ₹299 (group:68602032, var:205649465)]\n   Addons (Add on Drink): [Oreo Milkshake ₹199 (group:254891483, choice:124585314), Mint Mojito with Popping Bubbles ₹149 (group:254891483, choice:124585308), Le Chocolate Milkshake ₹199 (group:254891483, choice:124585319), Taro Lava Bubble Tea ₹199 (group:254891483, choice:124585309)]\n   Addons (Add-on): [Fries + Coke (250 ML) + Dip (save 10%) ₹199 (group:259399047, choice:142236273), Crispy Fries ₹149 (group:259399047, choice:142518770), Korean Buffalo Sauce Dip ₹49 (group:259399047, choice:142236274), Korean Honey Sriracha Sauce Dip ₹49 (group:259399047, choice:142236277), Korean Hot Sauce Dip ₹49 (group:259399047, choice:142236278), With Jalapeño ₹0 (group:259399047, choice:142518769), Peri Peri Fries ₹169 (group:259399047, choice:142518768)]\n3. Dakgangjeong -Korean Spicy Honey Gochujang Fried Chicken (12 Pcs) — ₹249 | 5.0★ (16) (ID: 169732955)\n   Addons (Add on Sauce): [Korean Buffalo Sauce ₹49 (group:293084295, choice:115016421), Korean BBQ Sauce ₹49 (group:293084295, choice:115016420), Korean Honey Sriracha Sauce ₹49 (group:293084295, choice:115016442), Korean Hot Sauce ₹59 (group:293084295, choice:115016431), Honey Gochujang ₹49 (group:293084295, choice:128918259)]\n4. Korean Honey Gochujang Paneer Burger (Its extra Spicy) — ₹279 | Veg | 4.0★ (1) (ID: 187075711)\n   Variants (Choice of Cheese): [With Cheese ₹298 (group:68602033, var:205649466), Without Cheese ₹279 (group:68602033, var:205649467)]\n   Addons (Add on Drink): [Oreo Milkshake ₹199 (group:254891489, choice:124585314), Mint Mojito with Popping Bubbles ₹149 (group:254891489, choice:124585308), Le Chocolate Milkshake ₹199 (group:254891489, choice:124585319), Taro Lava Bubble Tea ₹199 (group:254891489, choice:124585309)]\n   Addons (Add-on): [Fries + Coke (250 ML) + Dip (save 10%) ₹199 (group:259399050, choice:142236273), Crispy Fries ₹149 (group:259399050, choice:142518770), Korean Buffalo Sauce Dip ₹49 (group:259399050, choice:142236274), Korean Honey Sriracha Sauce Dip ₹49 (group:259399050, choice:142236277), Korean Hot Sauce Dip ₹49 (group:259399050, choice:142236278), With Jalapeño ₹0 (group:259399050, choice:142518769), Peri Peri Fries ₹169 (group:259399050, choice:142518768)]\n5. Korean Honey Gochujang Veggie Burger (Its extra Spicy) — ₹299 | Veg | 4.0★ (20) (ID: 187075712)\n   Variants (Choice of Cheese): [With Cheese ₹318 (group:68602034, var:205649468), Without Cheese ₹299 (group:68602034, var:205649469)]\n   Addons (Add-on): [Fries + Coke (250 ML) + Dip (save 10%) ₹199 (group:259399040, choice:142236273), Crispy Fries ₹149 (group:259399040, choice:142518770), Korean Buffalo Sauce Dip ₹49 (group:259399040, choice:142236274), Korean Honey Sriracha Sauce Dip ₹49 (group:259399040, choice:142236277), Korean Hot Sauce Dip ₹49 (group:259399040, choice:142236278), With Jalapeño ₹0 (group:259399040, choice:142518769), Peri Peri Fries ₹169 (group:259399040, choice:142518768)]\n6. Korean Honey Sriracha Chicken Wings - (4 Pcs) — ₹269 |  (ID: 155833243)\n   Addons (Add on Sauce): [Korean Buffalo Sauce ₹49 (group:306232170, choice:115016421), Korean BBQ Sauce ₹49 (group:306232170, choice:115016420), Korean Honey Sriracha Sauce ₹49 (group:306232170, choice:115016442), Korean Hot Sauce ₹59 (group:306232170, choice:115016431), Honey Gochujang ₹49 (group:306232170, choice:128918259)]\n7. Korean Buffalo Sauce Chicken Wings - (4 Pcs) — ₹269 | 5.0★ (3) (ID: 155833242)\n8. Korean Honey Sriracha Fried Chicken (12 Pcs) — ₹249 | 4.4★ (15) (ID: 169732959)\n   Addons (Add on Sauce): [Korean Buffalo Sauce ₹49 (group:293084300, choice:115016421), Korean BBQ Sauce ₹49 (group:293084300, choice:115016420), Korean Honey Sriracha Sauce ₹49 (group:293084300, choice:115016442), Korean Hot Sauce ₹59 (group:293084300, choice:115016431), Honey Gochujang ₹49 (group:293084300, choice:128918259)]\n9. Korean BBQ Chicken Wings - (4 Pcs) — ₹249 | 3.8★ (4) (ID: 155833241)\n10. Korean BBQ Fried Chicken (12 Pcs) — ₹249 | 4.0★ (1) (ID: 169732961)\n   Addons (Add on Sauce): [Korean Buffalo Sauce ₹49 (group:293084308, choice:115016421), Korean BBQ Sauce ₹49 (group:293084308, choice:115016420), Korean Honey Sriracha Sauce ₹49 (group:293084308, choice:115016442), Korean Hot Sauce ₹59 (group:293084308, choice:115016431), Honey Gochujang ₹49 (group:293084308, choice:128918259)]\n\nSome items have customisation options (variants/add-ons). Briefly ask the user which variant/size they prefer or if they want any add-ons. Keep it concise, e.g. "Chicken Biryani has size options (Reg/Large) — which do you prefer? Any add-ons?"\n\n⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget already displays — instead, provide a brief recommendation or ask what they\'d like to do next.'


BROAD_RESPONSE_TEXT = 'Found 10 menu items for "Hummus & pita":\n1. Falafel Hummus Pita Pocket — ₹249 | Veg | 4.8★ | Caterspoint (restaurantId: 300897) (ID: 186053967) [has addons]\n2. Hummus With Pita Bread — ₹210 | Veg | 4.0★ | Shawarmao Na (restaurantId: 1374628) (ID: 200616039) [has addons]\n3. Mini Hummus And Pita — ₹295 | Veg | 4.8★ | Layr\'d (restaurantId: 1096649) (ID: 182906973) [has addons]\n4. Hummus Falafel Pita — ₹399 | Veg | 3.8★ | Nutriobox (restaurantId: 979965) (ID: 155354051) [has addons]\n5. Avocado & Chili Oil Hummus Veg Pita Pocket — ₹369 | Veg | Witchery-The Sandwich Co. (restaurantId: 592168) (ID: 207555927) [has addons]\n6. Hummus With Pita Triangles — ₹299 | Veg | 4.2★ | Caterspoint (restaurantId: 300897) (ID: 63799487) [has addons]\n7. The Hummus & Pita Plate — ₹439 | Veg | Coffee Island (restaurantId: 1024381) (ID: 193115134)\n8. Hummus And Pita Platter — ₹315 | Veg | 4.1★ | Third Wave Coffee (restaurantId: 450915) (ID: 127378953)\n9. Falafel Pita Hummus — ₹379 | Veg | The Big Tree (restaurantId: 1074936) (ID: 169539818)\n10. Hummus, Pita Strips & PERi-PERi Drizzle — ₹399 | Veg | 4.3★ | Nando\'s (restaurantId: 17464) (ID: 90359506)\n\nSome items have customisation options (variants/add-ons). Briefly ask the user which variant/size they prefer or if they want any add-ons. Keep it concise, e.g. "Chicken Biryani has size options (Reg/Large) — which do you prefer? Any add-ons?"\n\n⚠️ A rich UI widget may be shown to the user with this data. Avoid restating everything the widget already displays — instead, provide a brief recommendation or ask what they\'d like to do next.'


# Reproduces the live 2026-09-28 regression (see logger.txt): the user
# picked craving_tag="sandwich" then craving="Grilled Cheese", and Swiggy's
# own free-text search_menu came back with dishes that have nothing to do
# with grilled cheese at all (a wrap, a burger, an unrelated sandwich)
# mixed in among the genuine matches.
GRILLED_CHEESE_SEARCH_TEXT = (
    'Found 10 menu items for "Grilled Cheese":\n'
    "1. Grilled Cheese \u2014 \u20b9125 | 5.0\u2605 | AKU'S - The Burger Co. (restaurantId: 2001) (ID: 184895813)\n"
    "2. Truffle Shroom Wrap \u2014 \u20b9335 | 4.9\u2605 | Good Flippin' Burgers (restaurantId: 2002) (ID: 184895814)\n"
    "3. Grilled Cheese Toastie \u2014 \u20b9299 | 4.5\u2605 | Sandburgs (restaurantId: 2003) (ID: 184895815)\n"
    "4. Grilled Cheese Sandwich \u2014 \u20b9159 | 4.4\u2605 | Burger Hut (restaurantId: 2004) (ID: 184895816)\n"
    "5. Pan Grilled Herb Cheese Salad \u2014 \u20b9329 | 4.2\u2605 | Saladspoint (restaurantId: 2005) (ID: 184895817)\n"
    "6. Veg Achaari Sandwich \u2014 \u20b9159 | 4.1\u2605 | Burger Hut (restaurantId: 2004) (ID: 184895818)\n"
    "7. Louis Grand Royale \u2014 \u20b9888 | 4.0\u2605 | Louis Burger (restaurantId: 2006) (ID: 184895819)\n"
    "8. Cheesy Aloo Bunto Sandwich \u2014 \u20b9195 | 3.7\u2605 | FLYP Coffee (restaurantId: 2007) (ID: 184895820)\n"
    "9. Pesto Grilled Cheese Sandwich \u2014 \u20b9349 | 3.3\u2605 | Slamwich by Burgerama (restaurantId: 2008) (ID: 184895821)\n"
    "10. Masala Grilled Cheese Sandwich \u2014 \u20b9150 | 3.1\u2605 | Dohful Cookies & Coffee (restaurantId: 2009) (ID: 184895822)\n"
)


class MatchesCravingTests(unittest.TestCase):
    def test_exact_and_substring_names_match(self):
        self.assertTrue(_matches_craving("Grilled Cheese", "Grilled Cheese"))
        self.assertTrue(_matches_craving("Grilled Cheese Toastie", "Grilled Cheese"))

    def test_a_name_sharing_no_real_words_with_the_craving_does_not_match(self):
        self.assertFalse(_matches_craving("Truffle Shroom Wrap", "Grilled Cheese"))
        self.assertFalse(_matches_craving("Louis Grand Royale", "Grilled Cheese"))

    def test_empty_craving_matches_everything(self):
        self.assertTrue(_matches_craving("Anything At All", ""))


class SearchDishesCravingRelevanceTests(unittest.TestCase):
    """search_dishes' primary (search_menu) path only ever re-applied
    _apply_filters (dietary/price/cuisine) to what Swiggy's free-text
    search handed back -- never a relevance check against the craving
    itself, unlike the search_restaurants fallback path below it, which
    already ran dishes through _matches_craving. That meant the one path
    actually taken whenever search_menu finds anything (the normal case)
    was the one path with no relevance filtering at all."""

    def test_dishes_unrelated_to_the_craving_are_dropped(self):
        from unittest.mock import patch

        with patch("cravr.food_mcp.call_tool_sync", return_value=GRILLED_CHEESE_SEARCH_TEXT):
            results = search_dishes({"craving": "Grilled Cheese", "course": "main"}, "addr1")

        names = {r["item_name"] for r in results}
        self.assertIn("Grilled Cheese", names)
        self.assertIn("Grilled Cheese Toastie", names)
        self.assertIn("Grilled Cheese Sandwich", names)
        self.assertIn("Pan Grilled Herb Cheese Salad", names)
        self.assertIn("Pesto Grilled Cheese Sandwich", names)
        self.assertIn("Masala Grilled Cheese Sandwich", names)

        self.assertNotIn("Truffle Shroom Wrap", names)
        self.assertNotIn("Veg Achaari Sandwich", names)
        self.assertNotIn("Louis Grand Royale", names)
        self.assertNotIn("Cheesy Aloo Bunto Sandwich", names)


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


class ParseDetailedSearchMenuItemsVariantsTests(unittest.TestCase):
    def test_addon_only_item_has_no_variant_groups(self):
        parsed = _parse_detailed_search_menu_items(KOREAN_WINGS_RESPONSE_TEXT)
        item = parsed["182602816"]
        self.assertEqual(item["variant_groups"], [])
        self.assertEqual(len(item["addon_groups"]), 1)
        sauce_group = item["addon_groups"][0]
        self.assertEqual(sauce_group["group_name"], "Add on Sauce")
        self.assertEqual(sauce_group["group_id"], "306232169")
        self.assertEqual(len(sauce_group["choices"]), 5)

    def test_variants_group_is_captured_separately_with_its_var_id_token(self):
        parsed = _parse_detailed_search_menu_items(KOREAN_WINGS_RESPONSE_TEXT)
        item = parsed["187075710"]
        self.assertEqual(len(item["variant_groups"]), 1)
        cheese_group = item["variant_groups"][0]
        self.assertEqual(cheese_group["group_name"], "Choice of Cheese")
        self.assertEqual(cheese_group["group_id"], "68602032")
        self.assertEqual(
            [(c["name"], c["id"], c["price_inr"]) for c in cheese_group["choices"]],
            [("With Cheese", "205649464", 318.0), ("Without Cheese", "205649465", 299.0)],
        )
        addon_names = [g["group_name"] for g in item["addon_groups"]]
        self.assertEqual(addon_names, ["Add on Drink", "Add-on"])


if __name__ == "__main__":
    unittest.main()


class DetailedAddonGroupNameWithBracketsTests(unittest.TestCase):
    def test_group_names_containing_brackets_are_kept(self):
        from cravr.food_mcp import _parse_detailed_search_menu_items
        text = (
            'Found 1 menu items for "x":\n'
            "1. Greek Falafel Salad (Veg) — ₹299 | Veg | 3.8★ (3) (ID: 205083764)\n"
            "   Addons (Add Grain): [Brown Rice ₹40 (group:308598175, choice:149928102)]\n"
            "   Addons ( Add Protein And Veggie Boosts (Veg & Non-veg)): "
            "[Extra Herbed Paneer ₹90 (group:308598176, choice:151898192)]\n"
            "   Addons (Make It A Meal (Sides & Drinks)): "
            "[Mixed Berry Cooler ₹99 (group:308598179, choice:152248148)]\n"
        )
        groups = _parse_detailed_search_menu_items(text)["205083764"]["addon_groups"]
        self.assertEqual(
            [g["group_name"] for g in groups],
            ["Add Grain", "Add Protein And Veggie Boosts (Veg & Non-veg)", "Make It A Meal (Sides & Drinks)"],
        )
        self.assertEqual(groups[1]["choices"][0]["id"], "151898192")
