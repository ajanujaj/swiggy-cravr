import unittest
from datetime import datetime

from cravr.intent.meal_clock import (
    MEAL_WINDOWS,
    current_meal_window,
    meal_intent_options,
    meal_intent_question,
    menu_meal_type,
    resolve_meal_intent,
)


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 8, 9, hour, minute)


class MealWindowTests(unittest.TestCase):
    def test_ten_pm_offers_dinner_quick_bite_and_snack(self):
        """The worked example: 10pm is dinner, or something lighter."""
        window = current_meal_window(at(22, 0))

        self.assertEqual(window.name, "dinner")
        self.assertEqual(
            [value for value, _label in window.options],
            ["dinner", "quick_bite", "snack"],
        )

    def test_ten_pm_question_names_the_time_and_the_choices(self):
        question = meal_intent_question(at(22, 4))

        self.assertIn("10:04 PM", question)
        self.assertIn("dinner", question.lower())
        self.assertIn("snack", question.lower())

    def test_windows_tile_the_whole_day_without_gaps(self):
        """Every minute of the day must resolve, including across midnight."""
        for hour in range(24):
            for minute in (0, 29, 30, 59):
                with self.subTest(hour=hour, minute=minute):
                    self.assertIsNotNone(current_meal_window(at(hour, minute)))

    def test_window_boundaries(self):
        cases = {
            (5, 0): "breakfast",
            (10, 29): "breakfast",
            (10, 30): "lunch",
            (15, 29): "lunch",
            (15, 30): "evening",
            (18, 59): "evening",
            (19, 0): "dinner",
            (22, 59): "dinner",
            (23, 0): "late_night",
            (2, 30): "late_night",
            (4, 59): "late_night",
        }
        for (hour, minute), expected in cases.items():
            with self.subTest(time=f"{hour:02d}:{minute:02d}"):
                self.assertEqual(current_meal_window(at(hour, minute)).name, expected)

    def test_breakfast_is_not_offered_at_night(self):
        values = {value for value, _ in current_meal_window(at(22, 0)).options}
        self.assertNotIn("breakfast", values)

    def test_options_are_payload_shaped(self):
        options = meal_intent_options(at(22, 0))

        self.assertTrue(all({"value", "label"} == set(o) for o in options))

    def test_every_window_has_at_least_one_option(self):
        for window in MEAL_WINDOWS:
            with self.subTest(window=window.name):
                self.assertTrue(window.options)
                self.assertEqual(window.default_intent, window.options[0][0])


class ResolveMealIntentTests(unittest.TestCase):
    def test_resolves_by_position(self):
        self.assertEqual(resolve_meal_intent("2", at(22, 0)), "quick_bite")

    def test_resolves_by_value(self):
        self.assertEqual(resolve_meal_intent("dinner", at(22, 0)), "dinner")

    def test_resolves_by_label_word(self):
        self.assertEqual(resolve_meal_intent("just a snack", at(22, 0)), "snack")

    def test_resolves_synonym_not_in_the_labels(self):
        self.assertEqual(resolve_meal_intent("supper", at(22, 0)), "dinner")

    def test_out_of_range_position_is_not_a_choice(self):
        self.assertIsNone(resolve_meal_intent("9", at(22, 0)))

    def test_unmatched_reply_returns_none(self):
        """A typo must not silently resolve to the first option."""
        self.assertIsNone(resolve_meal_intent("qwerty", at(22, 0)))

    def test_empty_reply_returns_none(self):
        self.assertIsNone(resolve_meal_intent("", at(22, 0)))


class MenuMealTypeTests(unittest.TestCase):
    def test_breakfast_maps_to_the_menu_column_value(self):
        self.assertEqual(menu_meal_type("breakfast"), "Breakfast")

    def test_other_intents_do_not_constrain_meal_type(self):
        """48 of 80 rows are All Day; filtering on it would only lose choices."""
        for intent in ("dinner", "snack", "quick_bite", "late_night_craving", None):
            with self.subTest(intent=intent):
                self.assertIsNone(menu_meal_type(intent))


if __name__ == "__main__":
    unittest.main()
