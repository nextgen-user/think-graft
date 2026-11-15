import unittest
from datetime import date
from decimal import Decimal
from orders.parser import parse_order, parse_code
from orders.pricing import total
from orders.report import customer_totals
D = date(2026, 5, 31)


class HiddenPromo(unittest.TestCase):
    def test_parse_code(self):
        self.assertEqual(parse_code("CODE: spring10\n2x A @ 1"), "SPRING10")
        self.assertEqual(parse_code("2x A @ 1\n  code:  big50  "), "BIG50")
        self.assertIsNone(parse_code("2x A @ 1"))

    def test_code_line_not_an_item(self):
        self.assertEqual([l.item for l in parse_order("CODE: WELCOME5\n2x A @ 1\n1x B @ 2")], ["A", "B"])

    def test_spring_valid_and_expired(self):
        lines = parse_order("10x A @ 10")
        self.assertEqual(total(lines, "SPRING10", D), Decimal("106.20"))          # 100 -> 90 -> 106.20
        self.assertEqual(total(lines, "SPRING10", date(2026, 6, 1)), Decimal("118.00"))

    def test_code_case_insensitive(self):
        self.assertEqual(total(parse_order("10x A @ 10"), "spring10", D), Decimal("106.20"))

    def test_big50_minimum(self):
        self.assertEqual(total(parse_order("30x A @ 10"), "BIG50", D), Decimal("295.00"))     # 300 -> 250
        self.assertEqual(total(parse_order("1x A @ 299.99"), "BIG50", D), Decimal("353.99"))  # below minimum

    def test_order_tier_then_promo_then_tax(self):
        self.assertEqual(total(parse_order("60x A @ 10"), "BIG50", D), Decimal("613.60"))     # 600 -> 570 -> 520
        self.assertEqual(total(parse_order("60x A @ 10"), "SPRING10", D), Decimal("605.34"))  # 600 -> 570 -> 513

    def test_welcome_and_floor(self):
        self.assertEqual(total(parse_order("2x A @ 10"), "WELCOME5", D), Decimal("17.70"))
        self.assertEqual(total(parse_order("1x A @ 3"), "WELCOME5", D), Decimal("0.00"))

    def test_unknown_and_none(self):
        self.assertEqual(total(parse_order("2x A @ 10"), "BOGUS", D), Decimal("23.60"))
        self.assertEqual(total(parse_order("2x A @ 10")), Decimal("23.60"))

    def test_report_applies_codes(self):
        got = customer_totals([("a", "CODE: WELCOME5\n2x A @ 10"), ("b", "1x B @ 5"), ("a", "1x B @ 5")], on=D)
        self.assertEqual(got, {"a": Decimal("23.60"), "b": Decimal("5.90")})
        self.assertTrue(all(isinstance(v, Decimal) for v in got.values()))
