import unittest
from decimal import Decimal
from orders.parser import parse_line
from orders.pricing import total, discount_rate
from orders.parser import parse_order
from orders.report import customer_totals


class Hidden(unittest.TestCase):
    def test_three_digit_quantity(self):
        self.assertEqual(parse_line("100x Bulk Item @ 0.05").qty, 100)

    def test_bad_line_rejected(self):
        with self.assertRaises(ValueError):
            parse_line("x Widget @ 2")

    def test_exact_tier_totals(self):
        self.assertEqual(discount_rate(Decimal("999.99")), Decimal("0.05"))
        # 20 x 50 = 1000 -> 10% off -> 900 * 1.18 = 1062.00
        self.assertEqual(total(parse_order("20x Crate @ 50")), Decimal("1062.00"))

    def test_half_up_cases(self):
        # 3 x 0.35 = 1.05 * 1.18 = 1.239 -> 1.24 ; 25 x 0.85 = 21.25 * 1.18 = 25.075 -> 25.08
        self.assertEqual(total(parse_order("3x Y @ 0.35")), Decimal("1.24"))
        self.assertEqual(total(parse_order("25x Y @ 0.85")), Decimal("25.08"))

    def test_report_order_and_type(self):
        got = customer_totals([("zed", "1x A @ 1"), ("amy", "1x A @ 1"), ("zed", "1x A @ 1")])
        self.assertEqual(list(got), ["zed", "amy"])
        self.assertTrue(all(isinstance(v, Decimal) for v in got.values()))
