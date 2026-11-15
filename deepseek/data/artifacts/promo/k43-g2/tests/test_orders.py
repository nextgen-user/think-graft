import unittest
from decimal import Decimal
from orders.parser import parse_line, parse_order
from orders.pricing import discount_rate, total
from orders.report import customer_totals


class TestParser(unittest.TestCase):
    def test_simple(self):
        l = parse_line("3x Widget @ 2.50")
        self.assertEqual((l.qty, l.item, l.unit_price), (3, "Widget", Decimal("2.50")))

    def test_two_digit_quantity(self):
        l = parse_line("12 x Gadget Pro @ 10")
        self.assertEqual((l.qty, l.item, l.unit_price), (12, "Gadget Pro", Decimal("10")))

    def test_block(self):
        lines = parse_order("# weekly\n2x A @ 1\n\n 1x B @ 2.25\n")
        self.assertEqual([l.item for l in lines], ["A", "B"])


class TestPricing(unittest.TestCase):
    def test_tier_boundaries(self):
        self.assertEqual(discount_rate(Decimal("499.99")), Decimal("0"))
        self.assertEqual(discount_rate(Decimal("500")), Decimal("0.05"))
        self.assertEqual(discount_rate(Decimal("1000")), Decimal("0.10"))

    def test_total_rounding(self):
        # 1x @ 0.85 -> 0.85 * 1.18 = 1.003 -> 1.00 ; 5x @ 0.85 -> 4.25 * 1.18 = 5.015 -> 5.02 (half up)
        self.assertEqual(total(parse_order("1x X @ 0.85")), Decimal("1.00"))
        self.assertEqual(total(parse_order("5x X @ 0.85")), Decimal("5.02"))

    def test_total_discounted(self):
        # 10 x 50 = 500 -> 5% off -> 475 -> *1.18 = 560.50
        self.assertEqual(total(parse_order("10x Crate @ 50")), Decimal("560.50"))


class TestReport(unittest.TestCase):
    def test_customer_totals(self):
        got = customer_totals([("acme", "2x A @ 10"), ("bolt", "1x B @ 5"), ("acme", "1x C @ 1")])
        self.assertEqual(got, {"acme": Decimal("24.78"), "bolt": Decimal("5.90")})
        self.assertIsInstance(got["acme"], Decimal)


if __name__ == "__main__":
    unittest.main()
