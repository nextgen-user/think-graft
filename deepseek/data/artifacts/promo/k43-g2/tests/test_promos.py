import unittest
from datetime import date
from decimal import Decimal

from orders.parser import parse_code, parse_order
from orders.pricing import total
from orders.report import customer_totals


def order_total(text, on=None):
    """Total for an order text, applying the code embedded in the text."""
    return total(parse_order(text), parse_code(text), on)


class TestParseCode(unittest.TestCase):
    def test_code_line(self):
        text = "# weekly\nCODE: SPRING10\n2x A @ 1\n"
        self.assertEqual(parse_code(text), "SPRING10")

    def test_case_and_spaces(self):
        self.assertEqual(parse_code("  code :   spring10  "), "SPRING10")

    def test_no_code(self):
        self.assertIsNone(parse_code("2x A @ 1\n"))

    def test_parse_order_ignores_code(self):
        lines = parse_order("CODE: welcome5\n2x A @ 1\n1x B @ 2\n")
        self.assertEqual([l.item for l in lines], ["A", "B"])

    def test_item_line_named_code_is_not_a_code_line(self):
        self.assertEqual(parse_code("2x CODE @ 1"), None)
        self.assertEqual(parse_order("2x CODE @ 1")[0].item, "CODE")


class TestPromoPricing(unittest.TestCase):
    def test_percent_after_tier(self):
        # 10 x 50 = 500 -> tier 5% -> 475 -> SPRING10 10% -> 427.50 -> tax 18% -> 504.45
        self.assertEqual(
            order_total("CODE: SPRING10\n10x Crate @ 50", on=date(2026, 5, 31)),
            Decimal("504.45"),
        )

    def test_spring10_expired(self):
        # On 2026-06-01 the code is expired, so only the tier applies: 560.50
        self.assertEqual(
            order_total("CODE: SPRING10\n10x Crate @ 50", on=date(2026, 6, 1)),
            Decimal("560.50"),
        )

    def test_fixed_amount_above_threshold(self):
        # 6 x 50 = 300 (eligible) -> no tier -> -50 -> 250 -> tax -> 295.00
        self.assertEqual(
            order_total("CODE: BIG50\n6x Thing @ 50", on=date(2026, 5, 31)),
            Decimal("295.00"),
        )

    def test_fixed_amount_threshold_boundary(self):
        # Exactly $300 before discount qualifies.
        self.assertEqual(
            order_total("CODE: BIG50\n1x Thing @ 300", on=date(2026, 5, 31)),
            Decimal("295.00"),
        )

    def test_fixed_amount_ineligible(self):
        # 299.99 < 300: no promo; 299.99 * 1.18 = 353.9882 -> 353.99
        self.assertEqual(
            order_total("CODE: BIG50\n1x Thing @ 299.99", on=date(2026, 5, 31)),
            Decimal("353.99"),
        )

    def test_welcome5(self):
        self.assertEqual(
            order_total("CODE: WELCOME5\n1x X @ 10", on=date(2026, 5, 31)),
            Decimal("5.90"),
        )

    def test_fixed_amount_never_negative(self):
        # 1x @ 2 -> WELCOME5 would go to -3, clamped to 0 -> tax 0
        self.assertEqual(
            order_total("CODE: WELCOME5\n1x X @ 2", on=date(2026, 5, 31)),
            Decimal("0.00"),
        )

    def test_code_is_case_insensitive(self):
        lines = parse_order("1x X @ 10")
        self.assertEqual(total(lines, code="welcome5", on=date(2026, 5, 31)), Decimal("5.90"))
        self.assertEqual(total(lines, code="  Welcome5  ", on=date(2026, 5, 31)), Decimal("5.90"))

    def test_unknown_code_no_discount(self):
        self.assertEqual(total(parse_order("1x X @ 10"), code="NOPE"), Decimal("11.80"))
        self.assertEqual(total(parse_order("1x X @ 10"), code=None), Decimal("11.80"))

    def test_rounding_only_at_end(self):
        # 1 x 0.85 -> promo 10% -> 0.765 -> *1.18 = 0.9027 -> 0.90
        self.assertEqual(
            order_total("CODE: SPRING10\n1x X @ 0.85", on=date(2026, 5, 31)),
            Decimal("0.90"),
        )

    def test_default_on_is_today(self):
        self.assertIsInstance(total(parse_order("1x X @ 1")), Decimal)


class TestReportWithCodes(unittest.TestCase):
    def test_customer_totals_apply_codes(self):
        orders = [
            ("acme", "CODE: WELCOME5\n2x A @ 10"),   # 20 -> -5 = 15 -> 17.70
            ("acme", "1x B @ 5"),                     # 5 -> 5.90
            ("bolt", "CODE: BIG50\n6x C @ 50"),       # 300 -> -50 = 250 -> 295.00
        ]
        got = customer_totals(orders, on=date(2026, 5, 31))
        self.assertEqual(got, {"acme": Decimal("23.60"), "bolt": Decimal("295.00")})
        self.assertIsInstance(got["acme"], Decimal)

    def test_report_expired_code(self):
        got = customer_totals(
            [("acme", "CODE: SPRING10\n10x C @ 50")], on=date(2026, 6, 1)
        )
        self.assertEqual(got, {"acme": Decimal("560.50")})


if __name__ == "__main__":
    unittest.main()
