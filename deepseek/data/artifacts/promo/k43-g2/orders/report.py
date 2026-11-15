"""Totals per customer from several orders."""
from collections import defaultdict
from decimal import Decimal

from .parser import parse_code, parse_order
from .pricing import total


def customer_totals(orders, on=None):
    """orders: list of (customer, order_text).  Returns {customer: Decimal total}."""
    out = defaultdict(lambda: Decimal("0"))
    for customer, text in orders:
        out[customer] += total(parse_order(text), parse_code(text), on)
    return dict(out)
