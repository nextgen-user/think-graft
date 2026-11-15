"""Order pricing: volume tier discount, then promo code, then tax, rounded to cents (half up)."""
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from . import promos

# (threshold, rate): a subtotal of at least `threshold` gets `rate` off
TIERS = [(Decimal("1000"), Decimal("0.10")), (Decimal("500"), Decimal("0.05")), (Decimal("0"), Decimal("0"))]
TAX = Decimal("0.18")
CENT = Decimal("0.01")


def subtotal(lines) -> Decimal:
    return sum((l.unit_price * l.qty for l in lines), Decimal("0"))


def discount_rate(amount: Decimal) -> Decimal:
    for threshold, rate in TIERS:
        if amount >= threshold:
            return rate
    return Decimal("0")


def total(lines, code=None, on=None) -> Decimal:
    """Price an order, applying tier discount, then promo code, then tax."""
    if on is None:
        on = date.today()

    sub = subtotal(lines)
    after = sub * (1 - discount_rate(sub))

    promo = promos.resolve(code)
    after -= promos.discount(promo, after, sub, on)
    if after < Decimal("0"):
        after = Decimal("0")

    with_tax = after * (1 + TAX)
    return with_tax.quantize(CENT, rounding=ROUND_HALF_UP)
