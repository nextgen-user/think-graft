"""Promo-code definitions and small helpers for applying them."""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal


@dataclass(frozen=True)
class Promo:
    code: str
    kind: str  # "percent" or "fixed"
    value: Decimal
    valid_through: date | None = None
    min_subtotal: Decimal | None = None


PROMOS = {
    "SPRING10": Promo(
        "SPRING10", "percent", Decimal("0.10"), valid_through=date(2026, 5, 31)
    ),
    "BIG50": Promo(
        "BIG50", "fixed", Decimal("50.00"), min_subtotal=Decimal("300.00")
    ),
    "WELCOME5": Promo("WELCOME5", "fixed", Decimal("5.00")),
}


def resolve(code):
    """Look up a code case-insensitively; return the Promo or None if unknown."""
    if code is None:
        return None
    return PROMOS.get(str(code).strip().upper())


def is_applicable(promo, subtotal_before_discount, on):
    """A promo is eligible if it exists, is not expired, and meets its minimum subtotal."""
    if promo is None:
        return False
    if promo.min_subtotal is not None and subtotal_before_discount < promo.min_subtotal:
        return False
    if promo.valid_through is not None and on > promo.valid_through:
        return False
    return True


def discount(promo, amount, subtotal_before_discount, on):
    """Promo discount in dollars on `amount` (post-tier subtotal); zero if ineligible."""
    if not is_applicable(promo, subtotal_before_discount, on):
        return Decimal("0")
    if promo.kind == "percent":
        return amount * promo.value
    return min(promo.value, amount)
