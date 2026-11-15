"""Parse order lines such as "12x Widget @ 3.50" or "3 x Gadget Pro @ 10" into Line records."""
import re
from dataclasses import dataclass
from decimal import Decimal

LINE = re.compile(r"^\s*(\d+)\s*x\s+(.+?)\s*@\s*(\d+(?:\.\d{1,2})?)\s*$", re.IGNORECASE)
CODE = re.compile(r"^\s*CODE\s*:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Line:
    qty: int
    item: str
    unit_price: Decimal


def parse_line(text: str) -> Line:
    m = LINE.match(text)
    if not m:
        raise ValueError(f"bad order line: {text!r}")
    qty, item, price = m.groups()
    return Line(int(qty), item.strip(), Decimal(price))


def parse_code(text: str):
    """Return the upper-case code from a `CODE: <code>` line, or None if absent."""
    m = CODE.search(text)
    return m.group(1).upper() if m else None


def parse_order(block: str) -> list:
    """One line per item; blank lines, '#' comments and `CODE: <code>` lines are ignored."""
    out = []
    for raw in block.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if CODE.match(raw):
            continue
        out.append(parse_line(raw))
    return out
