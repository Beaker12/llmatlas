"""Number-aware text comparison shared by the fence runner and the notebook output checks.

Rule for the numbers on a line (the text around them must be identical):
- no decimal point (`10`, `-3`, `1e-3`) is an integer and must match exactly;
- a comma-grouped number (`1,000`, `12,345,678`) is one integer and must match exactly, but a
  comma-separated list such as `[1,2,3]` is three integers;
- a number with a decimal point (`3.14`, `1.2e-3`) agrees to within one unit in its last printed digit
  (for scientific notation, in the last digit of the mantissa at the same exponent);
- a token with two or more dots (`2.15.0`) is a version and must match exactly; a number never starts directly
  after a `.` or a word character (so `v2.15.0` is plain text);
- a percent sign is not part of the number.
An absolute tolerance `tol` replaces the one-unit rule for numbers with a decimal point.
"""
import re
from decimal import Decimal
from itertools import zip_longest

NUMBER = re.compile(r"(?<![\w.])-?(?:\d+(?:\.\d+){2,}|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse(text):
    """The Decimal value of a number token (commas removed)."""
    return Decimal(text.replace(",", ""))


def one_unit(shown):
    """The value of one unit in the last digit of the number text `shown` (it has a decimal point)."""
    m = re.fullmatch(r"-?[\d,]*(?:\.(\d+))?(?:[eE]([-+]?\d+))?", shown)
    return Decimal(1).scaleb(int(m.group(2) or 0) - len(m.group(1) or ""))


def numbers_match(shown, actual, tol=None):
    """Whether two number tokens agree under the rule in the module docstring."""
    if shown.count(".") >= 2:  # a dotted version such as 2.15.0 is one exact token
        return shown == actual
    if actual.count(".") >= 2:
        return False
    if "." not in shown:
        return parse(shown) == parse(actual)
    allowed = one_unit(shown) if tol is None else Decimal(str(tol))
    return abs(parse(actual) - parse(shown)) <= allowed


def lines_match(shown, actual, shape, tol=None):
    if shape:
        return NUMBER.sub("#", shown) == NUMBER.sub("#", actual)
    if NUMBER.sub("\0", shown) != NUMBER.sub("\0", actual):
        return False
    return all(numbers_match(s, a, tol) for s, a in zip(NUMBER.findall(shown), NUMBER.findall(actual)))


def compare_output(stdout, block, shape, tol=None):
    """None if stdout matches the shown block, else the reason it does not.

    With shape=True only the line count and the text around the numbers are compared.
    """
    def clean(text):
        lines = [l.rstrip() for l in text.splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        return lines
    shown, actual = clean(block), clean(stdout)
    if shape and len(shown) != len(actual):
        return f"output-shape: expected {len(shown)} lines, got {len(actual)}"
    for n, (s, a) in enumerate(zip_longest(shown, actual), start=1):
        if s is None or a is None or not lines_match(s, a, shape, tol):
            if shape and s is not None and a is not None:
                s, a = NUMBER.sub("#", s), NUMBER.sub("#", a)
            return f"output line {n}: expected {s!r}, got {a!r}"
    return None
