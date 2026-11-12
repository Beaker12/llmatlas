"""Number-aware text comparison shared by the fence runner and the notebook output checks."""
import re
from decimal import Decimal
from itertools import zip_longest

NUMBER = re.compile(r"(?<!\w)-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")


def one_unit(shown):
    """The value of one unit in the last digit of the number text `shown`."""
    m = re.fullmatch(r"-?\d*(?:\.(\d+))?(?:[eE]([-+]?\d+))?", shown)
    return Decimal(1).scaleb(int(m.group(2) or 0) - len(m.group(1) or ""))


def lines_match(shown, actual, shape, tol=None):
    if shape:
        return NUMBER.sub("#", shown) == NUMBER.sub("#", actual)
    if NUMBER.sub("\0", shown) != NUMBER.sub("\0", actual):
        return False
    return all(abs(Decimal(a) - Decimal(s)) <= (one_unit(s) if tol is None else Decimal(str(tol)))
               for s, a in zip(NUMBER.findall(shown), NUMBER.findall(actual)))


def compare_output(stdout, block, shape, tol=None):
    """None if stdout matches the shown block, else the reason it does not.

    Numbers agree to within one unit in the last printed digit, or within the absolute
    tolerance `tol` when given; with shape=True only the line count and the text around
    the numbers are compared.
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
