"""Tests for the fence-aware math-region parser in tools/unescape_math.py."""
from tools.unescape_math import math_regions

SAMPLE = """Intro text with $a\\_b$ inline.

```python
price = "$1,250.00"  # not math
x = "$"
```

$$
E = \\\\sum_i x\\_i
$$

More prose $c$ here.
"""


def test_math_regions_skips_fenced_code_and_finds_display_and_inline():
    regions = math_regions(SAMPLE)
    texts = [SAMPLE[s:e] for s, e in regions]
    assert any(t.startswith("$$") for t in texts), "display block not found"
    assert "$a\\_b$" in texts
    assert "$c$" in texts
    assert not any("1,250" in t for t in texts), "fenced code must be skipped"
    assert len(texts) == 3


def test_a_math_directive_body_is_display_math():
    text = "Prose $a$.\n\n```{math}\n:label: eq-x\n\\chi = \\sum_i x_i\n```\n\n:::{math}\ny = 1\n:::\n\n```python\nz = '$b$'\n```\n"
    texts = [text[s:e] for s, e in math_regions(text)]
    assert texts == ["$a$", "\\chi = \\sum_i x_i\n", "y = 1\n"]
