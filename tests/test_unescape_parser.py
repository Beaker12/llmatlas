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
