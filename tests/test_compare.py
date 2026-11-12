"""The number-aware comparison in tests/_compare.py."""
import pytest

from tests._compare import compare_output, lines_match


@pytest.mark.parametrize("shown,actual,ok", [
    # integers match exactly
    ("10", "10", True),
    ("10", "11", False),
    ("-3", "-3", True),
    ("-3", "-4", False),
    ("GPT-2", "GPT-2", True),
    ("GPT-2", "GPT-3", False),
    # comma-grouped numbers are one integer, matched exactly
    ("1,000", "1,000", True),
    ("1,000", "2,000", False),
    ("12,345,678", "12,345,679", False),
    ("12,345,678", "12,345,678", True),
    # a comma-separated list is separate integers
    ("[1,2,3]", "[1,2,3]", True),
    ("[1,2,3]", "[1,2,4]", False),
    # decimals: one unit in the last printed digit
    ("3.14", "3.14", True),
    ("3.14", "3.15", True),
    ("3.14", "3.13", True),
    ("3.14", "3.16", False),
    ("-0.50", "-0.51", True),
    ("-0.50", "-0.52", False),
    ("1,234.5", "1,234.6", True),
    ("1,234.5", "1,234.7", False),
    # scientific notation: one unit in the last mantissa digit at the same exponent
    ("1.2e-3", "1.3e-3", True),
    ("1.2e-3", "1.4e-3", False),
    ("1e-3", "2e-3", False),
    ("1e-3", "1e-3", True),
    ("2.50E+4", "2.51E+4", True),
    # the percent sign is not part of the number
    ("95.5%", "95.6%", True),
    ("95%", "96%", False),
    ("95.5%", "95.5 %", False),
    # the text around the numbers must be identical
    ("float32", "float64", False),
    ("float32", "float32", True),
    ("loss 0.5", "cost 0.5", False),
    # dotted versions are exact tokens, also behind a letter
    ("2.15.0", "2.15.0", True),
    ("2.15.0", "2.14.0", False),
    ("2.15.0", "2.15.1", False),
    ("v2.15.0", "v2.15.1", False),
    ("v2.15.0", "v2.15.0", True),
    # a differing count of numbers on a line
    ("1 2 3", "1 2", False),
    ("1 2", "1 2 3", False),
    ("a 1.5", "a", False),
])
def test_lines_match(shown, actual, ok):
    assert lines_match(shown, actual, shape=False) is ok


@pytest.mark.parametrize("shown,actual,ok", [
    ("loss 0.5 at 10", "loss 9.9 at 1234", True),
    ("loss 0.5 at 10", "loss 9.9", False),
    ("float32", "float64", False),
])
def test_shape_comparison_replaces_numbers_only(shown, actual, ok):
    assert lines_match(shown, actual, shape=True) is ok


def test_tolerance_replaces_one_unit_for_decimals_but_not_for_integers():
    assert lines_match("loss 0.50", "loss 0.54", shape=False, tol=0.05)
    assert not lines_match("loss 0.50", "loss 0.56", shape=False, tol=0.05)
    assert not lines_match("step 10 loss 0.50", "step 11 loss 0.50", shape=False, tol=0.05)


def test_compare_output_names_the_first_differing_line():
    assert compare_output("a 1\nb 2\n", "a 1\nb 2\n", shape=False) is None
    assert compare_output("a 1\nb 3\n", "a 1\nb 2\n", shape=False) == "output line 2: expected 'b 2', got 'b 3'"
    assert compare_output("a\n", "a\nb\n", shape=True) == "output-shape: expected 2 lines, got 1"
