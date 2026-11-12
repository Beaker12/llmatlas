"""The symbol grammar of tools/list_symbols.py, pinned by tests/fixtures/symbols/."""
import subprocess
import sys

from tests._files import ROOT
from tools.list_symbols import definitions, normalise, symbols

FIXTURE = ROOT / "tests" / "fixtures" / "symbols"


def on_line(prefix):
    """The set of symbols list_symbols reports on the grammar.md line starting with `prefix`."""
    lines = (FIXTURE / "grammar.md").read_text(encoding="utf-8").splitlines()
    number = next(n for n, line in enumerate(lines, start=1) if line.startswith(prefix))
    return {sym for sym, places in symbols(FIXTURE).items() if ("grammar.md", number) in places}


def test_text_and_braces_are_stripped_from_a_subscript():
    assert on_line("Subscript text:") == {"W_{gate}"}


def test_a_named_subscript_is_part_of_the_symbol():
    assert on_line("Named subscripts kept:") == {"d_k", "n_h", "n_{kv}", "n_{train}", "\\epsilon_{ls}"}


def test_a_subscript_written_as_text_is_never_a_running_index():
    assert normalise("\\epsilon_{\\text{ls}} + \\epsilon_{ls} + x_{\\mathrm{ij}}") == ["\\epsilon_{ls}", "\\epsilon", "x_{ij}"]


def test_a_running_index_subscript_is_dropped():
    assert on_line("Running indices dropped:") == {"x", "\\omega", "q", "u", "U", "\\Sigma", "y"}


def test_digit_and_command_subscripts_are_kept():
    assert on_line("Digit and command subscripts kept:") == {"\\beta_1", "\\lambda_{\\max}"}


def test_wrappers_are_kept():
    assert on_line("Wrappers kept:") == {"\\mathcal{L}", "\\mathbf{z}", "\\mathbb{E}", "\\mathrm{i}",
                                         "\\mathcal{B}", "\\mathbf{W}_Q"}


def test_accents_are_kept_and_braces_do_not_matter():
    assert on_line("Accents kept:") == {"\\tilde{\\rho}", "\\bar{\\rho}", "\\hat{a}"}


def test_superscripts_and_primes_are_not_part_of_the_symbol():
    # h^{(l)}_i is h; the l inside the superscript is listed on its own, as are a_N and C.
    assert on_line("Superscripts dropped:") == {"h", "l", "n", "C", "a_N", "A"}


def test_a_capital_superscript_label_is_not_scanned():
    assert on_line("Capital superscript labels:") == {"W", "d_h", "V"}


def test_a_condition_subscript_is_scanned_not_attached():
    assert on_line("Conditions:") == {"x", "i", "\\mathbb{E}", "z", "p", "w", "N"}


def test_operator_and_function_names_are_not_symbols():
    assert on_line("Operators are not symbols:") == {"p", "z", "i", "a", "j", "T", "b", "r", "S", "e", "v", "c",
                                                      "\\tau", "g"}


def test_display_math_is_read_and_code_is_not():
    places = symbols(FIXTURE)
    assert ("grammar.md", 25) in places["\\mathcal{O}"] and ("grammar.md", 25) in places["Q"]
    assert "o" not in places and "not" not in places


def test_the_local_list_is_not_a_page():
    assert "\\zeta_{never}" not in symbols(FIXTURE)


def test_normalise_gives_the_normal_form():
    assert normalise("W_{\\text{gate}} + \\tilde\\rho_i") == ["W_{gate}", "\\tilde{\\rho}"]
    assert normalise("h^{(l)}_i") == ["h", "l"]


def test_defining_phrases_are_extracted():
    found = definitions(FIXTURE)
    assert found["d_k"] == [("definitions.md", 3, "the per-head width")]
    assert found["\\tau"] == [("definitions.md", 3, "the temperature")]
    assert found["B"] == [("definitions.md", 4, "the batch size")]
    assert found["\\theta"] == [("definitions.md", 4, "the parameters")]
    assert found["n_h"] == [("definitions.md", 4, "the number of heads")]
    assert found["W"] == [("definitions.md", 8, "a weight matrix")]
    assert "d" not in found and "\\mathcal{L}" not in found


def test_the_command_line_prints_symbols_and_definitions():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "list_symbols.py"), str(FIXTURE), "--definitions"],
                         capture_output=True, text=True, check=True).stdout
    assert "d_k\tdefinitions.md:3; grammar.md:7" in out
    assert "    defined definitions.md:L3: the per-head width" in out
