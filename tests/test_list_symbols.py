"""The symbol grammar of tools/list_symbols.py, pinned by tests/fixtures/symbols/."""
import subprocess
import sys

from tests._files import ROOT
from tools.list_symbols import (concordance, definitions, definitions_in_text, is_transpose, meanings, normalise,
                                symbols)

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


def test_a_whitelisted_superscript_label_is_not_scanned():
    # The whitelist is of (base, label) pairs: V^{UK} is not one, so U and K are listed.
    assert on_line("Capital superscript labels:") == {"W", "d_h", "V", "U", "K"}
    assert normalise("x^{R} + W^{UK} + d_h^{R}") == ["x", "R", "W", "d_h"]


def test_any_other_capital_superscript_is_scanned():
    # W_i^Q is W and Q (so chapter 1's per-head projection cannot pass as the generic W), R^{V} lists V.
    assert on_line("Other capital superscripts listed:") == {"W", "Q", "\\mathbb{R}", "V", "C", "N"}


def test_a_transpose_is_not_a_symbol_but_an_upper_limit_is():
    # T over \sum or \prod is the token count; T after x_i, \mathbf{1}, a bracket or K is a transpose.
    assert on_line("Transposes and limits:") == {"T", "i", "x", "a", "A", "B", "Q", "K"}
    assert normalise("\\mathbf{1}^T + \\right)^T") == []


def test_is_transpose_tells_a_transpose_from_a_limit():
    for math, expected in [("\\sum_{i=1}^{T}", [False]), ("\\prod_{i=1}^T", [False]),
                           ("\\sum_{i=1}^{T} x_i^T", [False, True]), ("QK^T", [True]),
                           ("\\mathbf{1}^T", [True]), ("(AB)^T", [True]), ("x^\\top", [True]), ("x^2", [False]),
                           ("\\sum\\limits_{i=1}^{T}", [False]), ("\\bigoplus_{i=1}^{T}", [False]),
                           ("\\sum^{T}_{i=1}", [False]), ("\\prod\\nolimits^T_{i=1}", [False]),
                           ("\\bigcup_{j}^{T}", [False]), ("\\gamma^T", [True])]:
        assert [is_transpose(math, k) for k, ch in enumerate(math) if ch == "^"] == expected, math


def test_an_upper_limit_keeps_its_symbol():
    assert normalise("\\sum\\limits_{i=1}^{T} x_i") == ["i", "T", "x"]
    assert normalise("\\bigoplus_{i=1}^{T} x_i") == ["i", "T", "x"]


def test_a_relation_defines_its_left_hand_side():
    text = ("Given $O$, let $G = \\frac{\\partial \\mathcal{L}}{\\partial O} V^\\top$ be the gradient with respect "
            "to the weights $A$. The pass is $2N + 2 L n d_{attn}$, where $d_{attn} = n_h d_k$ is the attention width.\n")
    assert [(s, p) for s, _, p in definitions_in_text(text)] == [
        ("G", "the gradient with respect to the weights"), ("d_{attn}", "the attention width")]


def test_each_symbol_in_a_list_of_definitions_takes_its_phrase():
    text = ("For a smoothed model, $q(w) = 1$ where $c(w)$ is the reference count of $w$, $C$ the total reference "
            "count and $|V|$ the vocabulary size. If $h$ is a hash function and $\\min h(A)$ denotes the smallest "
            "hash value among $A$'s shingles, then it holds.\n")
    assert [(s, p) for s, _, p in definitions_in_text(text)] == [
        ("c", "the reference count of"), ("C", "the total reference count"), ("V", "the vocabulary size")]


def test_an_operator_applied_to_a_symbol_defines_nothing():
    # "$\\min h(A)$ denotes the smallest hash value" defines \\min h(A), not h.
    text = ("If $\\min h(A)$ denotes the smallest hash value, where $\\max_i x_i$ is the largest entry and "
            "$\\max x$ denotes the peak, let $\\sum_i w_i$ be the total.\n")
    assert list(definitions_in_text(text)) == []


def test_junk_phrases_stay_excluded():
    text = ("With $T = 1$ the weights are used as given; $T > 1$ upsamples. Both are consumed by the same sampler, "
            "the latter with $w_s$ a function of the step.\n")
    assert list(definitions_in_text(text)) == []


def test_a_word_is_one_name_not_a_product():
    assert on_line("Words are names:") == {"pos", "i"}


def test_a_condition_subscript_is_scanned_not_attached():
    assert on_line("Conditions:") == {"x", "i", "\\mathbb{E}", "z", "p", "w", "N"}


def test_operator_and_function_names_are_not_symbols():
    assert on_line("Operators are not symbols:") == {"p", "z", "i", "a", "j", "T", "b", "r", "S", "e", "v", "c",
                                                      "\\tau", "g"}


def test_display_math_is_read_and_code_is_not():
    places = symbols(FIXTURE)
    assert ("grammar.md", 31) in places["\\mathcal{O}"] and ("grammar.md", 31) in places["Q"]
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
    assert found["D"] == [("definitions.md", 10, "data")]  # the phrase stops at "and"
    assert "T" not in found and "n" not in found and "w" not in found  # "with $w$ a ...", "$T = 1$ the ...", "is used as given": not definitions
    assert "d" not in found and "\\mathcal{L}" not in found


def test_the_command_line_prints_symbols_and_definitions():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "list_symbols.py"), str(FIXTURE), "--definitions"],
                         capture_output=True, text=True, check=True).stdout
    assert "d_k\tdefinitions.md:3; grammar.md:7" in out
    assert "    defined definitions.md:L3: the per-head width" in out


# --- the concordance (tests/fixtures/symbols/concordance.md) ----------------------------------------------

def on_page(name="concordance.md"):
    """The concordance restricted to one fixture page: {symbol: [(line, unit), ...]}, in the concordance's order."""
    found = {}
    for sym, places in concordance(FIXTURE).items():
        here = [(line, unit) for page, line, unit in places if page == name]
        if here:
            found[sym] = here
    return found


def run_concordance(*options):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "list_symbols.py"), str(FIXTURE), *options],
                          capture_output=True, text=True, check=True).stdout


def test_the_concordance_is_grouped_in_the_order_of_the_tables():
    # Global table order (tau, h, V, A, O, gamma), then the local list (chi), then the undeclared (Upsilon).
    assert list(on_page()) == ["\\tau", "h", "V", "A", "O", "\\gamma", "\\chi", "\\Upsilon"]
    undeclared = [s for s in concordance(FIXTURE) if s not in meanings(FIXTURE)]
    assert list(concordance(FIXTURE))[-len(undeclared):] == undeclared == sorted(undeclared)


def test_a_symbol_twice_in_one_unit_is_listed_once_for_that_unit():
    assert on_page()["A"][:2] == [
        (3, "The attention weights $A$ multiply $V$, and $A$ is a square matrix."),
        (3, "The weights $A$ sum to one per row.")]
    assert on_page()["\\gamma"] == [(13, "A list item with the gain $\\gamma$ and $\\gamma$ again.")]


def test_display_math_is_shown_with_the_sentence_that_introduces_it():
    assert on_page()["A"][2] == (10, "The attention output is: $$ O = A V $$")
    assert on_page()["O"] == [(10, "The attention output is: $$ O = A V $$")]
    # A display block without an introducing colon is shown alone.
    assert ("definitions.md", 6, "$$y = W x$$") in concordance(FIXTURE)["y"]


def test_the_concordance_command_prints_one_symbol():
    out = run_concordance("--symbol", "A").splitlines()
    assert out[0] == "== A — attention weights =="
    assert sum(line.startswith("== ") for line in out) == 1
    assert [line for line in out if line.startswith("concordance.md")] == ["concordance.md:L3: The attention weights $A$ multiply $V$, and $A$ is a square matrix.",
                       "concordance.md:L3: The weights $A$ sum to one per row.",
                       "concordance.md:L10: The attention output is: $$ O = A V $$"]


def test_the_concordance_command_prints_only_the_undeclared_symbols():
    out = run_concordance("--concordance", "--undeclared")
    headers = [line for line in out.splitlines() if line.startswith("== ")]
    assert "== \\Upsilon — UNDECLARED ==" in headers
    assert "concordance.md:L14: An undeclared $\\Upsilon$ here." in out
    assert all(line.endswith("— UNDECLARED ==") for line in headers)
    assert "== A " not in out and "== \\chi " not in out


def test_a_local_symbol_shows_its_local_meaning():
    out = run_concordance("--concordance")
    assert "== h — the fixture's hash function ==" in out  # local, though the global table has h as well
    assert "== \\chi — the fixture's shingle id ==" in out
    assert "== \\tau — temperature (sampling, source mixing, YaRN) ==" in out
    assert "concordance.md:L5: A hash $h$ maps a shingle $\\chi$ to a bucket." in out
