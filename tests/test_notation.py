"""Notation: one meaning per symbol, book-wide and in each chapter.

Check name `notation`, one test per chapter. Failures, all in the chapter's math regions:
- a banned pattern from BANNED (derived from the "Replaces" column of the plan's global notation table, each
  scoped to the chapters and meaning its row names);
- a banned phrase from BANNED_PHRASES: a use of a global symbol's spelling with another meaning that only the
  prose around the math identifies ("capacity factor $c$"), matched against each line of the page;
- "O(" in prose outside code and math (asymptotic order is \\mathcal{O} everywhere);
- a symbol tools/list_symbols.py reports that is in neither tests/notation_global.txt nor the chapter's
  notation-local.md;
- a symbol with two different defining phrases ("where $x$ is ...", "$x$ denotes ...") in the chapter.

Book-wide (not per chapter): across notation_global.txt and the six notation-local.md files no symbol has two
meanings unless tests/notation_exceptions.txt lists it with exactly those meanings and a reason; every symbol in
the tables is in the normal form list_symbols.py prints.
"""
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import subprocess

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._pages import blank_code, blank_inline_code, blank_math
from tools.list_symbols import LOCAL_LIST, definitions, is_transpose, normalise, pages, symbols
from tools.unescape_math import math_regions

GLOBAL = ROOT / "tests" / "notation_global.txt"
EXCEPTIONS = ROOT / "tests" / "notation_exceptions.txt"
ALL = frozenset(range(0, 7))  # every chapter, and a directory without a chapter number (the sample chapter)



def _transpose(region, match):
    return is_transpose(region, match.start())


# (regex over the text of one math region, the global row it comes from, the chapters it applies to[, a filter
# (region, match) -> bool that keeps only the matches that are the banned use]).
# Rows marked "(proposed)" are additions from the Task 1 step 9 inventory, to be folded into the plan's table.
BANNED: list[tuple] = [
    (r"d_\{?(?:\\(?:text|mathrm)\{)?model", "Model (residual) width: d_model in ch. 6", frozenset({6})),
    (r"\\sqrt\{N\}", "Layers: N for residual layers at CH6:139", frozenset({6})),
    (r"\\text\{head\}_\{?h\b", "Attention heads: h as head count in ch. 1-2", frozenset({1, 2})),
    (r"\\cdot\s*h\s*\\cdot|\\,h\\,", "Attention heads: h as head count in ch. 2", frozenset({2})),
    (r"/\s*h(?![\w^_])", "Attention heads: h as head count at CH6:82", frozenset({6})),
    (r"d_\{?h(?![\w])", "Per-head width: d_h in ch. 6", frozenset({6})),
    (r"(?<![\\\w{])L(?![\w])", "Sequence or context length: L in ch. 5", frozenset({5})),
    (r"L\\,\s*s\\,", "Sequence or context length: s at CH6:502", frozenset({6})),
    (r"L'|[<>]\s*L(?![\w])|L\\,\\theta|\[0,\s*L\)",
     "Trained and target context length: L, L' in ch. 6", frozenset({6})),
    (r"(?<![\\\w])pos(?![\w])", "Positions: pos in ch. 1", frozenset({1})),
    (r"i\s*-\s*j", "Positions: i - j as a relative offset in ch. 1", frozenset({1})),
    (r"m\s*-\s*n(?![\w])", "Positions: n as key position at CH6:261", frozenset({6})),
    (r"q_t\^\\top\s*k_s", "Positions: q_t^T k_s at CH6:549", frozenset({6})),
    (r"O\(nW\)|L\s*\\cdot\s*W|(?<![\\\w])W\s*=\s*\d", "Sliding-window span: W in ch. 6", frozenset({6})),
    (r"(?<![\\\w{])B_\{?t\b", "Batch size: B_t in the C2-10 fix", frozenset({2})),
    (r"(?<![\\\w{])\|V\|", "Vocabulary: |V| as a size", frozenset({4})),
    (r"\{V\s*\\times", "Vocabulary: V as a size at CH1:277", frozenset({1})),
    (r"\+\s*V\s*d(?![\w])", "Vocabulary: V as a size at CH2:450", frozenset({2})),
    (r"V\s*\\cdot\s*d", "Vocabulary: V as a size at CH6:395", frozenset({6})),
    (r"\^\{N\}|\\frac\{1\}\{N\}", "Tokens in a loss or perplexity sum: N", frozenset({1, 2, 4})),
    (r"w_\{?(?:N|1:N)\b", "Tokens in a loss or perplexity sum: N at CH4:462", frozenset({4})),
    (r"1/T\b|(?<![\\\w])T\s*[=>]\s*1", "Temperature: T at CH5:428", frozenset({5})),
    (r"\\tau\}\{\\\|g", "Gradient-clipping threshold: tau at CH2:223", frozenset({2})),
    (r"\\mathcal\{L\}\s*=\s*-\s*\\sum", "Loss: the sum at CH1:1060", frozenset({1})),
    (r"(?<![\\\w{])L\((?:N|D|C)", "Loss: L(N) in ch. 2", frozenset({2})),
    (r"(?<![\\\w])[PQ](?:\(|_\{|\s*\|\||,\s*Q)", "Data and model distributions: P, Q in ch. 1 and ch. 2",
     frozenset({1, 2})),
    (r"P\s*=\s*\\mathrm\{softmax\}|=\s*PV|P\s*\\odot|\\odot\s*P|P\^(?:T|\\top)",
     "Attention: P as attention weights at CH2:337-345", frozenset({2})),
    (r"g_j\s*=\s*\\gamma_j", "Upstream gradient: g_j at CH2:325-327", frozenset({2})),
    (r"W_\{?[123]\}?(?![\d\w])", "Feed-forward matrices: W_1, W_2, W_3", frozenset({1, 6})),
    (r"d_\{(?:\\text\{)?ffn", "Feed-forward width: d_ffn at CH6:160", frozenset({6})),
    (r"b_\{?[12]\b|b_\{i\s*-\s*j\}", "Biases: b_1, b_2, b_{i-j} in ch. 1", frozenset({1})),
    (r"(?<![\\\w])E\s*\\in|(?<![\\\w])E\[", "Embedding matrix: E at CH1:277", frozenset({1})),
    (r"C\^\{?[ab]\b|(?<![\\\w_])[ab]\s*\\approx", "Compute-optimal exponents: C^a, C^b in ch. 2",
     frozenset({2})),
    (r"\\times\s*s(?![\w])", "Loss-scaling factor: s at CH2:513", frozenset({2})),
    (r"b_\{(?:\\text\{)?out|W_\{(?:\\text\{)?out\}?\}\s*h",
     "Unembedding and vector convention: b_out and W_out h at CH1:1064", frozenset({1})),
    (r"N_s", "Source s: N_s at CH5:429", frozenset({5})),
    (r"\\theta_", "RoPE frequencies: theta_i in ch. 6", frozenset({6})),
    (r"2i/d(?![\w])|d/2\s*-\s*1|\(d\s*-\s*2\)|d/\(d", "RoPE frequencies: bare d in ch. 6", frozenset({6})),
    (r"s\s*=\s*L'|s\^\{\\,?d|1/s(?![\w])", "Context scale factor: bare s in ch. 6", frozenset({6})),
    (r"(?<![\\\w])[sg]_\{?[ij]\b|\\mathrm\{TopK\}\(s\)", "Router: s_i, g_i at CH6:324", frozenset({6})),
    (r"P_i", "Router: P_i at CH6:352", frozenset({6})),
    (r"E_i\s*\(", "Experts: E_i(x) at CH6:324", frozenset({6})),
    (r"k\s*>\s*1|\\,\s*k\\,\s*T", "Experts: k as top-k in ch. 6", frozenset({6})),
    (r"C\s*=\s*\\lceil|\\lceil\s*c", "Experts: C and c at CH6:354", frozenset({6})),
    (r"(?<![\\\w])[hc]_\{?t\b|c_s\b", "Multi-head latent attention: h_t, c_t in ch. 6", frozenset({6})),
    (r"(?<![\\\w])s_[ij]\b", "Document of position i: s_i at CH5:385", frozenset({5})),
    (r"v_\{t\s*\+\s*1\}|\\beta\s*v_t", "Momentum: beta and v_t at CH2:180-183", frozenset({2})),
    (r"T_\{(?:\\text\{)?(?:warmup|\\max)", "Optimizer step: T_warmup, T_max in ch. 2", frozenset({2})),
    (r"n_\{(?:\\text\{)?(?:in|out)\b", "Fan-in and fan-out: n_in, n_out in ch. 2", frozenset({2})),
    (r"(?<![\\\w{])O\(", "Asymptotic order: O(.) in every chapter", ALL),
    (r"k\s*\+\s*1", "Index letters: k as a training step at CH5:435", frozenset({5})),
    (r"(?<![\\\w])r(?![\w])", "Index letters: r as a worker rank at CH5:534", frozenset({5})),
    (r"\|D\|", "YaRN: the paper's |D| in the attention scale", frozenset({6})),
    # (proposed) rows from the Task 1 step 9 inventory
    (r"\^\s*\{?\s*T\b", "(proposed) Transpose: ^\\top, never ^T", ALL, _transpose),
    (r"W_\{?i\}?\^\{?[QKV]\b", "(proposed) Attention projections: W_i^Q at CH1:839 is W_Q^{(i)}", frozenset({1})),
    (r"(?<![\\\w{])W(?![\w])", "(proposed) World size: W at CH5:532-535 is n_{ranks}", frozenset({5})),
    (r"(?<![\\\w])P\s*[\(\[]|P\\left", "(proposed) Probability of an event: \\Pr, not P, in ch. 4", frozenset({4})),
    (r"(?<![\\\w{])X\s*(?:>|\\sim)|[HI]\(X|X;\s*Y", "(proposed) Random variables: \\mathrm{X}, \\mathrm{Y} in ch. 2; "
     "\\Pr[\\text{Pareto}(9) > 1 - \\hat{p}] in ch. 4", frozenset({2, 4})),
    (r"(?<![\\\w])h_\{?(?:l|L|1)(?:\s*[+-]\s*1)?\}?(?![\w])|\\theta_1",
     "(proposed) Hidden state and parameters of layer l are h^{(l)}, theta^{(l)} in ch. 2", frozenset({2})),
    (r"p\s*-\s*y|-y/p", "(proposed) Kronecker delta: the one-hot target y at CH2:58", frozenset({2})),
    (r"\\mathbf\{x\}", "(proposed) Hidden state: the embedding x_i at CH1:279 is h^{(0)}_i", frozenset({1})),
    (r"x_\{?t\b", "(proposed) Index letters: t as a position at CH1:1040", frozenset({1})),
    (r"(?<![\\\w])[kv]_t\b", "(proposed) Index letters: k_t, v_t as positions at CH6:236", frozenset({6})),
    (r"(?<![\\\w])w_\{?(?:1|i|<i)\b|[qc]\(w", "(proposed) Tokens: w for tokens at CH4:462-466", frozenset({4})),
    (r"C\s*\+\s*\\alpha", "(proposed) Reference total: C at CH4:466", frozenset({4})),
    (r"8N|N\s*\\log\s*N", "(proposed) Suffix-array length: N at CH4:722", frozenset({4})),
    (r"J\(A,\s*B\)|h\([AB]\)|A\s*\\c[au]p\s*B", "(proposed) Shingle sets: A, B in ch. 4", frozenset({4})),
    (r"s\^r|k\s*=\s*br|/k(?![\w])|b_i|\\bmod\s*p\b", "(proposed) MinHash and LSH symbols in ch. 4",
     frozenset({4})),
    (r"\\alpha\\,\s*E", "(proposed) Auxiliary-loss weight: alpha at CH6:352", frozenset({6})),
]


# (chapter, regex over one line of the page, the global row it comes from, what to write instead).
# Each entry matches its chapter's text at commit 4f4ee01 (test_every_banned_phrase_matched_the_chapter_at_4f4ee01)
# and must match nothing once the chapter is fixed.
BANNED_PHRASES: list[tuple[int, str, str, str]] = [
    (1, r"where \$V\$ is the vocabulary size", "Vocabulary: V as a size at CH1:277", "$|\\mathcal{V}|$"),
    (2, r"distributions \$P\$ and \$Q\$|from \$P\$ using code optimized for \$Q\$",
     "Data and model distributions: P, Q at CH2:104", "$p$ and $q$"),
    (2, r"momentum parameter \$\\beta\$", "Momentum coefficient: beta at CH2:183", "$\\beta_{\\text{mom}}$"),
    (2, r"where \$\\tau\$ is the clipping threshold", "Gradient-clipping threshold: tau at CH2:225",
     "$g_{\\max}$"),
    (2, r"\(g_i - \\frac\{1\}\{d\}\\sum_j g_j", "Upstream gradient: g_i, g_j at CH2:325", "$\\tilde{g}_i$, $\\tilde{g}_j$"),
    (2, r"and loss \$L\$", "Loss: L at CH2:398", "$\\mathcal{L}$"),
    (2, r"and \$V d\$ is the embedding table", "Vocabulary: V as a size at CH2:450", "$|\\mathcal{V}| d$"),
    (2, r"and \$h\$ the number of heads", "Attention heads: h at CH2:451", "$n_h$"),
    (2, r"where \$s\$ is the scale factor", "Loss-scaling factor: s at CH2:513", "$\\zeta$"),
    (4, r"\$C\$ the total reference count", "(proposed) Reference total: C at CH4:466", "$T_{\\text{ref}}$"),
    (4, r"With \$k\$ independent hash functions", "(proposed) MinHash: k at CH4:613", "$n_{\\text{hash}}$"),
    (4, r"Split the \$k\$ positions", "(proposed) MinHash: k at CH4:615", "$n_{\\text{hash}}$"),
    (4, r"\$k\$-means", "Index letters: k in k-means at CH4:760", "k-means, outside math"),
    (5, r"with \$W\$ processes|every \$W\$-th shard|depends on \$W\$|independent of \$W\$",
     "(proposed) World size: W at CH5:532-535", "$n_{\\text{ranks}}$"),
    (6, r"with \$N\$ the number of residual layers", "Layers: N at CH6:139", "$2L$, the number of residual layers"),
    (6, r"only to the previous \$W\$ tokens", "Sliding-window span: W at CH6:246", "$n_{\\text{win}}$"),
    (6, r"receptive field by \$W\$", "Sliding-window span: W at CH6:246", "$n_{\\text{win}}$"),
    (6, r"key at position \$n\$", "Positions: n as key position at CH6:261", "$m'$"),
    (6, r"sequences up to \$L\$", "Trained context length: L at CH6:265", "$n_{\\text{train}}$"),
    (6, r"divides the lowest by exactly \$s\$", "Context scale factor: s at CH6:275", "$s_{\\text{ctx}}$"),
    (6, r"context length \$s\$", "Sequence or context length: s at CH6:502", "$n$"),
    (6, r"chooses \$k\$ of them|grows with \$k\$", "Experts active per token: k at CH6:320", "$n_{\\text{act}}$"),
    (6, r"keeps the top \$k\$", "Experts active per token: k at CH6:322", "$n_{\\text{act}}$"),
    (6, r"top-\$k\$", "Experts active per token: top-k at CH6:362", "top-$n_{\\text{act}}$"),
    (6, r"capacity factor \$c\$", "Capacity factor: c at CH6:354", "$\\phi$"),
]
PROSE_ORDER = re.compile(r"\bO\(")


def chapter_number(chapter_dir) -> int:
    name = Path(chapter_dir).name
    return int(name[:2]) if name[:2].isdigit() else 0


def _rows(path: Path, columns: int):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.startswith("#"):
            parts = line.split("\t")
            assert len(parts) == columns, f"{path.name}:{number}: expected {columns} tab-separated columns"
            rows.append(tuple(p.strip() for p in parts))
    return rows


def global_rows(path=GLOBAL) -> list[tuple[str, str]]:
    return _rows(path, 2)


def exceptions(path=EXCEPTIONS) -> dict[str, tuple[set, str]]:
    return {sym: ({m.strip() for m in meanings.split(" | ")}, reason) for sym, meanings, reason in _rows(path, 3)}


def local_rows(chapter_dir) -> list[tuple[str, str]]:
    """(symbol, meaning) from the chapter's notation-local.md table (Symbol, Meaning, First used, Rename)."""
    path = Path(chapter_dir) / LOCAL_LIST
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 3 or cells[0] in ("Symbol", "") or set(cells[0]) <= set("-:"):
            continue
        rows.append((cells[0].strip("$").strip(), cells[1]))
    return rows


def banned_failures(chapter_dir, number=None):
    number = chapter_number(chapter_dir) if number is None else number
    active = [(re.compile(entry[0]), entry[1], entry[3] if len(entry) > 3 else None)
              for entry in BANNED if number in entry[2]]
    failures = []
    for page in pages(chapter_dir):
        text = page.read_text(encoding="utf-8")
        for start, end in math_regions(text):
            region = text[start:end]
            for pattern, row, keep in active:
                for m in pattern.finditer(region):
                    if keep and not keep(region, m):
                        continue
                    line = text.count("\n", 0, start) + 1 + region.count("\n", 0, m.start())
                    failures.append(f"{page.name}:L{line}: banned {m.group(0)!r} ({row})")
    return failures


def phrase_failures(chapter_dir, number=None):
    number = chapter_number(chapter_dir) if number is None else number
    active = [(re.compile(p), row, instead) for chapter, p, row, instead in BANNED_PHRASES if chapter == number]
    failures = []
    for page in pages(chapter_dir):
        for line_number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern, row, instead in active:
                for m in pattern.finditer(line):
                    failures.append(f"{page.name}:L{line_number}: banned phrase {m.group(0)!r} ({row}); "
                                    f"write {instead}")
    return failures


def prose_failures(chapter_dir):
    """'O(' in prose, outside code, inline code and math."""
    failures = []
    for page in pages(chapter_dir):
        text = blank_inline_code(blank_math(blank_code(page.read_text(encoding="utf-8"))))
        for m in PROSE_ORDER.finditer(text):
            failures.append(f"{page.name}:L{text.count(chr(10), 0, m.start()) + 1}: 'O(' in prose "
                            f"(Asymptotic order: write $\\mathcal{{O}}(\\cdot)$)")
    return failures


def membership_failures(chapter_dir, global_path=GLOBAL):
    known = {sym for sym, _ in global_rows(global_path)} | {sym for sym, _ in local_rows(chapter_dir)}
    failures = []
    for sym, places in symbols(chapter_dir).items():
        if sym not in known:
            page, line = places[0]
            failures.append(f"{page}:L{line}: symbol {sym!r} is in neither notation_global.txt nor "
                            f"{LOCAL_LIST} ({len(places)} place(s))")
    return failures


ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def _phrase(text):
    return " ".join(ARTICLES.sub(" ", text.lower()).split())


def same_phrase(a, b):
    """Whole normalised phrases are compared: identical, or near-identical (a typo or a plural), never a prefix."""
    a, b = _phrase(a), _phrase(b)
    return a == b or SequenceMatcher(None, a, b).ratio() >= 0.9


def definition_failures(chapter_dir):
    failures = []
    for sym, found in definitions(chapter_dir).items():
        first_page, first_line, first = found[0]
        for page, line, phrase in found[1:]:
            if not same_phrase(first, phrase):
                failures.append(f"{page}:L{line}: {sym!r} defined as {phrase!r}, but as {first!r} at "
                                f"{first_page}:L{first_line}")
    return failures


def notation_failures(chapter_dir, number=None, global_path=GLOBAL):
    return (banned_failures(chapter_dir, number) + phrase_failures(chapter_dir, number) + prose_failures(chapter_dir)
            + membership_failures(chapter_dir, global_path) + definition_failures(chapter_dir))


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_notation(chapter_dir):
    chapter_check("notation")(chapter_dir, notation_failures(chapter_dir))


def meaning_failures(global_path=GLOBAL, exceptions_path=EXCEPTIONS, chapter_dirs=CHAPTER_DIRS):
    """Book-wide: every symbol has one meaning across the global table and the local lists, or an exception."""
    meanings = defaultdict(dict)  # symbol -> {meaning: [where]}
    for sym, meaning in global_rows(global_path):
        meanings[sym].setdefault(meaning, []).append("notation_global.txt")
    global_meanings = {sym: set(found) for sym, found in meanings.items()}
    failures = []
    for chapter_dir in chapter_dirs:
        for sym, meaning in local_rows(chapter_dir):
            if meaning in global_meanings.get(sym, set()):
                failures.append(f"{Path(chapter_dir).name}/{LOCAL_LIST}: {sym!r} repeats a global row")
            meanings[sym].setdefault(meaning, []).append(f"{Path(chapter_dir).name}/{LOCAL_LIST}")
    allowed = exceptions(exceptions_path)
    for sym, found in sorted(meanings.items()):
        if len(found) > 1 and sym not in allowed:
            listing = "; ".join(f"{m!r} ({', '.join(w)})" for m, w in found.items())
            failures.append(f"{sym!r} has {len(found)} meanings and no exception: {listing}")
    for sym, (listed, _reason) in sorted(allowed.items()):
        found = set(meanings.get(sym, {}))
        if found != listed:
            failures.append(f"exception {sym!r} lists {sorted(listed)}, the tables give {sorted(found)}")
    return failures


def test_no_symbol_has_two_meanings_book_wide():
    failures = meaning_failures()
    assert not failures, "\n".join(failures)


def as_text_subscript(sym):
    """The symbol with its subscript written as text, the way a page writes \\epsilon_{\\text{ls}}."""
    base, _, sub = sym.partition("_")
    return f"{base}_{{\\text{{{sub.strip('{}')}}}}}" if sub else sym


def test_every_table_symbol_is_in_normal_form():
    rows = [("notation_global.txt", sym) for sym, _ in global_rows()]
    rows += [(f"{d.name}/{LOCAL_LIST}", sym) for d in CHAPTER_DIRS for sym, _ in local_rows(d)]
    rows += [("notation_exceptions.txt", sym) for sym in exceptions()]
    bad = [f"{where}: {sym!r} normalises to {normalise(sym)}" for where, sym in rows
           if normalise(sym) != [sym] and normalise(as_text_subscript(sym)) != [sym]]
    assert not bad, "\n".join(bad)


def _text_at_4f4ee01(number):
    directory = next(d for d in CHAPTER_DIRS if chapter_number(d) == number)
    try:
        out = subprocess.run(["git", "show", f"4f4ee01:chapters/{directory.name}/index.md"], cwd=ROOT,
                             capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


@pytest.mark.parametrize("entry", BANNED_PHRASES, ids=lambda e: f"ch{e[0]}:{e[1][:30]}")
def test_every_banned_phrase_matched_the_chapter_at_4f4ee01(entry):
    """A phrase pattern that matches nothing at 4f4ee01 would guard nothing; each must have matched then."""
    number, pattern, row, _instead = entry
    text = _text_at_4f4ee01(number)
    if text is None:
        pytest.skip("git or commit 4f4ee01 not available")
    assert any(re.search(pattern, line) for line in text.splitlines()), f"{pattern!r} ({row}) matches nothing"


def test_every_chapter_has_a_local_list():
    assert all((d / LOCAL_LIST).exists() for d in CHAPTER_DIRS)


# --- the check on small chapters --------------------------------------------------------------------------

def chapter_with(tmp_path, page, local="| Symbol | Meaning | First used | Rename |\n|---|---|---|---|\n"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "index.md").write_text(page, encoding="utf-8")
    (tmp_path / LOCAL_LIST).write_text(local, encoding="utf-8")
    return tmp_path


def test_a_clean_chapter_passes(tmp_path):
    chapter = chapter_with(tmp_path, "Width $d$ and $d_k$, where $d_k$ is the per-head width.\n\n"
                                     "Again, where $d_k$ is the per-head widths.\n",
                           local="| Symbol | Meaning | First used | Rename |\n|---|---|---|---|\n"
                                 "| $J$ | Jaccard similarity | index.md:L1 |  |\n")
    assert notation_failures(chapter, number=6) == []


def test_a_banned_pattern_is_reported_only_in_its_chapters(tmp_path):
    chapter = chapter_with(tmp_path, "The width $d_{model}$.\n")
    assert banned_failures(chapter, 6) == ["index.md:L1: banned 'd_{model' (Model (residual) width: d_model in ch. 6)"]
    assert banned_failures(chapter, 2) == []


@pytest.mark.parametrize("math,banned", [
    ("\\sum_{i=1}^{T} x_i", []), ("\\prod_{i=1}^T x_i", []), ("\\sum_{i=1}^{T} x_i^T", ["^T"]),
    ("QK^T", ["^T"]), ("\\mathbf{1}^T", ["^T"]), ("(AB)^T", ["^T"]), ("QK^\\top", [])])
def test_the_transpose_ban_spares_upper_limits(tmp_path, math, banned):
    chapter = chapter_with(tmp_path, f"Math ${math}$ here.\n")
    found = [f.split("banned ")[1].split(" (")[0].strip("'") for f in banned_failures(chapter, 1)
             if "Transpose" in f]
    assert found == banned


def test_a_banned_phrase_is_reported_only_in_its_chapter(tmp_path):
    chapter = chapter_with(tmp_path, "Intro.\n\nEach buffer at capacity factor $c$ drops tokens.\n")
    assert phrase_failures(chapter, 6) == ["index.md:L3: banned phrase 'capacity factor $c$' (Capacity factor: c at "
                                           "CH6:354); write $\\phi$"]
    assert phrase_failures(chapter, 4) == []


def test_order_notation_in_prose_is_reported(tmp_path):
    chapter = chapter_with(tmp_path, "1. **O(1) seeking.** Fine: $\\mathcal{O}(1)$, `O(n)` and\n\n"
                                     "```python\nO(1)\n```\n")
    assert prose_failures(chapter) == ["index.md:L1: 'O(' in prose (Asymptotic order: write $\\mathcal{O}(\\cdot)$)"]


def test_a_prefix_is_not_the_same_phrase():
    assert not same_phrase("the number", "the number of keys")
    assert same_phrase("the per-head width", "per-head widths")


def test_an_undeclared_symbol_is_reported(tmp_path):
    chapter = chapter_with(tmp_path, "Intro.\n\nThe Jaccard similarity $J$ and $\\chi$.\n",
                           local="| Symbol | Meaning | First used | Rename |\n|---|---|---|---|\n"
                                 "| $J$ | Jaccard similarity | index.md:L3 |  |\n")
    assert membership_failures(chapter) == [
        "index.md:L3: symbol '\\\\chi' is in neither notation_global.txt nor notation-local.md (1 place(s))"]


def test_a_symbol_defined_twice_is_reported(tmp_path):
    chapter = chapter_with(tmp_path, "Here, where $d_k$ is the per-head width.\n\n"
                                     "Later, where $d_k$ is the number of keys.\n")
    failures = definition_failures(chapter)
    assert len(failures) == 1 and failures[0].startswith("index.md:L3: 'd_k' defined as 'the number of keys'")


def test_two_meanings_without_an_exception_are_reported(tmp_path):
    table = tmp_path / "g.txt"
    table.write_text("d\tmodel width\n", encoding="utf-8")
    chapter = chapter_with(tmp_path / "01-x", "x\n", local="| Symbol | Meaning | First used | Rename |\n"
                                                           "|---|---|---|---|\n| $d$ | distance | L1 |  |\n")
    none = tmp_path / "none.txt"
    failures = meaning_failures(table, none, [chapter])
    assert failures == ["'d' has 2 meanings and no exception: 'model width' (notation_global.txt); "
                        "'distance' (01-x/notation-local.md)"]
    none.write_text("d\tmodel width | distance\tbecause\n", encoding="utf-8")
    assert meaning_failures(table, none, [chapter]) == []
    none.write_text("d\tmodel width | other\tbecause\n", encoding="utf-8")
    assert len(meaning_failures(table, none, [chapter])) == 1
