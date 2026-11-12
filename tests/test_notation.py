"""Notation: one meaning per symbol, book-wide and in each chapter.

Check name `notation`, one test per chapter. Failures, in what each layer does and does not guarantee:
- Formula bans (BANNED) are exact: a pattern in a math region, scoped to chapters, from the plan's "Replaces"
  column; the bans built from NO_MEANING report a symbol that has no meaning in that chapter, read with the
  symbol grammar of tools/list_symbols.py. A wrong use the formula itself does not show is not caught here.
- Phrase entries (BANNED_PHRASES) are a regression guard for wording close to the recorded lines, nothing
  more: keyword cues near the symbol can miss a reworded wrong use and can match a correct sentence (which
  the chapter then declares in its NOTATION_OK). test_known_limits_of_the_phrase_layer_are_recorded keeps one
  example of each.
- "O(" in prose; a symbol in neither notation_global.txt nor the chapter's notation-local.md; a symbol with
  two different defining phrases ("where $x$ is ...") in the chapter.
Book-wide: no symbol has two meanings across the tables unless notation_exceptions.txt lists it; every table
symbol is in normal form. None of this confirms what a symbol means in a sentence: for a finished chapter a
person reads `python tools/list_symbols.py <chapter-dir> --concordance` against the tables.
"""
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, chapter_constants
from tests._pages import blank_code, blank_inline_code, blank_math
from tests.test_notebook_prose import prose_sentences
from tools.list_symbols import LOCAL_LIST, definitions, is_transpose, normalise, pages, symbols, symbols_in_math
from tools.unescape_math import math_regions

GLOBAL = ROOT / "tests" / "notation_global.txt"
EXCEPTIONS = ROOT / "tests" / "notation_exceptions.txt"
ALL = frozenset(range(0, 7))  # every chapter, and a directory without a chapter number (the sample chapter)



def _transpose(region, match):
    return is_transpose(region, match.start())


# (regex over the text of one math region, the global row it comes from, the chapters it applies to[, a filter
# (region, match) -> bool that keeps only the matches that are the banned use[, a regex the heading of the
# section must match]]). Chapter 5's L and the plain-letter uses that only prose identifies are BANNED_PHRASES.
# Rows marked "(proposed)" are additions from the Task 1 step 9 inventory, to be folded into the plan's table.
BANNED: list[tuple] = [
    (r"d_\{?(?:\\(?:text|mathrm)\{)?model", "Model (residual) width: d_model in ch. 6", frozenset({6})),
    (r"\\sqrt\{N\}", "Layers: N for residual layers at CH6:139", frozenset({6})),
    (r"\\text\{head\}_\{?h\b", "Attention heads: h as head count in ch. 1-2", frozenset({1, 2})),
    (r"\\cdot\s*h\s*\\cdot|\\,h\\,", "Attention heads: h as head count in ch. 2", frozenset({2})),
    (r"/\s*h(?![\w^_])", "Attention heads: h as head count at CH6:82", frozenset({6})),
    (r"d_\{?h(?![\w])", "Per-head width: d_h in ch. 6", frozenset({6})),
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
    (r"(?<![\\\w])[kv]_t\b", "(proposed) Index letters: k_t, v_t as positions at CH6:236", frozenset({6}), None,
     r"latent attention"),
    (r"(?<![\\\w])w_\{?(?:1|i|<i)\b|[qc]\(w", "(proposed) Tokens: w for tokens at CH4:462-466", frozenset({4})),
    (r"C\s*\+\s*\\alpha", "(proposed) Reference total: C at CH4:466", frozenset({4})),
    (r"8N|N\s*\\log\s*N", "(proposed) Suffix-array length: N at CH4:722", frozenset({4})),
    (r"J\(A,\s*B\)|h\([AB]\)|A\s*\\c[au]p\s*B", "(proposed) Shingle sets: A, B in ch. 4", frozenset({4})),
    (r"s\^r|k\s*=\s*br|/k(?![\w])|b_i|\\bmod\s*p\b", "(proposed) MinHash and LSH symbols in ch. 4",
     frozenset({4})),
    (r"\\alpha\\,\s*E", "(proposed) Auxiliary-loss weight: alpha at CH6:352", frozenset({6})),
]


# (chapter, symbol in normal form, what the chapter's tables use instead): symbols with no meaning in that chapter
# under notation_global.txt and the chapter's notation-local.md, each found in the chapter's math when this list
# was made. Each is banned exactly: a match counts only where list_symbols reads that symbol (so in chapter 2
# P, P \odot G and O = PV are banned; \Pr, p, p_\theta, \mathrm{PE} and PPL are not). Several are also caught by
# an older pattern above; these stay banned however the formula is written.
NO_MEANING: list[tuple[int, str, str]] = [
    (1, "P", "p_\\theta or p, a distribution"),
    (1, "W_1", "W_{gate}, W_{up}, W_{down}"), (1, "W_2", "W_{gate}, W_{up}, W_{down}"),
    (1, "W_3", "W_{gate}, W_{up}, W_{down}"),
    (1, "\\mathbf{x}", "h^{(0)}_i, the embedding"),
    (1, "b_1", "the plan's Biases row"), (1, "b_2", "the plan's Biases row"),
    (1, "b_{out}", "the plan's Unembedding row"), (1, "pos", "m, the position"),
    (2, "P", "A for attention weights, p for a distribution"),
    (2, "P_{\\theta}", "p_\\theta"), (2, "P_{data}", "p_{data}"),
    (2, "T_{\\max}", "t_{end}"), (2, "T_{warmup}", "t_{warm}"),
    (2, "Y", "\\mathrm{Y}"), (2, "a", "a_N"),
    (2, "\\theta_1", "\\theta^{(1)}"), (2, "h_1", "h^{(1)}"), (2, "h_L", "h^{(L)}"),
    (2, "n_{in}", "d_{in}"), (2, "n_{out}", "d_{out}"),
    (4, "w_1", "x_1"), (4, "w_N", "x_T"),
    (6, "P", "\\bar{\\rho}, the batch-mean router probability"),
    (6, "W_1", "W_{gate}, W_{up}, W_{down}"), (6, "W_2", "W_{gate}, W_{up}, W_{down}"),
    (6, "\\alpha", "\\lambda_{aux}"),
    (6, "\\theta_0", "\\omega_0"), (6, "\\theta_{d/2-1}", "\\omega_{d_k/2-1}"),
    (6, "d_h", "d_k"), (6, "d_{ffn}", "d_{ff}"), (6, "d_{model}", "d"),
]


def _symbol_at(symbol):
    """A BANNED filter: keep a match only where list_symbols reads exactly `symbol` starting at it."""
    return lambda region, match: (symbol, match.start()) in symbols_in_math(region)


def _symbol_start(symbol):
    """A regex for where `symbol` can start: its leading command, or its first letter not inside a word."""
    head = re.match(r"\\[A-Za-z]+|[A-Za-z]", symbol).group(0)
    return re.escape(head) if head.startswith("\\") else rf"(?<![\\A-Za-z]){head}"


BANNED += [(_symbol_start(sym), f"No meaning in ch. {chapter}: {sym} (write {instead})", frozenset({chapter}),
            _symbol_at(sym)) for chapter, sym, instead in NO_MEANING]


# Uses of a global symbol's spelling with another meaning that only the words around the math identify.
# Entry: (chapter, symbol in normal form, wrong cues, right cues, row, what to write). Cues are case-insensitive
# regexes searched in the words of a sentence or list-item unit, where a math span counts as one word. An
# occurrence of the symbol in a math span is a banned use when a wrong cue is within WRONG_WORDS words of it and
# no right cue is within RIGHT_WORDS words (its own math span is distance zero, so a formula can be its own cue).
# Right cues are the words of the symbol's global meaning; None when the symbol has no meaning in that chapter.
# tests/fixtures/banned_phrase_lines.txt holds, per entry index, the chapter lines each entry is for.
BANNED_PHRASES: list[tuple[int, str, str, str | None, str, str]] = [
    (1, "V", r"vocabular|\bsize\b", r"(?-i:^\s*[QK]\s*$)|\bvalue", "Vocabulary: V as a size at CH1:277",
     "$|\\mathcal{V}|$"),
    (2, "P", r"distribution|\bcode|encod|entropy", None, "Data and model distributions: P at CH2:104", "$p$"),
    (2, "Q", r"distribution|\bcode|encod|entropy", r"(?-i:^\s*[KV]\s*$)|\bquer", "Data and model distributions: Q at CH2:104",
     "$q$"),
    (2, "\\beta", r"momentum|velocity", r"\bshift|layernorm|normaliz", "Momentum coefficient: beta at CH2:183",
     "$\\beta_{\\text{mom}}$"),
    (2, "\\tau", r"\bclip|threshold|\bnorm\b", r"temperature|softmax|sampl", "Gradient-clipping threshold: tau at CH2:225",
     "$g_{\\max}$"),
    (2, "g", r"\\frac\{1\}\{\\sigma\}", r"upstream", "Upstream gradient: g_i, g_j in the LayerNorm gradient at CH2:325",
     "$\\tilde{g}_i$, $\\tilde{g}_j$"),
    (2, "L", r"\bloss", r"\blayers?\b|\bdepth", "Loss: L at CH2:398", "$\\mathcal{L}$"),
    (2, "V", r"vocabular|embedding", r"(?-i:^\s*[QK]\s*$)|\bvalue", "Vocabulary: V as a size at CH2:450",
     "$|\\mathcal{V}| d$"),
    (2, "h", r"\bheads?\b", r"hidden|\bstate|(?-i:h\^)", "Attention heads: h at CH2:451", "$n_h$"),
    (2, "s", r"scal", r"\bsource", "Loss-scaling factor: s at CH2:511-513", "$\\zeta$"),
    (4, "C", r"\bcount|\btotal|reference", r"compute|flop|budget", "(proposed) Reference total: C at CH4:466",
     "$T_{\\text{ref}}$"),
    (4, "k", r"\bhash|signature|position|\bband", r"\bkeys?\b|(?-i:k_)", "(proposed) MinHash: k at CH4:613-615",
     "$n_{\\text{hash}}$"),
    (4, "k", r"means", r"\bkeys?\b", "Index letters: k in k-means at CH4:760", "k-means, outside math"),
    (5, "L", r"length|sequences?\b|window|\bpad|longer|chunk|tokens?\b", r"\blayers?\b",
     "Sequence or context length: L in ch. 5", "$n$"),
    (6, "N", r"residual|\blayers?\b", r"parameter|\bcount|\brule|(?-i:\d\s*N)", "Layers: N at CH6:139",
     "$2L$, the number of residual layers"),
    (6, "W", r"window|previous|attend|receptive|\bspan", r"weight|matri|projection|(?-i:W_|W\^)",
     "Sliding-window span: W at CH6:246", "$n_{\\text{win}}$"),
    (6, "n", r"\bkey\b|\bpositions?\b", r"\btokens?\b|length|context|sequence|cache",
     "Positions: n as key position at CH6:261", "$m'$"),
    (6, "L", r"\btrained\b|\bseen\b|sequences?\b|beyond|\bup\b|\btokens\b", r"\blayers?\b|\bdepth",
     "Trained context length: L at CH6:265", "$n_{\\text{train}}$"),
    (6, "s", r"scale|context|frequenc|length", r"\bsource", "Context scale factor and context length: s at CH6:271-275, :502",
     "$s_{\\text{ctx}}$ (scale factor), $n$ (context length)"),
    (6, "k", r"expert|select|(?<!\\)\btop\b|top-|best|rout|choos", r"\bkeys?\b|query|(?-i:k_)",
     "Experts active per token: k at CH6:320-362", "$n_{\\text{act}}$"),
    (6, "c", r"capacity", r"latent|(?-i:c_)", "Capacity factor: c at CH6:354", "$\\phi$"),
    (5, "r", r"\brank|worker|process", r"router|logit", "Index letters: r as a worker rank at CH5:534", "worker $j$"),
]
WRONG_WORDS = 6
RIGHT_WORDS = 3
HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.*)$", re.M)
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
    active = [(re.compile(entry[0]), entry[1], entry[3] if len(entry) > 3 else None,
               re.compile(entry[4], re.I) if len(entry) > 4 else None)
              for entry in BANNED if number in entry[2]]
    failures = []
    for page in pages(chapter_dir):
        text = page.read_text(encoding="utf-8")
        headings = [(m.start(), m.group(1)) for m in HEADING.finditer(blank_code(text))]
        for start, end in math_regions(text):
            region = text[start:end]
            section = next((h for pos, h in reversed(headings) if pos < start), "")
            for pattern, row, keep, where in active:
                if where and not where.search(section):
                    continue
                for m in pattern.finditer(region):
                    if keep and not keep(region, m):
                        continue
                    line = text.count("\n", 0, start) + 1 + region.count("\n", 0, m.start())
                    failures.append(f"{page.name}:L{line}: banned {m.group(0)!r} ({row})")
    return failures


def _words(sentence):
    """(start, end, text, is_math) for each word of a sentence: a math span is one word; prose is split at white
    space, and chunks without a letter or digit (a lone comma) are not words."""
    words, pos = [], 0
    for a, b in math_regions(sentence) + [(len(sentence), len(sentence))]:
        for m in re.finditer(r"\S+", sentence[pos:a]):
            if re.search(r"\w", m.group(0)):
                words.append((pos + m.start(), pos + m.end(), m.group(0), False))
        if a < b:
            words.append((a, b, sentence[a:b].strip("$"), True))
        pos = b
    return words


def phrase_hits(text, entries):
    """(line, entry index, math, unit) for each banned use in a page's text (see BANNED_PHRASES)."""
    for line, sentence in prose_sentences(text):
        words = _words(sentence)
        for index, (_chapter, symbol, wrong, right, _row, _instead) in entries:
            wrong_re = re.compile(wrong, re.I)
            right_re = re.compile(right, re.I) if right else None
            for k, (a, _b, math, is_math) in enumerate(words):
                if not is_math or symbol not in normalise(math):
                    continue
                near = words[max(0, k - WRONG_WORDS):k + WRONG_WORDS + 1]
                close = words[max(0, k - RIGHT_WORDS):k + RIGHT_WORDS + 1]
                if any(wrong_re.search(w[2]) for w in near) and not (
                        right_re and any(right_re.search(w[2]) for w in close)):
                    yield line + sentence.count("\n", 0, a), index, sentence[a:_b], sentence
                    break


def notation_ok(chapter_dir):
    """The chapter's NOTATION_OK declarations (unit regex -> reason) from its tests/test_chNN_snippets.py."""
    resolved = Path(chapter_dir).resolve()
    if resolved not in {d.resolve() for d in CHAPTER_DIRS}:
        return {}
    return dict(chapter_constants(chapter_dir).NOTATION_OK)


def phrase_failures(chapter_dir, number=None, allowed=None):
    number = chapter_number(chapter_dir) if number is None else number
    allowed = notation_ok(chapter_dir) if allowed is None else allowed
    entries = [(i, e) for i, e in enumerate(BANNED_PHRASES) if e[0] == number]
    failures, used = [], set()
    for page in pages(chapter_dir):
        for line, index, math, unit in phrase_hits(page.read_text(encoding="utf-8"), entries):
            declared = [p for p in allowed if re.search(p, unit)]
            if declared:
                used.update(declared)
                continue
            row, instead = BANNED_PHRASES[index][4:6]
            failures.append(f"{page.name}:L{line}: banned use of {math!r} ({row}); write {instead}")
    for pattern, reason in allowed.items():
        if pattern not in used:
            failures.append(f"NOTATION_OK {pattern!r} ({reason}) matches no reported unit; remove it")
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


PHRASE_LINES = ROOT / "tests" / "fixtures" / "banned_phrase_lines.txt"

# Per BANNED_PHRASES index: fix (replacements that write the plan's symbol), wrong (rewordings that keep the
# wrong symbol and must match), right (sentences using the symbol with its correct meaning, which must not
# match), cue (a literal wrong cue and right cue for the property test; right is None when the symbol has no
# meaning in that chapter).
PHRASE_CASES = {
    0: dict(fix=[("^{V \\times d}", "^{|\\mathcal{V}| \\times d}"), ("$V$", "$|\\mathcal{V}|$")],
            wrong=["Row $i$ belongs to token $i$ of a vocabulary of $V$ entries.",
                   "The table has $V$ rows, one per vocabulary item."],
            right=["Project the vocabulary embeddings to $Q$, $K$ and $V$."], cue=("vocabulary", "$K$")),
    1: dict(fix=[("$P$", "$p$")],
            wrong=["The cross-entropy relative to the distribution $P$ counts bits.",
                   "Encoding data from $P$ costs extra bits."],
            right=["The softmax output $P$ feeds the next layer."], cue=("distribution", None)),
    2: dict(fix=[("$Q$", "$q$")],
            wrong=["The cross-entropy of $Q$ relative to the true distribution counts bits.",
                   "A code built for $Q$ costs extra bits."],
            right=["The positional encoding is added before $Q$, $K$ and $V$ are computed."], cue=("distribution", "$K$")),
    3: dict(fix=[("$\\beta$", "$\\beta_{\\text{mom}}$")],
            wrong=["The momentum coefficient $\\beta$ (typically 0.9) sets the memory.",
                   "With $\\beta = 0.9$ the velocity averages about ten gradients."],
            right=["LayerNorm's shift $\\beta$, unlike the momentum coefficient, is learned."], cue=("momentum", "shift")),
    4: dict(fix=[("$\\tau$", "$g_{\\max}$")],
            wrong=["where $\\tau$ is the clip threshold (typically 1.0).", "Gradients whose norm exceeds $\\tau$ are rescaled."],
            right=["Dividing the logits by the temperature $\\tau$ flattens the softmax; clipping is separate."],
            cue=("threshold", "temperature")),
    5: dict(fix=[("g_i", "\\tilde{g}_i"), ("g_j", "\\tilde{g}_j")],
            wrong=["$$\\frac{\\partial \\mathcal{L}}{\\partial x_i} = \\frac{1}{\\sigma}\\left(g_i - \\frac{1}{d}\\sum_j g_j "
                   "(1 + \\hat{x}_i \\hat{x}_j)\\right)$$",
                   "To first order the input gradient is $\\frac{1}{\\sigma}(g - \\bar{g})$."],
            right=["The upstream gradient $g$ is then scaled by $\\frac{1}{\\sigma}$."], cue=("$\\frac{1}{\\sigma}$", "upstream")),
    6: dict(fix=[("$L$", "$\\mathcal{L}$")],
            wrong=["The test loss $L$ falls as a power law in $N$.", "$L$, the loss, scales with the parameter count."],
            right=["The loss gradient flows back through all $L$ layers."], cue=("loss", "layers")),
    7: dict(fix=[("V d", "|\\mathcal{V}| d")],
            wrong=["The embedding table holds $V d$ weights.", "Add $V d$ for the vocabulary."],
            right=["The embeddings are projected to $Q$, $K$ and $V$ by three matrices."], cue=("embedding", "$K$")),
    8: dict(fix=[("\\cdot h \\cdot", "\\cdot n_h \\cdot"), ("$h$", "$n_h$"), ("\\,h\\,", "\\,n_h\\,")],
            wrong=["where $h$ is the number of heads", "With $h = 12$ attention heads the scores grow."],
            right=["The language-model head maps the final hidden state $h^{(L)}_i$ to logits."], cue=("heads", "hidden")),
    9: dict(fix=[("$s$", "$\\zeta$"), ("\\times s$", "\\times \\zeta$")],
            wrong=["Multiply the loss by a scaling factor $s$ first.", "$s$ scales the loss before the backward pass."],
            right=["Each source $s$ has its own scaled weight."], cue=("scale", "source")),
    10: dict(fix=[("$C$", "$T_{\\text{ref}}$"), ("(C +", "(T_{\\text{ref}} +")],
             wrong=["where $C$ is the total count of reference tokens", "divided by the reference total $C$"],
             right=["Under a fixed total compute $C$ the best model is smaller."], cue=("total", "compute")),
    11: dict(fix=[("$k$", "$n_{\\text{hash}}$"), ("/k$", "/n_{\\text{hash}}$"), ("k = br", "n_{\\text{hash}} = br")],
             wrong=["Use $k$ hash functions per document.", "The signature holds $k$ minimum values, split into bands."],
             right=["Each hash bucket is looked up by its key $k$."], cue=("hash", "key")),
    12: dict(fix=[("$k$-means", "k-means")],
             wrong=["cluster the embeddings with $k$ means", "run the $k$-means algorithm on the embeddings"],
             right=["The key $k$ means nothing to the clustering step."], cue=("means", "key")),
    13: dict(fix=[("$L$", "$n$")],
             wrong=["Every training sequence holds $L$ tokens.", "Cut the stream into windows of $L$ tokens each."],
             right=["A model with $L$ layers processes a sequence of length $n$."], cue=("length", "layers")),
    14: dict(fix=[("\\sqrt{N}", "\\sqrt{2L}"), ("with $N$ the number", "with $2L$ the number")],
             wrong=["where $N$ is the number of residual layers", "GPT-2 divides by $\\sqrt{N}$ for $N$ residual layers."],
             right=["An MoE layer raises the parameter count $N$ without raising compute.",
                    "the $6N$ rule counts $N$ over all layers"], cue=("residual", "parameter")),
    15: dict(fix=[("$W$", "$n_{\\text{win}}$"), ("nW", "n\\,n_{\\text{win}}"), ("L \\cdot W", "L \\cdot n_{\\text{win}}"),
                  ("W = 4096", "n_{\\text{win}} = 4096")],
             wrong=["Each token attends to the previous $W$ tokens only.", "With a window of $W$ positions the cost is linear."],
             right=["Inside the window, attention uses the projection $W^{UK}$ as before."], cue=("window", "projection")),
    16: dict(fix=[("$n$ then", "$m'$ then"), ("m - n", "m - m'")],
             wrong=["the dot product between a query at $m$ and a key at $n$", "a key whose position is $n$"],
             right=["The key and value cache holds $2\\,n\\,n_{kv} d_k$ bytes for a context of $n$ tokens."],
             cue=("key", "tokens")),
    17: dict(fix=[("$L$", "$n_{\\text{train}}$"), ("m < L", "m < n_{\\text{train}}"),
                  ("L\\,\\theta_i", "n_{\\text{train}}\\,\\theta_i"), ("m > L", "m > n_{\\text{train}}")],
             wrong=["A model trained on sequences up to length $L$ never rotates further.",
                    "Positions beyond $L$ were never seen in training.", "It was trained on sequences of up to $L$ tokens."],
             right=["Llama 2 7B has $L = 32$ layers and was trained on 2T tokens."], cue=("trained", "layers")),
    18: dict(fix=[("exactly $s$", "exactly $s_{\\text{ctx}}$"), ("context length $s$", "context length $n$"),
                  ("L\\,s\\,d", "L\\,n\\,d")],
             wrong=["The context scale factor $s$ divides the lowest frequency.",
                    "The extra term grows with the context length $s$."],
             right=["Each source $s$ is scaled by its weight."], cue=("scale", "source")),
    19: dict(fix=[("top-$k$", "top-$n_{\\text{act}}$"), ("$k$", "$n_{\\text{act}}$")],
             wrong=["The router keeps the best $k$ experts for each token.", "Only the top $k$ experts receive the token.",
                    "A router that selects $k$ of them."],
             right=["Show that $q_t^\\top k_s = c_s^\\top W$ holds for the key.", "The router does not see the key $k$."],
             cue=("experts", "key")),
    20: dict(fix=[("$c$", "$\\phi$"), ("\\lceil c\\,", "\\lceil \\phi\\,")],
             wrong=["Each expert has a capacity factor of $c$.", "With $c = 1.25$ the buffer has spare capacity."],
             right=["The latent $c_i$ has less capacity than full keys."], cue=("capacity", "latent")),
    21: dict(fix=[("rank $r$", "rank $j$")],
             wrong=["Worker $r$ reads its own slice.", "The process with rank $r$ opens its shard."],
             right=["The router logit $r$ of each expert is a dot product."], cue=("rank", "router")),
}


def phrase_lines():
    """{entry index: [line, ...]} from tests/fixtures/banned_phrase_lines.txt."""
    lines = {}
    for row in PHRASE_LINES.read_text(encoding="utf-8").splitlines():
        index, _, line = row.partition("\t")
        lines.setdefault(int(index), []).append(line)
    return lines


def _hits(index, text):
    return [h for h in phrase_hits(text + "\n", [(index, BANNED_PHRASES[index])])]


@pytest.mark.parametrize("index", range(len(BANNED_PHRASES)),
                         ids=lambda i: f"ch{BANNED_PHRASES[i][0]}-{i}")
def test_every_banned_phrase_matches_its_lines_and_not_their_fix(index):
    """Each recorded line matches; with the plan's symbol written in it does not; rewordings that keep the wrong
    symbol match; sentences with the symbol's correct meaning do not."""
    lines = phrase_lines().get(index)
    assert lines, f"no line for entry {index} in {PHRASE_LINES.name}"
    case = PHRASE_CASES[index]
    for line in lines:
        assert _hits(index, line), f"entry {index} does not match its line: {line[:80]!r}"
        fixed = line
        for old, new in case["fix"]:
            fixed = fixed.replace(old, new)
        assert not _hits(index, fixed), f"entry {index} matches the corrected line: {fixed[:120]!r}"
    assert len(case["wrong"]) >= 2 and case["right"]
    for text in case["wrong"]:
        assert _hits(index, text), f"entry {index} misses the rewording {text!r}"
    for text in case["right"]:
        assert not _hits(index, text), f"entry {index} matches a correct use: {text!r}"


@pytest.mark.parametrize("index", range(len(BANNED_PHRASES)),
                         ids=lambda i: f"ch{BANNED_PHRASES[i][0]}-{i}")
def test_a_wrong_cue_beside_the_symbol_matches_and_a_right_cue_does_not(index):
    wrong, right = PHRASE_CASES[index]["cue"]
    symbol = BANNED_PHRASES[index][1]
    assert _hits(index, f"{wrong} ${symbol}$")
    if right is not None:
        assert not _hits(index, f"{right} ${symbol}$")
        assert not _hits(index, f"{wrong} {right} ${symbol}$")


def test_the_phrase_cases_cover_every_entry():
    assert set(PHRASE_CASES) == set(range(len(BANNED_PHRASES))) == set(phrase_lines())


def test_every_chapter_declares_notation_ok():
    for chapter_dir in CHAPTER_DIRS:
        assert isinstance(chapter_constants(chapter_dir).NOTATION_OK, dict)


def test_a_declared_unit_is_not_reported_and_an_unused_declaration_is(tmp_path):
    chapter = chapter_with(tmp_path, "Each buffer at capacity factor $c$ drops tokens.\n")
    assert phrase_failures(chapter, 6, allowed={r"buffer at capacity": "quoted from the paper"}) == []
    failures = phrase_failures(chapter, 6, allowed={r"no such unit": "stale"})
    assert len(failures) == 2 and "NOTATION_OK 'no such unit' (stale) matches no reported unit" in failures[1]


def test_global_symbols_pass_every_math_ban(tmp_path):
    """Each global symbol, and each form its meaning gives as an example (v_t, q_m), written alone in every
    section without a scoped ban, passes every chapter's math bans, unless a row replaces it there."""
    replaced = {("W", 5)}  # chapter 5's world size: W is banned there, the generic weight matrix is not used
    forms = set()
    for sym, meaning in global_rows():
        forms.add(sym)
        for token in re.findall(r"\((\S+)", meaning):
            token = token.rstrip(")")
            if re.search(r"[_\\^]", token) and token.count("{") == token.count("}"):
                forms.add(token)
    page = "# Notation\n\n" + "\n\n".join(f"Here ${form}$ alone." for form in sorted(forms)) + "\n"
    chapter = chapter_with(tmp_path, page)
    hits = []
    for number in range(1, 7):
        for failure in banned_failures(chapter, number):
            line = int(failure.split(":L")[1].split(":")[0])
            form = page.split("\n")[line - 1].split("$")[1]
            if (form, number) not in replaced:
                hits.append(f"ch{number}: {failure}")
    assert not hits, "\n".join(hits)


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
    assert banned_failures(chapter, 6) == ["index.md:L1: banned 'd_{model' (Model (residual) width: d_model in ch. 6)",
                                           "index.md:L1: banned 'd' (No meaning in ch. 6: d_{model} (write d))"]
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
    assert phrase_failures(chapter, 6) == ["index.md:L3: banned use of '$c$' (Capacity factor: c at CH6:354); "
                                           "write $\\phi$"]
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


# --- symbols with no meaning in a chapter (NO_MEANING) ------------------------------------------------------

def _banned(tmp_path, math, number):
    chapter = chapter_with(tmp_path, f"Math ${math}$ here.\n")
    return [f for f in banned_failures(chapter, number) if "No meaning" in f]


@pytest.mark.parametrize("number,symbol", [(n, s) for n, s, _ in NO_MEANING],
                         ids=[f"ch{n}-{s}" for n, s, _ in NO_MEANING])
def test_a_symbol_with_no_meaning_in_its_chapter_is_banned_there_alone(tmp_path, number, symbol):
    known = {sym for sym, _ in global_rows()} | {sym for sym, _ in local_rows(CHAPTER_DIRS[number - 1])}
    assert symbol not in known, f"{symbol!r} has a meaning in chapter {number}: no ban"
    assert normalise(symbol) == [symbol]
    assert len(_banned(tmp_path, symbol, number)) == 1
    banned_in = {c for c, s, _ in NO_MEANING if s == symbol}
    assert all(_banned(tmp_path, symbol, n) == [] for n in range(1, 7) if n not in banned_in)


@pytest.mark.parametrize("math,banned", [
    ("P", True), ("P \\odot G", True), ("O = PV", True), ("P_i", True),
    ("\\Pr[A]", False), ("p", False), ("p_\\theta", False), ("\\mathrm{PE}", False), ("PPL", False),
    ("\\mathbf{P}", False)])
def test_bare_p_is_banned_in_chapter_2(tmp_path, math, banned):
    found = [f for f in _banned(tmp_path, math, 2) if ": P (" in f]
    assert bool(found) == banned, found


def test_p_theta_is_banned_in_chapter_2_by_the_distribution_row_and_exactly(tmp_path):
    # The older pattern for distributions catches P_{\theta} but not P_\theta; the exact ban catches both.
    chapter = chapter_with(tmp_path, "Math $P_{\\theta}$ and $P_\\theta(x)$ here.\n")
    failures = banned_failures(chapter, 2)
    assert len([f for f in failures if "Data and model distributions" in f and "'P_{'" in f]) == 1
    assert len([f for f in failures if "No meaning in ch. 2: P_{\\theta}" in f]) == 2
    assert not [f for f in failures if ": P (" in f]


def test_known_limits_of_the_phrase_layer_are_recorded():
    """Two sentences an independent reviewer wrote. This records a limit, not a requirement: keyword cues cannot
    tell these apart, and the control for both is a person reading the --concordance of tools/list_symbols.py."""
    def hits(number, text):
        return list(phrase_hits(text + "\n", [(i, e) for i, e in enumerate(BANNED_PHRASES) if e[0] == number]))
    # A correct use that matches: L is the number of layers, but "tokens" is a wrong cue for chapter 6's L.
    assert hits(6, "Each of the $L$ blocks processes all $n$ tokens.")
    # A wrong use that escapes: tau as the clipping threshold, with none of the cues clip, threshold or norm.
    assert not hits(2, "Gradients are rescaled whenever their magnitude exceeds $\\tau$.")
