#!/usr/bin/env python3
"""List the mathematical symbols a chapter uses, with the page and line of every occurrence.

Usage:
    python tools/list_symbols.py chapters/02-mathematics
    python tools/list_symbols.py chapters/02-mathematics --definitions
    python tools/list_symbols.py chapters/02-mathematics --concordance [--symbol 'd_k'] [--undeclared]

--concordance prints, per symbol (in the order of tests/notation_global.txt, then the chapter's local list,
then the undeclared ones), a header "== <symbol> — <meaning or UNDECLARED> ==" and one line
"<page>:L<line>: <unit>" per sentence, list item or display block that uses it. A person reads it against the
tables to confirm each symbol's meaning in a finished chapter; tests/test_notation.py cannot.

Math regions are found with tools.unescape_math.math_regions (fenced code excluded). The pages are the
*.md files directly in the chapter directory, except notation-local.md.

Symbol grammar
    A symbol is a Latin or Greek base letter, keeping any \\mathcal, \\mathbf, \\mathrm or \\mathbb wrapper
    and any \\tilde, \\bar or \\hat accent, plus its subscript after stripping \\text{}, \\mathrm{} and
    braces. A subscript that is only a running index (one or two of i, j, l, t, m, s, or an index
    expression such as i-1, 2i+1 or m') is dropped, unless it is written as text (\\epsilon_{\\text{ls}} is
    \\epsilon_{ls}, \\epsilon_{ls} is \\epsilon). Superscripts and primes are not part of the symbol;
    the symbols inside a superscript are listed on their own, except a label in LABELS (W^{UK} is W) and a
    transpose (T or \\top after a symbol, a bracket or \\mathbf{1}); the T over a big operator (\\sum, \\prod,
    \\bigoplus, \\lim, ..., with or without \\limits, scripts in either order) is a limit and is listed. A power
    such as \\gamma^T cannot be told from a transpose and is read as one. A run of three or more letters that is not an operator word is one name (pos). A subscript holding a relation or range
    (x_{<i}, \\mathbb{E}_{x \\sim p}, w_{1:N}) is a condition: not part of the symbol, its symbols listed. Operator and function names (OPERATOR_NAMES,
    \\operatorname{...}, \\text{...}, multi-letter \\mathrm{...} and the bare words in OPERATOR_WORDS) are
    not symbols; the symbols in their subscripts and arguments are.

    Normal form: base, then "_x" for a one-character subscript or "_{xy}" for a longer one, e.g.
    W_{gate}, d_k, \\tilde{\\rho}, \\mathcal{L}, \\lambda_{\\max}.

Defining phrases (--definitions): "where $x$ is|are|denotes ...", "$x$ denotes ...", "with $x$ the
...", "let $x$ be ...". The phrase runs to the next comma, semicolon, colon, full stop, parenthesis,
dollar sign, brace, backtick, "and" or line end, and must be a noun phrase (it starts with
the, a, an or a noun). The math must hold exactly one symbol and no relation. The symbol is the first one in the inline math.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.unescape_math import math_regions  # noqa: E402

LOCAL_LIST = "notation-local.md"

# Operator and function names: commands that are never symbols. Any other command that is not a Greek
# letter, a wrapper or an accent is ignored as well (\frac, \cdot, \left, ...); this list documents the
# names a reader might otherwise take for symbols.
OPERATOR_NAMES = {
    "log", "ln", "exp", "max", "min", "sum", "prod", "arg", "argmax", "argmin", "lim", "sup", "inf",
    "sin", "cos", "tan", "det", "dim", "deg", "gcd", "Pr", "int", "partial", "nabla", "infty", "top",
}
# Commands whose single braced argument is a name or a layout instruction, never symbols.
NAME_COMMANDS = {
    "operatorname", "text", "textrm", "textit", "textbf", "texttt", "textsf", "mbox", "mathit", "mathsf",
    "mathtt", "begin", "end", "label", "tag", "hspace", "vspace", "color", "phantom", "hphantom",
}
# Bare words written in math as function names (softmax(x), RMSNorm(x), ...).
OPERATOR_WORDS = {
    "softmax", "argmax", "argmin", "sigmoid", "Swish", "SwiGLU", "GELU", "ReLU", "GLU", "SiLU", "LayerNorm",
    "RMSNorm", "Norm", "FFN", "MLP", "MoE", "Attention", "MultiHead", "MHA", "Concat", "head", "Linear",
    "Embed", "PPL", "PE", "RoPE", "ALiBi", "TopK", "topk", "TopP", "mean", "std", "Var", "var", "rank",
    "rms", "RMS", "erf", "tanh", "sign", "diag", "tr", "Tr", "len", "Dropout", "KL", "CE", "BPE", "BPC",
    "IDF", "TF", "idf", "tf", "Jaccard", "MinHash", "minhash", "hash", "bucket", "doc", "rb", "clip",
    "round", "floor", "ceil", "mod", "lr", "loss", "logits", "logit", "Adam", "AdamW", "SGD", "nats",
    "bits", "bytes", "GB", "TB", "MB", "KB", "FLOPs", "FLOP", "PF", "days", "tokens",
}
WRAPPERS = {"mathcal", "mathbf", "mathrm", "mathbb"}
ACCENTS = {"tilde", "bar", "hat", "widetilde", "widehat", "overline"}
ACCENT_NORMAL = {"widetilde": "tilde", "widehat": "hat", "overline": "bar"}
GREEK = {
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "vartheta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
    "Sigma", "Upsilon", "Phi", "Psi", "Omega", "ell",
}
INDEX_LETTERS = set("ijltms")
INDEX = re.compile(r"[ijltms]{1,2}|(?=.*[ijltms])[ijltms0-9+\-',]+")
# Superscript labels that are part of a symbol's name, as (base, label) pairs: the ones chapter 6 uses today
# (W^{UK}, W^{UV}, W^{DKV}, d_h^{R}). x^{R} is not one, so its R is listed. Their content is not listed; every other superscript is scanned for symbols.
LABELS = {("W", "UK"), ("W", "UV"), ("W", "DKV"), ("d_h", "R")}
# A superscript T or \top on a symbol, a bracket or \mathbf{1} is a transpose, not the symbol T; on a big operator
# (\sum, \prod, \bigcup, \bigcap, \lim, ...) it is an upper limit and is scanned.
TRANSPOSE = re.compile(r"\s*(?:T|\\top)\s*")
LIMIT_OPERATORS = re.compile(r"\\(?:sum|prod|coprod|int|bigcup|bigcap|bigoplus|bigotimes|bigvee|bigwedge|lim|limsup|"
                             r"liminf|max|min|sup|inf)\s*$")
# A run of three or more Latin letters that is not an operator word is one name (pos), not a product of symbols.
MIN_NAME_RUN = 3
# A subscript holding a relation or range is a condition (x_{<i}, \mathbb{E}_{x \sim p}, w_{1:N}): it is
# not part of the symbol, and the symbols in it are listed on their own.
CONDITION = re.compile(r"[<>=:]|\\(?:sim|in|le|leq|ge|geq|lt|gt|mid)(?![A-Za-z])")
# A subscript written as text is a name, never a running index: \\epsilon_{\\text{ls}} is \\epsilon_{ls}.
TEXT_SUBSCRIPT = re.compile(r"\s*\\(?:text|mathrm|textrm|operatorname)\s*\{[^{}]*\}\s*")
STRIP_IN_SUBSCRIPT = re.compile(r"\\(?:text|mathrm|textrm|mathit|operatorname)\s*(?=\{)")


class _Parser:
    """Walks one math string and records (symbol, offset) pairs."""

    def __init__(self, text: str):
        self.s = text
        self.found: list[tuple[str, int]] = []

    # --- reading primitives -------------------------------------------------------------
    def _skip_space(self, i: int) -> int:
        while i < len(self.s) and self.s[i].isspace():
            i += 1
        return i

    def _group_end(self, i: int) -> int:
        """Index just past the {...} group opening at i."""
        depth = 0
        while i < len(self.s):
            if self.s[i] == "\\":
                i += 2
                continue
            depth += self.s[i] == "{"
            depth -= self.s[i] == "}"
            i += 1
            if depth == 0:
                return i
        return i

    def _arg(self, i: int) -> tuple[int, int, int]:
        """(start, end, next): the argument at i, braces excluded; next is the index after it."""
        i = self._skip_space(i)
        if i >= len(self.s):
            return i, i, i
        if self.s[i] == "{":
            end = self._group_end(i)
            return i + 1, end - 1, end
        if self.s[i] == "\\":
            m = re.compile(r"\\([A-Za-z]+|.)").match(self.s, i)
            end = m.end()
            if m.group(1) in NAME_COMMANDS | WRAPPERS | ACCENTS:  # x_\text{gate}: the command takes its group
                j = self._skip_space(end)
                end = self._group_end(j) if j < len(self.s) and self.s[j] == "{" else self._arg(end)[2]
            return i, end, end
        return i, i + 1, i + 1

    # --- grammar ------------------------------------------------------------------------
    def parse(self, start: int = 0, end: int | None = None) -> None:
        end = len(self.s) if end is None else end
        i = start
        while i < end:
            ch = self.s[i]
            if ch == "\\":
                m = re.compile(r"\\([A-Za-z]+|.)").match(self.s, i)
                name = m.group(1)
                if name in GREEK:
                    i = self._symbol("\\" + name, i, m.end())
                elif name in WRAPPERS or name in ACCENTS:
                    i = self._decorated(name, i, m.end())
                elif name in NAME_COMMANDS:
                    i = self._name(self._arg(m.end())[2])
                else:
                    i = m.end()
            elif ch.isascii() and ch.isalpha():
                run = re.compile(r"[A-Za-z]+").match(self.s, i).group(0)
                if run in OPERATOR_WORDS:
                    i = self._name(i + len(run))
                elif len(run) >= MIN_NAME_RUN:  # pos: a name written as a word; listed once, as itself
                    self.found.append((run, i))
                    i = self._name(i + len(run))
                else:
                    i = self._symbol(ch, i, i + 1)
            elif ch.isalpha():
                i = self._symbol(ch, i, i + 1)
            elif ch in "_^":
                i = self._name(i)  # a script with no base: scan its content
            else:
                i += 1

    def _name(self, i: int) -> int:
        """After an operator or function name: its scripts are scanned for symbols, never attached."""
        while True:
            j = self._skip_space(i)
            if j < len(self.s) and self.s[j] in "_^":
                a, b, i = self._arg(j + 1)
                if not (self.s[j] == "^" and is_transpose(self.s, j)):
                    self.parse(a, b)
            elif j < len(self.s) and self.s[j] == "'":
                i = j + 1
            else:
                return i

    def _decorated(self, name: str, at: int, i: int) -> int:
        a, b, nxt = self._arg(i)
        inner = _Parser(self.s[a:b])
        inner.parse()
        bases = inner.found
        if name == "mathrm" and not re.fullmatch(r"\s*([A-Za-z]|\\[A-Za-z]+)\s*", self.s[a:b]):
            return self._name(nxt)  # \mathrm{softmax}, \mathrm{FFN}: a name
        if len(bases) != 1:
            for sym, off in bases:
                self.found.append((sym, a + off))
            return self._scripts(None, at, nxt)
        base = bases[0][0]
        if name in ACCENTS:
            return self._symbol(f"\\{ACCENT_NORMAL.get(name, name)}{{{base}}}", at, nxt)
        if "_" in base:  # \mathbf{W_Q}: keep the wrapper on the letter only
            letter, sub = base.split("_", 1)
            return self._symbol(f"\\{name}{{{letter}}}_{sub}", at, nxt, attached=True)
        return self._symbol(f"\\{name}{{{base}}}", at, nxt)

    def _symbol(self, base: str, at: int, i: int, attached: bool = False) -> int:
        return self._scripts(base, at, i, attached)

    def _scripts(self, base: str | None, at: int, i: int, attached: bool = False) -> int:
        subscript, named = None, False
        while True:
            j = self._skip_space(i)
            if j < len(self.s) and self.s[j] == "'":
                i = j + 1
            elif j < len(self.s) and self.s[j] == "^":
                a, b, i = self._arg(j + 1)
                label = re.sub(r"\\top|\s", "", self.s[a:b])
                named_base = base if not subscript else base + (f"_{subscript}" if len(subscript) == 1
                                                                 else f"_{{{subscript}}}") if base else None
                if not (TRANSPOSE.fullmatch(self.s[a:b]) or (named_base, label) in LABELS):
                    self.parse(a, b)
            elif j < len(self.s) and self.s[j] == "_" and subscript is None:
                a, b, i = self._arg(j + 1)
                subscript = normal_subscript(self.s[a:b])
                named = bool(TEXT_SUBSCRIPT.fullmatch(self.s[a:b]))
                if base is None or CONDITION.search(self.s[a:b]):
                    self.parse(a, b)
                    subscript = ""
            else:
                break
        if base is not None:
            if subscript and not attached and (named or not INDEX.fullmatch(subscript)):
                base += f"_{subscript}" if len(subscript) == 1 else f"_{{{subscript}}}"
            self.found.append((base, at))
        return i


def _script_head(text: str, caret: int) -> str:
    """The text before the script chain that the superscript at `caret` belongs to.

    Scripts already in the chain (either order) and \\limits or \\nolimits are skipped, so the result ends
    with the chain's base: \\sum in \\sum\\limits_{i=1}^{T}, x in x_i^T.
    """
    before = text[:caret]
    while True:
        before = before.rstrip()
        m = re.search(r"\\(?:no)?limits$", before)
        if m:
            before = before[:m.start()]
            continue
        if before.endswith("}"):
            depth, k = 0, len(before) - 1
            while k >= 0:
                depth += before[k] == "}"
                depth -= before[k] == "{"
                if depth == 0:
                    break
                k -= 1
            head = before[:k].rstrip()
            if head.endswith(("_", "^")):
                before = head[:-1]
                continue
            return before
        m = re.search(r"[_^]\s*(?:\\[A-Za-z]+|[A-Za-z0-9])$", before)
        if m:
            before = before[:m.start()]
            continue
        return before


def is_transpose(text: str, caret: int) -> bool:
    """True when the superscript at text[caret] == '^' is a transpose (T or \\top), not a big operator's limit."""
    m = re.compile(r"\^\s*(?:\{\s*(?:T|\\top)\s*\}|T(?![A-Za-z])|\\top(?![A-Za-z]))").match(text, caret)
    return bool(m) and not LIMIT_OPERATORS.search(_script_head(text, caret))


def normal_subscript(text: str) -> str:
    """The subscript with \\text{}, \\mathrm{} (and the like), braces and spaces removed."""
    text = STRIP_IN_SUBSCRIPT.sub("", text)
    text = re.sub(r"[{}\s]", "", text)
    return text


def symbols_in_math(math: str) -> list[tuple[str, int]]:
    """(symbol, offset) for every symbol occurrence in a math string, in order."""
    parser = _Parser(math)
    parser.parse()
    return sorted(parser.found, key=lambda item: item[1])


def normalise(math: str) -> list[str]:
    """The symbols of a math string, in order of appearance."""
    return [sym for sym, _ in symbols_in_math(math)]


def pages(chapter_dir) -> list[Path]:
    return sorted(p for p in Path(chapter_dir).glob("*.md") if p.name != LOCAL_LIST)


def _occurrences(page: Path):
    text = page.read_text(encoding="utf-8")
    for start, end in math_regions(text):
        line0 = text.count("\n", 0, start) + 1
        region = text[start:end]
        for sym, off in symbols_in_math(region):
            yield sym, line0 + region.count("\n", 0, off)


def symbols(chapter_dir) -> dict[str, list[tuple[str, int]]]:
    """{symbol: [(page name, line), ...]} over the chapter's pages, sorted by symbol."""
    result: dict[str, list[tuple[str, int]]] = {}
    for page in pages(chapter_dir):
        for sym, line in _occurrences(page):
            where = (page.name, line)
            if where not in result.setdefault(sym, []):
                result[sym].append(where)
    return dict(sorted(result.items()))


_PHRASE = r"((?:(?!\band\b)[^,;:.$(){}`\n]){1,100})"
# Words that cannot start a noun phrase, so "where $T$ is used as given" is not a definition.
_NOT_NOUN = {"not", "used", "given", "also", "then", "large", "small", "computed", "defined", "chosen", "set",
             "typically", "usually", "often", "always", "never", "equal", "close", "kept", "shown", "written"}
_RELATION = re.compile(r"[=<>]|\\(?:approx|sim|in|le|leq|ge|geq|propto|times|cdot)(?![A-Za-z])")


def _noun_phrase(phrase: str) -> bool:
    words = phrase.split()
    if not words:
        return False
    first = words[0].lower()
    if first in ("the", "a", "an"):
        return len(words) > 1
    return first.isalpha() and first not in _NOT_NOUN and not first.endswith(("ly", "ed", "ing"))
_BEFORE = {
    "where": re.compile(r"\b[Ww]here\s+$"),
    "with": re.compile(r"\b[Ww]ith\s+$"),
    "let": re.compile(r"\b[Ll]et\s+$"),
}
_AFTER = {
    "where": re.compile(r"\s+(?:is|are|denotes)\s+" + _PHRASE),
    "with": re.compile(r"\s+((?:the)\s+(?:(?!\band\b)[^,;:.$(){}`\n]){1,100})"),
    "let": re.compile(r"\s+be\s+" + _PHRASE),
    None: re.compile(r"\s+denotes\s+" + _PHRASE),
}


_LHS = re.compile(r"^(.*?)(?:=|\\approx|\\equiv)", re.S)
_APPLIED = re.compile(r"^\s*(?:\\[A-Za-z]+\s*)*(.+?)\s*\((.*)\)\s*$", re.S)
_CONTINUES = re.compile(r"(?:,|\band)\s*$")
_LISTED = re.compile(r"\s+(?:is\s+|are\s+)?((?:the|a|an)\s+(?:(?!\band\b)[^,;:.$(){}`\n]){1,100})")
_SENTENCE_END = re.compile(r"[.!?]\s|\n\s*\n")


def _defined_symbol(math: str):
    """(symbol, is_relation) for the math of a defining phrase, or (None, ...) when it defines no one symbol.

    A bare symbol defines itself; a relation defines its left-hand side if that is one symbol ($G = ...$); an
    application defines its function ($c(w)$ defines c); an operator applied to an expression
    ($\\min h(A)$, $\\max_i x_i$, $\\sum ...$) defines nothing.
    """
    head = re.match(r"\s*\\([A-Za-z]+)", math)
    if head and head.group(1) not in GREEK | WRAPPERS | ACCENTS:
        return None, False  # an operator applied to an expression ($\\min h(A)$, $\\max_i x_i$) defines no symbol
    lhs = _LHS.match(math)
    if lhs:
        side = normalise(lhs.group(1))
        return (side[0] if len(side) == 1 else None), True
    if _RELATION.search(math):
        return None, True
    found = normalise(math)
    if len(found) == 1:
        return found[0], False
    applied = _APPLIED.match(math)
    if applied:
        head = normalise(applied.group(1))
        if len(head) == 1:
            return head[0], False
    return None, False


def definitions_in_text(text: str):
    """(symbol, line, phrase) for every defining phrase in a page's text, in order."""
    chain_end = None  # end of the last definition that a list ("..., $b$ the ..., and $c$ the ...") may continue
    for start, end in math_regions(text):
        if text.startswith("$$", start):
            continue
        symbol, relation = _defined_symbol(text[start:end].strip("$"))
        if symbol is None:
            continue
        line = text.count("\n", 0, start) + 1
        before = text[max(0, start - 20):start]
        kind = next((k for k, pattern in _BEFORE.items() if pattern.search(before)), None)
        if relation and kind not in ("where", "let"):
            continue  # "with $T = 1$ the weights ...": a condition, not a definition
        m = _AFTER[kind].match(text, end) or (_AFTER[None].match(text, end) if kind else None)
        if m and _noun_phrase(m.group(1).strip()):
            chain_end = m.end() if kind == "where" else None
            yield symbol, line, m.group(1).strip()
            continue
        if (chain_end is not None and not relation and not _SENTENCE_END.search(text, chain_end, start)
                and _CONTINUES.search(text[chain_end:start])):
            m = _LISTED.match(text, end)
            if m and _noun_phrase(m.group(1).strip()):
                chain_end = m.end()
                yield symbol, line, m.group(1).strip()


def _definitions_in(text: str):
    yield from definitions_in_text(text)


def definitions(chapter_dir) -> dict[str, list[tuple[str, int, str]]]:
    """{symbol: [(page name, line, defining phrase), ...]} over the chapter's pages, sorted by symbol."""
    result: dict[str, list[tuple[str, int, str]]] = {}
    for page in pages(chapter_dir):
        for sym, line, phrase in _definitions_in(page.read_text(encoding="utf-8")):
            result.setdefault(sym, []).append((page.name, line, phrase))
    return dict(sorted(result.items()))


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _covering(sentences, line: int):
    """The (index, start line, sentence) of each sentence whose lines include `line`, in order."""
    return [(k, a, s) for k, (a, s) in enumerate(sentences) if a <= line <= a + s.count("\n")]


def _display_intro(text: str, start: int, sentences) -> str:
    """The sentence that introduces the display block at `start`, when the line before it ends with a colon."""
    line_start = text.rfind("\n", 0, start) + 1
    n = text.count("\n", 0, start) + 1
    before = text[line_start:start]
    if not before.strip():  # the block starts its line: look at the nearest non-blank line above
        lines = text[:line_start].split("\n")[:-1]
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ""
        before, n = lines[-1], len(lines)
    if not before.rstrip().endswith(":"):
        return ""
    covering = _covering(sentences, n)
    if not covering:
        return _collapse(before)
    _k, a, sentence = covering[-1]
    intro = "\n".join(sentence.split("\n")[:n - a + 1])
    return _collapse(intro.split("$$")[0])


def concordance_of_text(text: str, page: str = "") -> list[tuple[str, str, int, tuple, str]]:
    """(symbol, page, line, unit key, unit) for every symbol occurrence in a page's math, in order.

    The unit is the sentence or list item holding the occurrence, from the splitter the phrase layer of
    tests/test_notation.py uses (tests.test_notebook_prose.prose_sentences), math in place and white space
    collapsed. An occurrence in display math has the display block as its unit, preceded by the sentence that
    introduces it when the line before the block ends with a colon.
    """
    from tests.test_notebook_prose import prose_sentences  # the phrase layer's unit splitter

    sentences = list(prose_sentences(text))
    consumed: dict[tuple[int, str], int] = {}  # (sentence index, math) -> occurrences already assigned
    found = []
    for start, end in math_regions(text):
        region = text[start:end]
        line0 = text.count("\n", 0, start) + 1
        if region.startswith("$$"):
            intro = _display_intro(text, start, sentences)
            key, unit = ("display", start), (f"{intro} " if intro else "") + _collapse(region)
        else:
            # Regions come in document order, so the n-th "$A$" on a line belongs to the sentence holding the
            # n-th "$A$" among the sentences that cover the line.
            needle = _collapse(region)
            match = next(((k, s) for k, _a, s in _covering(sentences, line0)
                          if _collapse(s).count(needle) > consumed.get((k, needle), 0)), None)
            if match:
                consumed[(match[0], needle)] = consumed.get((match[0], needle), 0) + 1
                key, unit = ("sentence", match[0]), _collapse(match[1])
            else:  # math outside any prose unit (a directive option): the line itself
                line_start = text.rfind("\n", 0, start) + 1
                line_end = text.find("\n", end)
                key, unit = ("line", line0), _collapse(text[line_start:line_end if line_end >= 0 else len(text)])
        for sym, off in symbols_in_math(region):
            found.append((sym, page, line0 + region.count("\n", 0, off), key, unit))
    return found


def _tables(chapter_dir, global_path=None):
    """(global rows, local rows) as (symbol, meaning) pairs, read with tests/test_notation.py's readers."""
    from tests.test_notation import GLOBAL, global_rows, local_rows

    return global_rows(global_path or GLOBAL), local_rows(chapter_dir)


def concordance(chapter_dir, global_path=None) -> dict[str, list[tuple[str, int, str]]]:
    """{symbol: [(page name, line, unit), ...]} for every symbol the chapter's math uses.

    Each unit is listed once per symbol, at the line of the symbol's first occurrence in it. Symbols come in
    the order of tests/notation_global.txt, then the chapter's notation-local.md, then the undeclared ones
    sorted.
    """
    global_table, local_table = _tables(chapter_dir, global_path)
    occurrences: dict[str, list[tuple[str, int, str]]] = {}
    seen = set()
    for page in pages(chapter_dir):
        for sym, name, line, key, unit in concordance_of_text(page.read_text(encoding="utf-8"), page.name):
            if (sym, name, key) not in seen:
                seen.add((sym, name, key))
                occurrences.setdefault(sym, []).append((name, line, unit))
    order = list(dict.fromkeys([s for s, _ in global_table] + [s for s, _ in local_table]))
    order += sorted(s for s in occurrences if s not in order)
    return {s: occurrences[s] for s in order if s in occurrences}


def meanings(chapter_dir, global_path=None) -> dict[str, str]:
    """{symbol: meaning}: the chapter's local list where it has the symbol, else the global table's row(s)."""
    global_table, local_table = _tables(chapter_dir, global_path)
    result: dict[str, str] = {}
    for sym, meaning in global_table:
        result[sym] = f"{result[sym]} | {meaning}" if sym in result else meaning
    local: dict[str, str] = {}
    for sym, meaning in local_table:
        local[sym] = f"{local[sym]} | {meaning}" if sym in local else meaning
    result.update(local)
    return result


def concordance_lines(chapter_dir, symbol: str | None = None, undeclared: bool = False) -> list[str]:
    """The --concordance output: a header per symbol, then one line per unit."""
    known = meanings(chapter_dir)
    out = []
    for sym, places in concordance(chapter_dir).items():
        if (symbol is not None and sym != symbol) or (undeclared and sym in known):
            continue
        out.append(f"== {sym} — {known.get(sym, 'UNDECLARED')} ==")
        out += [f"{page}:L{line}: {unit}" for page, line, unit in places]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("chapter_dir", type=Path)
    ap.add_argument("--definitions", action="store_true", help="also print defining phrases")
    ap.add_argument("--concordance", action="store_true",
                    help="print every sentence or list item that uses each symbol, with the symbol's meaning")
    ap.add_argument("--symbol", help="with --concordance: only this symbol (normal form, e.g. 'd_k')")
    ap.add_argument("--undeclared", action="store_true",
                    help="with --concordance: only symbols in neither the global table nor the local list")
    args = ap.parse_args(argv)
    if args.concordance or args.symbol is not None or args.undeclared:
        lines = concordance_lines(args.chapter_dir, args.symbol, args.undeclared)
        print("\n".join(lines))
        headers = sum(line.startswith("== ") for line in lines)
        print(f"# {headers} symbols, {len(lines) - headers} units", file=sys.stderr)
        return 0
    found = symbols(args.chapter_dir)
    defs = definitions(args.chapter_dir) if args.definitions else {}
    for sym, places in found.items():
        by_page: dict[str, list[str]] = {}
        for page, line in places:
            by_page.setdefault(page, []).append(str(line))
        print(f"{sym}\t" + "; ".join(f"{page}:{','.join(lines)}" for page, lines in by_page.items()))
        for page, line, phrase in defs.get(sym, []):
            print(f"    defined {page}:L{line}: {phrase}")
    print(f"# {len(found)} distinct symbols", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
