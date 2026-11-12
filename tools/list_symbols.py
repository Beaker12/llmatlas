#!/usr/bin/env python3
"""List the mathematical symbols a chapter uses, with the page and line of every occurrence.

Usage:
    python tools/list_symbols.py chapters/02-mathematics
    python tools/list_symbols.py chapters/02-mathematics --definitions

Math regions are found with tools.unescape_math.math_regions (fenced code excluded). The pages are the
*.md files directly in the chapter directory, except notation-local.md.

Symbol grammar
    A symbol is a Latin or Greek base letter, keeping any \\mathcal, \\mathbf, \\mathrm or \\mathbb wrapper
    and any \\tilde, \\bar or \\hat accent, plus its subscript after stripping \\text{}, \\mathrm{} and
    braces. A subscript that is only a running index (one or two of i, j, l, t, m, s, or an index
    expression such as i-1, 2i+1 or m') is dropped, unless it is written as text (\\epsilon_{\\text{ls}} is
    \\epsilon_{ls}, \\epsilon_{ls} is \\epsilon). Superscripts and primes are not part of the symbol;
    the symbols inside a superscript are listed on their own, except in a superscript of capital letters
    on a symbol, which is part of its name (W^{UK} is W). A subscript holding a relation or range
    (x_{<i}, \\mathbb{E}_{x \\sim p}, w_{1:N}) is a condition: not part of the symbol, its symbols listed. Operator and function names (OPERATOR_NAMES,
    \\operatorname{...}, \\text{...}, multi-letter \\mathrm{...} and the bare words in OPERATOR_WORDS) are
    not symbols; the symbols in their subscripts and arguments are.

    Normal form: base, then "_x" for a one-character subscript or "_{xy}" for a longer one, e.g.
    W_{gate}, d_k, \\tilde{\\rho}, \\mathcal{L}, \\lambda_{\\max}.

Defining phrases (--definitions): "where $x$ is|are|denotes ...", "$x$ denotes ...", "with $x$ the|a|an
...", "let $x$ be ...". The phrase runs to the next comma, semicolon, colon, full stop, parenthesis,
dollar sign, brace, backtick or line end. The symbol is the first one in the inline math.
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
# A superscript of capital letters on a symbol is part of its name (W^{UK}, d_h^{R}): not scanned.
LABEL = re.compile(r"\s*[A-Z]+\s*(?:\\top\s*)?")
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
                if base is None or not LABEL.fullmatch(self.s[a:b]):
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


_PHRASE = r"([^,;:.$(){}`\n]{1,100})"
_BEFORE = {
    "where": re.compile(r"\b[Ww]here\s+$"),
    "with": re.compile(r"\b[Ww]ith\s+$"),
    "let": re.compile(r"\b[Ll]et\s+$"),
}
_AFTER = {
    "where": re.compile(r"\s+(?:is|are|denotes)\s+" + _PHRASE),
    "with": re.compile(r"\s+((?:the|a|an)\s+[^,;:.$(){}`\n]{1,100})"),
    "let": re.compile(r"\s+be\s+" + _PHRASE),
    None: re.compile(r"\s+denotes\s+" + _PHRASE),
}


def _definitions_in(text: str):
    for start, end in math_regions(text):
        if text.startswith("$$", start):
            continue
        found = symbols_in_math(text[start:end])
        if not found:
            continue
        before = text[max(0, start - 20):start]
        kind = next((k for k, pattern in _BEFORE.items() if pattern.search(before)), None)
        m = _AFTER[kind].match(text, end) or (_AFTER[None].match(text, end) if kind else None)
        if m and m.group(1).strip():
            yield found[0][0], text.count("\n", 0, start) + 1, m.group(1).strip()


def definitions(chapter_dir) -> dict[str, list[tuple[str, int, str]]]:
    """{symbol: [(page name, line, defining phrase), ...]} over the chapter's pages, sorted by symbol."""
    result: dict[str, list[tuple[str, int, str]]] = {}
    for page in pages(chapter_dir):
        for sym, line, phrase in _definitions_in(page.read_text(encoding="utf-8")):
            result.setdefault(sym, []).append((page.name, line, phrase))
    return dict(sorted(result.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("chapter_dir", type=Path)
    ap.add_argument("--definitions", action="store_true", help="also print defining phrases")
    args = ap.parse_args(argv)
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
