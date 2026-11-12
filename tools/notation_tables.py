"""Pieces shared by tools/list_symbols.py and the tests, with no test framework behind them.

The prose unit splitter (prose_sentences: list items and paragraphs outside code, cut into sentences), used by
tests/test_notebook_prose.py, the phrase layer of tests/test_notation.py and the concordance of
tools/list_symbols.py; and the readers of the notation tables (tests/notation_global.txt and each chapter's
notation-local.md).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ROOT / "tests" / "notation_global.txt"
LOCAL_LIST = "notation-local.md"

# --- the prose unit splitter --------------------------------------------------------------------------------

FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,}|:{3,})[ \t]*(.*?)[ \t]*$")
# MyST directives whose content is code or markup, not prose; every other directive keeps its text
CODE_DIRECTIVES = {"code-cell", "code-block", "sourcecode", "code", "math", "literalinclude", "raw", "eval-rst",
                   "mermaid", "graphviz"}
LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
ABBREVIATION = re.compile(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bcf|\bal|\bFig|\bEq|\bSec|\bapprox|\bNo|(?<![A-Za-z])[A-Z])\.$")


def blank_code(text):
    """The text with every fenced code block (any info string, none, or a code directive) blanked.

    Line count is kept. The fence lines of prose directives ({admonition}, {note}, ...) are blanked but their
    content stays; a ::: fence is always a prose directive.
    """
    out, stack = [], []   # stack of (marker char, length, is_code)
    for line in text.split("\n"):
        m = FENCE_LINE.match(line)
        if m:
            marker, info = m.groups()
            char, length = marker[0], len(marker)
            if stack and stack[-1][0] == char and length >= stack[-1][1] and not info:
                stack.pop()
                out.append("")
                continue
            if not (stack and stack[-1][2]):
                directive = re.match(r"\{([\w:-]+)\}", info)
                is_code = ((directive is None and char != ":")
                           or (directive is not None and directive.group(1) in CODE_DIRECTIVES))
                stack.append((char, length, is_code))
                out.append("")
                continue
        out.append("" if stack and stack[-1][2] else line)
    return "\n".join(out)


def prose_units(text):
    """(first line, text) for each unit: a list item (with its wrapped lines) or a paragraph.

    A blank line, a list-item marker or a heading starts a new unit; a wrapped line continues the current one.
    """
    units, current = [], None
    for n, line in enumerate(blank_code(text).split("\n"), start=1):
        if not line.strip() or re.match(r"^[ \t]*:[\w-]+:|^[ \t]*%", line):
            current = None
        elif re.match(r"^[ ]{0,3}#{1,6}[ \t]", line):
            units.append((n, line))
            current = None
        elif LIST_ITEM.match(line):
            current = [n, LIST_ITEM.sub("", line)]
            units.append(current)
        elif current is None:
            current = [n, line.strip()]
            units.append(current)
        else:
            current[1] += "\n" + line.strip()
    return [(n, t) for n, t in units]


def split_sentences(unit):
    """(offset, sentence) pairs: split after . ! ? and white space, but not after an abbreviation."""
    cuts = [0]
    for m in re.finditer(r"(?<=[.!?])\s+", unit):
        if not ABBREVIATION.search(unit[: m.start()]):
            cuts.append(m.end())
    cuts.append(len(unit))
    return [(a, unit[a:b]) for a, b in zip(cuts, cuts[1:])]


def prose_sentences(text):
    """(line, sentence) for every sentence outside code: admonition bodies, list items and paragraphs.

    The detector is deliberately wider than the admonition directives: chapter pages link notebook labels in
    the Hands-On list and in body paragraphs as well, and those sentences may state results too. Each list
    item is its own unit; sentences end at . ! ? followed by white space.
    """
    for start, unit in prose_units(text):
        for a, piece in split_sentences(unit):
            sentence = piece.strip()
            if sentence:
                yield start + unit.count("\n", 0, a), sentence


# --- the notation tables ------------------------------------------------------------------------------------

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
    return _rows(Path(path), 2)


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
