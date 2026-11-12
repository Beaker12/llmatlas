"""Pieces shared by tools/list_symbols.py and the tests, with no test framework behind them.

The prose unit splitter (prose_sentences: list items and paragraphs outside code, cut into sentences), used by
tests/test_notebook_prose.py, the phrase layer of tests/test_notation.py and the concordance of
tools/list_symbols.py; paper_note_spans, the math a note gives as a paper's own symbols ("The YaRN paper writes
the ramp as $\\gamma$"), which tests/test_notation.py exempts; and the readers of the notation tables
(tests/notation_global.txt and each chapter's notation-local.md).
"""
from __future__ import annotations

import re
from pathlib import Path

from tools.unescape_math import math_regions

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


# --- notes that name a paper's own symbols ------------------------------------------------------------------

# The verb of a paper note: "the (RoFormer) paper writes", "the papers write", "the authors write", "<Surname> et
# al. write", "<Surname> and <Surname> write", or a {cite:t} role followed by write, writes or wrote.
PAPER_NOTE = re.compile(
    r"\b[Tt]he\s+(?:[\w-]+\s+){0,3}?(?:paper\s+(?:writes|wrote)|papers\s+(?:write|wrote)|authors\s+(?:write|wrote))\b"
    r"|\b[A-Z][\w'-]+\s+et\s+al\.\s+(?:write|wrote)\b"
    r"|\b[A-Z][\w'-]+\s+and\s+[A-Z][\w'-]+\s+(?:write|wrote)\b"
    r"|\{cite:t\}`[^`\n]*`\s+(?:writes?|wrote)\b")


def paper_note_math(sentence):
    """(start, end) in `sentence` of each inline math span after the verb of a paper note; display math never."""
    regions = math_regions(sentence)
    masked = list(sentence)
    for a, b in regions:
        masked[a:b] = " " * (b - a)
    m = PAPER_NOTE.search("".join(masked))
    if not m:
        return []
    return [(a, b) for a, b in regions if a >= m.end() and not sentence.startswith("$$", a)]


def paper_note_spans(text):
    """(start, end) character spans in `text` of the math after the verb in each paper-note sentence.

    Sentences come from prose_sentences, which joins a unit's wrapped lines stripped; each sentence is found
    again in the text and its characters mapped back, a joined line break to the next line's first non-blank.
    """
    starts = [0]
    for line in text.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    spans, cursor = [], 0
    for line, sentence in prose_sentences(text):
        first = sentence.split("\n", 1)[0]
        pos = text.find(first, max(cursor, starts[line - 1]))
        if pos < 0:
            continue
        local = paper_note_math(sentence)
        if not local:
            cursor = pos + len(first)
            continue
        mapping, t = [], pos
        for ch in sentence:
            if ch == "\n":
                t = text.index("\n", t) + 1
                while t < len(text) and text[t] in " \t":
                    t += 1
                mapping.append(t)
                continue
            mapping.append(t)
            t += 1
        cursor = t
        spans += [(mapping[a], mapping[b - 1] + 1) for a, b in local]
    return spans


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


def local_table(chapter_dir) -> list[dict[str, str]]:
    """The rows of the chapter's notation-local.md table as {"symbol", "meaning", "also"} dicts.

    Columns: Symbol, Meaning, First used, Rename and an optional last column "Also means". A chapter fills
    "Also means" when its symbol's normal form already has another meaning (U_s, e_s and w_s drop their running
    index and read as U, e, w): it names the other meaning and says why the double use is acceptable, and
    tests/test_notation.py then accepts the two meanings without a line in tests/notation_exceptions.txt.
    The column is found by its header; a row with fewer cells leaves it empty.
    """
    path = Path(chapter_dir) / LOCAL_LIST
    if not path.exists():
        return []
    rows, also = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) < 3 or set(cells[0]) <= set("-:"):
            continue
        if cells[0] == "Symbol":
            also = next((k for k, c in enumerate(cells) if c.lower() == "also means"), None)
            continue
        if cells[0] == "":
            continue
        rows.append({"symbol": cells[0].strip("$").strip(), "meaning": cells[1],
                     "also": cells[also] if also is not None and also < len(cells) else ""})
    return rows


def local_rows(chapter_dir) -> list[tuple[str, str]]:
    """(symbol, meaning) from the chapter's notation-local.md table (see local_table)."""
    return [(row["symbol"], row["meaning"]) for row in local_table(chapter_dir)]


def local_also_means(chapter_dir) -> dict[str, str]:
    """{symbol: the filled "Also means" cell} for the chapter's local rows that fill it."""
    return {row["symbol"]: row["also"] for row in local_table(chapter_dir) if row["also"]}
