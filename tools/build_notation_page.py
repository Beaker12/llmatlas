"""Build notation.md, the reader's notation page, from the notation tables.

Usage: python tools/build_notation_page.py          write notation.md
       python tools/build_notation_page.py --check  exit 1 if notation.md differs from what would be written

Sources: tests/notation_global.txt (book-wide symbols), tests/notation_exceptions.txt (letters with two
meanings and why) and each chapter's notation-local.md (symbol and meaning only; the internal "First used"
and "Rename" columns are not shown). Each chapter-local symbol links to the labelled section that holds its
first use on the chapter's pages (tools/list_symbols.py finds the uses); a symbol with no use links to the
chapter. Deterministic, no network; tests/test_notation_page.py checks the committed page is current.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import list_symbols  # noqa: E402
from tools.notation_tables import ROOT, global_rows, local_table  # noqa: E402

PAGE = ROOT / "notation.md"
EXCEPTIONS = ROOT / "tests" / "notation_exceptions.txt"
LABEL = re.compile(r"^\(([A-Za-z0-9_-]+)\)=\s*$")
HEADING = re.compile(r"^#{1,6} ")
MATHLIKE = re.compile(r"[\\_^]|^sec-|^[A-Za-z]'+$")  # TeX, a section label, or a primed letter (n')
SECTION = re.compile(r"sec-[A-Za-z0-9_-]+")
# A code name such as a configuration key (beta_fast): a word joined by underscores with no TeX, where some part
# after the first underscore has two or more characters. In math it would print as italic letters with a
# one-letter subscript, so it is set as code. A symbol's subscript (d_k, x_i) has one character or braces.
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*_[A-Za-z0-9]{2,}(?:_[A-Za-z0-9]+)*")

INTRO = """\
(sec-notation)=
# Notation

This page collects the symbols the book uses. The first table holds the symbols every chapter shares, with one
meaning each. A few letters knowingly carry two meanings; the second table lists them and says why the two never
meet in one formula. The chapter sections after that list the symbols that only one chapter uses, each with a link
to the section where it is introduced.

Symbols are given in a normal form: a subscript that is only a running index ($i$, $j$, $t$, ...) is dropped, so
$\\eta$ stands for $\\eta_t$ and $d_k$ keeps its $k$ because it is a name, not an index. Vectors are rows, and a
layer computes $xW$.
"""


COLUMNS_2 = "```{tabularcolumns} \\X{1}{5}\\X{4}{5}\n```"
COLUMNS_3 = "```{tabularcolumns} \\X{3}{20}\\X{11}{20}\\X{6}{20}\n```"


def exception_rows(path=EXCEPTIONS):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            symbol, meanings, reason = (p.strip() for p in line.split("\t"))
            rows.append((symbol, [m.strip() for m in meanings.split(" | ")], reason))
    return rows


def _math_token(token: str) -> str:
    match = re.match(r"^([(\[]*)(.*?)([)\],;.:]*)$", token)
    lead, core, tail = match.groups()
    if core.count("(") > core.count(")") and tail.startswith(")"):
        core, tail = core + ")", tail[1:]
    if not core or not MATHLIKE.search(core) or "`" in core:
        return token
    if SECTION.fullmatch(core):
        return f"{lead}{{ref}}`{core}`{tail}"
    if IDENTIFIER.fullmatch(core):
        return f"{lead}`{core}`{tail}"
    return f"{lead}${core}${tail}"


def _letter_token(token: str) -> str:
    """Inside parentheses a lone letter is a symbol ("at position m", "pair p, q", "the i-th"): math."""
    match = re.match(r"^([(\[]*)([A-Za-z])(-th)?([)\],;.:]*)$", token)
    if match and match.group(2) not in ("a", "A"):
        lead, letter, suffix, tail = match.groups()
        return f"{lead}${letter}${suffix or ''}{tail}"
    return _math_token(token)


def render(text: str) -> str:
    """A meaning for a table cell: bare TeX tokens outside $...$ set as math, pipes escaped."""
    parts = re.split(r"(\$[^$]*\$)", text)
    out, depth = [], 0
    for k, part in enumerate(parts):
        if k % 2:
            out.append(part)
        else:
            tokens = []
            for t in part.split(" "):
                tokens.append(_letter_token(t) if depth > 0 or t.startswith("(") else _math_token(t))
                depth += t.count("(") - t.count(")")
            out.append(" ".join(tokens))
    return "".join(out).replace("|", "\\vert ")


def chapters():
    return sorted(p for p in (ROOT / "chapters").iterdir() if (p / "index.md").exists())


def _title(chapter: Path) -> str:
    for line in (chapter / "index.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return chapter.name


def _section_labels(page: Path):
    """[(line of the heading, label)] for each heading on the page that carries a label."""
    lines = page.read_text(encoding="utf-8").splitlines()
    out = []
    for number, line in enumerate(lines, start=1):
        if HEADING.match(line):
            k = number - 2
            while k >= 0 and not lines[k].strip():
                k -= 1
            found = LABEL.match(lines[k]) if k >= 0 else None
            if found:
                out.append((number, found.group(1)))
    return out


def _first_used_labels(chapter: Path) -> dict[str, str]:
    """{symbol: label} for local rows whose "First used" cell names a section label (sec-...)."""
    out = {}
    path = chapter / "notation-local.md"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| $") and len(cells) >= 3:
            found = re.search(r"\b(sec-[A-Za-z0-9_-]+)", cells[2])
            if found:
                out[cells[0].strip("$").strip()] = found.group(1)
    return out


def _link(chapter: Path, uses, label=None) -> str:
    if label:
        return f"{{ref}}`{label}`"
    if uses:
        uses = sorted(uses, key=lambda u: (u[0] != "index.md", u[0], u[1]))
        page, line = uses[0]
        labels = [label for at, label in _section_labels(chapter / page) if at <= line]
        if labels:
            return f"{{ref}}`{labels[-1]}`"
    rel = (chapter / "index").relative_to(ROOT).as_posix()
    return f"{{doc}}`{_title(chapter)} </{rel}>`"


def build() -> str:
    # tabularcolumns sets the column widths of the LaTeX output only (a long table otherwise gets
    # unbreakable `l` columns that run off the page); the HTML is unchanged.
    out = [INTRO, "## Symbols used throughout the book", "", COLUMNS_2, "", "| Symbol | Meaning |", "|---|---|"]
    out += [f"| ${s}$ | {render(m)} |" for s, m in global_rows()]
    out += ["", "## Letters with two meanings", "", COLUMNS_3, "", "| Symbol | Meanings | Why both are kept |", "|---|---|---|"]
    out += [f"| ${s}$ | {'; '.join(render(m) for m in ms)} | {render(r)} |" for s, ms, r in exception_rows()]
    out += ["", "## Symbols used in one chapter", ""]
    for chapter in chapters():
        rows = local_table(chapter)
        if not rows:
            continue
        uses = list_symbols.symbols(chapter)
        named = _first_used_labels(chapter)
        number = int(chapter.name.split("-")[0])
        out += [f"### Chapter {number}: {_title(chapter)}", "", COLUMNS_3, "", "| Symbol | Meaning | Introduced in |",
                "|---|---|---|"]
        for row in rows:
            meaning = render(row["meaning"])
            if row["also"]:
                meaning += f"; also {render(row['also'])}"
            out.append(f"| ${row['symbol']}$ | {meaning} | {_link(chapter, uses.get(row['symbol']), named.get(row['symbol']))} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv):
    text = build()
    if "--check" in argv:
        current = PAGE.read_text(encoding="utf-8") if PAGE.exists() else ""
        if current != text:
            print("notation.md is out of date: run python tools/build_notation_page.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
