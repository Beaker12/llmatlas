#!/usr/bin/env python3
"""List every exemption a chapter has declared, for the person who checks the chapter.

Usage:
    python tools/list_exemptions.py chapters/04-preprocessing

Kinds, in this order:
    SYNC_EXEMPT, NOT_A_RESULT, NOTATION_OK, INTERFACE_LABELS
        entries of the chapter's tests/test_chNN_snippets.py, read with ast (the test module is not imported);
    Also means
        rows of notation-local.md that fill the "Also means" column (a second meaning of the symbol);
    lifted no-meaning ban
        rows of notation-local.md whose symbol has a NO_MEANING ban in tests/test_notation.py for this chapter,
        which the row switches off;
    spelling-allow.txt, proper-nouns.txt
        lines of the chapter's own allow lists;
    static-numbers, tol/compare
        notebook cells tagged static-numbers, and cells with "tol" or "compare" metadata (cell index, first line);
    paper note
        sentences of the pages whose math the paper-note rule exempts from the notation check.

Output: a header "== <kind> (<count>) ==" per kind that has entries, one line "<entry> — <reason>" per entry,
and a count line "# <n> exemptions" on stderr. Importable: exemptions(chapter_dir) -> [(kind, entry, reason)].
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.notation_tables import local_table, pages, paper_note_math, prose_sentences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ("SYNC_EXEMPT", "NOT_A_RESULT", "NOTATION_OK", "INTERFACE_LABELS")
NOTATION_TESTS = ROOT / "tests" / "test_notation.py"


def chapter_number(chapter_dir) -> int:
    name = Path(chapter_dir).name
    return int(name[:2]) if name[:2].isdigit() else 0


def constants_file(chapter_dir):
    """tests/test_chNN_snippets.py for a numbered chapter, else None."""
    number = chapter_number(chapter_dir)
    path = ROOT / "tests" / f"test_ch{number:02d}_snippets.py"
    return path if number and path.exists() else None


def literal_assignments(path, names) -> dict:
    """{name: value} for the module-level assignments of `names` in the Python file `path`, by ast.literal_eval."""
    found = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        else:
            continue
        if target in names and value is not None:
            found[target] = ast.literal_eval(value)
    return found


def no_meaning_entries(path=NOTATION_TESTS):
    """NO_MEANING of tests/test_notation.py: (chapter, symbol, what to write instead)."""
    return literal_assignments(path, {"NO_MEANING"}).get("NO_MEANING", [])


def _lines(path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def _first_line(cell):
    source = cell.get("source", "")
    text = "".join(source) if isinstance(source, list) else source
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def exemptions(chapter_dir, constants=None, no_meaning=None) -> list[tuple[str, str, str]]:
    """(kind, entry, reason) for every exemption the chapter declares (see the module docstring).

    `constants` is the Python file holding the chapter's constants (default: its tests/test_chNN_snippets.py);
    `no_meaning` the NO_MEANING entries (default: read from tests/test_notation.py).
    """
    chapter_dir = Path(chapter_dir)
    found: list[tuple[str, str, str]] = []
    constants = constants_file(chapter_dir) if constants is None else constants
    if constants:
        values = literal_assignments(constants, set(CONSTANTS))
        for kind in CONSTANTS:
            for entry, reason in dict(values.get(kind, {})).items():
                found.append((kind, entry, reason))
    rows = local_table(chapter_dir)
    found += [("Also means", f"{row['symbol']} ({row['meaning']})", row["also"]) for row in rows if row["also"]]
    number = chapter_number(chapter_dir)
    bans = {sym: instead for chapter, sym, instead in (no_meaning_entries() if no_meaning is None else no_meaning)
            if chapter == number}
    found += [("lifted no-meaning ban", f"{row['symbol']} ({row['meaning']})", f"the ban said: {bans[row['symbol']]}")
              for row in rows if row["symbol"] in bans]
    for name in ("spelling-allow.txt", "proper-nouns.txt"):
        found += [(name, line, f"chapters/{chapter_dir.name}/{name}") for line in _lines(chapter_dir / name)]
    for notebook in sorted((chapter_dir / "examples").glob("*.ipynb")):
        cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
        for index, cell in enumerate(cells):
            meta = cell.get("metadata", {})
            where = f"{notebook.name} cell {index}: {_first_line(cell)}"
            if "static-numbers" in meta.get("tags", []):
                found.append(("static-numbers", where, "numbers in this markdown cell are not checked"))
            settings = {k: meta[k] for k in ("tol", "compare") if k in meta}
            if settings:
                found.append(("tol/compare", where, ", ".join(f"{k}={v!r}" for k, v in settings.items())))
    for page in pages(chapter_dir):
        for line, sentence in prose_sentences(page.read_text(encoding="utf-8")):
            spans = paper_note_math(sentence)
            if spans:
                found.append(("paper note", f"{page.name}:L{line}: {' '.join(sentence.split())}",
                              "exempt: " + ", ".join(sentence[a:b] for a, b in spans)))
    return found


def report_lines(found) -> list[str]:
    kinds: dict[str, list[tuple[str, str]]] = {}
    for kind, entry, reason in found:
        kinds.setdefault(kind, []).append((entry, reason))
    out = []
    for kind, entries in kinds.items():
        out.append(f"== {kind} ({len(entries)}) ==")
        out += [f"{entry} — {reason}" for entry, reason in entries]
    return out


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: list_exemptions.py <chapter-dir>", file=sys.stderr)
        return 2
    found = exemptions(args[0])
    lines = report_lines(found)
    if lines:
        print("\n".join(lines))
    print(f"# {len(found)} exemptions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
