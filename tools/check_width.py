#!/usr/bin/env python3
"""List code and output lines wider than the limit (88 characters).

Usage: python tools/check_width.py [chapter directory name ... | all] [--limit N] [--quiet]

Checked, for each chapter directory under chapters/: every line inside a fence (backtick or tilde, any info
string, including the `text` fences that show output) of the chapter's Markdown pages, except its
notation-local.md; and in the chapter's notebooks (examples/*.ipynb) every code-cell source line and every
line of stream and text/plain output. 88 characters fit the PDF's text block at its code font size without
wrapping. Exit status 1 if any line is over the limit. tests/test_line_width.py runs the same check per
chapter.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 88
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def page_lines(path: Path):
    """(line number, text) for every line inside a fence of a Markdown page; the fence lines are skipped."""
    opener = None  # the opening fence's characters, e.g. "````"
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        match = FENCE.match(line)
        if match:
            mark, rest = match.groups()
            if opener is None:
                opener = mark
            elif mark[0] == opener[0] and len(mark) >= len(opener) and not rest.strip():
                opener = None
            continue
        if opener is not None:
            yield number, line


def _text(value) -> str:
    return "".join(value) if isinstance(value, list) else (value or "")


def notebook_lines(path: Path):
    """(where, text) for every code-cell source line and stream or text/plain output line of a notebook."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for j, line in enumerate(_text(cell.get("source")).split("\n"), start=1):
            yield f"cell {i} source line {j}", line
        for k, output in enumerate(cell.get("outputs", [])):
            text = output.get("text") if "text" in output else output.get("data", {}).get("text/plain")
            for j, line in enumerate(_text(text).split("\n"), start=1):
                yield f"cell {i} output {k} line {j}", line


def chapter_failures(chapter_dir: Path, limit: int = LIMIT) -> list[str]:
    """Failure strings ("index.md:L12 (93 characters): ...") for the lines of a chapter over the limit."""
    chapter_dir = Path(chapter_dir)
    failures = []
    for page in sorted(chapter_dir.glob("*.md")):
        if page.name == "notation-local.md":
            continue
        for number, line in page_lines(page):
            if len(line) > limit:
                failures.append(f"{page.name}:L{number} ({len(line)} characters): {line[:60]}")
    for notebook in sorted(chapter_dir.glob("examples/*.ipynb")):
        for where, line in notebook_lines(notebook):
            if len(line) > limit:
                rel = notebook.relative_to(chapter_dir).as_posix()
                failures.append(f"{rel}: {where} ({len(line)} characters): {line[:60]}")
    return failures


def main(argv: list[str]) -> int:
    limit = LIMIT
    if "--limit" in argv:
        at = argv.index("--limit")
        limit = int(argv[at + 1])
        argv = argv[:at] + argv[at + 2:]
    quiet = "--quiet" in argv
    names = [a for a in argv if not a.startswith("--")]
    chapters = sorted(p for p in (ROOT / "chapters").iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if names and names != ["all"]:
        chapters = [p for p in chapters if p.name in names]
        missing = set(names) - {p.name for p in chapters}
        if missing:
            print(f"no such chapter directory: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    total = 0
    for chapter in chapters:
        failures = chapter_failures(chapter, limit)
        if not quiet:
            for failure in failures:
                print(f"{chapter.name}/{failure}")
        print(f"{chapter.name}: {len(failures)} lines over {limit}")
        total += len(failures)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
