#!/usr/bin/env python3
"""List code and output lines wider than the limit (88 characters).

Usage: python tools/check_width.py [chapter directory name ... | all] [--limit N] [--quiet]

Checked, for each chapter directory under chapters/: every line inside a code fence (a plain-language info
string such as python, text or bash, or {code-cell}, {code-block}, {sourcecode}; backtick, tilde or colon
fence, at any nesting depth) of the chapter's Markdown pages, except its notation-local.md. The prose and
options of any other {directive} (admonition, note, figure, math...) are not checked, since prose wraps in
the PDF, but code fences nested in one are; and in the chapter's notebooks (examples/*.ipynb) every code-cell source line and every
line of stream and text/plain output. 88 characters fit the PDF's text block at its code font size without
wrapping. Exit status 1 if any line is over the limit. tests/test_line_width.py runs the same check per
chapter.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fences import directive_name, fences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 88
CHECKED_DIRECTIVES = {"code-cell", "code-block", "code", "sourcecode"}


def page_lines(path: Path):
    """(line number, text) for every line in the body of a code fence of a Markdown page.

    A code fence has a plain-language info string (or none) or is one of CHECKED_DIRECTIVES; the fence lines
    themselves are skipped. Any other {directive} is a container (prose, options, {math}, {literalinclude}):
    its own lines are not checked, but code fences nested in it are. The fence parser is tools/fences.py.
    """
    text = Path(path).read_text(encoding="utf-8")
    for fence in fences(text):
        info = fence["info"]
        if info.startswith("{") and directive_name(info.split("}")[0] + "}") not in CHECKED_DIRECTIVES:
            continue
        for offset, line in enumerate(fence["body"].split("\n")[:-1], start=1):
            yield fence["line"] + offset, line


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
