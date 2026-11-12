#!/usr/bin/env python3
"""List the functions and classes a chapter shares with its notebook.

A name is shared when it is defined at top level both in a ```python fence of the chapter's pages
(pseudo-code fences included) and in a code cell of the chapter's notebook. "Top level" includes
definitions inside module-level `if`, `try`, `with`, `for` and `while` blocks, but not inside a function
or class. When a name is defined more than once on one side, the LAST definition is used: it is the one in
effect after running the pages or the notebook from top to bottom.

A notebook cell or a fence that does not parse is reported by parse_failures() as
"<source>: cannot parse: <error>"; pseudo-code fences that do not parse are skipped quietly. In notebook
cells a line is blanked as an IPython magic only when it starts with `%` or `!` at bracket depth zero.

Usage: python tools/list_shared_defs.py chapters/03-data-acquisition
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

FENCE = re.compile(r"```python\n([\s\S]*?)```")
PSEUDO = re.compile(r"# pseudo-code(?![\w-])")
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def blank_magics(source: str) -> str:
    """Blank lines that start with % or ! at bracket depth zero outside strings (IPython magics)."""
    out, depth, quote = [], 0, None
    for line in source.split("\n"):
        if quote is None and depth == 0 and line.lstrip().startswith(("%", "!")):
            out.append("")
            continue
        out.append(line)
        i = 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if line.startswith(quote, i):
                    i += len(quote)
                    quote = None
                    continue
            elif ch == "#":
                break
            elif ch in "\"'":
                quote = ch * 3 if line.startswith(ch * 3, i) else ch
                i += len(quote)
                continue
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            i += 1
        if quote and len(quote) == 1:
            quote = None  # a single-quoted string ends at the end of its line
    return "\n".join(out)


def _collect(body: list[ast.stmt], lines: list[str], found: dict[str, str]) -> None:
    for node in body:
        if isinstance(node, DEFS):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            found[node.name] = "\n".join(lines[start - 1: node.end_lineno])
        else:
            for field in ("body", "orelse", "finalbody"):
                _collect(getattr(node, field, None) or [], lines, found)
            for handler in getattr(node, "handlers", []):
                _collect(handler.body, lines, found)
            for case in getattr(node, "cases", []):
                _collect(case.body, lines, found)


def top_level_defs(source: str, magics: bool = True) -> dict[str, str]:
    """name -> source text of each top-level function or class (decorators included); the last one wins.

    Raises SyntaxError if the source does not parse.
    """
    if magics:
        source = blank_magics(source)
    tree = ast.parse(source)
    found: dict[str, str] = {}
    _collect(tree.body, source.splitlines(), found)
    return found


def chapter_sources(chapter_dir: Path):
    """(label, source, is_pseudo_code) for each python fence of the chapter's pages."""
    for page in sorted(Path(chapter_dir).glob("*.md")):
        text = page.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            line = text[: m.start()].count("\n") + 1
            yield f"{page.name}:L{line}", m.group(1), bool(PSEUDO.match(m.group(1).lstrip()))


def notebook_sources(notebook: Path):
    """(label, source) for each code cell of the notebook."""
    for i, cell in enumerate(json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            yield f"{Path(notebook).name}:cell {i}", source


def _defs(sources, magics: bool) -> dict[str, str]:
    found: dict[str, str] = {}
    for _, source in sources:
        try:
            found.update(top_level_defs(source, magics))
        except SyntaxError:
            pass
    return found


def chapter_defs(chapter_dir: Path) -> dict[str, str]:
    return _defs(((label, src) for label, src, _ in chapter_sources(chapter_dir)), magics=False)


def notebook_defs(notebook: Path) -> dict[str, str]:
    return _defs(notebook_sources(notebook), magics=True)


def parse_failures(chapter_dir: Path) -> list[str]:
    """'<source>: cannot parse: <error>' for each notebook cell and non-pseudo-code fence that does not parse."""
    chapter_dir = Path(chapter_dir)
    failures = []
    for label, source, pseudo in chapter_sources(chapter_dir):
        if not pseudo:
            try:
                ast.parse(source)
            except SyntaxError as e:
                failures.append(f"{label}: cannot parse: {e.msg} (line {e.lineno})")
    for notebook in sorted((chapter_dir / "examples").glob("*.ipynb")):
        for label, source in notebook_sources(notebook):
            try:
                ast.parse(blank_magics(source))
            except SyntaxError as e:
                failures.append(f"{label}: cannot parse: {e.msg} (line {e.lineno})")
    return failures


def shared_defs(chapter_dir: Path) -> dict[str, tuple[str, str]]:
    """name -> (chapter source, notebook source) for every name defined in both (last definitions)."""
    chapter_dir = Path(chapter_dir)
    in_chapter = chapter_defs(chapter_dir)
    in_notebook: dict[str, str] = {}
    for notebook in sorted((chapter_dir / "examples").glob("*.ipynb")):
        in_notebook.update(notebook_defs(notebook))
    return {name: (in_chapter[name], in_notebook[name]) for name in in_chapter if name in in_notebook}


def normalised(source: str) -> str:
    """The definition after ast.unparse with docstrings removed (comments and layout are gone already)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, DEFS):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: list_shared_defs.py <chapter-dir>", file=sys.stderr)
        return 2
    for name in shared_defs(Path(argv[1])):
        print(name)
    for failure in parse_failures(Path(argv[1])):
        print(failure, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
