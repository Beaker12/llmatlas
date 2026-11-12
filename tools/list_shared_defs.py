#!/usr/bin/env python3
"""List the functions and classes a chapter shares with its notebook.

A name is shared when it is defined at top level both in a ```python fence of the chapter's pages
(pseudo-code fences included) and in a code cell of the chapter's notebook.

Usage: python tools/list_shared_defs.py chapters/03-data-acquisition

Fences and cells that do not parse are skipped. When a name is defined more than once on one side,
the first definition is used.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

FENCE = re.compile(r"```python\n([\s\S]*?)```")


def top_level_defs(source: str) -> dict[str, str]:
    """name -> source text of each top-level function or class in `source` (decorators included)."""
    source = "\n".join("" if line.lstrip().startswith(("%", "!")) else line for line in source.splitlines())
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    lines = source.splitlines()
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name not in found:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            found[node.name] = "\n".join(lines[start - 1: node.end_lineno])
    return found


def chapter_defs(chapter_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for page in sorted(Path(chapter_dir).glob("*.md")):
        for m in FENCE.finditer(page.read_text(encoding="utf-8")):
            for name, source in top_level_defs(m.group(1)).items():
                found.setdefault(name, source)
    return found


def notebook_defs(notebook: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for cell in json.loads(Path(notebook).read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for name, text in top_level_defs(source).items():
                found.setdefault(name, text)
    return found


def shared_defs(chapter_dir: Path) -> dict[str, tuple[str, str]]:
    """name -> (chapter source, notebook source) for every name defined in both."""
    chapter_dir = Path(chapter_dir)
    in_chapter = chapter_defs(chapter_dir)
    in_notebook: dict[str, str] = {}
    for notebook in sorted((chapter_dir / "examples").glob("*.ipynb")):
        for name, text in notebook_defs(notebook).items():
            in_notebook.setdefault(name, text)
    return {name: (in_chapter[name], in_notebook[name]) for name in in_chapter if name in in_notebook}


def normalised(source: str) -> str:
    """The definition after ast.unparse with docstrings removed (comments and layout are gone already)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
