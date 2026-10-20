"""Fail if any Markdown page contains mdformat-style escaped LaTeX inside math."""
import re
import subprocess
from pathlib import Path

import pytest

from tools.unescape_math import math_regions

ROOT = Path(__file__).resolve().parents[1]
ESCAPED = re.compile(r"\\\\[A-Za-z]|\\_")
DOUBLED = re.compile(r"\\\\(?!\s|$)")


def tracked_markdown():
    out = subprocess.run(
        ["git", "ls-files", "*.md", "**/*.md"], cwd=ROOT, capture_output=True, text=True
    ).stdout.split()
    return sorted({ROOT / p for p in out})


@pytest.mark.parametrize("path", tracked_markdown(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_escaped_latex_inside_math(path):
    text = path.read_text(encoding="utf-8")
    bad = []
    for start, end in math_regions(text):
        region = text[start:end]
        if ESCAPED.search(region) or DOUBLED.search(region):
            line = text[:start].count("\n") + 1
            bad.append(f"{path.relative_to(ROOT)}:{line}: {region[:60]!r}")
    assert not bad, "escaped LaTeX inside math:\n" + "\n".join(bad)


@pytest.mark.parametrize("path", tracked_markdown(), ids=lambda p: str(p.relative_to(ROOT)))
def test_display_math_blocks_start_after_a_blank_line(path):
    """A `$$` block that directly follows a prose line is a paragraph continuation and
    renders inline instead of as display math (seen in the Phase 4 render check)."""
    lines = path.read_text(encoding="utf-8").split("\n")
    bad = []
    for n, line in enumerate(lines):
        if not line.lstrip().startswith("$$") or n == 0:
            continue
        prev = lines[n - 1]
        if prev.strip() and "$$" not in prev:
            bad.append(f"{path.relative_to(ROOT)}:{n + 1}")
    assert not bad, "display math without a preceding blank line:\n" + "\n".join(bad)
