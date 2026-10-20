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
