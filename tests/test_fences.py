"""Every ```python fence in a chapter either passes pyflakes (no undefined names,
no syntax errors) or is marked on its first line as pseudo-code."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```python\n([\s\S]*?)```")
MARKER = "# pseudo-code"


def fences():
    out = subprocess.run(["git", "ls-files", "chapters/*.md", "chapters/**/*.md"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    for rel in sorted(set(out)):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            line = text[: m.start()].count("\n") + 2
            yield pytest.param(rel, line, m.group(1), id=f"{rel}:{line}")


@pytest.mark.parametrize("rel,line,code", list(fences()))
def test_fence_is_runnable_or_marked_pseudo_code(rel, line, code):
    if code.lstrip().startswith(MARKER):
        pytest.skip("marked pseudo-code")
    proc = subprocess.run([sys.executable, "-m", "pyflakes"], input=code,
                          capture_output=True, text=True)
    problems = [l for l in proc.stdout.splitlines()
                if "undefined name" in l or "invalid syntax" in l or "SyntaxError" in l]
    assert not problems, f"{rel}:{line}:\n" + "\n".join(problems)
