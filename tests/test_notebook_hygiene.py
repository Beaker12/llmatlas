"""Notebooks must be publishable: no leaked local paths, no machine-specific kernel,
no runtime pip installs, and small enough to review."""
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 1_000_000
PIP_INSTALL = re.compile(r"""(!|%)pip\s+install|pip["']?\s*,\s*["']install|pip install""")


def notebooks():
    out = subprocess.run(["git", "ls-files", "*.ipynb", "**/*.ipynb"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    return sorted({ROOT / p for p in out})


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_is_clean(path):
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    problems = []
    if "/mnt/ztank" in raw or "/home/" in raw:
        problems.append("contains a local filesystem path")
    display = nb.get("metadata", {}).get("kernelspec", {}).get("display_name", "")
    if ".venv" in display or "(" in display:
        problems.append(f"machine-specific kernelspec display name {display!r}")
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and PIP_INSTALL.search(src):
            problems.append(f"cell {i} runs pip install at runtime")
    if path.stat().st_size > MAX_BYTES:
        problems.append(f"{path.stat().st_size} bytes exceeds {MAX_BYTES}")
    assert not problems, "; ".join(problems)
