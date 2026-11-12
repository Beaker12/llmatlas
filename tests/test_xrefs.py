"""Chapter admonitions that point readers at a notebook must not use 'Section N' or
'cells N-M' references, which drift; they use MyST {ref} labels instead."""
import re
from pathlib import Path

import pytest

from tests._files import tracked

ROOT = Path(__file__).resolve().parents[1]
ADMONITION = re.compile(r"```\{admonition\}[^\n]*\n[\s\S]*?```", re.MULTILINE)
BAD = re.compile(r"\b(Sections?\s+\d+(?:-\d+)?|cells?\s+\d+(?:-\d+)?)\b", re.IGNORECASE)


def chapter_pages():
    return tracked("chapters/*/*.md")


@pytest.mark.parametrize("path", chapter_pages(), ids=lambda p: str(p.relative_to(ROOT)))
def test_notebook_admonitions_use_labels_not_numbers(path):
    text = path.read_text(encoding="utf-8")
    bad = []
    for m in ADMONITION.finditer(text):
        body = m.group(0)
        if "notebook" not in body.lower():
            continue
        for hit in BAD.finditer(body):
            line = text[: m.start() + hit.start()].count("\n") + 1
            bad.append(f"{path.relative_to(ROOT)}:{line}: {hit.group(0)!r}")
    assert not bad, "numbered notebook references in admonitions:\n" + "\n".join(bad)
