"""The book's file lists, one home each, for the tools and the tests.

pages(chapter_dir)      every *.md directly in the chapter directory except notation-local.md, sorted
notebooks(chapter_dir)  the chapter's notebooks, examples/*.ipynb, sorted
bib_files(root)         references.bib, then bib/*.bib sorted (whether or not they exist)
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_LIST = "notation-local.md"


def pages(chapter_dir) -> list[Path]:
    """Every *.md directly in the chapter directory except notation-local.md (a symbol list, not a page), sorted."""
    return sorted(p for p in Path(chapter_dir).glob("*.md") if p.name != LOCAL_LIST)


def notebooks(chapter_dir) -> list[Path]:
    return sorted((Path(chapter_dir) / "examples").glob("*.ipynb"))


def bib_files(root=ROOT) -> list[Path]:
    root = Path(root)
    return [root / "references.bib"] + sorted((root / "bib").glob("*.bib"))
