"""Every file in _toc.yml exists, and no chapter page is a placeholder stub."""
from pathlib import Path

import yaml

from tests._files import tracked

ROOT = Path(__file__).resolve().parents[1]
STUB_MARKER = "Phase 1 Skeleton"


def toc_files():
    toc = yaml.safe_load((ROOT / "_toc.yml").read_text(encoding="utf-8"))
    files = [toc["root"]]

    def walk(node):
        if isinstance(node, dict):
            if "file" in node:
                files.append(node["file"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(toc.get("parts", []) + toc.get("chapters", []))
    return files


def test_every_toc_entry_exists():
    missing = []
    for f in toc_files():
        if not any((ROOT / f"{f}{ext}").exists() for ext in (".md", ".ipynb", ".rst")):
            missing.append(f)
    assert not missing, f"_toc.yml entries without a file: {missing}"


def test_no_chapter_page_is_a_stub():
    stubs = [p for p in tracked("chapters/**/*.md")
             if STUB_MARKER in p.read_text(encoding="utf-8")]
    assert not stubs, f"{len(stubs)} placeholder chapter(s) still present, e.g. {stubs[:3]}"
