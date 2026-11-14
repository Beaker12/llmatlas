"""Fold per-chapter bibliography files (bib/chNN.bib) into references.bib.

Usage: python tools/fold_bib.py bib/ch01.bib [bib/ch02.bib ...]

For each file, in the order given: its entries are appended verbatim to references.bib under a chapter
heading comment (the file's own leading `%` comment lines are dropped), the file is deleted, and its line
under `bibtex_bibfiles` in _config.yml is removed. The bib/ directory is removed once it is empty.

The fold refuses to run (and changes nothing) when a key of a file being folded is already defined, or
when two entries would share a normalised title: the same work under two keys has to be merged by hand
first (keep the key already in references.bib and update the citations).
"""
import re
import sys
from pathlib import Path

from pybtex.database import parse_string

ROOT = Path(__file__).resolve().parent.parent
RULE = "% " + "-" * 75


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"\\[a-zA-Z]+|[{}]", "", title).lower())


def _entries(text: str):
    return parse_string(text, bib_format="bibtex").entries


def _body(text: str) -> str:
    lines = text.splitlines()
    while lines and (lines[0].startswith("%") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).rstrip() + "\n"


def fold(paths, root=ROOT):
    root = Path(root)
    ref = root / "references.bib"
    combined = ref.read_text(encoding="utf-8").rstrip() + "\n"
    problems = []
    titles = {_title_key(e.fields.get("title", "")): k for k, e in _entries(combined).items()}
    for path in paths:
        path = root / path
        body = _body(path.read_text(encoding="utf-8"))
        known = _entries(combined)
        for key, entry in _entries(body).items():
            if key in known:
                problems.append(f"{path.name}: key {key} already defined")
            t = _title_key(entry.fields.get("title", ""))
            if t and t in titles and titles[t] != key:
                problems.append(f"{path.name}: {key} has the same title as {titles[t]}")
            titles.setdefault(t, key)
        chapter = int(re.search(r"(\d+)", path.stem).group(1))
        if body.strip():
            combined += f"\n{RULE}\n% Chapter {chapter} (folded from bib/{path.name})\n{RULE}\n\n{body}"
    if problems:
        raise SystemExit("fold refused:\n" + "\n".join(problems))
    _entries(combined)  # strict parse of the result
    ref.write_text(combined, encoding="utf-8")
    config = root / "_config.yml"
    text = config.read_text(encoding="utf-8")
    for path in paths:
        name = Path(path).name
        text = re.sub(rf"^[ \t]*- bib/{re.escape(name)}[ \t]*\n", "", text, flags=re.M)
        (root / path).unlink()
    config.write_text(text, encoding="utf-8")
    bib = root / "bib"
    if bib.is_dir() and not any(bib.iterdir()):
        bib.rmdir()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    fold(sys.argv[1:])
