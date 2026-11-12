"""Source hygiene: no invisible characters in chapter pages or notebook code, and hashed data is never normalised.

Check name `hygiene`, one test per chapter.
- No literal U+200B, U+200C, U+200D, U+2060, U+FEFF, U+00A0 and no combining mark (category Mn) anywhere in a
  chapter page or in a notebook code-cell source. Escapes such as \\u200b written as text are fine.
- Every file listed in a checksums.json under the chapter directory has the attribute `-text`
  (`check-attr text` reports `unset`). Skipped when the directory is not in a version-control checkout.
"""
import json
import subprocess
import unicodedata
from pathlib import Path

import nbformat
import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check
from tests._notebooks import notebooks, read
from tests._pages import pages

INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00a0"}
VCS = "git"


def bad_characters(text, where):
    failures = []
    for number, line in enumerate(text.split("\n"), start=1):
        for ch in line:
            if ch in INVISIBLE or unicodedata.category(ch) == "Mn":
                failures.append(f"{where}:L{number}: literal U+{ord(ch):04X} {unicodedata.name(ch, 'unnamed')}")
    return failures


def character_failures(chapter_dir):
    failures = []
    for page in pages(chapter_dir):
        failures += bad_characters(page.read_text(encoding="utf-8"), page.name)
    for path in notebooks(chapter_dir):
        for index, cell in enumerate(read(path).cells):
            if cell.cell_type == "code":
                failures += bad_characters(cell.source, f"{path.name}:cell {index}")
    return failures


def in_checkout(directory):
    try:
        done = subprocess.run([VCS, "-C", str(directory), "rev-parse", "--is-inside-work-tree"],
                              capture_output=True, text=True)
    except OSError:
        return False
    return done.returncode == 0 and done.stdout.strip() == "true"


def checksum_failures(chapter_dir):
    chapter_dir = Path(chapter_dir)
    if not in_checkout(chapter_dir):
        return []
    failures = []
    for listing in sorted(chapter_dir.rglob("checksums.json")):
        data = json.loads(listing.read_text(encoding="utf-8"))
        names = data.get("sha256", data)
        for name in sorted(names):
            target = listing.parent / name
            out = subprocess.run([VCS, "-C", str(chapter_dir), "check-attr", "text", "--", str(target)],
                                 capture_output=True, text=True).stdout.strip()
            if not out.endswith(": text: unset"):
                failures.append(f"{target.relative_to(chapter_dir)}: listed in {listing.name} but attribute text is "
                                f"{out.rsplit(': ', 1)[-1] or 'unknown'}, not unset; add `-text` in .gitattributes")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_source_hygiene(chapter_dir):
    chapter_check("hygiene")(chapter_dir, character_failures(chapter_dir) + checksum_failures(chapter_dir))


def test_invisible_characters_in_a_page_are_reported(tmp_path):
    text = "ok\n\n```python\nx = 1\u200b\n```\n\nnon\u00a0breaking e\u0301 \\u200b\n"
    (tmp_path / "index.md").write_text(text, encoding="utf-8")
    assert character_failures(tmp_path) == [
        "index.md:L4: literal U+200B ZERO WIDTH SPACE",
        "index.md:L7: literal U+00A0 NO-BREAK SPACE",
        "index.md:L7: literal U+0301 COMBINING ACUTE ACCENT"]


def test_invisible_characters_in_notebook_code_are_reported(tmp_path):
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("a\u200bb"),
                                         nbformat.v4.new_code_cell("x = 1\ny = '\ufeff'")])
    (tmp_path / "examples").mkdir()
    nbformat.write(nb, str(tmp_path / "examples" / "n.ipynb"))
    assert character_failures(tmp_path) == ["n.ipynb:cell 1:L2: literal U+FEFF ZERO WIDTH NO-BREAK SPACE"]


def test_a_checksummed_file_without_minus_text_is_reported(tmp_path):
    subprocess.run([VCS, "-C", str(tmp_path), "init", "-q"], check=True, capture_output=True)
    data = tmp_path / "examples" / "data"
    data.mkdir(parents=True)
    (data / "checksums.json").write_text(json.dumps({"sha256": {"a.json": "0", "sub/b.json": "0"}}))
    (tmp_path / ".gitattributes").write_text("* text=auto\nexamples/data/a.json -text\n")
    failures = checksum_failures(tmp_path)
    assert len(failures) == 1 and failures[0].startswith("examples/data/sub/b.json: listed in checksums.json")
    (tmp_path / ".gitattributes").write_text("examples/data/** -text\n")
    assert checksum_failures(tmp_path) == []


def test_the_checksum_half_is_skipped_outside_a_checkout(tmp_path, monkeypatch):
    (tmp_path / "checksums.json").write_text(json.dumps({"sha256": {"a.json": "0"}}))
    monkeypatch.setattr("tests.test_source_hygiene.in_checkout", lambda directory: False)
    assert checksum_failures(tmp_path) == []
