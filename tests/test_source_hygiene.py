"""Source hygiene: no invisible characters in chapter pages or notebook code, and hashed data is never normalised.

Check name `hygiene`, one test per chapter.
- No literal U+200B, U+200C, U+200D, U+2060, U+FEFF, U+00A0 and no combining mark (category Mn) anywhere in a
  chapter page or in a notebook code-cell source. Escapes such as \\u200b written as text are fine.
- Every file listed in a checksums.json under the chapter directory has the attribute `-text`
  (`check-attr text` reports `unset`). Skipped when the directory is not in a version-control checkout.
"""
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import nbformat
import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check
from tests._notebooks import notebooks, read
from tests._pages import pages

INVISIBLE = ({"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00a0", "\u00ad", "\u200e", "\u200f"}
             | {chr(c) for c in range(0x202A, 0x202F)} | {chr(c) for c in range(0x2061, 0x2065)})
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


def git_text_attribute(chapter_dir, target):
    """The `text` attribute of `target` as reported by `check-attr`: set, unset, auto or unspecified."""
    out = subprocess.run([VCS, "-C", str(chapter_dir), "check-attr", "text", "--", str(target)],
                         capture_output=True, text=True).stdout.strip()
    return out.rsplit(": ", 1)[-1]


def attributes_file(start):
    """The nearest .gitattributes in `start` or a parent directory, or None."""
    for directory in [Path(start).resolve(), *Path(start).resolve().parents]:
        if (directory / ".gitattributes").is_file():
            return directory / ".gitattributes"
    return None


def file_text_attribute(target, gitattributes):
    """The `text` attribute of `target` from a .gitattributes file read directly (no git needed).

    Handles the pattern forms the repository uses: `*`, `*.ext`, `dir/**` and a plain relative path. The last
    matching line wins. Any other form raises NotImplementedError instead of guessing.
    """
    gitattributes = Path(gitattributes)
    relative = Path(target).resolve().relative_to(gitattributes.parent.resolve()).as_posix()
    value = "unspecified"
    for number, raw in enumerate(gitattributes.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, *attrs = line.split()
        if pattern == "*":
            matches = True
        elif re.fullmatch(r"\*\.[\w.-]+", pattern):
            matches = relative.endswith(pattern[1:])
        elif re.fullmatch(r"[\w./-]+/\*\*", pattern):
            matches = relative.startswith(pattern[:-2])
        elif re.fullmatch(r"[\w./-]+", pattern):
            matches = relative == pattern
        else:
            raise NotImplementedError(f"{gitattributes}:{number}: .gitattributes pattern {pattern!r} is not handled "
                                      "by tests/test_source_hygiene.py; extend file_text_attribute")
        if matches:
            for attr in attrs:
                if attr == "text":
                    value = "set"
                elif attr == "-text":
                    value = "unset"
                elif attr == "!text":
                    value = "unspecified"
                elif attr == "text=auto":
                    value = "auto"
    return value


def checksum_failures(chapter_dir, method=None):
    """Files listed in a checksums.json whose `text` attribute is not unset.

    method "git" asks git, "file" reads .gitattributes directly; the default is git inside a checkout and the
    file otherwise, so the result does not depend on whether .git is present.
    """
    chapter_dir = Path(chapter_dir)
    method = method or ("git" if in_checkout(chapter_dir) else "file")
    gitattributes = attributes_file(chapter_dir) if method == "file" else None
    failures = []
    for listing in sorted(chapter_dir.rglob("checksums.json")):
        data = json.loads(listing.read_text(encoding="utf-8"))
        for name in sorted(data.get("sha256", data)):
            target = listing.parent / name
            value = (git_text_attribute(chapter_dir, target) if method == "git" else
                     file_text_attribute(target, gitattributes) if gitattributes else "unspecified")
            if value != "unset":
                failures.append(f"{target.relative_to(chapter_dir)}: listed in {listing.name} but attribute text is "
                                f"{value}, not unset; add `-text` in .gitattributes")
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


def test_the_checksum_half_does_not_depend_on_git_being_present(tmp_path, monkeypatch):
    data = tmp_path / "examples" / "data"
    data.mkdir(parents=True)
    (data / "checksums.json").write_text(json.dumps({"sha256": {"a.json": "0"}}))
    (tmp_path / ".gitattributes").write_text("* text=auto\n*.json text\n")
    monkeypatch.setattr("tests.test_source_hygiene.in_checkout", lambda directory: False)
    failures = checksum_failures(tmp_path)
    assert len(failures) == 1 and "attribute text is set, not unset" in failures[0]
    (tmp_path / ".gitattributes").write_text("* text=auto\nexamples/data/** -text\n")
    assert checksum_failures(tmp_path) == []
    (tmp_path / ".gitattributes").write_text("* text=auto\n")
    assert "attribute text is auto" in checksum_failures(tmp_path)[0]
    (tmp_path / ".gitattributes").unlink()
    assert "attribute text is unspecified" in checksum_failures(tmp_path)[0]


@pytest.mark.parametrize("pattern", ["data/*.json", "**/data/*", "[ab].json", "!x.json"])
def test_an_unhandled_gitattributes_pattern_fails_loudly(tmp_path, pattern):
    (tmp_path / ".gitattributes").write_text(f"{pattern} -text\n")
    with pytest.raises(NotImplementedError, match="not handled"):
        file_text_attribute(tmp_path / "a.json", tmp_path / ".gitattributes")


def test_the_two_attribute_methods_agree_for_every_checksummed_file():
    from tests._files import ROOT
    if not in_checkout(ROOT):
        pytest.skip("not a git checkout")
    attributes = ROOT / ".gitattributes"
    listed = [p for d in CHAPTER_DIRS for p in sorted(d.rglob("checksums.json"))]
    assert listed
    for listing in listed:
        for name in json.loads(listing.read_text(encoding="utf-8")).get("sha256", {}):
            target = listing.parent / name
            assert git_text_attribute(ROOT, target) == file_text_attribute(target, attributes), target


def test_the_two_attribute_methods_give_the_same_failures():
    for chapter in CHAPTER_DIRS:
        if not in_checkout(chapter):
            pytest.skip("not a git checkout")
        assert checksum_failures(chapter, method="file") == checksum_failures(chapter, method="git")


def test_more_invisible_characters_are_reported(tmp_path):
    banned = ["­", "‎", "‏", "‪", "‮", "⁡", "⁤"]
    (tmp_path / "index.md").write_text("\n".join(f"a{c}b" for c in banned) + "\nplain – dash\n", encoding="utf-8")
    assert [f.split(":")[1] for f in character_failures(tmp_path)] == [f"L{i}" for i in range(1, 8)]
