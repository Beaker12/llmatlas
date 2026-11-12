"""Chapter admonitions that point readers at a notebook must not use 'Section N' or
'cells N-M' references, which drift; they use MyST {ref} labels instead."""
import re
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check, tracked
from tests._pages import blank_code, blank_inline_code, blank_math, line_of, pages

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


# Body text: references go through {ref} and {numref}, never a literal section number or an unwritten chapter.
LAST_CHAPTER = 6
SECTION_NUMBER = re.compile(r"\bSections?\s+\d+\.\d+")
CHAPTER_NUMBERS = re.compile(r"\bChapters?\s+(\d+(?:\s*(?:,|and|to|or|–|-)\s*\d+)*)")
ROLE = re.compile(r"\{[\w:-]+\}`[^`\n]*`")


def body_text_failures(chapter_dir):
    failures = []
    for page in pages(chapter_dir):
        text = blank_code(page.read_text(encoding="utf-8"))
        text = blank_inline_code(blank_math(ROLE.sub(lambda m: " " * len(m.group(0)), text)))
        for m in SECTION_NUMBER.finditer(text):
            failures.append(f"{page.name}:L{line_of(text, m.start())}: literal {m.group(0)!r}; use {{ref}}")
        for m in CHAPTER_NUMBERS.finditer(text):
            numbers = [int(n) for n in re.findall(r"\d+", m.group(1))]
            if any(n > LAST_CHAPTER for n in numbers):
                failures.append(f"{page.name}:L{line_of(text, m.start())}: {m.group(0)!r} names an unwritten chapter")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_body_text_has_no_literal_section_or_unwritten_chapter(chapter_dir):
    chapter_check("xrefs")(chapter_dir, body_text_failures(chapter_dir))


def _page(tmp_path, text):
    (tmp_path / "index.md").write_text(text, encoding="utf-8")
    return tmp_path


def test_a_literal_section_number_in_body_text_is_reported(tmp_path):
    chapter = _page(tmp_path, "Intro.\n\nSee Section 2.1 and Sections 3.2 and 3.3.\n")
    assert body_text_failures(chapter) == ["index.md:L3: literal 'Section 2.1'; use {ref}",
                                           "index.md:L3: literal 'Sections 3.2'; use {ref}"]


def test_a_chapter_above_six_is_reported(tmp_path):
    chapter = _page(tmp_path, "As Chapter 10 shows, and Chapters 3 and 7, but Chapter 6 is fine.\n")
    assert body_text_failures(chapter) == [
        "index.md:L1: 'Chapter 10' names an unwritten chapter",
        "index.md:L1: 'Chapters 3 and 7' names an unwritten chapter"]


def test_roles_code_and_math_are_exempt(tmp_path):
    text = ("See {ref}`Section 2.1 <sec-ch1-a>` and {numref}`Chapter 10`, `Section 2.2`, $Chapter 9$.\n\n"
            "```text\nSection 4.4 and Chapter 12\n```\n\nSection 2 and Chapter 5 are fine.\n")
    assert body_text_failures(_page(tmp_path, text)) == []
