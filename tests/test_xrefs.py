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
LAST_CHAPTER = len(CHAPTER_DIRS)        # the chapters written so far: one directory each
JOIN = r"\s*(?:,|and|or|to|–|-)\s*"
SECTION_NUMBERS = re.compile(rf"\b(Sections?)\s+(\d+\.\d+(?:{JOIN}\d+\.\d+)*)")
SIGN_NUMBERS = re.compile(rf"(§+)\s*(\d+(?:\.\d+)*(?:{JOIN}\d+(?:\.\d+)*)*)")
CHAPTER_NUMBERS = re.compile(rf"\bChapters?\s+(\d+(?:{JOIN}\d+)*)")
ROLE = re.compile(r"\{[\w:-]+\}`[^`\n]*`")


def body_text_failures(chapter_dir):
    failures = []
    for page in pages(chapter_dir):
        text = blank_code(page.read_text(encoding="utf-8"))
        text = blank_inline_code(blank_math(ROLE.sub(lambda m: " " * len(m.group(0)), text)))
        for pattern in (SECTION_NUMBERS, SIGN_NUMBERS):
            for m in pattern.finditer(text):
                for number in re.findall(r"\d+(?:\.\d+)*", m.group(2)):
                    failures.append(f"{page.name}:L{line_of(text, m.start())}: literal '{m.group(1)} {number}'; use {{ref}}")
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
                                           "index.md:L3: literal 'Sections 3.2'; use {ref}",
                                           "index.md:L3: literal 'Sections 3.3'; use {ref}"]


def test_section_signs_are_reported_and_the_last_chapter_comes_from_the_directories(tmp_path):
    chapter = _page(tmp_path, "See §3.1 and §§4.2-4.3 for details.\n")
    assert body_text_failures(chapter) == ["index.md:L1: literal '§ 3.1'; use {ref}",
                                           "index.md:L1: literal '§§ 4.2'; use {ref}",
                                           "index.md:L1: literal '§§ 4.3'; use {ref}"]
    assert LAST_CHAPTER == len(CHAPTER_DIRS) == 6


def test_a_chapter_above_six_is_reported(tmp_path):
    chapter = _page(tmp_path, "As Chapter 10 shows, and Chapters 3 and 7, but Chapter 6 is fine.\n")
    assert body_text_failures(chapter) == [
        "index.md:L1: 'Chapter 10' names an unwritten chapter",
        "index.md:L1: 'Chapters 3 and 7' names an unwritten chapter"]


def test_roles_code_and_math_are_exempt(tmp_path):
    text = ("See {ref}`Section 2.1 <sec-ch1-a>` and {numref}`Chapter 10`, `Section 2.2`, $Chapter 9$.\n\n"
            "```text\nSection 4.4 and Chapter 12\n```\n\nSection 2 and Chapter 5 are fine.\n")
    assert body_text_failures(_page(tmp_path, text)) == []


# Section labels: every level 2-4 heading of the chapter pages and the deep-dive carries a target line
# `(label)=` and a blank line directly above it (a host heading may carry two stacked target lines).
# Interface labels, which other chapters link to, are pinned to a page and heading text in this table:
# a chapter task that moves one to a new heading edits its row.
INTERFACE_LABELS = {
    "sec-ch1-perplexity": ("01-foundations", "4.3 Evaluation Metrics"),
    "sec-ch1-layernorm": ("01-foundations", "Layer Normalization"),
    "sec-ch1-modern-block": ("01-foundations", "3.3 What Changed Since 2017"),
    "sec-ch1-rope": ("01-foundations", "Positional Encodings"),
    "sec-ch1-relative-bias": ("01-foundations", "Positional Encodings"),
    "sec-ch2-sqrt-dk": ("02-mathematics", "0.1 A matrix is a map between spaces"),
    "sec-ch2-pre-post-ln": ("02-mathematics", "Residual Connections"),
    "sec-ch2-flops": ("02-mathematics", "Training Complexity"),
    "sec-ch2-residual-init": ("02-mathematics", "Transformer-Specific Initialization"),
    "sec-ch3-epochs": ("03-data-acquisition", "1.3 How much is enough"),
    "sec-ch4-dedup-scope": ("04-preprocessing", "5.3 Near-duplicate detection with MinHash and LSH"),
}
HEADING = re.compile(r"^#{2,4} (.*?)\s*$")
TARGET = re.compile(r"^\(([A-Za-z0-9_.:-]+)\)=$")
LABEL_FORM = re.compile(r"sec-(ch[1-6]|tdd)-[a-z0-9]+(-[a-z0-9]+)*")


def test_every_heading_has_a_section_label():
    bad, pinned = [], set()
    for chapter_dir in CHAPTER_DIRS:
        for page in pages(chapter_dir):
            prefix = "sec-tdd-" if page.name == "tokenization-deep-dive.md" else f"sec-ch{int(chapter_dir.name[:2])}-"
            text = page.read_text(encoding="utf-8")
            lines, prose = text.split("\n"), blank_code(text).split("\n")
            in_reading_list = False
            for i, line in enumerate(prose):
                heading = HEADING.match(line)
                if not heading:
                    continue
                labels, k = [], i - 1
                while k >= 1 and lines[k] == "" and TARGET.match(lines[k - 1]):
                    labels.append(TARGET.match(lines[k - 1]).group(1))
                    k -= 2
                if not labels:
                    bad.append(f"{page.name}:L{i + 1}: heading has no target line above it: {line!r}")
                for label in labels:
                    where = f"{page.name}:L{i + 1}: label {label!r}"
                    if not LABEL_FORM.fullmatch(label):
                        bad.append(f"{where} does not match sec-(ch1-6|tdd)-words")
                    if label in INTERFACE_LABELS:
                        if (chapter_dir.name, heading.group(1)) != INTERFACE_LABELS[label]:
                            bad.append(f"{where} is on {heading.group(1)!r}, expected {INTERFACE_LABELS[label][1]!r}")
                        else:
                            pinned.add(label)
                    elif not label.startswith(prefix):
                        bad.append(f"{where} lacks the prefix {prefix!r}")
                    if heading.group(1) == "Further Reading":
                        in_reading_list = True
                    elif in_reading_list != label.startswith(prefix + "further-reading-"):
                        bad.append(f"{where}: labels below 'Further Reading' start {prefix}further-reading-, "
                                   f"and no other label does")
    bad += [f"interface label {label!r} is not above {INTERFACE_LABELS[label][1]!r}"
            for label in sorted(set(INTERFACE_LABELS) - pinned)]
    assert not bad, "\n".join(bad)


def test_target_labels_are_unique_book_wide():
    seen, bad = {}, []
    for path in tracked("*.md"):
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            target = TARGET.match(line)
            if target:
                where = f"{path.relative_to(ROOT)}:L{number}"
                if target.group(1) in seen:
                    bad.append(f"{where}: {target.group(1)!r} also at {seen[target.group(1)]}")
                seen[target.group(1)] = where
    assert not bad, "\n".join(bad)
