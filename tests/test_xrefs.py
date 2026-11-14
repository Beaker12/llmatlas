"""Chapter admonitions that point readers at a notebook must not use 'Section N' or
'cells N-M' references, which drift; they use MyST {ref} labels instead."""
import re
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check, chapter_constants, tracked
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
# Interface labels are the section labels other chapters link to. Which labels they are, and the chapter each
# belongs to, is fixed here; the heading each sits above is declared by its chapter, in INTERFACE_LABELS
# (label -> heading text) of tests/test_chNN_snippets.py, so a chapter task that moves one edits only its own files.
INTERFACE_LABEL_NAMES = {
    "sec-ch1-perplexity": "01-foundations",
    "sec-ch1-layernorm": "01-foundations",
    "sec-ch1-modern-block": "01-foundations",
    "sec-ch1-rope": "01-foundations",
    "sec-ch1-relative-bias": "01-foundations",
    "sec-ch2-sqrt-dk": "02-mathematics",
    "sec-ch2-pre-post-ln": "02-mathematics",
    "sec-ch2-flops": "02-mathematics",
    "sec-ch2-residual-init": "02-mathematics",
    "sec-ch3-epochs": "03-data-acquisition",
    "sec-ch4-dedup-scope": "04-preprocessing",
}
HEADING = re.compile(r"^#{2,4} (.*?)\s*$")
TARGET = re.compile(r"^\(([A-Za-z0-9_.:-]+)\)=$")
LABEL_FORM = re.compile(r"sec-(ch[1-6]|tdd)-[a-z0-9]+(-[a-z0-9]+)*")


def labels_above(lines, i):
    """The target labels stacked directly above line i, each followed by one blank line, nearest first."""
    labels, k = [], i - 1
    while k >= 1 and lines[k] == "" and TARGET.match(lines[k - 1]):
        labels.append(TARGET.match(lines[k - 1]).group(1))
        k -= 2
    return labels


def heading_labels(chapter_dir):
    """{label: [(page name, heading text), ...]} for the labels directly above each level 2-4 heading."""
    found = {}
    for page in pages(chapter_dir):
        text = page.read_text(encoding="utf-8")
        lines = text.split("\n")
        for i, line in enumerate(blank_code(text).split("\n")):
            heading = HEADING.match(line)
            if heading:
                for label in labels_above(lines, i):
                    found.setdefault(label, []).append((page.name, heading.group(1)))
    return found


def interface_label_failures(chapter_dir, declared, names=None):
    """Failures for a chapter's INTERFACE_LABELS `declared` (label -> heading text): each interface label of the
    chapter is declared, each declared one is an interface label of the chapter, and each sits directly above a
    heading with the declared text in the chapter's pages."""
    names = INTERFACE_LABEL_NAMES if names is None else names
    fix = f"update INTERFACE_LABELS in tests/test_ch{Path(chapter_dir).name[:2]}_snippets.py"
    expected = {label for label, chapter in names.items() if chapter == Path(chapter_dir).name}
    failures = [f"interface label {label!r} is not declared in INTERFACE_LABELS; {fix}"
                for label in sorted(expected - set(declared))]
    failures += [f"INTERFACE_LABELS declares {label!r}, which is not an interface label of this chapter"
                 for label in sorted(set(declared) - expected)]
    found = heading_labels(chapter_dir)
    for label, heading in sorted(declared.items()):
        places = found.get(label, [])
        if heading not in [h for _page, h in places]:
            where = f"it is above {places[0][1]!r} in {places[0][0]}" if places else "no heading carries it"
            failures.append(f"interface label {label!r} is not above {heading!r} ({where}); {fix}")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_interface_labels_sit_above_the_headings_their_chapter_declares(chapter_dir):
    declared = chapter_constants(chapter_dir).INTERFACE_LABELS
    assert isinstance(declared, dict) and all(isinstance(v, str) for v in declared.values())
    failures = interface_label_failures(chapter_dir, declared)
    assert not failures, "\n".join(failures)


def test_every_interface_label_belongs_to_a_chapter():
    assert set(INTERFACE_LABEL_NAMES.values()) <= set(CHAPTER_IDS)
    assert all(label.startswith(f"sec-ch{int(chapter[:2])}-") for label, chapter in INTERFACE_LABEL_NAMES.items())


def test_a_moved_undeclared_or_unlisted_interface_label_is_reported(tmp_path):
    chapter = tmp_path / "02-x"
    chapter.mkdir()
    (chapter / "index.md").write_text("# T\n\n(sec-ch2-flops)=\n\n## Training Complexity\n\n(sec-ch2-a)=\n\n"
                                      "## Other\n", encoding="utf-8")
    names = {"sec-ch2-flops": "02-x", "sec-ch2-gone": "02-x", "sec-ch1-x": "01-y"}
    ok = {"sec-ch2-flops": "Training Complexity", "sec-ch2-gone": "Other"}
    fix = "; update INTERFACE_LABELS in tests/test_ch02_snippets.py"
    assert interface_label_failures(chapter, ok, names) == [
        "interface label 'sec-ch2-gone' is not above 'Other' (no heading carries it)" + fix]
    moved = {"sec-ch2-flops": "Other", "sec-ch2-gone": "Other", "sec-ch2-a": "Other"}
    assert interface_label_failures(chapter, moved, names) == [
        "INTERFACE_LABELS declares 'sec-ch2-a', which is not an interface label of this chapter",
        "interface label 'sec-ch2-flops' is not above 'Other' (it is above 'Training Complexity' in index.md)" + fix,
        "interface label 'sec-ch2-gone' is not above 'Other' (no heading carries it)" + fix]
    assert interface_label_failures(chapter, {}, names) == [
        "interface label 'sec-ch2-flops' is not declared in INTERFACE_LABELS" + fix,
        "interface label 'sec-ch2-gone' is not declared in INTERFACE_LABELS" + fix]


FURTHER_READING = re.compile(r"further reading", re.I)


def heading_label_failures(chapter_dirs):
    """Headings without a target line, labels of the wrong form or prefix, and the Further Reading rule (the
    heading matched case-insensitively): labels below it start <prefix>further-reading-, and no other does."""
    bad = []
    for chapter_dir in chapter_dirs:
        for page in pages(chapter_dir):
            prefix = "sec-tdd-" if page.name == "tokenization-deep-dive.md" else f"sec-ch{int(chapter_dir.name[:2])}-"
            text = page.read_text(encoding="utf-8")
            lines, prose = text.split("\n"), blank_code(text).split("\n")
            in_reading_list = False
            for i, line in enumerate(prose):
                heading = HEADING.match(line)
                if not heading:
                    continue
                labels = labels_above(lines, i)
                if not labels:
                    bad.append(f"{page.name}:L{i + 1}: heading has no target line above it: {line!r}")
                for label in labels:
                    where = f"{page.name}:L{i + 1}: label {label!r}"
                    if not LABEL_FORM.fullmatch(label):
                        bad.append(f"{where} does not match sec-(ch1-6|tdd)-words")
                    if not label.startswith(prefix):
                        bad.append(f"{where} lacks the prefix {prefix!r}")
                    if FURTHER_READING.fullmatch(heading.group(1).strip()):
                        in_reading_list = True
                    elif in_reading_list != label.startswith(prefix + "further-reading-"):
                        bad.append(f"{where}: labels below 'Further Reading' start {prefix}further-reading-, "
                                   f"and no other label does")
    return bad


def test_every_heading_has_a_section_label():
    bad = heading_label_failures(CHAPTER_DIRS)
    assert not bad, "\n".join(bad)


def test_the_further_reading_heading_is_matched_in_any_case(tmp_path):
    chapter = tmp_path / "02-x"
    chapter.mkdir()
    page = ("# T\n\n(sec-ch2-intro)=\n\n## Intro\n\n(sec-ch2-further-reading)=\n\n## Further reading\n\n"
            "(sec-ch2-further-reading-books)=\n\n### Books\n")
    (chapter / "index.md").write_text(page, encoding="utf-8")
    assert heading_label_failures([chapter]) == []
    (chapter / "index.md").write_text(page.replace("further-reading-books", "books"), encoding="utf-8")
    assert heading_label_failures([chapter]) == [
        "index.md:L13: label 'sec-ch2-books': labels below 'Further Reading' start sec-ch2-further-reading-, "
        "and no other label does"]


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
