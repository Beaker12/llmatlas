"""Page order, exercises with solutions, and section size.

`exercises` (one test per chapter; tokenization-deep-dive.md is exempt from this check)
  The `##` headings of the page are, in this order: an overview ("Overview" or "Chapter Overview"), "Learning
  Objectives", "Hands-On" (or "Hands-On Learning"), one or more numbered sections ("1. ...", "2 ..."), "Summary",
  "Exercises", "Further Reading". Titles are matched whole, case-insensitively, after stripping a leading section
  number ("10. Summary" is the Summary). The Exercises section has at least one exercise, written either as numbered
  list items directly under the heading or as `### Exercise N` / `#### Exercise N` headings (not both). Each exercise
  is followed, before the next one, by an admonition titled `Solution N` with `:class: dropdown` (N is the heading's
  number, or the item's position), and the numbers of exercises and solutions are equal.
`structure` (one test per chapter, every page including the deep-dive)
  No `##` section exceeds 35% of the page's lines (lines from its heading to the next `##` heading or the end).
"""
import re
import shutil

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._pages import blank_code, pages

DEEP_DIVE = "tokenization-deep-dive.md"
FRONT = ["Overview", "Learning Objectives", "Hands-On"]
BACK = ["Summary", "Exercises", "Further Reading"]
TITLES = {"Overview": {"overview", "chapter overview"}, "Learning Objectives": {"learning objectives"},
          "Hands-On": {"hands-on", "hands-on learning"}, "Summary": {"summary"}, "Exercises": {"exercises"},
          "Further Reading": {"further reading"}}
EXERCISE_HEADING = re.compile(r"^#{3,4}[ \t]+Exercise[ \t]+(\d+)\b")


def canonical(title):
    """The title without a leading section number, case-folded."""
    return re.sub(r"^\d+(?:\.\d+)*\.?[ \t]+", "", title).strip().casefold()
MAX_SHARE = 0.35
HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
NUMBERED = re.compile(r"^\d+(?:\.|\s)")
ITEM = re.compile(r"^[ ]{0,3}\d+\.[ \t]+\S")
SOLUTION = re.compile(r"^[ \t]*(?:`{3,}|:{3,})\{admonition\}[ \t]+Solution[ \t]+(\d+)[ \t]*$")
ADMONITION = re.compile(r"^[ \t]*(?:`{3,}|:{3,})\{admonition\}")


def headings(text, level=2):
    """[(line number, title)] of the headings of exactly `level`, outside code and directives."""
    found = []
    for number, line in enumerate(blank_code(text, directives=True).split("\n"), start=1):
        m = HEADING.match(line)
        if m and len(m.group(1)) == level:
            found.append((number, m.group(2)))
    return found


def order_failures(name, text):
    top = headings(text)
    titles = [t for _, t in top]
    position = {}
    for phrase in FRONT + BACK:
        for i, title in enumerate(titles):
            if canonical(title) in TITLES[phrase]:
                position[phrase] = i
                break
        else:
            return [f"{name}:L1: no '##' heading titled {phrase!r}"]
    named = set().union(*TITLES.values())
    numbered = [i for i, t in enumerate(titles) if NUMBERED.match(t) and canonical(t) not in named]
    if not numbered:
        return [f"{name}:L1: no numbered '##' sections"]
    failures = []
    expected = FRONT + ["<numbered>"] + BACK
    first_numbered, last_numbered = numbered[0], numbered[-1]
    order = [position[p] for p in FRONT] + [first_numbered] + [position[p] for p in BACK]
    wrong = next((b for a, b in zip(order, order[1:]) if b < a), None)
    if wrong is None and last_numbered > position["Summary"]:
        wrong = last_numbered
    if wrong is not None:
        failures.append(f"{name}:L{top[wrong][0]}: '{top[wrong][1]}' is out of order, expected "
                        f"{' < '.join(expected)}")
    return failures


def exercise_failures(name, text):
    top = headings(text)
    start = next((i for i, (_, t) in enumerate(top) if canonical(t) in TITLES["Exercises"]), None)
    if start is None:
        return []                      # reported by order_failures
    first = top[start][0]
    last = top[start + 1][0] - 1 if start + 1 < len(top) else len(text.split("\n"))
    lines = text.split("\n")
    shown = blank_code(text, directives=True).split("\n")
    events = []                        # (line, "item"|"heading"|"solution", number)
    for number in range(first + 1, last + 1):
        heading = EXERCISE_HEADING.match(shown[number - 1])
        if ITEM.match(shown[number - 1]):
            events.append((number, "item", None))
        elif heading:
            events.append((number, "heading", int(heading.group(1))))
        elif SOLUTION.match(lines[number - 1]):
            events.append((number, "solution", int(SOLUTION.match(lines[number - 1]).group(1))))
    failures = []
    exercises = [e for e in events if e[1] != "solution"]
    solutions = [e for e in events if e[1] == "solution"]
    if not exercises:
        return [f"{name}:L{first}: the Exercises section has no exercises (numbered list items or 'Exercise N' headings)"]
    if {kind for _, kind, _ in exercises} == {"item", "heading"}:
        return [f"{name}:L{first}: the Exercises section mixes numbered list items and 'Exercise N' headings; use one form"]
    for k, event in enumerate(exercises, start=1):
        line, kind, written = event
        wanted = written if kind == "heading" else k
        position = events.index(event)
        after = events[position + 1] if position + 1 < len(events) else None
        if not after or after[1] != "solution" or after[2] != wanted:
            failures.append(f"{name}:L{line}: exercise {wanted} is not followed by a 'Solution {wanted}' admonition")
    for line, _, k in solutions:
        options = []
        for text_line in lines[line:line + 6]:
            if not text_line.strip().startswith(":"):
                break
            options.append(text_line.strip())
        if not any(re.fullmatch(r":class:\s+(?:\S+\s+)*dropdown(?:\s+\S+)*", o) for o in options):
            failures.append(f"{name}:L{line}: Solution {k} lacks ':class: dropdown'")
    if len(exercises) != len(solutions):
        failures.append(f"{name}:L{first}: {len(exercises)} exercises but {len(solutions)} solutions")
    return sorted(failures, key=lambda f: int(re.search(r":L(\d+):", f).group(1)))


def structure_failures(name, text):
    total = len(text.splitlines())
    top = headings(text)
    failures = []
    for i, (line, title) in enumerate(top):
        end = top[i + 1][0] if i + 1 < len(top) else total + 1
        share = (end - line) / total
        if share > MAX_SHARE:
            failures.append(f"{name}:L{line}: section '{title}' is {share:.0%} of the page's lines (limit {MAX_SHARE:.0%})")
    return failures


def page_failures(chapter_dir):
    failures = []
    for page in pages(chapter_dir):
        if page.name != DEEP_DIVE:
            text = page.read_text(encoding="utf-8")
            failures += order_failures(page.name, text) + exercise_failures(page.name, text)
    return failures


def size_failures(chapter_dir):
    failures = []
    for page in pages(chapter_dir):
        failures += structure_failures(page.name, page.read_text(encoding="utf-8"))
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_page_order_and_exercise_solutions(chapter_dir):
    chapter_check("exercises")(chapter_dir, page_failures(chapter_dir))


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_no_section_exceeds_35_percent_of_the_page(chapter_dir):
    chapter_check("structure")(chapter_dir, size_failures(chapter_dir))


# ---- the checking functions on small pages ---------------------------------------------------------------------

def solution(n, extra=":class: dropdown"):
    return f"```{{admonition}} Solution {n}\n{extra}\n\nThe answer.\n```\n"


def page(sections=None, exercises=None, filler=6):
    """A page that passes both checks; `sections` replaces the heading list, `exercises` the Exercises body."""
    sections = sections or ["Chapter Overview", "Learning Objectives", "Hands-On Learning", "1. First", "2. Second",
                            "3. Third", "Summary", "Exercises", "Further Reading"]
    exercises = exercises if exercises is not None else (
        "1. First exercise.\n\n" + solution(1) + "\n2. Second exercise.\n\n" + solution(2))
    parts = ["# Title\n"]
    for title in sections:
        parts.append(f"## {title}\n")
        parts.append(exercises if title == "Exercises" else "\n".join(f"Line {i}." for i in range(filler)) + "\n")
    return "\n".join(parts)


def test_a_complete_page_passes():
    text = page()
    assert order_failures("p.md", text) == [] and exercise_failures("p.md", text) == []
    assert structure_failures("p.md", text) == []


def test_a_missing_solution_is_reported():
    text = page(exercises="1. First.\n\n" + solution(1) + "\n2. Second.\n\n3. Third.\n\n" + solution(3))
    failures = exercise_failures("p.md", text)
    assert len(failures) == 2 and failures[0].endswith("3 exercises but 2 solutions")
    assert "exercise 2 is not followed by a 'Solution 2' admonition" in failures[1]


def test_a_page_with_no_exercises_fails():
    failures = exercise_failures("p.md", page(exercises="Think about it.\n"))
    assert len(failures) == 1 and "has no exercises" in failures[0]


def heading_exercises(*solved):
    return "\n".join(f"### Exercise {n}\n\nDo the work.\n\n" + (solution(n) if ok else "") for n, ok in enumerate(solved, 1))


@pytest.mark.parametrize("marks", ["###", "####"])
def test_exercise_headings_with_solutions_pass(marks):
    text = page(exercises=heading_exercises(True, True).replace("###", marks))
    assert exercise_failures("p.md", text) == [] and order_failures("p.md", text) == []


def test_an_exercise_heading_without_its_solution_fails():
    failures = exercise_failures("p.md", page(exercises=heading_exercises(True, False, True)))
    assert len(failures) == 2 and failures[0].endswith("3 exercises but 2 solutions")
    assert "exercise 2 is not followed by a 'Solution 2' admonition" in failures[1]


def test_mixed_exercise_forms_fail_with_a_clear_message():
    text = page(exercises="1. A list item.\n\n" + solution(1) + "\n### Exercise 2\n\nText.\n\n" + solution(2))
    failures = exercise_failures("p.md", text)
    assert len(failures) == 1 and "mixes numbered list items and 'Exercise N' headings" in failures[0]


def test_headings_are_matched_whole_after_a_section_number():
    sections = ["Chapter Overview", "Learning Objectives", "Hands-On Learning", "1. First", "10. Summary",
                "Exercises", "Further Reading"]
    assert order_failures("p.md", page(sections=sections)) == []
    for old, wrong in (("10. Summary", "Summary of results"), ("10. Summary", "Executive Summary"),
                       ("Exercises", "More Exercises")):
        replaced = [wrong if s == old else s for s in sections]
        assert order_failures("p.md", page(sections=replaced))[0].startswith("p.md:L1: no '##' heading titled")


def test_a_solution_without_dropdown_is_reported():
    text = page(exercises="1. First.\n\n" + solution(1, extra=":class: note"))
    failures = exercise_failures("p.md", text)
    assert len(failures) == 1 and failures[0].endswith("Solution 1 lacks ':class: dropdown'")


def test_numbered_lists_inside_a_solution_are_not_exercises():
    inner = "```{admonition} Solution 1\n:class: dropdown\n\n1. step one\n2. step two\n```\n"
    assert exercise_failures("p.md", page(exercises="1. First.\n\n" + inner)) == []


def test_summary_after_exercises_is_reported():
    sections = ["Overview", "Learning Objectives", "Hands-On", "1. A", "2. B", "Exercises", "Summary", "Further Reading"]
    failures = order_failures("p.md", page(sections=sections))
    assert len(failures) == 1 and failures[0].startswith("p.md:L") and "out of order" in failures[0]


@pytest.mark.parametrize("missing", ["Overview", "Learning Objectives", "Hands-On", "Summary", "Exercises",
                                     "Further Reading", "1. A"])
def test_a_missing_part_is_reported(missing):
    sections = [s for s in ["Overview", "Learning Objectives", "Hands-On", "1. A", "Summary", "Exercises",
                            "Further Reading"] if s != missing]
    failures = order_failures("p.md", page(sections=sections))
    assert len(failures) == 1 and ("no '##' heading" in failures[0] or "no numbered" in failures[0])


def test_numbered_sections_after_summary_are_reported():
    sections = ["Overview", "Learning Objectives", "Hands-On", "1. A", "Summary", "2. B", "Exercises", "Further Reading"]
    assert len(order_failures("p.md", page(sections=sections))) == 1


def test_a_section_over_35_percent_is_reported():
    text = page(filler=2).replace("## 2. Second\n", "## 2. Second\n" + "Long.\n" * 40)
    failures = structure_failures("p.md", text)
    assert len(failures) == 1 and "section '2. Second' is" in failures[0] and "(limit 35%)" in failures[0]
    assert order_failures("p.md", text) == []


def test_headings_in_fences_do_not_count():
    text = page() + "\n```python\n## Summary\n```\n"
    assert structure_failures("p.md", text) == [] and order_failures("p.md", text) == []


FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"


def fixture_copy(tmp_path, old=None, new=""):
    chapter = tmp_path / "chapter"
    shutil.copytree(FIXTURE, chapter)
    if old is not None:
        text = (chapter / "index.md").read_text(encoding="utf-8")
        assert old in text
        (chapter / "index.md").write_text(text.replace(old, new, 1), encoding="utf-8")
    return chapter


def test_the_sample_chapter_passes_both_checks():
    assert page_failures(FIXTURE) == [] and size_failures(FIXTURE) == []


def test_the_sample_chapter_without_its_solution_fails(tmp_path):
    solution_block = "```{admonition} Solution 1\n:class: dropdown\n\nThe sum is ten and there are four values, so the mean is 2.5.\n```\n"
    failures = page_failures(fixture_copy(tmp_path, solution_block))
    assert len(failures) == 2 and "1 exercises but 0 solutions" in failures[0]
    assert "exercise 1 is not followed by a 'Solution 1'" in failures[1]


def test_the_sample_chapter_with_summary_after_exercises_fails(tmp_path):
    text = (FIXTURE / "index.md").read_text(encoding="utf-8")
    summary = text[text.index("## Summary"):text.index("## Exercises")]
    chapter = fixture_copy(tmp_path, summary)
    (chapter / "index.md").write_text((chapter / "index.md").read_text(encoding="utf-8").replace(
        "## Further Reading", summary + "## Further Reading"), encoding="utf-8")
    failures = page_failures(chapter)
    assert len(failures) == 1 and "out of order" in failures[0]


def test_the_sample_chapter_with_a_section_over_35_percent_fails(tmp_path):
    chapter = fixture_copy(tmp_path, "## Learning Objectives\n", "## Learning Objectives\n" + "Padding line.\n" * 150)
    failures = size_failures(chapter)
    assert len(failures) == 1 and "section 'Learning Objectives' is" in failures[0]


def test_the_deep_dive_is_exempt_from_order_and_exercises_only(tmp_path):
    (tmp_path / "tokenization-deep-dive.md").write_text("# Deep dive\n\n## Part\n" + "text\n" * 5, encoding="utf-8")
    assert page_failures(tmp_path) == []
    assert len(size_failures(tmp_path)) == 1
    (tmp_path / "index.md").write_text("# Chapter\n\n## Part\n" + "text\n" * 5, encoding="utf-8")
    assert page_failures(tmp_path) != []
