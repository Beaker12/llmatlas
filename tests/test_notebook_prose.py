"""Notebook prose carries no hand-typed results, and the chapter's claims about results agree with the notebook.

(a) After the first code cell, a markdown cell contains no digit and none of the number words below
    ("one" is allowed) unless its metadata tags include `static-numbers`. Exempt: math, inline code and code
    blocks, heading lines, link targets and MyST labels, and the names in tests/notebook_proper_nouns.txt and
    in the chapter's own proper-nouns.txt (one name per line).
(b) Every RESULT_CLAIMS entry (chapter regex, notebook-output regex, comparator) holds. The comparator is
    applied as `chapter_value <op> notebook_value`: eq, gt (chapter > notebook), lt, or approx:<rel>
    (|chapter - notebook| <= rel * |notebook|).
(c) Every sentence in a chapter admonition that links a notebook label ({ref} to a target starting `nb-`) and
    contains a digit or a number word is matched by the chapter regex of some RESULT_CLAIMS entry.
One test per chapter covers all three, check name `prose`.
"""
import re
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, chapter_constants
from tests._notebooks import cell_text, notebooks, read

NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "twice", "half", "double", "hundred", "thousand", "million", "billion"]
WORD = re.compile(r"\b(?:%s)\b" % "|".join(NUMBER_WORDS), re.I)
DIGIT = re.compile(r"\d")
GLOBAL_NOUNS = ROOT / "tests" / "notebook_proper_nouns.txt"
ADMONITION = re.compile(r"^(`{3,}|:{3,})\{(?:admonition|note|tip|warning|important|hint|caution|attention|"
                        r"danger|error|seealso)\}[^\n]*\n([\s\S]*?)^\1[ \t]*$", re.M)
NB_REF = re.compile(r"\{ref\}`nb-[^`]*`")
ROLE = re.compile(r"\{[\w:-]+\}`[^`]*`")


def read_names(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = (l.strip() for l in path.read_text(encoding="utf-8").splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def proper_nouns(chapter_dir, global_file=GLOBAL_NOUNS):
    return read_names(global_file) + read_names(Path(chapter_dir) / "proper-nouns.txt")


def strip_exempt(text, nouns=()):
    """The markdown text with math, code, headings, link targets, labels and proper nouns removed."""
    text = re.sub(r"(`{3,}|~{3,})[\s\S]*?\1", " ", text)                  # code blocks
    text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text)                          # display math
    text = re.sub(r"(?<![\\\w])\$[^$\n]+\$", " ", text)                    # inline math
    text = re.sub(r"``[^\n]*?``|`[^`\n]*`", " ", text)                     # inline code (and roles' targets)
    text = re.sub(r"^[ ]{0,3}#{1,6}[ \t].*$", " ", text, flags=re.M)       # heading lines
    text = re.sub(r"^\(\S+\)=[ \t]*$", " ", text, flags=re.M)              # MyST labels
    text = re.sub(r"\]\([^)\n]*\)", "]", text)                             # link targets
    text = re.sub(r"^\[[^\]\n]+\]:[ \t]*\S+.*$", " ", text, flags=re.M)    # reference link definitions
    text = re.sub(r"https?://[^\s)>]+", " ", text)                         # bare URLs
    for name in sorted(nouns, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])(?!\.\d)", " ", text)
    return text


def prose_hits(text, nouns=()):
    """Digits and number words left in `text` after the exemptions."""
    text = strip_exempt(text, nouns)
    return re.findall(r"\d[\d,._]*\d|\d", text) + WORD.findall(text)


def check_prose_cells(path, nouns=()):
    """Failures for markdown cells after the first code cell of notebook `path`."""
    nb = read(path)
    failures = []
    seen_code = False
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            seen_code = True
            continue
        if cell.cell_type != "markdown" or not seen_code:
            continue
        if "static-numbers" in cell.metadata.get("tags", []):
            continue
        hits = prose_hits(cell.source, nouns)
        if hits:
            shown = ", ".join(repr(h) for h in dict.fromkeys(hits))
            failures.append(f"{Path(path).name}: cell {i}: number in prose: {shown}")
    return failures


def number(text):
    """A captured number as a float: `10,000`, `10_000`, `1e4` and a trailing `%` are accepted."""
    return float(text.strip().replace(",", "").replace("_", "").rstrip("%").strip())


def holds(op, chapter_value, notebook_value):
    if op == "eq":
        return chapter_value == notebook_value
    if op == "gt":
        return chapter_value > notebook_value
    if op == "lt":
        return chapter_value < notebook_value
    if op.startswith("approx:"):
        return abs(chapter_value - notebook_value) <= float(op.split(":", 1)[1]) * abs(notebook_value)
    raise ValueError(f"unknown comparator {op!r}")


def chapter_text(chapter_dir):
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(Path(chapter_dir).glob("*.md")))


def notebook_output_text(chapter_dir):
    return "\n".join(cell_text(c) for path in notebooks(chapter_dir)
                     for c in read(path).cells if c.cell_type == "code")


def check_claims(chapter_dir, claims):
    """Failures for RESULT_CLAIMS entries that do not hold (or cannot be evaluated)."""
    failures = []
    pages, outputs = chapter_text(chapter_dir), notebook_output_text(chapter_dir)
    for n, (chapter_re, notebook_re, op) in enumerate(claims):
        label = f"claim {n} ({chapter_re!r})"
        try:
            c_match, n_match = re.compile(chapter_re), re.compile(notebook_re)
        except re.error as e:
            failures.append(f"{label}: bad regex: {e}")
            continue
        if c_match.groups != 1 or n_match.groups != 1:
            failures.append(f"{label}: each regex needs exactly one capture group")
            continue
        found_c, found_n = c_match.search(pages), n_match.search(outputs)
        if not found_c:
            failures.append(f"{label}: chapter regex not found in the chapter pages")
        elif not found_n:
            failures.append(f"{label}: notebook regex {notebook_re!r} not found in the notebook outputs")
        else:
            try:
                c_value, n_value = number(found_c.group(1)), number(found_n.group(1))
                ok = holds(op, c_value, n_value)
            except ValueError as e:
                failures.append(f"{label}: {e}")
                continue
            if not ok:
                failures.append(f"{label}: chapter says {found_c.group(1)!r}, notebook shows "
                                f"{found_n.group(1)!r}, which fails {op}")
    return failures


def admonition_sentences(text):
    """(line, sentence) for each sentence in each admonition of the page text."""
    for m in ADMONITION.finditer(text):
        body, start = m.group(2), m.start(2)
        cuts = [0] + [b.end() for b in re.finditer(r"(?<=[.!?])\s+", body)] + [len(body)]
        for a, b in zip(cuts, cuts[1:]):
            sentence = body[a:b].strip()
            sentence = re.sub(r"^(?::[\w-]+:.*\n)+", "", sentence).strip()
            if sentence:
                yield text.count("\n", 0, start + a) + 1, sentence


def check_admonitions(chapter_dir, claims):
    """Failures for sentences that link a notebook result with a number but match no claim."""
    matchers = []
    for chapter_re, _, _ in claims:
        try:
            matchers.append(re.compile(chapter_re))
        except re.error:
            pass
    failures = []
    for page in sorted(Path(chapter_dir).glob("*.md")):
        for line, sentence in admonition_sentences(page.read_text(encoding="utf-8")):
            if not NB_REF.search(sentence):
                continue
            rest = ROLE.sub(" ", sentence)
            if not (DIGIT.search(rest) or WORD.search(rest)):
                continue
            if not any(rx.search(sentence) for rx in matchers):
                failures.append(f"{page.name}:L{line}: states a notebook result with no RESULT_CLAIMS entry: "
                                f"{sentence[:90]!r}")
    return failures


def check_chapter(chapter_dir, claims=None, global_file=GLOBAL_NOUNS):
    if claims is None:
        claims = chapter_constants(chapter_dir).RESULT_CLAIMS
    nouns = proper_nouns(chapter_dir, global_file)
    failures = []
    for path in notebooks(chapter_dir):
        failures += check_prose_cells(path, nouns)
    return failures + check_claims(chapter_dir, claims) + check_admonitions(chapter_dir, claims)


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_notebook_prose_agrees_with_outputs(chapter_dir):
    assert notebooks(chapter_dir), f"no notebook under {chapter_dir.name}/examples"
    chapter_check("prose")(chapter_dir, check_chapter(chapter_dir))


# Fixture tests: the checks on deliberately broken copies of the sample chapter.

FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
CLAIMS = [(r"mean of (\d+\.\d+)", r"mean (\d+\.\d+)", "approx:0.001")]


def sample_chapter(tmp_path, cell=None, source=None, tags=None, page=None, nouns=None):
    """A copy of the fixture chapter; optionally replace a notebook cell's source or tags, or the page text."""
    import json
    import shutil
    chapter = tmp_path / "sample"
    shutil.copytree(FIXTURE, chapter)
    nb_path = chapter / "examples" / "sample_demo.ipynb"
    if cell is not None:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        if source is not None:
            nb["cells"][cell]["source"] = source
        if tags is not None:
            nb["cells"][cell]["metadata"]["tags"] = tags
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    if page is not None:
        old, new = page
        text = (chapter / "index.md").read_text(encoding="utf-8")
        assert old in text
        (chapter / "index.md").write_text(text.replace(old, new, 1), encoding="utf-8")
    if nouns is not None:
        (chapter / "proper-nouns.txt").write_text(nouns, encoding="utf-8")
    return chapter


def failures_of(chapter):
    return check_chapter(chapter, CLAIMS, global_file=GLOBAL_NOUNS)


def test_sample_chapter_passes():
    assert check_chapter(FIXTURE, CLAIMS) == []


def test_a_digit_typed_into_a_markdown_cell_is_reported(tmp_path):
    chapter = sample_chapter(tmp_path, 2, "The mean printed above is 3 times the count.")
    assert failures_of(chapter) == ["sample_demo.ipynb: cell 2: number in prose: '3'"]


def test_a_number_word_typed_into_a_markdown_cell_is_reported(tmp_path):
    chapter = sample_chapter(tmp_path, 2, "The mean printed above is Twice the count, about half of a million.")
    assert failures_of(chapter) == ["sample_demo.ipynb: cell 2: number in prose: 'Twice', 'half', 'million'"]


def test_the_word_one_and_longer_words_are_allowed(tmp_path):
    chapter = sample_chapter(tmp_path, 2, "One value per item, a tenth of the total, and someone's doubled twin.")
    assert failures_of(chapter) == []


def test_a_static_numbers_tag_exempts_the_cell(tmp_path):
    chapter = sample_chapter(tmp_path, 2, "The paper used 3 layers and two heads.", tags=["static-numbers"])
    assert failures_of(chapter) == []


def test_text_before_the_first_code_cell_is_not_scanned(tmp_path):
    chapter = sample_chapter(tmp_path, 0, "# Title\n\nThis needs 12 things and two hands.")
    assert failures_of(chapter) == []


def test_math_code_headings_and_link_targets_are_exempt(tmp_path):
    source = ("## Section 4 with 3 words\n\nInline $x_2 = 3$ and\n\n$$y = 2x + 10$$\n\nand `n = 5` and\n\n"
              "```python\nx = 7\n```\n\nand [a page](https://example.org/a/2) and\n\n(sec-two)=\n\nplain text.")
    assert failures_of(sample_chapter(tmp_path, 2, source)) == []


def test_a_name_in_the_proper_nouns_list_is_exempt(tmp_path):
    source = "The mean printed above is like Zorb-7 and Quux 4.1 and Quux 4, not the others."
    assert len(failures_of(sample_chapter(tmp_path, 2, source))) == 1
    chapter = sample_chapter(tmp_path / "b", 2, source, nouns="# names\nZorb-7\nQuux 4\nQuux 4.1\n")
    assert failures_of(chapter) == []
    assert failures_of(sample_chapter(tmp_path / "c", 2, "It is Zorb-70 here.", nouns="Zorb-7\n")) != []


def test_a_false_result_claim_is_reported(tmp_path):
    chapter = sample_chapter(tmp_path, page=("mean of 2.5 over", "mean of 2.7 over"))
    failures = failures_of(chapter)
    assert len(failures) == 1 and failures[0].startswith("claim 0") and "'2.7'" in failures[0]
    assert failures_of(sample_chapter(tmp_path / "b", page=("mean of 2.5 over", "mean of 2.5 over"))) == []


@pytest.mark.parametrize("claim,expected_ok", [
    ((r"mean of (\d+\.\d+)", r"mean (\d+\.\d+)", "eq"), True),
    ((r"mean of (\d+\.\d+)", r"mean (\d+\.\d+)", "gt"), False),
    ((r"mean of (\d+\.\d+)", r"items (\d+)", "lt"), True),
    ((r"mean of (\d+\.\d+)", r"items (\d+)", "gt"), False),
    ((r"mean of (\d+\.\d+)", r"mean (\d+\.\d+)", "approx:0.01"), True),
    ((r"mean of (\d+\.\d+)", r"items (\d+)", "approx:0.2"), False),
    ((r"mean of (\d+\.\d+)", r"items (\d+)", "approx:0.5"), True),
])
def test_comparators(claim, expected_ok):
    assert (check_claims(FIXTURE, [claim]) == []) is expected_ok


def test_claim_problems_are_reported():
    assert "not found in the chapter" in check_claims(FIXTURE, [(r"absent (\d+)", r"mean (\d+)", "eq")])[0]
    assert "not found in the notebook" in check_claims(FIXTURE, [(r"mean of (\d+)", r"absent (\d+)", "eq")])[0]
    assert "exactly one capture group" in check_claims(FIXTURE, [(r"mean of \d+", r"mean (\d+)", "eq")])[0]


def test_numbers_are_normalised():
    assert [number(t) for t in ["10,000", "10_000", "1e4", "95%", " 2.5 "]] == [10000, 10000, 10000, 95, 2.5]


def test_an_admonition_sentence_with_a_notebook_result_and_no_claim_is_reported(tmp_path):
    chapter = sample_chapter(tmp_path, page=("reports a mean of 2.5 over", "reports a mean of 2.5 and 4 items over"))
    assert check_admonitions(chapter, CLAIMS) == []   # the same sentence still matches the claim regex
    failures = check_admonitions(chapter, [])
    assert len(failures) == 1 and "no RESULT_CLAIMS entry" in failures[0] and failures[0].startswith("index.md:L")
    worded = sample_chapter(tmp_path / "b", page=("a mean of 2.5 over", "twelve values over"))
    failures = check_admonitions(worded, CLAIMS)
    assert len(failures) == 1 and "twelve values" in failures[0] and failures[0].startswith("index.md:L")


def test_an_admonition_without_a_notebook_link_or_number_needs_no_claim():
    assert check_admonitions(FIXTURE, [(r"mean of (\d+\.\d+)", r"x", "eq")]) == []
    text = "```{admonition} T\nThe notebook {ref}`nb-x` lists every value.\n\nSee 3 items and {numref}`fig-ch1-a`.\n```\n"
    assert [s for _, s in admonition_sentences(text)] == [
        "The notebook {ref}`nb-x` lists every value.", "See 3 items and {numref}`fig-ch1-a`."]
