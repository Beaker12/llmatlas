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


def strip_nouns(text, nouns=()):
    for name in sorted(nouns, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])(?!\.\d)", " ", text)
    return text


def strip_exempt(text, nouns=()):
    """The markdown text with math, code, headings, link targets, labels and proper nouns removed."""
    text = re.sub(r"(`{3,}|~{3,})[\s\S]*?\1", " ", text)                  # code blocks
    text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text)                          # display math
    text = re.sub(r"(?<![\\\w])\$[^$\n]+\$", " ", text)                    # inline math
    text = re.sub(r"``[^\n]*?``|`[^`\n]*`", " ", text)                     # inline code (and roles' targets)
    text = re.sub(r"^[ ]{0,3}#{1,6}[ \t].*$", " ", text, flags=re.M)       # heading lines
    text = re.sub(r"^[^\n]+\n[ ]{0,3}(?:=+|-+)[ \t]*$", " ", text, flags=re.M)  # setext headings
    text = re.sub(r"^\(\S+\)=[ \t]*$", " ", text, flags=re.M)              # MyST labels
    text = re.sub(r"\]\([^)\n]*\)", "]", text)                             # link targets
    text = re.sub(r"^\[[^\]\n]+\]:[ \t]*\S+.*$", " ", text, flags=re.M)    # reference link definitions
    text = re.sub(r"https?://[^\s)>]+", " ", text)                         # bare URLs
    return strip_nouns(text, nouns)


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


INLINE_CODE = r"``(?:[^`\n]|\n(?!\n))*?``|`[^`\n]*(?:\n[^`\n]*)?`"   # a span may wrap across one line break
FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,}|:{3,})[ \t]*(.*?)[ \t]*$")
# MyST directives whose content is code or markup, not prose; every other directive keeps its text
CODE_DIRECTIVES = {"code-cell", "code-block", "sourcecode", "code", "math", "literalinclude", "raw", "eval-rst",
                   "mermaid", "graphviz"}
LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
NOT_A_RESULT = [re.compile(p, re.I) for p in (
    r"\bChapters?\s+\d+(?:\s*(?:,|and|&|-|\u2013)\s*\d+)*",
    r"\bSections?\s+\d+(?:\.\d+)*",
    r"\bParts?\s+\d+", r"\bExercises?\s+\d+", r"\bFigures?\s+\d+", r"\bTables?\s+\d+", r"\bSteps?\s+\d+",
)]
# A four-digit number 1900-2099 is a year (not a result) only in these positions. "of" is not in the list of
# preceding words: "a context of 2048" states a number.
YEAR = r"(?:19|20)\d{2}"
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
YEAR_CONTEXTS = [re.compile(p, re.I) for p in (
    rf"\b(?:in|since|by|from|until|before|after|during|early|late|{MONTHS})\.?\s+{YEAR}(?!\d)",
    rf"\bmid-{YEAR}(?!\d)",
    rf"[(\[]\s*{YEAR}[a-z]?\s*[)\]]",
    rf"(?<!\d){YEAR}[a-z]?(?=[,\s]*\{{cite)",
    rf"\bet al\.,?\s+{YEAR}(?!\d)",
)]
# A whitespace-free chunk with a letter/digit junction (fp16, 13-gram, top-2, o200k, fp16/bf16) is a name.
NAME_CHUNK = re.compile(r"\S*(?:[A-Za-z]\d|\d(?![eE][-+]?\d)[A-Za-z]|[A-Za-z]-\d|\d-[A-Za-z])\S*")
ABBREVIATION = re.compile(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bcf|\bal|\bFig|\bEq|\bSec|\bapprox|\bNo|(?<![A-Za-z])[A-Z])\.$")


def blank_code(text):
    """The text with every fenced code block (any info string, none, or a code directive) blanked.

    Line count is kept. The fence lines of prose directives ({admonition}, {note}, ...) are blanked but their
    content stays; a ::: fence is always a prose directive.
    """
    out, stack = [], []   # stack of (marker char, length, is_code)
    for line in text.split("\n"):
        m = FENCE_LINE.match(line)
        if m:
            marker, info = m.groups()
            char, length = marker[0], len(marker)
            if stack and stack[-1][0] == char and length >= stack[-1][1] and not info:
                stack.pop()
                out.append("")
                continue
            if not (stack and stack[-1][2]):
                directive = re.match(r"\{([\w:-]+)\}", info)
                is_code = ((directive is None and char != ":")
                           or (directive is not None and directive.group(1) in CODE_DIRECTIVES))
                stack.append((char, length, is_code))
                out.append("")
                continue
        out.append("" if stack and stack[-1][2] else line)
    return "\n".join(out)


def prose_units(text):
    """(first line, text) for each unit: a list item (with its wrapped lines) or a paragraph.

    A blank line, a list-item marker or a heading starts a new unit; a wrapped line continues the current one.
    """
    units, current = [], None
    for n, line in enumerate(blank_code(text).split("\n"), start=1):
        if not line.strip() or re.match(r"^[ \t]*:[\w-]+:|^[ \t]*%", line):
            current = None
        elif re.match(r"^[ ]{0,3}#{1,6}[ \t]", line):
            units.append((n, line))
            current = None
        elif LIST_ITEM.match(line):
            current = [n, LIST_ITEM.sub("", line)]
            units.append(current)
        elif current is None:
            current = [n, line.strip()]
            units.append(current)
        else:
            current[1] += "\n" + line.strip()
    return [(n, t) for n, t in units]


def split_sentences(unit):
    """(offset, sentence) pairs: split after . ! ? and white space, but not after an abbreviation."""
    cuts = [0]
    for m in re.finditer(r"(?<=[.!?])\s+", unit):
        if not ABBREVIATION.search(unit[: m.start()]):
            cuts.append(m.end())
    cuts.append(len(unit))
    return [(a, unit[a:b]) for a, b in zip(cuts, cuts[1:])]


def prose_sentences(text):
    """(line, sentence) for every sentence outside code: admonition bodies, list items and paragraphs.

    The detector is deliberately wider than the admonition directives: chapter pages link notebook labels in
    the Hands-On list and in body paragraphs as well, and those sentences may state results too. Each list
    item is its own unit; sentences end at . ! ? followed by white space.
    """
    for start, unit in prose_units(text):
        for a, piece in split_sentences(unit):
            sentence = piece.strip()
            if sentence:
                yield start + unit.count("\n", 0, a), sentence


def states_a_number(sentence, nouns=()):
    """Whether the sentence states a number: a digit or number word outside roles, inline code, link targets,
    proper nouns, years in citation-like positions, `Chapter 2`-style references and names such as `fp16`,
    `13-gram` or `top-2` (names exempt digits only; number words always count)."""
    rest = sentence
    for pattern in YEAR_CONTEXTS:       # before roles are removed: a year may be followed by {cite}
        rest = pattern.sub(" ", rest)
    rest = ROLE.sub(" ", rest)
    rest = re.sub(INLINE_CODE, " ", rest)
    rest = re.sub(r"\]\([^)\n]*\)", "]", rest)
    rest = strip_nouns(rest, nouns)
    for pattern in NOT_A_RESULT:
        rest = pattern.sub(" ", rest)
    return bool(DIGIT.search(NAME_CHUNK.sub(" ", rest)) or WORD.search(rest))


def check_admonitions(chapter_dir, claims, nouns=(), declared=None):
    """Failures for sentences that link a notebook result with a number but match no claim.

    Every sentence of the pages is examined (see prose_sentences), so the check cannot miss a link because of
    the kind of block it sits in. Proper nouns such as FineWeb-2 are not counted as numbers. `declared` maps a
    regex over a sentence to a reason (the chapter's NOT_A_RESULT): matching sentences are not flagged, and an
    entry that matches no sentence of the chapter is a failure.
    """
    declared = dict(declared or {})
    used = set()
    matchers = []
    for chapter_re, _, _ in claims:
        try:
            matchers.append(re.compile(chapter_re))
        except re.error:
            pass
    failures = []
    for page in sorted(Path(chapter_dir).glob("*.md")):
        for line, sentence in prose_sentences(page.read_text(encoding="utf-8")):
            used.update(rx for rx in declared if re.search(rx, sentence))
            if not NB_REF.search(sentence):
                continue
            hit = [rx for rx in declared if re.search(rx, sentence)]
            used.update(hit)
            if hit or not states_a_number(sentence, nouns):
                continue
            if not any(rx.search(sentence) for rx in matchers):
                failures.append(f"{page.name}:L{line}: states a notebook result with no RESULT_CLAIMS entry: "
                                f"{sentence!r}")
    failures += [f"NOT_A_RESULT entry matches nothing: {rx}" for rx in declared if rx not in used]
    return failures


def check_chapter(chapter_dir, claims=None, global_file=GLOBAL_NOUNS, declared=None):
    if claims is None:
        claims = chapter_constants(chapter_dir).RESULT_CLAIMS
    nouns = proper_nouns(chapter_dir, global_file)
    failures = []
    for path in notebooks(chapter_dir):
        failures += check_prose_cells(path, nouns)
    declared = chapter_constants(chapter_dir).NOT_A_RESULT if declared is None else declared
    return failures + check_claims(chapter_dir, claims) + check_admonitions(chapter_dir, claims, nouns, declared)


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
    return check_chapter(chapter, CLAIMS, global_file=GLOBAL_NOUNS, declared={})


def test_sample_chapter_passes():
    assert check_chapter(FIXTURE, CLAIMS, declared={}) == []


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


def test_every_notebook_link_in_the_chapter_pages_is_in_a_sentence_the_check_examines():
    missed = []
    for chapter_dir in CHAPTER_DIRS:
        for page in sorted(chapter_dir.glob("*.md")):
            text = page.read_text(encoding="utf-8")
            outside_code = re.sub(r"^(`{3,}|~{3,})[A-Za-z][^\n]*\n[\s\S]*?\n\1[ \t]*$", "", text, flags=re.M)
            expected = len(NB_REF.findall(outside_code))
            seen = sum(len(NB_REF.findall(sentence)) for _, sentence in prose_sentences(text))
            if seen != expected:
                missed.append(f"{chapter_dir.name}/{page.name}: {expected} links outside code, {seen} in examined sentences")
    assert not missed, "\n".join(missed)


def test_sentences_in_lists_paragraphs_and_admonitions_are_all_examined():
    text = ("Intro paragraph with {ref}`nb-a` and 3 things.\n\n- A list item ({ref}`nb-b`) with two parts.\n\n"
            "```{admonition} T\n:class: tip\nBody sentence {ref}`nb-c` shows 4.\n```\n\n```python\nprint('{ref}`nb-d`')\n```\n")
    found = [sentence for _, sentence in prose_sentences(text)]
    assert [bool(NB_REF.search(x)) for x in found] == [True, True, True]
    assert [line for line, _ in prose_sentences(text)] == [1, 3, 7]


def sentence_failures(tmp_path, text, claims=(), nouns=(), declared=None):
    (tmp_path / "index.md").write_text(text, encoding="utf-8")
    return check_admonitions(tmp_path, list(claims), nouns, declared)


def test_unpunctuated_bullets_are_separate_units(tmp_path):
    text = "- Reads a snapshot ({ref}`nb-a`)\n- Counts things\n- Runs 12 steps\n"
    assert sentence_failures(tmp_path, text) == []
    assert [s for _, s in prose_sentences(text)] == ["Reads a snapshot ({ref}`nb-a`)", "Counts things", "Runs 12 steps"]
    assert [l for l, _ in prose_sentences(text)] == [1, 2, 3]


def test_a_bullet_with_a_link_and_a_number_is_flagged_at_its_own_line(tmp_path):
    failures = sentence_failures(tmp_path, "- First\n- Runs 12 steps ({ref}`nb-a`)\n- Third\n")
    assert len(failures) == 1 and failures[0].startswith("index.md:L2: ")


def test_a_claim_regex_cannot_match_across_bullets(tmp_path):
    text = "- Mean ({ref}`nb-a`) of\n- 2.5 items\n"
    assert sentence_failures(tmp_path, text, [(r"of\s+(2\.5)", r"x (\d)", "eq")]) == []


def test_wrapped_nested_and_numbered_items(tmp_path):
    wrapped = "- A long item that wraps ({ref}`nb-a`)\n  and then reaches 12 steps.\n- Next\n"
    failures = sentence_failures(tmp_path, wrapped)
    assert len(failures) == 1 and failures[0].startswith("index.md:L1: ")
    nested = "- Outer\n  - Inner links {ref}`nb-a`\n  - Inner has 7\n"
    assert sentence_failures(tmp_path, nested) == []
    nested_hit = "- Outer\n  - Inner links {ref}`nb-a` and has 7\n"
    assert sentence_failures(tmp_path, nested_hit)[0].startswith("index.md:L2: ")
    numbered = "1. Links {ref}`nb-a`\n2. Has 7\n10) Third {ref}`nb-b`\n"
    assert sentence_failures(tmp_path, numbered) == []
    assert sentence_failures(tmp_path, "3. Links {ref}`nb-a` and has 7\n")[0].startswith("index.md:L1: ")


def test_a_blank_line_ends_a_unit_and_a_wrapped_line_continues_a_sentence():
    text = "Para one links {ref}`nb-a`\ncontinues here\n\nPara two has 5.\n"
    assert [s for _, s in prose_sentences(text)] == ["Para one links {ref}`nb-a`\ncontinues here", "Para two has 5."]


def test_numbers_in_inline_code_and_any_fence_are_not_counted(tmp_path):
    assert sentence_failures(tmp_path, "The notebook {ref}`nb-a` sets `n = 12` and ``k = 3``.\n") == []
    assert sentence_failures(tmp_path, "Links {ref}`nb-a`.\n\n```\nx = 12\n```\n") == []
    assert sentence_failures(tmp_path, "```{code-cell} ipython3\nx = 12  # {ref}`nb-a` 5\n```\n") == []
    assert sentence_failures(tmp_path, "~~~text\nsee {ref}`nb-a` 12\n~~~\n") == []
    assert len(sentence_failures(tmp_path, "```{note}\nThe notebook {ref}`nb-a` reaches 12.\n```\n")) == 1


def test_chapter_and_section_numbers_and_years_are_not_results(tmp_path):
    for text in ["The Chapter 2 notebook ({ref}`nb-a`) shows it.", "Chapters 2 and 3 use {ref}`nb-a`.",
                 "See Section 3.1 and {ref}`nb-a`.", "Part 2 uses {ref}`nb-a`.", "Exercise 4 uses {ref}`nb-a`.",
                 "Figure 5 comes from {ref}`nb-a`.", "Table 6 comes from {ref}`nb-a`.",
                 "Step 3 of {ref}`nb-a` repeats.", "Introduced in 2017 ({ref}`nb-a`).", "Since 1999 ({ref}`nb-a`)."]:
        assert sentence_failures(tmp_path, text + "\n") == [], text
    for text in ["Chapter 2's notebook ({ref}`nb-a`) reaches a loss of 0.5.", "See 12345 in {ref}`nb-a`.",
                 "Section 3.1 and {ref}`nb-a` give 7.", "In 2017 it reached 3 ({ref}`nb-a`)."]:
        assert len(sentence_failures(tmp_path, text + "\n")) == 1, text


def test_a_list_item_that_links_a_notebook_and_states_a_number_needs_a_claim(tmp_path):
    page = tmp_path / "index.md"
    page.write_text("- Runs 12 steps ({ref}`nb-x`)\n", encoding="utf-8")
    assert len(check_admonitions(tmp_path, [])) == 1
    assert check_admonitions(tmp_path, [], nouns=[]) != []
    page.write_text("- Uses FineWeb-2 ({ref}`nb-x`)\n", encoding="utf-8")
    assert check_admonitions(tmp_path, [], nouns=["FineWeb-2"]) == []


def test_setext_headings_are_exempt_but_hyphenated_number_words_are_not(tmp_path):
    assert prose_hits("Title with 3 digits\n===\n\nplain text.") == []
    assert prose_hits("Another 2 words\n---\n\nplain text.") == []
    assert prose_hits("Two-thirds of the data.") == ["Two"]


LINK = "{ref}`nb-a`"


def flagged(tmp_path, sentence, declared=None):
    return sentence_failures(tmp_path, sentence + "\n", declared=declared)


@pytest.mark.parametrize("text", ["Since 2020 it grew ({ref}`nb-a`).", "In 2017 it was proposed ({ref}`nb-a`).",
                                  "It ({ref}`nb-a`) was proposed (2019).", "Vaswani et al., 2017 ({ref}`nb-a`).",
                                  "Vaswani et al. 2017 ({ref}`nb-a`).", "Released in June 2025 ({ref}`nb-a`).",
                                  "By 2030 nothing ({ref}`nb-a`) changed, mid-2024 aside, as in [2020a].",
                                  "Attention 2017 {cite}`vaswani2017` shows {ref}`nb-a`.",
                                  "Attention 2017, {cite}`vaswani2017` shows {ref}`nb-a`."])
def test_years_in_citation_positions_are_not_results(tmp_path, text):
    assert flagged(tmp_path, text.replace("By 2030 nothing", "By 2030 nothing")) == [], text


@pytest.mark.parametrize("text", ["It has 2048 tokens ({ref}`nb-a`).", "It runs 2000 steps ({ref}`nb-a`).",
                                  "A context of 2048 ({ref}`nb-a`).", "The loss reaches 2000 ({ref}`nb-a`)."])
def test_four_digit_numbers_outside_citation_positions_count(tmp_path, text):
    assert len(flagged(tmp_path, text)) == 1, text


@pytest.mark.parametrize("abbreviation", ["e.g.", "i.e.", "etc.", "vs.", "cf.", "al.", "Fig.", "Eq.", "Sec.",
                                          "approx.", "No.", "J."])
def test_sentences_are_not_split_after_abbreviations(abbreviation):
    sentences = [s for _, s in prose_sentences(f"Runs {LINK}, {abbreviation} 5 tokens. Next one.")]
    assert sentences == [f"Runs {LINK}, {abbreviation} 5 tokens.", "Next one."]


def test_a_number_after_an_abbreviation_is_still_in_the_sentence_with_the_link(tmp_path):
    assert len(flagged(tmp_path, f"It runs {LINK}, e.g. 5 tokens.")) == 1


@pytest.mark.parametrize("name", ["fp16", "bf16", "float32", "o200k", "GPT2", "13-gram", "top-2", "4-bit",
                                  "Llama-3", "2-D", "fp16/bf16", "8B", "fp16/bf16."])
def test_numbers_that_are_part_of_a_name_are_not_results(tmp_path, name):
    assert flagged(tmp_path, f"It uses {name} here ({LINK}).") == [], name


@pytest.mark.parametrize("bare", ["13 grams", "top 2", "-2", "2 - 1", "16/32", "1e5 and 3"])
def test_bare_numbers_next_to_units_still_count(tmp_path, bare):
    assert len(flagged(tmp_path, f"It gives {bare} here ({LINK}).")) == 1, bare


def test_number_words_inside_names_still_count(tmp_path):
    assert len(flagged(tmp_path, f"A top-two cut ({LINK}).")) == 1


def test_a_declared_sentence_is_not_flagged_and_a_stale_entry_is_reported(tmp_path):
    text = f"It parses two files ({LINK}).\n"
    assert len(sentence_failures(tmp_path, text)) == 1
    assert sentence_failures(tmp_path, text, declared={r"parses two files": "a count of inputs, not a result"}) == []
    stale = sentence_failures(tmp_path, text, declared={r"parses two files": "ok", r"nothing like this": "stale"})
    assert stale == ["NOT_A_RESULT entry matches nothing: nothing like this"]


def test_code_directives_behind_colon_fences_and_wrapped_inline_code_are_code(tmp_path):
    assert flagged(tmp_path, f":::{{code-cell}} ipython3\nx = 12  # {LINK}\n:::") == []
    assert flagged(tmp_path, f":::{{code-block}} python\nx = 12  # {LINK}\n:::") == []
    assert len(flagged(tmp_path, f":::{{note}}\nIt has 12 ({LINK}).\n:::")) == 1
    assert flagged(tmp_path, f"Sets `n =\n12` ({LINK}).") == []


def test_every_chapter_declares_a_not_a_result_dictionary():
    for chapter_dir in CHAPTER_DIRS:
        assert chapter_constants(chapter_dir).NOT_A_RESULT == {}
