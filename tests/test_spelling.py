"""American spelling throughout. Check name `spelling`, one test per chapter.

Scanned: chapter prose (every *.md in the chapter directory), notebook markdown cells, string literals in notebook
code cells and the committed text outputs of those cells, and (chapter 6) the string literals of
tools/build_architecture_notebook.py.
Not scanned: code fences, inline code, math, block quotations, quoted text immediately followed by a citation,
URLs and link targets, MyST labels and role targets ((label)=, {ref}`..`, {cite}`..`), directive option lines such
as :name:, and the bibliography files.
Words are matched whole and case-insensitively against tests/spelling_uk_us.txt (`british<TAB>american`, built from
VarCon by tests/make_spelling_list.py) plus the phrase "per cent". No suffix patterns.
Exceptions: tests/spelling_allow.txt (proper names and titles) and chapters/<dir>/spelling-allow.txt, one phrase per
line, `#` comments; an allowed phrase is matched case-insensitively and hides itself from the scan.
"""
import io
import re
import tokenize
from functools import lru_cache
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._notebooks import cell_text, notebooks, read
from tests._pages import blank_code, blank_inline_code, blank_math, line_of, pages

WORDLIST = ROOT / "tests" / "spelling_uk_us.txt"
ALLOW = ROOT / "tests" / "spelling_allow.txt"
GENERATOR = ROOT / "tools" / "build_architecture_notebook.py"
WORD = re.compile(r"\b[A-Za-z]+\b")
PER_CENT = re.compile(r"\bper\s+cent\b", re.I)
QUOTED_CITED = re.compile(r"\"[^\"]{1,600}\"|“[^”]{1,600}”")
CITED_AFTER = re.compile(r"\s*[,.;]?\s*\{cite(?::\w+)?\}`")


@lru_cache(maxsize=None)
def load_pairs(path=WORDLIST):
    pairs = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            british, american = line.split("\t")
            pairs[british] = american
    return pairs


def read_phrases(path):
    path = Path(path)
    if not path.exists():
        return []
    lines = (l.strip() for l in path.read_text(encoding="utf-8").splitlines())
    return [l for l in lines if l and not l.startswith("#")]


def allowed_phrases(chapter_dir, global_file=ALLOW):
    return read_phrases(global_file) + read_phrases(Path(chapter_dir) / "spelling-allow.txt")


def _blank(match):
    return re.sub(r"[^\n]", " ", match.group(0))


def blank_phrases(text, phrases):
    for phrase in sorted(phrases, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(phrase)}(?![\w-])", _blank, text, flags=re.I)
    return text


def mask_markdown(text):
    """Markdown text with everything the check ignores replaced by spaces (line numbers kept)."""
    text = blank_code(text)
    for m in list(QUOTED_CITED.finditer(text)):      # quoted text immediately followed by a citation
        if CITED_AFTER.match(text, m.end()):
            text = text[:m.start()] + _blank(m) + text[m.end():]
    text = re.sub(r"^[ \t]*>.*$", _blank, text, flags=re.M)                    # block quotations
    text = re.sub(r"\{[\w:-]+\}`[^`\n]*`", _blank, text)                       # role targets and roles
    text = re.sub(r"^\(\S+\)=[ \t]*$", _blank, text, flags=re.M)               # MyST labels
    text = re.sub(r"^[ \t]*:(?:name|label|target|class|width|height|align|scale|figclass):.*$", _blank, text,
                  flags=re.M)                                                  # directive options
    text = re.sub(r"(\{(?:figure|image)\}[ \t]+)\S+", lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))),
                  text)                                                        # figure targets
    text = blank_inline_code(blank_math(text))
    text = re.sub(r"\]\([^)\n]*\)", lambda m: "]" + " " * (len(m.group(0)) - 1), text)   # link targets
    text = re.sub(r"<?https?://[^\s)>\]]+>?", _blank, text)                    # URLs
    return re.sub(r"^\[[^\]\n]+\]:[ \t]*\S+.*$", _blank, text, flags=re.M)    # reference link definitions


def words_in(text, where, pairs, phrases, first_line=1):
    """Failure strings for the British spellings in `text` (already masked of what is ignored)."""
    text = blank_phrases(text, phrases)
    hits = []
    for m in WORD.finditer(text):
        american = pairs.get(m.group(0).lower())
        if american:
            hits.append((m.start(), m.group(0), american))
    hits += [(m.start(), m.group(0), "percent") for m in PER_CENT.finditer(text)]
    return [f"{where}:L{first_line + line_of(text, pos) - 1}: {word!r} -> {american!r}"
            for pos, word, american in sorted(hits)]


def string_literals(source):
    """[(first line, text)] of the string literals of Python source; quoted-text fallback if it does not tokenize."""
    found = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                found.append((tok.start[0], re.sub(r"\\[nrt]", "  ", tok.string)))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        found = [(line_of(source, m.start()), m.group(0))
                 for m in re.finditer(r"\"[^\"\n]*\"|'[^'\n]*'", source)]
    return found


def code_failures(source, where, pairs, phrases, first_line=1):
    failures = []
    for line, text in string_literals(source):
        failures += words_in(text, where, pairs, phrases, first_line=first_line + line - 1)
    return failures


def spelling_failures(chapter_dir, pairs=None, phrases=None, generator=GENERATOR):
    chapter_dir = Path(chapter_dir)
    pairs = load_pairs() if pairs is None else pairs
    phrases = allowed_phrases(chapter_dir) if phrases is None else phrases
    failures = []
    for page in pages(chapter_dir):
        failures += words_in(mask_markdown(page.read_text(encoding="utf-8")), page.name, pairs, phrases)
    for path in notebooks(chapter_dir):
        for index, cell in enumerate(read(path).cells):
            where = f"{path.name}:cell {index}"
            if cell.cell_type == "markdown":
                failures += words_in(mask_markdown(cell.source), where, pairs, phrases)
            elif cell.cell_type == "code":
                failures += code_failures(cell.source, where, pairs, phrases)
                failures += words_in(cell_text(cell), f"{where} output", pairs, phrases)
    if chapter_dir.name.startswith("06") and Path(generator).exists():
        failures += code_failures(Path(generator).read_text(encoding="utf-8"), "tools/build_architecture_notebook.py",
                                  pairs, phrases)
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_american_spelling(chapter_dir):
    chapter_check("spelling")(chapter_dir, spelling_failures(chapter_dir))


# ---- the checking function on small cases -------------------------------------------------------------------

PAIRS = {"normalisation": "normalization", "colour": "color", "centre": "center", "behaviour": "behavior"}


def failures_of_page(tmp_path, text, phrases=(), pairs=PAIRS):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "index.md").write_text(text, encoding="utf-8")
    return spelling_failures(tmp_path, pairs=pairs, phrases=list(phrases), generator="/nonexistent")


def test_the_word_list_is_a_us_uk_mapping_with_a_licence_header():
    pairs = load_pairs()
    assert len(pairs) > 10000
    assert pairs["normalisation"] == "normalization" and pairs["behaviour"] == "behavior"
    for correct in ("smaller", "dialogue", "grey", "analyze", "color"):
        assert correct not in pairs
    head = WORDLIST.read_text(encoding="utf-8")[:600]
    assert "VarCon" in head and "version 2020.12.07" in head


def test_a_british_word_is_reported_with_its_line(tmp_path):
    text = "# Title\n\nThe Colour of normalisation, and per cent too.\n"
    assert failures_of_page(tmp_path, text) == [
        "index.md:L3: 'Colour' -> 'color'", "index.md:L3: 'normalisation' -> 'normalization'",
        "index.md:L3: 'per cent' -> 'percent'"]


def test_an_allow_listed_proper_name_passes(tmp_path):
    text = "The Centre for Data Ethics published it, but the centre of the plot is not that.\n"
    assert failures_of_page(tmp_path, text) == ["index.md:L1: 'Centre' -> 'center'", "index.md:L1: 'centre' -> 'center'"]
    assert failures_of_page(tmp_path / "b", text, phrases=["Centre for Data Ethics"]) == [
        "index.md:L1: 'centre' -> 'center'"]


def test_a_chapter_allow_file_is_read(tmp_path):
    (tmp_path / "spelling-allow.txt").write_text("# names\nColour Study\n", encoding="utf-8")
    assert allowed_phrases(tmp_path, global_file=tmp_path / "none.txt") == ["Colour Study"]


def test_code_math_quotes_urls_labels_and_roles_are_exempt(tmp_path):
    text = ("`colour` and $colour$ and https://example.org/colour/centre and [x](https://e.org/behaviour) and\n"
            "{ref}`sec-colour` and {cite}`colour2020`\n\n(sec-behaviour)=\n\n"
            "```python\ncolour = 1\n```\n\n> A quotation about colour.\n\n"
            "He wrote \"the colour of it\" {cite}`smith2020x` and “normalisation”, {cite}`smith2020x`.\n\n"
            "```{figure} figures/colour.svg\n:name: fig-ch1-colour\n:alt: A plain alt text\n\nA caption.\n```\n")
    assert failures_of_page(tmp_path, text) == []
    assert failures_of_page(tmp_path / "b", "He wrote \"the colour of it\" without a citation.\n") == [
        "index.md:L1: 'colour' -> 'color'"]


def test_directive_content_is_prose(tmp_path):
    text = "```{admonition} A note\nThe behaviour is odd.\n\n```python\nx = 'colour'\n```\n```\n"
    assert failures_of_page(tmp_path, text) == ["index.md:L2: 'behaviour' -> 'behavior'"]


def test_notebook_markdown_string_literals_and_outputs_are_scanned(tmp_path):
    import nbformat
    cells = [nbformat.v4.new_markdown_cell("# T\n\nA colour here and `colour`."),
             nbformat.v4.new_code_cell("colour = 1  # colour comment\nprint(f'the {colour} centre\\nbehaviour')\n"),
             nbformat.v4.new_code_cell("print('x')")]
    cells[1]["outputs"] = [nbformat.v4.new_output("stream", name="stdout", text="the 1 centre\n")]
    (tmp_path / "examples").mkdir()
    nbformat.write(nbformat.v4.new_notebook(cells=cells), str(tmp_path / "examples" / "n.ipynb"))
    assert spelling_failures(tmp_path, pairs=PAIRS, phrases=[], generator="/nonexistent") == [
        "n.ipynb:cell 0:L3: 'colour' -> 'color'",
        "n.ipynb:cell 1:L2: 'centre' -> 'center'",
        "n.ipynb:cell 1:L2: 'behaviour' -> 'behavior'",
        "n.ipynb:cell 1 output:L1: 'centre' -> 'center'"]


def test_generator_string_literals_are_scanned_for_chapter_6(tmp_path):
    chapter = tmp_path / "06-x"
    chapter.mkdir()
    (chapter / "index.md").write_text("Fine.\n", encoding="utf-8")
    generator = tmp_path / "gen.py"
    generator.write_text("x = 1  # colour\nCELL = '''a\nnormalisation'''\n", encoding="utf-8")
    assert spelling_failures(chapter, pairs=PAIRS, phrases=[], generator=generator) == [
        "tools/build_architecture_notebook.py:L3: 'normalisation' -> 'normalization'"]
