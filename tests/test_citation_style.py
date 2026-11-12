"""Citation style: {cite:t} when the authors are the grammatical subject, {cite} otherwise.

Check name `citations`, one test per chapter. Failures:
- "et al." directly followed by a plain {cite};
- the surname of an author of a cited key written immediately before a plain {cite} of that key
  ("Press and Wolf {cite}`press2017using`"; the fix is {cite:t}).
Surnames come from references.bib and bib/*.bib in the repository root.
"""
import re
import unicodedata

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._pages import blank_code, blank_inline_code, line_of, pages

ET_AL = re.compile(r"et al\.,?\s+\{cite\}")
PLAIN_CITE = re.compile(r"\{cite\}`([^`\n]+)`")
BEFORE = re.compile(r"([^\W\d_][\w\-]*)(?:['\u2019]s)?\s*$")


def bib_files():
    return [ROOT / "references.bib"] + sorted((ROOT / "bib").glob("*.bib"))


def _braced(text, start):
    """The text of the {...} group whose opening brace is at `start`."""
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "{"
        depth -= text[i] == "}"
        if depth == 0:
            return text[start + 1:i]
    return text[start + 1:]


def fold(word):
    """Case- and accent-folded form, so that Unal, Ünal and (from the bib) {\\\"U}nal compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c)).casefold()


def _clean(name):
    name = re.sub(r"\\[a-zA-Z]+\s*|\\.", "", name)
    return re.sub(r"[{}]", "", name).strip()


def surnames(files):
    """{citation key: set of author surnames} from the BibTeX files that exist."""
    result = {}
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for entry in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
            body = _braced(text, text.index("{", entry.start()))
            field = re.search(r"\bauthor\s*=\s*", body)
            names = set()
            if field:
                rest = body[field.end():]
                value = _braced(rest, 0) if rest.startswith("{") else rest[1:rest.index('"', 1)]
                for author in re.split(r"\s+and\s+", value):
                    author = _clean(author)
                    surname = author.split(",")[0].strip() if "," in author else author
                    if surname:
                        names.add(surname)
                        names.add(surname.split()[-1])
            result[entry.group(1)] = names
    return result


def citation_failures(chapter_dir, files=None):
    known = surnames(bib_files() if files is None else files)
    failures = []
    for page in pages(chapter_dir):
        text = blank_inline_code(blank_code(page.read_text(encoding="utf-8")), keep_roles=True)
        for m in ET_AL.finditer(text):
            failures.append(f"{page.name}:L{line_of(text, m.start())}: 'et al.' followed by plain {{cite}}; use {{cite:t}}")
        for m in PLAIN_CITE.finditer(text):
            before = BEFORE.search(text[max(0, m.start() - 80):m.start()])
            if not before:
                continue
            word = before.group(1)
            parts = {fold(p) for p in (word, *word.split("-"))}
            for key in (k.strip() for k in m.group(1).split(",")):
                if parts & {fold(s) for s in known.get(key, set())}:
                    failures.append(f"{page.name}:L{line_of(text, m.start())}: surname {word!r} before "
                                    f"{{cite}}`{key}`; use {{cite:t}}")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_citation_style(chapter_dir):
    chapter_check("citations")(chapter_dir, citation_failures(chapter_dir))


BIB = ("@article{press2017using,\n  title={Using the output embedding},\n"
       "  author={Press, Ofir and Wolf, Lior},\n  year={2017}\n}\n"
       "@inproceedings{smith2020x,\n  author = {Smith, Jo and {\\\"U}nal, Ay\\c{s}e and Van Der Berg, Kim},\n"
       "  title = {X}\n}\n")


def chapter_with(tmp_path, text):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "index.md").write_text(text, encoding="utf-8")
    bib = tmp_path / "tiny.bib"
    bib.write_text(BIB, encoding="utf-8")
    return tmp_path, [bib]


def test_surnames_are_read_from_the_bib(tmp_path):
    _, files = chapter_with(tmp_path, "")
    assert surnames(files) == {"press2017using": {"Press", "Wolf"},
                               "smith2020x": {"Smith", "Unal", "Van Der Berg", "Berg"}}


def test_et_al_before_a_plain_cite_is_reported(tmp_path):
    chapter, files = chapter_with(tmp_path, "Intro.\n\nSmith et al. {cite}`smith2020x` found it.\n")
    assert citation_failures(chapter, files) == ["index.md:L3: 'et al.' followed by plain {cite}; use {cite:t}"]
    chapter, files = chapter_with(tmp_path / "b", "Smith et al., {cite}`smith2020x` found it.\n")
    assert len(citation_failures(chapter, files)) == 1


def test_accents_are_folded_on_both_sides(tmp_path):
    chapter, files = chapter_with(tmp_path, "The work of Ünal {cite}`smith2020x` and Unal {cite}`smith2020x`.\n")
    assert [f.split(": ", 1)[1].split(" before")[0] for f in citation_failures(chapter, files)] == [
        "surname 'Ünal'", "surname 'Unal'"]


def test_a_line_break_between_et_al_and_cite_is_reported(tmp_path):
    chapter, files = chapter_with(tmp_path, "Smith et al.\n{cite}`smith2020x` found it; Jones et al.   {cite}`x`.\n")
    assert [f.split(": ")[0] for f in citation_failures(chapter, files)] == ["index.md:L1", "index.md:L2"]


def test_a_surname_before_a_plain_cite_is_reported(tmp_path):
    chapter, files = chapter_with(tmp_path, "Press and Wolf {cite}`press2017using` tie them.\n")
    assert citation_failures(chapter, files) == [
        "index.md:L1: surname 'Wolf' before {cite}`press2017using`; use {cite:t}"]


def test_the_fix_and_other_uses_pass(tmp_path):
    text = ("Press and Wolf {cite:t}`press2017using` tie them; ties are common {cite}`press2017using`.\n"
            "Smith {cite}`press2017using` is another author. In `Smith et al. {cite}` code and\n\n"
            "```text\nSmith et al. {cite}`smith2020x`\n```\n")
    chapter, files = chapter_with(tmp_path, text)
    assert citation_failures(chapter, files) == []
