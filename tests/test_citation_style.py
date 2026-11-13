"""Citation style: {cite:t} when the authors are the grammatical subject, {cite} otherwise.

Check name `citations`, one test per chapter. Failures:
- "et al." directly followed by a plain {cite};
- the surname of an author of a cited key written immediately before a plain {cite} of that key
  ("Press and Wolf {cite}`press2017using`"; the fix is {cite:t});
- a hand-written "Surname et al." or "Surname and Surname" followed, within HAND_GAP characters of the same
  sentence, by any cite role ({cite}, {cite:p} or {cite:t}) of a work that author wrote ("Xiong et al. (2020)
  {cite:p}`key` show", or "Press and Wolf {cite:t}`press2017using`", which prints the names twice). The fix is
  the role alone: {cite:t}`key` show. A failure is reported once per role. Two forms that need the written
  name stay legal: a possessive ("Xiong et al.'s {cite}`key` result", which {cite:t} cannot make) and a
  quotation (a quotation mark directly before the name or between the name and the role).
Surnames come from references.bib and bib/*.bib in the repository root.
"""
import re
import unicodedata

import pytest

from tools.book_files import bib_files as book_bib_files

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._pages import blank_code, blank_inline_code, line_of, pages

ET_AL = re.compile(r"et al\.,?\s+\{cite\}")
PLAIN_CITE = re.compile(r"\{cite\}`([^`\n]+)`")
HAND_GAP = 15
HAND = re.compile(r"\b(?P<first>[A-Z][\w\-]*)(?:\s+et\s+al\.|\s+and\s+(?P<second>[A-Z][\w\-]*))(?P<poss>['’]s)?"
                  r"(?P<gap>[^.;!?{]{0,%d}?)\{cite(?::[pt])?\}`(?P<keys>[^`\n]+)`" % HAND_GAP)
QUOTES = "\"“”"
BEFORE = re.compile(r"([^\W\d_][\w\-]*)(?:['\u2019]s)?\s*$")


def bib_files():
    return book_bib_files(ROOT)


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
        reported = set()  # offsets of the cite roles already reported
        for m in ET_AL.finditer(text):
            reported.add(m.end() - len("{cite}"))
            failures.append(f"{page.name}:L{line_of(text, m.start())}: 'et al.' followed by plain {{cite}}; use {{cite:t}}")
        for m in PLAIN_CITE.finditer(text):
            before = BEFORE.search(text[max(0, m.start() - 80):m.start()])
            if not before:
                continue
            word = before.group(1)
            parts = {fold(p) for p in (word, *word.split("-"))}
            for key in (k.strip() for k in m.group(1).split(",")):
                if parts & {fold(s) for s in known.get(key, set())}:
                    reported.add(m.start())
                    failures.append(f"{page.name}:L{line_of(text, m.start())}: surname {word!r} before "
                                    f"{{cite}}`{key}`; use {{cite:t}}")
        for m in HAND.finditer(text):
            quoted = any(q in m.group("gap") or text[m.start() - 1:m.start()] == q for q in QUOTES)
            if m.group("poss") or quoted or m.end("gap") in reported:
                continue
            names = {fold(n) for n in (m.group("first"), m.group("second")) if n}
            same = [k for k in (k.strip() for k in m.group("keys").split(","))
                    if names & {fold(s) for s in known.get(k, set())}]
            if same:
                written = text[m.start():m.end("poss") if m.group("poss") else m.start("gap")]
                failures.append(f"{page.name}:L{line_of(text, m.start())}: hand-written {' '.join(written.split())!r} "
                                f"before a cite of {same[0]}; use {{cite:t}}`{same[0]}` alone")
    return sorted(failures, key=lambda f: int(f.split(":L")[1].split(":")[0]))


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
    text = ("{cite:t}`press2017using` tie them; ties are common {cite}`press2017using`.\n"
            "Smith {cite}`press2017using` is another author. In `Smith et al. {cite}` code and\n\n"
            "```text\nSmith et al. {cite}`smith2020x`\n```\n")
    chapter, files = chapter_with(tmp_path, text)
    assert citation_failures(chapter, files) == []


def test_a_hand_written_name_before_any_cite_role_of_the_same_work_is_reported(tmp_path):
    text = ("Smith et al. (2020) {cite:p}`smith2020x` found it.\n"
            "Press and Wolf {cite:t}`press2017using` tie them.\n"
            "Smith et al., in {cite}`smith2020x`, found it.\n")
    chapter, files = chapter_with(tmp_path, text)
    failures = citation_failures(chapter, files)
    assert [f.split(": ")[0] for f in failures] == ["index.md:L1", "index.md:L2", "index.md:L3"]
    assert failures[1] == ("index.md:L2: hand-written 'Press and Wolf' before a cite of press2017using; "
                           "use {cite:t}`press2017using` alone")


def test_a_possessive_a_quotation_another_work_or_a_distant_role_passes(tmp_path):
    text = ("Smith et al.'s {cite}`smith2020x` result holds.\n"
            'The phrase "Press and Wolf" {cite}`press2017using` is a name.\n'
            "Press and Wolf {cite:t}`smith2020x` is another work.\n"
            "Smith et al. reported this long before the later study {cite}`smith2020x`.\n")
    chapter, files = chapter_with(tmp_path, text)
    assert citation_failures(chapter, files) == []
