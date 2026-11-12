"""The sample chapter passes every page-level check, and each check fails on a deliberately broken copy of it.

Exercises, structure and figures have their broken-copy tests in their own modules; this module covers the
spelling, citation, cross-reference, hygiene and notation checks on copies of the fixture page.
"""
import shutil

import pytest

from tests._files import ROOT
from tests.test_citation_style import citation_failures
from tests.test_exercises import page_failures, size_failures
from tests.test_figures import figure_failures
from tests.test_notation import notation_failures
from tests.test_source_hygiene import character_failures, checksum_failures
from tests.test_spelling import spelling_failures
from tests.test_xrefs import body_text_failures

FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
BIB = [FIXTURE / "refs.bib"]


def all_failures(chapter):
    bib = [chapter / "refs.bib"]
    return {"spelling": spelling_failures(chapter, generators={}),
            "figures": figure_failures(chapter, 2),
            "exercises": page_failures(chapter), "structure": size_failures(chapter),
            "citations": citation_failures(chapter, bib), "xrefs": body_text_failures(chapter),
            "hygiene": character_failures(chapter) + checksum_failures(chapter),
            "notation": notation_failures(chapter)}


def broken(tmp_path, old, new):
    chapter = tmp_path / "chapter"
    shutil.copytree(FIXTURE, chapter)
    text = (chapter / "index.md").read_text(encoding="utf-8")
    assert old in text, old
    (chapter / "index.md").write_text(text.replace(old, new, 1), encoding="utf-8")
    return chapter


def test_the_sample_chapter_passes_every_page_check():
    assert all_failures(FIXTURE) == {name: [] for name in
                                     ("spelling", "figures", "exercises", "structure", "citations", "xrefs", "hygiene",
                                      "notation")}


@pytest.mark.parametrize("old,new,check,fragment", [
    ("print(7)\n", "print(7)​\n", "hygiene", "literal U+200B ZERO WIDTH SPACE"),
    ("shows the ramp", "shows the ramp, see Section 2.1", "xrefs", "literal 'Section 2.1'"),
    ("shows the ramp", "shows the ramp, see Chapter 10", "xrefs", "'Chapter 10' names an unwritten chapter"),
    ("shows the ramp", "shows the colour ramp", "spelling", "'colour' -> 'color'"),
    ("Demo {cite:t}`demo2020` that", "Demo et al. {cite}`demo2020` that", "citations", "'et al.' followed by plain {cite}"),
    ("Demo {cite:t}`demo2020` that", "Sample {cite}`demo2020` that", "citations", "surname 'Sample' before {cite}`demo2020`"),
    ("shows the ramp", "shows the ramp of $\\chi$", "notation", "symbol '\\\\chi' is in neither"),
    ("shows the ramp", "shows the ramp in $O(n)$", "notation", "banned 'O(' (Asymptotic order"),
    ("shows the ramp", "shows the ramp, where $d$ is the width, where $d$ is the depth", "notation",
     "'d' defined as 'the depth', but as 'the width'"),
])
def test_each_check_fails_on_a_broken_copy_of_the_sample_chapter(tmp_path, old, new, check, fragment):
    failures = all_failures(broken(tmp_path, old, new))
    assert [f for f in failures[check] if fragment in f], failures[check]
    assert all(not v for name, v in failures.items() if name != check)


def test_an_allow_listed_british_proper_name_passes_in_the_sample_chapter(tmp_path):
    chapter = broken(tmp_path, "shows the ramp", "shows the ramp from the Centre for Fixtures")
    assert any("'Centre'" in f for f in spelling_failures(chapter, generators={}))
    (chapter / "spelling-allow.txt").write_text("# names\nCentre for Fixtures\n", encoding="utf-8")
    assert spelling_failures(chapter, generators={}) == []
