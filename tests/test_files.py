"""The shared test helpers in tests/_files.py."""
import pytest

from tests._files import CHAPTER_DIRS, chapter_constants, pending, report, tracked


def test_tracked_fails_when_nothing_matches():
    with pytest.raises(AssertionError, match="no-such-file-anywhere"):
        tracked("chapters/**/no-such-file-anywhere.md")


def test_tracked_finds_chapter_pages():
    names = {p.name for p in tracked("chapters/*/index.md")}
    assert names == {"index.md"} and len(tracked("chapters/*/index.md")) == len(CHAPTER_DIRS)


def test_chapter_dirs_are_the_six_chapters():
    assert [p.name for p in CHAPTER_DIRS] == sorted(p.name for p in CHAPTER_DIRS)
    assert len(CHAPTER_DIRS) == 6


def test_pending_ignores_blank_lines_and_comments(tmp_path):
    assert pending(tmp_path) == set()
    (tmp_path / ".review-pending").write_text("# why\n\nfences\n  xrefs  \n", encoding="utf-8")
    assert pending(tmp_path) == {"fences", "xrefs"}


def test_report_passes_or_fails_when_not_pending(tmp_path):
    report(tmp_path, "fences", [])
    with pytest.raises(AssertionError, match="index.md:L3: boom"):
        report(tmp_path, "fences", ["index.md:L3: boom"])


def test_report_prints_failures_and_xfails_when_pending(tmp_path, capsys):
    (tmp_path / ".review-pending").write_text("fences\n", encoding="utf-8")
    with pytest.raises(pytest.xfail.Exception):
        report(tmp_path, "fences", ["index.md:L3: boom"])
    assert "index.md:L3: boom" in capsys.readouterr().out
    report(tmp_path, "other", [])  # a different check name is not pending


def test_report_fails_strictly_when_a_pending_check_passes(tmp_path):
    (tmp_path / ".review-pending").write_text("fences\n", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="remove it"):
        report(tmp_path, "fences", [])


EXPECTED_MIN_FIGURES = [15, 12, 10, 10, 11, 12]


def test_chapter_constants_load_for_every_chapter():
    for chapter_dir, minimum in zip(CHAPTER_DIRS, EXPECTED_MIN_FIGURES):
        module = chapter_constants(chapter_dir)
        assert module.MIN_FIGURES == minimum, chapter_dir.name
        assert isinstance(module.SYNC_EXEMPT, dict)
        assert isinstance(module.RESULT_CLAIMS, list)
