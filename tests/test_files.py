"""The shared test helpers in tests/_files.py."""
import subprocess

import pytest

from tests import _files
from tests._files import CHAPTER_DIRS, KNOWN_CHECKS, chapter_constants, pending, report, tracked


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
    report(tmp_path, "xrefs", [])  # a different check name is not pending


def test_xfail_reason_and_assertion_message_list_every_failure(tmp_path):
    failures = ["index.md:L3: boom", "index.md:L9: bang"]
    with pytest.raises(AssertionError) as info:
        report(tmp_path, "fences", failures)
    assert all(f in str(info.value) for f in failures)
    (tmp_path / ".review-pending").write_text("fences\n", encoding="utf-8")
    with pytest.raises(pytest.xfail.Exception) as info:
        report(tmp_path, "fences", failures)
    assert all(f in str(info.value) for f in failures) and "2 failure(s)" in str(info.value)


def test_unknown_check_names_are_rejected(tmp_path):
    (tmp_path / ".review-pending").write_text("fences\n# note\nfencez\n", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"\.review-pending:3: unknown check name 'fencez'"):
        pending(tmp_path)
    with pytest.raises(AssertionError, match="unknown check name 'nope'"):
        report(tmp_path / "x", "nope", [])


def test_known_checks_are_exactly_the_documented_names():
    assert KNOWN_CHECKS == {"fences", "fences-live", "outputs", "outputs-live", "prose", "sync-chapter",
                            "sync-generator", "spelling", "figures", "structure", "exercises", "citations",
                            "xrefs", "notation", "hygiene"}


@pytest.mark.parametrize("pattern", ["chapters/*/index.md", "tests/fixtures/*/*.md", "pytest.ini"])
@pytest.mark.parametrize("how", ["git missing", "git empty"])
def test_tracked_fallback_equals_the_git_result(monkeypatch, pattern, how):
    expected = tracked(pattern)

    def fake_run(*args, **kwargs):
        if how == "git missing":
            raise FileNotFoundError("git")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(_files.subprocess, "run", fake_run)
    assert tracked(pattern) == expected


def test_report_fails_strictly_when_a_pending_check_passes(tmp_path):
    (tmp_path / ".review-pending").write_text("fences\n", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="remove it"):
        report(tmp_path, "fences", [])


EXPECTED_MIN_FIGURES = [15, 12, 10, 10, 11, 12]


def test_chapter_constants_load_for_every_chapter():
    for chapter_dir, minimum in zip(CHAPTER_DIRS, EXPECTED_MIN_FIGURES):
        module = chapter_constants(chapter_dir)
        # A chapter may raise its minimum in its own file; the floor is the count at 774011c.
        assert isinstance(module.MIN_FIGURES, int) and module.MIN_FIGURES >= minimum, chapter_dir.name
        for name in ("SYNC_EXEMPT", "NOT_A_RESULT", "NOTATION_OK", "INTERFACE_LABELS"):
            assert isinstance(getattr(module, name), dict), f"{chapter_dir.name}: {name}"
        assert isinstance(module.RESULT_CLAIMS, list), chapter_dir.name
