"""Shared helpers for the tests: file discovery and the per-chapter pending mechanism.

File discovery
    tracked(pattern) -> list[Path]

Per-chapter checks
    Each guarded check is ONE test per chapter. The test collects failure strings
    ("index.md:L234: reason") and hands them to chapter_check(name), which prints them
    and then either asserts (chapter not pending) or xfails strictly (the check name is
    listed in chapters/<dir>/.review-pending). Remove the name from that file once the
    chapter is fixed; a strict xfail that starts passing fails the suite until you do.

        @pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
        def test_something(chapter_dir):
            chapter_check("something")(chapter_dir, my_failures(chapter_dir))
"""
import importlib
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PENDING_FILE = ".review-pending"
SKIP_DIRS = {"_build", ".venv", "venv", ".claude", ".superpowers", ".jupyter_cache", ".git",
             ".ipynb_checkpoints", "__pycache__", ".pytest_cache", "node_modules"}
KNOWN_CHECKS = {"fences", "fences-live", "outputs", "outputs-live", "prose", "sync-chapter", "sync-generator",
                "spelling", "figures", "structure", "exercises", "citations", "xrefs", "notation", "hygiene"}

CHAPTER_DIRS = sorted(p for p in (ROOT / "chapters").iterdir()
                      if p.is_dir() and not p.name.startswith((".", "_")))
CHAPTER_IDS = [p.name for p in CHAPTER_DIRS]


def tracked(pattern: str) -> list[Path]:
    """Absolute paths of the files matching a `git ls-files` pathspec, sorted.

    Falls back to walking the tree (skipping build, environment and tooling directories)
    when git is unavailable or returns nothing, e.g. in an archive without .git. The
    fallback matches like a plain git pathspec: `*` also matches `/`.
    Raises AssertionError if nothing matches either way.
    """
    try:
        out = subprocess.run(["git", "ls-files", "-z", pattern], cwd=ROOT,
                             capture_output=True, text=True).stdout.split("\0")
    except OSError:
        out = []
    found = {ROOT / rel for rel in out if rel and (ROOT / rel).exists()}
    if not found:
        for path in ROOT.rglob("*"):
            rel = path.relative_to(ROOT)
            if SKIP_DIRS.intersection(rel.parts) or not path.is_file():
                continue
            if fnmatchcase(rel.as_posix(), pattern):
                found.add(path)
    assert found, f"no files match {pattern!r} (git ls-files and directory walk both empty)"
    return sorted(found)


def pending(chapter_dir: Path) -> set[str]:
    """Check names listed in <chapter_dir>/.review-pending (blank lines and # comments ignored).

    Raises AssertionError naming the file and line for a name not in KNOWN_CHECKS.
    """
    path = Path(chapter_dir) / PENDING_FILE
    if not path.exists():
        return set()
    names = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#"):
            assert line in KNOWN_CHECKS, (f"{path}:{number}: unknown check name {line!r} "
                                          f"(known: {', '.join(sorted(KNOWN_CHECKS))})")
            names.add(line)
    return names


def report(chapter_dir: Path, name: str, failures: list[str]) -> None:
    """Print the failures, then apply the pending rule for check `name` in `chapter_dir`.

    The failure list is also in the xfail reason and in the assertion message, so `pytest -rx` shows it.
    """
    assert name in KNOWN_CHECKS, f"unknown check name {name!r}"
    chapter_dir = Path(chapter_dir)
    listing = "\n".join(f"{chapter_dir.name}/{f}" for f in failures)
    if failures:
        print(listing)
    if name in pending(chapter_dir):
        if failures:
            pytest.xfail(f"{name} pending for {chapter_dir.name}: {len(failures)} failure(s)\n{listing}")
        pytest.fail(f"[XPASS(strict)] {name} passes for {chapter_dir.name}; "
                    f"remove it from {chapter_dir.name}/{PENDING_FILE}", pytrace=False)
    assert not failures, f"{name}: {len(failures)} failure(s) in {chapter_dir.name}:\n{listing}"


def chapter_check(name: str):
    """Return `check(chapter_dir, failures)`, the report function bound to check `name`."""
    def check(chapter_dir: Path, failures: list[str]) -> None:
        report(chapter_dir, name, failures)
    return check


def chapter_constants(chapter_dir: Path):
    """The per-chapter test module of `chapter_dir`, e.g. tests.test_ch03_snippets for 03-*.

    Read SYNC_EXEMPT, RESULT_CLAIMS and MIN_FIGURES from the returned module.
    """
    number = Path(chapter_dir).name[:2]
    return importlib.import_module(f"tests.test_ch{number}_snippets")
