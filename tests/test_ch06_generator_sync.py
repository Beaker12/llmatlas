"""The chapter 6 notebook equals what tools/build_architecture_notebook.py builds.

The generator's cell list (its module-level CELLS, built on import; `main`/`build` is not run and nothing is
written) must equal the committed notebook's cells, kind and source. Check name `sync-generator`, chapter 6 only.
"""
import importlib
import json

import pytest

from tests._files import CHAPTER_DIRS, ROOT, chapter_check

CH06 = next(p for p in CHAPTER_DIRS if p.name.startswith("06-"))


def generator_cells():
    """The generator's cells as (kind, source) pairs."""
    return [tuple(cell) for cell in importlib.import_module("tools.build_architecture_notebook").CELLS]


def notebook_cells(path):
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    return [(c["cell_type"], "".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in cells]


def compare_cells(built, committed):
    """Failure strings for the differences between the generator's cells and the notebook's."""
    failures = []
    if len(built) != len(committed):
        failures.append(f"generator builds {len(built)} cells, notebook has {len(committed)}")
    for i, (a, b) in enumerate(zip(built, committed)):
        if a == b:
            continue
        if a[0] != b[0]:
            failures.append(f"cell {i}: generator builds a {a[0]} cell, notebook has a {b[0]} cell")
            continue
        if a[1].rstrip("\n") == b[1].rstrip("\n"):
            failures.append(f"cell {i}: trailing newline differs")
            continue
        old, new = a[1].splitlines(), b[1].splitlines()
        line = next((n for n, (x, y) in enumerate(zip(old, new), start=1) if x != y), min(len(old), len(new)) + 1)
        x = old[line - 1] if line <= len(old) else "<end>"
        y = new[line - 1] if line <= len(new) else "<end>"
        failures.append(f"cell {i}: line {line}: generator {x!r}, notebook {y!r}")
    return failures


def test_generator_cells_equal_the_committed_notebook():
    notebook = next((CH06 / "examples").glob("*.ipynb"))
    try:
        failures = compare_cells(generator_cells(), notebook_cells(notebook))
    except SystemExit as e:  # the generator exits when a chapter fence it copies cannot be found
        failures = [f"generator could not build its cells: {e}"]
    chapter_check("sync-generator")(CH06, failures)


# Fixture tests: the comparison on the sample notebook and deliberately broken copies of its cells.

SAMPLE = ROOT / "tests" / "fixtures" / "sample-chapter" / "examples" / "sample_demo.ipynb"


def test_identical_cell_lists_pass():
    cells = notebook_cells(SAMPLE)
    assert len(cells) == 4 and compare_cells(list(cells), cells) == []


def test_an_edited_generator_cell_is_reported_with_its_line():
    cells = notebook_cells(SAMPLE)
    built = list(cells)
    built[1] = ("code", built[1][1].replace("mean = average(values)", "mean = average(values) + 0"))
    failures = compare_cells(built, cells)
    assert len(failures) == 1 and failures[0].startswith("cell 1: line 10:") and "+ 0" in failures[0]


def test_a_changed_cell_kind_and_a_missing_cell_are_reported():
    cells = notebook_cells(SAMPLE)
    assert compare_cells([("code", cells[0][1])] + cells[1:], cells) == [
        "cell 0: generator builds a code cell, notebook has a markdown cell"]
    assert compare_cells(cells[:-1], cells) == ["generator builds 3 cells, notebook has 4"]


def test_generator_cells_have_the_expected_shape():
    cells = generator_cells()
    assert cells and all(kind in ("markdown", "code") and isinstance(text, str) for kind, text in cells)


def test_importing_the_generator_writes_nothing():
    target = ROOT / "chapters" / "06-architectures" / "examples" / "architecture_demo.ipynb"
    before = target.stat().st_mtime_ns
    importlib.reload(importlib.import_module("tools.build_architecture_notebook"))
    assert target.stat().st_mtime_ns == before


def test_cells_that_differ_only_by_a_trailing_newline_are_named_as_such():
    assert compare_cells([("code", "x = 1")], [("code", "x = 1\n")]) == ["cell 0: trailing newline differs"]
    assert compare_cells([("code", "x = 1\n\n")], [("code", "x = 1")]) == ["cell 0: trailing newline differs"]
