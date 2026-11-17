"""The chapter 7 notebook equals what tools/build_training_notebook.py builds.

The generator's cell list (its module-level CELLS, built on import; `main`/`build` is not run and nothing is written)
must equal the committed notebook's cells, kind and source. Check name `sync-generator`, chapter 7 only. The
comparison itself is chapter 6's (tests/test_ch06_generator_sync.compare_cells), whose fixture tests cover it.
"""
import importlib

from tests._files import CHAPTER_DIRS, ROOT, chapter_check
from tests.test_ch06_generator_sync import compare_cells, notebook_cells

CH07 = next(p for p in CHAPTER_DIRS if p.name.startswith("07-"))


def generator_cells():
    return [tuple(cell) for cell in importlib.import_module("tools.build_training_notebook").CELLS]


def test_generator_cells_equal_the_committed_notebook():
    notebook = next((CH07 / "examples").glob("*.ipynb"))
    try:
        failures = compare_cells(generator_cells(), notebook_cells(notebook))
    except SystemExit as e:  # the generator exits when a chapter fence it copies cannot be found
        failures = [f"generator could not build its cells: {e}"]
    chapter_check("sync-generator")(CH07, failures)


def test_importing_the_generator_writes_nothing():
    target = ROOT / "chapters" / "07-training-objectives" / "examples" / "training_objectives_demo.ipynb"
    before = target.stat().st_mtime_ns
    importlib.reload(importlib.import_module("tools.build_training_notebook"))
    assert target.stat().st_mtime_ns == before


def test_recorded_schedule_runs_match_the_training_cell():
    """The figure's data was recorded by the notebook's current training cell, and its claims hold on it."""
    import json

    generator = importlib.import_module("tools.build_training_notebook")
    record = json.loads(generator.RECORD_JSON.read_text(encoding="utf-8"))
    assert record["train_cell_sha256"] == generator.train_cell_sha256(), \
        "the training cell changed; rerun tools/build_training_notebook.py --record"
    assert sorted(int(s) for s in record["runs"]) == list(generator.RECORD_SEEDS)
    failed = [claim for claim, holds in generator.record_claims(record).items() if not holds]
    assert not failed, failed
