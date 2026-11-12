"""Each chapter notebook executed online (no LLMATLAS_OFFLINE) has the same output shape as the committed one.

Shape: same number of lines per cell, same text once numbers are replaced by a placeholder. Run with
`pytest -m live`. One test per chapter, check name `outputs-live`.
"""
import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check
from tests._notebooks import aborted, compare_cells, execute, notebooks, read
from tests.test_notebook_outputs import MASKS


def check_chapter_live(chapter_dir):
    failures = []
    for path in notebooks(chapter_dir):
        executed = execute(path, offline=False)
        if aborted(executed):
            failures.append(f"{path.name}: execution aborted at {aborted(executed)}")
            continue
        failures += [f"{path.name}: {m}"
                     for m in compare_cells(read(path), executed, MASKS, shape_only=True)]
    return failures


@pytest.mark.live
@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_notebook_outputs_have_the_committed_shape_online(chapter_dir):
    assert notebooks(chapter_dir), f"no notebook under {chapter_dir.name}/examples"
    chapter_check("outputs-live")(chapter_dir, check_chapter_live(chapter_dir))


# Fixture test: the shape comparison used above, on the sample notebook.

def test_shape_comparison_ignores_numbers_and_reports_changed_text(tmp_path):
    from tests.test_notebook_outputs import SAMPLE, sample_copy
    executed = execute(SAMPLE, offline=False)
    assert compare_cells(read(SAMPLE), executed, MASKS, shape_only=True) == []
    numbers = sample_copy(tmp_path, "mean 2.5000", "mean 9.9999")
    assert compare_cells(read(numbers), executed, MASKS, shape_only=True) == []
    words = sample_copy(tmp_path, "items 4", "things 4")
    assert len(compare_cells(read(words), executed, MASKS, shape_only=True)) == 1
