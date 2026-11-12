"""Each chapter notebook, executed offline in a temporary copy, reproduces its committed outputs.

Only stream text and text/plain are compared (images and HTML are ignored). Text must match after the
MASKS below are applied; numbers must agree to within one unit in the last printed digit, or the absolute
tolerance in the cell's "tol" metadata; a cell with "compare": "shape" is compared by shape only. The
committed kernel metadata must name Python 3.12. One test per chapter, check name `outputs`.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check
from tests._notebooks import compare_cells, execute, kernel_failures, mask, notebooks, read

# Text that legitimately differs between runs or machines; replaced before comparing.
MASKS = [
    (re.compile(r"(?<=version: )\S+"), "<version>"),                         # PyTorch version: 2.15.0+cpu
    (re.compile(r"\b\d+\.\d+\.\d+\+[0-9A-Za-z.]+"), "<version>"),            # 2.15.0+cpu
    (re.compile(r"(?<=: )\d+(?:\.\d+)? s(?=\])"), "<seconds>"),              # [configs: 1.3 s]
    (re.compile(r"(?<=took )\d+(?:\.\d+)? ?(?:ms|s)\b"), "<seconds>"),       # took 3.2 s
    (re.compile(r"(?:/tmp|/var/folders|/private/var)/[^\s'\")\]]+"), "<tmp>"),  # absolute temp paths
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "<address>"),                         # memory addresses
]


def check_notebook(path, executed=None):
    """Failure strings for one notebook; `executed` is a pre-executed copy (else it is executed here)."""
    path = Path(path)
    committed = read(path)
    failures = [f"{path.name}: {m}" for m in kernel_failures(committed)]
    if executed is None:
        executed = execute(path, offline=True)
    failures += [f"{path.name}: {m}" for m in compare_cells(committed, executed, MASKS)]
    return failures


def check_chapter(chapter_dir):
    failures = []
    for path in notebooks(chapter_dir):
        failures += check_notebook(path)
    return failures


@pytest.mark.slow
@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_notebook_outputs_reproduce(chapter_dir):
    assert notebooks(chapter_dir), f"no notebook under {chapter_dir.name}/examples"
    chapter_check("outputs")(chapter_dir, check_chapter(chapter_dir))


# Fixture tests: the checker on deliberately broken copies of the sample notebook.

SAMPLE_DIR = ROOT / "tests" / "fixtures" / "sample-chapter" / "examples"
SAMPLE = SAMPLE_DIR / "sample_demo.ipynb"


@pytest.fixture(scope="module")
def executed():
    return execute(SAMPLE, offline=True)


def sample_copy(tmp_path, old, new, metadata=None):
    """A copy of the sample notebook (with its data/) where `old` is replaced by `new` in the JSON text.

    `metadata` maps a cell index to a dict merged into that cell's metadata.
    """
    text = SAMPLE.read_text(encoding="utf-8")
    assert old in text
    nb = json.loads(text.replace(old, new, 1))
    for index, extra in (metadata or {}).items():
        nb["cells"][index]["metadata"].update(extra)
    path = tmp_path / SAMPLE.name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    shutil.copytree(SAMPLE_DIR / "data", tmp_path / "data", dirs_exist_ok=True)
    return path


def test_sample_notebook_passes(executed):
    assert check_notebook(SAMPLE, executed) == []


def test_executing_a_copy_reads_its_data_directory_and_runs_in_this_interpreter(tmp_path):
    path = sample_copy(tmp_path, "print(f", "print(sys.executable)\\nprint(f")
    path.write_text(path.read_text(encoding="utf-8").replace("import json", "import json, sys", 1), encoding="utf-8")
    shown = execute(path, offline=True).cells[1].outputs[0]["text"].splitlines()
    assert shown[0] == sys.executable
    assert shown[1] == "mean 2.5000"


def test_a_changed_printed_number_is_reported(tmp_path, executed):
    failures = check_notebook(sample_copy(tmp_path, "mean 2.5000", "mean 2.6000"), executed)
    assert len(failures) == 1 and failures[0].startswith("sample_demo.ipynb: cell 1: output line 1")


def test_a_number_changed_within_one_last_digit_passes(tmp_path, executed):
    assert check_notebook(sample_copy(tmp_path, "mean 2.5000", "mean 2.5001"), executed) == []


def test_a_number_changed_by_two_last_digits_is_reported(tmp_path, executed):
    assert len(check_notebook(sample_copy(tmp_path, "mean 2.5000", "mean 2.5002"), executed)) == 1


def test_tol_in_the_cell_metadata_replaces_the_tolerance(tmp_path, executed):
    assert len(check_notebook(sample_copy(tmp_path, "mean 2.5000", "mean 2.5200"), executed)) == 1
    within = sample_copy(tmp_path, "mean 2.5000", "mean 2.5200", metadata={1: {"tol": 0.05}})
    assert check_notebook(within, executed) == []
    outside = sample_copy(tmp_path, "mean 2.5000", "mean 2.6000", metadata={1: {"tol": 0.05}})
    assert len(check_notebook(outside, executed)) == 1


def test_compare_shape_ignores_numbers_but_not_text(tmp_path, executed):
    shape = {1: {"compare": "shape"}}
    assert check_notebook(sample_copy(tmp_path, "mean 2.5000", "mean 9.9999", metadata=shape), executed) == []
    assert len(check_notebook(sample_copy(tmp_path, "items 4", "things 4", metadata=shape), executed)) == 1


def test_masks_hide_versions_timings_addresses_and_temp_paths():
    for a, b in [("PyTorch version: 2.15.0+cpu", "PyTorch version: 2.14.1+cpu"),
                 ("[configs: 1.3 s]", "[configs: 22.0 s]"),
                 ("at 0x7f3a9c1b2d40", "at 0x7f00deadbeef"),
                 ("in /tmp/tmpab12cd/data", "in /tmp/tmpzz99yy/data")]:
        assert mask(a, MASKS) == mask(b, MASKS)
    assert mask("mean 2.5", MASKS) == "mean 2.5"


def test_a_cell_whose_code_now_prints_something_else_is_reported_after_real_execution(tmp_path):
    path = sample_copy(tmp_path, "mean = average(values)", "mean = average(values) + 1")
    failures = check_notebook(path)
    assert any("cell 1: output line 1" in f for f in failures)


def test_kernel_metadata_of_python_3_13_is_reported(tmp_path, executed):
    path = sample_copy(tmp_path, '"version": "3.12.12"', '"version": "3.13.5"')
    assert check_notebook(path, executed) == [
        "sample_demo.ipynb: kernel metadata names Python 3.13.5, expected 3.12.x"]


def test_an_error_in_a_cell_is_reported(tmp_path):
    failures = check_notebook(sample_copy(tmp_path, "mean = average(values)", "mean = undefined_name"))
    assert any("NameError" in f for f in failures)
