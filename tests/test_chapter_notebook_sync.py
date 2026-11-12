"""Code a chapter shares with its notebook stays identical.

Every function or class that tools/list_shared_defs.py reports for a chapter must be identical in the chapter's
python fences and in its notebook after ast.unparse (docstrings and comments ignored), unless the chapter's
SYNC_EXEMPT lists the name. One test per chapter, check name `sync-chapter`. A second test asserts that
ScaledDotProductAttention is identical in the chapter 1 and chapter 2 notebooks.
"""
import difflib
import json
import shutil
import subprocess
import sys

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, chapter_constants
from tools.list_shared_defs import normalised, notebook_defs, shared_defs


def check_sync(chapter_dir, exempt=None):
    """Failure strings for shared names whose chapter and notebook definitions differ."""
    if exempt is None:
        exempt = chapter_constants(chapter_dir).SYNC_EXEMPT
    failures = []
    for name, (in_chapter, in_notebook) in shared_defs(chapter_dir).items():
        if name in exempt:
            continue
        a, b = normalised(in_chapter), normalised(in_notebook)
        if a != b:
            diff = [l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), "chapter", "notebook",
                                                    n=0, lineterm="") if l[0] in "+-" and l[:3] not in ("+++", "---")]
            failures.append(f"{name}: differs between chapter and notebook; first difference: "
                            f"{diff[0].strip()!r} / {diff[1].strip() if len(diff) > 1 else ''!r}")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_shared_code_is_identical_in_chapter_and_notebook(chapter_dir):
    chapter_check("sync-chapter")(chapter_dir, check_sync(chapter_dir))


def first_notebook(chapter_name):
    return next((ROOT / "chapters" / chapter_name / "examples").glob("*.ipynb"))


def test_scaled_dot_product_attention_is_identical_in_notebooks_1_and_2():
    one = notebook_defs(first_notebook("01-foundations"))["ScaledDotProductAttention"]
    two = notebook_defs(first_notebook("02-mathematics"))["ScaledDotProductAttention"]
    assert normalised(one) == normalised(two)


def test_exemptions_in_every_chapter_name_a_shared_definition_and_a_reason():
    for chapter_dir in CHAPTER_DIRS:
        for name, reason in chapter_constants(chapter_dir).SYNC_EXEMPT.items():
            assert name in shared_defs(chapter_dir) and reason.strip(), f"{chapter_dir.name}: {name}"


# Fixture tests: the check on deliberately broken copies of the sample chapter.

FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"


def sample_chapter(tmp_path, old=None, new=None, page=False):
    """A copy of the fixture chapter; `old` is replaced by `new` in the notebook JSON (or in the page)."""
    chapter = tmp_path / "sample"
    shutil.copytree(FIXTURE, chapter)
    path = chapter / ("index.md" if page else "examples/sample_demo.ipynb")
    if old is not None:
        text = path.read_text(encoding="utf-8")
        assert old in text
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return chapter


def test_the_shared_names_of_the_sample_chapter():
    assert list(shared_defs(FIXTURE)) == ["average"]


def test_sample_chapter_passes_although_docstring_and_comments_differ():
    in_chapter, in_notebook = shared_defs(FIXTURE)["average"]
    assert "Arithmetic mean" in in_chapter and "comment differently" in in_notebook
    assert check_sync(FIXTURE, {}) == []


def test_an_edited_shared_function_is_reported(tmp_path):
    chapter = sample_chapter(tmp_path, "return sum(xs) / len(xs)", "return sum(xs) / (len(xs) + 1)")
    failures = check_sync(chapter, {})
    assert len(failures) == 1 and failures[0].startswith("average: differs") and "len(xs) + 1" in failures[0]


def test_an_edit_in_the_chapter_page_is_reported_too(tmp_path):
    chapter = sample_chapter(tmp_path, "def average(xs):", "def average(values, xs=None):", page=True)
    assert len(check_sync(chapter, {})) == 1


def test_an_exempt_name_is_not_reported(tmp_path):
    chapter = sample_chapter(tmp_path, "return sum(xs) / len(xs)", "return sum(xs) / (len(xs) + 1)")
    assert check_sync(chapter, {"average": "the notebook rounds differently"}) == []


def test_a_name_defined_on_one_side_only_is_not_shared(tmp_path):
    chapter = sample_chapter(tmp_path, "def average(xs):", "def mean_of(xs):")
    assert shared_defs(chapter) == {} and check_sync(chapter, {}) == []


def test_the_command_line_tool_prints_the_shared_names():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "list_shared_defs.py"), str(FIXTURE)],
                         capture_output=True, text=True, check=True).stdout
    assert out.split() == ["average"]


def test_decorated_classes_and_magics_are_read(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": "%matplotlib inline\n@dec\nclass A:\n    '''doc'''\n    x = 1\n"}]}
    path = tmp_path / "n.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    source = notebook_defs(path)["A"]
    assert source.startswith("@dec") and normalised(source) == "@dec\nclass A:\n    x = 1"
