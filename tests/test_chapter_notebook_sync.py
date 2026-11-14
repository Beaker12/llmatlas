"""Code a chapter shares with its notebook stays identical.

Every function or class that tools/list_shared_defs.py reports for a chapter must be identical in the chapter's
python fences and in its notebook (last definition on each side) after ast.unparse (docstrings and comments ignored), unless the chapter's
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
from tools.list_shared_defs import normalised, notebook_defs, parse_failures, shared_defs, top_level_defs


def check_sync(chapter_dir, exempt=None):
    """Failure strings for shared names whose chapter and notebook definitions differ."""
    if exempt is None:
        exempt = chapter_constants(chapter_dir).SYNC_EXEMPT
    failures = parse_failures(chapter_dir)
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
    # Deliberate: the two chapters share this class; a chapter that renames or changes it changes both notebooks.
    found = []
    for chapter in ("01-foundations", "02-mathematics"):
        path = first_notebook(chapter)
        defs = notebook_defs(path)
        assert "ScaledDotProductAttention" in defs, \
            f"class ScaledDotProductAttention not found in {path.relative_to(ROOT)}"
        found.append(defs["ScaledDotProductAttention"])
    assert normalised(found[0]) == normalised(found[1]), \
        "ScaledDotProductAttention differs between the chapter 1 and chapter 2 notebooks"


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


# Fixture tests for what the tool sees.

def write_notebook(path, *sources):
    cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": s}
             for s in sources]
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
                    encoding="utf-8")


def mini_chapter(tmp_path, page, *cells):
    chapter = tmp_path / "ch"
    (chapter / "examples").mkdir(parents=True)
    (chapter / "index.md").write_text(page, encoding="utf-8")
    write_notebook(chapter / "examples" / "n.ipynb", *cells)
    return chapter


def fence(code):
    return f"```python\n{code}\n```\n\n"


def test_a_percent_on_a_continuation_line_is_not_a_magic():
    source = "x = (1\n%2)\n\ndef f():\n    return 1\n"
    assert "f" in top_level_defs(source)
    assert "%2" in blank_lines_kept(source)


def blank_lines_kept(source):
    from tools.list_shared_defs import blank_magics
    return blank_magics(source)


def test_real_magics_at_depth_zero_are_blanked():
    source = "%matplotlib inline\n!pip list\ndef f():\n    return 1\n"
    assert list(top_level_defs(source)) == ["f"]


def test_unparsable_cells_and_fences_are_reported_and_pseudo_code_is_not(tmp_path):
    page = fence("def f(:\n    pass") + fence("# pseudo-code\ndef g(:\n") + fence("def h():\n    return 1")
    chapter = mini_chapter(tmp_path, page, "x = 1", "def broken(:\n  pass")
    failures = parse_failures(chapter)
    assert len(failures) == 2
    assert failures[0].startswith("index.md:L1: cannot parse: ") and failures[1].startswith("n.ipynb:cell 1: cannot parse: ")
    assert check_sync(chapter, {})[:2] == failures


def test_definitions_inside_if_try_and_with_are_top_level():
    source = ("if True:\n    def a(): return 1\n"
              "try:\n    def b(): return 2\nexcept ImportError:\n    def c(): return 3\n"
              "with open('x') as f:\n    class D: pass\n")
    assert list(top_level_defs(source)) == ["a", "b", "c", "D"]


def test_the_last_definition_on_each_side_is_compared(tmp_path):
    page = fence("def f():\n    return 1") + fence("def f():\n    return 2")
    chapter = mini_chapter(tmp_path, page, "def f():\n    return 1", "def f():\n    return 2")
    assert check_sync(chapter, {}) == []
    chapter = mini_chapter(tmp_path / "b", page, "def f():\n    return 2", "def f():\n    return 1")
    assert len(check_sync(chapter, {})) == 1


def test_async_decorated_nested_docstring_and_comment_cases(tmp_path):
    page = fence("@dec\nasync def a():\n    'doc'\n    return 1\n\n\ndef outer():\n    def inner():\n        return 1\n"
                 "    return inner\n\n\ndef k():\n    return 1  # comment")
    chapter = mini_chapter(tmp_path, page,
                           "@dec\nasync def a():\n    'another doc'\n    return 1\n\n\n"
                           "def outer():\n    def inner():\n        return 2\n    return inner\n\n\n"
                           "def k():\n    # different comment\n    return 1")
    assert set(shared_defs(chapter)) == {"a", "outer", "k"}   # inner is nested: not reported
    failures = check_sync(chapter, {})
    assert len(failures) == 1 and failures[0].startswith("outer: differs")   # a, k differ only by doc/comment
    assert "inner" not in shared_defs(chapter)
    assert shared_defs(chapter)["a"][0].startswith("@dec")
