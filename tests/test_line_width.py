"""Code and output lines are at most 88 characters, so they fit the PDF's text block without wrapping.

tools/check_width.py holds the rule: every line inside a code fence of a chapter's pages (not the prose of a directive such as an admonition or figure), and every code-cell
source line and stream or text/plain output line of its notebooks. A chapter directory that contains a
file named .width-pending is an expected failure, strictly: the test xfails while the chapter has wide
lines and fails once it has none, until the file is removed.
"""
import json

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS
from tools.check_width import LIMIT, chapter_failures, main

PENDING = ".width-pending"
WIDE = "x" * (LIMIT + 1)
FITS = "x" * LIMIT


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_code_and_output_lines_fit(chapter_dir):
    failures = chapter_failures(chapter_dir)
    listing = "\n".join(f"{chapter_dir.name}/{f}" for f in failures)
    if (chapter_dir / PENDING).exists():
        if failures:
            pytest.xfail(f"width pending for {chapter_dir.name}: {len(failures)} line(s)\n{listing}")
        pytest.fail(f"[XPASS(strict)] no line is over {LIMIT} characters in {chapter_dir.name}; "
                    f"remove {chapter_dir.name}/{PENDING}", pytrace=False)
    assert not failures, f"{len(failures)} line(s) over {LIMIT} characters:\n{listing}"


def _page(chapter, fenced_line, prose_line=WIDE):
    (chapter / "index.md").write_text(
        f"# Title\n\n{prose_line}\n\n````{{code-cell}} python\n```\n{fenced_line}\n```\n````\n\n"
        f"~~~text\n{FITS}\n~~~\n", encoding="utf-8")


def _notebook(chapter, source, output, output_type="stream"):
    out = {"output_type": "stream", "name": "stdout", "text": [output]} if output_type == "stream" else {
        "output_type": "execute_result", "data": {"text/plain": [output]}, "metadata": {}, "execution_count": 1}
    cell = {"cell_type": "code", "source": [source], "outputs": [out], "metadata": {}, "execution_count": 1}
    md = {"cell_type": "markdown", "source": [WIDE], "metadata": {}}
    (chapter / "examples").mkdir(exist_ok=True)
    (chapter / "examples" / "demo.ipynb").write_text(json.dumps({"cells": [md, cell]}), encoding="utf-8")


def test_page_with_a_wide_fenced_line_fails(tmp_path):
    _page(tmp_path, WIDE)
    failures = chapter_failures(tmp_path)
    assert len(failures) == 1 and failures[0].startswith("index.md:L7 (89 characters)"), failures


def test_page_with_lines_of_exactly_the_limit_passes_and_prose_is_not_checked(tmp_path):
    _page(tmp_path, FITS)
    (tmp_path / "notation-local.md").write_text(f"```\n{WIDE}\n```\n", encoding="utf-8")
    assert chapter_failures(tmp_path) == []


@pytest.mark.parametrize("where,output_type", [("source", "stream"), ("output", "stream"), ("output", "result")])
def test_notebook_with_a_wide_code_or_output_line_fails(tmp_path, where, output_type):
    _notebook(tmp_path, WIDE if where == "source" else FITS, WIDE if where == "output" else FITS, output_type)
    failures = chapter_failures(tmp_path)
    assert len(failures) == 1 and f"cell 1 {where}" in failures[0] and "(89 characters)" in failures[0], failures


@pytest.mark.parametrize("output_type", ["stream", "result"])
def test_notebook_within_the_limit_passes_and_markdown_cells_are_not_checked(tmp_path, output_type):
    _notebook(tmp_path, FITS, FITS, output_type)
    assert chapter_failures(tmp_path) == []


def test_command_line_reports_an_unknown_chapter():
    assert main(["no-such-chapter"]) == 2


def _check(chapter, text):
    (chapter / "index.md").write_text(text, encoding="utf-8")
    return chapter_failures(chapter)


def test_prose_inside_a_dropdown_admonition_is_not_checked(tmp_path):
    page = f"````{{admonition}} Solution\n:class: dropdown\n\n{WIDE}\n\n```python\n{FITS}\n```\n````\n"
    assert _check(tmp_path, page) == []


def test_code_fence_nested_in_an_admonition_is_checked(tmp_path):
    page = f"````{{admonition}} Solution\n:class: dropdown\n\n{WIDE}\n\n```python\n{WIDE}\n```\n````\n"
    failures = _check(tmp_path, page)
    assert len(failures) == 1 and failures[0].startswith("index.md:L7 (89 characters)"), failures


def test_colon_fenced_figure_caption_is_not_checked(tmp_path):
    page = f":::{{figure}} plot.png\n:name: fig-plot\n\n{WIDE}\n:::\n"
    assert _check(tmp_path, page) == []


def test_text_output_fence_nested_in_a_colon_fenced_directive_is_checked(tmp_path):
    page = f":::{{note}}\n{WIDE}\n\n```text\n{WIDE}\n```\n:::\n"
    failures = _check(tmp_path, page)
    assert len(failures) == 1 and failures[0].startswith("index.md:L5 (89 characters)"), failures


@pytest.mark.parametrize("opener", ["```{code-block} python", "```{sourcecode} python", "```bash", "```", "```json"])
def test_code_fences_are_checked(tmp_path, opener):
    failures = _check(tmp_path, f"{opener}\n{WIDE}\n```\n")
    assert len(failures) == 1 and failures[0].startswith("index.md:L2"), failures


def test_math_directive_is_not_checked(tmp_path):
    assert _check(tmp_path, f"```{{math}}\n{WIDE}\n```\n") == []
