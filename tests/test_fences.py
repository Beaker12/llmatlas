"""Every ```python fence in a chapter either passes pyflakes (no undefined names,
no syntax errors) or is marked on its first line as pseudo-code, and runs standalone."""
import ast
import os
import re
import subprocess
import sys
import tempfile

import pytest

from tests._compare import compare_output
from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, tracked
from tests._pages import pages as chapter_pages
from tools.fences import is_python, python_fences as shared_python_fences

PSEUDO_MARKER = re.compile(r"# pseudo-code(?![\w-])")
NETWORK_MARKER = re.compile(r"# network(?![\w-])")


def fences():
    for path in tracked("chapters/**/*.md"):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for fence in shared_python_fences(text):
            line = fence["line"] + 1
            yield pytest.param(rel, line, fence["body"], id=f"{rel}:{line}")


@pytest.mark.parametrize("rel,line,code", list(fences()))
def test_fence_is_runnable_or_marked_pseudo_code(rel, line, code):
    if PSEUDO_MARKER.match(code.lstrip()):
        pytest.skip("marked pseudo-code")
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise AssertionError(f"{rel}:{line}: syntax error: {e}") from None
    proc = subprocess.run([sys.executable, "-m", "pyflakes"], input=code,
                          capture_output=True, text=True)
    assert proc.returncode in (0, 1) and not proc.stderr, (
        f"pyflakes did not run (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
    problems = [l for l in proc.stdout.splitlines() if "undefined name" in l]
    assert not problems, f"{rel}:{line}:\n" + "\n".join(problems)


# --- standalone execution of the fences -------------------------------------------------
# Each ```python fence is run on its own. A fence followed by `% output` and a ```text
# fence must print exactly that text (numbers to within one unit of the last digit shown);
# `% output-shape` compares only the text around the numbers and the line count.
# First-line markers: `# pseudo-code` is never run, `# network` only by the live test.
# Every other fence must print at least one line and be followed by its output block (the page convention:
# a runnable listing shows a key result). A `# network` fence must have its block too; it is not run here.
# A `# pseudo-code` fence must not be a program: see stale_pseudo_code().
# Failure strings are "<page>:L<line of the opening fence>: <reason>".

RUN_TIMEOUT = 120
OUTPUT_BLOCK = re.compile(r"\s*% output(-shape)?[ \t]*\n\s*```text\n([\s\S]*?)```")
OUTPUT_MARKER = re.compile(r"\s*% output(?:-shape)?[ \t]*(?:\n|$)")
ANY_MARKER = re.compile(r"\s*%\s*output", re.I)  # a marker, possibly mistyped (reported by marker_failures)


def run_fence(code, timeout):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run(
            [sys.executable, "-W", "error", "-W", "ignore:FigureCanvasAgg is non-interactive", "-"],
            input=code, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=timeout)


def python_fences(text):
    """(line of the opening fence, code, text after the fence) for each ```python fence."""
    for fence in shared_python_fences(text):
        yield fence["line"], fence["body"], text[fence["end"]:]


def marker_failures(text, name):
    """`% output` lines that are mistyped or not directly after a python fence."""
    failures = []
    fence = None          # language of the open fence, if any
    after_python = False  # only blank lines since the closing line of a python fence
    for n, line in enumerate(text.splitlines(), start=1):
        if fence is not None:
            if line.startswith("```"):
                fence, after_python = None, is_python(fence)
            continue
        if line.startswith("```"):
            fence, after_python = line[3:].strip(), False
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"%\s*output", stripped, re.I):
            if stripped not in ("% output", "% output-shape"):
                failures.append(f"{name}:L{n}: mistyped marker {stripped!r} (use '% output' or '% output-shape')")
            elif not after_python:
                failures.append(f"{name}:L{n}: {stripped!r} is not directly after a python fence")
        after_python = False
    return failures


def stale_pseudo_code(code):
    """True when a `# pseudo-code` fence is plainly a runnable program, so the marker hides real code.

    The rule: the fence parses as Python, pyflakes finds no undefined name in it, and at top level it both
    defines a function and calls a function it defines. Sketches (with `...`, undefined helpers or prose that
    does not parse) and fragments that only define stay legal.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    called = {c.func.id for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
              for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
    if not defined & called:
        return False
    proc = subprocess.run([sys.executable, "-m", "pyflakes"], input=code, capture_output=True, text=True)
    return "undefined name" not in proc.stdout


def has_network_fence(page):
    return any(NETWORK_MARKER.match(code.lstrip())
               for _, code, _ in python_fences(page.read_text(encoding="utf-8")))


def check_page(page, network=False, timeout=RUN_TIMEOUT):
    """Run the page's python fences (or only its `# network` ones) and list the failures."""
    text = page.read_text(encoding="utf-8")
    failures = [] if network else marker_failures(text, page.name)
    for line, code, rest in python_fences(text):
        first = code.lstrip()
        where = f"{page.name}:L{line}"
        if PSEUDO_MARKER.match(first):
            if not network and stale_pseudo_code(code):
                failures.append(f"{where}: marked '# pseudo-code' but defines and calls its own functions and runs; "
                                "drop the marker and show its output")
            continue
        block = OUTPUT_BLOCK.match(rest)
        if bool(NETWORK_MARKER.match(first)) != network:
            if not network and block is None and not ANY_MARKER.match(rest):
                failures.append(f"{where}: no '% output' block after the fence")
            continue
        try:
            proc = run_fence(code, timeout)
        except subprocess.TimeoutExpired:
            failures.append(f"{where}: timeout after {timeout}s")
            continue
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            failures.append(f"{where}: exit {proc.returncode}: {err[-1][:200] if err else 'no stderr'}")
            continue
        if not proc.stdout.strip() and not network:
            failures.append(f"{where}: prints nothing; print a key result and show it in a '% output' block")
            continue
        if block is None:
            if OUTPUT_MARKER.match(rest):
                failures.append(f"{where}: '% output' is not followed by a ```text fence")
            elif not network and not ANY_MARKER.match(rest):
                failures.append(f"{where}: no '% output' block after the fence")
            continue
        reason = compare_output(proc.stdout, block.group(2), shape=bool(block.group(1)))
        if reason:
            failures.append(f"{where}: {reason}")
    return failures


def check_chapter(chapter_dir, network=False):
    failures = []
    for page in chapter_pages(chapter_dir):
        failures += check_page(page, network=network)
    return failures


FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
SAMPLE = FIXTURE / "index.md"


def _line_start(text, line):
    """Offset of the first character of the 1-based line `line`."""
    return sum(len(x) + 1 for x in text.split("\n")[:line - 1])


def chunk(marker):
    """The fixture fence containing `marker`, with its output block, as a page of its own (fence at L1)."""
    text = SAMPLE.read_text(encoding="utf-8")
    for fence in shared_python_fences(text):
        if marker in fence["body"]:
            start = _line_start(text, fence["line"])
            block = OUTPUT_BLOCK.match(text[fence["end"]:])
            return text[start: fence["end"] + (block.end() if block else 0)] + "\n"
    raise AssertionError(f"no fixture fence contains {marker!r}")


def page_of(tmp_path, text):
    page = tmp_path / "index.md"
    page.write_text(text, encoding="utf-8")
    return page


def broken(tmp_path, marker, old, new):
    """A page holding only the fixture fence that contains `marker`, with `old` replaced by `new`."""
    text = chunk(marker)
    assert old in text
    return page_of(tmp_path, text.replace(old, new, 1))


def test_sample_chapter_passes():
    assert check_chapter(FIXTURE) == []


def test_sample_chapter_network_fence_runs_only_when_asked():
    text = SAMPLE.read_text(encoding="utf-8")
    line = next(n for n, l in enumerate(text.splitlines(), start=1) if l == "# network") - 1
    assert check_page(SAMPLE, network=True) == [f"index.md:L{line}: exit 1: RuntimeError: network fence ran"]


def test_number_within_one_unit_of_the_last_digit_is_accepted(tmp_path):
    assert check_page(broken(tmp_path, "3.14159", "3.1416", "3.1415")) == []


def test_text_that_differs_is_reported_with_the_first_differing_line(tmp_path):
    failures = check_page(broken(tmp_path, "3.14159", "3.1416\ndone", "3.1416\nfinished"))
    assert failures == ["index.md:L1: output line 3: expected 'finished', got 'done'"]


def test_number_off_by_more_than_one_unit_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "3.14159", "3.1416", "3.1418"))
    assert failures == ["index.md:L1: output line 2: expected '3.1418', got '3.14159'"]


def test_an_integer_that_differs_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "3.14159", "7\n3.1416", "8\n3.1416"))
    assert failures == ["index.md:L1: output line 1: expected '8', got '7'"]


def test_output_shape_with_the_wrong_line_count_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "loss 0.123", "loss 4.5\n", ""))
    assert failures == ["index.md:L1: output-shape: expected 1 lines, got 2"]


def test_output_shape_with_different_text_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "loss 0.123", "loss 4.5", "cost 4.5"))
    assert failures == ["index.md:L1: output line 2: expected 'cost #', got 'loss #'"]


def test_a_fence_that_raises_is_reported_with_the_last_stderr_line(tmp_path):
    failures = check_page(broken(tmp_path, "3.14159", "print(7)", "raise ValueError('bad')"))
    assert failures == ["index.md:L1: exit 1: ValueError: bad"]


def test_a_fence_that_times_out_is_reported(tmp_path):
    page = page_of(tmp_path, "```python\nimport time\ntime.sleep(30)\n```\n\n% output\n\n```text\n\n```\n")
    assert check_page(page, timeout=1) == ["index.md:L1: timeout after 1s"]


def test_output_marker_without_a_text_fence_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "3.14159", "```text\n7\n", "```\n7\n"))
    assert len(failures) == 1 and "% output" in failures[0] and failures[0].startswith("index.md:L1:")


def test_trailing_whitespace_in_the_output_is_ignored(tmp_path):
    assert check_page(broken(tmp_path, "3.14159", "done\n```", "done   \n```")) == []


def test_markers_match_as_whole_words(tmp_path):
    ran = "```python\n# networking\nprint('ran')\n```\n\n% output\n\n```text\nran\n```\n"
    page = page_of(tmp_path, ran)
    assert check_page(page) == []                      # run as an ordinary fence, not skipped as network
    assert check_page(page, network=True) == []        # and not run as a network fence
    broken_ran = page_of(tmp_path, ran.replace("ran\n```\n", "other\n```\n"))
    assert len(check_page(broken_ran)) == 1
    colon = page_of(tmp_path, "```python\n# pseudo-code: a sketch\nnot python (\n```\n")
    assert check_page(colon) == []


@pytest.mark.parametrize("marker", ["% outputs", "% output shape", "%output", "% Output-shape"])
def test_a_mistyped_output_marker_is_reported(tmp_path, marker):
    page = page_of(tmp_path, f"```python\nprint(1)\n```\n\n{marker}\n\n```text\n1\n```\n")
    failures = check_page(page)
    assert len(failures) == 1 and failures[0].startswith("index.md:L5: mistyped marker")


@pytest.mark.parametrize("before", ["Some prose.\n", "```text\nx\n```\n"])
def test_an_output_marker_not_after_a_python_fence_is_reported(tmp_path, before):
    page = page_of(tmp_path, f"{before}\n% output\n\n```text\n1\n```\n")
    failures = check_page(page)
    assert len(failures) == 1 and "not directly after a python fence" in failures[0]


def test_a_fence_that_prints_nothing_is_reported(tmp_path):
    page = page_of(tmp_path, "```python\nx = 1 + 1\n```\n\n% output\n\n```text\n2\n```\n")
    assert check_page(page) == ["index.md:L1: prints nothing; print a key result and show it in a '% output' block"]


def test_a_fence_without_its_output_block_is_reported(tmp_path):
    page = page_of(tmp_path, "```python\nprint(2)\n```\n\nProse.\n")
    assert check_page(page) == ["index.md:L1: no '% output' block after the fence"]
    network = page_of(tmp_path, "```python\n# network\nprint(2)\n```\n\nProse.\n")
    assert check_page(network) == ["index.md:L1: no '% output' block after the fence"]


def test_a_pseudo_code_fence_that_runs_as_written_is_reported(tmp_path):
    stale = "```python\n# pseudo-code\ndef f(x):\n    return x + 1\n\n\nprint(f(1))\n```\n"
    assert check_page(page_of(tmp_path, stale)) == [
        "index.md:L1: marked '# pseudo-code' but defines and calls its own functions and runs; "
        "drop the marker and show its output"]
    for legal in ("```python\n# pseudo-code\ndef f(x):\n    return helper(x)\n\n\nf(1)\n```\n",
                  "```python\n# pseudo-code\ndef f(x):\n    ...\n```\n",
                  "```python\n# pseudo-code\nfor step in training:\n    update(step)\n```\n"):
        assert check_page(page_of(tmp_path, legal)) == []


@pytest.mark.parametrize("info", ["py", "python3", "python title=\"demo.py\"", "py3", "{code-block} python"])
def test_a_python_like_info_string_is_held_to_the_python_rules(tmp_path, info):
    """A fence tagged py, python3 or python with options is a python listing: it gets every fence rule."""
    assert [f["info"] for f in shared_python_fences(f"```{info}\nprint(2)\n```\n")] == [info]
    page = page_of(tmp_path, f"```{info}\nprint(2)\n```\n\nProse.\n")
    assert check_page(page) == ["index.md:L1: no '% output' block after the fence"]
    shown = page_of(tmp_path, f"```{info}\nprint(2)\n```\n\n% output\n\n```text\n2\n```\n")
    assert check_page(shown) == []
    silent = page_of(tmp_path, f"```{info}\nx = 2\n```\n\n% output\n\n```text\n2\n```\n")
    assert check_page(silent) == ["index.md:L1: prints nothing; print a key result and show it in a '% output' block"]


@pytest.mark.parametrize("info", ["text", "pycon", "pythonic", "json", "{code-block} text"])
def test_other_info_strings_are_not_python(info):
    assert shared_python_fences(f"```{info}\nprint(2)\n```\n") == []


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_python_fences_run_standalone(chapter_dir):
    chapter_check("fences")(chapter_dir, check_chapter(chapter_dir))


@pytest.mark.live
@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_network_fences_run_standalone(chapter_dir):
    if not any(has_network_fence(p) for p in chapter_pages(chapter_dir)):
        pytest.skip("no # network fences in this chapter")
    chapter_check("fences-live")(chapter_dir, check_chapter(chapter_dir, network=True))
