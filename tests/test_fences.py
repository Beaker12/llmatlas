"""Every ```python fence in a chapter either passes pyflakes (no undefined names,
no syntax errors) or is marked on its first line as pseudo-code, and runs standalone."""
import ast
import os
import re
import subprocess
import sys
import tempfile
from decimal import Decimal
from itertools import zip_longest
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, chapter_check, tracked

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```python\n([\s\S]*?)```")
MARKER = "# pseudo-code"


def fences():
    for path in tracked("chapters/**/*.md"):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            line = text[: m.start()].count("\n") + 2
            yield pytest.param(rel, line, m.group(1), id=f"{rel}:{line}")


@pytest.mark.parametrize("rel,line,code", list(fences()))
def test_fence_is_runnable_or_marked_pseudo_code(rel, line, code):
    if code.lstrip().startswith(MARKER):
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
# Failure strings are "<page>:L<line of the opening fence>: <reason>".

RUN_TIMEOUT = 120
NETWORK_MARKER = "# network"
NUMBER = re.compile(r"(?<!\w)-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
OUTPUT_BLOCK = re.compile(r"\s*% output(-shape)?[ \t]*\n\s*```text\n([\s\S]*?)```")
OUTPUT_MARKER = re.compile(r"\s*% output(?:-shape)?[ \t]*(?:\n|$)")


def one_unit(shown):
    """The value of one unit in the last digit of the number text `shown`."""
    m = re.fullmatch(r"-?\d*(?:\.(\d+))?(?:[eE]([-+]?\d+))?", shown)
    return Decimal(1).scaleb(int(m.group(2) or 0) - len(m.group(1) or ""))


def lines_match(shown, actual, shape):
    if shape:
        return NUMBER.sub("#", shown) == NUMBER.sub("#", actual)
    if NUMBER.sub("\0", shown) != NUMBER.sub("\0", actual):
        return False
    return all(abs(Decimal(a) - Decimal(s)) <= one_unit(s)
               for s, a in zip(NUMBER.findall(shown), NUMBER.findall(actual)))


def compare_output(stdout, block, shape):
    """None if stdout matches the shown block, else the reason it does not."""
    def clean(text):
        lines = [l.rstrip() for l in text.splitlines()]
        while lines and not lines[-1]:
            lines.pop()
        return lines
    shown, actual = clean(block), clean(stdout)
    if shape and len(shown) != len(actual):
        return f"output-shape: expected {len(shown)} lines, got {len(actual)}"
    for n, (s, a) in enumerate(zip_longest(shown, actual), start=1):
        if s is None or a is None or not lines_match(s, a, shape):
            if shape and s is not None and a is not None:
                s, a = NUMBER.sub("#", s), NUMBER.sub("#", a)
            return f"output line {n}: expected {s!r}, got {a!r}"
    return None


def run_fence(code, timeout):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory() as cwd:
        return subprocess.run(
            [sys.executable, "-W", "error", "-W", "ignore:FigureCanvasAgg is non-interactive", "-"],
            input=code, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
            timeout=timeout)


def python_fences(text):
    """(line of the opening fence, code, text after the fence) for each ```python fence."""
    for m in FENCE.finditer(text):
        yield text[: m.start()].count("\n") + 1, m.group(1), text[m.end():]


def has_network_fence(page):
    return any(code.lstrip().startswith(NETWORK_MARKER)
               for _, code, _ in python_fences(page.read_text(encoding="utf-8")))


def check_page(page, network=False, timeout=RUN_TIMEOUT):
    """Run the page's python fences (or only its `# network` ones) and list the failures."""
    failures = []
    for line, code, rest in python_fences(page.read_text(encoding="utf-8")):
        first = code.lstrip()
        if first.startswith(MARKER) or first.startswith(NETWORK_MARKER) != network:
            continue
        where = f"{page.name}:L{line}"
        try:
            proc = run_fence(code, timeout)
        except subprocess.TimeoutExpired:
            failures.append(f"{where}: timeout after {timeout}s")
            continue
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines()
            failures.append(f"{where}: exit {proc.returncode}: {err[-1][:200] if err else 'no stderr'}")
            continue
        block = OUTPUT_BLOCK.match(rest)
        if block is None:
            if OUTPUT_MARKER.match(rest):
                failures.append(f"{where}: '% output' is not followed by a ```text fence")
            continue
        reason = compare_output(proc.stdout, block.group(2), shape=bool(block.group(1)))
        if reason:
            failures.append(f"{where}: {reason}")
    return failures


def check_chapter(chapter_dir, network=False):
    failures = []
    for page in sorted(chapter_dir.glob("*.md")):
        failures += check_page(page, network=network)
    return failures


FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
SAMPLE = FIXTURE / "index.md"


def broken(tmp_path, old, new):
    """A copy of the sample page with `old` replaced by `new`."""
    text = SAMPLE.read_text(encoding="utf-8")
    assert old in text
    page = tmp_path / "index.md"
    page.write_text(text.replace(old, new, 1), encoding="utf-8")
    return page


def test_sample_chapter_passes():
    assert check_page(SAMPLE) == []
    assert check_chapter(FIXTURE) == []


def test_sample_chapter_network_fence_runs_only_when_asked():
    failures = check_page(SAMPLE, network=True)
    assert len(failures) == 1
    assert failures[0].startswith("index.md:L") and "exit 1" in failures[0] and "URLError" in failures[0]


def test_number_within_one_unit_of_the_last_digit_is_accepted(tmp_path):
    assert check_page(broken(tmp_path, "3.1416", "3.1415")) == []


def test_text_that_differs_is_reported_with_the_first_differing_line(tmp_path):
    failures = check_page(broken(tmp_path, "3.1416\ndone", "3.1416\nfinished"))
    assert failures == ["index.md:L7: output line 3: expected 'finished', got 'done'"]


def test_number_off_by_more_than_one_unit_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "3.1416", "3.1418"))
    assert failures == ["index.md:L7: output line 2: expected '3.1418', got '3.14159'"]


def test_output_shape_with_the_wrong_line_count_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "loss 4.5\n", ""))
    assert failures == ["index.md:L23: output-shape: expected 1 lines, got 2"]


def test_output_shape_with_different_text_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "loss 4.5", "cost 4.5"))
    assert failures == ["index.md:L23: output line 2: expected 'cost #', got 'loss #'"]


def test_a_fence_that_raises_is_reported_with_the_last_stderr_line(tmp_path):
    failures = check_page(broken(tmp_path, "print(7)", "raise ValueError('bad')"))
    assert failures == ["index.md:L7: exit 1: ValueError: bad"]


def test_a_fence_that_times_out_is_reported(tmp_path):
    page = tmp_path / "index.md"
    page.write_text("```python\nimport time\ntime.sleep(30)\n```\n", encoding="utf-8")
    assert check_page(page, timeout=1) == ["index.md:L1: timeout after 1s"]


def test_output_marker_without_a_text_fence_is_reported(tmp_path):
    failures = check_page(broken(tmp_path, "```text\n7\n", "```\n7\n"))
    assert len(failures) == 1 and "% output" in failures[0] and failures[0].startswith("index.md:L7:")


def test_trailing_whitespace_in_the_output_is_ignored(tmp_path):
    assert check_page(broken(tmp_path, "done\n```", "done   \n```")) == []


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_python_fences_run_standalone(chapter_dir):
    chapter_check("fences")(chapter_dir, check_chapter(chapter_dir))


@pytest.mark.live
@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_network_fences_run_standalone(chapter_dir):
    if not any(has_network_fence(p) for p in sorted(chapter_dir.glob("*.md"))):
        pytest.skip("no # network fences in this chapter")
    chapter_check("fences-live")(chapter_dir, check_chapter(chapter_dir, network=True))
