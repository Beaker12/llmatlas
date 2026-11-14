"""Figures: labels, alt text, captions, files, references, plotting fences, count, sources, regeneration.

`figures` (one test per chapter, fast). For every `{figure}` directive on a page of the chapter:
  - a `:name:` (or `:label:`) of the form fig-chN-<slug>, a non-empty `:alt:`, and a caption;
  - the target is `figures/<name>.*` and both figures/<name>.svg and figures/<name>.pdf exist.
Then, for the chapter:
  - every figure label is used by at least one {numref};
  - every file in figures/ other than sources (.dot, .py, .json, SOURCES.sha256, anything under figures/data/) is
    referenced by a directive;
  - every `python` fence that plots (plt.show, plt.savefig, .savefig, or .plot/.imshow on plt/ax/axs/axes) is
    followed, after its optional `% output` block, by a `{figure}` directive (`# pseudo-code` fences excepted);
  - the number of figures is at least the chapter's MIN_FIGURES;
  - every .dot file's SHA-256 equals its `<hash>  <file>` line in figures/SOURCES.sha256.
Slow, one test per chapter: if figures/make_figures.py exists, run it with a temporary output directory and assert
each SVG it writes is byte-identical to the committed one. A chapter without the script passes trivially; with the
script this test is asserted directly (not through .review-pending), because a stale figure is never pending.
"""
import ast
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, chapter_constants
from tests._pages import blank_code, fences, line_of, pages

LABEL = re.compile(r"fig-ch\d+-[a-z0-9]+(?:-[a-z0-9]+)*")
FIGURE = re.compile(r"^([ \t]*)(`{3,}|:{3,})\{figure\}[ \t]+(\S+)[ \t]*\n((?:[ \t]*:[\w-]+:.*\n)*)([\s\S]*?)^\1\2[ \t]*$", re.M)
NUMREF = re.compile(r"\{numref\}`(?:[^`<]*<)?([^`>]+?)>?`")
TARGET_LINE = re.compile(r"\(([^)\s]+)\)=[ \t]*")
OUTPUT_MARK = re.compile(r"\s*%[ \t]*output(?:-shape)?[ \t]*\n\s*(`{3,})text[ \t]*\n")
FIGURE_START = re.compile(r"\s*(?:\([^)\s]+\)=[ \t]*\n\s*)?(?:`{3,}|:{3,})\{figure\}")
PLOT_ATTRS = {"savefig", "plot", "imshow", "bar", "hist", "scatter"}


def _plot_call(node):
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return False
    attr, base = node.func.attr, node.func.value
    return attr in PLOT_ATTRS or (attr == "show" and isinstance(base, ast.Name) and base.id in ("plt", "pyplot"))


def plots(source):
    """True when a plotting call is reachable from module level, directly or in a function the fence calls there."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    top = [n for n in tree.body if n not in functions.values()]
    seen, queue = set(), list(top)
    while queue:
        node = queue.pop()
        for sub in ast.walk(node):
            if _plot_call(sub):
                return True
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in functions \
                    and sub.func.id not in seen:
                seen.add(sub.func.id)
                queue.append(functions[sub.func.id])
    return False


def followed_by_figure(text, pos):
    """True when, after optional blank lines and an optional `% output` block, a {figure} directive opens."""
    m = OUTPUT_MARK.match(text, pos)
    if m:
        closing = re.compile(rf"^{re.escape(m.group(1))}[ \t]*$", re.M).search(text, m.end())
        if not closing:
            return False
        pos = closing.end()
    return bool(FIGURE_START.match(text, pos))
SOURCE_SUFFIXES = {".dot", ".py", ".json"}


def target_above(shown, start, indent):
    """The label of a `(fig-chN-slug)=` target line above the figure at offset `start`: the previous non-blank
    line, past any blank lines, written at the figure's own indentation (so a figure inside an admonition works)."""
    before = shown[:start].split("\n")[:-1]
    k = len(before) - 1
    while k >= 0 and not before[k].strip():
        k -= 1
    if k < 0 or not before[k].startswith(indent) or before[k][len(indent):len(indent) + 1] in (" ", "\t"):
        return None
    m = TARGET_LINE.fullmatch(before[k][len(indent):])
    return m.group(1) if m else None


def parse_figures(text):
    """[(line, target, options dict, caption)] for the figure directives of `text`."""
    figures = []
    shown = blank_code(text)     # offsets below are into `shown`, which keeps every line number
    for m in FIGURE.finditer(shown):
        options = {}
        for line in m.group(4).splitlines():
            key, _, value = line.strip()[1:].partition(":")
            options[key] = value.strip()
        above = target_above(shown, m.start(), m.group(1))    # a `(fig-chN-slug)=` target line
        if above and not (options.get("name") or options.get("label")):
            options["name"] = above
        figures.append((line_of(shown, m.start()), m.group(3), options, m.group(5).strip()))
    return figures


def figure_failures(chapter_dir, min_figures):
    chapter_dir = Path(chapter_dir)
    figures_dir = chapter_dir / "figures"
    failures, labels, stems, numrefs, count = [], {}, set(), set(), 0
    for page in pages(chapter_dir):
        text = page.read_text(encoding="utf-8")
        numrefs |= set(NUMREF.findall(blank_code(text)))
        for line, target, options, caption in parse_figures(text):
            count += 1
            where = f"{page.name}:L{line}"
            label = options.get("name") or options.get("label") or ""
            if not LABEL.fullmatch(label):
                failures.append(f"{where}: figure label {label!r} is not of the form fig-chN-<slug>")
            else:
                labels[label] = where
            if not options.get("alt"):
                failures.append(f"{where}: figure {label or target!r} has no non-empty :alt:")
            if not caption:
                failures.append(f"{where}: figure {label or target!r} has no caption")
            m = re.fullmatch(r"figures/([\w.-]+)\.\*", target)
            if not m:
                failures.append(f"{where}: figure target {target!r} is not of the form figures/<name>.*")
                continue
            stems.add(m.group(1))
            for suffix in (".svg", ".pdf"):
                if not (figures_dir / (m.group(1) + suffix)).exists():
                    failures.append(f"{where}: figures/{m.group(1)}{suffix} does not exist")
        for fence in fences(text):
            if fence["info"] != "python" or fence["body"].lstrip().startswith("# pseudo-code") or not plots(fence["body"]):
                continue
            if not followed_by_figure(text, fence["end"]):
                failures.append(f"{page.name}:L{fence['line']}: plotting fence is not followed by a {{figure}}")
    for label, where in labels.items():
        if label not in numrefs:
            failures.append(f"{where}: figure {label} is never referenced by {{numref}}")
    if figures_dir.is_dir():
        for path in sorted(figures_dir.rglob("*")):
            relative = path.relative_to(figures_dir)
            if (not path.is_file() or path.suffix in SOURCE_SUFFIXES or path.name == "SOURCES.sha256"
                    or relative.parts[0] == "data"):
                continue
            if len(relative.parts) > 1 or path.stem not in stems:
                failures.append(f"figures/{relative.as_posix()}: not referenced by any {{figure}} directive")
        recorded = {}
        sources = figures_dir / "SOURCES.sha256"
        if sources.exists():
            for line in sources.read_text(encoding="utf-8").splitlines():
                digest, _, name = line.partition("  ")
                if name:
                    recorded[name.strip()] = digest.strip()
        for dot in sorted(figures_dir.glob("*.dot")):
            actual = hashlib.sha256(dot.read_bytes()).hexdigest()
            if dot.name not in recorded:
                failures.append(f"figures/{dot.name}: no line in figures/SOURCES.sha256")
            elif recorded[dot.name] != actual:
                failures.append(f"figures/{dot.name}: hash differs from figures/SOURCES.sha256 (re-render and record the new hash)")
            for suffix in (".svg", ".pdf"):
                if not dot.with_suffix(suffix).exists():
                    failures.append(f"figures/{dot.stem}{suffix}: missing, but figures/{dot.name} is its source")
        script = figures_dir / "make_figures.py"
        if script.exists():
            code = script.read_text(encoding="utf-8")
            for dot in sorted(figures_dir.glob("*.dot")):
                if re.search(rf"""["']{re.escape(dot.stem)}(?:\.(?:svg|pdf))?["']""", code):
                    failures.append(f"figures/{dot.stem}: has a .dot source but make_figures.py also writes it (schematics are rendered from Graphviz)")
    if count < min_figures:
        failures.append(f"index.md:L1: {count} figures, at least {min_figures} required")
    return failures


def regeneration_failures(chapter_dir, out_dir):
    figures_dir = Path(chapter_dir) / "figures"
    script = figures_dir / "make_figures.py"
    written, failures = set(), []
    if script.exists():
        env = {**os.environ, "MPLBACKEND": "Agg"}
        done = subprocess.run([sys.executable, str(script), str(out_dir)], capture_output=True, text=True, env=env)
        if done.returncode:
            return [f"figures/make_figures.py: exit {done.returncode}: {done.stderr.strip().splitlines()[-1:]}"]
        for svg in sorted(Path(out_dir).glob("*.svg")):
            written.add(svg.name)
            if svg.with_suffix(".dot").name in {d.name for d in figures_dir.glob("*.dot")}:
                failures.append(f"figures/{svg.name}: has a .dot source but make_figures.py also writes it")
            committed = figures_dir / svg.name
            if not committed.exists():
                failures.append(f"figures/{svg.name}: written by make_figures.py but not committed")
            elif committed.read_bytes() != svg.read_bytes():
                failures.append(f"figures/{svg.name}: differs from the committed file; run make_figures.py and commit")
    for svg in sorted(figures_dir.glob("*.svg")):
        if svg.name not in written and not svg.with_suffix(".dot").exists():
            failures.append(f"figures/{svg.name}: no .dot source and not written by make_figures.py (orphan plot)")
    return failures


@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_figures(chapter_dir):
    chapter_check("figures")(chapter_dir, figure_failures(chapter_dir, chapter_constants(chapter_dir).MIN_FIGURES))


@pytest.mark.slow
@pytest.mark.parametrize("chapter_dir", CHAPTER_DIRS, ids=CHAPTER_IDS)
def test_figures_regenerate_byte_for_byte(chapter_dir, tmp_path):
    failures = regeneration_failures(chapter_dir, tmp_path / "out")
    assert not failures, "\n".join(failures)


# ---- the checking functions on the sample chapter and broken copies -----------------------------------------------

FIXTURE = ROOT / "tests" / "fixtures" / "sample-chapter"
RAMP = ("```{figure} figures/ramp.*\n:name: fig-ch0-ramp\n:alt: A straight line rising from one to three.\n\n"
        "A ramp, drawn by the fence above and written by `figures/make_figures.py`.\n```\n")


def copy(tmp_path, old=None, new=""):
    chapter = tmp_path / "chapter"
    shutil.copytree(FIXTURE, chapter)
    if old is not None:
        text = (chapter / "index.md").read_text(encoding="utf-8")
        assert old in text, old
        (chapter / "index.md").write_text(text.replace(old, new, 1), encoding="utf-8")
    return chapter


def only(failures, fragment):
    assert len(failures) == 1 and fragment in failures[0], failures


def test_the_sample_chapter_passes():
    assert figure_failures(FIXTURE, 2) == []


def test_too_few_figures_are_reported():
    only(figure_failures(FIXTURE, 3), "2 figures, at least 3 required")


def test_a_missing_alt_is_reported(tmp_path):
    only(figure_failures(copy(tmp_path, ":alt: A straight line rising from one to three.\n"), 2), "has no non-empty :alt:")


def test_a_missing_caption_is_reported(tmp_path):
    chapter = copy(tmp_path, "A ramp, drawn by the fence above and written by `figures/make_figures.py`.\n")
    only(figure_failures(chapter, 2), "has no caption")


def test_a_bad_label_is_reported(tmp_path):
    only(figure_failures(copy(tmp_path, ":name: fig-ch0-ramp", ":name: ramp"), 2), "'ramp' is not of the form fig-chN-<slug>")


def test_a_figure_without_numref_is_reported(tmp_path):
    only(figure_failures(copy(tmp_path, "Figure {numref}`fig-ch0-ramp` shows the ramp. "), 2),
         "fig-ch0-ramp is never referenced by {numref}")


def test_a_missing_pdf_is_reported(tmp_path):
    chapter = copy(tmp_path)
    (chapter / "figures" / "ramp.pdf").unlink()
    only(figure_failures(chapter, 2), "figures/ramp.pdf does not exist")


def test_a_target_without_wildcard_is_reported(tmp_path):
    failures = figure_failures(copy(tmp_path, "figures/ramp.*", "figures/ramp.svg"), 2)
    assert "is not of the form figures/<name>.*" in failures[0]      # the files are then reported as unreferenced
    assert failures[1:] == ["figures/ramp.pdf: not referenced by any {figure} directive",
                            "figures/ramp.svg: not referenced by any {figure} directive"]


def test_an_unreferenced_figure_file_is_reported(tmp_path):
    chapter = copy(tmp_path)
    (chapter / "figures" / "orphan.svg").write_text("<svg/>", encoding="utf-8")
    (chapter / "figures" / "data").mkdir()
    (chapter / "figures" / "data" / "points.csv").write_text("x\n1\n", encoding="utf-8")
    only(figure_failures(chapter, 2), "figures/orphan.svg: not referenced")


def test_a_plotting_fence_without_a_figure_is_reported(tmp_path):
    extra = "\n```python\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot([1, 2])\n```\n"
    chapter = copy(tmp_path, "## 3. Shared function\n", "## 3. Shared function\n" + extra)
    only(figure_failures(chapter, 2), "plotting fence is not followed by a {figure}")
    text = (chapter / "index.md").read_text(encoding="utf-8").replace(
        "ax.plot([1, 2])\n```\n", "ax.plot([1, 2])\n```\n\n% output\n\n```text\nnothing\n```\n\n" + RAMP, 1)
    (chapter / "index.md").write_text(text, encoding="utf-8")
    assert not any("plotting fence" in f for f in figure_failures(chapter, 2))


def test_a_changed_dot_without_a_new_hash_is_reported(tmp_path):
    chapter = copy(tmp_path)
    with open(chapter / "figures" / "schematic.dot", "a", encoding="utf-8") as f:
        f.write("// edited\n")
    only(figure_failures(chapter, 2), "figures/schematic.dot: hash differs")
    (chapter / "figures" / "SOURCES.sha256").unlink()
    only(figure_failures(chapter, 2), "no line in figures/SOURCES.sha256")


TARGET_FIGURE = ("(fig-ch0-ramp)=\n\n```{figure} figures/ramp.*\n:alt: A straight line rising from one to three.\n\n"
                 "A ramp, drawn by the fence above and written by `figures/make_figures.py`.\n```\n")


def test_a_label_may_be_a_myst_target_line_above_the_directive(tmp_path):
    chapter = copy(tmp_path, RAMP, TARGET_FIGURE)
    assert figure_failures(chapter, 2) == []
    # still checked against the {numref} uses and the slug form
    text = (chapter / "index.md").read_text(encoding="utf-8")
    (chapter / "index.md").write_text(text.replace("(fig-ch0-ramp)=", "(ramp)="), encoding="utf-8")
    assert any("'ramp' is not of the form" in f for f in figure_failures(chapter, 2))
    (chapter / "index.md").write_text(text.replace("Figure {numref}`fig-ch0-ramp` shows the ramp. ", ""), encoding="utf-8")
    only(figure_failures(chapter, 2), "fig-ch0-ramp is never referenced by {numref}")


def test_a_target_line_is_found_past_blank_lines_and_inside_an_admonition():
    figure = "```{figure} figures/ramp.*\n:alt: A.\n\nCaption.\n```\n"
    indented = "".join(f"  {line}\n" if line else "\n" for line in figure.rstrip("\n").split("\n"))
    inside = ("::::{note}\n" + "A long admonition sentence that pushes the target far back. " * 5 + "\n\n"
              "  (fig-ch0-ramp)=\n" + "\n" * 3 + indented + "::::\n")
    assert [f[2].get("name") for f in parse_figures(inside)] == ["fig-ch0-ramp"]
    far = "(fig-ch0-ramp)=\n" + "   \n" * 120 + figure
    assert [f[2].get("name") for f in parse_figures(far)] == ["fig-ch0-ramp"]
    # Not a target of this figure: a prose line in between, or a different indentation.
    assert [f[2].get("name") for f in parse_figures("(fig-ch0-ramp)=\n\nSome prose.\n\n" + figure)] == [None]
    assert [f[2].get("name") for f in parse_figures("  (fig-ch0-ramp)=\n\n" + figure)] == [None]
    assert [f[2].get("name") for f in parse_figures("(fig-ch0-ramp)=\n\n" + indented)] == [None]


def test_a_directive_without_name_or_target_line_has_an_empty_label(tmp_path):
    chapter = copy(tmp_path, ":name: fig-ch0-ramp\n")
    assert any("figure label '' is not of the form" in f for f in figure_failures(chapter, 2))


def test_a_fence_that_only_defines_a_plotting_function_is_not_a_plotting_fence():
    defines = "import matplotlib.pyplot as plt\n\ndef draw(xs):\n    plt.plot(xs)\n    plt.show()\n"
    assert not plots(defines)
    assert plots(defines + "\ndraw([1, 2])\n")
    assert plots("import matplotlib.pyplot as plt\nplt.show()\n")
    assert plots("fig, ax = plt.subplots()\nax.bar([1], [2])\n")
    assert not plots("print('plot')\nx.append(1)\n") and not plots("def broken(:\n")


def test_four_backtick_fences_and_fences_in_admonitions_are_checked(tmp_path):
    body = "import matplotlib.pyplot as plt\nplt.show()\n"
    page = f"````python\n{body}````\n\n```{{admonition}} X\n```python\n{body}```\n```\n"
    chapter = copy(tmp_path, "## 3. Shared function\n", "## 3. Shared function\n\n" + page)
    failures = [f for f in figure_failures(chapter, 2) if "plotting fence" in f]
    assert len(failures) == 2


def test_the_check_is_fast_on_the_longest_chapter():
    import time
    longest = max(CHAPTER_DIRS, key=lambda d: sum(len(p.read_text(encoding="utf-8")) for p in pages(d)))
    start = time.perf_counter()
    figure_failures(longest, 0)
    assert time.perf_counter() - start < 0.5


def test_an_orphan_svg_is_reported_by_the_regeneration_check(tmp_path):
    chapter = copy(tmp_path)
    (chapter / "figures" / "orphan.svg").write_text("<svg/>", encoding="utf-8")
    only(regeneration_failures(chapter, tmp_path / "out"), "figures/orphan.svg: no .dot source and not written")


@pytest.mark.slow
def test_the_sample_chapter_regenerates_byte_for_byte(tmp_path):
    assert regeneration_failures(FIXTURE, tmp_path / "out") == []


@pytest.mark.slow
def test_a_plotting_fence_edited_without_regenerating_is_reported(tmp_path):
    chapter = copy(tmp_path, "plt.plot([1, 2, 3])", "plt.plot([1, 2, 5])")
    only(regeneration_failures(chapter, tmp_path / "out"), "figures/ramp.svg: differs from the committed file")


def _with_dot(tmp_path, name, pdf):
    chapter = copy(tmp_path)
    figures = chapter / "figures"
    (figures / f"{name}.dot").write_text("digraph g { a -> b }\n", encoding="utf-8")
    digest = hashlib.sha256((figures / f"{name}.dot").read_bytes()).hexdigest()
    (figures / "SOURCES.sha256").write_text(f"{digest}  {name}.dot\n", encoding="utf-8")
    (figures / f"{name}.svg").write_text("<svg/>", encoding="utf-8")
    if pdf:
        (figures / f"{name}.pdf").write_bytes(b"%PDF")
    return chapter


def test_a_figure_with_a_dot_source_and_a_script_is_reported(tmp_path):
    chapter = _with_dot(tmp_path, "ramp", pdf=True)
    failures = figure_failures(chapter, 2)
    assert any("make_figures.py also writes it" in f for f in failures), failures


def test_a_dot_without_its_pdf_is_reported(tmp_path):
    chapter = _with_dot(tmp_path, "flow", pdf=False)
    failures = figure_failures(chapter, 2)
    assert any("flow.pdf: missing" in f for f in failures), failures
