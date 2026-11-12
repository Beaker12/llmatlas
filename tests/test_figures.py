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
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests._files import CHAPTER_DIRS, CHAPTER_IDS, ROOT, chapter_check, chapter_constants
from tests._pages import blank_code, line_of, pages

LABEL = re.compile(r"fig-ch\d+-[a-z0-9]+(?:-[a-z0-9]+)*")
FIGURE = re.compile(r"^([ \t]*)(`{3,}|:{3,})\{figure\}[ \t]+(\S+)[ \t]*\n((?:[ \t]*:[\w-]+:.*\n)*)([\s\S]*?)^\1\2[ \t]*$", re.M)
NUMREF = re.compile(r"\{numref\}`(?:[^`<]*<)?([^`>]+?)>?`")
PYTHON_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
PLOTS = re.compile(r"plt\.(?:show|savefig)\(|\.savefig\(|\b(?:plt|ax|axs|axes|ax\d+)(?:\[[^\]]*\])?\.(?:plot|imshow)\(")
AFTER_FENCE = re.compile(r"\s*(?:%[ \t]*output(?:-shape)?[ \t]*\n\s*(`{3,})text[ \t]*\n[\s\S]*?\n\1[ \t]*\n)?\s*(?:`{3,}|:{3,})\{figure\}")
SOURCE_SUFFIXES = {".dot", ".py", ".json"}


def parse_figures(text):
    """[(line, target, options dict, caption)] for the figure directives of `text`."""
    figures = []
    for m in FIGURE.finditer(blank_code(text)):
        options = {}
        for line in m.group(4).splitlines():
            key, _, value = line.strip()[1:].partition(":")
            options[key] = value.strip()
        figures.append((line_of(text, m.start()), m.group(3), options, m.group(5).strip()))
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
        for m in PYTHON_FENCE.finditer(text):
            body = m.group(1)
            if body.lstrip().startswith("# pseudo-code") or not PLOTS.search(body):
                continue
            if not AFTER_FENCE.match(text, m.end()):
                failures.append(f"{page.name}:L{line_of(text, m.start())}: plotting fence is not followed by a {{figure}}")
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
    if count < min_figures:
        failures.append(f"index.md:L1: {count} figures, at least {min_figures} required")
    return failures


def regeneration_failures(chapter_dir, out_dir):
    script = Path(chapter_dir) / "figures" / "make_figures.py"
    if not script.exists():
        return []
    env = {**os.environ, "MPLBACKEND": "Agg"}
    done = subprocess.run([sys.executable, str(script), str(out_dir)], capture_output=True, text=True, env=env)
    if done.returncode:
        return [f"figures/make_figures.py: exit {done.returncode}: {done.stderr.strip().splitlines()[-1:]}"]
    failures = []
    for svg in sorted(Path(out_dir).glob("*.svg")):
        committed = Path(chapter_dir) / "figures" / svg.name
        if not committed.exists():
            failures.append(f"figures/{svg.name}: written by make_figures.py but not committed")
        elif committed.read_bytes() != svg.read_bytes():
            failures.append(f"figures/{svg.name}: differs from the committed file; run make_figures.py and commit")
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


@pytest.mark.slow
def test_the_sample_chapter_regenerates_byte_for_byte(tmp_path):
    assert regeneration_failures(FIXTURE, tmp_path / "out") == []


@pytest.mark.slow
def test_a_plotting_fence_edited_without_regenerating_is_reported(tmp_path):
    chapter = copy(tmp_path, "plt.plot([1, 2, 3])", "plt.plot([1, 2, 5])")
    only(regeneration_failures(chapter, tmp_path / "out"), "figures/ramp.svg: differs from the committed file")
