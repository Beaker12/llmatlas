"""Chapter 5 (dataset management): per-chapter constants read by the notebook, figure and sync tests, and the
numerical answers of the exercises."""
import os
import re
import subprocess
import sys

import pytest

from tests._files import ROOT

# --- Per-chapter constants -----------------------------------------------------------------------------------
# The shared tests read every entry of these through tests/_files.chapter_constants; the chapter changes them
# here and nowhere else.
# SYNC_EXEMPT       name -> reason a definition may differ between the chapter page and its notebook.
#                   Example: {"train_step": "the notebook version also logs the learning rate"}
# RESULT_CLAIMS     (chapter regex, notebook-output regex, comparator) for a number the chapter states and the
#                   notebook prints; comparator eq, gt, lt or approx:<rel>. Captured numbers are normalised, so
#                   10,000 and 1e4 are the same number. Example: (r"loss of (\d+\.\d+)", r"final loss: (\S+)", "approx:0.01")
# NOT_A_RESULT      sentence regex -> reason for a sentence that links a notebook and holds a number but states no
#                   result. Example: {r"parses two files": "a count of inputs, not a result"}
# NOTATION_OK       unit regex -> reason for a sentence the notation check's banned phrases match on purpose.
#                   Example: {r"capacity factor of the paper": "quoted from the Switch paper"}
# MIN_FIGURES       the least number of figures the chapter must have. Example: 12
# INTERFACE_LABELS  label -> text of the heading the label sits directly above, for the section labels other
#                   chapters link to. Example: {"sec-ch2-flops": "Training Complexity"}
# A NOT_A_RESULT or NOTATION_OK entry that matches nothing is a failure: remove it with the sentence it was for.
# The chapter's checker reads every exemption with `python tools/list_exemptions.py chapters/<dir>`.
SYNC_EXEMPT: dict[str, str] = {}  # name -> reason it may differ between the chapter and its notebook

# (chapter regex, notebook-output regex, comparator); each regex has exactly one capture group.
# Comparator: eq, gt, lt or approx:<rel>, applied as `chapter_value <op> notebook_value`.
RESULT_CLAIMS: list[tuple[str, str, str]] = []

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {
    r"and diffing two versions": "names the two inputs of the diff (an old and a new version), not a measured result",
    r"checks resumption at a hundred random steps": "the number of resumption points the notebook tests, an "
                                                    "experiment setting it asserts over, not a result",
    r"simulates shard assignment for one, two, four and eight ranks": "the simulated world sizes and shard count, experiment settings, not results",
}

MIN_FIGURES: int = 13
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {}

# --- Exercise answers ----------------------------------------------------------------------------------------
# Each exercise's solution listing is run from the page, and the numbers its answer states are checked against
# what the listing prints. Solution 1's timings and compressed size depend on the machine; its exact numbers are
# the token count and the binaries' bytes per token.
PAGE = ROOT / "chapters" / "05-dataset-management" / "index.md"
SOLUTION = re.compile(r"^`{4}\{admonition\} Solution (\d+)\n(.*?)^`{4}$", re.S | re.M)
LISTING = re.compile(r"^```python\n(.*?)^```$", re.S | re.M)
SHOWN = re.compile(r"^```\n\n% output\n\n```text\n(.*?)^```$", re.S | re.M)  # exact output after the listing
ANSWERS = {
    1: [r"608,751 tokens in 2,000 documents", r"uint16 binary +2\.00 bytes per token", r"uint32 binary +4\.00 bytes"],
    2: [r"kept 850 at 0\.30 and 838 at 0\.27: 12 documents removed", r"count layout: 16 of 17 shards changed",
        r"hash  layout: 12 of 64 shards changed"],
    3: [r"mean    100: chunking splits 2\.4% of documents; best-fit pads 0\.04% and splits 0\.0%",
        r"mean  1,000: chunking splits 23\.2% of documents; best-fit pads 0\.03% and splits 1\.6%",
        r"mean 10,000: chunking splits 84\.0% of documents; best-fit pads 0\.48% and splits 66\.5%"],
    4: [r"removes 62\.0% of the causal pairs", r"changes the context of 75\.6% of the tokens",
        r"seed 0, gap to document mask ", r"seed 1, gap to mask, EOD opens "],
    5: [r"agrees with iterating at 1000 random steps", r"step 5,000: \{'code': 750, 'web': 3500, 'wiki': 750\}",
        r"step 12,345: \{'code': 2586, 'web': 7173, 'wiki': 2586\}"],
    6: [r"global indexing : 0 samples seen twice, 0 never seen", r"shard assignment: 16,000 samples seen twice, 16,000"],
    7: [r"without the index, 2,900 documents", r"differs from v1\.2 in \['s1'\]",
        r"\['model-a', 'model-b', 'model-c'\]"],
}
# Numbers the prose of each answer states, which must also appear in the listing's printed output
STATED = {2: ["16 of the 17", "12 of the 64"], 3: ["2.4%", "23.2%", "1.6%", "0.03%", "84.0%", "66.5%"],
          4: ["62.0%", "75.6%"], 5: ["7,173", "2,586"], 6: ["16,000"], 7: ["2,900"]}


def solutions() -> dict:
    return {int(n): body for n, body in SOLUTION.findall(PAGE.read_text(encoding="utf-8"))}


def test_every_exercise_has_its_answers_listed():
    assert sorted(solutions()) == sorted(ANSWERS) == list(range(1, 8))


@pytest.mark.parametrize("n", sorted(ANSWERS))
def test_exercise_answer(n):
    body = solutions()[n]
    listings = LISTING.findall(body)
    assert len(listings) == 1, f"solution {n} should hold one listing"
    shown = SHOWN.search(body)
    if shown:
        # An exact `% output` block: tests/test_fences.py runs this listing and compares its output with the block,
        # so the answers are checked against the block instead of running the listing a second time.
        run = subprocess.CompletedProcess([], 0, stdout=shown.group(1), stderr="")
    else:
        run = subprocess.run([sys.executable, "-W", "error", "-c", listings[0]], capture_output=True, text=True,
                             timeout=600, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stderr[-2000:]
    for pattern in ANSWERS[n]:
        assert re.search(pattern, run.stdout), f"solution {n}: {pattern!r} not in\n{run.stdout}"
    prose = LISTING.sub("", body)
    for number in STATED.get(n, []):
        assert number in prose, f"solution {n}'s answer no longer states {number}"
        printed = number.replace(" of the ", " of ")
        assert printed in run.stdout or printed.replace(",", "") in run.stdout, f"solution {n} states {number} but the listing prints\n{run.stdout}"


# --- Committed notebook keeps its plots ------------------------------------------------------------------------
# The notebook is published without execution (_config.yml: execute_notebooks "off"), so its plots exist only as
# committed outputs. Executing it with a non-interactive MPLBACKEND (Agg) silently drops them.
NOTEBOOK = ROOT / "chapters" / "05-dataset-management" / "examples" / "dataset_management_demo.ipynb"
MIN_NOTEBOOK_IMAGES = 3  # packing histogram, document mask, temperature


def test_committed_notebook_has_its_plots():
    import json

    cells = [c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    has_image = [any("image/png" in o.get("data", {}) for o in c.get("outputs", [])) for c in cells]
    assert sum(has_image) >= MIN_NOTEBOOK_IMAGES, "the committed notebook lost its plots; re-execute without MPLBACKEND"
    tagged = [(t, img) for c, img in zip(cells, has_image) for t in c["metadata"].get("tags", []) if t.startswith("figure:")]
    assert len(tagged) >= MIN_NOTEBOOK_IMAGES
    missing = [t for t, img in tagged if not img]
    assert not missing, f"figure cells without an image output: {missing}"
