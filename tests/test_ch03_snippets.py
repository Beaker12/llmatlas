"""Chapter 3 (data acquisition): per-chapter constants read by the notebook, figure and sync tests."""

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
SYNC_EXEMPT: dict[str, str] = {  # name -> reason it may differ between the chapter and its notebook
    "fetch": "the notebook version also retries once, can send HEAD, and records every fetch (URL, size, hash, "
             "bytes) for the provenance manifest of its last section; the page shows only the GET it needs",
}

# (chapter regex, notebook-output regex, comparator); each regex has exactly one capture group.
# Comparator: eq, gt, lt or approx:<rel>, applied as `chapter_value <op> notebook_value`.
RESULT_CLAIMS: list[tuple[str, str, str]] = [
    (r"\\approx (20\.8)\$T tokens", r"D = (\d+\.\d+)T tokens", "approx:0.01"),
]

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {}

MIN_FIGURES: int = 10
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    "sec-ch3-epochs": "1.3 How much is enough",
}


# --- Tests -----------------------------------------------------------------------------------------------------
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

FIGURE_SCRIPT = Path(__file__).resolve().parents[1] / "chapters" / "03-data-acquisition" / "figures" / "make_figures.py"


PAGE = FIGURE_SCRIPT.parents[1] / "index.md"

# Exercise 1: Content-Length of ten WARC files of CC-MAIN-2026-39 drawn with random.Random(0).sample(paths, 10),
# measured by HEAD request on 12 November 2026; the published WARC total is 82.79 TiB.
WARC_SIZES = [910_707_757, 966_719_339, 903_957_552, 915_750_155, 911_469_743,
              933_101_733, 896_086_136, 935_203_619, 918_424_076, 939_744_129]


def test_exercise_1_snapshot_size():
    import re
    import statistics

    text = PAGE.read_text()
    shown = [int(n.replace(",", "")) for n in re.findall(r"\b\d{3},\d{3},\d{3}\b", text.split("Solution 1")[1].split("Solution 2")[0])]
    assert shown[:10] == WARC_SIZES and shown[10] == 923_116_424   # the ten sizes, then their mean
    mean, sd = statistics.mean(WARC_SIZES), statistics.stdev(WARC_SIZES)
    total = 100_000 * mean
    assert round(mean) == 923_116_424
    assert round(sd / 1e6, 1) == 20.8
    assert round(total / 1e12, 1) == 92.3 and round(total / 2**40, 2) == 83.96
    assert round(sd / 10**0.5 / 1e6, 1) == 6.6 and round(sd / 10**0.5 / mean * 100, 2) == 0.71
    assert round((total / 2**40 / 82.79 - 1) * 100, 1) == 1.4
    assert 1.5 < (total / 2**40 - 82.79) / (sd / 10**0.5 * 100_000 / 2**40) < 2.5      # about two standard errors
    assert round(sd / 100**0.5 / mean * 100, 1) == 0.2                                 # a hundred files
    assert round(82.79 * 2**40 / 1e12, 1) == 91.0


def _plan_mixture():
    import re

    block = next(b for b in re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S) if "def plan_mixture" in b)
    namespace: dict = {}
    exec(block.split("plan = plan_mixture")[0], namespace)
    return namespace["plan_mixture"], namespace["sources"]


def test_exercise_5_budget_sweep():
    plan_mixture, sources = _plan_mixture()

    def capped(budget):
        return {s for s, (_, _, e) in plan_mixture(budget, sources).items() if e >= 4 - 1e-9}

    def first_capped(name, lo=0.5e12, hi=20e12):
        if name not in capped(hi):
            return None
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if name in capped(mid) else (mid, hi)
        return hi

    assert capped(0.5e12) == set()
    assert round(first_capped("wikipedia") / 1e12, 3) == 0.867
    assert round(first_capped("books") / 1e12, 3) == 3.906
    assert round(first_capped("papers") / 1e12, 2) == 14.25
    assert first_capped("web") is None and first_capped("code") is None
    plan = plan_mixture(20e12, sources)
    shown = {"web": (0.877, 17.54, 1.46), "code": (0.086, 1.71, 1.71), "papers": (0.030, 0.60, 4.00),
             "wikipedia": (0.0013, 0.026, 4.00), "books": (0.006, 0.12, 4.00)}
    for s, (w, t, e) in shown.items():
        digits = len(str(w).split(".")[1])
        assert round(plan[s][0], digits) == w and round(plan[s][1] / 1e12, len(str(t).split(".")[1])) == t
        assert round(plan[s][2], 2) == e
    small = sum(plan[s][0] for s in ("papers", "wikipedia", "books"))
    assert round(small * 100, 1) == 3.7


def test_exercise_6_wilson_interval():
    import math

    n, k, z = 50, 12, 1.96
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert round(p * 100) == 24
    assert (round((center - half) * 100, 1), round((center + half) * 100, 1)) == (14.3, 37.4)


@pytest.mark.slow
def test_make_figures_twice_gives_identical_files(tmp_path):
    """Two runs of the chapter's figure script write byte-identical SVGs (no dates, fixed hash salt)."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    outputs = []
    for run in ("first", "second"):
        out = tmp_path / run
        subprocess.run([sys.executable, str(FIGURE_SCRIPT), str(out)], check=True, capture_output=True, env=env)
        outputs.append({p.name: p.read_bytes() for p in sorted(out.glob("*.svg"))})
    assert len(outputs[0]) == 5, sorted(outputs[0])
    assert outputs[0] == outputs[1]
