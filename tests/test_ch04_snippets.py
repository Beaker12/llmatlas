"""Chapter 4 (preprocessing): per-chapter constants read by the notebook, figure and sync tests."""

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
NOT_A_RESULT: dict[str, str] = {}

MIN_FIGURES: int = 10
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    "sec-ch4-dedup-scope": "5.3 Near-duplicate detection with MinHash and LSH",
}


# --- Tests of the chapter's code ------------------------------------------------------------------------------
import itertools  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

CH04 = Path(__file__).resolve().parent.parent / "chapters" / "04-preprocessing" / "index.md"
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def fence_containing(marker: str) -> str:
    for m in FENCE.finditer(CH04.read_text(encoding="utf-8")):
        if marker in m.group(1):
            return m.group(1)
    raise AssertionError(f"no python fence in ch04 contains {marker!r}")


def run(marker: str) -> dict:
    ns: dict = {}
    exec(fence_containing(marker), ns)  # noqa: S102 - executing the book's own example
    return ns


def test_repeated_spans_keeps_only_the_earliest_of_three_copies_in_every_ordering():
    # C4-02 (a): 36 orderings of three copies of a 5-token run; every copy but the first is removed.
    repeated_spans = run("def repeated_spans")["repeated_spans"]
    run_ = "p q r s t".split()
    for followers in itertools.permutations(["a", "m", "z"]):
        for preceders in itertools.permutations(["b", "n", "y"]):
            tokens, starts = [], []
            for f, p in zip(followers, preceders):
                tokens.append(p + "0")
                starts.append(len(tokens))
                tokens += run_ + [f + "1"]
            assert repeated_spans(tokens, 5) == [(s, s + 5) for s in starts[1:]], (followers, preceders)


def test_repeated_spans_returns_one_maximal_span_when_the_repeat_is_longer_than_min_len():
    # C4-02 (b): an 8-token repeat with min_len 4 gives one span, not five overlapping ones.
    repeated_spans = run("def repeated_spans")["repeated_spans"]
    tokens = ["x1"] + "a b c d e f g h".split() + ["x2"] + "a b c d e f g h".split() + ["x3"]
    assert repeated_spans(tokens, 4) == [(10, 18)]
