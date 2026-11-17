"""Chapter 7's per-chapter constants, and checks of the numbers its prose states.

The shared tests read the constants below through tests/_files.chapter_constants. The exercise answers are checked
against the output blocks, which tests/test_fences.py compares with what each listing prints.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "chapters" / "07-training-objectives" / "index.md"

# --- Per-chapter constants (see tests/test_ch06_snippets.py for the meaning of each) ------------------------
SYNC_EXEMPT: dict[str, str] = {}
RESULT_CLAIMS: list[tuple[str, str, str]] = []
NOT_A_RESULT: dict[str, str] = {}
MIN_FIGURES: int = 9
NOTATION_OK: dict[str, str] = {}
INTERFACE_LABELS: dict[str, str] = {}

OUTPUT = re.compile(r"```python\n([\s\S]*?)```\s*% output\s*```text\n([\s\S]*?)```")


def output_of(marker: str) -> str:
    """The output block shown after the fence that contains `marker`."""
    for code, shown in OUTPUT.findall(CHAPTER.read_text(encoding="utf-8")):
        if marker in code:
            return shown
    raise AssertionError(f"no fence with an output block contains {marker!r}")


def text() -> str:
    return CHAPTER.read_text(encoding="utf-8")


def test_prose_states_the_accumulation_counts_the_listing_prints():
    assert "[64, 40, 16, 6]" in output_of("mean_of_means")
    assert "64, 40, 16 and 6 loss-bearing tokens" in text()


def test_span_corruption_lengths_in_prose_match_the_listing():
    shown = output_of("restored = []")
    assert "input 461 tokens, target 104 tokens (77 original + 27 sentinels)" in shown
    assert "the target holds the 77 removed tokens, 26 sentinels" in text()


def test_adam_jump_numbers_in_prose_match_the_listing():
    shown = output_of("def adam_steps")
    assert "largest step 6.12, steps above 1.5 after the jump: 100" in shown
    assert "largest step 1.10" in shown
    assert "the largest is about 6; with $\\beta_2 = 0.95$ the largest is 1.1" in text()


def test_fit_comparison_in_prose_matches_the_listing():
    shown = output_of("FITS = {")
    hoffmann, besiroglu = shown.split("Besiroglu et al.")
    assert "N =   32.5B, D =  3.02T" in hoffmann and "N =   73.0B, D =  1.34T" in besiroglu
    # "under half Chinchilla's size trained on more than twice its tokens"; "near Chinchilla's 70B and 1.4T"
    assert 32.5 < 70 / 2 and 3.02 > 2 * 1.4


def test_overtraining_numbers_in_prose_match_the_listing():
    shown = output_of("brentq")
    assert "(5.5x less), N = 34B" in shown and "1,868 tokens per parameter" in shown
    assert "about $1.31 \\times 10^{23}$" not in text()  # the exercise states the number exactly
    assert "$1.31 \\times 10^{23}$ training FLOPs instead of $7.23 \\times 10^{23}$" in text()


def test_budget_numbers_in_prose_match_the_listing():
    shown = output_of("PEAK = 989e12")
    days = [int(d) for d in re.findall(r"(\d+) days", shown)]
    assert all(55 <= d <= 75 for d in days)  # "about two months"
    assert "35% of the bf16 peak" in shown
