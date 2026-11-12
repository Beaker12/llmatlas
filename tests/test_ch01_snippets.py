"""Execute selected code fences from chapter 01 and check what they define."""
import math
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

FENCE = re.compile(r"```python\n([\s\S]*?)```")

CH01_DIR = Path(__file__).resolve().parents[1] / "chapters" / "01-foundations"

SYNC_EXEMPT: dict[str, str] = {}  # name -> reason it may differ between the chapter and its notebook

# (chapter regex, notebook-output regex, comparator); each regex has exactly one capture group.
# Comparator: eq, gt, lt or approx:<rel>, applied as `chapter_value <op> notebook_value`.
RESULT_CLAIMS: list[tuple[str, str, str]] = []

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {}

MIN_FIGURES: int = 15
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    "sec-ch1-perplexity": "4.3 Evaluation Metrics",
    "sec-ch1-layernorm": "Layer Normalization",
    "sec-ch1-modern-block": "3.3 What Changed Since 2017",
    "sec-ch1-rope": "Positional Encodings",
    "sec-ch1-relative-bias": "Positional Encodings",
}


def ch01_fence(marker: str) -> str:
    for page in sorted(CH01_DIR.glob("*.md")):
        for m in FENCE.finditer(page.read_text(encoding="utf-8")):
            if marker in m.group(1):
                return m.group(1)
    raise AssertionError(f"no python fence under chapters/01-foundations contains {marker!r}")


def test_comprehensive_tokenizer_round_trips_mixed_case_text():
    ns = {"torch": torch, "np": np, "math": math}
    exec(ch01_fence("class ComprehensiveTokenizer"), ns)  # noqa: S102
    tok = ns["tokenizer"]
    assert tok.vocab["<PAD>"] == 0 and tok.vocab["<UNK>"] == 1
    text = "The quick brown fox"
    assert tok.decode(tok.encode(text)) == text


def test_apply_rope_preserves_norm_and_depends_only_on_relative_offset():
    ns = {"torch": torch, "math": math}
    exec(ch01_fence("def apply_rope"), ns)  # noqa: S102
    torch.manual_seed(0)
    q, k = torch.randn(8, 16), torch.randn(8, 16)
    pos = torch.arange(8)
    qr, kr = ns["apply_rope"](q, pos), ns["apply_rope"](k, pos)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    qs, ks = ns["apply_rope"](q, pos + 7), ns["apply_rope"](k, pos + 7)
    assert torch.allclose(qr @ kr.T, qs @ ks.T, atol=1e-4)
    # a no-op would pass both checks above; the rotation must actually move position 1+
    assert not torch.allclose(qr[1:], q[1:], atol=1e-3)
    # and position 0 (angle 0) must be untouched
    assert torch.allclose(qr[0], q[0], atol=1e-6)


def test_evaluate_tokenization_quality_reports_only_measured_metrics():
    ns = {"torch": torch, "np": np, "math": math}
    exec(ch01_fence("class ComprehensiveTokenizer"), ns)  # noqa: S102
    exec(ch01_fence("def evaluate_tokenization_quality"), ns)  # noqa: S102
    metrics = ns["evaluate_tokenization_quality"](ns["tokenizer"], ["The quick brown fox", "hello world"])
    assert set(metrics) == {"compression_ratio", "oov_rate", "reversibility_errors"}
    assert metrics["reversibility_errors"] == 0.0
