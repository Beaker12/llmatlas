"""Execute selected code fences from chapter 01 and check what they define."""
import math
import re
from pathlib import Path

import numpy as np
import torch

from tests._pages import pages

FENCE = re.compile(r"```python\n([\s\S]*?)```")

CH01_DIR = Path(__file__).resolve().parents[1] / "chapters" / "01-foundations"

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
    "MultiHeadAttention": "the notebook version delegates to ScaledDotProductAttention, which is frozen in both "
    "notebooks during Phase 1, and returns the attention weights its heatmaps plot; the chapter fence is "
    "self-contained and returns only the output",
    "TransformerBlock": "calls MultiHeadAttention, whose notebook version returns (output, weights) and the "
    "chapter version only the output (see MultiHeadAttention)",
    "RelativePositionBias": "the notebook keeps a short clipping distance (10) so its bias heatmap is readable "
    "at the demo's sequence length; the chapter uses the T5-scale 128. The offset sign is settled with C1-12",
}

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
    for page in pages(CH01_DIR):
        for m in FENCE.finditer(page.read_text(encoding="utf-8")):
            if marker in m.group(1):
                return m.group(1)
    raise AssertionError(f"no python fence under chapters/01-foundations contains {marker!r}")


def test_comprehensive_tokenizer_round_trips_mixed_case_text():
    ns = {"torch": torch, "np": np, "math": math}
    exec(ch01_fence("class ComprehensiveTokenizer"), ns)  # noqa: S102
    tok = ns["tokenizer"]
    # byte-level: ids 0..255 are the bytes, the five special tokens follow, then the learned merges (D6)
    assert tok.special_tokens == {"<PAD>": 256, "<UNK>": 257, "<BOS>": 258, "<EOS>": 259, "<MASK>": 260}
    assert all(tok.vocab[i] == bytes([i]) for i in range(256))
    assert len(tok.merges) == 39
    for text in ["The quick brown fox", "Hello, world!", "The quick brown fox.", "two  spaces", "a\nnewline",
                 "你好世界", "tabs\tand  \n\n mixed   whitespace "]:
        ids = tok.encode(text)
        assert tok.special_tokens["<UNK>"] not in ids
        assert tok.decode(ids) == text, (text, tok.decode(ids))
    # C1-08: frequent words merge, an unseen word still splits
    assert tok.pieces("The") == ["The"]
    assert len(tok.pieces("unthinkable")) > 1


def test_simple_bpe_splits_an_unseen_word():
    ns = {}
    exec(ch01_fence("class SimpleBPE"), ns)  # noqa: S102
    assert ns["bpe"].encode("helper") == ["hel", "p", "e", "r", "</w>"]


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
    assert set(metrics) == {"chars_per_token", "bytes_per_token", "reversibility_errors"}
    assert metrics["chars_per_token"] > 1.0
    assert metrics["reversibility_errors"] == 0.0
