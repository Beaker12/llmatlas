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
    "sec-ch1-rope": "Rotary Position Embedding (RoPE)",
    "sec-ch1-relative-bias": "Relative Position Bias (T5)",
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


def t5_reference_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """T5Attention._relative_position_bucket from Hugging Face transformers 4.46.3, transcribed.

    Its argument is HF's relative_position = key position - query position, i.e. m' - m.
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large,
                                                                                        num_buckets - 1))
    return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)


def test_relative_position_bias_is_t5_bucketing_with_the_page_sign():
    ns = {"torch": torch, "math": math}
    exec(ch01_fence("class RelativePositionBias"), ns)  # noqa: S102
    bucket = ns["relative_position_bucket"]
    offsets = torch.arange(-1000, 1001)  # m - m'
    for bidirectional in (True, False):
        assert torch.equal(bucket(offsets, bidirectional), t5_reference_bucket(-offsets, bidirectional))
    # known values, bidirectional (encoder): exact up to 7, log-spaced to 15 at 128, keys after the query in 16..31
    known = {0: 0, 1: 1, 7: 7, 8: 8, 12: 9, 16: 10, 64: 14, 127: 15, 128: 15, 1000: 15, -1: 17, -8: 24, -128: 31}
    assert {o: int(bucket(torch.tensor(o), True)) for o in known} == known
    # causal (decoder): every key at or after the query shares bucket 0; exact up to 15, then log-spaced to 31
    known_causal = {-5: 0, 0: 0, 1: 1, 15: 15, 16: 16, 32: 21, 127: 31, 128: 31, 1000: 31}
    assert {o: int(bucket(torch.tensor(o), False)) for o in known_causal} == known_causal
    rpb = ns["RelativePositionBias"](num_heads=8)
    assert rpb.table.weight.shape == (32, 8)  # one scalar per (bucket, head), for the whole model
    bias = rpb(seq_len=16)
    assert bias.shape == (8, 16, 16)
    # entry (m, m') is rb(bucket(m - m')): query 5 looking back at key 2 uses bucket 3, not the future-key bucket 19
    assert torch.equal(bias[:, 5, 2], rpb.table.weight[3])
    assert torch.equal(bias[:, 2, 5], rpb.table.weight[16 + 3])


def test_multi_head_attention_handles_cross_attention_and_padding_masks():
    ns = {}
    exec(ch01_fence("class MultiHeadAttention"), ns)  # noqa: S102
    torch.manual_seed(0)
    mha = ns["MultiHeadAttention"](d_model=32, num_heads=4).eval()
    x, memory = torch.randn(2, 5, 32), torch.randn(2, 7, 32)
    out, weights = mha(x, memory, memory)  # keys and values longer than the queries (C1-18)
    assert out.shape == (2, 5, 32) and weights.shape == (2, 4, 5, 7)
    padding = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    padding[1, ..., 5:] = False  # the last two keys of the second sequence are padding
    _, weights = mha(x, memory, memory, padding)
    assert torch.all(weights[1, ..., 5:] == 0)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5), atol=1e-6)


def test_token_embedding_scaled_by_sqrt_d_has_unit_variance():
    ns = {}
    exec(ch01_fence("class TokenEmbeddingLayer"), ns)  # noqa: S102
    torch.manual_seed(0)
    layer = ns["TokenEmbeddingLayer"](vocab_size=5000, d_model=512, layer_norm=False, dropout=0.0)
    token_rms = (layer.token_embeddings.weight * math.sqrt(512)).pow(2).mean().sqrt().item()
    position_rms = layer.position_encoding.pe.pow(2).mean().sqrt().item()
    assert abs(token_rms - 1.0) < 0.01  # N(0, 1/d) times sqrt(d): unit variance (C1-03)
    assert abs(position_rms - 0.5 ** 0.5) < 0.01  # sin^2 and cos^2 average 1/2
    assert 1.5 < token_rms**2 / position_rms**2 < 2.5  # comparable power, not 24.7x the other way
    ns = {}
    exec(ch01_fence("class TokenEmbedding("), ns)  # noqa: S102
    emb = ns["TokenEmbedding"](5000, 512)
    assert abs(emb(torch.arange(5000)).pow(2).mean().sqrt().item() - 1.0) < 0.01


def test_rope_equals_the_block_diagonal_rotation_of_the_derivation():
    ns = {"torch": torch, "math": math}
    exec(ch01_fence("def apply_rope"), ns)  # noqa: S102
    d, b = 8, 10000.0

    def rotation(m):  # R_m of eq-ch1-rope: 2x2 blocks [[cos, -sin], [sin, cos]] at angle m * b^(-2i/d)
        r = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d // 2):
            a = m * b ** (-2 * i / d)
            r[2 * i:2 * i + 2, 2 * i:2 * i + 2] = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return r

    torch.manual_seed(0)
    x = torch.randn(1, d, dtype=torch.float64)
    for m in (0, 1, 5, 37):
        rotated = ns["apply_rope"](x.float(), torch.tensor([m])).double()
        assert torch.allclose(rotated[0], rotation(m) @ x[0], atol=1e-5)
    # the identity the proof rests on: R_m^T R_m' = R_{m'-m}
    assert torch.allclose(rotation(3).T @ rotation(11), rotation(8), atol=1e-12)


def test_evaluate_tokenization_quality_reports_only_measured_metrics():
    ns = {"torch": torch, "np": np, "math": math}
    exec(ch01_fence("class ComprehensiveTokenizer"), ns)  # noqa: S102
    exec(ch01_fence("def evaluate_tokenization_quality"), ns)  # noqa: S102
    metrics = ns["evaluate_tokenization_quality"](ns["tokenizer"], ["The quick brown fox", "hello world"])
    assert set(metrics) == {"chars_per_token", "bytes_per_token", "reversibility_errors"}
    assert metrics["chars_per_token"] > 1.0
    assert metrics["reversibility_errors"] == 0.0
