"""Execute selected code fences from chapter 02 and check the numbers they print."""
import math
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CH02 = Path(__file__).resolve().parents[1] / "chapters" / "02-mathematics" / "index.md"
FENCE = re.compile(r"```python\n([\s\S]*?)```")


def fence_containing(marker: str) -> str:
    for m in FENCE.finditer(CH02.read_text(encoding="utf-8")):
        if marker in m.group(1):
            return m.group(1)
    raise AssertionError(f"no python fence in ch02 contains {marker!r}")


def run(marker: str) -> dict:
    ns = {"torch": torch, "np": np, "F": F, "math": math}
    exec(fence_containing(marker), ns)  # noqa: S102 - executing the book's own example
    return ns


def test_memory_estimate_reproduces_gpt3_parameter_count():
    ns = run("def estimate_memory_usage")
    est = ns["estimate_memory_usage"](num_layers=96, d_model=12288, seq_length=2048, batch_size=8, vocab_size=50257)
    assert abs(est["parameters_B"] / 175 - 1) < 0.02, est


def test_attention_gradients_fence_matches_autograd():
    ns = run("def attention_gradients")
    torch.manual_seed(0)
    Q = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    K = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    V = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, 3, dtype=torch.float64)
    O = torch.softmax(Q @ K.T / 2.0, dim=-1) @ V
    auto = torch.autograd.grad(O, (Q, K, V), grad_outputs=g)
    ours = ns["attention_gradients"](Q.detach(), K.detach(), V.detach(), g)
    for a, b in zip(auto, ours):
        assert torch.allclose(a, b, atol=1e-8)


def test_warmup_cosine_schedule_is_never_zero_and_holds_at_lr_min():
    ns = run("class AdamWWithWarmup")
    opt = ns["AdamWWithWarmup"]([torch.zeros(1, requires_grad=True)], lr=1e-3, lr_min=1e-5,
                                warmup_steps=100, max_steps=1000)
    assert opt.get_lr(0) > 0
    assert math.isclose(opt.get_lr(1000), 1e-5, rel_tol=1e-9)
    assert math.isclose(opt.get_lr(1500), 1e-5, rel_tol=1e-9)


CH01_DIR = Path(__file__).resolve().parents[1] / "chapters" / "01-foundations"


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
