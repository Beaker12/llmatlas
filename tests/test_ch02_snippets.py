"""Execute selected code fences from chapter 02 and check the numbers they print."""
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CH02 = Path(__file__).resolve().parents[1] / "chapters" / "02-mathematics" / "index.md"
FENCE = re.compile(r"```python\n([\s\S]*?)```")

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

MIN_FIGURES: int = 12
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    "sec-ch2-sqrt-dk": "0.1 A matrix is a map between spaces",
    "sec-ch2-pre-post-ln": "Residual Connections",
    "sec-ch2-flops": "Training Complexity",
    "sec-ch2-residual-init": "Transformer-Specific Initialization",
}


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


def test_kl_divergence_is_infinite_where_q_has_no_mass():
    """D11 / C2-11: D_KL(p || q) is infinite when some q_i = 0 < p_i, and terms with p_i = 0 add nothing."""
    kl = run("def kl_divergence")["kl_divergence"]
    assert math.isinf(kl(torch.tensor([[0.5, 0.5]]), torch.tensor([[1.0, 0.0]]))[0].item())
    assert math.isclose(kl(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.5, 0.5]]))[0].item(), 1.0, rel_tol=1e-6)
    p = torch.tensor([[0.2, 0.3, 0.5]])
    assert abs(kl(p, p)[0].item()) < 1e-7


def test_kl_divergence_asymmetry_on_the_notebook_distributions():
    """C2-01: peaked (0.9 on one of 10) against uncertain (0.5, 0.3, rest 0.2/8) gives 0.6064 and 1.2365 bits."""
    kl = run("def kl_divergence")["kl_divergence"]
    peaked = torch.full((1, 10), 0.1 / 9)
    peaked[0, 0] = 0.9
    uncertain = torch.full((1, 10), 0.2 / 8)
    uncertain[0, 0], uncertain[0, 1] = 0.5, 0.3
    a, b = kl(peaked, uncertain)[0].item(), kl(uncertain, peaked)[0].item()
    assert abs(a - 0.6064) < 1e-3 and abs(b - 1.2365) < 1e-3, (a, b)
    assert abs(a - b) > 0.1
