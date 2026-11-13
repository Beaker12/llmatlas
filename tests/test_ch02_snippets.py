"""Execute selected code fences from chapter 02 and check the numbers they print."""
import math
import re
from pathlib import Path

import numpy as np
import pytest
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
RESULT_CLAIMS: list[tuple[str, str, str]] = [
    (r"is a fraction (\d\.\d+e-\d+) of the gradient at the last layer",
     r"No LayerNorm, std 0\.02\s+layer0=\S+\s+last=\S+\s+ratio=(\S+)", "approx:0.001"),
    (r"it gives (\d+\.\d+) billion", r"GPT-3 175B\s+(\d+\.\d+)B", "eq"),
]

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {}

MIN_FIGURES: int = 12
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    "sec-ch2-sqrt-dk": "Why attention scores are divided by the key dimension's square root",
    "sec-ch2-pre-post-ln": "Pre-LN versus Post-LN",
    "sec-ch2-flops": "Training Compute per Token",
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


CHAPTER_DIR = CH02.parent
NOTEBOOK_PATH = CHAPTER_DIR / "examples" / "mathematical_foundations_demo.ipynb"


def test_notebook_keeps_its_plots():
    """Executing with MPLBACKEND=Agg drops inline images; every plotting cell must still carry an image/png."""
    import json

    cells = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]
    images = {i: sum("image/png" in o.get("data", {}) for o in c.get("outputs", []))
              for i, c in enumerate(cells) if c["cell_type"] == "code"}
    plotting = [i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and "plt.show()" in "".join(c["source"])]
    assert sum(images.values()) >= len(plotting), f"{sum(images.values())} images for {len(plotting)} plotting cells"
    tagged = [i for i, c in enumerate(cells)
              if any(t.startswith("figure:") for t in c.get("metadata", {}).get("tags", []))]
    assert len(tagged) == 3, tagged
    assert all(images[i] >= 1 for i in tagged), [i for i in tagged if not images[i]]


@pytest.mark.slow
def test_make_figures_is_deterministic(tmp_path):
    """Two runs of make_figures.py write byte-identical SVG files."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "MPLBACKEND": "Agg"}
    for run in ("a", "b"):
        subprocess.run([sys.executable, str(CHAPTER_DIR / "figures" / "make_figures.py"), str(tmp_path / run)],
                       check=True, env=env, capture_output=True)
    names = sorted(p.name for p in (tmp_path / "a").glob("*.svg"))
    assert len(names) == 10
    assert all((tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes() for n in names)


# --- Exercises: every number a worked solution states ---------------------------------------------------------
def exercise_text(n: int) -> str:
    text = CH02.read_text(encoding="utf-8")
    start = text.index(f"{{admonition}} Solution {n}")
    return text[start:text.index("```", start + 20)]


def test_exercise_1_entropy_cross_entropy_and_kl():
    p = [0.5, 0.25, 0.125, 0.125]
    H = -sum(x * math.log2(x) for x in p)
    Hpq = -sum(x * math.log2(0.25) for x in p)
    assert H == 1.75 and Hpq == 2.0 and Hpq - H == 0.25
    assert round(2 ** H, 3) == 3.364
    lengths = [1, 2, 3, 3]                                    # the code 0, 10, 110, 111
    assert sum(2.0 ** -k for k in lengths) == 1 and sum(x * k for x, k in zip(p, lengths)) == H
    s = exercise_text(1)
    assert "= 1.75$ bits" in s and "2^{1.75} = 3.364" in s and "0.25$ bits" in s


def test_exercise_2_adam_first_step_and_rosenbrock_bound():
    for g in (0.003, 300.0):
        p = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([p], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        p.grad = torch.tensor([g], dtype=torch.float64)
        opt.step()
        step = -p.item()
        assert math.isclose(step, 1e-3 * g / (g + 1e-8), rel_tol=1e-12)
        uncorrected = 1e-3 * 0.1 * g / (math.sqrt(0.001) * g + 1e-8)
        assert round(uncorrected, 6) == 0.003162
    assert round(1e-3 * 0.003 / 0.00300001, 9) == 9.99997e-4
    x, y = torch.tensor(-1.0, dtype=torch.float64), torch.tensor(-1.0, dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2, torch.stack([x, y]))
    assert H.tolist() == [[1602.0, 400.0], [400.0, 200.0]]
    lam = torch.linalg.eigvalsh(H).max().item()
    assert round(lam) == 1708 and round(math.sqrt(701**2 + 400**2), 1) == 807.1
    assert round(2 / lam, 6) == 1.171e-3 and round(3.8 / lam, 6) == 2.225e-3
    s = exercise_text(2)
    assert "9.99997 \\times 10^{-4}" in s and "3.162 \\times 10^{-3}" in s and "\\lambda_{\\max} = 1708" in s


def test_exercise_3_kaplan_and_chinchilla_by_hand():
    kaplan = lambda N: (8.8e13 / N) ** 0.076                 # noqa: E731
    chin = lambda N, D: 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28  # noqa: E731
    assert round(0.9 ** (-1 / 0.076), 3) == 4.000
    assert round(kaplan(1e9), 3) == 2.376 and round(kaplan(1e10), 3) == 1.994 and round(1 - 10**-0.076, 2) == 0.16
    C = 5.88e23
    for N, D, loss in ((70e9, 1.4e12, 1.937), (280e9, 350e9, 1.983), (17.5e9, 5.6e12, 1.934)):
        assert math.isclose(6 * N * D, C, rel_tol=1e-12)
        assert round(chin(N, D), 3) == loss
    assert [round(406.4 / N**0.34, 4) for N in (70e9, 280e9, 17.5e9)] == [0.0835, 0.0521, 0.1338]
    assert [round(410.7 / D**0.28, 4) for D in (1.4e12, 350e9, 5.6e12)] == [0.1632, 0.2405, 0.1107]
    G = (0.34 * 406.4 / (0.28 * 410.7)) ** (1 / 0.62)
    N_star = G * (C / 6) ** (0.28 / 0.62)
    D_star = C / (6 * N_star)
    assert round(G, 3) == 1.345 and round(0.28 / 0.62, 4) == 0.4516
    assert round(N_star / 1e10, 2) == 3.25 and round(D_star / 1e12, 2) == 3.02 and round(D_star / N_star) == 93
    assert round(chin(N_star, D_star), 3) == 1.930 and round(chin(70e9, 1.4e12) - chin(N_star, D_star), 3) == 0.007
    grid = [10 ** (9 + k / 1000) for k in range(3000)]       # brute force agrees with the closed form
    best = min(grid, key=lambda N: chin(N, C / (6 * N)))
    assert abs(best / N_star - 1) < 0.01
    s = exercise_text(3)
    assert "= 4.000" in s and "2.376" in s and "1.937, 1.983 and 1.934" in s and "about 93 tokens" in s


def test_exercise_4_softmax_saturation_and_residual_variance():
    def slope(gap):
        p1 = 1 / (1 + math.exp(-gap))
        return p1, p1 * (1 - p1)
    assert slope(0) == (0.5, 0.25)
    assert round(slope(1)[0], 4) == 0.7311 and round(slope(1)[1], 4) == 0.1966
    assert round(slope(8)[0], 5) == 0.99966 and round(slope(8)[1] * 1e4, 3) == 3.352 and round(0.25 / slope(8)[1]) == 746
    z = torch.tensor([8.0, 0.0], dtype=torch.float64, requires_grad=True)
    torch.softmax(z, 0)[0].backward()
    assert math.isclose(z.grad[0].item(), slope(8)[1], rel_tol=1e-12)
    torch.manual_seed(0)
    q, k = torch.randn(200_000, 64), torch.randn(200_000, 64)
    assert abs((q * k).sum(-1).std().item() - 8) < 0.05 and abs(((q * k).sum(-1) / 8).std().item() - 1) < 0.01
    g = 2 ** 0.5; pg = 1 / (1 + math.exp(-g))
    assert round(g, 2) == 1.41 and round(pg * (1 - pg), 3) == 0.157 and round(pg * (1 - pg) / slope(8)[1], -1) == 470
    assert [1 + 2 * L for L in (12, 48)] == [25, 97] and all(1 + 2 * L / (2 * L) == 2 for L in (12, 48))
    s = exercise_text(4)
    assert "3.352 \\times 10^{-4}" in s and "746 times" in s and "25 after 12 blocks and 97 after 48" in s


def test_exercise_5_fp16_underflow_loss_scaling_and_lost_updates():
    half = lambda v, dt=torch.float16: torch.tensor(v, dtype=dt).item()  # noqa: E731
    assert half(1e-8) == 0.0 and half(2**-24) == 2**-24 and round(2**-24 * 1e8, 2) == 5.96
    scaled = half(1e-8 * 2**16)
    assert round(1e-8 * 2**16 * 1e4, 3) == 6.554 and round(scaled * 1e4, 3) == 6.552
    assert round(scaled / 2**16 * 1e8, 2) == 1.00
    assert 65504 / 3 < 21835 and half(3.0 * 2**14) <= 65504 and math.isinf(half(3.0 * 2**15))
    for dt, gap in ((torch.float16, 2**-10), (torch.bfloat16, 2**-7)):
        assert torch.finfo(dt).eps == gap
        assert (torch.tensor(1.0, dtype=dt) + torch.tensor(1e-4, dtype=dt)).item() == 1.0
    assert round((torch.tensor(1.0) + torch.tensor(1e-4)).item(), 4) == 1.0001
    s = exercise_text(5)
    assert "2^{14} = 16{,}384" in s and "6.552 \\times 10^{-4}" in s and "$1.0001$" in s
