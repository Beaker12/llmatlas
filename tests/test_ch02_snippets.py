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


def test_fix_round_numbers_in_prose_and_solutions():
    """Numbers added in the chapter 2 fix round: KL asymmetry, Shannon's alphabet, Adam's uncorrected step,
    the Chinchilla exponent rounding range, the fp16 scale bound, SwiGLU width and full recomputation memory."""
    import math

    kl = lambda a, b: sum(x * math.log2(x / y) for x, y in zip(a, b))
    assert round(kl((0.9, 0.1), (0.5, 0.5)), 3) == 0.531 and round(kl((0.5, 0.5), (0.9, 0.1)), 3) == 0.737
    assert round(kl((0.25, 0.5, 0.25), (0.5, 0.25, 0.25)), 3) == 0.25
    assert round(math.log2(27), 2) == 4.75
    ratio = [(1 - 0.9 ** t) / math.sqrt(1 - 0.999 ** t) for t in range(1, 5001)]
    assert max(range(5000), key=ratio.__getitem__) + 1 == 12 and round(max(ratio), 2) == 6.57
    assert [round(ratio[t - 1], 2) for t in (1, 200, 1000)] == [3.16, 2.35, 1.26]
    assert round(0.28 / 0.62, 4) == 0.4516 and round(0.275 / 0.62, 3) == 0.444 and round(0.285 / 0.62, 3) == 0.46
    assert round(65504 / 3, 2) == 21834.67 and 3 * 2 ** 14 <= 65504 < 3 * 2 ** 15
    assert round(8 * 4096 / 3) == 10923
    assert round(2 * 8 * 2048 * 12288 * 96 / 1e9, 1) == 38.7
    assert round(1e-10 / (1e-10 + 1e-8), 4) == 0.0099


def test_adam_step_without_epsilon_exceeds_kingma_ba_bound_but_not_cauchy_schwarz_bound():
    """Chapter 2, sec-ch2-epsilon-placement: with negligible eps the step |m_hat|/sqrt(v_hat) (in units of lr) is
    3.15 on the first nonzero gradient after 5,000 zeros, as Kingma and Ba's (1 - beta1)/sqrt(1 - beta2) = 3.16 says,
    but rises past it to 4.24 and a peak of 6.55 at the 12th step if the gradient stays constant. No sequence exceeds
    the Cauchy-Schwarz bound (1 - beta1)/sqrt((1 - beta2)(1 - beta1^2/beta2)) = 7.27, which a geometrically growing
    gradient approaches. Steps are taken by torch.optim.Adam itself, in float64."""
    b1, b2 = 0.9, 0.999

    def steps(grads):
        p = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        # eps = 0 would make the zero-gradient steps 0/0 (the first job of eps); 1e-300 avoids that and is
        # negligible against sqrt(v_hat) once a gradient is nonzero.
        opt = torch.optim.Adam([p], lr=1.0, eps=1e-300)
        out = []
        for g in grads:
            before = p.item()
            p.grad = torch.tensor([g], dtype=torch.float64)
            opt.step()
            out.append(abs(p.item() - before))
        return out

    kingma_ba = (1 - b1) / math.sqrt(1 - b2)
    q = b1 * b1 / b2
    cauchy_schwarz = (1 - b1) / math.sqrt((1 - b2) * (1 - q))
    assert round(kingma_ba, 2) == 3.16 and round(cauchy_schwarz, 2) == 7.27
    assert round(q, 4) == 0.8108 and round(1 - q, 4) == 0.1892 and round(0.1 / math.sqrt(0.001 * 0.1892), 2) == 7.27

    resumed = steps([0.0] * 5000 + [1.0] * 200)[5000:]
    assert [round(resumed[k], 2) for k in (0, 1, 11)] == [3.15, 4.24, 6.55]
    assert max(resumed) == resumed[11] and resumed[0] <= kingma_ba < max(resumed) < cauchy_schwarz
    assert abs(steps([1.0] * 100)[-1] - 1.0) < 1e-9  # dense constant gradient: a step of lr

    # Zeros-then-constant gradients never reach 7.27, whatever the length of the zero run.
    assert all(max(steps([0.0] * z + [1.0] * 40)[z:]) < cauchy_schwarz for z in (0, 10, 100, 1000, 20000))

    # The bias-corrected bound over a history of t steps rises toward 7.27 and never exceeds it ...
    def finite_bound(t):
        return (1 - b1) / (1 - b1**t) * math.sqrt((1 - b2**t) / (1 - b2) * (1 - q**t) / (1 - q))

    bounds = [finite_bound(t) for t in range(1, 100_001)]
    assert all(a <= b for a, b in zip(bounds, bounds[1:])) and bounds[-1] <= cauchy_schwarz + 1e-12  # rounding
    assert abs(cauchy_schwarz - bounds[-1]) < 1e-3
    # ... and a gradient growing by beta2/beta1 per step attains it at step t (here t = 3000: 7.09, past 6.55).
    t = 3000
    growing = steps([(b2 / b1) ** (s - t) for s in range(1, t + 1)])
    assert math.isclose(growing[-1], finite_bound(t), rel_tol=1e-9) and round(growing[-1], 2) == 7.09
    assert math.isclose(finite_bound(1), 1.0)
    text = CH02.read_text(encoding="utf-8")
    assert "0.8108" in text and "0.001 \\times 0.1892" in text and "$7.09\\,\\eta$ after 3,000 steps" in text


def test_exercises_are_graded_and_labelled():
    text = CH02.read_text(encoding="utf-8")
    for n in range(1, 9):
        assert f"(sec-ch2-exercise-{n})=\n\n### Exercise {n}: " in text
        assert re.search(rf"### Exercise {n}: .+ \(level [123]\)\n", text)
        assert f"{{admonition}} Solution {n}" in text


def test_exercise_6_attention_backward_by_hand():
    for q1, expected in ((math.log(3), 0.375), (8.0, 6.705e-4)):
        Q = torch.tensor([[q1], [0.0]], dtype=torch.float64, requires_grad=True)
        K = torch.tensor([[1.0], [0.0]], dtype=torch.float64, requires_grad=True)
        V = torch.tensor([[2.0], [0.0]], dtype=torch.float64, requires_grad=True)
        A = torch.softmax(Q @ K.T, dim=-1)
        O = A @ V
        O[0, 0].backward()
        assert math.isclose(Q.grad[0, 0].item(), expected, rel_tol=1e-3) and Q.grad[1, 0].item() == 0
        if q1 < 2:
            assert torch.allclose(A, torch.tensor([[0.75, 0.25], [0.5, 0.5]], dtype=torch.float64))
            assert torch.allclose(O.detach().flatten(), torch.tensor([1.5, 1.0], dtype=torch.float64))
            assert [round(v, 4) for v in K.grad.flatten().tolist()] == [0.412, -0.412]
            assert torch.allclose(V.grad.flatten(), torch.tensor([0.75, 0.25], dtype=torch.float64))
            small = Q.grad[0, 0].item()
        else:
            assert round(small / Q.grad[0, 0].item()) == 559
    s = exercise_text(6)
    assert "(0.375, -0.375)" in s and "(0.4120, -0.4120)" in s and "6.705 \\times 10^{-4}" in s and "559 times" in s


def test_exercise_7_layernorm_gradient_by_hand():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
    y = F.layer_norm(x, (3,), eps=0.0)
    y[0].backward()
    assert [round(v, 4) for v in y.tolist()] == [-1.2247, 0.0, 1.2247]
    assert [round(v, 4) for v in x.grad.tolist()] == [0.2041, -0.4082, 0.2041]
    assert torch.allclose(x.grad, torch.tensor([1.0, -2.0, 1.0], dtype=torch.float64) / (2 * math.sqrt(6)))
    assert abs(x.grad.sum().item()) < 1e-12
    s = exercise_text(7)
    assert "\\sigma = 0.8165" in s and "0.2041" in s and "-0.4082" in s


def test_exercise_8_schedule_clipping_accumulation():
    def lr(t, mx=3e-4, mn=3e-5, tw=2000, te=100_000):
        if t < tw:
            return mx * (t + 1) / tw
        return mn + 0.5 * (mx - mn) * (1 + math.cos((t - tw) / (te - tw) * math.pi))

    got = [lr(t) for t in (0, 999, 26_500, 51_000, 100_000)]
    assert [float(f"{v:.4g}") for v in got] == [1.5e-7, 1.5e-4, 2.605e-4, 1.65e-4, 3e-5]
    p = torch.zeros(2, requires_grad=True)
    p.grad = torch.tensor([3.0, 4.0])
    assert torch.nn.utils.clip_grad_norm_([p], 1.0).item() == 5.0
    assert torch.allclose(p.grad, torch.tensor([0.6, 0.8]), atol=1e-6)
    n, n_acc = 10, 4
    divisors = [min(n_acc, n - (i - 1) // n_acc * n_acc) for i in range(1, n + 1)]
    steps = [i for i in range(1, n + 1) if i % n_acc == 0 or i == n]
    assert steps == [4, 8, 10] and divisors == [4] * 8 + [2] * 2
    s = exercise_text(8)
    assert "2.605 \\times 10^{-4}" in s and "1.65 \\times 10^{-4}" in s and "(0.6, 0.8)" in s


def test_condition_number_rate_and_rosenbrock_minimum():
    """sec-ch2-stability-condition: (cond - 1)/(cond + 1) per step at eta = 2/(lambda_max + lambda_min)."""
    assert round(99 / 101, 4) == 0.9802 and round(math.log(10) / math.log(101 / 99)) == 115
    assert round(math.log(10) / math.log(3), 1) == 2.1
    lo, hi = np.linalg.eigvalsh(np.array([[802.0, -400.0], [-400.0, 200.0]]))
    cond = hi / lo
    assert round(hi, 1) == 1001.6 and round(lo, 3) == 0.399 and round(cond) == 2508
    rate = (cond - 1) / (cond + 1)
    assert round(rate, 4) == 0.9992 and round(math.log(10) / -math.log(rate), -2) == 2900
    for eta in (2 / (hi + lo),):  # both extreme directions shrink by the same factor
        assert math.isclose(abs(1 - eta * lo), abs(1 - eta * hi)) and math.isclose(abs(1 - eta * lo), rate)
    text = CH02.read_text(encoding="utf-8")
    assert "$99/101 = 0.9802$" in text and "1001.6 and 0.399" in text and "= 2508$" in text


def test_mutual_information_worked_table():
    """sec-ch2-mutual-information: MI = H(Y) - H(Y | X) = 1 - 0.7219 = 0.2781 bits."""
    P = np.array([[0.4, 0.1], [0.1, 0.4]])
    px, py = P.sum(1), P.sum(0)
    mi = sum(P[i, j] * math.log2(P[i, j] / (px[i] * py[j])) for i in range(2) for j in range(2))
    hy = -sum(p * math.log2(p) for p in py)
    hyx = -sum(P[i, j] * math.log2(P[i, j] / px[i]) for i in range(2) for j in range(2))
    assert hy == 1.0 and round(hyx, 4) == 0.7219 and round(mi, 4) == 0.2781 and math.isclose(mi, hy - hyx)
    text = CH02.read_text(encoding="utf-8")
    assert "= 0.7219$ bits" in text and "0.2781$ bits" in text
