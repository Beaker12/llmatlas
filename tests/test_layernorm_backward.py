"""Numerically verify the hand-written gradient equations in chapter 02 (step 7 of the
audit plan) and pin the exact LaTeX strings so the text cannot drift from the test."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

CH02 = Path(__file__).resolve().parents[1] / "chapters" / "02-mathematics" / "index.md"

# Exact LaTeX strings that must appear in chapters/02-mathematics/index.md.
LATEX_STRINGS = [
    r"S = QK^T/\sqrt{d_k}",
    r"P = \mathrm{softmax}(S)",
    r"O = PV",
    r"G = \frac{\partial \mathcal{L}}{\partial O} V^T",
    r"\frac{\partial \mathcal{L}}{\partial S} = P \odot \left(G - (G \odot P)\mathbf{1}\mathbf{1}^T\right)",
    r"\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S} K",
    r"\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}}\left(\frac{\partial \mathcal{L}}{\partial S}\right)^T Q",
    r"\frac{\partial \mathcal{L}}{\partial V} = P^T \frac{\partial \mathcal{L}}{\partial O}",
    r"y = \gamma \odot \frac{x - \mu}{\sigma} + \beta",
    r"\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}",
    r"\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(g_i - \frac{1}{d}\sum_j g_j - \hat{x}_i \frac{1}{d}\sum_j g_j \hat{x}_j\right)",
    r"g_j = \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}",
]


@pytest.mark.parametrize("snippet", LATEX_STRINGS)
def test_chapter_contains_step7_equation(snippet):
    assert snippet in CH02.read_text(encoding="utf-8"), f"missing from ch02: {snippet}"


def _attention_grads_from_equations(Q, K, V, dO):
    d_k = Q.shape[-1]
    S = Q @ K.T / d_k**0.5
    P = torch.softmax(S, dim=-1)
    G = dO @ V.T
    dS = P * (G - (G * P).sum(dim=-1, keepdim=True))  # (G ⊙ P) 1 1ᵀ == row sums
    dQ = dS @ K / d_k**0.5
    dK = dS.T @ Q / d_k**0.5
    dV = P.T @ dO
    return dQ, dK, dV


def test_attention_gradient_equations_match_autograd():
    torch.manual_seed(0)
    n, d_k, d_v = 5, 4, 3
    Q = torch.randn(n, d_k, dtype=torch.float64, requires_grad=True)
    K = torch.randn(n, d_k, dtype=torch.float64, requires_grad=True)
    V = torch.randn(n, d_v, dtype=torch.float64, requires_grad=True)
    dO = torch.randn(n, d_v, dtype=torch.float64)
    O = torch.softmax(Q @ K.T / d_k**0.5, dim=-1) @ V
    auto = torch.autograd.grad(O, (Q, K, V), grad_outputs=dO)
    ours = _attention_grads_from_equations(Q.detach(), K.detach(), V.detach(), dO)
    for a, b in zip(auto, ours):
        assert torch.allclose(a, b, atol=1e-8)


def _layernorm_backward_from_equation(x, gamma, dy, eps):
    d = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    sigma = torch.sqrt(((x - mu) ** 2).mean(dim=-1, keepdim=True) + eps)
    xhat = (x - mu) / sigma
    g = gamma * dy
    return (g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True)) / sigma


def test_layernorm_backward_equation_matches_autograd_with_per_feature_gamma():
    torch.manual_seed(0)
    b, d, eps = 3, 7, 1e-5
    x = torch.randn(b, d, dtype=torch.float64, requires_grad=True)
    ln = torch.nn.LayerNorm(d, eps=eps).double()
    with torch.no_grad():
        ln.weight.copy_(torch.randn(d, dtype=torch.float64))
        ln.bias.copy_(torch.randn(d, dtype=torch.float64))
    dy = torch.randn(b, d, dtype=torch.float64)
    (auto,) = torch.autograd.grad(ln(x), x, grad_outputs=dy)
    ours = _layernorm_backward_from_equation(x.detach(), ln.weight.detach(), dy, eps)
    assert torch.allclose(auto, ours, atol=1e-8)
