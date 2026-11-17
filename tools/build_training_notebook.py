#!/usr/bin/env python3
"""Build the chapter 7 notebook (unexecuted) from the chapter text and the cell sources below.

Functions the chapter prints (z_loss, span_corrupt, fim_transform, adam_steps, param_groups, newton_schulz, wsd,
cosine, chinchilla_loss, compute_optimal, deepseek_llm_hyperparameters) are copied from the chapter's fences, so
the notebook runs exactly what the text shows.

Usage: python tools/build_training_notebook.py
Then execute it in place from its own directory (see the README).

`--record` runs the schedule experiment of Section 4.3 (three seeds) with the notebook's own training cell and
writes the validation curves to examples/data/schedule_runs.json, which figures/make_figures.py plots for the
chapter. `--record --check` recomputes them, writes nothing, and fails unless the training cell's hash matches the
recorded one, every recomputed loss is within RECORD_TOLERANCE of the recorded one, and the notebook's asserted
statements hold on the recomputed curves.
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CH07 = ROOT / "chapters" / "07-training-objectives" / "index.md"
OUT = ROOT / "chapters" / "07-training-objectives" / "examples" / "training_objectives_demo.ipynb"
RECORD_JSON = OUT.parent / "data" / "schedule_runs.json"
RECORD_TOLERANCE = 1e-4
RECORD_SEEDS = (0, 1, 2)
FENCE = re.compile(r"```python\n([\s\S]*?)```")


def function(name: str) -> str:
    """One top-level definition (with any decorators and a preceding constant block it needs) from the chapter's
    last fence that defines it, up to the next top-level statement."""
    found = None
    for m in FENCE.finditer(CH07.read_text(encoding="utf-8")):
        hit = re.search(rf"^def {name}\b[\s\S]*?(?=\n\n\S|\Z)", m.group(1), re.M)
        if hit:
            found = hit.group(0).rstrip()
    if found is None:
        raise SystemExit(f"no python fence in {CH07.name} defines {name!r}")
    return found


CELLS: list[tuple[str, str]] = []
COMPARE: dict[int, dict] = {}  # cell index -> {"compare": "shape"} where printed numbers depend on the CPU
TAGS: dict[int, list[str]] = {}  # cell index -> metadata tags


def md(text: str, tags: tuple[str, ...] = ()) -> None:
    CELLS.append(("markdown", text.strip("\n")))
    if tags:
        TAGS[len(CELLS) - 1] = list(tags)


def code(text: str, tags: tuple[str, ...] = (), compare: dict | None = None) -> None:
    CELLS.append(("code", text.strip("\n")))
    if compare:
        COMPARE[len(CELLS) - 1] = dict(compare)
    if tags:
        TAGS[len(CELLS) - 1] = list(tags)


md(r"""
# Training Objectives Demo

**Purpose**: This notebook checks the mechanisms of Chapter 7 on small inputs: gradient accumulation that equals one large batch, the z-loss, span corruption, a prefix-LM mask, the fill-in-the-middle transformation, an AdamW step written by hand, Muon's Newton-Schulz iteration, a small model trained with a cosine and with a warm-up-stable-decay schedule, and the compute-optimal sizes of the two fits of the Chinchilla data. Every section ends in assertions, so a silent change to any formula fails the notebook.

**Data**: the training text is the first 150 KB of Jane Austen's *Pride and Prejudice* (public domain, from Project Gutenberg), the same file chapter 6's notebook uses, copied into `data/` beside this notebook and checked against its SHA-256. Nothing is downloaded.

**Time**: the training section runs three seeds of a 135,488-parameter model, each in its own cell of about two minutes on one CPU thread; everything else takes seconds.
""")

md("## Setup")
code(r'''
import copy
import hashlib
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
plt.rcParams["figure.dpi"] = 100

TEXT_FILE = Path("data/pride_and_prejudice_150k.txt")
TEXT_SHA256 = "8c394b472b75396071e8d70d0c2f7bb4dfb76565aeca62c127961960d4f57560"


@contextmanager
def timed(label):
    start = time.perf_counter()
    yield
    print(f"[{label}: {time.perf_counter() - start:.1f} s]")


raw = TEXT_FILE.read_bytes()
assert hashlib.sha256(raw).hexdigest() == TEXT_SHA256, "the text file has changed"
print(f"text: {len(raw):,} bytes")
''')

md(r"""
(nb-train-loss)=
## The Loss: Accumulation and the Z-Loss

A batch of sequences with different numbers of loss-bearing tokens is split into micro-batches. Dividing each micro-batch's summed loss by the token count of the whole batch reproduces the one-batch gradient; averaging the micro-batch means does not. The check runs over several random splits and paddings, on a small embedding-and-linear model.
""")
code(r'''
def accumulation_errors(seed, vocab=40, width=12, n_micro=4, length=24):
    """Relative gradient errors of (mean of means, sum over all tokens) against one
    batch."""
    g = torch.Generator().manual_seed(seed)
    model = nn.Sequential(nn.Embedding(vocab, width), nn.Linear(width, vocab))
    for p in model.parameters():
        nn.init.normal_(p, std=0.5, generator=g)
    inputs = torch.randint(vocab, (n_micro, 2, length), generator=g)
    targets = torch.randint(vocab, (n_micro, 2, length), generator=g)
    kept = torch.randint(1, length + 1, (n_micro, 2), generator=g)
    for i in range(n_micro):
        for j in range(2):
            targets[i, j, kept[i, j]:] = -100

    def grad(losses):
        model.zero_grad()
        sum(losses).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    def losses(i, reduction):
        return F.cross_entropy(model(inputs[i]).reshape(-1, vocab),
                               targets[i].reshape(-1), ignore_index=-100,
                               reduction=reduction)

    full = grad([losses(slice(None), "mean")])
    total = (targets != -100).sum()
    mean_of_means = grad([losses(i, "mean") / n_micro for i in range(n_micro)])
    token_mean = grad([losses(i, "sum") / total for i in range(n_micro)])
    return [((g - full).norm() / full.norm()).item()
            for g in (mean_of_means, token_mean)]


errors = np.array([accumulation_errors(seed) for seed in range(20)])
print(f"mean of means: relative error {errors[:, 0].min():.3f} "
      f"to {errors[:, 0].max():.3f}")
print(f"sum / all tokens: largest relative error {errors[:, 1].max():.1e}")
assert errors[:, 1].max() < 1e-5 and errors[:, 0].min() > 1e-3
''')
code(function("z_loss") + r'''


def fit_logits(weight, steps=3000):
    """Free logits fitted to fixed target distributions, with and without the z-loss."""
    g = torch.Generator().manual_seed(1)
    targets = torch.randn(8, 50, generator=g).softmax(-1)
    logits = (torch.randn(8, 50, generator=g) * 3 + 12).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(steps):
        loss = -(targets * logits.log_softmax(-1)).sum(-1).mean()
        if weight:
            loss = loss + z_loss(logits, weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return logits.detach(), targets


plain, targets = fit_logits(0.0)
with_z, _ = fit_logits(1e-2)
for name, fitted in (("without z-loss", plain), ("with z-loss   ", with_z)):
    gap = (fitted.softmax(-1) - targets).abs().max()
    print(f"{name}: largest probability error {gap:.1e}, "
          f"mean |log Z| {torch.logsumexp(fitted, -1).abs().mean():.3f}")
assert (with_z.softmax(-1) - targets).abs().max() < 1e-3
assert torch.logsumexp(with_z, -1).abs().max() < 0.05
assert torch.logsumexp(plain, -1).abs().min() > 0.05
''')

md(r"""
(nb-train-objectives)=
## Objectives: Span Corruption, Prefix LM and Fill-in-the-Middle

The chapter's `span_corrupt` is applied to a passage of the book, read as bytes, and the passage is restored from the input and the target. The prefix-LM mask is the causal mask with the prefix block made bidirectional; the fill-in-the-middle transformation is applied to many documents to check its rate and that every transformed document can be put back together.
""")
code(function("span_corrupt") + r'''


def restore(inputs, targets, base=32_000):
    """Put the spans of a span-corruption target back into its input."""
    spans, current = {}, None
    for t in targets:
        if t >= base:
            current = t
            spans[current] = []
        else:
            spans[current].append(t)
    out = []
    for t in inputs:
        out += spans[t] if t >= base else [t]
    return out


passage = list(raw[10_000:10_512])
for seed in range(5):
    inputs, targets = span_corrupt(passage, seed=seed)
    assert restore(inputs, targets) == passage
inputs, targets = span_corrupt(passage)
removed = [t for t in targets if t < 32_000]
print(f"input {len(inputs)} bytes, target {len(targets)} "
      f"({len(removed)} removed bytes)")
print("first removed span:", bytes(removed[:targets.index(32_001) - 1]))
''')
code(r'''
def prefix_lm_mask(n, prefix):
    """True where query i may attend to key j: bidirectional in the prefix,
    causal after it."""
    mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
    mask[:prefix, :prefix] = True
    return mask


n = 8
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
for prefix in (0, 1, 3, n):
    mask = prefix_lm_mask(n, prefix)
    extra = int((mask & ~causal).sum())
    print(f"prefix {prefix}: {extra} attention pairs beyond the causal mask")
    assert extra == prefix * (prefix - 1) // 2
    # the rows after the prefix stay causal
    assert torch.equal(mask[prefix:], causal[prefix:])
''')
code(function("fim_transform") + r'''


def unfim(text):
    """Recover the original document from a PSM or SPM rearrangement."""
    if not text.startswith("<PRE>"):
        return text
    body = text[len("<PRE>"):-len("<EOT>")]
    if body.startswith("<SUF>"):  # SPM: suffix, then prefix and middle together
        suffix, rest = body[len("<SUF>"):].split("<MID>", 1)
        return rest + suffix
    prefix, rest = body.split("<SUF>", 1)
    suffix, middle = rest.split("<MID>", 1)
    return prefix + middle + suffix


document = raw[20_000:20_400].decode("utf-8")
rng = random.Random(0)
outputs = [fim_transform(document, rng=rng) for _ in range(4000)]
share = sum(o.startswith("<PRE>") for o in outputs) / len(outputs)
assert all(unfim(o) == document for o in outputs)
print(f"share transformed {share:.3f}; every document restored")
assert abs(share - 0.5) < 0.03
''')
code(r'''
n, prefix, masked = 512, 128, round(0.15 * 512)
inputs, targets = span_corrupt(list(range(n)))
shares = {
    "causal": (n - 1) / n,  # every position but the last predicts a token
    "fill-in-the-middle": (n - 1) / n,
    "prefix LM": (n - prefix) / n,
    "masked LM": masked / n,
    "span corruption": len(targets) / n,
}
for name, share in shares.items():
    print(f"{name:20s} {share:.3f} of the sequence length carries a loss")
assert (shares["masked LM"] < shares["span corruption"] < shares["prefix LM"]
        < shares["causal"])
''')

md(r"""
(nb-train-optimizers)=
## Optimizers: AdamW by Hand, Adam's Second Moment and Muon

One AdamW step, with bias correction and decoupled weight decay, written out and compared with `torch.optim.AdamW` over several steps; then the chapter's sudden-gradient experiment and its parameter groups; then the Newton-Schulz iteration on matrices of several shapes.
""")
code(r'''
def adamw_by_hand(params, grads, state, lr, betas=(0.9, 0.95), eps=1e-8,
                  weight_decay=0.1):
    """One AdamW step on lists of tensors; `state` holds step, m and v."""
    beta1, beta2 = betas
    state["step"] += 1
    t = state["step"]
    for i, (p, g) in enumerate(zip(params, grads)):
        p.mul_(1 - lr * weight_decay)  # decoupled weight decay
        state["m"][i] = beta1 * state["m"][i] + (1 - beta1) * g
        state["v"][i] = beta2 * state["v"][i] + (1 - beta2) * g * g
        m_hat = state["m"][i] / (1 - beta1 ** t)
        v_hat = state["v"][i] / (1 - beta2 ** t)
        p.sub_(lr * m_hat / (v_hat.sqrt() + eps))


torch.manual_seed(0)
reference = nn.Linear(16, 8).double()
mine = copy.deepcopy(reference)
optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2, betas=(0.9, 0.95),
                              eps=1e-8, weight_decay=0.1)
state = {"step": 0, "m": [torch.zeros_like(p) for p in mine.parameters()],
         "v": [torch.zeros_like(p) for p in mine.parameters()]}
for step in range(20):
    x = torch.randn(32, 16, dtype=torch.float64)
    for model in (reference, mine):
        model.zero_grad()
        model(x).pow(2).mean().backward()
    optimizer.step()
    with torch.no_grad():
        params = list(mine.parameters())
        adamw_by_hand(params, [p.grad for p in params], state, lr=1e-2)
pairs = zip(reference.parameters(), mine.parameters())
gap = max((a - b).abs().max().item() for a, b in pairs)
print(f"after 20 steps, largest difference from torch.optim.AdamW: {gap:.1e}")
assert gap < 1e-12
''')
code(function("adam_steps") + r'''


grads = [0.01] * 2000 + [1.0] * 100
largest = {beta2: max(adam_steps(grads, beta2=beta2)[2000:]) for beta2 in (0.999, 0.95)}
print({k: round(v, 2) for k, v in largest.items()})
assert largest[0.999] > 5 and largest[0.95] < 1.2
''')
code(function("param_groups") + r'''


model = nn.Sequential(nn.Embedding(100, 16), nn.RMSNorm(16), nn.Linear(16, 16))
groups = param_groups(model, decay_embeddings=False)
names = {id(p): n for n, p in model.named_parameters()}
for group in groups:
    members = sorted(names[id(p)] for p in group["params"])
    print(f"weight decay {group['weight_decay']}:", members)
assert [names[id(p)] for p in groups[0]["params"]] == ["2.weight"]
''')
code(function("newton_schulz") + r'''


torch.manual_seed(0)
for rows, cols in ((64, 64), (256, 64), (64, 256), (512, 128)):
    u = torch.randn(rows, cols, dtype=torch.float64)
    values = torch.linalg.svdvals(newton_schulz(u))
    in_band = ((values >= 0.68) & (values <= 1.21)).float().mean().item()
    print(f"{rows:3d} x {cols:3d}: singular values after five steps "
          f"{values.min():.3f} to {values.max():.3f}, {in_band:.1%} in the band")
    assert newton_schulz(u).shape == u.shape and values.max() <= 1.21
    # a square random matrix has singular values near zero; a rectangular one does not
    assert in_band > 0.95 and (rows == cols or in_band == 1.0)
''')

md(r"""
(nb-train-schedules)=
## Schedules: Cosine Against Warm-Up-Stable-Decay
""")
md(r"""
A byte-level model with two pre-norm blocks of width 64 (135,488 parameters) is trained on the first 90% of the text in batches of 8 sequences of 64 bytes, at a peak learning rate of $10^{-2}$ after 50 warm-up steps, with AdamW ($\beta_2 = 0.95$, weight decay 0.1) and clipping at 1.0. The validation loss on 64 fixed sequences from the last 10% of the text is recorded every 50 steps.

For each seed, one **stable** run holds the peak rate for 1,200 steps. At step 640 and at step 960 it is copied, model, optimizer and data position together, and each copy is decayed to zero along the chapter's `wsd` cooldown, ending at 800 and 1,200 steps: two **WSD** runs for the price of one stable run and two short decays. Two **cosine** runs, of 800 and 1,200 steps, use the chapter's `cosine` schedule with its floor of 10% of the peak and see the same batches.

Single-threaded training is deterministic on one machine, but the last bits of a float32 matrix product depend on the CPU's instruction set, and 1,200 steps amplify them. The printed losses therefore agree between machines only roughly: the notebook test compares the layout of those printouts, and every statement asserted below holds on every machine tried. The chapter's figure plots the curves recorded in `data/schedule_runs.json` by `tools/build_training_notebook.py --record`, which runs this cell.
""", tags=("static-numbers",))
TRAIN_CELL = function("wsd") + "\n\n\n" + function("cosine") + r'''


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(width), nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        b, n, w = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, w // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.out(attended.transpose(1, 2).reshape(b, n, w))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class ByteLM(nn.Module):
    def __init__(self, width=64, layers=2, heads=4, context=64):
        super().__init__()
        self.embed = nn.Embedding(256, width)
        self.position = nn.Parameter(torch.zeros(context, width))
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.RMSNorm(width)
        self.head = nn.Linear(width, 256, bias=False)

    def forward(self, idx):
        x = self.embed(idx) + self.position[:idx.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


TOKENS = torch.tensor(list(raw), dtype=torch.long)
SPLIT = int(0.9 * len(TOKENS))
TRAIN, VALID = TOKENS[:SPLIT], TOKENS[SPLIT:]
BATCH, SEQ, PEAK, WARM, EVERY = 8, 64, 1e-2, 50, 50


def sample(tokens, count, generator):
    start = torch.randint(0, len(tokens) - SEQ - 1, (count,), generator=generator)
    return tokens[start[:, None] + torch.arange(SEQ + 1)]


VALID_BATCH = sample(VALID, 64, torch.Generator().manual_seed(123))


def validation_loss(model):
    with torch.no_grad():
        logits = model(VALID_BATCH[:, :-1])
        targets = VALID_BATCH[:, 1:].reshape(-1)
        return F.cross_entropy(logits.reshape(-1, 256), targets).item()


def train_step(model, optimizer, generator, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
    batch = sample(TRAIN, BATCH, generator)
    logits = model(batch[:, :-1]).reshape(-1, 256)
    loss = F.cross_entropy(logits, batch[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def new_run(seed):
    torch.manual_seed(seed)
    model = ByteLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK, betas=(0.9, 0.95),
                                  weight_decay=0.1)
    return model, optimizer, torch.Generator().manual_seed(1000 + seed)


def run_cosine(seed, t_end):
    model, optimizer, generator = new_run(seed)
    curve = []
    for t in range(t_end):
        train_step(model, optimizer, generator, cosine(t, PEAK, WARM, t_end))
        if (t + 1) % EVERY == 0:
            curve.append(validation_loss(model))
    return curve


def run_wsd(seed, ends=(800, 1200), decay_fraction=0.2):
    """One stable run; at 80% of each length in `ends`, a copy is decayed to zero."""
    model, optimizer, generator = new_run(seed)
    starts = {round(end * (1 - decay_fraction)): end for end in ends}
    stable, branches = [], {}
    for t in range(max(ends)):
        if t in starts:
            end = starts[t]
            b_model, b_optimizer = copy.deepcopy((model, optimizer))
            b_generator = torch.Generator()
            b_generator.set_state(generator.get_state())
            curve = []
            for u in range(t, end):
                train_step(b_model, b_optimizer, b_generator,
                           wsd(u, PEAK, WARM, end, decay_fraction))
                if (u + 1) % EVERY == 0:
                    curve.append(validation_loss(b_model))
            branches[end] = {"start": t, "curve": curve}
        train_step(model, optimizer, generator, PEAK * min(1.0, (t + 1) / WARM))
        if (t + 1) % EVERY == 0:
            stable.append(validation_loss(model))
    return {"stable": stable, "wsd": branches,
            "cosine": {end: run_cosine(seed, end) for end in ends}}


print(f"{sum(p.numel() for p in ByteLM().parameters()):,} parameters; "
      f"{BATCH * SEQ} tokens per step; validation every {EVERY} steps")
RUNS = {}
'''
code(TRAIN_CELL)
md("One cell per seed keeps each well inside the time limit of a notebook cell; each prints the time its runs took.")
for _seed in RECORD_SEEDS:
    code(f'''
with timed("seed {_seed}"):
    RUNS[{_seed}] = run_wsd({_seed})
r = RUNS[{_seed}]
print("final validation loss: stable", round(r["stable"][-1], 3),
      "| wsd 800", round(r["wsd"][800]["curve"][-1], 3),
      "| wsd 1200", round(r["wsd"][1200]["curve"][-1], 3))
print("  cosine 800", round(r["cosine"][800][-1], 3),
      "| cosine 1200", round(r["cosine"][1200][-1], 3))
''', compare={"compare": "shape"})
CLAIMS_CELL = r'''
def schedule_claims(runs):
    """The statements this section asserts, evaluated on {seed: run_wsd(seed)}."""
    claims = {}
    for seed, r in runs.items():
        stable = dict(zip(range(EVERY, 1201, EVERY), r["stable"]))
        for end, branch in r["wsd"].items():
            name = f"seed {seed}: the {end}-step WSD branch ends below"
            claims[f"{name} the stable run"] = branch["curve"][-1] < stable[end] - 0.02
            cosine_end = r["cosine"][end][-1]
            claims[f"{name} cosine of {end}"] = branch["curve"][-1] < cosine_end
    return claims
'''
code(CLAIMS_CELL + r'''

for claim, holds in schedule_claims(RUNS).items():
    print(f"{'holds' if holds else 'FAILS'}: {claim}")
    assert holds, claim
''')
code(r'''
steps = np.arange(EVERY, 1201, EVERY)
mean = {key: np.mean([fn(RUNS[s]) for s in RUNS], axis=0) for key, fn in {
    "stable": lambda r: r["stable"],
    "cosine 1200": lambda r: r["cosine"][1200],
    "cosine 800": lambda r: r["cosine"][800],
    "wsd 1200": lambda r: r["wsd"][1200]["curve"],
    "wsd 800": lambda r: r["wsd"][800]["curve"],
}.items()}
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(steps, mean["stable"], color="#8c8c8c", label="stable (constant rate)")
ax.plot(steps, mean["cosine 1200"], color="#4c72b0", label="cosine, 1,200 steps")
ax.plot(steps[:16], mean["cosine 800"], color="#4c72b0", ls="--",
        label="cosine, 800 steps")
ax.plot(steps[19:], mean["wsd 1200"], color="#c44e52", label="WSD decay to 1,200")
ax.plot(steps[12:16], mean["wsd 800"], color="#c44e52", ls="--",
        label="WSD decay to 800")
ax.set_xlabel("step")
ax.set_ylabel("validation loss (nats per byte)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
''', compare={"compare": "shape"})

md(r"""
(nb-train-scaling)=
## Scaling: Two Fits of the Chinchilla Data

The closed-form compute-optimal size is checked against a numerical minimization along each isoFLOP curve, for both fits, and Llama 3 8B's shape is placed against the optimum of the refit.
""")
code(r'''
from scipy.optimize import brentq, minimize_scalar

FITS = {  # L_inf, c_N, c_D, gamma_N, gamma_D
    "Hoffmann et al.": (1.69, 406.4, 410.7, 0.34, 0.28),
    "Besiroglu et al.": (1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}


''' + function("chinchilla_loss") + "\n\n\n" + function("compute_optimal") + "\n\n\n" + function("isoflop_loss") + r'''


for name, fit in FITS.items():
    for compute in np.logspace(19, 26, 8):
        closed, _ = compute_optimal(compute, fit)
        result = minimize_scalar(isoflop_loss, args=(compute, fit), bounds=(5, 15),
                                 method="bounded", options={"xatol": 1e-10})
        assert abs(math.log10(closed) - result.x) < 1e-6
    n, d = compute_optimal(5.88e23, fit)
    print(f"{name}: at Chinchilla's budget N = {n / 1e9:.1f}B, D = {d / 1e12:.2f}T")
    print(f"  ({d / n:.1f} tokens per parameter)")
print("closed form matches the numerical optimum at 8 budgets for both fits")
''')
code(r'''
fit = FITS["Besiroglu et al."]
n, d = 8.03e9, 15e12
loss = chinchilla_loss(n, d, fit)
same_loss = brentq(lambda c: chinchilla_loss(*compute_optimal(c, fit), fit) - loss,
                   1e21, 6 * n * d)
n_small, _ = compute_optimal(same_loss, fit)
print(f"Llama 3 8B shape: {6 * n * d / same_loss:.1f}x the compute of the optimum")
print(f"at equal fitted loss; that optimum has {n_small / 1e9:.0f}B parameters")
assert 5 < 6 * n * d / same_loss < 6 and 30e9 < n_small < 40e9
''')
code(function("deepseek_llm_hyperparameters") + r'''


lrs, batches = zip(*(deepseek_llm_hyperparameters(c) for c in (1e20, 1e22, 1e24)))
print("learning rates", [f"{x:.2e}" for x in lrs])
print("batch sizes (M tokens)", [round(x / 1e6, 2) for x in batches])
assert lrs[0] > lrs[1] > lrs[2] and batches[0] < batches[1] < batches[2]
''')

md(r"""
## Summary

- Dividing summed micro-batch losses by the token count of the whole batch reproduces the one-batch gradient; the mean of micro-batch means does not.
- The z-loss drives the log-normalizer to zero without changing a single predicted probability.
- Span corruption, the prefix-LM mask and fill-in-the-middle are each undone exactly, and they differ in the share of positions that carry a loss.
- A hand-written AdamW step matches PyTorch's; a short second-moment memory removes the large steps after a sudden change in the gradient; Muon's Newton-Schulz steps bring the singular values of a random matrix into a band around one, all of them when the matrix is rectangular.
- In the small model, decay branches taken from one constant-rate run end below that run and below the cosine runs of the same lengths.
- The closed-form compute-optimal size matches a numerical minimization, and the published fit and the refit of the Chinchilla data disagree widely in the model size they recommend.
""")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    for index, (kind, text) in enumerate(CELLS):
        cell = nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text)
        if index in TAGS:
            cell.metadata["tags"] = TAGS[index]
        if index in COMPARE:
            cell.metadata.update(COMPARE[index])
        nb.cells.append(cell)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} with {len(nb.cells)} cells")


# --- the recorded schedule runs behind the chapter's figure ---------------------------------------------------
_RECORD_NS: dict = {}


def train_cell_sha256() -> str:
    """SHA-256 of the training cell's source exactly as it appears in the notebook."""
    import hashlib
    return hashlib.sha256(TRAIN_CELL.strip("\n").encode("utf-8")).hexdigest()


def _record_worker_init() -> None:
    """Run the notebook's code cells up to and including the training cell, without output."""
    import contextlib
    import io
    import os
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(OUT.parent)
    stop = CELLS.index(("code", TRAIN_CELL.strip("\n")))
    with contextlib.redirect_stdout(io.StringIO()):
        for kind, text in CELLS[:stop + 1]:
            if kind == "code":
                exec(compile(text, "<notebook cell>", "exec"), _RECORD_NS)


def _record_run(seed):
    return seed, _RECORD_NS["run_wsd"](seed)


def run_record(processes: int) -> dict:
    import multiprocessing
    with multiprocessing.get_context("spawn").Pool(processes, initializer=_record_worker_init) as pool:
        runs = dict(pool.map(_record_run, RECORD_SEEDS, chunksize=1))
    import torch
    return {
        "description": "Validation loss every 50 steps of the Section 4.3 schedule experiment, per seed; "
                       "written by tools/build_training_notebook.py --record.",
        "torch": torch.__version__,
        "train_cell_sha256": train_cell_sha256(),
        "every": 50,
        "runs": {str(seed): {"stable": r["stable"],
                             "wsd": {str(end): b for end, b in r["wsd"].items()},
                             "cosine": {str(end): c for end, c in r["cosine"].items()}}
                 for seed, r in runs.items()},
    }


def as_runs(record: dict) -> dict:
    """The recorded runs with integer keys, in the shape run_wsd returns."""
    return {int(seed): {"stable": r["stable"], "wsd": {int(e): b for e, b in r["wsd"].items()},
                        "cosine": {int(e): c for e, c in r["cosine"].items()}}
            for seed, r in record["runs"].items()}


def record_claims(record: dict) -> dict:
    namespace: dict = {"EVERY": record["every"]}
    exec(compile(CLAIMS_CELL, "<claims cell>", "exec"), namespace)
    return namespace["schedule_claims"](as_runs(record))


def main(argv=None) -> int:
    import argparse
    import json
    import time
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="run and record the schedule experiment")
    parser.add_argument("--check", action="store_true", help="with --record: recompute and check, write nothing")
    parser.add_argument("--processes", type=int, default=len(RECORD_SEEDS))
    args = parser.parse_args(argv)
    if not args.record:
        build()
        return 0
    start = time.perf_counter()
    record = run_record(args.processes)
    print(f"recorded {len(RECORD_SEEDS)} seeds in {time.perf_counter() - start:.0f} s")
    if not args.check:
        RECORD_JSON.parent.mkdir(parents=True, exist_ok=True)
        RECORD_JSON.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {RECORD_JSON.relative_to(ROOT)}")
        ok = True
    else:
        recorded = json.loads(RECORD_JSON.read_text(encoding="utf-8"))
        ok = recorded["train_cell_sha256"] == record["train_cell_sha256"]
        print("recorded training-cell hash matches:", ok)
        a, b = as_runs(recorded), as_runs(record)
        pairs = [(x, y) for s in a for x, y in zip(
            a[s]["stable"] + sum((a[s]["wsd"][e]["curve"] for e in a[s]["wsd"]), []) +
            sum((a[s]["cosine"][e] for e in a[s]["cosine"]), []),
            b[s]["stable"] + sum((b[s]["wsd"][e]["curve"] for e in b[s]["wsd"]), []) +
            sum((b[s]["cosine"][e] for e in b[s]["cosine"]), []))]
        worst = max(abs(x - y) for x, y in pairs)
        print(f"recorded losses reproduced (largest difference {worst:.2e}, tolerance {RECORD_TOLERANCE:g}; "
              f"torch {record['torch']} now, {recorded['torch']} recorded):", worst <= RECORD_TOLERANCE)
        ok = ok and worst <= RECORD_TOLERANCE
    for name, holds in record_claims(record).items():
        print(f"{'holds' if holds else 'FAILS'}: {name}")
        ok = ok and holds
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
