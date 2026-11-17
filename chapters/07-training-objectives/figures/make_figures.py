"""Write the data plots of chapter 7 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

- `schedule-runs.svg` plots `examples/data/schedule_runs.json`, the validation curves recorded by
  `tools/build_training_notebook.py --record` with the notebook's own training cell.
- The other plots are computed here from the chapter's own numbers (the fits, schedules, token counts and
  byte counts stated on the page), which the asserts below check where the page states a result.

The output is deterministic: Agg backend, fixed hash salt, glyphs as paths, no date metadata, DejaVu Sans.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

STYLE = {"svg.hashsalt": "llmatlas-ch7", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 11,
         "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10, "xtick.labelsize": 10,
         "ytick.labelsize": 10, "figure.dpi": 100}
matplotlib.rcParams.update(STYLE)

CHAPTER = Path(__file__).resolve().parents[1]
DATA = CHAPTER / "examples" / "data"
BLUE, RED, GREEN, ORANGE, GREY, PURPLE = "#4c72b0", "#c44e52", "#55a868", "#dd8452", "#8c8c8c", "#8172b3"
DARK, LIGHT, PALE = "#4c72b0", "#dfe7f2", "#f4f4f4"


def save(fig, path):
    """Write `path` (SVG, the HTML figure) and the same figure as PDF beside it (the LaTeX figure)."""
    fig.savefig(path, metadata={"Date": None})
    fig.savefig(Path(path).with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


# --- Section 2.1: attention patterns and loss positions -----------------------------------------------------

def attention_masks(out):
    n, prefix = 8, 3
    causal = np.tril(np.ones((n, n), dtype=bool))
    prefix_lm = causal.copy()
    prefix_lm[:prefix, :prefix] = True
    full = np.ones((n, n), dtype=bool)
    loss = {"causal": np.ones(n, bool), "prefix LM (prefix 3)": np.arange(n) >= prefix,
            "bidirectional (masked LM)": np.isin(np.arange(n), [2, 6])}
    masks = {"causal": causal, "prefix LM (prefix 3)": prefix_lm, "bidirectional (masked LM)": full}
    assert prefix_lm.sum() - causal.sum() == prefix * (prefix - 1) // 2
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.9))
    for ax, (name, mask) in zip(axes, masks.items()):
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((j, n - 1 - i), 0.92, 0.92, color=DARK if mask[i, j] else PALE))
            ax.add_patch(Rectangle((i, -1.6), 0.92, 0.92, color=RED if loss[name][i] else PALE))
        ax.text(n / 2, -2.25, "carries a loss", ha="center", va="top", fontsize=9)
        ax.set_xlim(-0.2, n + 0.1)
        ax.set_ylim(-2.9, n + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=11, pad=16)
        ax.text(-0.35, n / 2, "query", rotation=90, ha="right", va="center", fontsize=9)
        ax.text(n / 2, n + 0.05, "key", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save(fig, out / "attention-masks.svg")


# --- Section 2.4: five objectives on one sequence ------------------------------------------------------------

def objectives(out):
    n = 16
    tokens = [f"x{i}" for i in range(1, n + 1)]
    rows = []
    rows.append(("causal", [(t, True) for t in tokens]))
    masked = {4, 11}
    rows.append(("masked LM", [("[M]" if i in masked else t, i in masked) for i, t in enumerate(tokens)]))
    spans = [(4, 6), (11, 12)]  # half-open index ranges removed
    inputs, targets, position = [], [], 0
    for k, (a, b) in enumerate(spans):
        inputs += [(t, False) for t in tokens[position:a]] + [(f"S{k}", False)]
        targets += [(f"S{k}", True)] + [(t, True) for t in tokens[a:b]]
        position = b
    inputs += [(t, False) for t in tokens[position:]]
    targets += [(f"S{len(spans)}", True)]
    rows.append(("span corruption", inputs + [("", None)] + targets))
    prefix = 6
    rows.append(("prefix LM", [(t, i >= prefix) for i, t in enumerate(tokens)]))
    i, j = 5, 10  # prefix x1-x5, middle x6-x10, suffix x11-x16
    fim = [("PRE", True)] + [(t, True) for t in tokens[:i]] + [("SUF", True)] + \
        [(t, True) for t in tokens[j:]] + [("MID", True)] + [(t, True) for t in tokens[i:j]]
    rows.append(("fill-in-the-middle", fim))
    width = max(len(r) for _, r in rows)
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    for r, (name, row) in enumerate(rows):
        y = len(rows) - 1 - r
        ax.text(-0.3, y + 0.42, name, ha="right", va="center", fontsize=10)
        for c, (label, carries) in enumerate(row):
            if carries is None:
                ax.text(c + 0.45, y + 0.42, "→", ha="center", va="center", fontsize=11)
                continue
            special = label.startswith(("S", "[", "P", "M")) and not label.startswith("x")
            face = DARK if carries else (LIGHT if not special else "#f2d7d7")
            ax.add_patch(Rectangle((c, y), 0.9, 0.84, color=face))
            ax.text(c + 0.45, y + 0.42, label, ha="center", va="center", fontsize=7,
                    color="white" if carries else "black")
    ax.set_xlim(-3.6, width + 0.2)
    ax.set_ylim(-0.3, len(rows))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    save(fig, out / "objectives.svg")


# --- Section 3.3: Newton-Schulz on one singular value ---------------------------------------------------------

def newton_schulz(out):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = np.logspace(-4, 0, 2000)
    y = x.copy()
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.axhspan(0.68, 1.21, color=GREEN, alpha=0.15, lw=0)
    ax.plot(x, x, color=GREY, ls=":", label="input (no step)")
    colors = [ORANGE, PURPLE, BLUE, RED, "#222222"]
    for k in range(1, 6):
        y = a * y + b * y ** 3 + c * y ** 5
        ax.plot(x, y, color=colors[k - 1], lw=1.6 if k == 5 else 1.0, label=f"{k} step{'s' if k > 1 else ''}")
    fine = np.logspace(-6, 0, 600_001)
    z = fine.copy()
    for _ in range(5):
        z = a * z + b * z ** 3 + c * z ** 5
    inside = (z >= 0.68) & (z <= 1.21)
    first = fine[np.where(~inside)[0].max() + 1]
    assert 0.0014 < first < 0.0016 and z.max() <= 1.21  # "every singular value above about 0.0015 ... 0.68 and 1.21"
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("singular value before (after normalization)")
    ax.set_ylabel("singular value after")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, out / "newton-schulz.svg")


# --- Section 3.4: bytes per parameter -------------------------------------------------------------------------

def optimizer_memory(out):
    state = {"AdamW": 8, "Lion": 4, "Muon": 4, "Adafactor (no momentum)": 0, "SGD": 0}
    parts = [("bf16 weight", 2, BLUE), ("bf16 gradient", 2, ORANGE), ("fp32 master weight", 4, GREEN)]
    assert sum(p[1] for p in parts) + state["AdamW"] == 16
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = list(state)[::-1]
    for row, name in enumerate(names):
        left = 0
        for label, size, color in parts:
            ax.barh(row, size, left=left, color=color, label=label if row == 0 else None)
            left += size
        if state[name]:
            ax.barh(row, state[name], left=left, color=RED, label="optimizer state" if name == "Lion" else None)
        ax.text(left + state[name] + 0.2, row, f"{left + state[name]} bytes", va="center", fontsize=10)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 19.5)
    ax.set_xticks(range(0, 20, 2))
    ax.set_xlabel("bytes per parameter")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, out / "optimizer-memory.svg")


# --- Section 4.1: decay shapes --------------------------------------------------------------------------------

def schedules(out):
    p = np.linspace(0, 1, 2001)
    cosine = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * p))
    multistep = np.where(p < 0.8, 1.0, np.where(p < 0.9, 0.316, 0.1))
    wsd = np.where(p < 0.8, 1.0, 1 - np.sqrt(np.clip((p - 0.8) / 0.2, 0, 1)))
    total = 14.8
    b1, b2, b3 = 10 / total, 14.3 / total, (14.3 + 0.333) / total
    v3 = np.where(p < b1, 1.0, np.where(p < b2, 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (p - b1) / (b2 - b1))),
                                        np.where(p < b3, 0.1, 7.3e-6 / 2.2e-4)))
    assert abs(b1 - 0.676) < 1e-3 and abs(b2 - 0.966) < 1e-3
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(p, cosine, color=BLUE, label="cosine to 10% (Llama 2)")
    ax.plot(p, multistep, color=ORANGE, label="multi-step (DeepSeek LLM)")
    ax.plot(p, wsd, color=RED, label="WSD, 20% 1 - sqrt cooldown")
    ax.plot(p, v3, color=GREEN, label="DeepSeek-V3")
    ax.set_xlabel("fraction of training completed")
    ax.set_ylabel("learning rate / peak")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, out / "schedules.svg")


# --- Section 4.3: the recorded schedule experiment ------------------------------------------------------------

def schedule_runs(out):
    record = json.loads((DATA / "schedule_runs.json").read_text(encoding="utf-8"))
    runs = record["runs"].values()
    every = record["every"]

    def mean(get):
        return np.mean([get(r) for r in runs], axis=0)

    stable = mean(lambda r: r["stable"])
    steps = np.arange(every, every * (len(stable) + 1), every)
    cos = {e: mean(lambda r, e=e: r["cosine"][e]) for e in ("800", "1200")}
    wsd = {e: mean(lambda r, e=e: r["wsd"][e]["curve"]) for e in ("800", "1200")}
    starts = {e: next(iter(runs))["wsd"][e]["start"] for e in ("800", "1200")}
    assert wsd["1200"][-1] < cos["1200"][-1] and wsd["800"][-1] < cos["800"][-1]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(steps, stable, color=GREY, label="stable (constant rate)")
    ax.plot(steps[:len(cos["1200"])], cos["1200"], color=BLUE, label="cosine, 1,200 steps")
    ax.plot(steps[:len(cos["800"])], cos["800"], color=BLUE, ls="--", label="cosine, 800 steps")
    for e, ls in (("1200", "-"), ("800", "--")):
        x = np.arange(math.ceil((starts[e] + 1) / every) * every, int(e) + 1, every)  # the branch's own records
        ax.plot(x, wsd[e], color=RED, ls=ls, marker=".", ms=4, label=f"WSD branch to {int(e):,}")
        ax.axvline(starts[e], color=RED, lw=0.6, alpha=0.4)
    ax.set_xlim(0, 1220)
    ax.set_ylim(1.6, 2.6)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (nats per byte)")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, out / "schedule-runs.svg")


# --- Section 5.1: isoFLOP curves of the refit -----------------------------------------------------------------

FIT = (1.8172, 482.01, 2085.43, 0.3478, 0.3658)  # Besiroglu et al.


def loss(n, d):
    l_inf, c_n, c_d, g_n, g_d = FIT
    return l_inf + c_n / n ** g_n + c_d / d ** g_d


def optimal(compute):
    _, c_n, c_d, g_n, g_d = FIT
    k = (g_n * c_n / (g_d * c_d)) ** (1 / (g_n + g_d))
    n = k * (compute / 6) ** (g_d / (g_n + g_d))
    return n, compute / (6 * n)


def isoflop(out):
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    budgets = [1e18, 1e20, 1e22, 5.88e23, 1e26]
    colors = [PURPLE, ORANGE, GREEN, RED, BLUE]
    for budget, color in zip(budgets, colors):
        n = np.logspace(7, 13, 400)
        d = budget / (6 * n)
        keep = d > 1e8
        label = "5.88e23 (Chinchilla)" if budget == 5.88e23 else f"{budget:.0e}".replace("e+", "e")
        ax.plot(n[keep], loss(n[keep], d[keep]), color=color, label=f"C = {label}")
        n_opt, d_opt = optimal(budget)
        ax.plot([n_opt], [loss(n_opt, d_opt)], "o", color=color, ms=5)
    ax.plot([70e9], [loss(70e9, 1.4e12)], "*", color="black", ms=11, label="Chinchilla, 70B on 1.4T")
    n_c, _ = optimal(5.88e23)
    assert 65e9 < n_c < 80e9  # the refit lands near Chinchilla's own size
    ax.set_xscale("log")
    ax.set_ylim(1.8, 4.2)
    ax.set_xlabel("parameters N")
    ax.set_ylabel("fitted loss")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, out / "isoflop.svg")


# --- Section 5.2: tokens per parameter of published models ----------------------------------------------------

MODELS = [  # name, parameters, training tokens, label offset (points)
    ("GPT-3 175B", 175.0e9, 300e9, (-36, 8)),
    ("Chinchilla", 70e9, 1.4e12, (-62, 6)),
    ("Llama 1 7B", 6.7e9, 1.0e12, (6, -10)),
    ("Llama 1 65B", 65.2e9, 1.4e12, (6, -12)),
    ("Llama 2 7B", 7e9, 2e12, (-66, 2)),
    ("Llama 2 70B", 70e9, 2e12, (6, 2)),
    ("Llama 3 8B", 8.03e9, 15e12, (-62, -12)),
    ("Llama 3 70B", 70.55e9, 15e12, (6, -2)),
    ("Llama 3 405B", 405e9, 15.6e12, (6, -10)),
    ("OLMo 2 7B", 7.30e9, 4.05e12, (6, -4)),
    ("Gemma 2 9B", 9.24e9, 8e12, (6, -6)),
    ("Gemma 3 12B", 11.77e9, 12e12, (6, -8)),
    ("Qwen2.5 7B", 7.62e9, 18e12, (6, 3)),
    ("Qwen3 8B", 8.19e9, 36e12, (6, 0)),
    ("DeepSeek-V3 (active)", 37e9, 14.8e12, (6, 4)),
]


def tokens_per_parameter(out):
    ratios = {name: d / n for name, n, d, _ in MODELS}
    assert ratios["GPT-3 175B"] < 2 and 1850 < ratios["Llama 3 8B"] < 1900 and ratios["Qwen3 8B"] > 4000
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    n = np.logspace(9.5, 12, 10)
    ax.plot(n, 20 * n, color=GREY, ls="--", label="20 tokens per parameter")
    ax.plot(n, 2000 * n, color=GREY, ls=":", label="2,000 tokens per parameter")
    for name, params, tokens, offset in MODELS:
        ax.plot([params], [tokens], "o", color=BLUE, ms=5)
        ax.annotate(name, (params, tokens), textcoords="offset points", xytext=offset, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2e9, 1e12)
    ax.set_ylim(1e11, 1e14)
    ax.set_xlabel("parameters")
    ax.set_ylabel("training tokens")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, out / "tokens-per-parameter.svg")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for make in (attention_masks, objectives, newton_schulz, optimizer_memory, schedules, schedule_runs, isoflop,
                 tokens_per_parameter):
        make(out)
        matplotlib.rcParams.update(STYLE)


if __name__ == "__main__":
    main()
