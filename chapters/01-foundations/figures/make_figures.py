"""Write the data plots of chapter 1 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

Each figure is written as <name>.svg (the file the HTML build uses, compared byte for byte by the test suite) and
<name>.pdf (the file the LaTeX build uses). The sources are:

  - the pages' own listings: the BPE trace, the sinusoidal plots, the position-cost plot and the word-frequency plot
    execute the text of the fence that defines them, so editing the fence changes the figure;
  - numbers stated on the page (the worked attention example, the RoPE example, the history milestones), repeated
    below next to the figure that uses them;
  - figures/data/tokens_per_language.json, written by the deep-dive's `# network` fence with LLMATLAS_WRITE_DATA=1.

The output is deterministic: Agg backend, fixed seeds, fixed SVG hash salt, glyphs as paths, no date metadata,
DejaVu Sans.
"""
import contextlib
import io
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "index.md"
DEEP_DIVE = HERE.parent / "tokenization-deep-dive.md"

BASE_RC = {
    "svg.hashsalt": "llmatlas-ch01",
    "svg.fonttype": "path",
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "path.simplify": True,
    "axes.prop_cycle": matplotlib.cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]),
}
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#4a5568"


def save(fig, out, name):
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.05, metadata={"Date": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def page_fence(page, marker):
    """The text of the page's first python fence that contains `marker`."""
    for match in re.finditer(r"^```python\n(.*?)^```", page.read_text(encoding="utf-8"), re.M | re.S):
        if marker in match.group(1):
            return match.group(1)
    raise SystemExit(f"no python fence on {page.name} contains {marker!r}")


def run_fence(page, marker, call=None):
    """Execute a fence (and optionally one extra statement); return the figures it showed, in order."""
    shown = []
    namespace = {}
    original = plt.show
    plt.show = lambda *a, **k: shown.append(plt.gcf())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(page_fence(page, marker), str(page), "exec"), namespace)
            if call:
                exec(call, namespace)
    finally:
        plt.show = original
    return namespace, shown


# BPE merges (sec-ch1-byte-pair-encoding-bpe): the page's corpus and SimpleBPE, one row per merge.
def bpe_merges(out):
    namespace, _ = run_fence(PAGE, "class SimpleBPE")
    bpe = namespace["SimpleBPE"]()
    corpus = namespace["corpus"]
    bpe.train(corpus, num_merges=4)
    words = list(dict.fromkeys(corpus))
    counts = {w: corpus.count(w) for w in words}
    states = [[tuple(w) + ("</w>",) for w in words]]
    for pair in bpe.merges:
        states.append([bpe.merge_word(w, pair) for w in states[-1]])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_axis_off()
    right = 0.0
    labels = ["start"] + [f"merge {i}: {a} + {b}" for i, (a, b) in enumerate(bpe.merges, start=1)]
    for row, state in enumerate(states):
        y = -row
        ax.text(-0.3, y, labels[row], ha="right", va="center", fontsize=10)
        x = 0.0
        new = "".join(bpe.merges[row - 1]) if row else None
        for word, symbols in zip(words, state):
            for s in symbols:
                width = 0.36 + 0.18 * len(s) if s != "</w>" else 0.95
                color = "#fde2b6" if s == new else ("#edf2f7" if s == "</w>" else "white")
                ax.add_patch(Rectangle((x, y - 0.32), width, 0.64, facecolor=color, edgecolor=GREY, lw=1))
                ax.text(x + width / 2, y, s, ha="center", va="center", fontsize=9.5 if s != "</w>" else 8)
                x += width
            if row == 0:
                ax.text(x - 0.05, 0.5, f"×{counts[word]}", ha="right", va="bottom", fontsize=9, color=GREY)
            x += 0.35
        right = max(right, x)
    ax.set_xlim(-3.1, right)
    ax.set_ylim(-len(states) + 0.4, 0.95)
    save(fig, out, "bpe-merges")


# Sinusoidal encodings (sec-ch1-sinusoidal-positions): the two plots of the page's fence.
def sinusoidal(out):
    _, shown = run_fence(PAGE, "def visualize_positional_encoding", "")
    if len(shown) != 2:
        raise SystemExit(f"expected two plots from the sinusoidal fence, got {len(shown)}")
    shown[0].axes[0].grid(False)
    save(shown[0], out, "sinusoidal-heatmap")
    save(shown[1], out, "sinusoidal-lines")


# RoPE (sec-ch1-rope): the page's d_k = 2 example, frequency 1, q = k = (1, 0).
def rope_rotation(out):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    for ax, (m, m2) in zip(axes, ((3, 5), (10, 12))):
        circle = np.linspace(0, 2 * math.pi, 200)
        ax.plot(np.cos(circle), np.sin(circle), color=GREY, lw=0.8)
        for pos, color, name, dash in ((m, BLUE, r"$\tilde{q}_m$", "-"), (m2, ORANGE, r"$\tilde{k}_{m'}$", "--")):
            ax.add_patch(FancyArrowPatch((0, 0), (math.cos(pos), math.sin(pos)), arrowstyle="-|>",
                                         mutation_scale=12, color=color, lw=2, linestyle=dash))
            ax.text(1.18 * math.cos(pos), 1.18 * math.sin(pos), name, color=color, ha="center", va="center",
                    fontsize=11)
        arc = np.linspace(m, m2, 50)
        ax.plot(0.45 * np.cos(arc), 0.45 * np.sin(arc), color=GREEN, lw=1.5)
        mid = (m + m2) / 2
        ax.text(0.2 * math.cos(mid), 0.2 * math.sin(mid), "2", color=GREEN, ha="center", va="center",
                fontsize=10.5)
        ax.set_title(f"$m = {m}$, $m' = {m2}$")
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.text(0.5, 0.02, r"angle between them $(m' - m)\,\omega_0 = 2$ in both, so the score is "
             f"$\\cos 2 = {math.cos(2):.3f}$ in both", ha="center", color=GREEN, fontsize=10)
    save(fig, out, "rope-rotation")


def matrix(ax, values, fmt, cmap, row_labels, col_labels, title, vmin=None, vmax=None):
    ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text = fmt(values[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="white" if vmax is not None and values[i, j] > 0.75 * vmax else "black")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)


# Worked attention example (sec-ch1-attention-worked-example): the page's X, identity projections.
def attention_worked_example(out):
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = x @ x.T / math.sqrt(2)
    weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    tokens = ["token 1", "token 2", "token 3"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
    matrix(axes[0], scores, lambda v: f"{v:.4f}", "Blues", tokens, tokens, r"Scores $QK^\top/\sqrt{2}$",
           vmin=0, vmax=1.6)
    matrix(axes[1], weights, lambda v: f"{v:.4f}", "Oranges", tokens, tokens, "Weights (each row sums to 1)",
           vmin=0, vmax=1.0)
    for ax in axes:
        ax.set_xlabel("key")
    axes[0].set_ylabel("query")
    fig.tight_layout()
    save(fig, out, "attention-worked-example")


def mask_panel(ax, allowed, rows, cols, title):
    ax.imshow(np.where(allowed, 1.0, 0.0), cmap="Blues", vmin=-0.4, vmax=1.6)
    ax.set_xticks(range(len(cols)), cols, fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_title(title, fontsize=10.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


# Mask patterns of the three variants (sec-ch1-architectural-variants): 4 source and 3 target tokens.
def mask_patterns(out):
    src, tgt = ["s1", "s2", "s3", "s4"], ["t1", "t2", "t3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [4, 4, 7]})
    mask_panel(axes[0], np.ones((4, 4), bool), src, src, "Encoder-only:\nevery pair")
    mask_panel(axes[1], np.tril(np.ones((4, 4), bool)), src, src, "Decoder-only:\nearlier or equal")
    cross = np.hstack([np.ones((3, 4), bool), np.tril(np.ones((3, 3), bool))])
    mask_panel(axes[2], cross, tgt, src + tgt, "Decoder of encoder-decoder:\nall source, earlier target")
    axes[2].axvline(3.5, color=RED, lw=1.5)
    for ax in axes:
        ax.set_xlabel("key")
    axes[0].set_ylabel("query")
    fig.tight_layout()
    save(fig, out, "mask-patterns")


# Causal mask (sec-ch1-causal-masking): the teacher-forcing inputs of the page, 0 = kept, -inf = masked.
def causal_mask(out):
    tokens = ["<bos>", "the", "cat", "sat"]
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    allowed = np.tril(np.ones((n, n), bool))
    mask_panel(ax, allowed, tokens, tokens, "Added to the scores before the softmax")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "0" if allowed[i, j] else "−∞", ha="center", va="center", fontsize=11,
                    color="white" if allowed[i, j] else "black")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    save(fig, out, "causal-mask")


# History (sec-ch1-historical-context): the milestones the section names, on a year axis.
MILESTONES = [(1997, "LSTM"), (2003, "Neural LM (Bengio)"), (2013, "word2vec"), (2014, "seq2seq"),
              (2014.6, "Attention (Bahdanau)"), (2017, "Transformer"), (2018, "ELMo"), (2018.4, "GPT"),
              (2018.8, "BERT"), (2019, "GPT-2, 1.5B"), (2019.5, "T5, BART"), (2020, "GPT-3, 175B"),
              (2022, "InstructGPT, ChatGPT"), (2023, "LLaMA, Mistral 7B"), (2024, "Llama 3, 405B")]


def history_timeline(out):
    # Two panels on one year scale with a break at 2012, so the crowded 2013-2024 span gets the width.
    # (level, alignment) per milestone keeps every label clear of the others.
    place = [(1, "center"), (-1, "center"), (1, "left"), (-1, "left"), (2, "left"), (-2, "left"), (1, "left"),
             (-1, "left"), (3, "left"), (-3, "left"), (2, "left"), (-1, "left"), (4, "right"), (-4, "right"),
             (1, "right")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"width_ratios": [1, 2.4]})
    for ax, (lo, hi) in zip(axes, ((1995, 2008), (2012.5, 2025.5))):
        for (year, label), (level, align) in zip(MILESTONES, place):
            if not lo <= year <= hi:
                continue
            ax.plot([year, year], [0, level * 0.7], color=GREY, lw=0.8)
            ax.plot(year, 0, "o", color=BLUE, ms=4)
            ax.text(year, level * 0.7 + (0.08 if level > 0 else -0.08), label, ha=align,
                    va="bottom" if level > 0 else "top", fontsize=9)
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
    axes[0].set_xticks([1995, 2000, 2005])
    axes[1].set_xticks(range(2013, 2026, 2))
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_ylim(-3.4, 3.4)
    fig.supxlabel("year (axis broken between 2008 and 2012)", fontsize=10.5)
    fig.tight_layout()
    save(fig, out, "history-timeline")


# Tokens per 100 characters (sec-tdd-linguistic-bias), from the committed counts.
def tokens_per_language(out):
    data = json.loads((HERE / "data" / "tokens_per_language.json").read_text(encoding="utf-8"))
    languages, names = list(data["counts"]), data["encodings"]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    width = 0.26
    for k, (name, hatch) in enumerate(zip(names, ("", "//", ".."))):
        values = [100 * data["counts"][lang][name] / data["counts"][lang]["characters"] for lang in languages]
        ax.bar(np.arange(len(languages)) + (k - 1) * width, values, width, label=name, hatch=hatch,
               edgecolor="white")
    ax.set_xticks(range(len(languages)), languages)
    ax.set_ylabel("tokens per 100 characters")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, out, "tokens-per-language")


# Position-scheme parameter cost (sec-tdd-position-encoding-trade-offs): the deep-dive fence.
def position_cost(out):
    _, shown = run_fence(DEEP_DIVE, "def compare_position_encodings")
    save(shown[0], out, "position-parameter-cost")


# Word frequencies (sec-tdd-best-practices-tokenization): the fence's function run on this chapter's main page.
class WordTokenizer:
    def encode(self, text):
        return re.findall(r"[a-z]+", text.lower())


def word_frequency(out):
    paragraphs = [p for p in PAGE.read_text(encoding="utf-8").split("\n\n") if not p.startswith(("```", "$$"))]
    namespace, shown = run_fence(DEEP_DIVE, "def analyze_token_frequency")
    shown.clear()
    plt.show = lambda *a, **k: shown.append(plt.gcf())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["analyze_token_frequency"](paragraphs, WordTokenizer())
    finally:
        pass  # main() restores plt.show
    save(shown[0], out, "word-frequency")


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    show = plt.show
    with matplotlib.rc_context(BASE_RC):
        for make in (bpe_merges, sinusoidal, rope_rotation, attention_worked_example, mask_patterns, causal_mask,
                     history_timeline, tokens_per_language, position_cost, word_frequency):
            make(out)
            plt.show = show


if __name__ == "__main__":
    main()
