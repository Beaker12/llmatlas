"""Write the data plots of chapter 2 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

Each figure is written as <name>.svg (the file the HTML build uses, compared byte for byte by the test suite) and
<name>.pdf (the file the LaTeX build uses). Three sources feed the figures:

  - numbers stated on the page (the fitted constants of the scaling laws, the worked KL example, the formats of
    the floating-point table), repeated below next to the figure that uses them;
  - the page's own listings: the memory and attention-share figures execute the text of the fence that defines
    `estimate_memory_usage` and `attention_share`, so editing the fence changes the figure;
  - the companion notebook: a figure that reproduces a notebook plot executes the notebook's code cells in order
    and saves the plot of the cell tagged `figure:<name>`.

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

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "index.md"
NOTEBOOK = HERE.parent / "examples" / "mathematical_foundations_demo.ipynb"
NOTEBOOK_FIGURES = ("optimizer-trajectories", "lr-schedules", "gradient-flow-depth")

BASE_RC = {
    "svg.hashsalt": "llmatlas-ch02",
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
}
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#4a5568"


def save(fig, out, name):
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.05, metadata={"Date": None})
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def page_fence(marker):
    """The text of the page's first python fence that contains `marker`."""
    for match in re.finditer(r"^```python\n(.*?)^```", PAGE.read_text(encoding="utf-8"), re.M | re.S):
        if marker in match.group(1):
            return match.group(1)
    raise SystemExit(f"no python fence on the page contains {marker!r}")


def run_fence(marker):
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(page_fence(marker), str(PAGE), "exec"), namespace)
    return namespace


# Softmax saturation (sec-ch2-softmax-jacobian): dp_i/dz_i = p_i (1 - p_i) when z_i leads the other logits by gap.
def softmax_saturation(out):
    gap = np.linspace(0, 14, 400)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    # The text of each annotation sits on the side of its point away from the other curve.
    for count, color, offset, align in ((2, BLUE, (-12, -10), ("right", "top")),
                                        (64, ORANGE, (12, 8), ("left", "bottom"))):
        p1 = np.exp(gap) / (np.exp(gap) + (count - 1))
        ax.plot(gap, p1 * (1 - p1), color=color, lw=2, label=f"{count} logits")
        p8 = math.exp(8) / (math.exp(8) + count - 1)
        ax.plot(8, p8 * (1 - p8), "o", color=color, ms=6)
        ax.annotate(f"$p_i = {p8:.4f}$, derivative {p8 * (1 - p8):.2g}", (8, p8 * (1 - p8)), xytext=offset,
                    textcoords="offset points", ha=align[0], va=align[1], color=color)
    ax.axvline(8, color=GREY, ls="--", lw=1)
    ax.text(7.85, 1.3e-6, "gap 8: the standard deviation\nof one unscaled score at $d_k = 64$\n(the difference of two scores\nhas $8\\sqrt{2} = 11.3$)", ha="right",
            va="bottom", color=GREY)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.6)
    ax.set_xlim(0, 14)
    ax.set_xlabel("logit gap $z_i - z_j$ (all other logits equal)")
    ax.set_ylabel(r"$\partial p_i / \partial z_i = p_i(1 - p_i)$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, out, "softmax-saturation")


# Cross-entropy = entropy + KL (sec-ch2-cross-entropy-kl-divergence), with the page's worked p and q, in bits.
def cross_entropy_decomposition(out):
    p = np.array([0.5, 0.25, 0.25])
    cases = [("$q = p$", p), ("$q = (1/3, 1/3, 1/3)$", np.full(3, 1 / 3)),
             ("$q = (0.25, 0.5, 0.25)$", np.array([0.25, 0.5, 0.25]))]
    h_p = float(-(p * np.log2(p)).sum())
    fig, ax = plt.subplots(figsize=(7, 2.9))
    for row, (label, q) in enumerate(cases):
        h_pq = float(-(p * np.log2(q)).sum())
        ax.barh(row, h_p, color="#9fb7d6", edgecolor=GREY, label="$H(p)$" if row == 0 else None)
        ax.barh(row, h_pq - h_p, left=h_p, color=ORANGE, edgecolor=GREY,
                label=r"$D_{KL}(p\,\|\,q)$" if row == 0 else None)
        ax.text(h_pq + 0.03, row, f"$H(p, q) = {h_pq:.3f}$  ($D_{{KL}} = {h_pq - h_p:.3f}$)", va="center")
    ax.axvline(h_p, color=GREY, ls="--", lw=1)
    ax.set_yticks(range(len(cases)), [c[0] for c in cases])
    ax.invert_yaxis()
    ax.set_xlim(0, 2.75)
    ax.set_xlabel("bits per symbol, for $p = (0.5, 0.25, 0.25)$")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2, frameon=False)
    save(fig, out, "cross-entropy-decomposition")


# Kaplan et al.'s parameter and data laws with the page's constants (sec-ch2-parameter-scaling, data scaling).
def kaplan_power_laws(out):
    laws = [("$N$, non-embedding parameters", 8.8e13, 0.076, (768, 1.5e9), (1e2, 1e13), "N", BLUE),
            ("$D$, training tokens", 5.4e13, 0.095, (22e6, 23e9), (1e6, 1e14), "D", ORANGE)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for ax, (xlabel, scale, alpha, fit, span, sym, color) in zip(axes, laws):
        x = np.logspace(math.log10(span[0]), math.log10(span[1]), 300)
        y = (scale / x) ** alpha
        inside = (x >= fit[0]) & (x <= fit[1])
        ax.axvspan(*fit, color=color, alpha=0.12, lw=0, label="range of the fit")
        ax.plot(x[inside], y[inside], color=color, lw=2.2, label="fitted law")
        ax.plot(x[x >= fit[1]], y[x >= fit[1]], color=color, lw=1.6, ls="--", label="extrapolation")
        ax.plot(x[x <= fit[0]], y[x <= fit[0]], color=color, lw=1.6, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(rf"$\mathcal{{L}}({sym}) = ({sym}_c/{sym})^{{\alpha_{sym}}}$, $\alpha_{sym} = {alpha}$")
        end = (scale / fit[1]) ** alpha
        ax.annotate(f"{end:.2f}", (fit[1], end), xytext=(4, 4), textcoords="offset points")
    axes[0].set_ylabel(r"$\mathcal{L}$ (nats per token)")
    axes[0].set_yticks([1.5, 2, 3, 4, 6], ["1.5", "2", "3", "4", "6"])
    axes[0].minorticks_off()
    axes[1].minorticks_off()
    axes[1].legend(loc="upper right")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "kaplan-power-laws")


# Optimal N* and D* against compute: Kaplan's exponents (0.73, 0.27) against Chinchilla's (0.50, 0.50), both
# drawn through Chinchilla's own size, with the four rows of Hoffmann et al.'s Table 3 quoted on the page.
def allocation_lines(out):
    n0, d0 = 70e9, 1.4e12
    c0 = 6 * n0 * d0
    compute = np.logspace(19, 26, 200)
    table3 = [(1e9, 20.2e9), (10e9, 205.1e9), (67e9, 1.5e12), (175e9, 3.7e12)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True)
    for ax, base, exps, ylabel, column in ((axes[0], n0, (0.73, 0.50), "optimal parameters $N^*$", 0),
                                           (axes[1], d0, (0.27, 0.50), "optimal tokens $D^*$", 1)):
        ax.plot(compute, base * (compute / c0) ** exps[0], color=BLUE, lw=2, label=f"Kaplan, exponent {exps[0]:.2f}", zorder=2)
        ax.plot(compute, base * (compute / c0) ** exps[1], color=ORANGE, lw=2,
                label=f"Chinchilla, exponent {exps[1]:.2f}", zorder=2)
        # The hollow star leaves the 67 B Table 3 dot, which sits almost on Chinchilla's own size, visible.
        ax.plot(c0, base, "*", mfc="none", mec="black", mew=1.2, ms=16, zorder=3, label="Chinchilla, 70 B")
        ax.plot([6 * n * d for n, d in table3], [row[column] for row in table3], "o", color=ORANGE, mec="black",
                ms=5, zorder=4, label="Table 3, 4 rows")
        ax.legend(loc="upper left", fontsize=10, frameon=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training compute $C = 6ND$ (FLOPs)")
        ax.set_ylabel(ylabel)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "allocation-lines")


# Training memory of the GPT-3 configuration, from the page's estimate_memory_usage fence.
def memory_stack(out):
    gpt3 = run_fence("def estimate_memory_usage")["gpt3"]
    params = gpt3["parameters_B"] * 1e9
    parts = [("fp16 weights, 2 bytes per parameter", 2 * params / 1e9, "#9fb7d6"),
             ("fp16 gradients, 2 bytes per parameter", 2 * params / 1e9, "#6f93c4"),
             ("fp32 master weights, 4 bytes per parameter", 4 * params / 1e9, "#3d6aa8"),
             ("Adam moments $m$ and $v$, 8 bytes per parameter", 8 * params / 1e9, "#1f3f73"),
             ("activations, batch $B = 8$, $n = 2{,}048$, no recomputation", gpt3["activation_GB"], ORANGE)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bottom = 0.0
    for label, size, color in parts:
        ax.bar(0, size, bottom=bottom, width=0.5, color=color, edgecolor="white")
        ax.annotate(f"{label}: {size:,.0f} GB", (0.25, bottom + size / 2), xytext=(0.42, bottom + size / 2),
                    va="center", arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8})
        bottom += size
    ax.text(-0.27, bottom + 220, f"total {bottom:,.0f} GB: {gpt3['param_and_optimizer_GB']:,.0f} GB of weights and "
            f"optimizer state, {gpt3['activation_GB']:,.0f} GB of activations", ha="left", va="bottom")
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, bottom * 1.16)
    ax.set_xticks([])
    ax.set_ylabel("GB")
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="x", visible=False)
    save(fig, out, "memory-stack")


# Attention's share of the per-layer work, 2n^2 d / (12 n d^2 + 2 n^2 d), from the page's attention_share fence.
def attention_crossover(out):
    attention_share = run_fence("def attention_share")["attention_share"]
    n = np.logspace(math.log10(256), math.log10(262144), 300)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for d, color in ((4096, BLUE), (12288, ORANGE)):
        ax.plot(n, 100 * attention_share(n, d), color=color, lw=2, label=f"$d = {d:,}$".replace(",", "{,}"))
        ax.plot(6 * d, 50, "o", color=color, ms=6)
        left = d == 4096
        ax.annotate(f"$n = 6d = {6 * d:,}$".replace(",", "{,}"), (6 * d, 50), xytext=(-8 if left else 8, 8 if left else -8),
                    textcoords="offset points", ha="right" if left else "left", va="bottom" if left else "top",
                    color=color)
    ax.plot(2048, 100 * attention_share(2048, 12288), "s", color=ORANGE, mec="black", ms=6)
    gpt3_share = 100 * attention_share(2048, 12288)
    ax.annotate(f"GPT-3: $n = 2{{,}}048$, {gpt3_share:.1f}%", (2048, gpt3_share), xytext=(700, 24), ha="center",
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8, "shrinkB": 4})
    ax.axhline(50, color=GREY, ls="--", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(256, 262144)
    ax.set_ylim(0, 100)
    ax.set_xlabel("sequence length $n$ (tokens)")
    ax.set_ylabel("attention share of layer work (%)")
    ax.legend(loc="upper left")
    save(fig, out, "attention-crossover")


# Bit layouts and ranges of the page's floating-point table (sec-ch2-floating-point-precision).
def float_formats(out):
    formats = [("fp32", 8, 23, 1.175e-38, 3.403e38, 1.192e-7),
               ("fp16", 5, 10, 6.104e-5, 65504, 9.766e-4),
               ("bf16", 8, 7, 1.175e-38, 3.390e38, 7.812e-3)]
    fig, (bits, ranges) = plt.subplots(2, 1, figsize=(7.2, 4.6), gridspec_kw={"height_ratios": [1, 1]})
    colors = {"sign": "#d9d9d9", "exponent": "#f4b183", "mantissa": "#9fb7d6"}
    for row, (name, exp, man, *_rest) in enumerate(formats):
        start = 0
        for part, width in (("sign", 1), ("exponent", exp), ("mantissa", man)):
            for b in range(width):
                bits.add_patch(plt.Rectangle((start + b, row - 0.35), 1, 0.7, facecolor=colors[part],
                                             edgecolor="white", lw=0.6))
            bits.add_patch(plt.Rectangle((start, row - 0.35), width, 0.7, fill=False, edgecolor=GREY, lw=1))
            if part != "sign":
                bits.text(start + width / 2, row, f"{part} {width}",
                          ha="center", va="center")
            start += width
    bits.set_xlim(0, 32)
    bits.set_ylim(len(formats) - 0.5, -0.5)
    bits.set_yticks(range(len(formats)), [f[0] for f in formats])
    bits.set_xticks([0, 8, 16, 24, 32])
    bits.set_xlabel("bits (sign bit first, grey)")
    bits.grid(False)
    for spine in ("left", "bottom"):
        bits.spines[spine].set_visible(False)
    def sci(value):
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"{value:,.0f}" if 1 <= value < 1e6 else rf"${mantissa} \times 10^{{{int(exponent)}}}$"

    for row, (name, _e, _m, low, high, eps) in enumerate(formats):
        ranges.plot([low, high], [row + 0.12, row + 0.12], color="#3d6aa8", lw=6, solid_capstyle="butt")
        ranges.text(1, row - 0.12, f"{sci(low)} to {sci(high)};  machine epsilon {sci(eps)}", ha="center",
                    va="bottom")
    ranges.set_xscale("log")
    ranges.set_xlim(1e-45, 1e45)
    ranges.set_ylim(len(formats) - 0.5, -0.5)
    ranges.set_yticks(range(len(formats)), [f[0] for f in formats])
    ranges.set_xticks([1e-40, 1e-20, 1, 1e20, 1e40])
    ranges.minorticks_off()
    ranges.set_xlabel("magnitude: smallest normal value to largest value")
    fig.tight_layout(h_pad=1.2)
    save(fig, out, "float-formats")


def notebook_figures(out):
    """Execute the notebook's code cells in order; save the figure of each cell tagged figure:<name>."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    tagged = {}
    for index, cell in enumerate(cells):
        for tag in cell.get("metadata", {}).get("tags", []):
            if tag.startswith("figure:"):
                tagged[index] = tag.split(":", 1)[1]
    missing = set(NOTEBOOK_FIGURES) - set(tagged.values())
    if missing:
        raise SystemExit(f"no notebook cell is tagged figure:{sorted(missing)[0]}")
    current = {"name": None}

    def show(*args, **kwargs):
        if current["name"]:
            save(plt.gcf(), out, current["name"])
            current["name"] = None
        plt.close("all")

    namespace = {"__name__": "__main__"}
    with plt.rc_context(), contextlib.redirect_stdout(io.StringIO()):
        plt.show = show
        for index, cell in enumerate(cells[:max(tagged) + 1]):
            if cell["cell_type"] != "code":
                continue
            current["name"] = tagged.get(index)
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}[{index}]", "exec"), namespace)
            plt.rcParams.update({k: v for k, v in BASE_RC.items() if k.startswith("svg.")})
            if current["name"]:
                raise SystemExit(f"cell {index} tagged figure:{current['name']} never called plt.show()")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    for figure in (softmax_saturation, cross_entropy_decomposition, kaplan_power_laws, allocation_lines,
                   memory_stack, attention_crossover, float_formats):
        with plt.rc_context(BASE_RC):
            figure(out)
    plt.rcParams.update(BASE_RC)
    notebook_figures(out)


if __name__ == "__main__":
    main(sys.argv[1])
