"""Write the data plots of chapter 6 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

- `kv-bytes.svg` is the notebook's own plot: the code cells of `examples/architecture_demo.ipynb` are executed in
  order, offline, up to and including the last cell tagged `figure:<slug>`, and each figure such a cell shows is
  written to `<slug>.svg`. The training cells come later and are not run.
- `norm-sweep.svg` plots `examples/data/norm_sweep.json`, the recorded learning-rate sweep (decision D22).
- The other plots are computed here from the chapter's own numbers: the committed model configurations in
  `examples/data/configs/`, and the constants stated on the page, which the asserts below check.

The output is deterministic: Agg backend, fixed hash salt, glyphs as paths, no date metadata, DejaVu Sans.
"""
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

STYLE = {"svg.hashsalt": "llmatlas-ch6", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 11,
         "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10, "xtick.labelsize": 10,
         "ytick.labelsize": 10, "figure.dpi": 100}
matplotlib.rcParams.update(STYLE)

CHAPTER = Path(__file__).resolve().parents[1]
NOTEBOOK = CHAPTER / "examples" / "architecture_demo.ipynb"
DATA = CHAPTER / "examples" / "data"
BLUE, RED, GREEN, ORANGE, GREY = "#4c72b0", "#c44e52", "#55a868", "#dd8452", "#8c8c8c"


def save(fig, path):
    """Write `path` (SVG, the HTML figure) and the same figure as PDF beside it (the LaTeX figure)."""
    fig.savefig(path, metadata={"Date": None})
    fig.savefig(Path(path).with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def run_notebook(out):
    """Execute the notebook's code cells up to the last `figure:` cell; save what those cells show."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    tagged = [i for i, c in enumerate(cells)
              if c["cell_type"] == "code" and any(t.startswith("figure:") for t in c["metadata"].get("tags", []))]
    namespace = {"__name__": "__main__"}
    current = {"slug": None}

    def show(*args, **kwargs):
        if current["slug"]:
            save(plt.gcf(), out / f"{current['slug']}.svg")
        plt.close("all")

    plt.show = show
    os.environ["LLMATLAS_OFFLINE"] = "1"
    here = os.getcwd()
    os.chdir(NOTEBOOK.parent)
    try:
        for index, cell in enumerate(cells[:tagged[-1] + 1]):
            if cell["cell_type"] != "code":
                continue
            tags = [t for t in cell["metadata"].get("tags", []) if t.startswith("figure:")]
            current["slug"] = tags[0].split(":", 1)[1] if tags else None
            source = "".join(cell["source"])
            exec(compile(source, f"{NOTEBOOK.name}[{index}]", "exec"), namespace)
            matplotlib.rcParams.update(STYLE)
    finally:
        os.chdir(here)


# --- Section 4.5: which keys each query sees --------------------------------------------------------------------

def attention_patterns(out):
    n, window, sinks = 16, 4, 2
    masks = {
        "full causal": lambda i, j: j <= i,
        f"sliding window, $n_{{win}}$ = {window}": lambda i, j: i - window < j <= i,
        f"{sinks} sink tokens + window of {window}": lambda i, j: j <= i and (j < sinks or i - window < j),
    }
    fig = plt.figure(figsize=(7.4, 4.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.25], hspace=0.45, wspace=0.12)
    for k, (title, rule) in enumerate(masks.items()):
        ax = fig.add_subplot(grid[0, k])
        image = [[1.0 if rule(i, j) else 0.0 for j in range(n)] for i in range(n)]
        ax.imshow(image, cmap=matplotlib.colors.ListedColormap(["#f3f4f6", BLUE]), vmin=0, vmax=1)
        ax.set_title(f"{title}\n{sum(map(sum, image)):.0f} scores", fontsize=10.5)
        ax.set_xticks([0, n - 1], ["0", str(n - 1)])
        ax.set_yticks([0, n - 1], ["0", str(n - 1)] if k == 0 else [])
        ax.set_xlabel("key position")
        if k == 0:
            ax.set_ylabel("query position")
    ax = fig.add_subplot(grid[1, :])
    patterns = {"Gemma 2 9B (42 layers)": ["L" if i % 2 == 0 else "G" for i in range(42)],
                "Gemma 3 12B (48 layers)": ["G" if i % 6 == 5 else "L" for i in range(48)]}
    assert patterns["Gemma 3 12B (48 layers)"].count("G") == 8   # Section 4.5: 8 global, 40 local
    for row, (name, layers) in enumerate(patterns.items()):
        for i, kind in enumerate(layers):
            ax.add_patch(plt.Rectangle((i, -row - 0.8), 0.9, 0.7, color=ORANGE if kind == "G" else "#9fb7d9"))
        ax.text(-1, -row - 0.45, name, ha="right", va="center", fontsize=10)
    ax.set_xlim(-15.5, 48.5)
    ax.set_ylim(-2.0, 0.1)
    ax.axis("off")
    ax.legend(handles=[Patch(color="#9fb7d9", label="local (window) layer"), Patch(color=ORANGE, label="global layer")],
              loc="upper center", bbox_to_anchor=(0.6, 1.45), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.03)
    save(fig, out / "attention-patterns.svg")


# --- Section 5.2: what each rescaling does to each frequency ----------------------------------------------------

def rope_scaling(out):
    d, base, n_train, s = 128, 10000.0, 4096, 8
    beta_slow, beta_fast = 1.0, 32.0
    pairs = list(range(d // 2))
    omega = [base ** (-2 * i / d) for i in pairs]
    pi = [1 / s for _ in pairs]
    ntk_base = base * s ** (d / (d - 2))
    ntk = [ntk_base ** (-2 * i / d) / w for i, w in zip(pairs, omega)]
    psi = [min(1.0, max(0.0, (n_train * w / (2 * math.pi) - beta_slow) / (beta_fast - beta_slow))) for w in omega]
    yarn = [(1 - p) / s + p for p in psi]
    kept, interpolated = sum(p == 1.0 for p in psi), sum(p == 0.0 for p in psi)
    assert (kept, len(pairs) - kept - interpolated, interpolated) == (21, 25, 18)      # Section 5.2's listing
    assert abs(ntk[0] - 1) < 1e-12 and abs(ntk[-1] - 1 / s) < 1e-12                    # NTK-aware identities

    fig, ax = plt.subplots(figsize=(7, 3.9))
    ax.axvspan(-0.5, kept - 0.5, color="#e8f0e8")
    ax.axvspan(len(pairs) - interpolated - 0.5, len(pairs) - 0.5, color="#f6e7e7")
    ax.plot(pairs, pi, color=RED, lw=2, label="position interpolation")
    ax.plot(pairs, ntk, color=BLUE, lw=2, label="NTK-aware base")
    ax.plot(pairs, yarn, color=GREEN, lw=2.4, label="YaRN (NTK-by-parts)")
    ax.set_yscale("log")
    ax.set_yticks([1, 1 / 2, 1 / 4, 1 / 8], ["1", "1/2", "1/4", "1/8"])
    ax.minorticks_off()
    ax.set_ylim(0.1, 1.45)
    ax.set_xlim(-0.5, len(pairs) - 0.5)
    ax.text(kept / 2 - 0.5, 1.2, f"YaRN keeps {kept}", ha="center", fontsize=10)
    ax.text(len(pairs) - interpolated / 2 - 0.5, 1.2, f"divides {interpolated} by 8", ha="center", fontsize=10)
    ax.set_xlabel("pair index i (high frequency at left, low at right)")
    ax.set_ylabel(r"new frequency / old,  $\omega'_i / \omega_i$")
    ax.set_title(r"Extending 4,096 trained positions 8×: $d_k$ = 128, $b$ = 10,000")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), framealpha=1)
    fig.tight_layout()
    save(fig, out / "rope-scaling.svg")


# --- Section 5.4: what the context numbers say ------------------------------------------------------------------

def context_claims(out):
    # (model, [(length, marker meaning)], accepted by the configuration); every number is on the page.
    rows = [
        ("Mistral 7B v0.1", [(4096, "window"), (8192, "trained")], 32768),
        ("Qwen2.5 7B", [(4096, "pretrained"), (32768, "extended")], 131072),
        ("Llama 3 405B", [(8192, "pretrained"), (131072, "extended")], 131072),
        ("DeepSeek-V3", [(4096, "pretrained"), (32768, "extended"), (131072, "extended")], 163840),
    ]
    style = {"window": ("s", GREY), "pretrained": ("o", BLUE), "trained": ("o", BLUE), "extended": ("D", GREEN)}
    fig, ax = plt.subplots(figsize=(7, 3.0))
    for y, (name, points, accepted) in enumerate(reversed(rows)):
        ax.plot([2048, accepted], [y, y], color="#d0d5dd", lw=6, solid_capstyle="butt", zorder=1)
        ax.plot([accepted, accepted], [y - 0.28, y + 0.28], color=RED, lw=2.5, zorder=2)
        for length, meaning in points:
            marker, color = style[meaning]
            ax.plot(length, y, marker=marker, color=color, ms=9, zorder=3, ls="none")
    ax.set_xscale("log", base=2)
    ticks = [4096, 8192, 32768, 131072]
    ax.set_xticks(ticks, ["4K", "8K", "32K", "128K"])
    ax.minorticks_off()
    ax.set_xlim(2048, 260000)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("tokens (log scale)")
    ax.legend(handles=[plt.Line2D([], [], color=RED, lw=2.5, label="config accepts (claim 1)"),
                       plt.Line2D([], [], marker="o", color=BLUE, ls="none", ms=8, label="pretraining length"),
                       plt.Line2D([], [], marker="D", color=GREEN, ls="none", ms=8, label="long-context stage"),
                       plt.Line2D([], [], marker="s", color=GREY, ls="none", ms=8, label="attention window")],
              loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=2, frameon=False)
    ax.set_title("Accepted positions against trained lengths")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.36)
    save(fig, out / "context-claims.svg")


# --- Section 6.4: where the parameters of a sparse model are, and which are active -------------------------------

def moe_breakdown(cfg):
    """Parameter groups of a mixture-of-experts configuration (the chapter's count_params, split by role)."""
    H, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    kind, nh = cfg["model_type"], cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads") or nh
    dh = cfg.get("head_dim") or H // nh
    if kind.startswith("deepseek"):
        dn, dr, dv = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        kv_rank, q_rank = cfg["kv_lora_rank"], cfg["q_lora_rank"]
        q = H * nh * (dn + dr) if q_rank is None else H * q_rank + q_rank + q_rank * nh * (dn + dr)
        attn = q + H * (kv_rank + dr) + kv_rank + kv_rank * nh * (dn + dv) + nh * dv * H
        dense_layers = cfg["first_k_dense_replace"]
        E, k, width = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
        router = H * E + (E if kind == "deepseek_v3" else 0)
        always = dense_layers * 3 * H * cfg["intermediate_size"] + (L - dense_layers) * cfg["n_shared_experts"] * 3 * H * width
    else:
        attn = 2 * H * nh * dh + 2 * H * nkv * dh + (nh * dh + nkv * dh if kind == "olmoe" else 0)
        dense_layers, E = 0, cfg.get("num_local_experts") or cfg["num_experts"]
        k, width, router, always = cfg["num_experts_per_tok"], cfg["intermediate_size"], H * E, 0
    moe_layers = L - dense_layers
    groups = {
        "input embedding": V * H,
        "output head": V * H,
        "attention, norms, routers": L * (attn + 2 * H) + H + moe_layers * router,
        "dense and shared FFN": always,
        "routed experts, active": moe_layers * k * 3 * H * width,
        "routed experts, idle": moe_layers * (E - k) * 3 * H * width,
    }
    return groups


def parameters(out):
    models = [("Mixtral 8x7B", "mistralai__Mixtral-8x7B-v0.1", 46_702_792_704),
              ("OLMoE 1B-7B", "allenai__OLMoE-1B-7B-0924", 6_919_161_856),
              ("DeepSeek-V2-Lite", "deepseek-ai__DeepSeek-V2-Lite", 15_706_484_224),
              ("DeepSeek-V3", "deepseek-ai__DeepSeek-V3", 671_026_419_200)]
    colors = {"input embedding": "#b9c6d9", "output head": "#7f93b3", "attention, norms, routers": BLUE,
              "dense and shared FFN": GREEN, "routed experts, active": ORANGE, "routed experts, idle": "#e7e2da"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for y, (name, file, total) in enumerate(reversed(models)):
        groups = moe_breakdown(json.loads((DATA / "configs" / f"{file}.json").read_text()))
        assert sum(groups.values()) == total, (name, sum(groups.values()))     # Section 8.2's exact totals
        left = 0.0
        for group, value in groups.items():
            ax.barh(y, 100 * value / total, left=left, color=colors[group], edgecolor="white", lw=0.5)
            left += 100 * value / total
        active = total - groups["routed experts, idle"] - groups["input embedding"]
        ax.text(101, y, f"{total / 1e9:.2f}B total\n{100 * active / total:.1f}% active", va="center", fontsize=10)
    ax.set_yticks(range(len(models)), [m[0] for m in reversed(models)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total parameters (%); active excludes the input embedding")
    ax.set_title("Total against active: idle experts are memory, not compute")
    ax.legend(handles=[Patch(color=c, label=g) for g, c in colors.items()], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.82, top=0.91, bottom=0.33)
    save(fig, out / "moe-parameters.svg")


# --- Section 8.3: the weights' 2N against the attention term ----------------------------------------------------

def flops_context(out):
    L, d_attn = 32, 32 * 128
    N = 8_030_261_248 - 128_256 * 4096          # Llama 3.1 8B less its input embedding
    two_n = 2 * N / 1e9
    assert round(two_n, 1) == 15.0 and round(2 * L * 131072 * d_attn / 1e9, 1) == 34.4
    crossover = N / (L * d_attn)
    ns = [2 ** (e / 8) for e in range(8 * 9, 8 * 17 + 1)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(ns, [two_n] * len(ns), color=BLUE, lw=2.2, label=r"weights, $2N$ = 15.0")
    ax.plot(ns, [2 * L * n * d_attn / 1e9 for n in ns], color=RED, lw=2.2, label=r"attention, $2\,L\,n\,d_{attn}$")
    ax.plot(ns, [L * n * d_attn / 1e9 for n in ns], color=RED, lw=1.6, ls="--",
            label="attention, averaged over a causal sequence")
    ax.axvline(4096, color=GREY, lw=1, ls=":")
    ax.text(4096 * 1.08, 40, r"$n = d$ = 4,096:" "\nattention 7% of the total", fontsize=10, va="top")
    ax.plot([crossover], [two_n], "o", color="black")
    ax.annotate(f"equal at $n$ ≈ {crossover / 1000:.0f}K", (crossover, two_n), xytext=(-12, 14),
                textcoords="offset points", ha="right", fontsize=10)
    ax.plot([131072], [2 * L * 131072 * d_attn / 1e9], "o", color=RED)
    ax.annotate("34.4 at 128K", (131072, 34.4), xytext=(-8, 4), textcoords="offset points", ha="right", fontsize=10)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 2048, 8192, 32768, 131072], ["512", "2K", "8K", "32K", "128K"])
    ax.minorticks_off()
    ax.set_xlim(512, 131072 * 1.15)
    ax.set_ylim(0, 40)
    ax.set_xlabel(r"context length $n$ (tokens before the current one)")
    ax.set_ylabel("forward GFLOPs per token")
    ax.set_title("Llama 3.1 8B: when attention outgrows the weights")
    ax.text(600, two_n + 0.8, r"weights, $2N$ = 15.0", color=BLUE, fontsize=10.5)
    ax.text(24000, 27, r"attention, $2\,L\,n\,d_{attn}$", color=RED, fontsize=10.5, ha="right")
    ax.text(90000, 5, "causal\naverage", color=RED, fontsize=10.5, ha="center")
    fig.tight_layout()
    save(fig, out / "flops-context.svg")


# --- Section 2.1: the recorded learning-rate sweep (D22) --------------------------------------------------------

def norm_sweep(out):
    sweep = json.loads((DATA / "norm_sweep.json").read_text())
    rates = sorted({arm["lr"] for arm in sweep["arms"]})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, warmup in zip(axes, (0, 100)):
        for style, color, shift in (("pre", BLUE, 0.94), ("post", RED, 1.06)):
            means = []
            for lr in rates:
                arm = next(a for a in sweep["arms"] if a["lr"] == lr and a["warmup"] == warmup)
                ax.plot([lr * shift] * len(arm[style]), arm[style], "o", color=color, ms=5, alpha=0.75)
                means.append(sum(arm[style]) / len(arm[style]))
            ax.plot([lr * shift for lr in rates], means, color=color, lw=1.5, label=f"{style}-norm (mean of {len(sweep['seeds'])} seeds)")
        ax.set_xscale("log")
        ax.set_xticks(rates, [f"{lr:g}" for lr in rates])
        ax.minorticks_off()
        ax.set_xlabel("learning rate")
        ax.set_title("no warm-up" if warmup == 0 else f"{warmup}-step warm-up")
        ax.grid(axis="y", color="#e5e7eb")
    axes[0].set_ylabel("training loss after 400 steps")
    axes[0].legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    save(fig, out / "norm-sweep.svg")


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for make in (attention_patterns, rope_scaling, context_claims, parameters, flops_context, norm_sweep):
        make(out)
        matplotlib.rcParams.update(STYLE)
    run_notebook(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
