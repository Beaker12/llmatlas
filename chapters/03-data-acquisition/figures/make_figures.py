"""Write the data plots of chapter 3 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

Four plots are drawn from committed files in `figures/data/` (each file's header names its source): the published
mixtures, corpus size by year, the robots.txt restrictions measured by Consent in Crisis, and FineWeb-2's filter
rate by duplicate-cluster size. The fifth comes from the notebook: every code cell of
`examples/data_acquisition_demo.ipynb` is executed in order in one namespace with LLMATLAS_OFFLINE=1 (so it reads the
committed samples instead of the network), and a cell tagged `figure:<slug>` has the figure it shows written to
`<slug>.svg`.

The output is deterministic: Agg backend, fixed hash salt, glyphs as paths, no date metadata, DejaVu Sans.
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

STYLE = {"svg.hashsalt": "llmatlas-ch3", "svg.fonttype": "path", "font.family": "DejaVu Sans"}
matplotlib.rcParams.update(STYLE)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
NOTEBOOK = HERE.parent / "examples" / "data_acquisition_demo.ipynb"


def save(fig, path):
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def mixtures(out):
    """Section 1.1: the five published mixtures as stacked horizontal bars."""
    data = rows("mixtures.csv")
    parts = [("web_crawl", "Web crawl", "#4c72b0"), ("curated_web", "Curated web", "#8fb3de"),
             ("books", "Books", "#dd8452"), ("papers", "Papers", "#55a868"), ("code", "Code", "#8172b3")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    names = [r["name"] for r in data][::-1]
    left = [0.0] * len(names)
    for key, label, color in parts:
        values = [float(r[key]) for r in data][::-1]
        ax.barh(names, values, left=left, color=color, edgecolor="white", linewidth=0.8, label=label)
        for i, (x0, v) in enumerate(zip(left, values)):
            if v >= 6:
                ax.text(x0 + v / 2, i, f"{v:g}%", ha="center", va="center", fontsize=8,
                        color="white" if key in ("web_crawl", "papers", "code") else "black")
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlim(0, 101)
    ax.set_xlabel("share of the mixture (%)")
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, out / "mixtures.svg")


def corpus_sizes(out):
    """Section 1.3: tokens per model or corpus by year, against the estimated stock of public text."""
    data = rows("corpus_sizes.csv")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.axhspan(100e12, 1000e12, xmin=0, xmax=1, color="#fde0c5", zorder=0)
    ax.axhline(300e12, color="#d9822b", linestyle="--", linewidth=1)
    ax.text(2019.6, 300e12, "stock of public text: about 300T tokens\n(90% interval 100T to 1,000T)",
            va="center", fontsize=8, color="#8a4b0f", bbox={"facecolor": "#fde0c5", "edgecolor": "none", "pad": 1})
    ax.axvspan(2026, 2032, ymin=0, ymax=1, color="#eeeeee", zorder=0)
    ax.text(2029, 4e10, "projected\nexhaustion\n2026 to 2032", ha="center", fontsize=8, color="#444444")
    SHIFT = {"FineWeb": -0.15, "Llama 3": 0.15}
    offsets = {"GPT-3": (6, -4), "LLaMA 1": (6, -4), "Dolma": (6, -4), "FineWeb": (-8, -3), "Llama 3": (8, -3)}
    for r in data:
        x, y = int(r["year"]) + SHIFT.get(r["name"], 0), float(r["tokens"])
        marker = "o" if r["kind"] == "trained" else "s"
        ax.scatter([x], [y], marker=marker, color="#2b5d8a", s=36, zorder=3)
        label = f"{r['name']}  {y / 1e12:g}T" if y >= 1e12 else f"{r['name']}  {y / 1e9:g}B"
        dx, dy = offsets[r["name"]]
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    ha="right" if dx < 0 else "left")
    ax.scatter([], [], marker="o", color="#2b5d8a", label="tokens a model was trained on")
    ax.scatter([], [], marker="s", color="#2b5d8a", label="tokens in a released corpus")
    ax.set_yscale("log")
    ax.set_ylim(1e10, 3e15)
    ax.set_xlim(2019.5, 2032.5)
    ax.set_xticks(range(2020, 2033, 2))
    ax.set_xlabel("year")
    ax.set_ylabel("tokens (log scale)")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, out / "corpus-size.svg")


def consent(out):
    """Section 5.3: share of tokens fully restricted by robots.txt, full corpora against their head domains."""
    data = rows("consent_robots_restricted.csv")
    data = [r for r in data if r["month"] >= "2022-07"]
    months = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    tiers = [("head", "#c0392b", "--", "head domains (most-used, best-maintained)"),
             ("full", "#2b5d8a", "-", "full corpus")]
    for tier, color, style, label in tiers:
        for k, line in enumerate(("blue", "orange", "red")):
            values = [float(r[f"{tier}_{line}"]) for r in data]
            ax.plot(months, values, color=color, linestyle=style, linewidth=1.6,
                    label=label if k == 0 else None)
    ticks = [i for i, r in enumerate(data) if r["month"][5:] in ("01", "07")]
    ax.set_xticks(ticks)
    ax.set_xticklabels([("Jan " if data[i]["month"][5:] == "01" else "Jul ") + data[i]["month"][:4] for i in ticks])
    last = len(data) - 1
    head = sorted(float(data[last][f"head_{c}"]) for c in ("blue", "orange", "red"))
    full = sorted(float(data[last][f"full_{c}"]) for c in ("blue", "orange", "red"))
    ax.text(last + 0.5, sum(head) / 3, f"{head[0]:.0f} to {head[-1]:.0f}%", va="center", fontsize=8, color="#c0392b")
    ax.text(last + 0.5, sum(full) / 3, f"{full[0]:.0f} to {full[-1]:.0f}%", va="center", fontsize=8, color="#2b5d8a")
    ax.set_xlim(0, last + 4)
    ax.set_ylim(0, 35)
    ax.set_ylabel("tokens fully restricted (%)")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, out / "consent-restrictions.svg")


def rehydration(out):
    """Section 6.2: FineWeb-2 French removal rate by MinHash cluster size, bars colored by upsampling weight."""
    data = rows("fineweb2_fra_cluster_filtering.csv")
    sizes = [int(r["cluster_size"]) for r in data]
    rates = [float(r["removal_pct"]) for r in data]
    weights = [int(r["weight"]) for r in data]
    cmap = ListedColormap(plt.get_cmap("viridis_r")(range(0, 256, 26))[:10])
    norm = BoundaryNorm([w - 0.5 for w in range(1, 12)], cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.bar(sizes, rates, width=1.0, color=cmap(norm(weights)), linewidth=0)
    ax.axhline(62.4, color="#c0392b", linestyle="--", linewidth=1)
    ax.text(120, 63.6, "global removal rate 62.4%", ha="center", va="bottom", fontsize=8, color="#c0392b")
    low = min(range(len(rates)), key=rates.__getitem__)
    ax.annotate(f"lowest removal, weight 10\n(size {sizes[low]}: {rates[low]:.1f}%)", (sizes[low], rates[low]),
                xytext=(40, 22), textcoords="data", fontsize=8, arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate(f"unique pages\n(size 1: {rates[0]:.1f}%)", (1, rates[0]), xytext=(14, 69), textcoords="data",
                fontsize=8, va="center", arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.annotate("last bar: the most-copied\n0.1% grouped", (sizes[-1], rates[-1]), xytext=(178, 71),
                textcoords="data", fontsize=8, ha="center", arrowprops={"arrowstyle": "->", "lw": 0.8})
    ax.set_xlim(0, sizes[-1] + 1)
    ax.set_ylim(0, 75)
    ax.set_xlabel("MinHash duplicate-cluster size (copies of the document)")
    ax.set_ylabel("documents removed by filters (%)")
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=range(1, 11), pad=0.02)
    bar.set_label("upsampling weight")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, out / "fineweb2-rehydration.svg")


def run_notebook(out):
    """Execute the notebook's code cells offline; save the figure shown by a cell tagged figure:<slug>."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "__main__"}
    current = {"slug": None}

    def show(*args, **kwargs):
        if current["slug"]:
            save(plt.gcf(), out / f"{current['slug']}.svg")
        plt.close("all")

    os.environ["LLMATLAS_OFFLINE"] = "1"
    plt.show = show
    cwd = os.getcwd()
    os.chdir(NOTEBOOK.parent)
    try:
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            tags = [t for t in cell["metadata"].get("tags", []) if t.startswith("figure:")]
            current["slug"] = tags[0].split(":", 1)[1] if tags else None
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}[{index}]", "exec"), namespace)
            matplotlib.rcParams.update(STYLE)
    finally:
        os.chdir(cwd)


def main():
    out = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    mixtures(out)
    corpus_sizes(out)
    consent(out)
    rehydration(out)
    run_notebook(out)


if __name__ == "__main__":
    main()
