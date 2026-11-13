"""Write the data plots of chapter 5 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

The plots come from the chapter's notebook. Every code cell of `examples/dataset_management_demo.ipynb` is executed
in order in one namespace (the notebook takes about ten seconds). A cell tagged `figure:<slug>` has each figure it
shows written to `<slug>.svg`, so editing that cell changes the figure. One further plot, the epoch-cap timeline,
is drawn here from the draws the notebook's epoch-cap cell made (`stream`), because the notebook only prints it.

The output is deterministic: Agg backend, fixed hash salt, glyphs as paths, no date metadata, bundled DejaVu Sans.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {"svg.hashsalt": "llmatlas-ch5", "svg.fonttype": "path", "font.family": "DejaVu Sans"}
matplotlib.rcParams.update(STYLE)

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "dataset_management_demo.ipynb"


def save(fig, path):
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def run_notebook(out):
    """Execute the notebook's code cells; save the figures shown by cells tagged figure:<slug>."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace = {"__name__": "__main__"}
    current = {"slug": None}

    def show(*args, **kwargs):
        if current["slug"]:
            save(plt.gcf(), out / f"{current['slug']}.svg")
        plt.close("all")

    plt.show = show
    os.chdir(NOTEBOOK.parent)
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        tags = [t for t in cell["metadata"].get("tags", []) if t.startswith("figure:")]
        current["slug"] = tags[0].split(":", 1)[1] if tags else None
        source = "".join(cell["source"])
        exec(compile(source, f"{NOTEBOOK.name}[{index}]", "exec"), namespace)
        matplotlib.rcParams.update(STYLE)
    return namespace


def epoch_cap_timeline(ns, out):
    """Cumulative draws per source in the notebook's capped stream, with each source's cap."""
    stream, sizes, cap = ns["stream"], ns["sizes"], 4
    names = list(sizes)
    counts = {name: 0 for name in names}
    curves = {name: [] for name in names}
    for source, _ in stream:
        counts[source] += 1
        for name in names:
            curves[name].append(counts[name])
    first_cap = ns["first_cap"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    colors = {"web": "#1f77b4", "wiki": "#d62728", "code": "#2ca02c"}
    steps = range(1, len(stream) + 1)
    for name in names:
        color = colors.get(name)
        ax.plot(steps, curves[name], color=color, label=f"{name} ({sizes[name]} docs)")
        if cap * sizes[name] <= max(curves[n][-1] for n in names) * 1.1:
            ax.axhline(cap * sizes[name], color=color, linestyle=":", linewidth=1)
            ax.annotate(f"{name} cap = {cap} × {sizes[name]} = {cap * sizes[name]}", (0, cap * sizes[name]),
                        xytext=(6, 3), textcoords="offset points", ha="left", va="bottom", fontsize=8, color=color)
    ax.axvline(first_cap, color="black", linestyle="--", linewidth=1)
    ax.annotate(f"wiki capped at step {first_cap}", (first_cap, 0), xytext=(-4, 6), textcoords="offset points",
                ha="right", fontsize=8)
    ax.set_xlabel("sample drawn (step)")
    ax.set_ylabel("documents drawn so far")
    ax.set_title("Epoch cap at 4 epochs: once wiki is exhausted, only web and code are drawn")
    ax.title.set_fontsize(10)
    ax.set_xlim(0, len(stream))
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, out / "epoch-cap.svg")


SIX_DOCS = [11, 7, 5, 9, 3, 13]   # lengths in tokens, each including its end-of-document token
SEQ = 16                          # sequence length n of the example
DOC_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#9c755f"]


def best_fit_decreasing(lengths, n):
    """Bins of (doc, start, length) pieces: longest piece first, into the fullest bin it fits."""
    pieces = sorted(((d, s, min(n, length - s)) for d, length in enumerate(lengths) for s in range(0, length, n)),
                    key=lambda p: -p[2])
    bins, free = [], []
    for piece in pieces:
        fits = [i for i, room in enumerate(free) if piece[2] <= room]
        if fits:
            i = min(fits, key=lambda k: free[k])
            bins[i].append(piece)
            free[i] -= piece[2]
        else:
            bins.append([piece])
            free.append(n - piece[2])
    return bins


def packing_schematic(out):
    """The same six documents laid out by concatenate-and-chunk, by padding, and by best-fit packing."""
    chunk_rows, pos = [[] for _ in range(sum(SIX_DOCS) // SEQ)], 0
    for d, length in enumerate(SIX_DOCS):
        for t in range(pos, pos + length):
            row = t // SEQ
            if row < len(chunk_rows):
                split = pos // SEQ != (pos + length - 1) // SEQ
                chunk_rows[row].append((d, t % SEQ, split))
        pos += length
    pad_rows = [[(d, t, False) for t in range(length)] for d, length in enumerate(SIX_DOCS)]
    pack_rows = []
    for b in best_fit_decreasing(SIX_DOCS, SEQ):
        cells, col = [], 0
        for d, _, length in b:
            cells += [(d, col + t, False) for t in range(length)]
            col += length
        pack_rows.append(cells)
    panels = [("Concatenate and chunk", chunk_rows), ("Pad (one document per sequence)", pad_rows),
              ("Best-fit pack", pack_rows)]
    total_rows = sum(len(rows) for _, rows in panels) + 1.4 * len(panels)
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * (total_rows + 0.6) / SEQ * 0.62))
    y = 0.0
    for title, rows in panels:
        used = sum(len(r) for r in rows)
        splits = len({d for r in rows for d, _, s in r if s})
        stats = (f"{len(rows)} sequences, {SEQ * len(rows) - used} padding tokens, "
                 f"{splits} document{'' if splits == 1 else 's'} split")
        ax.text(0, y - 0.45, f"{title}: {stats}", fontsize=9, va="center")
        y -= 1.0
        for cells in rows:
            for col in range(SEQ):
                ax.add_patch(plt.Rectangle((col, y - 0.8), 1, 0.8, facecolor="#eeeeee", edgecolor="#4a5568", lw=0.5))
            for i, (d, col, split) in enumerate(cells):
                ax.add_patch(plt.Rectangle((col, y - 0.8), 1, 0.8, facecolor=DOC_COLORS[d], edgecolor="#4a5568",
                                           lw=0.5, hatch="///" if split else None))
                if i == 0 or cells[i - 1][0] != d:
                    ax.text(col + 0.5, y - 0.4, f"D{d + 1}", ha="center", va="center", fontsize=8, color="white",
                            fontweight="bold")
            y -= 1.0
        y -= 0.4
    ax.set_xlim(-0.05, SEQ + 0.05)
    ax.set_ylim(y + 0.3, 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Six documents of {', '.join(map(str, SIX_DOCS))} tokens (each with its EOD), n = {SEQ}",
                 fontsize=10)
    fig.tight_layout()
    save(fig, out / "packing-schematic.svg")


def main(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    packing_schematic(out)
    namespace = run_notebook(out)
    epoch_cap_timeline(namespace, out)


if __name__ == "__main__":
    main(sys.argv[1])
