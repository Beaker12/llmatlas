"""Write the data plots and to-scale drawings of chapter 4 into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

Figures that depend on a listing of the page are made by executing that listing's text (found by a phrase it
contains), so editing the listing changes the figure. Figures built from measured or published numbers read the
committed files in figures/data/ and never make a network call. The output is deterministic: Agg backend, fixed
hash salt, glyphs as paths, no date metadata, DejaVu Sans. Each figure is written as .svg and .pdf.
"""
import contextlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

matplotlib.rcParams.update({
    "svg.hashsalt": "llmatlas-ch4", "svg.fonttype": "path", "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "index.md"
DATA = HERE / "data"
NOTEBOOK = HERE.parent / "examples" / "preprocessing_pipeline_demo.ipynb"
BLUE, ORANGE, GREY, GREEN, RED, INK = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a", "#c53030", "#1a202c"


def listing(phrase):
    """Execute the page's python fence containing `phrase` and return its namespace."""
    text = PAGE.read_text(encoding="utf-8")
    for m in re.finditer(r"^```python\n(.*?)^```", text, re.M | re.S):
        if phrase in m.group(1):
            namespace = {"__name__": "listing"}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(m.group(1), str(PAGE), "exec"), namespace)
            return namespace
    raise SystemExit(f"no listing contains {phrase!r}")


def save(fig, out, name):
    fig.savefig(out / f"{name}.svg", metadata={"Date": None}, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}, bbox_inches="tight")
    plt.close(fig)


def retention_funnel(out):
    stages = json.loads((DATA / "refinedweb_stages.json").read_text(encoding="utf-8"))["stages"]
    colors = {"Document preparation": GREY, "Filtering": BLUE, "Deduplication": ORANGE}
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, [100] * len(stages), color="#edf2f7", height=0.72)
    for yi, s in zip(y, stages):
        ax.barh(yi, s["kept"], color=colors[s["group"]], height=0.72)
        label = f"{s['kept']:.1f}% left, stage removes {s['removed']:.1f}%"
        inside = s["kept"] > 55  # long bars carry their label inside, short ones beside
        ax.text(s["kept"] - 1 if inside else s["kept"] + 1, yi, label, va="center", ha="right" if inside else "left",
                fontsize=11, color="#ffffff" if inside and s["group"] != "Document preparation" else INK)
    ax.set_yticks(y, [s["stage"] for s in stages])
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of the original Common Crawl still present (%)", fontsize=11)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, ["preparation (documents)", "filtering (tokens)", "deduplication (tokens)"],
              loc="upper center", bbox_to_anchor=(0.4, -0.15), ncol=3, fontsize=11, frameon=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, out, "retention-funnel")


def page_blocks(out):
    ns = listing("def main_text(")
    parser = ns["Blocks"]()
    parser.feed(ns["PAGE"])
    parser.flush()
    names = ["<title>", "cookie banner", "<nav> menu", "<h1> headline", "article paragraph 1",
             "article paragraph 2", "share line", "<aside> links", "<footer>"]
    blocks = parser.blocks
    assert len(blocks) == len(names), [t for t, _ in blocks]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.add_patch(Rectangle((15, -0.13), 21, 0.43, color="#c6f6d5", zorder=0))
    ax.text(15.6, 0.27, "kept: at least 15 words,\nlink density at most 0.3", ha="left", va="top", fontsize=9,
            color=GREEN)
    ax.axvline(15, color=GREEN, lw=1, ls="--")
    ax.axhline(0.3, color=GREEN, lw=1, ls="--")
    # Each point carries its number; the key beside the plot names the block. Numbers sit clear of the
    # threshold lines: below the cookie banner (just under 0.3), above everything else.
    below = {1}
    key = []
    for i, ((text, density), name) in enumerate(zip(blocks, names)):
        words = len(text.split())
        kept = words >= 15 and density <= 0.3
        ax.scatter(words, density, s=46, color=GREEN if kept else RED, zorder=3)
        ax.annotate(str(i + 1), (words, density), textcoords="offset points", xytext=(0, -13 if i in below else 6),
                    ha="center", va="bottom" if i not in below else "top", fontsize=9, fontweight="bold",
                    color=GREEN if kept else RED)
        key.append(f"{i + 1}  {name}: {words} words, {density:.2f}")
    for col, part in enumerate((key[:5], key[5:])):
        ax.text(-0.08 + 0.58 * col, -0.2, "\n".join(part), transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color=INK, linespacing=1.5)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, 36)
    ax.set_ylim(-0.13, 1.08)
    ax.set_xlabel("words in the block (stands in for text density)", fontsize=9)
    ax.set_ylabel("link density\n(link characters / characters)", fontsize=9)
    save(fig, out, "page-blocks")


def unicode_forms(out):
    inputs = ["é", "é", "ﬁ", "²"]
    labels = ["e + combining acute", "precomposed é", "ligature fi", "superscript two"]
    forms = ["NFC", "NFD", "NFKC", "NFKD"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_axis_off()
    col_x = [0.0, 2.0, 3.2, 4.4, 5.6, 6.8]
    for j, head in enumerate(["input", "code points"] + forms):
        ax.text(col_x[j] + (0 if j == 0 else 0.55), 4.6, head, fontsize=10, fontweight="bold", ha="left" if j == 0
                else "center")
    for i, (s, label) in enumerate(zip(inputs, labels)):
        y = 3.6 - i
        ax.text(col_x[0], y + 0.12, label, fontsize=9, va="center")
        ax.text(col_x[1] + 0.55, y, " ".join(f"{ord(c):04X}" for c in s), fontsize=8.5, ha="center", va="center",
                family="DejaVu Sans Mono")
        for j, form in enumerate(forms):
            t = unicodedata.normalize(form, s)
            changed = t != s
            ax.add_patch(Rectangle((col_x[j + 2] - 0.02, y - 0.42), 1.14, 0.84,
                                   color="#fefcbf" if changed else "#f7fafc", ec="#4a5568", lw=0.6))
            cx = col_x[j + 2] + 0.55
            if t == "fi":  # two letters set apart, so the font cannot draw its ligature
                for dx, letter in ((-0.07, "f"), (0.07, "i")):
                    ax.text(cx + dx, y + 0.14, letter, fontsize=13, ha="center", va="center")
            else:
                ax.text(cx, y + 0.14, t, fontsize=13, ha="center", va="center")
            ax.text(col_x[j + 2] + 0.55, y - 0.25, " ".join(f"{ord(c):04X}" for c in t), fontsize=8,
                    ha="center", va="center", family="DejaVu Sans Mono")
    ax.text(col_x[0], -0.25, "Shaded cells differ from the input. Code points are hexadecimal (U+ omitted).",
            fontsize=8.5, color="#4a5568")
    ax.set_xlim(-0.05, 8.0)
    ax.set_ylim(-0.5, 4.9)
    save(fig, out, "unicode-forms")


def wet_rows():
    return np.array(json.loads((DATA / "wet_stats.json").read_text(encoding="utf-8"))["rows"])


def gopher_stats(out):
    rows = wet_rows()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    mean_len, alpha = rows[:, 1], rows[:, 2]
    bins = np.arange(2, 14.01, 0.25)
    a1.hist(np.clip(mean_len, 2, 14), bins=bins, color=BLUE)
    for t in (3, 10):
        a1.axvline(t, color=RED, lw=1.2, ls="--")
    out1 = np.mean((mean_len < 3) | (mean_len > 10))
    a1.set_title(f"mean word length: keep 3 to 10\n{out1:.1%} of pages outside", fontsize=9.5)
    a1.set_xlabel("characters per word (clipped at 2 and 14)")
    a1.set_ylabel("pages")
    a2.hist(alpha, bins=np.arange(0, 1.001, 0.02), color=BLUE)
    a2.axvline(0.8, color=RED, lw=1.2, ls="--")
    out2 = np.mean(alpha < 0.8)
    a2.set_title(f"words with a letter: keep at least 0.8\n{out2:.1%} of pages below", fontsize=9.5)
    a2.set_xlabel("fraction of words with an alphabetic character")
    fig.tight_layout()
    save(fig, out, "gopher-stats")


def perplexity_terciles(out):
    rows = wet_rows()
    ppl = rows[:, 3]
    lo, hi = np.percentile(ppl, [100 / 3, 200 / 3])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bins = np.logspace(np.log10(ppl.min()), np.log10(ppl.max()), 60)
    bins = np.sort(np.concatenate([bins, [lo, hi]]))          # the tercile cuts are bin edges
    _, edges, patches = ax.hist(ppl, bins=bins)
    for left, patch in zip(edges[:-1], patches):
        patch.set_color(GREEN if left < lo else BLUE if left < hi else ORANGE)
    counts = [(ppl < lo).sum(), ((ppl >= lo) & (ppl < hi)).sum(), (ppl >= hi).sum()]
    ax.legend([Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, BLUE, ORANGE)],
              [f"{name}: {k:,} pages" for name, k in zip(("head", "middle", "tail"), counts)],
              frameon=False, fontsize=9)
    for t in (lo, hi):
        ax.axvline(t, color=INK, lw=1, ls="--")
        ax.text(t, ax.get_ylim()[1] * 1.0, f"{t:,.0f}", fontsize=8.5, va="bottom", ha="center")
    ax.set_xscale("log")
    ticks = [2000, 3000, 5000, 10000, 20000]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("perplexity under the unigram model (log scale)")
    ax.set_ylabel("pages")
    save(fig, out, "perplexity-terciles")


def pareto_keep(out):
    p = np.linspace(0, 1, 401)
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(p, (2 - p) ** -9.0, color=BLUE, lw=2, label=r"GPT-3, shape 9: $(2-\hat{p})^{-9}$")
    ax.plot(p, (2 - p) ** -3.0, color=ORANGE, lw=1.5, ls="--", label=r"The Pile, shape 3: $(2-\hat{p})^{-3}$")
    for s in (0.0, 0.5, 0.9, 1.0):
        v = (2 - s) ** -9.0
        ax.scatter([s], [v], color=BLUE, zorder=3, s=22)
        ax.annotate(f"{v:.2g}", (s, v), textcoords="offset points", xytext=(-8 if s == 1 else 4, 6),
                    ha="right" if s == 1 else "left", fontsize=8.5)
    ax.set_xlabel(r"classifier score $\hat{p}$")
    ax.set_ylabel("probability the page is kept")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, "pareto-keep")


def minhash_bands(out):
    rng = np.random.default_rng(4)
    b, r = 4, 4
    first = rng.integers(10, 99, size=b * r)
    second = first.copy()
    for k in (1, 10, 11, 13):                       # positions where the minima differ
        second[k] = (first[k] + 37) % 89 + 10
    same = first == second
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.set_axis_off()
    for row, (sig, name) in enumerate(((first, "document 1"), (second, "document 2"))):
        y = 1.6 - row * 0.9
        ax.text(-0.3, y + 0.35, name, ha="right", va="center", fontsize=9.5)
        for k, v in enumerate(sig):
            x = k + (k // r) * 0.5
            ax.add_patch(Rectangle((x, y), 1, 0.7, fc="#c6f6d5" if same[k] else "#fed7d7", ec="#4a5568", lw=0.7))
            ax.text(x + 0.5, y + 0.35, str(v), ha="center", va="center", fontsize=9)
    for band in range(b):
        x0 = band * (r + 0.5)
        ax.add_patch(Rectangle((x0 - 0.1, 0.6), r + 0.2, 1.8, fill=False, ec=INK, lw=1.2))
        match = bool(same[band * r:(band + 1) * r].all())
        ax.text(x0 + r / 2, 2.62, f"band {band + 1}", ha="center", fontsize=9.5, fontweight="bold")
        ax.text(x0 + r / 2, 0.45, "same bucket:\ncandidate pair" if match else "different\nbuckets", ha="center",
                va="top", fontsize=9, color=GREEN if match else RED)
    ax.text(-0.3, -0.45, rf"$n_{{hash}} = br = {b * r}$ minima, $b = {b}$ bands of $r = {r}$ rows; "
            f"{int(same.sum())} of {b * r} positions agree, so the estimate of $J$ is {same.mean():.4g}.",
            fontsize=9.5)
    ax.set_xlim(-2.4, b * r + (b - 1) * 0.5 + 0.2)
    ax.set_ylim(-0.65, 2.85)
    save(fig, out, "minhash-bands")


def notebook_cell(tag):
    """Execute the notebook cell tagged `tag` (with the notebook's SEED, np and plt) and return its namespace."""
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    seed = next(int(m.group(1)) for c in cells for m in [re.search(r"^SEED = (\d+)", "".join(c["source"]), re.M)]
                if c["cell_type"] == "code" and m)
    source = next("".join(c["source"]) for c in cells if tag in c["metadata"].get("tags", []))
    namespace = {"np": np, "plt": plt, "SEED": seed}
    show, plt.show = plt.show, (lambda *args, **kwargs: None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    finally:
        plt.show = show
        plt.close("all")
    return namespace


def lsh_curves(out):
    measured = notebook_cell("figure:lsh-curves")
    j = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(measured["exact_j"], measured["empirical"], "o", mfc="white", mec=INK, ms=5, zorder=4,
            label=rf"measured, $b = {measured['BANDS']}$, $r = {measured['ROWS']}$ "
                  f"({measured['TRIALS']} trials per point, notebook)")
    settings = [(32, 4, BLUE, "-"), (16, 8, ORANGE, "-"), (8, 16, RED, "-"), (14, 8, GREEN, "--")]
    for b, r, color, ls in settings:
        mid = (1 / b) ** (1 / r)
        tag = " (FineWeb)" if (b, r) == (14, 8) else ""
        ax.plot(j, 1 - (1 - j ** r) ** b, color=color, ls=ls, lw=1.8,
                label=rf"$b = {b}$, $r = {r}${tag}: $(1/b)^{{1/r}} = {mid:.2f}$")
        ax.plot([mid], [1 - (1 - mid ** r) ** b], "o", color=color, ms=4)
    ax.set_xlabel(r"Jaccard similarity $J$ of the pair")
    ax.set_ylabel("probability of becoming a candidate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    save(fig, out, "lsh-curves")


def dedup_scope(out):
    rows = ["page A (good)", "page B (good)", "page C (spam)", "page D (spam)", "page E (spam)", "page F (spam)"]
    crawled = {0: [0, 1, 2, 3], 1: [2, 3], 2: [0], 3: [1], 4: [2], 5: [3]}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, scope in zip(axes, ("per snapshot", "global")):
        kept_good = kept_all = 0
        for i, name in enumerate(rows):
            for s in crawled[i]:
                keep = scope == "per snapshot" or s == max(crawled[i])     # global keeps the newest copy
                kept_all += keep
                kept_good += keep and i < 2
                ax.add_patch(Rectangle((s, 5 - i), 0.9, 0.8, fc=(GREEN if i < 2 else ORANGE) if keep else "white",
                                       ec="#4a5568", lw=0.8, hatch=None if keep else "///"))
        ax.set_xlim(-0.1, 4)
        ax.set_ylim(-0.2, 6)
        ax.set_xticks(np.arange(4) + 0.45, [f"snapshot {k + 1}" for k in range(4)], fontsize=8.5)
        ax.set_yticks(np.arange(6)[::-1] + 0.4, rows)
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{scope} deduplication\n{kept_all} copies kept, {kept_good} of them good "
                     f"({kept_good / kept_all:.0%})",
                     fontsize=9.5)
    fig.text(0.5, -0.04, "filled: copy kept; hatched: copy removed as a duplicate", ha="center", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "dedup-scope")


def suffix_array(out):
    ns = listing("def repeated_spans(")
    tokens, (start, end), lic = ns["tokens"], ns["spans"][0], ns["license"]
    n, min_len = len(tokens), len(lic)
    sa = sorted(range(n), key=lambda i: tokens[i:])

    def lcp(i, k):
        m = 0
        while i + m < n and k + m < n and tokens[i + m] == tokens[k + m]:
            m += 1
        return m

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 6.6), gridspec_kw={"height_ratios": [1.5, 2.6]})
    top.set_axis_off()
    split = len(ns["doc_a"])
    first_copy = tokens.index(lic[0])
    for i, tok in enumerate(tokens):
        x, y = (i, 1) if i < split else (i - split, 0)
        in_lic = (first_copy <= i < first_copy + min_len) or (start <= i < end)
        fc = "#fed7d7" if start <= i < end else ("#fefcbf" if in_lic else "#ffffff")
        top.add_patch(Rectangle((x, y * 2.1), 0.96, 1.9, fc=fc, ec="#4a5568", lw=0.6))
        top.text(x + 0.48, y * 2.1 + 1.65, str(i), ha="center", va="center", fontsize=8, color="#4a5568")
        top.text(x + 0.48, y * 2.1 + 0.08, tok, ha="center", va="bottom", rotation=90, fontsize=8)
    top.text(-0.15, 3.05, "document a", ha="right", va="center", fontsize=8.5)
    top.text(-0.15, 0.95, "document b", ha="right", va="center", fontsize=8.5)
    top.add_patch(Rectangle((0, 4.25), 0.6, 0.4, fc="#fefcbf", ec="#4a5568", lw=0.6))
    top.text(0.8, 4.45, f"first copy of the {min_len}-token license (kept)", va="center", fontsize=8.5)
    top.add_patch(Rectangle((9.6, 4.25), 0.6, 0.4, fc="#fed7d7", ec="#4a5568", lw=0.6))
    top.text(10.4, 4.45, f"second copy, tokens {start} to {end - 1} (removed)", va="center", fontsize=8.5)
    top.set_xlim(-2.2, n - split + 0.1)
    top.set_ylim(0, 4.8)
    bottom.set_axis_off()
    first = sa.index(3) - 4
    window = range(max(1, first), min(n, first + 11))
    bottom.text(0.0, 11.4, "rank", fontsize=9, fontweight="bold")
    bottom.text(1.0, 11.4, "start", fontsize=9, fontweight="bold")
    bottom.text(2.0, 11.4, "suffix (first six tokens)", fontsize=9, fontweight="bold")
    bottom.text(9.6, 11.4, "LCP with the row above", fontsize=9, fontweight="bold")
    for row, k in enumerate(window):
        y = 10.4 - row
        common = lcp(sa[k - 1], sa[k])
        hit = common >= min_len
        if hit or (k + 1 < n and lcp(sa[k], sa[k + 1]) >= min_len):
            bottom.add_patch(Rectangle((-0.1, y - 0.3), 12.6, 0.8, color="#fefcbf", zorder=0))
        bottom.text(0.0, y, str(k), fontsize=8.5)
        bottom.text(1.0, y, str(sa[k]), fontsize=8.5)
        bottom.text(2.0, y, " ".join(tokens[sa[k]:sa[k] + 6]), fontsize=8.5)
        bottom.text(10.6, y, str(common), fontsize=8.5, fontweight="bold" if hit else "normal",
                    color=RED if hit else INK)
    bottom.set_xlim(-0.2, 12.6)
    bottom.set_ylim(10.4 - len(window) + 0.3, 12)
    fig.tight_layout()
    save(fig, out, "suffix-array")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for make in (retention_funnel, page_blocks, unicode_forms, gopher_stats, perplexity_terciles, pareto_keep,
                 minhash_bands, lsh_curves, dedup_scope, suffix_array):
        make(out)


if __name__ == "__main__":
    main(sys.argv[1])
