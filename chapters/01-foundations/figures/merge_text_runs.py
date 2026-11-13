"""Merge consecutive <text> runs on one baseline that include a sub/superscript into one <text> of <tspan>s,
so the subscript sits in the same text run as its letter and the browser lays the line out itself.

Graphviz places each run of an HTML-like label at an x position from its own font metrics; when the viewer's font
is narrower, a gap opens before a subscript ("√d k"). Usage: python merge_text_runs.py <file.svg> (edits in place;
lines without a sub/superscript are left as they are, so running it twice changes nothing more).

How to rebuild a schematic of this chapter (tests/test_figures.py allows only figure files and sources here,
so these steps live in this docstring rather than in a README):

The schematics are rendered with the graphviz-studio command-line renderer (`cli/render.ts`, white_paper theme).
From the renderer's directory, with `<figures>` the absolute path of this directory and `<LR|TB>` the layout
direction (keep the one the committed SVG shows: `TB` for a top-to-bottom flow, `LR` for a left-to-right one):

1. Render the SVG:
   `npx tsx cli/render.ts --format svg --theme white_paper --rankdir <LR|TB> -o <figures>/<name>.svg <figures>/<name>.dot`
2. Merge sub- and superscript runs so each subscript is in the same text run as its letter:
   `python merge_text_runs.py <figures>/<name>.svg`
   It changes only lines that contain a sub- or superscript, so it is safe on every schematic. In this round it was
   applied to `scaled_dot_product_attention.svg` (and run on `autoregressive_loop.svg`, which has no subscripts).
   `encoder_decoder.svg` and `multi_head_attention.svg` were rendered before the script existed and have not been
   re-rendered since; apply it when they next change.
3. Convert to the PDF the LaTeX build uses:
   `rsvg-convert -f pdf -o <figures>/<name>.pdf <figures>/<name>.svg`
4. Update the source hash: replace the `<name>.dot` line of `SOURCES.sha256` with the output of
   `sha256sum <name>.dot` (run in this directory), then check with `sha256sum -c SOURCES.sha256`.
5. View the SVG at the width the page shows it (the figure's `:width:` times the page's text width), for example
   with `rsvg-convert -w <pixels> -b white -o <a directory outside the repository>/preview.png <name>.svg`, and check that no label touches an edge.

In `.dot` sources, keep a subscript inside the same HTML-like label as its letter (`d<sub>k</sub>`) and outside
bold text, and give edge labels the node font size (`fontsize=13`).
"""
import re
import sys

path = sys.argv[1]
lines = open(path).read().split("\n")
pat = re.compile(r'<text ([^>]*)>(.*)</text>$')


def attrs(a):
    return dict(re.findall(r'([\w:-]+)="([^"]*)"', a))


out, i = [], 0
while i < len(lines):
    m = pat.match(lines[i])
    if not m:
        out.append(lines[i])
        i += 1
        continue
    group = [m]
    j = i + 1
    while j < len(lines):
        n = pat.match(lines[j])
        if not n or attrs(n.group(1))["y"] != attrs(m.group(1))["y"]:
            break
        group.append(n)
        j += 1
    if len(group) > 1 and any("baseline-shift" in g.group(1) for g in group):
        first = attrs(group[0].group(1))
        keep = {k: v for k, v in first.items() if k not in ("font-weight", "baseline-shift")}
        head = "<text " + " ".join(f'{k}="{v}"' for k, v in keep.items()) + ">"
        spans = []
        for g in group:
            a = attrs(g.group(1))
            extra = " ".join(f'{k}="{a[k]}"' for k in ("font-weight", "baseline-shift") if k in a)
            if "baseline-shift" in a:
                extra += ' font-size="9.75"'
            spans.append(f"<tspan {extra}>{g.group(2)}</tspan>" if extra else f"<tspan>{g.group(2)}</tspan>")
        out.append(head + "".join(spans) + "</text>")
    else:
        out.extend(lines[i:j])
    i = j
open(path, "w").write("\n".join(out))
