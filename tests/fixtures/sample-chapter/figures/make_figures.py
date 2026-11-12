"""Write the data plots of the sample chapter into the directory given as the only argument.

Usage: python make_figures.py <output-directory>

The plot is made by executing the text of the page's plotting fence (the first `python` fence that calls
plt.show), so editing the fence changes the figure. The output is deterministic: Agg backend, fixed hash salt,
glyphs as paths, no date metadata, bundled DejaVu Sans.
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({"svg.hashsalt": "sample-chapter", "svg.fonttype": "path", "font.family": "DejaVu Sans"})

PAGE = Path(__file__).resolve().parents[1] / "index.md"


def plotting_fence(text):
    for match in re.finditer(r"^```python\n(.*?)^```", text, re.M | re.S):
        if "plt.show()" in match.group(1):
            return match.group(1)
    raise SystemExit("no plotting fence found on the page")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plt.show = lambda *args, **kwargs: None
    exec(compile(plotting_fence(PAGE.read_text(encoding="utf-8")), str(PAGE), "exec"), {})
    plt.gcf().savefig(out / "ramp.svg", metadata={"Date": None})
    plt.close("all")


if __name__ == "__main__":
    main(sys.argv[1])
