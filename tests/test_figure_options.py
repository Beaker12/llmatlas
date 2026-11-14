"""Directive options survive the parser: every option is one physical line and the figure's alt and caption come out whole.

Fast: each page is parsed with the build's own parser (myst-parser's markdown-it configuration and its directive-text
parser), so the check follows what the build does, not how the source is laid out. For every `{figure}`, the parsed
options must end at a blank line before the caption (an indented continuation line ends the option block early: the alt
is cut, the continuation becomes the caption and the real caption becomes a legend), and no option may leak into the
caption. For every directive of any kind, no indented continuation line may follow an option line.
Slow: build the HTML once and check, for every figure, `img[alt]`, the caption and the legend.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from docutils.parsers.rst.directives.images import Figure
from myst_parser.config.main import MdParserConfig
from myst_parser.parsers import directives as mdirectives
from myst_parser.parsers.mdit import create_md_parser
from markdown_it.renderer import RendererHTML

from tests._files import ROOT
from tools.book_files import pages

OPTION = re.compile(r"^:[\w-]+:")
LEAK = re.compile(r"(^|\s):(width|alt|name|label|align|height|scale|class|target):", re.M)


def book_pages():
    root_pages = [ROOT / "notation.md", ROOT / "intro.md", ROOT / "bibliography.md", ROOT / "roadmap.md"]
    found = [p for p in root_pages if p.exists()]
    for chapter in sorted((ROOT / "chapters").iterdir()):
        if chapter.is_dir():
            found += pages(chapter)
    return found


def directives(path):
    """(line, name, first_line, content) for every directive fence of the page, as the build's parser sees it."""
    md = create_md_parser(MdParserConfig(), RendererHTML)
    for tok in md.parse(path.read_text(encoding="utf-8")):
        if tok.type == "fence" and tok.info.startswith("{"):
            name, _, first = tok.info[1:].partition("}")
            yield (tok.map[0] + 1 if tok.map else 0), name, first.strip(), tok.content


PAGES = book_pages()
IDS = [str(p.relative_to(ROOT)) for p in PAGES]


@pytest.mark.parametrize("path", PAGES, ids=IDS)
def test_directive_options_are_single_lines(path):
    bad = []
    for line, name, _, content in directives(path):
        lines = content.split("\n")
        i = 0
        while i < len(lines) and OPTION.match(lines[i]):
            i += 1
        if i and i < len(lines) and re.match(r"^\s+\S", lines[i]):
            bad.append(f"{path.name}:{line} {{{name}}} option wrapped onto: {lines[i].strip()[:50]!r}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("path", PAGES, ids=IDS)
def test_figures_parse_whole(path):
    bad = []
    for line, name, first, content in directives(path):
        if name != "figure":
            continue
        res = mdirectives.parse_directive_text(Figure, first, content)
        lines = content.split("\n")
        where = f"{path.name}:{line}"
        if res.body and res.body_offset and lines[res.body_offset - 1].strip():
            bad.append(f"{where} option block runs into the caption: {res.body[0][:50]!r}")
        if not res.options.get("alt", "").strip():
            bad.append(f"{where} empty alt")
        elif not re.search(r"[.!?)\w]$", res.options["alt"].strip()):
            bad.append(f"{where} alt ends mid-sentence: {res.options['alt'][-30:]!r}")
        if any(LEAK.search(b) for b in res.body):
            bad.append(f"{where} an option leaked into the caption")
    assert not bad, "\n".join(bad)


@pytest.fixture(scope="module")
def built_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("book")
    exe = shutil.which("jupyter-book", path=str(Path(sys.executable).parent)) or shutil.which("jupyter-book")
    subprocess.run([exe, "build", str(ROOT), "--path-output", str(out), "-n", "--keep-going"],
                   check=True, capture_output=True, text=True)
    return out / "_build" / "html"


@pytest.mark.slow
@pytest.mark.parametrize("path", [p for p in PAGES if "figure}" in p.read_text(encoding="utf-8")],
                         ids=lambda p: str(p.relative_to(ROOT)))
def test_built_figures(built_html, path):
    from bs4 import BeautifulSoup
    html = built_html / path.relative_to(ROOT).with_suffix(".html")
    soup = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser")
    bad = []
    for _, name, first, content in directives(path):
        if name != "figure":
            continue
        res = mdirectives.parse_directive_text(Figure, first, content)
        label = res.options["name"]
        fig = soup.find("figure", id=label)
        if fig is None:
            bad.append(f"{label}: not in the built page")
            continue
        alt = " ".join(res.options["alt"].split())
        if " ".join(fig.find("img").get("alt", "").split()) != alt:
            bad.append(f"{label}: img alt differs from the source alt")
        caption = " ".join(fig.find("span", class_="caption-text").get_text().split())
        sentence = " ".join(res.body[0].split()) if res.body else ""
        plain = re.split(r"[$`{*_'\"\\\[<]", sentence)[0]  # up to the first markup, quote or math
        if not caption.startswith(plain):
            bad.append(f"{label}: caption does not start with the source caption: {caption[:40]!r}")
        legend = fig.find("figcaption").find("div", class_="legend") if fig.find("figcaption") else None
        for text in (caption, legend.get_text() if legend else ""):
            if LEAK.search(text):
                bad.append(f"{label}: option leaked into the caption or legend")
    assert not bad, "\n".join(bad)
