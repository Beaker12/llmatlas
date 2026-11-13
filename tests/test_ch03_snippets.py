"""Chapter 3 (data acquisition): per-chapter constants read by the notebook, figure and sync tests."""

# --- Per-chapter constants -----------------------------------------------------------------------------------
# The shared tests read every entry of these through tests/_files.chapter_constants; the chapter changes them
# here and nowhere else.
# SYNC_EXEMPT       name -> reason a definition may differ between the chapter page and its notebook.
#                   Example: {"train_step": "the notebook version also logs the learning rate"}
# RESULT_CLAIMS     (chapter regex, notebook-output regex, comparator) for a number the chapter states and the
#                   notebook prints; comparator eq, gt, lt or approx:<rel>. Captured numbers are normalised, so
#                   10,000 and 1e4 are the same number. Example: (r"loss of (\d+\.\d+)", r"final loss: (\S+)", "approx:0.01")
# NOT_A_RESULT      sentence regex -> reason for a sentence that links a notebook and holds a number but states no
#                   result. Example: {r"parses two files": "a count of inputs, not a result"}
# NOTATION_OK       unit regex -> reason for a sentence the notation check's banned phrases match on purpose.
#                   Example: {r"capacity factor of the paper": "quoted from the Switch paper"}
# MIN_FIGURES       the least number of figures the chapter must have. Example: 12
# INTERFACE_LABELS  label -> text of the heading the label sits directly above, for the section labels other
#                   chapters link to. Example: {"sec-ch2-flops": "Training Complexity"}
# A NOT_A_RESULT or NOTATION_OK entry that matches nothing is a failure: remove it with the sentence it was for.
# The chapter's checker reads every exemption with `python tools/list_exemptions.py chapters/<dir>`.
SYNC_EXEMPT: dict[str, str] = {  # name -> reason it may differ between the chapter and its notebook
    "fetch": "the notebook version also retries once, can send HEAD, and records every fetch (URL, size, hash, "
             "bytes) for the provenance manifest of its last section; the page shows only the GET it needs",
}

# (chapter regex, notebook-output regex, comparator); each regex has exactly one capture group.
# Comparator: eq, gt, lt or approx:<rel>, applied as `chapter_value <op> notebook_value`.
RESULT_CLAIMS: list[tuple[str, str, str]] = [
    (r"\\approx (20\.8)\$T tokens", r"D = (\d+\.\d+)T tokens", "approx:0.01"),
    # \s+ between words: the page's prose inside admonitions is wrapped at 88 columns
    (r"with (428)\s+Disallow\s+rules\s+for\s+every\s+agent", r"Disallow lines: \{'\*': (\d+)\}", "eq"),
    (r"median article \((206\.5) words", r"(?m)^wikipedia\s+\d+\s+(\d+\.\d)", "eq"),
    (r"median page \((188\.5) words", r"(?m)^c4\s+\d+\s+(\d+\.\d)", "eq"),
    (r"rounds\s+it\s+to\s+about\s+(92)\s+TB", r"-> ~(\d+) TB of WARC", "eq"),
]

# regex over a sentence -> one-line reason; such a sentence links a notebook and mentions a number but states
# no result, so the notebook-link check (rule c of test_notebook_prose.py) does not flag it
NOT_A_RESULT: dict[str, str] = {
    r"samples\s+five\s+hosted\s+corpora\s+through\s+the\s+Hugging\s+Face\s+rows\s+API": "the numbers of rows and pulls the notebook "
        "requests from each corpus: sample sizes chosen in advance, not results",
    r"measures\s+domain\s+concentration\s+on\s+2,000\s+C4\s+host\s+names": "the size of the host sample and the number of pulls "
        "are settings of the notebook, not results",
    r"serves\s+only\s+the\s+first\s+2,207,027\s+rows": "a property of the hosting API (the row count the API reports), "
        "stated to explain the offsets, not a result of the notebook",
}

MIN_FIGURES: int = 10
# Units the notation check's banned phrases may match, declared on purpose: unit regex -> reason.
NOTATION_OK: dict[str, str] = {}
# Section labels other chapters link to: label -> the text of the heading it sits directly above.
INTERFACE_LABELS: dict[str, str] = {
    # fixed by the plan's list of eleven interface labels: chapter 4 links it (from its own branch) for the
    # four-epoch result this heading states; the chapter-internal label of 8.2 is sec-ch3-epoch-cap-per-source
    "sec-ch3-epochs": "1.3 How much is enough",
}


# --- Tests -----------------------------------------------------------------------------------------------------
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

FIGURE_SCRIPT = Path(__file__).resolve().parents[1] / "chapters" / "03-data-acquisition" / "figures" / "make_figures.py"


PAGE = FIGURE_SCRIPT.parents[1] / "index.md"

# Exercise 1: Content-Length of ten WARC files of CC-MAIN-2026-39 drawn with random.Random(0).sample(paths, 10),
# measured by HEAD request on 12 November 2026; the published WARC total is 82.79 TiB.
WARC_SIZES = [910_707_757, 966_719_339, 903_957_552, 915_750_155, 911_469_743,
              933_101_733, 896_086_136, 935_203_619, 918_424_076, 939_744_129]


def test_exercise_1_snapshot_size():
    import re
    import statistics

    text = PAGE.read_text()
    shown = [int(n.replace(",", "")) for n in re.findall(r"\b\d{3},\d{3},\d{3}\b", text.split("Solution 1")[1].split("Solution 2")[0])]
    assert shown[:10] == WARC_SIZES and shown[10] == 923_116_424   # the ten sizes, then their mean
    mean, sd = statistics.mean(WARC_SIZES), statistics.stdev(WARC_SIZES)
    total = 100_000 * mean
    assert round(mean) == 923_116_424
    assert round(sd / 1e6, 1) == 20.8
    assert round(total / 1e12, 1) == 92.3 and round(total / 2**40, 2) == 83.96
    assert round(sd / 10**0.5 / 1e6, 1) == 6.6 and round(sd / 10**0.5 / mean * 100, 2) == 0.71
    assert round((total / 2**40 / 82.79 - 1) * 100, 1) == 1.4
    assert 1.5 < (total / 2**40 - 82.79) / (sd / 10**0.5 * 100_000 / 2**40) < 2.5      # about two standard errors
    assert round(sd / 100**0.5 / mean * 100, 1) == 0.2                                 # a hundred files
    assert round(82.79 * 2**40 / 1e12, 1) == 91.0


def _plan_mixture():
    import re

    block = next(b for b in re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S) if "def plan_mixture" in b)
    namespace: dict = {}
    exec(block.split("plan = plan_mixture")[0], namespace)
    return namespace["plan_mixture"], namespace["sources"]


def test_exercise_5_budget_sweep():
    plan_mixture, sources = _plan_mixture()

    def capped(budget):
        return {s for s, (_, _, e) in plan_mixture(budget, sources).items() if e >= 4 - 1e-9}

    def first_capped(name, lo=0.5e12, hi=20e12):
        if name not in capped(hi):
            return None
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if name in capped(mid) else (mid, hi)
        return hi

    assert capped(0.5e12) == set()
    assert round(first_capped("wikipedia") / 1e12, 3) == 0.867
    assert round(first_capped("books") / 1e12, 3) == 3.906
    assert round(first_capped("papers") / 1e12, 2) == 14.25
    assert first_capped("web") is None and first_capped("code") is None
    plan = plan_mixture(20e12, sources)
    shown = {"web": (0.877, 17.54, 1.46), "code": (0.086, 1.71, 1.71), "papers": (0.030, 0.60, 4.00),
             "wikipedia": (0.0013, 0.026, 4.00), "books": (0.006, 0.12, 4.00)}
    for s, (w, t, e) in shown.items():
        digits = len(str(w).split(".")[1])
        assert round(plan[s][0], digits) == w and round(plan[s][1] / 1e12, len(str(t).split(".")[1])) == t
        assert round(plan[s][2], 2) == e
    small = sum(plan[s][0] for s in ("papers", "wikipedia", "books"))
    assert round(small * 100, 1) == 3.7


def test_exercise_6_wilson_interval():
    import math

    n, k, z = 50, 12, 1.96
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert round(p * 100) == 24
    assert (round((center - half) * 100, 1), round((center + half) * 100, 1)) == (14.3, 37.4)
    assert round((2 * half) * 100) == 23
    needed = z * z * p * (1 - p) / 0.05**2
    assert round(needed, 1) == 280.3 and math.ceil(needed) == 281


DATA = FIGURE_SCRIPT.parents[1] / "examples" / "data"


def test_exercise_2_robots_table():
    import re
    from urllib.robotparser import RobotFileParser

    host = (DATA / "news_robots_host.txt").read_text().strip()
    sites = {"wikipedia_robots.txt": "https://en.wikipedia.org/wiki/Language_model",
             "news_robots.txt": f"https://{host}/2026/01/01/world/example.html"}
    answers = {}
    for name, url in sites.items():
        rp = RobotFileParser()
        rp.parse((DATA / name).read_text().splitlines())
        answers[name] = {a: rp.can_fetch(a, url) for a in ("*", "CCBot", "GPTBot", "ClaudeBot", "Google-Extended")}
    assert all(answers["wikipedia_robots.txt"].values())
    assert answers["news_robots.txt"] == {"*": True, "CCBot": False, "GPTBot": False, "ClaudeBot": False,
                                          "Google-Extended": False}
    text = (DATA / "news_robots.txt").read_text()
    groups = [g for g in re.split(r"\n\s*\n", text) if g.strip()]
    agent_lines = re.findall(r"(?im)^user-agent:", text)
    whole_site = [g for g in groups
                  if [ln.strip().lower() for ln in g.strip().splitlines() if not ln.lower().startswith("user-agent")]
                  == ["disallow: /"]]
    assert len(agent_lines) == 69 and len(whole_site) == 55
    blocked = {ln.split(":", 1)[1].strip() for g in whole_site for ln in g.splitlines() if ln.lower().startswith("user-agent")}
    assert {"CCBot", "GPTBot", "ClaudeBot", "Google-Extended"} <= blocked


def test_exercise_3_median_ratios():
    import itertools
    import json
    import statistics

    medians = {n: statistics.median(r["full_words"] for r in json.loads((DATA / f"hf_{n}.json").read_text())["rows"])
               for n in ("c4", "wikipedia", "fineweb2_fra", "fineweb", "fineweb_edu")}
    assert medians == {"c4": 188.5, "wikipedia": 206.5, "fineweb2_fra": 266, "fineweb": 354.5, "fineweb_edu": 387}
    ratios = {(a, b): max(medians[a], medians[b]) / min(medians[a], medians[b])
              for a, b in itertools.combinations(medians, 2)}
    ordered = sorted(ratios, key=ratios.get)
    assert len(ratios) == 10
    assert set(ordered[0]) == {"fineweb", "fineweb_edu"} and round(ratios[ordered[0]], 2) == 1.09
    assert set(ordered[1]) == {"c4", "wikipedia"} and round(ratios[ordered[1]], 2) == 1.10
    assert set(ordered[-1]) == {"c4", "fineweb_edu"} and round(ratios[ordered[-1]], 2) == 2.05
    assert sum(r < 1.10 for r in ratios.values()) == 2


@pytest.mark.slow
def test_make_figures_twice_gives_identical_files(tmp_path):
    """Two runs of the chapter's figure script write byte-identical SVGs (no dates, fixed hash salt)."""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    outputs = []
    for run in ("first", "second"):
        out = tmp_path / run
        subprocess.run([sys.executable, str(FIGURE_SCRIPT), str(out)], check=True, capture_output=True, env=env)
        outputs.append({p.name: p.read_bytes() for p in sorted(out.glob("*.svg"))})
    assert len(outputs[0]) == 6, sorted(outputs[0])
    assert outputs[0] == outputs[1]


NOTEBOOK = FIGURE_SCRIPT.parents[1] / "examples" / "data_acquisition_demo.ipynb"


def test_notebook_commits_its_plots():
    """The committed notebook embeds its figures (executed without MPLBACKEND=Agg), one per plotting cell."""
    import json

    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    images = [c for c in cells if any("image/png" in o.get("data", {}) for o in c["outputs"])]
    plotting = [c for c in cells if "plt." in "".join(c["source"])]
    assert len(plotting) == 2 and images == plotting
    tagged = [c for c in cells if any(t.startswith("figure:") for t in c["metadata"].get("tags", []))]
    assert tagged and all(c in images for c in tagged)


def test_mixture_shares_match_the_papers_tables():
    """Section 1.1: The Pile's shares are its Table 1 "Weight" column (training bytes after upsampling, out of
    1,254.20 GiB; raw corpus 825.18 GiB); Dolma's are its Table 1 Llama-token counts out of 3,059B."""
    import csv

    with open(FIGURE_SCRIPT.parent / "data" / "mixtures.csv", encoding="utf-8") as f:
        rows = {r["name"]: r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))}
    raw = {"Pile-CC": 227.12, "PubMed Central": 90.27, "Books3": 100.96, "OpenWebText2": 62.77}
    epochs = {"Pile-CC": 1.0, "PubMed Central": 2.0, "Books3": 1.5, "OpenWebText2": 2.0}
    weight = {"Pile-CC": 18.11, "PubMed Central": 14.40, "Books3": 12.07, "OpenWebText2": 10.01}
    for k in raw:                                     # Weight = raw size x epochs / effective total
        assert abs(raw[k] * epochs[k] / 1254.20 * 100 - weight[k]) < 0.02, k
    assert round(227.12 / 825.18 * 100) == 28         # Pile-CC as a share of the raw corpus, as the page says
    pile = {"web_crawl": [18.11], "curated_web": [10.01, 5.13, 1.53, 0.62], "books": [12.07, 2.17, 0.75],
            "papers": [14.40, 8.96, 6.12, 3.65, 3.07, 0.38, 0.30], "code": [7.59]}
    for col, parts in pile.items():
        assert round(sum(parts)) == round(float(rows["The Pile"][col])), col
    assert round(1.55 + 1.24 + 0.88 + 0.73 + 0.60 + 0.14) == 5   # components no column holds: "Other"
    dolma = {"web_crawl": (2479, 0), "curated_web": (89 + 4.3, 0), "books": (6.0, 1), "papers": (70, 1),
             "code": (411, 1)}
    for col, (tokens, digits) in dolma.items():
        assert round(tokens / 3059 * 100, digits) == float(rows["Dolma"][col]), col
    assert round(100 - sum(float(rows["Dolma"][c]) for c in dolma), 1) == 0.1   # rounding only
    page = PAGE.read_text()
    assert "out of 1,254 GiB, not of the 825 GiB raw corpus" in page and "Llama tokens out of 3,059B" in page


def _page_block(marker):
    """The source of the page's python listing that contains `marker`."""
    text = PAGE.read_text(encoding="utf-8")
    end = text.index(marker)
    start = text.rindex("```python\n", 0, end) + len("```python\n")
    return text[start:text.index("\n```", end)]


def test_canonicalize_keeps_empty_segments_and_removes_dot_segments(capsys):
    ns = {}
    exec(_page_block("def remove_dot_segments"), ns)  # noqa: S102 - the page's own listing
    canon = ns["canonicalize"]
    # RFC 3986 section 5.2.4 removes only "." and ".."; an empty segment is significant
    assert canon("http://example.com/a//b") == "http://example.com/a//b"
    assert canon("http://example.com//a") == "http://example.com//a"
    assert canon("http://example.com/a/b/..") == "http://example.com/a/"
    assert canon("http://example.com/..") == "http://example.com/"
    assert canon("http://example.com/a/./b/../c") == "http://example.com/a/c"


def test_polite_crawler_waits_through_its_sleep_hook(capsys):
    ns = {}
    exec(_page_block("def remove_dot_segments"), ns)  # noqa: S102 - the page's own listing
    clock = {"t": 100.0}
    slept = []

    def sleep(seconds):          # a real clock that moves only when the crawler waits
        slept.append(seconds)
        clock["t"] += seconds

    bot = ns["PoliteCrawler"](ns["transport"], clock=lambda: clock["t"], sleep=sleep)
    bot.get("https://example.com/a")
    bot.get("https://example.com/b")          # same host, no time passed: must wait 2 s
    bot.get("https://example.org/")           # another host: no wait
    assert slept == [2.0]
    assert bot.now == 0.0                     # the simulated clock is not used when a clock is given


def _notebook_setup_namespace(monkeypatch):
    import json
    nb_path = PAGE.parent / "examples" / "data_acquisition_demo.ipynb"
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    src = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "def fetch(" in c["source"][0]
               + "".join(c["source"]))
    monkeypatch.chdir(nb_path.parent)
    monkeypatch.setenv("LLMATLAS_OFFLINE", "1")
    ns = {}
    exec(src, ns)  # noqa: S102 - the notebook's own setup cell
    return ns


def test_notebook_fetch_spaces_requests_per_host(monkeypatch, capsys):
    ns = _notebook_setup_namespace(monkeypatch)
    clock = {"t": 1000.0}
    slept = []

    class Response:
        headers = {}

        def read(self):
            return b"ok"

    monkeypatch.setattr(ns["time"], "monotonic", lambda: clock["t"])
    monkeypatch.setattr(ns["time"], "sleep", lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
    monkeypatch.setattr(ns["urllib"].request, "urlopen", lambda req, timeout: Response())
    ns["fetch"]("https://a.example/1")
    ns["fetch"]("https://a.example/2")        # same host at once: waits the full delay
    ns["fetch"]("https://b.example/1")        # another host: no wait
    assert slept == [ns["POLITE_DELAY"]] and ns["POLITE_DELAY"] >= 2.0
    assert all("fetched_at" in f for f in ns["FETCHED"])


def test_notebook_fallback_records_carry_the_collection_date(monkeypatch, capsys):
    ns = _notebook_setup_namespace(monkeypatch)
    ns["fetch_or_fallback"]("https://en.wikipedia.org/robots.txt", ns["DATA"] / "wikipedia_robots.txt")
    assert ns["FETCHED"][-1]["fetched_at"] == "2026-10-20"   # examples/data/README.md
    ns["fetch_or_fallback"]("https://example.com/x", ns["DATA"] / "cc_warc_slice.json")
    assert ns["FETCHED"][-1]["fetched_at"] == "2026-11-13"



def test_robots_unreachable_statuses_disallow_everything(capsys):
    ns = {}
    exec(_page_block("def remove_dot_segments"), ns)  # noqa: S102 - the page's own listing
    for status, expected in ((200, True), (404, True), (301, False), (503, False)):
        def transport(url, headers, status=status):
            if url.endswith("/robots.txt"):
                return status, {}, "User-agent: *\nAllow: /\n" if status == 200 else ""
            return 200, {}, "x"
        bot = ns["PoliteCrawler"](transport)
        assert bot.allowed("https://example.com/private/x") is expected, status


def test_sitemap_listing_keeps_undated_entries_and_filters_old_ones(capsys):
    ns = {}
    exec(_page_block("def sitemap_entries"), ns)  # noqa: S102 - the page's own listing
    kind, pages = ns["sitemap_entries"](ns["URLSET"], since="2026-06-01")
    assert kind == "pages"
    assert pages == ["https://example.com/blog/a", "https://example.com/blog/c"]
    kind, kids = ns["sitemap_entries"](ns["INDEX"], since="2026-01-01")
    assert (kind, kids) == ("sitemaps", ["https://example.com/sitemap-blog.xml"])


def test_effective_data_law_listing_prints_what_the_page_shows():
    ns = {}
    exec(_page_block("def effective"), ns)  # noqa: S102 - the page's own listing
    assert [round(ns["effective"](e), 2) for e in (1, 4, 10, 40)] == [1.0, 3.73, 7.81, 15.17]


def test_no_runnable_listing_queries_the_cdx_server():
    """The CDX request is taught as a documented format; nothing the live test runs
    may depend on the unreachable index server."""
    import re
    text = PAGE.read_text(encoding="utf-8")
    for block in re.findall(r"```python\n(.*?)\n```", text, flags=re.S):
        assert "index.commoncrawl.org" not in block
    network = re.findall(r"```python\n# network\b", text)
    assert len(network) == 5  # robots, range fetch, columnar index, crawler, manifest


def test_cdx_reply_example_agrees_with_the_index_line():
    import json
    import re
    text = PAGE.read_text(encoding="utf-8")
    lines = _page_block("INDEX_LINES = [")
    ns = {}
    exec(lines, ns)  # noqa: S102 - the page's own listing
    _, timestamp, fields = ns["INDEX_LINES"][0].split(" ", 2)
    hit = json.loads(fields)
    reply = re.search(r"```text\n(\{\"urlkey\".*?\})\n```", text, flags=re.S).group(1)
    sketch = json.loads("".join(l.strip() for l in reply.splitlines()))
    assert sketch["timestamp"] == timestamp
    for key in ("url", "digest", "length", "offset", "filename"):
        assert sketch[key] == hit[key]
    assert (hit["filename"], int(hit["offset"]), int(hit["length"])) == (
        "crawl-data/CC-MAIN-2026-39/segments/1788492699801.50/warc/"
        "CC-MAIN-20260908083825-20260908113825-00403.warc.gz", 48374322, 9882)


def test_phi4_listing_prints_what_the_page_shows(capsys):
    ns = {}
    exec(_page_block("PHI4 = {"), ns)  # noqa: S102 - the page's own listing
    out = capsys.readouterr().out.splitlines()
    assert out[2].split(":")[1].split(",")[0].strip() == "13.8 epochs"
    assert round(ns["effective"](13.8), 2) == 9.69


def test_exercise_8_bloom_sizing_and_range_header():
    import math
    bits = -math.log(0.01) / math.log(2) ** 2
    assert round(bits, 3) == 9.585
    assert round(bits * 1e9 / 8 / 1e9, 3) == 1.198  # gigabytes
    assert round(bits * math.log(2), 2) == 6.64
    start, length = 48_374_322, 9_882
    assert f"bytes={start}-{start + length - 1}" == "bytes=48374322-48384203"


def test_decay_law_figure_values_match_the_page():
    import math
    r_star = 15.387756
    marginal = {e: math.exp(-(e - 1) / r_star) for e in (4, 10, 40)}
    assert [f"{v:.0%}" for v in marginal.values()] == ["82%", "56%", "8%"]
    ns = {}
    exec(_page_block("def effective"), ns)  # noqa: S102 - the page's own listing
    h = 1e-6
    slope = (ns["effective"](10 + h) - ns["effective"](10 - h)) / (2 * h)
    assert abs(slope - marginal[10]) < 1e-6


def test_manifest_sample_sizes_are_the_exercise_sizes():
    import json
    manifest = json.loads((PAGE.parent / "examples" / "data" / "cc_manifest.json").read_text())
    assert manifest["sample_bytes"] == WARC_SIZES


def test_refresh_listing_prints_what_the_page_shows(capsys):
    ns = {}
    exec(_page_block("def freshness"), ns)  # noqa: S102 - the page's own listing
    out = capsys.readouterr().out
    assert "average freshness 0.372" in out and "average freshness 0.199" in out
    assert "average freshness 0.395" in out
