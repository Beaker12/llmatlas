# Data behind the chapter 4 figures

These files hold derived numbers only. No crawled page text and no Wikipedia text is committed.
`make_figures.py` reads them; it never makes a network call.

## refinedweb_stages.json

- Source: Penedo et al. (2023), "The RefinedWeb dataset for Falcon LLM", arXiv:2306.01116, Figure 2
  (https://arxiv.org/pdf/2306.01116). Fetched 2026-11-12.
- Derivation: the eight percentages printed in the figure, copied by hand: the share of the original
  Common Crawl left after each stage and the share each stage removes relative to the stage before it. The
  paper measures the first three stages in documents and the rest in tokens. Multiplying the removal rates
  reproduces the kept column to within rounding.
- Terms: these are factual figures quoted from the paper, with attribution.
- Used by: `retention-funnel`.

## wet_stats.json and fetch_wet_stats.py

- Source 1: bytes 0 to 41,943,039 (the first 40 MiB) of the Common Crawl WET file
  `crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/wet/CC-MAIN-20240220211055-20240221001055-00000.warc.wet.gz`
  (https://data.commoncrawl.org/). Fetched 2026-11-12.
- Source 2: 20 random English Wikipedia articles, 10,774 words in total, fetched as plain text through the
  MediaWiki extracts API on 2026-11-12. These are the reference corpus of the unigram model. Of 60
  requests, 40 were refused with HTTP 429.
- Derivation: `fetch_wet_stats.py` keeps every WET record whose `WARC-Identified-Content-Language` is exactly
  `eng` and that has at least 50 whitespace-separated words. That leaves 4,707 pages. For each page it writes four
  numbers: the word count, the mean word length, the fraction of words containing an alphabetic character
  (two Gopher statistics from section 4.1), and the page's perplexity under the add-0.5 unigram model of
  section 4.3 trained on the Wikipedia articles. The inputs stay outside the repository.
- Terms: Common Crawl data is used under the Common Crawl Terms of Use
  (https://commoncrawl.org/terms-of-use). Wikipedia text is licensed CC BY-SA 4.0. Only the per-page
  statistics above are redistributed, never the text itself.
- Used by: `gopher-stats` and `perplexity-terciles`.
