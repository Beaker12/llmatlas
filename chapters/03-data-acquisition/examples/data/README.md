# Committed samples for the data-acquisition notebook

These files let `data_acquisition_demo.ipynb` run offline (`LLMATLAS_OFFLINE=1`) and reproduce its outputs. They are
small excerpts of third-party data. **They are not covered by the book's own license.** Each file keeps the terms of its
source, listed below, and is included only so that the notebook's analysis can be checked. `checksums.json` holds the
SHA-256 of every sample as committed.

| File | Source | Fetched | License or terms | Attribution |
|---|---|---|---|---|
| `cc_manifest.json` | Common Crawl `CC-MAIN-2026-39`: file count, first five paths of `warc.paths.gz`, and the sizes of the first three from `HEAD` requests | 2026-10-20 | Common Crawl Terms of Use (https://commoncrawl.org/terms-of-use) | Common Crawl Foundation |
| `cc_warc_slice.json` | Common Crawl `CC-MAIN-2026-39`, 256 KB of WARC file 97,227 (seed 5, second draw) from byte 192,487,287, cut to complete records; headers and the first 1,500 characters of each payload | 2026-11-13 | Common Crawl Terms of Use; the captured pages remain the property of their sites' owners | Common Crawl Foundation; the sites named in each `WARC-Target-URI` |
| `wikipedia_robots.txt` | `https://en.wikipedia.org/robots.txt` | 2026-10-20 | Wikimedia Foundation; quoted as a factual record of the site's crawler policy | Wikimedia Foundation |
| `news_robots.txt`, `news_robots_host.txt` | the robots.txt of the news site named in `news_robots_host.txt` | 2026-10-20 | the site's terms; quoted as a factual record of its crawler policy | the site's publisher |
| `hf_fineweb.json` | `HuggingFaceFW/fineweb`, config `sample-10BT`, rows 0-99 via the dataset-viewer rows API; text cut to 1,200 characters | 2026-10-20 | ODC-BY 1.0, and Common Crawl's Terms of Use | Hugging Face (FineWeb); Common Crawl |
| `hf_fineweb_edu.json` | `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, rows 0-99; as above | 2026-10-20 | ODC-BY 1.0, and Common Crawl's Terms of Use | Hugging Face (FineWeb-Edu); Common Crawl |
| `hf_fineweb2_fra.json` | `HuggingFaceFW/fineweb-2`, config `fra_Latn`, rows 0-49; as above | 2026-10-20 | ODC-BY 1.0, and Common Crawl's Terms of Use | Hugging Face (FineWeb-2); Common Crawl |
| `hf_c4.json` | `allenai/c4`, config `en`, rows 0-99; as above | 2026-10-20 | ODC-BY 1.0, and Common Crawl's Terms of Use | Allen Institute for AI; Common Crawl |
| `c4_hosts_sample.txt` | `allenai/c4`, config `en`: host names only (no text) from the `url` column of 20 pulls of 100 rows at offsets drawn with seed 4; the header lines record the seed and offsets. The rows API reports 2,207,027 rows for this split, so the offsets cover that range only | 2026-11-13 | ODC-BY 1.0 | Allen Institute for AI; Common Crawl |
| `hf_wikipedia.json` | `wikimedia/wikipedia`, config `20231101.en`: 5 pulls of 10 rows at offsets drawn with seed 3 (recorded in the file); text cut to 1,200 characters | 2026-11-13 | CC BY-SA 3.0 and GFDL, as stated on the dataset card; each article's history page lists its authors | Wikipedia contributors; Wikimedia Foundation |
