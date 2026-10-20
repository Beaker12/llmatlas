# Data Acquisition and Sources

Every token a language model is trained on was written by someone, stored somewhere, and copied into the training set by a decision that a person or a pipeline made. This chapter is about those decisions: where the text that trains a modern model comes from, how it is obtained at the scale of trillions of tokens, what may legally and ethically be used, and how the pieces are combined into a corpus. Chapter 4 picks up where this one stops, with the cleaning that turns raw acquisitions into training-ready text, and Chapter 5 with keeping the result versioned and reproducible.

## Learning Objectives

By the end of this chapter, you will understand:

- What the pretraining corpora of GPT-3, the LLaMA family, The Pile, Dolma and FineWeb are made of, and roughly in what proportions
- How Common Crawl is organised (snapshots, segments, WARC/WET/WAT files, the index) and how to read a slice of it
- How to crawl responsibly: robots.txt, rate limits, sitemaps, and the economics of fetching a billion pages
- The main curated sources (Wikipedia, books, scientific papers, source code, forums and dialogue) and what each contributes
- The copyright, licensing and consent questions around training data, and how the major open corpora answer them
- How to turn a token budget into a mixture of sources, and how to document what was collected

## Hands-On Learning

The accompanying notebook, [Data Acquisition Demo](examples/data_acquisition_demo.ipynb), fetches small samples from real public sources and checks every claim with an assertion:

- Reading a Common Crawl snapshot's manifest and parsing records from a byte-range slice of a WARC file ({ref}`nb-acq-crawl`)
- Checking robots.txt rules for search and AI crawler user agents on an encyclopaedia and a news site ({ref}`nb-acq-robots`)
- Sampling FineWeb, FineWeb-Edu, C4, Wikipedia and FineWeb-2 through the Hugging Face rows API and comparing document lengths, language scores and domain concentration ({ref}`nb-acq-sources`)
- Turning a compute budget into a token budget and a source mixture with per-source epoch caps ({ref}`nb-acq-mixture`)
- Writing a provenance manifest with content hashes for everything fetched ({ref}`nb-acq-provenance`)

Everything runs on a CPU in about a minute. The network fetches total a few megabytes; if a source is unreachable, or if `LLMATLAS_OFFLINE=1` is set, the notebook falls back to committed samples under `examples/data/` so that the results, and the assertions, are reproducible.

## Chapter Overview

Imagine being asked to assemble a library for someone who will read every book in it exactly once and then has to answer any question anyone asks. The questions are not known in advance. The budget is enough for a few trillion words. The books have to be obtained lawfully, and whoever wrote them may have opinions about the arrangement.

**The web** is the only source large enough to fill most of the shelves. Common Crawl, a non-profit that has crawled the web since 2008 and publishes the result for free, supplies the raw material for nearly every open corpus and, by all accounts, for most closed ones.

**Curated sources** fill the gaps the web leaves: encyclopaedias for facts, books for long-form prose, scientific papers for technical reasoning, source code for programming, and conversation for dialogue. Each is a few per cent of the corpus and each is upweighted far beyond its size.

**Crawling** is how new material is obtained when the published crawls are not enough. It is a matter of HTTP mechanics, politeness conventions that are rapidly hardening into legal lines, and arithmetic about how many machines it takes to fetch a billion pages.

**Rights and consent** decide what may be kept. Copyright, dataset licences, robots.txt and opt-out registries all apply, and the answer has changed quickly: a quarter of the highest-quality web text became off limits to crawlers in a single year.

**Mixture design** turns the pile of sources into a training set. The scaling laws of Chapter 2 say how many tokens a model should see; the mixture says which ones, how many times, and in what proportion.

**Provenance** is the record that lets anyone, including you in a year's time, find out what is in the corpus and where it came from.

Acquisition is where a model's knowledge, its languages, its biases and its legal exposure are all fixed. The cleaning in Chapter 4 can remove what should not have been collected; it cannot add what was never acquired.

## 1. What a Pretraining Corpus Is Made Of

### 1.1 The published mixtures

The training sets of the major models are reported, when they are reported at all, as a table of sources with a share of the training tokens. The table below collects the published mixtures. For GPT-3 and LLaMA the shares are of training tokens, which differ from shares of the raw data because sources are sampled at different rates; for The Pile they are shares of the corpus by bytes, and for Dolma and FineWeb shares of the corpus by tokens.

| Model or corpus                                  | Web crawl                    | Curated web (Wikipedia, forums, news) | Books          | Papers           | Code            | Total tokens |
| ------------------------------------------------ | ---------------------------- | ------------------------------------- | -------------- | ---------------- | --------------- | ------------ |
| GPT-3 {cite}`brown2020language`                  | 60% (filtered Common Crawl)  | 22% WebText2, 3% Wikipedia            | 16% (two book corpora) | included in web | not separate | 300B |
| The Pile {cite}`gao2020pile`                     | 18% Pile-CC                  | 17% (OpenWebText2, StackExchange, Wikipedia, HN) | 15% (Books3, Gutenberg, BookCorpus2) | 37% (PubMed Central, arXiv, FreeLaw, USPTO, PubMed abstracts, …) | 8% GitHub | 825 GB (~300B tokens) |
| LLaMA 1 {cite}`touvron2023llama`                 | 67% CommonCrawl + 15% C4     | 4.5% Wikipedia, 2% StackExchange      | 4.5% (Gutenberg, Books3) | 2.5% arXiv | 4.5% GitHub | 1.4T |
| Dolma {cite}`soldaini2024dolma`                  | 80% (Common Crawl)           | 3% (Reddit, Wikipedia, Wikibooks)     | 0.2% (Gutenberg) | 2.3% (peS2o)     | 13% (The Stack) | 3T |
| FineWeb {cite}`penedo2024fineweb`                | 100%                         |                                       |                |                  |                 | 15T |

Three things stand out. The web is the bulk of every corpus, because nothing else is large enough. Curated sources are small in tokens but upweighted: Wikipedia is 3% of GPT-3's training tokens while being a tiny fraction of the raw data, and LLaMA samples it for more than two epochs. And the totals have grown fifty-fold in four years, from GPT-3's 300B tokens to the 15T of FineWeb and the roughly 15T used by Llama 3, which is why the question of whether the supply of human-written text is finite {cite}`villalobos2024run` is now taken seriously.

### 1.2 Tokens, documents and bytes

Corpus sizes are quoted in three units and the conversions are rough:

- **Bytes** are what storage and transfer cost. A Common Crawl snapshot is 80 to 100 TB compressed as WARC and about a tenth of that as extracted text.
- **Documents** are the unit of acquisition and deduplication. A snapshot holds two to three billion pages.
- **Tokens** are what training cost and the scaling laws count. English prose runs about four characters per token with a GPT-style tokenizer, so one terabyte of UTF-8 text is roughly 250 billion tokens; code and non-Latin scripts tokenize less efficiently.

The retention figures in Chapter 4 apply on top of this: of a snapshot's two billion pages, a cleaned English corpus keeps perhaps a tenth. Fifteen trillion tokens is therefore not one snapshot but ninety-six of them (FineWeb), each deduplicated on its own; FineWeb found that deduplicating across snapshots removed the good pages that recur in every crawl and kept the junk that appears once.

### 1.3 How much is enough

Chapter 2 derived the compute-optimal token count: for a budget of $C$ FLOPs and a model of $N$ parameters, $C \approx 6ND$ and the Chinchilla optimum is about $D \approx 20N$ tokens {cite}`hoffmann2022training`. Production models train well past that point, because inference cost scales with $N$ and not with $D$, so a smaller model trained longer is cheaper to serve. Llama 3's 8B model saw about 15T tokens, almost two thousand per parameter.

Two results set the limits of stretching a fixed corpus:

- **Repetition is cheap up to a point.** Training for up to about four epochs on the same data costs almost nothing in final loss relative to fresh data; beyond that the returns fall off steeply {cite}`muennighoff2023scaling`. A corpus of 1T unique tokens can support a 4T-token training run, not a 40T one.
- **The stock of public text is finite.** Villalobos et al. {cite}`villalobos2024run` estimate the stock of human-written public text at about 300 trillion tokens (a 90% interval of 100T to 1,000T) and project that training runs will use all of it at some point between 2026 and 2032.

Both results bear directly on acquisition. If repeated tokens are nearly free, then a small, high-quality source is worth collecting even though it will be seen many times. If the stock is finite, then every source lost to opt-outs or licensing is a permanent reduction, and synthetic data (Section 7) stops being optional.

## 2. The Open Web: Common Crawl

### 2.1 What Common Crawl is

Common Crawl is a non-profit foundation that has crawled the web since 2008, on a regular (roughly monthly to bi-monthly) schedule since 2013, and publishes every snapshot, free of charge, on Amazon S3 under the Open Data program. Each snapshot is a sample of the web, not a copy: a few billion pages selected by a crawl frontier that balances breadth across domains against depth within them, fetched over two to three weeks, with per-domain page budgets derived from a harmonic-centrality ranking of hosts so that no single site dominates. Common Crawl is the raw material for C4, RefinedWeb, FineWeb, Dolma's web portion, OSCAR, mC4 and CulturaX, and for the web portion of nearly every model whose training data has been described. A critical analysis of it as a training source {cite}`baack2024critical` is worth reading before building on it: the crawl over-represents English, under-represents the parts of the web behind login walls or heavy JavaScript, and, being a sample, is not the web.

### 2.2 Snapshots, segments and files

A snapshot is identified by a crawl id such as `CC-MAIN-2026-39` (the thirty-ninth ISO week of 2026). Under `crawl-data/<crawl-id>/` it is divided into a hundred segments, and each segment into files of three kinds:

| File type | Contents                                                                     | Size per snapshot | Use                                           |
| --------- | ---------------------------------------------------------------------------- | ----------------- | --------------------------------------------- |
| **WARC**  | Raw HTTP requests and responses, including headers and the full HTML payload | ~90 TB            | Re-extracting text with your own extractor    |
| **WET**   | Plain text that Common Crawl's own extractor pulled from each page           | ~9 TB             | Quick corpora (C4, OSCAR, mC4 used WET)       |
| **WAT**   | JSON metadata per page: links, headers, title, language guess                | ~20 TB            | Link graphs, URL filtering, domain statistics |

A WARC file is a sequence of gzip members, one per record, so that any record can be decompressed on its own once its byte offset is known. Each record has a header block (`WARC-Type`, `WARC-Target-URI`, `WARC-Date`, a content digest, the detected payload type) followed by the payload; a fetched page produces three records, `request`, `response` and `metadata`. The manifest files `warc.paths.gz`, `wet.paths.gz` and `wat.paths.gz` list every file in the snapshot, which is how a job divides the snapshot across machines.

```python
import gzip
import json
import urllib.request
import zlib

BASE = "https://data.commoncrawl.org/"
HEADERS = {"User-Agent": "llmatlas-notebook/0.1"}


def fetch(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={**HEADERS, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout).read()


def warc_paths(crawl_id):
    """The manifest: one line per WARC file in the snapshot."""
    return gzip.decompress(fetch(f"{BASE}crawl-data/{crawl_id}/warc.paths.gz")).decode().splitlines()


def warc_records(blob):
    """Split a byte string of concatenated gzip members into complete WARC records."""
    records, pos = [], 0
    while pos < len(blob):
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            chunk = d.decompress(blob[pos:])
        except zlib.error:
            break
        if not d.eof:  # the last member in a byte-range slice is usually cut off
            break
        records.append(chunk)
        pos = len(blob) - len(d.unused_data)
    return records


def parse_record(record):
    head, _, payload = record.partition(b"\r\n\r\n")
    headers = dict(
        line.split(": ", 1) for line in head.decode("utf-8", "replace").splitlines()[1:] if ": " in line
    )
    return headers, payload


paths = warc_paths("CC-MAIN-2026-39")
first_256k = fetch(BASE + paths[0], {"Range": "bytes=0-262143"})
records = [parse_record(r) for r in warc_records(first_256k)]
print(len(paths), "WARC files;", len(records), "complete records in the first 256 KB")
print({h.get("WARC-Type") for h, _ in records})  # {'warcinfo', 'request', 'response', 'metadata'}
```

The `Range` request is the key technique: it fetches a few hundred kilobytes out of a gigabyte-sized file, which makes it possible to look at a snapshot without downloading it. Production pipelines stream whole files, and the same `warc_records` logic, written in a streaming form, is the inner loop of every WARC reader.

### 2.3 The index and targeted retrieval

Common Crawl also publishes a URL index for each snapshot, queryable through the CDX API (`index.commoncrawl.org/<crawl-id>-index?url=…&output=json`), which returns the WARC filename, byte offset and length of every capture of a URL or URL prefix. Combined with a range request this retrieves a single page from a 90 TB snapshot in one round trip. It is how researchers study a particular site's history across snapshots, and how a corpus builder can pull every capture of a chosen list of domains without scanning the whole snapshot. The index server is shared and rate limited, so bulk use goes through the columnar index files (`cc-index-table`, in Parquet on S3) and a query engine rather than the HTTP API.

### 2.4 Corpora derived from Common Crawl

Most open web corpora are a choice of snapshots plus a choice of pipeline:

| Corpus                                              | Snapshots      | Extraction      | Language   | Notable choices                                                                   |
| --------------------------------------------------- | -------------- | --------------- | ---------- | --------------------------------------------------------------------------------- |
| C4 {cite}`raffel2019exploring`                      | 1 (April 2019) | WET             | English    | Strict line-level rules; a bad-words list that also removed benign text {cite}`dodge2021documenting` |
| mC4 {cite}`xue2021mt5`                              | 71             | WET             | 101 languages | Per-language C4 rules with a line-length rule instead of terminal punctuation  |
| OSCAR {cite}`abadji2022oscar`                       | 1 per release  | WET             | 150+       | Language id per line, then per document; minimal filtering by design              |
| RefinedWeb {cite}`penedo2023refinedweb`             | 90             | WARC + trafilatura | English | Web only; argued that enough filtered web matches curated mixtures                |
| FineWeb {cite}`penedo2024fineweb`                   | 96             | WARC + trafilatura | English | Every filter ablated with 1.7B-parameter training runs; 15T tokens                |
| FineWeb-Edu {cite}`penedo2024fineweb`               | as FineWeb     | as FineWeb      | English    | Classifier for educational value, 1.3T tokens                                     |
| CulturaX {cite}`nguyen2023culturax`                 | mC4 + OSCAR    | WET             | 167        | Merge and re-clean two multilingual corpora                                       |
| FineWeb-2 {cite}`penedo2025fineweb2`                | 96             | WARC + trafilatura | 1,000+ languages | Per-language filter thresholds derived automatically from reference-corpus statistics; duplicates "rehydrated" in proportion to cluster size |
| Dolma web {cite}`soldaini2024dolma`                 | 24             | WET (CCNet)     | English    | Part of a documented multi-source corpus with released tooling                    |

Two trends run through the table. Later corpora re-extract from WARC with a boilerplate-aware extractor rather than trusting the WET text, and later corpora use far more snapshots, because one snapshot cleans down to a few hundred billion tokens and trillion-token corpora need dozens. FineWeb's ablations add a caution: deduplicating each snapshot separately gave better models than deduplicating across all of them, since global deduplication preferentially removed the good pages that every crawl re-captures.

```{admonition} A Slice of the Crawl
:class: tip
The notebook section {ref}`nb-acq-crawl` downloads the manifest of a current snapshot, range-fetches the first 256 KB of its first WARC file, parses the records, tabulates record types and declared payload types, and shows the HTTP response and HTML of one page, so that you can see exactly what a crawl contains before any cleaning.
```

## 3. Crawling the Web Yourself

### 3.1 When a published crawl is not enough

Common Crawl covers the open, static, mostly English web as of a few weeks ago. Teams crawl for themselves when they need freshness (news, prices, documentation that changes), depth on a chosen set of sites (every page of a documentation site rather than the few a broad crawl sampled), content that needs a browser to render, or languages and regions the broad crawl under-samples. The cost is real: Common Crawl's snapshot of two to three billion pages represents two to three weeks of fetching at a sustained rate of one to two thousand pages per second, and a team crawling at that rate has to run the politeness and legal machinery below itself.

### 3.2 robots.txt and the politeness conventions

The Robots Exclusion Protocol (RFC 9309) lets a site state, in `/robots.txt`, which user agents may fetch which paths. It was a convention for thirty years and became a standard in 2022; it is not enforced by anything but the crawler's own code, which is why it has become the centre of the consent question in Section 5. A crawler is expected to:

- Fetch `/robots.txt` before anything else on a host, cache it, and obey the `Disallow` and `Allow` rules for its own user agent, falling back to the `*` group.
- Honour `Crawl-delay` where present (it is a widely used extension, not part of RFC 9309) and otherwise limit itself to a request every few seconds per host, with backoff on `429` and `503` responses.
- Identify itself with a user agent string that names the crawler and links to a page explaining it, so that site operators can block it or write to its operator.
- Use the `Sitemap:` entries, which list the URLs a site wants crawled, as the frontier for that site.

Python's standard library parses the file:

```python
import urllib.request
import urllib.robotparser

# Fetch with a named user agent: RobotFileParser.read() would send Python's default
# agent, which Wikipedia answers with 403, and the parser would then disallow everything.
req = urllib.request.Request("https://en.wikipedia.org/robots.txt", headers={"User-Agent": "llmatlas-notebook/0.1"})
robots_txt = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")

rp = urllib.robotparser.RobotFileParser()
rp.parse(robots_txt.splitlines())

for agent in ["*", "Googlebot", "CCBot", "GPTBot", "Google-Extended"]:
    article = rp.can_fetch(agent, "https://en.wikipedia.org/wiki/Language_model")
    search = rp.can_fetch(agent, "https://en.wikipedia.org/w/index.php?title=Special:Search&search=x")
    print(f"{agent:16} article: {article}  search: {search}")
```

The user agents in that list are the ones that matter for training data: `CCBot` is Common Crawl, `GPTBot` is OpenAI's crawler, `Google-Extended` is Google's token for opting out of model training without leaving search, and `ClaudeBot`, `anthropic-ai`, `Bytespider`, `PerplexityBot` and `Applebot-Extended` are the others most often named in robots files. A site that disallows `CCBot` removes itself from every future Common Crawl snapshot, and therefore from every corpus derived from one.

### 3.3 Crawler architecture

A crawler at corpus scale is a distributed system with four parts:

- **Frontier**: the queue of URLs to fetch, prioritised by some estimate of value (links in, freshness, site quality) and partitioned by host so that per-host politeness can be enforced locally. The frontier is the crawl's main state and its main engineering problem; Common Crawl uses Apache Nutch's.
- **Fetchers**: many workers, each holding a pool of connections, resolving DNS, fetching with timeouts, and respecting the per-host schedule. Throughput is bounded by politeness rather than bandwidth: at one request per two seconds per host, a hundred thousand hosts in flight give fifty thousand pages per second in theory and a tenth of that in practice.
- **Deduplication of URLs** before fetching (canonicalisation, a seen-set of URL hashes) and of content after fetching (so that mirrors and soft-404 pages do not fill the frontier with links).
- **Storage** as WARC, with the request, response and metadata records written together, because the raw response is what later re-extraction needs. Writing only extracted text saves space and throws away the option of re-extracting with a better tool next year.

Rendering is the expensive option. A page that builds its content with JavaScript yields a near-empty HTML response to a plain fetcher; running a headless browser recovers the content at ten to a hundred times the cost per page. Most training-data crawls do not render, which is one reason single-page applications are under-represented in corpora.

```{admonition} Reading robots.txt
:class: note
The notebook section {ref}`nb-acq-robots` parses two robots.txt files with the standard library: Wikipedia's (28 KB, one of the most detailed on the web, which treats search and training crawlers alike) and a major news site's, which allows Googlebot and disallows every AI crawler by name. It counts the Disallow rules per agent in each.
```

### 3.4 The economics

Order-of-magnitude arithmetic for a billion-page crawl: pages average about 100 KB of HTML, so the raw fetch is 100 TB of transfer, which is free as cloud ingress but costs a few thousand dollars in NAT-gateway or bandwidth charges, and far more if the crawl is then moved between regions or clouds. Compressed, the WARC output is about a third of that. Fetching at an aggregate 5,000 pages per second takes two to three days of wall-clock time on a few hundred small machines; politeness, not hardware, sets that rate. Extraction and filtering, by comparison, are a few CPU-milliseconds per page and run in hours. The dominant cost of a serious crawl is not any of these but the engineering of the frontier and the handling of the long tail of malformed responses, redirect loops, tar pits and hosts that respond slowly enough to tie up a fetcher without timing out.

## 4. Curated Sources

The web supplies volume; the sources in this section supply things the web is short of: long documents, reliable facts, technical reasoning, working code and natural conversation. Each is a small fraction of a corpus and each is sampled several times over.

### 4.1 Wikipedia and the Wikimedia projects

Wikipedia is in every corpus. The Wikimedia Foundation publishes complete database dumps of every language edition twice a month (`dumps.wikimedia.org`), licensed CC BY-SA 4.0, as XML with the wikitext markup; converting wikitext to clean text (templates, infoboxes, references, tables) is the main work, and the `wikimedia/wikipedia` dataset on Hugging Face is a widely used pre-converted version. English Wikipedia is about 4.5 billion words, roughly 5 billion tokens or 0.03% of a 15T-token corpus, and models typically see it two to four times. Wiktionary, Wikibooks, Wikisource and Wikidata are published the same way; Dolma includes Wikibooks alongside Wikipedia.

### 4.2 Books

Books are the best source of long-range structure (a plot, an argument, a chapter that refers to an earlier one), and the most legally fraught. The sources that have been used:

- **Project Gutenberg**: more than 75,000 public-domain books, mostly English works published before 1931 (the US public-domain cutoff as of 2026), downloadable in bulk. Legally clean and stylistically dated; it is the books portion of Dolma and part of LLaMA's.
- **BookCorpus**: 11,000 self-published novels scraped from Smashwords for the original BERT and GPT; a retrospective datasheet {cite}`bandy2021addressing` found duplicates, skewed genres and books whose licences did not permit the use.
- **Books3**: about 196,000 pirated books from a shadow library, included in The Pile and used by several models, withdrawn after takedown notices in 2023 and the subject of ongoing lawsuits. Its presence in a training set is now a legal liability.
- **Licensed collections**: publishers and aggregators now sell training licences, and the large labs increasingly buy them, or buy and scan print books, rather than rely on shadow libraries. The 2025 US rulings in *Bartz v. Anthropic* and *Kadrey v. Meta* turned on exactly this distinction: training on lawfully acquired copies was treated as fair use on those records, while keeping pirated libraries was not, and the former case settled for a sum reported at about $1.5 billion.

The practical rule in 2026: use public-domain text, text whose licence explicitly allows the use, or text you have licensed, and keep the receipts. Books3 in The Pile is the cautionary example of what happens otherwise.

### 4.3 Scientific and technical papers

Papers supply formal argument, mathematics and domain vocabulary. The accessible sources:

- **arXiv** publishes bulk access to its LaTeX sources and PDFs (on S3, requester pays). The LaTeX source, converted with macro expansion and with bibliographies and comments removed, is what The Pile and LLaMA used; it is also the main source of mathematical text in the open corpora.
- **S2ORC** {cite}`lo2020s2orc` and its successor **peS2o** {cite}`soldaini2023pes2o` provide full text for tens of millions of open-access papers from Semantic Scholar, already converted from PDF, with licences tracked per paper. Dolma's papers portion is peS2o.
- **PubMed Central** offers the open-access subset of biomedical literature in bulk XML, licence by licence.

PDF extraction is the technical problem: two-column layouts, equations, tables and footnotes come out scrambled from generic tools, which is why pipelines prefer LaTeX or publisher XML wherever it exists. Where only PDFs exist, document-understanding models trained for the job, Nougat {cite}`blecher2023nougat` for academic PDFs and olmOCR {cite}`poznanski2025olmocr` at corpus scale, now recover text, equations and tables well enough that scanned books and papers have become an acquisition channel in their own right.

### 4.4 Source code

Code is a few per cent of every mixture and is kept even in models not aimed at programming; the LLaMA and Gopher papers both report that it helps with structured output and, in ablations reported by several groups since, with reasoning benchmarks. The sources:

- **The Stack** {cite}`kocetkov2022stack` is 3 TB of source files from public GitHub repositories, restricted to permissively licensed repositories (MIT, Apache, BSD and similar) determined from the repository's licence file, near-deduplicated, with an opt-out form that lets developers remove their repositories from future versions. **The Stack v2** {cite}`lozhkov2024starcoder2` was built from Software Heritage's archive instead, which preserves provenance for every file.
- **GitHub archives** such as GH Archive (event streams) and the Software Heritage graph are the raw material behind both; building from them directly means doing the licence detection yourself.
- **Notebooks, issues, commits and documentation** are included alongside code in StarCoder's training data because they pair code with natural-language explanation.

Licence filtering is the acquisition-time decision that matters most for code: a permissive-only corpus is smaller but can be released and used without the copyleft question that GPL-licensed code raises for model outputs.

### 4.5 Forums, dialogue and question answering

Conversation teaches turn-taking, register and the question-answer form that instruction tuning (Chapter 10) later refines. The sources have become harder to obtain:

- **Reddit** was the backbone of WebText (GPT-2 used links with at least three upvotes as a quality signal) and of the Pushshift archive used by many corpora, until API access was closed to free bulk use in 2023 and licensed to model developers instead.
- **Stack Exchange** published a quarterly dump of every question and answer on its sites under CC BY-SA, which is in The Pile and LLaMA, and is the single best source of paired technical questions and answers; in 2024 the dump moved behind a login with terms that restrict its use for training.
- **Mailing lists, IRC logs and email archives**, the Ubuntu IRC logs and the Enron email corpus in The Pile, supply informal dialogue.
- **Subtitles and transcripts** (OpenSubtitles, YouTube captions) supply spoken-register dialogue; their licensing is unsettled and most recent open corpora omit them.

### 4.6 News, government and legal text

News archives are mostly licensed commercially now; the open sources are the CC-NEWS subset of Common Crawl (news sites crawled continuously, used by RoBERTa) and national-library collections with their own terms. Government text, court opinions (the FreeLaw project in The Pile), patents (USPTO), legislation and regulatory filings are public records in most jurisdictions, large, well edited and formal, and under-used relative to their quality because they have to be collected site by site.

```{admonition} Comparing Sources
:class: tip
The notebook section {ref}`nb-acq-sources` pulls a hundred documents each from FineWeb, FineWeb-Edu and C4, and fifty each from Wikipedia and the French portion of FineWeb-2, through the Hugging Face rows API, and compares document lengths, language scores, domain concentration and the FineWeb-Edu educational score, so that the differences described in this section are visible in real data rather than asserted.
```

## 5. Rights, Licences and Consent

This section describes the landscape as of 2026 and is not legal advice. The questions are live in several courts, the answers differ by jurisdiction, and a team assembling a corpus for anything beyond research should have counsel involved from the start.

### 5.1 Copyright and the training question

Almost every document on the web is under copyright from the moment it is written; the open question is whether copying it into a training set, and training on it, is permitted without a licence.

- In the **United States** the argument is fair use: that training is a transformative, non-expressive use that does not substitute for the work. Several suits by authors, publishers and news organisations against model developers were pending or partly decided in 2024 and 2025, with rulings so far turning on how the data was obtained (a pirated library is treated differently from a lawful copy) and on whether the model's outputs reproduce the works.
- The **European Union**'s Digital Single Market directive (2019) created a text-and-data-mining exception for research institutions and a general one that rights holders may opt out of in a machine-readable way; the AI Act (2024) requires providers to publish a summary of training content and to honour those opt-outs. A robots.txt directive is the usual machine-readable form.
- **Japan** permits training on copyrighted works under a 2018 exception with narrow limits; the **United Kingdom** has a research exception and has consulted repeatedly on a broader one; **Singapore** has a computational-analysis exception.

The practical consequences for acquisition: keep the record of where each document came from and under what terms (Section 9); prefer sources that are public domain, openly licensed, or licensed to you; treat shadow libraries and scraped paywalled content as liabilities, whatever the fair-use argument; and build the machinery to honour opt-outs, because a document that was lawful to collect may not be lawful to keep.

### 5.2 Dataset licences are not content licences

A dataset on a hub carries a licence (ODC-BY, CC BY, Apache 2.0, "other"), and it is tempting to read that licence as permission to train on the contents. It is not. The dataset licence covers the compilation and the curation work; the documents inside keep their own rights. FineWeb and Dolma are both released under ODC-BY with an explicit note that Common Crawl's terms of use apply to the underlying pages. The Data Provenance Initiative {cite}`longpre2023data` audited more than 1,800 widely used fine-tuning datasets and found that more than 70% of the licences were omitted on the hosting platforms and more than half of those stated were wrong, frequently in the permissive direction, with the original sources often restricting commercial use where the hub listing did not. The lesson is to trace every dataset to its sources and read the sources' terms, which is tedious and is the job.

### 5.3 robots.txt, opt-outs and the shrinking commons

Consent in Crisis {cite}`longpre2024consent` measured, across C4, RefinedWeb and Dolma, how the web's permissions for AI crawlers changed between 2023 and 2024: within a year, robots.txt restrictions had put off limits more than 5% of all of C4's tokens and more than 28% of the tokens from its most actively maintained, highest-quality domains (news organisations above all), with terms-of-service restrictions covering a further large share (about 45% of C4), and the restrictions were rising fastest exactly where the best text was. The restrictions are also inconsistent: many sites block OpenAI's crawler but not others, many block `CCBot` without realising that this also removes them from the open corpora that research depends on, and many state a prohibition in their terms of service that their robots.txt does not express.

Opt-out mechanisms beyond robots.txt:

- **Registries** such as Spawning's Do Not Train list and the `ai.txt` proposal, which some corpus builders query at acquisition time.
- **Per-dataset opt-out forms**, as The Stack provides for GitHub repositories, honoured in the next release.
- **Retroactive removal**: FineWeb and RefinedWeb state that they will remove pages whose owners ask, and Common Crawl removes content on request from future snapshots.

A corpus built today should record, per document, the robots.txt state at fetch time and should be rebuilt against current opt-outs before each new training run. The cost of doing so is small; the cost of not doing so is a model that cannot be shipped.

### 5.4 What the open corpora decided

| Corpus                                  | Content licence position                                        | Opt-out                                   | Known problems                                          |
| --------------------------------------- | --------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| C4 {cite}`raffel2019exploring`          | Common Crawl terms; first released as a script that regenerates it from Common Crawl, hosted as text from 2021 | None | Removed benign minority text with its word list {cite}`dodge2021documenting` |
| The Pile {cite}`gao2020pile`            | Mixed; included Books3                                          | None                                      | Books3 withdrawn after takedown; litigation            |
| The Stack {cite}`kocetkov2022stack`     | Permissive licences only, detected from licence files           | Form per repository                       | Licence detection errors; copyleft-adjacent files       |
| Dolma {cite}`soldaini2024dolma`         | ODC-BY on the compilation; sources' terms apply                 | Via Common Crawl and source hubs          | Reddit portion predates API closure                     |
| FineWeb {cite}`penedo2024fineweb`       | ODC-BY on the compilation; Common Crawl terms apply             | Email removal requests honoured           | Web only, so inherits every robots.txt gap              |
| RedPajama-V2 {cite}`weber2024redpajama` | Released with quality signals so users apply their own filters  | Via Common Crawl                          | 100T raw tokens, deliberately unfiltered                |

### 5.5 Personal data and sensitive sources

Data-protection law (the GDPR and its relatives) applies to personal data regardless of copyright, which is why acquisition should avoid sources that are mostly personal data (people directories, leaked databases, social profiles) rather than relying on the redaction in Chapter 4 to fix them later. Health records, children's content and content from regions with strict localisation rules are usually excluded at the source level for the same reason.

### 5.6 The infrastructure that is replacing robots.txt

Three developments in 2025 are turning the informal conventions above into machinery, and a corpus built in 2026 should expect to interact with all of them:

- **Default blocking and paid crawling.** Cloudflare, which fronts a large share of the web, began blocking known AI crawlers by default for new sites in July 2025 and introduced a pay-per-crawl mechanism; the IETF's AI Preferences working group is standardising a vocabulary for expressing training and inference permissions in robots.txt and HTTP headers.
- **Regulatory codes.** The EU's General-Purpose AI Code of Practice (2025) commits signatories to honour robots.txt and other machine-readable opt-outs and to publish a training-content summary, which makes the opt-out handling of Section 5.3 a compliance requirement rather than a courtesy.
- **Openly licensed corpora.** The Common Pile {cite}`kandpal2025common` (8 TB of public-domain and openly licensed text, with a model trained on it to show the approach is viable), Common Corpus and institutional collections of digitised public-domain books show what a corpus looks like when Section 5's advice is followed to the letter: smaller, older on average, and free of the exposure that the web corpora carry.

One acquisition-time risk is specific to how corpora are distributed. Many web datasets are published as URL lists rather than text, and Carlini et al. {cite}`carlini2024poisoning` showed that buying expired domains on those lists is enough to insert chosen content into anyone who later re-downloads the dataset; content hashes recorded at collection time (Section 9) are the defence.

## 6. Multilingual and Low-Resource Sources

### 6.1 The English skew

Roughly 45% of Common Crawl's pages are English, which is already a large over-representation relative to the world's speakers, and every filtering stage makes the skew worse because the filters were tuned on English. A model trained on an English-dominant corpus can still speak other languages, but it does so with less data, fewer facts and a tokenizer that spends several tokens per word, which raises its cost and lowers its quality for most of the world's people.

### 6.2 The multilingual corpora

| Corpus                                         | Languages | Source               | Approach                                                                                 |
| ---------------------------------------------- | --------- | -------------------- | ---------------------------------------------------------------------------------------- |
| mC4 {cite}`xue2021mt5`                         | 101       | 71 Common Crawl WET  | cld3 language id, per-language C4 rules; known to contain mislabelled and machine-translated text |
| OSCAR {cite}`abadji2022oscar`                  | 150+      | Common Crawl WET     | fastText language id per line, then document level; releases new versions per snapshot  |
| ROOTS {cite}`laurenccon2022bigscience`         | 46 + 13 programming | Curated datasets + crawl, chosen by language communities | Community-sourced catalogue of sources; per-language cleaning parameters |
| CulturaX {cite}`nguyen2023culturax`            | 167       | mC4 + OSCAR          | Merge, re-identify language, URL blocklist, per-language perplexity thresholds, MinHash |
| MADLAD-400 {cite}`kudugunta2023madlad`         | 419       | Common Crawl         | Manual audit of every language by inspection; drops the labels that were not real languages |
| FineWeb-2 {cite}`penedo2025fineweb2`           | 1,000+    | 96 Common Crawl snapshots | Per-language thresholds derived from reference-corpus statistics; re-adds ("rehydrates") duplicates in proportion to how many copies existed |

The recurring finding, first established by the audit of Kreutzer et al. {cite}`kreutzer2022quality` across mC4, OSCAR and other crawled corpora and repeated in the MADLAD-400 audit, is that automatic language identification produces hundreds of "languages" that on inspection are noise, boilerplate, mislabelled majority-language text or machine translation. Below the top hundred or so languages, a human who reads the language has to look at the data. FineWeb-2's rehydration step makes a second point: a page that many sites copied was, on average, worth copying, so after deduplication FineWeb-2 up-samples the surviving copy in proportion to its cluster size, and for languages with little data this recovers a large share of the usable signal.

### 6.3 Beyond the crawl

For languages the web under-serves, acquisition reaches for sources that the web pipeline skips: religious texts and their translations (the Bible and Quran exist in more languages than any other document), government publications and legal codes in official languages, broadcaster transcripts, Wikipedia editions, OPUS and other parallel corpora assembled for machine translation, and community collections such as those gathered for ROOTS and by Masakhane for African languages. These are small, but a few million tokens of genuine text in a language is worth more than a billion tokens of mislabelled crawl.

## 7. Synthetic Data

### 7.1 Why generate text

If the stock of human-written text is the binding constraint, text generated by a model is the obvious supplement, and it is now a standard component rather than an experiment:

- **Instruction data**: Self-Instruct {cite}`wang2023selfinstruct` showed that a model can generate the instruction-response pairs used to fine-tune it, bootstrapping from a few hundred human-written seeds; this is the basis of most instruction datasets (Chapter 10).
- **Textbook-style pretraining data**: phi-1 {cite}`gunasekar2023textbooks`, a 1.3B-parameter code model, trained on under a billion tokens of synthetic textbook-style text plus about six billion tokens of classifier-filtered code and matched much larger models on the HumanEval and MBPP coding benchmarks. Cosmopedia {cite}`benallal2024cosmopedia` reproduced the idea openly, generating 25 billion tokens of textbooks, blog posts and stories from seed topics with an open model.
- **Rephrased web**: rewriting crawled pages in a cleaner style with a model, then training on the rewritten version, gives a cleaner corpus of the same size.
- **Translation and augmentation** for low-resource languages, where a model translates high-resource text into a target language.

### 7.2 What can go wrong

Synthetic data inherits the generating model's errors, style and gaps, and feeding it back repeatedly compounds them. Shumailov et al. {cite}`shumailov2024curse` showed that models trained on the outputs of earlier models lose the tails of the distribution first, so that after a few generations rare content disappears and the model's outputs converge on a narrow, repetitive mode. The practical mitigations are to keep synthetic data a minority of the mixture, to keep human-written data in every generation, to anchor generation in real documents (rephrasing, grounded question writing) rather than free generation, and to mark synthetic data as such in provenance records. The last matters beyond your own pipeline: model-generated text is now a growing fraction of the open web, so future crawls contain it whether or not it was labelled.

## 8. Mixture Design and Token Budgeting

### 8.1 From compute to tokens

Chapter 2's relation $C \approx 6ND$ fixes the token count once the budget and the model size are chosen. For an 8B-parameter model and $10^{24}$ FLOPs that is about 20T tokens; for the Chinchilla optimum at the same budget, $N \approx \sqrt{C/120} \approx 91$B and $D \approx 1.8$T. Production practice sits between: a model sized for inference cost, trained on as many tokens as the data allows.

### 8.2 Epochs and unique tokens

Each source $s$ has $U_s$ unique tokens after cleaning. If the mixture assigns it a weight $w_s$ (its share of training tokens) and the run sees $D$ tokens in total, the source is repeated $e_s = w_s D / U_s$ times. The data-constrained scaling result {cite}`muennighoff2023scaling` says $e_s$ up to about 4 is nearly free and the value of further repeats decays quickly, so the design constraint is

$$e_s = \frac{w_s D}{U_s} \le e_{\max} \approx 4 \quad \text{for every source } s, \qquad \sum_s w_s = 1$$

Small, high-quality sources (Wikipedia, textbooks, papers) hit the cap first and are the reason mixtures cannot simply weight sources by preference.

### 8.3 Setting the weights

Three ways the weights get set:

- **By hand, from experience.** LLaMA's mixture (67% crawl, 15% C4, 4.5% each of GitHub, Wikipedia and books, 2.5% arXiv, 2% StackExchange, with Wikipedia and books at about two epochs) {cite}`touvron2023llama` is the template most open models started from.
- **By optimisation.** DoReMi {cite}`xie2023doremi` trains a small proxy model with weights adjusted to minimise the worst-case excess loss across domains against a reference, then trains the large model with the resulting weights; on The Pile it improved average downstream accuracy at the same token budget and reached the baseline's downstream accuracy 2.6 times faster.
- **By ablation.** The FineWeb {cite}`penedo2024fineweb` and DataComp-LM {cite}`li2024datacomp` protocol, training small models on candidate mixtures and comparing benchmarks, is the method of last resort and the one most teams actually use for the final decision.

A fourth, increasingly common choice is **curriculum**: training on the full mixture for most of the run and then "annealing" on a small, high-quality subset (FineWeb-Edu, textbooks, instruction-like data) in the final few per cent of steps, which the Llama 3 report and several open models describe as a measurable gain.

### 8.4 A budgeting procedure

The function below turns a token budget, a table of sources with unique-token counts and preferred weights, and an epoch cap into a feasible mixture. When a preferred weight would exceed the cap, the source is clipped at the cap and the surplus is redistributed proportionally to the sources that still have room.

```python
import math


def plan_mixture(total_tokens, sources, max_epochs=4.0):
    """Return {source: (weight, tokens, epochs)} obeying sum(weight) == 1 and epochs <= max_epochs.

    sources: {name: (unique_tokens, preferred_weight)}
    """
    caps = {s: min(1.0, max_epochs * u / total_tokens) for s, (u, _) in sources.items()}
    if sum(caps.values()) < 1.0:
        raise ValueError("not enough unique data for this budget at this epoch cap")
    weights = {s: w for s, (_, w) in sources.items() if w > 0}
    if sum(caps[s] for s in weights) < 1.0:
        raise ValueError("the sources with a positive preferred weight cannot absorb the budget")
    total = sum(weights.values())
    weights = {s: w / total for s, w in weights.items()}
    clipped = set()
    while True:
        over = {s for s in weights if s not in clipped and weights[s] > caps[s] + 1e-12}
        if not over:
            break
        surplus = sum(weights[s] - caps[s] for s in over)
        for s in over:
            weights[s] = caps[s]
        clipped |= over
        free = {s: weights[s] for s in weights if s not in clipped}
        scale = sum(free.values())
        for s in free:
            weights[s] += surplus * free[s] / scale
    return {
        s: (weights.get(s, 0.0), weights.get(s, 0.0) * total_tokens, weights.get(s, 0.0) * total_tokens / sources[s][0])
        for s in sources
    }


sources = {               # unique tokens, preferred weight
    "web":        (12e12, 0.82),
    "code":       (1.0e12, 0.08),
    "papers":     (0.15e12, 0.04),
    "wikipedia":  (0.0045e12, 0.03),
    "books":      (0.03e12, 0.03),
}
plan = plan_mixture(total_tokens=2e12, sources=sources)
assert abs(sum(w for w, _, _ in plan.values()) - 1) < 1e-9
assert all(e <= 4.0 + 1e-9 for _, _, e in plan.values())
assert math.isclose(plan["wikipedia"][2], 4.0)   # the smallest source is pinned at the epoch cap
assert plan["web"][0] > 0.82              # and its surplus went to the sources with room
for name, (w, t, e) in plan.items():
    print(f"{name:10} weight {w:6.3f}  tokens {t/1e12:5.2f}T  epochs {e:4.2f}")
```

The procedure says nothing about whether the preferred weights are good; it only keeps them feasible. Choosing them is the ablation work of Section 8.3, and documenting the choice is Section 9.

```{admonition} Planning a Mixture
:class: note
The notebook section {ref}`nb-acq-mixture` starts from a compute budget, derives the token count with the Chapter 2 relation, runs `plan_mixture` over a realistic table of sources, plots the planned epochs per source against the four-epoch line, and shows what happens to the plan when a quarter of the high-quality web slice is lost to opt-outs, the rate the consent study measured for the best-maintained domains.
```

## 9. Provenance and Documentation

### 9.1 What to record

Every document that enters the corpus should carry, as metadata that travels with it through Chapter 4's pipeline and into Chapter 5's versioned dataset:

- **Source and locator**: the corpus or site, the URL or file path, and for Common Crawl the snapshot, WARC file and byte offset, which together are enough to re-fetch the exact bytes.
- **Fetch time and crawl time**, because the page may have changed, been removed or opted out since.
- **Licence or terms** as understood at acquisition, and the robots.txt decision for the crawler that fetched it.
- **A content hash** of the raw bytes, so that the document can be recognised in later versions and so that exact deduplication has something to work with.
- **Synthetic or not**, with the generating model and prompt if synthetic.

```python
import hashlib
import json
import time


def provenance_record(source, locator, raw_bytes, licence, robots_decision, synthetic=False, **extra):
    """One manifest line per acquired document; stable field order, content-addressed."""
    return {
        "source": source,
        "locator": locator,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "licence": licence,
        "robots": robots_decision,
        "synthetic": synthetic,
        **extra,
    }


doc = b"<html><body><p>Example page.</p></body></html>"
rec = provenance_record("commoncrawl", "crawl-data/CC-MAIN-2026-39/.../00000.warc.gz#offset=0", doc,
                        licence="CC terms of use", robots_decision="allowed:CCBot")
other = provenance_record("commoncrawl", "same locator", doc + b"\n", "CC terms of use", "allowed:CCBot")
assert provenance_record("x", "y", doc, "l", "r")["sha256"] == rec["sha256"]  # same bytes, same id
assert other["sha256"] != rec["sha256"]                                        # one byte changed, different id
print(json.dumps(rec, indent=1)[:200])
```

### 9.2 Datasheets and data cards

Datasheets for Datasets {cite}`gebru2021datasheets` is the standard template: motivation, composition, collection process, preprocessing, uses, distribution and maintenance, each as a list of questions with answers. Dolma's datasheet is a good complete example, and Hugging Face dataset cards are the lightweight version that every hosted dataset should have. The questions that acquisition has to answer, and that are most often left blank, are the collection ones: who collected it, over what period, with what mechanism, under what terms, and what was deliberately excluded.

A documentation practice that goes further is to publish measurements of the corpus itself. What's In My Big Data {cite}`elazar2024whats` ran a fixed battery of analyses (duplication, benchmark contamination, toxic and personal content, domain and language distributions, the most frequent documents) over ten open corpora and found, among other things, that C4 and the Pile contained benchmark test sets and that several corpora were dominated by a handful of domains. Running the same battery over your own corpus before training is cheaper than discovering the answers from the model.

### 9.3 Handing off

The output of acquisition is a set of raw shards plus a manifest: one provenance record per document, the robots and licence decisions, the source table with unique-token counts, and the mixture plan. Chapter 4 consumes the shards and must preserve the records through every filter; Chapter 5 versions the result so that the model, the corpus and the manifest can be matched up later.

```{admonition} Provenance in Practice
:class: tip
The notebook section {ref}`nb-acq-provenance` writes a provenance record for every fetch made earlier in the notebook, checks every committed sample against the checksums recorded when it was collected, and prints the manifest that would accompany the samples into Chapter 4.
```

## Exercises

1. **Read a snapshot.** Using the manifest and range-request code from Section 2, estimate the size of a current Common Crawl snapshot from the number of WARC files and the sizes reported by a `HEAD` request on ten of them. Compare with Common Crawl's published figure.
1. **Who is blocked.** Fetch robots.txt from the fifty most-linked news sites you can list and tabulate which of `CCBot`, `GPTBot`, `ClaudeBot`, `Google-Extended` and `*` each disallows. How many block Common Crawl, and did they likely intend to leave the open corpora?
1. **Source comparison.** Extend the notebook's source comparison to a sixth source of your choosing from the Hugging Face hub. Which of the statistics in the notebook separates it from the others most clearly?
1. **Licence tracing.** Pick three popular datasets on the hub, read their stated licence, then trace each to its original sources and read those terms. Record every discrepancy, in the style of the Data Provenance Initiative audit.
1. **Budget sensitivity.** Using `plan_mixture`, hold the mixture preferences fixed and sweep the total budget from 0.5T to 20T tokens. At what budget does each source hit the epoch cap, and what does the mixture look like at 20T?
1. **Low-resource audit.** Take fifty documents labelled as a language you read from mC4 or OSCAR and classify each by hand as genuine text, boilerplate, mislabelled or machine-translated. Compare your rate with the MADLAD-400 audit for that language.
1. **A datasheet.** Write the collection-process section of a datasheet for the samples the notebook fetched, answering every question in the Datasheets template that applies.

## Further Reading

### Corpora and their papers

- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) {cite}`penedo2024fineweb` - the most carefully ablated open web corpus; Section 2 covers acquisition from Common Crawl
- [FineWeb2](https://arxiv.org/abs/2506.20920) {cite}`penedo2025fineweb2` - the same pipeline for a thousand languages, with the rehydration result
- [Dolma](https://arxiv.org/abs/2402.00159) {cite}`soldaini2024dolma` - a multi-source corpus whose datasheet is a model of documentation
- [The Pile](https://arxiv.org/abs/2101.00027) {cite}`gao2020pile` - the case for source diversity, and the Books3 cautionary tale
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) {cite}`penedo2023refinedweb` - the argument that filtered web alone is enough
- [RedPajama](https://arxiv.org/abs/2411.12372) {cite}`weber2024redpajama` - a 100T-token raw web corpus released with quality signals instead of filters
- [The Stack](https://arxiv.org/abs/2211.15533) {cite}`kocetkov2022stack` and [StarCoder 2 and The Stack v2](https://arxiv.org/abs/2402.19173) {cite}`lozhkov2024starcoder2` - licence-filtered code with an opt-out process
- [S2ORC](https://arxiv.org/abs/1911.02782) {cite}`lo2020s2orc` - open-access papers as full text with licences tracked
- [A Critical Analysis of Common Crawl](https://doi.org/10.1145/3630106.3659033) {cite}`baack2024critical` - what the crawl is and is not

### Rights, consent and provenance

- [Consent in Crisis](https://arxiv.org/abs/2407.14933) {cite}`longpre2024consent` - the year the web started opting out
- [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787) {cite}`longpre2023data` - the licensing audit
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) {cite}`gebru2021datasheets` - the documentation template
- [What's In My Big Data?](https://arxiv.org/abs/2310.20707) {cite}`elazar2024whats` - measuring corpora before training on them
- [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) {cite}`dodge2021documenting` - what C4's acquisition and filtering removed
- [The Common Pile v0.1](https://arxiv.org/abs/2506.05209) {cite}`kandpal2025common` - an 8 TB openly licensed corpus and a model trained on it
- [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149) {cite}`carlini2024poisoning` - why URL-list datasets need content hashes

### Scale, mixtures and synthetic data

- [Will we run out of data?](https://arxiv.org/abs/2211.04325) {cite}`villalobos2024run` - the size of the stock of human text
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) {cite}`muennighoff2023scaling` - how many epochs are safe
- [DoReMi](https://arxiv.org/abs/2305.10429) {cite}`xie2023doremi` - learning the mixture weights
- [A Pretrainer's Guide to Training Data](https://arxiv.org/abs/2305.13169) {cite}`longpre2023pretrainers` - age, domain and quality effects measured
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) {cite}`gunasekar2023textbooks` and [Self-Instruct](https://arxiv.org/abs/2212.10560) {cite}`wang2023selfinstruct` - synthetic pretraining and instruction data
- [AI models collapse when trained on recursively generated data](https://doi.org/10.1038/s41586-024-07566-y) {cite}`shumailov2024curse` - the failure mode of synthetic data

### Multilingual

- [mT5 and mC4](https://arxiv.org/abs/2010.11934) {cite}`xue2021mt5`, [OSCAR](https://arxiv.org/abs/2201.06642) {cite}`abadji2022oscar`, [CulturaX](https://arxiv.org/abs/2309.09400) {cite}`nguyen2023culturax`
- [Quality at a Glance](https://arxiv.org/abs/2103.12028) {cite}`kreutzer2022quality` and [MADLAD-400](https://arxiv.org/abs/2309.04662) {cite}`kudugunta2023madlad` - the audits that found which "languages" in crawled corpora were real
- [The BigScience ROOTS Corpus](https://arxiv.org/abs/2303.03915) {cite}`laurenccon2022bigscience` - community-sourced multilingual acquisition

### Tools and services

- [Common Crawl](https://commoncrawl.org/get-started) - snapshots, the CDX index API and the columnar index
- [warcio](https://github.com/webrecorder/warcio) - streaming WARC reader and writer
- [Hugging Face dataset viewer API](https://huggingface.co/docs/dataset-viewer) - the rows endpoint the notebook uses to sample hosted corpora
- [datatrove](https://github.com/huggingface/datatrove) and the [Dolma toolkit](https://github.com/allenai/dolma) - pipeline libraries with Common Crawl readers
- [Wikimedia dumps](https://dumps.wikimedia.org/) - Wikipedia and sister projects
- [Spawning](https://spawning.ai/) - the Do Not Train registry

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
