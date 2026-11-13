# Data Acquisition and Sources

Every token a language model is trained on was written by someone, stored somewhere, and copied into the training set by a decision that a person or a pipeline made. This chapter is about those decisions: where the text that trains a modern model comes from, how it is obtained at the scale of trillions of tokens, what may legally and ethically be used, and how the pieces are combined into a corpus. {ref}`Chapter 4 <sec-ch4-chapter-overview>` picks up where this one stops, with the cleaning that turns raw acquisitions into training-ready text, and {ref}`Chapter 5 <sec-ch5-chapter-overview>` with keeping the result versioned and reproducible.

(sec-ch3-chapter-overview)=

## Chapter Overview

Imagine being asked to assemble a library for someone who will read every book in it exactly once and then has to answer any question anyone asks. The questions are not known in advance. The budget is enough for a few trillion words. The books have to be obtained lawfully, and whoever wrote them may have opinions about the arrangement.

**The web** is the only source large enough to fill most of the shelves. Common Crawl, a non-profit that has crawled the web since 2008 and publishes the result for free, supplies the raw material for nearly every open corpus and, by all accounts, for most closed ones.

**Curated sources** fill the gaps the web leaves: encyclopedias for facts, books for long-form prose, scientific papers for technical reasoning, source code for programming, and conversation for dialogue. In most mixtures each is a few percent of the corpus (the exceptions are The Pile, where papers, court opinions and patents are 37% of the training bytes (papers alone 27%), and Dolma, where code is 13% of the tokens), and the smallest (Wikipedia, books) are typically seen two or more times.

**Crawling** is how new material is obtained when the published crawls are not enough. It is a matter of HTTP mechanics, politeness conventions that are rapidly hardening into legal lines, and arithmetic about how many machines it takes to fetch a billion pages.

**Rights and consent** decide what may be kept. Copyright, dataset licenses, robots.txt and opt-out registries all apply, and the answer has changed quickly: robots.txt put more than a quarter of the tokens from the best-maintained web domains off limits to AI crawlers in a single year.

**Mixture design** turns the pile of sources into a training set. The scaling laws of {ref}`Chapter 2 <sec-ch2-chapter-overview>` say how many tokens a model should see; the mixture says which ones, how many times, and in what proportion.

**Provenance** is the record that lets anyone, including you in a year's time, find out what is in the corpus and where it came from.

Acquisition is where a model's knowledge, its languages, its biases and its legal exposure are all fixed. The cleaning in {ref}`Chapter 4 <sec-ch4-chapter-overview>` can remove what should not have been collected; it cannot add what was never acquired.

{numref}`fig-ch3-acquisition-flow` shows how the parts connect: every source passes the rights checks and is written to the provenance manifest, from which Chapter 4 cleans the text and Chapter 5 samples it by the mixture plan.

:::{figure} figures/acquisition-flow.*
:name: fig-ch3-acquisition-flow
:alt: Flow chart from top to bottom. Four sources (Common Crawl snapshots, your own crawl, curated sources, synthetic text) feed a rights check, which feeds the provenance manifest. The manifest feeds the mixture plan and Chapter 4's cleaning, and both feed Chapter 5.

From sources to training data. Rights are decided at acquisition
({ref}`sec-ch3-rights-licenses-consent`), each kept document gets a manifest record
({ref}`sec-ch3-provenance-documentation`), and the mixture plan
({ref}`sec-ch3-mixture-design-token-budgeting`) fixes each source's weight $w_s$ (its
share of training tokens) and the number of times $e_s$ it is repeated, at most
$e_{\max}$ epochs, before Chapter 5 samples the cleaned shards.
:::

Every symbol the book uses is listed on the {ref}`notation page <sec-notation>`, with a link to the section that introduces each symbol used only in this chapter.

(sec-ch3-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- What the pretraining corpora of GPT-3, the LLaMA family, The Pile, Dolma and FineWeb are made of, and roughly in what proportions
- How Common Crawl is organized (snapshots, segments, WARC/WET/WAT files, the index) and how to read a slice of it
- How to crawl responsibly: robots.txt, rate limits, sitemaps, and the economics of fetching a billion pages
- How a crawler's frontier, scheduler and fetchers fit together, with URL canonicalization and conditional re-fetching in code
- The main curated sources (Wikipedia, books, scientific papers, source code, forums and dialogue, licensed news, and first-party conversations) and what each contributes
- The copyright, licensing, consent and personal-data questions around training data, and how the major open corpora answer them
- Why multilingual corpora need human audits, and what sources exist beyond the crawl for low-resource languages
- What synthetic data adds to a corpus and how it fails when it replaces human text
- How to turn a token budget into a mixture of sources under an epoch cap, and how to document what was collected

(sec-ch3-hands-on-learning)=

## Hands-On Learning

The accompanying notebook, [Data Acquisition Demo](examples/data_acquisition_demo.ipynb), fetches small samples from real public sources and checks every claim with an assertion:

- Reading a Common Crawl snapshot's manifest and parsing records from a byte-range slice of a WARC file ({ref}`nb-acq-crawl`)
- Checking robots.txt rules for search and AI crawler user agents on an encyclopedia and a news site ({ref}`nb-acq-robots`)
- Sampling FineWeb, FineWeb-Edu, C4, Wikipedia and FineWeb-2 through the Hugging Face rows API and comparing document lengths, language scores and domain concentration ({ref}`nb-acq-sources`)
- Turning a compute budget into a token budget and a source mixture with per-source epoch caps ({ref}`nb-acq-mixture`)
- Writing a provenance manifest with content hashes for everything fetched ({ref}`nb-acq-provenance`)

Everything runs on a CPU in about a minute. The network fetches total a few megabytes; if a source is unreachable, or if `LLMATLAS_OFFLINE=1` is set, the notebook falls back to committed samples under `examples/data/` so that the results, and the assertions, are reproducible.

(sec-ch3-pretraining-corpus-composition)=

## 1. What a Pretraining Corpus Is Made Of

(sec-ch3-published-mixtures)=

### 1.1 The published mixtures

The training sets of the major models are reported, when they are reported at all, as a table of sources with a share of the training tokens. The table below collects the published mixtures. The shares are not all of the same quantity. For GPT-3 and LLaMA they are shares of training tokens, which differ from shares of the raw data because sources are sampled at different rates. For The Pile they are shares of the training bytes after upsampling. For Dolma and FineWeb they are shares of the corpus by tokens.

```{raw} latex
\begingroup\footnotesize\setlength{\tabcolsep}{2pt}
```

```{tabularcolumns} \Y{0.13}\Y{0.16}\Y{0.21}\Y{0.16}\Y{0.12}\Y{0.09}\Y{0.13}
```

| Model or corpus                                  | Web crawl                    | Curated web (Wikipedia, forums, news) | Books          | Papers           | Code            | Total tokens |
| ------------------------------------------------ | ---------------------------- | ------------------------------------- | -------------- | ---------------- | --------------- | ------------ |
| GPT-3 {cite}`brown2020language`                  | 60% (filtered Common Crawl)  | 22% WebText2, 3% Wikipedia            | 16% (two book corpora) | included in web | not separate | 300B (trained on) |
| The Pile {cite}`gao2020pile`                     | 18% Pile-CC                  | 17% (OpenWebText2, StackExchange, Wikipedia, HN) | 15% (Books3, Gutenberg, BookCorpus2) | 37% (27% papers: PubMed Central, arXiv, PubMed abstracts, …; 10% court opinions and patents: FreeLaw, USPTO) | 8% GitHub | 825 GiB raw, 1,254 GiB after upsampling (bytes; no token count reported) |
| LLaMA 1 {cite}`touvron2023llama`                 | 67% CommonCrawl + 15% C4     | 4.5% Wikipedia, 2% StackExchange      | 4.5% (Gutenberg, Books3) | 2.5% arXiv | 4.5% GitHub | 1.4T (trained on) |
| Dolma {cite}`soldaini2024dolma`                  | 81% (Common Crawl)           | 3% (Reddit, Wikipedia, Wikibooks)     | 0.2% (Gutenberg) | 2.3% (peS2o)     | 13.4% (The Stack) | 3T (corpus) |
| FineWeb {cite}`penedo2024fineweb`                | 100%                         |                                       |                |                  |                 | 15T (corpus) |

```{raw} latex
\endgroup
```

The Pile's shares are the paper's "Weight" column: each component's raw size times its epochs, out of 1,254 GiB, not of the 825 GiB raw corpus. Pile-CC's 227 GiB is therefore 18% of the training bytes but 28% of the raw corpus. Dolma's shares are from its Table 1, in Llama tokens out of 3,059B.

{numref}`fig-ch3-mixtures` draws the same shares as bars, which makes the contrast visible at a glance: The Pile is the only mixture in which the web crawl is not the largest part.

:::{figure} figures/mixtures.*
:name: fig-ch3-mixtures
:alt: Five horizontal stacked bars, one per mixture, split into web crawl, curated web, books, papers, code and other, the last being 5 percent of The Pile. GPT-3 is 60, 25 and 16 percent; The Pile 18, 17, 15, 37 and 8; LLaMA 1 82, 6.5, 4.5, 2.5 and 4.5; Dolma 81, 3, 0.2, 2.3 and 13.4; FineWeb 100 percent web crawl.

The published mixtures of the table above. Numbers are percentages as published and
rounded, so GPT-3's row sums to 101%. "Other" is what the five columns leave of 100%:
The Pile's 5% are the components the table does not place in a column (OpenSubtitles, DM
Mathematics, Ubuntu IRC, EuroParl, YouTube subtitles and Enron emails); Dolma places
every source, so its 0.1% is rounding. Segments under 3% are too narrow to label inside
the bar; their values are written to the right of the bar. GPT-3 and LLaMA 1 are shares
of training tokens, The Pile of training bytes after upsampling, Dolma and FineWeb of
corpus tokens.
:::

Three things stand out. The web is the bulk of every corpus but The Pile, because nothing else is large enough; The Pile, built to be diverse, takes most of its bytes from papers and curated sources instead. Curated sources are small in tokens but upweighted: Wikipedia is 3% of GPT-3's training tokens while being a tiny fraction of the raw data, and LLaMA samples it for more than two epochs. And the sizes have grown fifty-fold in four years, from the 300B tokens GPT-3 trained on to FineWeb's corpus of 15T tokens (a corpus size, not a training count) and the roughly 15T tokens Llama 3 trained on {cite}`grattafiori2024llama`, which is why the question of whether the supply of human-written text is finite {cite}`villalobos2024run` is now taken seriously.

(sec-ch3-tokens-documents-bytes)=

### 1.2 Tokens, documents and bytes

Corpus sizes are quoted in three units and the conversions are rough:

- **Bytes** are what storage and transfer cost. A Common Crawl snapshot is 80 to 100 TB compressed as WARC and about 7% of that as extracted text (WET).
- **Documents** are the unit of acquisition and deduplication. A snapshot holds two to three billion pages.
- **Tokens** are what training cost and the scaling laws count. English prose runs about four characters per token with a GPT-style tokenizer, so one terabyte of UTF-8 text is roughly 250 billion tokens; code and non-Latin scripts tokenize less efficiently.

The retention figures in {ref}`Chapter 4 <sec-ch4-chapter-overview>` apply on top of this: of a snapshot's two to three billion pages, a cleaned English corpus keeps perhaps a tenth, some 250 million pages. FineWeb's 15T tokens came from 96 snapshots, about 156 billion tokens per cleaned snapshot (15T / 96), or about 600 tokens per kept page. Fifteen trillion tokens is therefore not one snapshot's worth but ninety-six snapshots' worth, each deduplicated on its own for the reason given in {ref}`sec-ch3-corpora-derived-common-crawl`.

(sec-ch3-epochs)=

### 1.3 How much is enough

Chapter 2 derived the compute-optimal token count ({ref}`sec-ch2-flops`): for a budget of $C$ FLOPs and a model of $N$ parameters, $C \approx 6ND$ and the Chinchilla optimum is about $D \approx 20N$ tokens {cite}`hoffmann2022training`. Production models train well past that point, because inference cost scales with $N$ and not with $D$, so a smaller model trained longer is cheaper to serve. Llama 3's 8B model saw more than 15T tokens {cite}`meta2024llama3card`, almost two thousand per parameter.

Two results bound how far a fixed corpus can be stretched:

- **Repetition is cheap up to a point.** Training for up to about four epochs on the same data costs almost nothing in final loss relative to fresh data; beyond that the returns fall off steeply {cite}`muennighoff2023scaling`. A corpus of 1T unique tokens can support a 4T-token training run, not a 40T one. The paper models the decline as decay: each repetition is worth less than a fresh pass, so a source of $U$ unique tokens seen for $e$ epochs is worth $U\,[1 + 15.4\,(1 - \exp(-(e - 1)/15.4))]$ fresh tokens, where 15.4 is the paper's fitted decay constant (15.387756 in its fit). The listing evaluates it. The hard cap $e_{\max}$ of {ref}`sec-ch3-epoch-cap-per-source` is a step-function approximation of this curve.
- **The stock of public text is finite.** {cite:t}`villalobos2024run` build the estimate in three steps: about 510T tokens of deduplicated text in the indexed web, about 100T once text of too low a quality to train on is set aside, and that figure multiplied by the number of epochs it can be repeated for, the step that uses the result above. They project that training runs will use the whole supply "at some point between 2026 and 2032". The number usually quoted is the end product, the *effective* stock, which Epoch AI's summary of the paper gives as "on the order of 300 trillion tokens, with a 90% confidence interval of 100T to 1000T", an estimate that "includes only data that is sufficiently high-quality to be used for training, and accounts for the possibility of training models for multiple epochs" {cite}`epoch2024runout`.

```python
import math

R_STAR = 15.387756  # fitted decay constant of the data-constrained scaling law


def effective(epochs):
    """Fresh-token equivalents of one unique token seen for `epochs` epochs."""
    return 1 + R_STAR * (1 - math.exp(-(epochs - 1) / R_STAR))


for e in (1, 4, 10, 40):
    print(f"{e:>2} epochs: {effective(e):5.2f} fresh-token equivalents,",
          f"{effective(e) / e:.0%} of fresh data per pass")
assert 3.7 < effective(4) < 3.8 and 15.1 < effective(40) < 15.3
```

% output

```text
 1 epochs:  1.00 fresh-token equivalents, 100% of fresh data per pass
 4 epochs:  3.73 fresh-token equivalents, 93% of fresh data per pass
10 epochs:  7.81 fresh-token equivalents, 78% of fresh data per pass
40 epochs: 15.17 fresh-token equivalents, 38% of fresh data per pass
```

Four epochs lose 7% of the value of a fresh pass, which is why the cap is "nearly free"; forty epochs lose 62%, so a run that repeats a corpus forty times gets about 15 passes of value from it, and the extra 25 are close to wasted.

{numref}`fig-ch3-decay-law` plots the quantity behind the listing's four numbers: the value of the $e$-th pass, which is the derivative of the listing's `effective` with respect to $e$, $\exp(-(e - 1)/15.4)$. The step at four epochs keeps every pass up to the fourth at full value and discards the rest; the curve falls smoothly, to 82% at the fourth pass and 8% at the fortieth.

:::{figure} figures/decay-law.*
:name: fig-ch3-decay-law
:alt: Line plot of the value of the e-th pass over the same tokens, as a fraction of a fresh pass, against epoch e from 1 to 40. A solid curve falls from 1.0 at e = 1 to 82% at e = 4, 56% at e = 10 and 8% at e = 40. A dashed line holds 1.0 up to e = 4 and drops to 0 there, the hard cap.

The decay law's value per pass against the hard cap $e_{\max} = 4$ of {ref}`sec-ch3-epoch-cap-per-source`. The solid curve is $\exp(-(e - 1)/15.4)$, the marginal value of the $e$-th epoch, with the fitted constant of the listing above; the dashed step is the cap, which treats a pass as worth everything or nothing.
:::

The effective stock is therefore a count of training tokens, not of distinct tokens, and the four-epoch result is already inside it: it should not be applied a second time. {numref}`fig-ch3-corpus-size` sets the training sets of the table in {ref}`sec-ch3-published-mixtures` against it in the same unit: on a log axis they climb about one order of magnitude every two years, and the largest, at 15T training tokens, is already within a factor of seven of the lower end of the interval.

:::{figure} figures/corpus-size.*
:name: fig-ch3-corpus-size
:alt: Log-scale plot of tokens against year. GPT-3 300 billion in 2020, LLaMA 1 1.4 trillion in 2023, Dolma 3 trillion, FineWeb 15 trillion and Llama 3 15 trillion in 2024. A shaded band from 100 to 1,000 trillion with a dashed line at 300 trillion marks the effective stock of public text, adjusted for quality and repetition; a gray span marks the projected exhaustion years 2026 to 2032.

Training-set and corpus sizes against the effective stock of public text. Circles are
tokens a model was trained on, squares tokens in a released corpus (The Pile is left out
because it reports bytes); each point sits at the month of its paper's arXiv identifier,
except that FineWeb (June 2024) and Llama 3 (July 2024), both at 15T, are drawn 0.1 year
to either side of their months so that both markers show. The band is the 90% interval
of 100T to 1,000T tokens and the dashed line the central 300T of the effective stock,
quality- and repetition-adjusted, as Epoch AI's summary {cite}`epoch2024runout` states
them; the gray span is the years in which {cite:t}`villalobos2024run` project it to be
used up.
:::

Both results bear directly on acquisition. If repeated tokens are nearly free, then a small, high-quality source is worth collecting even though it will be seen many times. If the stock is finite, then every source lost to opt-outs or licensing is a permanent reduction, and synthetic data ({ref}`sec-ch3-synthetic-data`) stops being optional.

Age matters as well as amount. {cite:t}`longpre2023pretrainers` pretrained models on data collected at different times and found that "a temporal shift between evaluation data and pretraining data leads to performance degradation, which is not overcome by finetuning". The newest documents in a corpus set the model's knowledge cutoff, so recording each document's crawl date ({ref}`sec-ch3-record-contents`) is what lets a team state the cutoff and choose snapshots to move it.

(sec-ch3-open-web-common-crawl)=

## 2. The Open Web: Common Crawl

(sec-ch3-common-crawl-overview)=

### 2.1 What Common Crawl is

Common Crawl is a non-profit foundation that has crawled the web since 2008, on a regular (roughly monthly to bi-monthly) schedule since 2013, and publishes every snapshot, free of charge, on Amazon S3 under the Open Data program. Each snapshot is a sample of the web, not a copy: a few billion pages selected by a crawl frontier that balances breadth across domains against depth within them, fetched over two to three weeks. Its crawler, CCBot, is "a Nutch-based web crawler" whose candidate list "is sorted by host (domain name) and then distributed to a set of crawler servers" {cite}`commoncrawl2026faq`. Common Crawl is the raw material for C4, RefinedWeb, FineWeb, Dolma's web portion, OSCAR, mC4 and CulturaX, and for the web portion of nearly every model whose training data has been described. A critical analysis of it as a training source {cite}`baack2024critical` is worth reading before building on it: the crawl over-represents English, under-represents the parts of the web behind login walls or heavy JavaScript, and, being a sample, is not the web.

(sec-ch3-snapshots-segments-files)=

### 2.2 Snapshots, segments and files

A snapshot is identified by a crawl id such as `CC-MAIN-2026-39` (the id carries a year and a week number). Under `crawl-data/<crawl-id>/` it is divided into a hundred segments, and each segment into files of three kinds:

```{tabularcolumns} \Y{0.13}\Y{0.40}\Y{0.15}\Y{0.32}
```

| File type | Contents                                                                     | Size per snapshot | Use                                           |
| --------- | ---------------------------------------------------------------------------- | ----------------- | --------------------------------------------- |
| **WARC**  | Raw HTTP requests and responses, including headers and the full HTML payload | ~90 TB            | Re-extracting text with your own extractor    |
| **WET**   | Plain text that Common Crawl's own extractor pulled from each page           | ~7 TB             | Quick corpora (C4, OSCAR, mC4 used WET)       |
| **WAT**   | JSON metadata per page: links, headers, title, language guess                | ~16 TB            | Link graphs, URL filtering, domain statistics |

A WARC file is a sequence of gzip members, one per record, so that any record can be decompressed on its own once its byte offset is known. Each record has a header block (`WARC-Type`, `WARC-Target-URI`, `WARC-Date`, a content digest, the detected payload type) followed by the payload; a fetched page produces three records, `request`, `response` and `metadata`. The manifest files `warc.paths.gz`, `wet.paths.gz` and `wat.paths.gz` list every file in the snapshot, which is how a job divides the snapshot across machines.

{numref}`fig-ch3-snapshot-layout` follows one path down this hierarchy to a single record, using the index entry that {ref}`sec-ch3-index-targeted-retrieval` looks up.

:::{figure} figures/snapshot-layout.*
:name: fig-ch3-snapshot-layout
:alt: Tree from crawl-data/CC-MAIN-2026-39 to 100 segments, each with warc, wet and wat directories of about 90, 7 and 16 TB per snapshot. One WARC file is drawn as a row of gzip members (request, response, metadata), with the response member at offset 48,374,322 and 9,882 bytes long highlighted. A URL index line with that filename, offset and length leads to a range request that fetches exactly that member.

The layout of a snapshot. The manifest lists 100,000 WARC files, 1,000 per segment, each
about a gigabyte and made of one gzip member per record. An index line gives a record's
file, offset and length; the range request `bytes=48374322-48384203` fetches that member
alone, and it decompresses without the rest of the file.
:::

The listings that touch the network send a `User-Agent` that names the tool (`llmatlas-notebook/0.1`) and that the environment variable `LLMATLAS_USER_AGENT` replaces. It names no person and no repository, which suits a reader's few requests and does not suit a crawler: the rule of {ref}`sec-ch3-robots-txt-politeness-conventions`, a string that names the crawler and links to a page explaining it, applies to anything you run beyond a handful of requests, and the variable is where to set that string. The listings also wait two seconds between requests to one host.

```python
# network
import gzip
import json
import os
import time
import urllib.request
import zlib

BASE = "https://data.commoncrawl.org/"
# LLMATLAS_USER_AGENT, when set, replaces the default agent string.
HEADERS = {"User-Agent": os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")}


def fetch(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers={**HEADERS, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout).read()


def warc_paths(crawl_id):
    """The manifest: one line per WARC file in the snapshot."""
    paths = fetch(f"{BASE}crawl-data/{crawl_id}/warc.paths.gz")
    return gzip.decompress(paths).decode().splitlines()


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
        line.split(": ", 1)
        for line in head.decode("utf-8", "replace").splitlines()[1:]
        if ": " in line
    )
    return headers, payload


paths = warc_paths("CC-MAIN-2026-39")
time.sleep(2)  # two seconds between requests to one host
first_256k = fetch(BASE + paths[0], {"Range": "bytes=0-262143"})
records = [parse_record(r) for r in warc_records(first_256k)]
print(len(paths), "WARC files;", len(records), "complete records in the first 256 KB")
print(sorted({h.get("WARC-Type") for h, _ in records}))
```

% output-shape

```text
100000 WARC files; 62 complete records in the first 256 KB
['metadata', 'request', 'response', 'warcinfo']
```

The `Range` request is the key technique: it fetches a few hundred kilobytes out of a gigabyte-sized file, which makes it possible to look at a snapshot without downloading it. Production pipelines stream whole files, and the same `warc_records` logic, written in a streaming form, is the inner loop of every WARC reader.

Large responses are not stored whole. Common Crawl's crawler stops reading a response at a fixed size, which protects it from endless streams and very large files: the limit was 1 MiB for the crawls before March 2025 and has been 5 MiB from `CC-MAIN-2025-13` onwards {cite}`commoncrawl2025truncated`. The WARC format marks such a record with a `WARC-Truncated` field whose value gives the reason (`length` when the size limit was reached) {cite}`iipc2017warc`. Most HTML pages are far below either limit, but long PDFs and large data files arrive cut off, so a pipeline that extracts from WARC should check the field and either drop truncated records or fetch the document again.

(sec-ch3-index-targeted-retrieval)=

### 2.3 The index and targeted retrieval

Common Crawl also publishes a URL index for each snapshot, queryable through the CDX API. A query returns the WARC filename, byte offset and length of every capture of a URL or URL prefix, and combined with a range request it retrieves a single page from a 90 TB snapshot in one round trip. It is how researchers study a particular site's history across snapshots, and how a corpus builder can pull every capture of a chosen list of domains without scanning the whole snapshot. The index server is shared and rate limited, so bulk use goes through the columnar index files (`cc-index-table`, in Parquet on S3) and a query engine rather than the HTTP API.

A worked lookup in two steps. An index entry gives, for one capture, the WARC file, the byte offset and the compressed length of its record. The three entries below are real lines of the CC-MAIN-2026-39 index, taken from the start of its first index file (the files under `cc-index/collections/<crawl-id>/indexes/` are themselves blocks of gzip members, located through `cluster.idx`, so they too can be read by range request). Each entry is over 400 characters on one line; the listing adds line breaks for display only, writing it as adjacent string literals split after each JSON field, which Python joins back into the exact line. Turning an entry into the request that retrieves the record is arithmetic:

```python
import json

# Three lines of the CC-MAIN-2026-39 URL index (cdx-00000.gz), copied verbatim
# and split after each field for display (adjacent strings are joined into one).
# Each is a SURT key (the host reversed, so that one site's pages sort
# together), a 14-digit capture time and a JSON record.
INDEX_LINES = [
    "0,11,214,3)/galleries/garys-gallery/yazyk 20260908100030 "
    '{"url": "https://3.214.11.0/galleries/garys-gallery/yazyk/", '
    '"mime": "text/html", '
    '"mime-detected": "text/html", '
    '"status": "200", '
    '"digest": "4FPPZ452ITDX4GHJQJ354OVFUQJCBZVS", '
    '"length": "9882", '
    '"offset": "48374322", '
    '"filename": "crawl-data/CC-MAIN-2026-39/segments/1788492699801.50/warc/'
    'CC-MAIN-20260908083825-20260908113825-00403.warc.gz", '
    '"charset": "UTF-8", '
    '"languages": "eng", '
    '"recordid": "01a08076-484a-783b-8b87-4cc051e242c1"}',
    "0,131,250,3)/ 20260917030102 "
    '{"url": "http://3.250.131.0/", '
    '"mime": "text/html", '
    '"mime-detected": "text/html", '
    '"status": "200", '
    '"digest": "Q2UDGWTKF2JOTDPWVJ7K45VJFAEWOL5C", '
    '"length": "8698", '
    '"offset": "272424", '
    '"filename": "crawl-data/CC-MAIN-2026-39/segments/1788493000724.71/warc/'
    'CC-MAIN-20260917025921-20260917055921-00427.warc.gz", '
    '"charset": "UTF-8", '
    '"languages": "por,eng", '
    '"recordid": "01a0ad4f-7be6-719d-a7fd-693aa3f9997c"}',
    "0,135,199,18)/stocks/johnson-johnson 20260910132650 "
    '{"url": "http://18.199.135.0/stocks/johnson-johnson", '
    '"mime": "text/html", '
    '"mime-detected": "text/html", '
    '"status": "200", '
    '"digest": "DKTNPR2H7QQVQGQ2XB2MKEBDXBJKLXDL", '
    '"length": "13816", '
    '"offset": "111752", '
    '"filename": "crawl-data/CC-MAIN-2026-39/segments/1788492699937.26/warc/'
    'CC-MAIN-20260910122853-20260910152853-00300.warc.gz", '
    '"charset": "UTF-8", '
    '"languages": "eng", '
    '"recordid": "01a08b7f-e621-764c-9a9a-02e3eabace87"}',
]

for line in INDEX_LINES:
    surt, timestamp, fields = line.split(" ", 2)
    hit = json.loads(fields)
    start, length = int(hit["offset"]), int(hit["length"])
    print(hit["url"], timestamp[:8], hit["status"])
    name = hit["filename"].rsplit("/", 1)[1]
    print(f"  GET {name}")
    print(f"      Range: bytes={start}-{start + length - 1}")
```

% output

```text
https://3.214.11.0/galleries/garys-gallery/yazyk/ 20260908 200
  GET CC-MAIN-20260908083825-20260908113825-00403.warc.gz
      Range: bytes=48374322-48384203
http://3.250.131.0/ 20260917 200
  GET CC-MAIN-20260917025921-20260917055921-00427.warc.gz
      Range: bytes=272424-281121
http://18.199.135.0/stocks/johnson-johnson 20260910 200
  GET CC-MAIN-20260910122853-20260910152853-00300.warc.gz
      Range: bytes=111752-125567
```

The same entry can be asked for by URL through the CDX API instead of being read out of an index file. The book does not run that query, because the index server is shared and its replies change with every crawl; the request and response below show the documented format only. Common Crawl's index server is, in the words of its repository, "a deployment of the pywb web archive replay and index server", and refers readers to the pywb "CDX Server API" page for the details. That page says that "the only required parameter to the cdx server api is the url", that `output=json` "will return each line as a proper JSON dictionary", and that the standard fields are "urlkey, timestamp, url, mime, status, digest, length, offset, filename". It also warns that without a `limit` "all the matching lines are returned, which may be slow". A request names the crawl in the path and the URL in the query:

```text
GET https://index.commoncrawl.org/CC-MAIN-2026-39-index
    ?url=https://3.214.11.0/galleries/garys-gallery/yazyk/&output=json&limit=1
```

The reply is one JSON object per line, one line per capture. For the entry above it carries the fields of the index line, with the SURT key and capture time as `urlkey` and `timestamp`; the sketch below is built from that index line and is not a server reply:

```text
{"urlkey": "0,11,214,3)/galleries/garys-gallery/yazyk",
 "timestamp": "20260908100030",
 "url": "https://3.214.11.0/galleries/garys-gallery/yazyk/",
 "mime": "text/html", "status": "200",
 "digest": "4FPPZ452ITDX4GHJQJ354OVFUQJCBZVS",
 "length": "9882", "offset": "48374322",
 "filename": "crawl-data/CC-MAIN-2026-39/segments/1788492699801.50/warc/
CC-MAIN-20260908083825-20260908113825-00403.warc.gz"}
```

(In the real reply each object is on a single line, and the filename is not broken.) The last three fields are all the range fetch needs, and they are the values the next listing fetches with.

The second step fetches exactly those bytes. Because each record is its own gzip member, the slice decompresses on its own:

```python
# network
import gzip
import os
import urllib.request

UA = os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")
# FILENAME, START and LENGTH are the filename, offset and length of the first index
# line above (the record is 9882 bytes at offset 48374322 of this file).
FILENAME = ("crawl-data/CC-MAIN-2026-39/segments/1788492699801.50/warc/"
            "CC-MAIN-20260908083825-20260908113825-00403.warc.gz")
START, LENGTH = 48374322, 9882

req = urllib.request.Request("https://data.commoncrawl.org/" + FILENAME,
                             headers={"User-Agent": UA,
                                      "Range": f"bytes={START}-{START + LENGTH - 1}"})
# one gzip member, one record
record = gzip.decompress(urllib.request.urlopen(req, timeout=60).read())
warc_head, http_head, body = record.split(b"\r\n\r\n", 2)
html = body.removesuffix(b"\r\n\r\n")  # the record ends with its own blank line
target = [l for l in warc_head.split(b"\r\n") if l.startswith(b"WARC-Target-URI")][0]
print(warc_head.split(b"\r\n")[0].decode(), "|", target.decode())
print(http_head.split(b"\r\n")[0].decode())
print(len(html), "bytes of HTML; truncated:", b"WARC-Truncated" in warc_head)
```

% output-shape

```text
WARC/1.0 | WARC-Target-URI: https://3.214.11.0/galleries/garys-gallery/yazyk/
HTTP/1.1 200 OK
34353 bytes of HTML; truncated: False
```

{numref}`fig-ch3-warc-record` shows what the three parts of that split are: the WARC header that Common Crawl wrote, the HTTP header the server sent, and the page itself.

:::{figure} figures/warc-record.*
:name: fig-ch3-warc-record
:alt: A gzip member of 9,882 bytes decompresses to one WARC record, drawn as a stack of five bands. The WARC header (WARC/1.0, WARC-Type response, the target URI, WARC-Date 2026-09-08T10:00:30Z, the payload digest, Content-Type application/http), a blank line, the HTTP response header starting HTTP/1.1 200 OK, a second blank line, and the HTML payload of 34,353 bytes.

Anatomy of the response record fetched above. The first two blank lines (`\r\n\r\n`)
divide the record into `warc_head`, `http_head` and `html`, which is the split the code
makes. The WARC date and the payload digest are the capture time and digest in the index
line; a record cut off at the size limit would also carry `WARC-Truncated: length`. A record ends with a further blank line, which belongs to the WARC format and not to the page, so the listing removes it before counting the HTML.
:::

Through the CDX API the same lookup is one request per URL, so a list of a hundred thousand URLs means a hundred thousand requests to the shared index server; at that scale the columnar index is the better route.

The columnar index stores the same fields as a table in Parquet files, one set of files per crawl, and a reader can fetch only the parts of a file that a query needs. Each file ends with a footer that records, for every row group (a block of about 1.6 million rows), the smallest and largest `url_surtkey` in it, so a lookup by URL prefix reads the footer, skips every row group whose range cannot contain the key, and reads only five columns of the one that can. The listing does that for one host in one of the crawl's 300 files, with a small class that turns HTTP range requests into a file for the Parquet reader. The rows come back in the same three fields (`warc_filename`, `warc_record_offset`, `warc_record_length`) as the CDX reply, so the range fetch below applies unchanged:

```python
# network
import io
import os
import time

import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

HEAD = {"User-Agent": os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")}
PART = ("https://data.commoncrawl.org/cc-index/table/cc-main/warc/"
        "crawl=CC-MAIN-2026-39/subset=warc/"
        "part-00000-c0b5852a-5611-4d53-9ab6-ef33a48d5e8b.c000.zstd.parquet")


class RangeFile(io.RawIOBase):
    """A read-only file whose bytes come from HTTP Range requests."""

    def __init__(self, url):
        self.url, self.pos, self.fetched = url, 0, 0
        self.size = int(requests.head(url, headers=HEAD).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: 0, 1: self.pos, 2: self.size}[whence] + offset
        return self.pos

    def readinto(self, buf):
        start, end = self.pos, min(self.pos + len(buf), self.size)
        if end <= start:
            return 0
        time.sleep(2)  # two seconds between requests to one host
        reply = requests.get(self.url, headers={
            **HEAD, "Range": f"bytes={start}-{end - 1}"})
        buf[:end - start] = reply.content
        self.fetched += end - start
        self.pos = end
        return end - start


key = "com,blogspot,dantevada)"  # SURT form of the host dantevada.blogspot.com
remote = RangeFile(PART)
part = pq.ParquetFile(remote)
meta = part.metadata
keep = []
for i in range(meta.num_row_groups):
    stats = meta.row_group(i).column(0).statistics  # column 0 is url_surtkey
    if stats.min <= key + "\uffff" and stats.max >= key:
        keep.append(i)
print(meta.num_rows, "rows in", meta.num_row_groups, "row groups; read", keep)

columns = ["url_surtkey", "fetch_status", "warc_filename",
           "warc_record_offset", "warc_record_length"]
rows = part.read_row_group(keep[0], columns=columns)
rows = rows.filter(pc.starts_with(rows["url_surtkey"], key))
assert rows.num_rows > 0
for r in rows.to_pylist():
    print(r["url_surtkey"], r["fetch_status"])
    print(" ", r["warc_filename"].rsplit("/", 1)[1],
          r["warc_record_offset"], r["warc_record_length"])
got, size = remote.fetched, remote.size
print(f"fetched {got / 1e6:.0f} MB of {size / 1e6:.0f} MB ({got / size:.0%})")
```

% output-shape

```text
6168911 rows in 4 row groups; read [1]
com,blogspot,dantevada)/ 200
  CC-MAIN-20260915202726-20260915232726-00076.warc.gz 161519017 13995
com,blogspot,dantevada)/2012/02/blog-post_08.html 200
  CC-MAIN-20260915202726-20260915232726-00332.warc.gz 155145929 15777
fetched 33 MB of 503 MB (7%)
```

The crawl's `warc` subset has 300 such files. Each file is sorted by key but covers keys from all over the alphabet, so a query for one host repeats the footer check on every file and reads the matching row group of each. This listing did it for one file and moved 7% of its bytes; a query engine such as Athena runs the same pruning over all 300 files in parallel, which is what makes the columnar route the one for lists of thousands of URLs.


(sec-ch3-corpora-derived-common-crawl)=

### 2.4 Corpora derived from Common Crawl

Most open web corpora are a choice of snapshots plus a choice of pipeline:

```{raw} latex
\begingroup\footnotesize\setlength{\tabcolsep}{3pt}
```

```{tabularcolumns} \Y{0.20}\Y{0.16}\Y{0.13}\Y{0.12}\Y{0.39}
```

| Corpus | Snapshots | Text from | Languages | Notable choices |
| --- | --- | --- | --- | --- |
| C4 {cite}`raffel2019exploring` | 1 (April 2019) | WET | English | Strict line-level rules; a bad-words list that also removed benign text {cite}`dodge2021documenting` |
| mC4 {cite}`xue2021mt5` | 71 | WET | 101 | Per-language C4 rules with a line-length rule instead of terminal punctuation |
| OSCAR {cite}`abadji2022oscar` | 1 per release | WET | 150+ | Language id per line, then per document; minimal filtering by design |
| RefinedWeb {cite}`penedo2023refinedweb` | about 90, up to `2023-06` | WARC, trafilatura | English | Web only; argued that enough filtered web matches curated mixtures |
| FineWeb {cite}`penedo2024fineweb` | 96 | WARC, trafilatura | English | Every filter ablated with 1.7B-parameter training runs; 15T tokens |
| FineWeb-Edu {cite}`penedo2024fineweb` | FineWeb's | FineWeb's | English | Classifier for educational value, 1.3T tokens |
| CulturaX {cite}`nguyen2023culturax` | mC4 + OSCAR | WET | 167 | Merge and re-clean two multilingual corpora |
| FineWeb-2 {cite}`penedo2025fineweb2` | 96 | WARC, trafilatura | over 1,000 | Per-language filter thresholds derived from reference-corpus statistics; duplicates "rehydrated" with weights of 1 to 10 set by cluster size |
| Dolma web {cite}`soldaini2024dolma` | 25 | WET (CCNet) | English | Part of a documented multi-source corpus with released tooling |

```{raw} latex
\endgroup
```

RefinedWeb used all crawls up to `CC-MAIN-2023-06`. The suffix of a crawl id is a year and a week number, so that crawl was published in February 2023, not June.

Two trends run through the table. Later corpora re-extract from WARC with a boilerplate-aware extractor rather than trusting the WET text, and later corpora use far more snapshots, because one snapshot cleans down to about 150 billion tokens (FineWeb: 15T from 96 snapshots) and trillion-token corpora need dozens. FineWeb's ablations add a caution: deduplicating each snapshot separately gave better models than deduplicating across all of them, since global deduplication preferentially removed the good pages that every crawl re-captures and kept the junk that appears once. Chapter 4 compares the two scopes ({ref}`sec-ch4-dedup-scope`).

```{admonition} A Slice of the Crawl
:class: tip
The notebook section {ref}`nb-acq-crawl` downloads the manifest of a current snapshot,
range-fetches a slice of a WARC file at a file and byte offset drawn with a fixed seed,
parses the records, tabulates record types and declared payload types, and shows the
HTTP response and HTML of one page, so that you can see exactly what a crawl contains
before any cleaning.
```

(sec-ch3-crawling-web-yourself)=

## 3. Crawling the Web Yourself

(sec-ch3-published-crawl-not-enough)=

### 3.1 When a published crawl is not enough

Common Crawl covers the open, static, mostly English web as of a few weeks ago. Teams crawl for themselves when they need freshness (news, prices, documentation that changes), depth on a chosen set of sites (every page of a documentation site rather than the few a broad crawl sampled), content that needs a browser to render, or languages and regions the broad crawl under-samples. The cost is real: Common Crawl's snapshot of two to three billion pages represents two to three weeks of fetching at a sustained rate of one to two thousand pages per second, and a team crawling at that rate has to run the politeness and legal machinery below itself.

(sec-ch3-robots-txt-politeness-conventions)=

### 3.2 robots.txt and the politeness conventions

The Robots Exclusion Protocol (RFC 9309) {cite}`koster2022robots` lets a site state, in `/robots.txt`, which user agents may fetch which paths. It was a convention for almost thirty years and became a standard in 2022; it is not enforced by anything but the crawler's own code, which is why it has become the center of the consent question in {ref}`sec-ch3-rights-licenses-consent`. A crawler is expected to:

- Fetch `/robots.txt` before anything else on a host and obey the `Disallow` and `Allow` rules for its own user agent, falling back to the `*` group. When several rules match a path, the most specific one (the longest match) wins.
- Treat the answer to that fetch as RFC 9309 says: a 4xx status (the file is "unavailable") means no restrictions, while a 5xx status or a network failure ("unreachable") means the whole site is disallowed. On a 3xx redirect the crawler "SHOULD follow at least five consecutive redirects, even across authorities", and the file reached is applied to the original host; after more than five redirects the crawler may treat the file as unavailable. Use a cached copy for no more than 24 hours, and parse at least the first 500 KiB of the file.
- Honor `Crawl-delay` where present (it is a widely used extension, not part of RFC 9309) and otherwise limit itself to a request every few seconds per host, with backoff on `429` and `503` responses.
- Identify itself with a user agent string that names the crawler and links to a page explaining it, so that site operators can block it or write to its operator.
- Use the `Sitemap:` entries, which list the URLs a site wants crawled, as the frontier for that site.

A sitemap is an XML file that lists a site's URLs {cite}`sitemapsorg2026`. A `<urlset>` holds one `<url>` per page, with its location in `<loc>` and optionally the date it last changed in `<lastmod>`; a `<sitemapindex>` holds `<sitemap>` entries of the same shape that point to further sitemaps. The `Sitemap:` line of robots.txt gives the starting file, and because one file may hold at most 50,000 URLs and 50 MB uncompressed, a large site lists many files in an index. A crawler that reads the `lastmod` dates can skip what has not changed since its last visit, and the locations it keeps are the seeds of the frontier below:

```python
import xml.etree.ElementTree as ET

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
INDEX = """<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/sitemap-blog.xml</loc>
           <lastmod>2026-11-12</lastmod></sitemap>
  <sitemap><loc>https://example.com/sitemap-docs.xml</loc>
           <lastmod>2025-02-01</lastmod></sitemap>
</sitemapindex>"""
URLSET = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/blog/a</loc><lastmod>2026-11-12</lastmod></url>
  <url><loc>https://example.com/blog/b</loc><lastmod>2026-03-30</lastmod></url>
  <url><loc>https://example.com/blog/c</loc></url>
</urlset>"""


def sitemap_entries(xml_text, since=""):
    """Return ("sitemaps" or "pages", locations changed on or after `since`).

    Untrusted XML needs a hardened parser such as defusedxml in a real crawler.
    """
    root = ET.fromstring(xml_text)
    kind = "sitemaps" if root.tag.endswith("sitemapindex") else "pages"
    found = []
    for entry in root:
        lastmod = entry.findtext("sm:lastmod", default="", namespaces=NS)
        if not lastmod or lastmod >= since:  # no date: nothing says it is old
            found.append(entry.findtext("sm:loc", namespaces=NS))
    return kind, found


print(*sitemap_entries(INDEX, since="2026-01-01"))
print(*sitemap_entries(URLSET, since="2026-06-01"))
```

% output

```text
sitemaps ['https://example.com/sitemap-blog.xml']
pages ['https://example.com/blog/a', 'https://example.com/blog/c']
```

{numref}`fig-ch3-robots-decision` lays out the first two rules as the decision a crawler makes before each URL.

:::{figure} figures/robots-decision.*
:name: fig-ch3-robots-decision
:alt: Flow chart from top to bottom. Fetch /robots.txt, reusing a copy for at most 24 hours. A 4xx answer means no restrictions; a 5xx answer or network failure means the whole site is disallowed; a 3xx answer is followed, at least five redirects in a row and even to another host, and the file reached is applied to the original host, while more than five redirects may be treated as unavailable; a 2xx answer is parsed for at least the first 500 KiB and the group for this user agent, or else the star group, is used. The longest matching rule then decides: Allow or no match leads to a fetch at the host's delay, Disallow to skipping the URL.

The robots.txt decision under RFC 9309. Note the asymmetry between the two failures: a
missing file (4xx) permits everything, while a server error or network failure (5xx)
forbids everything. A redirect (3xx) is followed to the file it leads to, up to at least
five in a row; only beyond five may the crawler give up and treat the file as
unavailable.
:::

Python's standard library parses the file:

```python
# network
import os
import urllib.request
import urllib.robotparser

# Fetch with a named user agent: RobotFileParser.read() would send Python's default
# agent, which Wikipedia answers with 403, and the parser would then disallow
# everything.
agent = os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")
req = urllib.request.Request("https://en.wikipedia.org/robots.txt",
                             headers={"User-Agent": agent})
robots_txt = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")

rp = urllib.robotparser.RobotFileParser()
rp.parse(robots_txt.splitlines())

for agent in ["*", "Googlebot", "CCBot", "GPTBot", "Google-Extended"]:
    article = rp.can_fetch(agent, "https://en.wikipedia.org/wiki/Language_model")
    search = rp.can_fetch(
        agent, "https://en.wikipedia.org/w/index.php?title=Special:Search&search=x")
    print(f"{agent:16} article: {article}  search: {search}")
```

% output-shape

```text
*                article: True  search: False
Googlebot        article: True  search: False
CCBot            article: True  search: False
GPTBot           article: True  search: False
Google-Extended  article: True  search: False
```

The standard-library parser predates RFC 9309: it ignores the `*` wildcard and the end-of-path anchor (a final dollar sign) and applies the first matching rule instead of the longest, so `Disallow: /*.pdf` does not block `/a/b.pdf`, and `Disallow: /a` followed by `Allow: /a/b` blocks `/a/b/c`, which the RFC allows. Use it for inspection, as here; a production crawler needs a compliant parser, such as Google's `robotstxt` library or `protego` in Python.

The user agents in that list are the ones that matter for training data: `CCBot` is Common Crawl, `GPTBot` is OpenAI's crawler, `Google-Extended` is Google's token for opting out of model training without leaving search, and `ClaudeBot`, `anthropic-ai`, `Bytespider`, `PerplexityBot` and `Applebot-Extended` are the others most often named in robots files. A site that disallows `CCBot` removes itself from every future Common Crawl snapshot, and therefore from every corpus derived from one.

(sec-ch3-crawler-architecture)=

### 3.3 Crawler architecture

A crawler at corpus scale is a distributed system with five parts:

- **Frontier**: the queue of URLs to fetch, prioritized by some estimate of value (links in, freshness, site quality) and partitioned by host so that per-host politeness can be enforced locally. The frontier is the crawl's main state and its main engineering problem; Common Crawl's is that of Apache Nutch {cite}`commoncrawl2026faq`.
- **Scheduler**: picks the next URL to hand out, from the host whose delay has expired, consulting a cache of that host's robots.txt. Politeness is enforced here and nowhere else.
- **Fetchers**: many workers, each holding a pool of connections, resolving DNS, fetching with timeouts, and respecting the per-host schedule. Throughput is bounded by politeness rather than bandwidth: at one request per two seconds per host, a hundred thousand hosts in flight give at most $10^5 / 2 =$ fifty thousand pages per second. That is a ceiling, not a rate: most hosts have only a few URLs queued, and slow servers hold connections open.
- **Deduplication of URLs** before fetching (canonicalization, a seen-set of URL hashes) and of content after fetching (so that mirrors and soft-404 pages do not fill the frontier with links).
- **Storage** as WARC, with the request, response and metadata records written together, because the raw response is what later re-extraction needs. Writing only extracted text saves space and throws away the option of re-extracting with a better tool next year.

{numref}`fig-ch3-crawler-architecture` arranges the five parts, and the stages between them, in the order a URL passes through them.

The frontier has to answer two questions for every URL: has it been seen, and when should it be fetched. The seen-set at web scale holds billions of URLs, too many to keep as strings in memory, so crawlers keep fixed-size fingerprints and test new URLs against them in batches {cite}`heydon1999mercator,lee2009irlbot`. A Bloom filter {cite}`bloom1970space`, the structure {ref}`Chapter 4 <sec-ch4-exact-deduplication>` uses for paragraph hashes, trades a small false-positive rate for less memory still: with the optimal number of hash functions it needs $-\ln p / (\ln 2)^2$ bits per URL for a false-positive rate $p$. A false positive here means a new URL is wrongly taken as seen and never fetched, which a crawl can afford at 1%:

```python
import math

p, urls = 0.01, 10e9                       # false-positive rate; URLs in the seen-set
bits = -math.log(p) / math.log(2) ** 2     # bits per URL, optimal number of hashes
probes = round(bits * math.log(2))
print(f"{bits:.2f} bits and {probes} hash functions per URL")
pairs = urls ** 2 / 2 ** 65               # expected colliding pairs of 64-bit hashes
print(f"Bloom filter: {bits * urls / 8 / 1e9:.1f} GB;",
      f"64-bit fingerprints: {8 * urls / 1e9:.0f} GB, {pairs:.1f} colliding pairs")
```

% output

```text
9.59 bits and 7 hash functions per URL
Bloom filter: 12.0 GB; 64-bit fingerprints: 80 GB, 2.7 colliding pairs
```

Ten billion URLs fit in 12 GB, a fraction of a single machine's memory, against 80 GB for 64-bit fingerprints. Those are not exact either: among $10^{10}$ URLs about 2.7 pairs collide, so a few URLs are wrongly taken as seen, a rate near $5 \times 10^{-10}$ per lookup against the Bloom filter's $10^{-2}$. When to fetch is a priority order within each host's queue, set by an estimate of a page's value (links pointing to it, its host's quality) and, for pages already fetched, of how likely it is to have changed since. Two limits protect the frontier from pages that generate links without end, such as calendars with a "next month" link or URLs carrying a fresh session id: a page budget per host, and a cap on URL length and link depth. How often to re-fetch pages that change at different rates is its own problem, worked through below; {cite:t}`cho2003effective` study such refresh policies analytically and experimentally, and {cite:t}`olston2010web` survey all of these questions.

:::{figure} figures/crawler-architecture.*
:name: fig-ch3-crawler-architecture
:alt: Left to right. Seed URLs and sitemaps fill a frontier of per-host queues. A scheduler, consulting a robots.txt cache, serves the host whose delay expires first, at most one request per host every two seconds. Fetchers handle DNS, connections, timeouts and back-off. Their output goes to a WARC writer and to content deduplication, then to link extraction and canonicalization, and the new URLs join their host's queue in the frontier.

A crawler's parts. Politeness lives in the scheduler, which is why throughput is bounded
by the number of hosts in flight rather than by bandwidth. The dashed box closes the
loop: links found in fetched pages, once canonicalized and checked against the seen-set,
go back into the frontier.
:::

Rendering is the expensive option. A page that builds its content with JavaScript yields a near-empty HTML response to a plain fetcher; running a headless browser recovers the content at many times the cost per page. Common Crawl's crawler does not render: its FAQ says "Currently, JavaScript is not executed" {cite}`commoncrawl2026faq`, which is one reason single-page applications are under-represented in corpora built on it.

**Re-fetch scheduling.** Suppose a page changes at random times at an average rate of $\chi_p$ changes per day (a Poisson process, the model {cite:t}`cho2003effective` propose and find a good description of real pages), and the crawler re-fetches it $f_p$ times per day at even intervals. A local copy is *fresh* while the page has not changed since the last fetch, and the expected fraction of time it is fresh is

$$F(\chi_p, f_p) = \frac{1 - e^{-\chi_p / f_p}}{\chi_p / f_p}.$$

With a fixed budget of fetches per day, the question is how to divide it. The obvious policy, refreshing in proportion to the change rate, gives every page the same ratio $\chi_p / f_p$ and so the same freshness; the uniform policy gives every page the same $f_p$. The listing compares them for two pages, one that changes daily and one nine times a day, with a budget of two fetches a day, and also searches for the best split:

```python
import math

RATES = [1.0, 9.0]  # changes per day of a slow page and a fast page
BUDGET = 2.0        # refreshes per day the crawler can afford, in total


def freshness(rate, refreshes_per_day):
    """Expected fraction of time a local copy is current when the page changes as a
    Poisson process of `rate` per day and is re-fetched every 1/f days."""
    x = rate / refreshes_per_day
    return (1 - math.exp(-x)) / x


def average(split):
    return sum(freshness(r, f) for r, f in zip(RATES, split)) / len(RATES)


uniform = [BUDGET / len(RATES)] * len(RATES)
proportional = [BUDGET * r / sum(RATES) for r in RATES]
best = max(([BUDGET * i / 1000, BUDGET * (1 - i / 1000)] for i in range(1, 1000)),
           key=average)
for name, split in [("uniform", uniform), ("proportional", proportional),
                    ("best split", best)]:
    print(f"{name:>12}: {split[0]:.2f} and {split[1]:.2f} fetches/day,",
          f"average freshness {average(split):.3f}")
assert average(uniform) > average(proportional)
assert average(best) > average(uniform) and best[1] < best[0]
```

% output

```text
     uniform: 1.00 and 1.00 fetches/day, average freshness 0.372
proportional: 0.20 and 1.80 fetches/day, average freshness 0.199
  best split: 1.76 and 0.24 fetches/day, average freshness 0.395
```

Proportional refreshing is the worse of the two, and the best split goes the other way from it: it spends most of the budget on the slow page. A page that changes nine times a day is stale again soon after any fetch, so an extra fetch buys little, while an extra fetch of a page that changes once a day buys a long stretch of freshness. The conditional requests of the next listing supply the data such a policy needs: a `304` answer says the page has not changed since the last fetch, and the fraction of `304` answers over time estimates $\chi_p$.

Two of these parts are worth seeing in code. URL canonicalization decides when two strings name the same page. RFC 3986 {cite}`berners2005rfc3986` defines the normalizations that never change what a URL refers to: lowercase the scheme and host, write the hexadecimal digits of percent-encodings in uppercase and decode those that stand for unreserved characters, remove `.` and `..` path segments (but not empty ones: `/a//b` and `/a/b` may name different resources), and drop a port that is the scheme's default. Crawlers add rules the RFC does not sanction: dropping the fragment, which never reaches the server, sorting the query parameters, and stripping session and tracking parameters. These merge many duplicates and occasionally two different pages, so each is a per-site decision. Conditional requests make re-crawling cheap: a server that sent an `ETag` with a page answers a later request that carries the tag in `If-None-Match` with `304 (Not Modified)` and no content when the page has not changed {cite}`fielding2022rfc9110`.

The loop below puts these together with the robots.txt rules of {ref}`sec-ch3-robots-txt-politeness-conventions`. The frontier is a queue per host, and the loop always serves the host whose delay expires first, so requests to different hosts interleave while each host sees at most one request every two seconds. robots.txt is fetched once per host and read as RFC 9309 says: a 4xx answer means no restrictions and a 5xx answer means nothing may be fetched. The transport and the clock are simulated so that the example runs offline and gives the same answer every time: when a host is not yet due, `get` calls the `sleep` hook, which by default only advances the simulated clock. A working single-threaded crawler needs both swapped: `transport` for an HTTP call, and `clock` and `sleep` for `time.monotonic` and `time.sleep`, so that the wait really happens.

```python
import re
from collections import Counter, deque
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
DEFAULT_PORT = {"http": 80, "https": 443}


def remove_dot_segments(path):
    """RFC 3986, section 5.2.4: drop "." and ".." segments. Empty segments ("//") are
    significant and are kept."""
    segments, out = path.split("/")[1:], []
    for seg in segments:
        if seg == "..":
            if out:
                out.pop()
        elif seg != ".":
            out.append(seg)
    trailing = segments[-1] in (".", "..") and out   # "/a/b/.." names "/a/"
    return "/" + "/".join(out) + ("/" if trailing else "")


def canonicalize(url):
    """RFC 3986 normalization (case, default port, percent-encoding, dot segments),
    then four crawler conventions that the RFC does not make: drop the fragment,
    sort the query parameters, drop an empty query ("?" alone, which RFC 3986
    6.2.3 says cannot be assumed equivalent), and drop any userinfo
    (user:password@), which a crawler should neither store nor send. IPv6 hosts
    keep their brackets."""
    p = urlsplit(url)
    scheme, host = p.scheme.lower(), (p.hostname or "").lower()
    if ":" in host:      # IPv6 literal: urlsplit strips the brackets RFC 3986 requires
        host = f"[{host}]"
    default = p.port in (None, DEFAULT_PORT.get(scheme))
    netloc = host if default else f"{host}:{p.port}"

    def percent(m):
        ch = chr(int(m[0][1:], 16))
        return ch if ch in UNRESERVED else m[0].upper()

    path = remove_dot_segments(re.sub(r"%[0-9A-Fa-f]{2}", percent, p.path or "/"))
    query = "&".join(sorted(p.query.split("&"))) if p.query else ""
    return urlunsplit((scheme, netloc, path, query, ""))


class PoliteCrawler:
    """One robots.txt per host, a minimum delay per host, and conditional re-fetches by
    ETag. The clock and the wait are hooks: by default a simulated clock that waiting
    advances; for real use pass clock=time.monotonic and sleep=time.sleep."""

    def __init__(self, transport, agent="examplebot", delay=2.0,
                 clock=None, sleep=None):
        self.transport, self.agent, self.delay = transport, agent, delay
        self.now = 0.0                          # the simulated clock, in seconds
        self.clock = clock or (lambda: self.now)
        self.sleep = sleep or self._advance
        self.robots, self.ready = {}, {}    # per host: robots.txt, next allowed time
        self.etags, self.store = {}, {}     # per URL: last ETag, last body

    def _advance(self, seconds):
        self.now += seconds

    def get(self, url):
        host = urlsplit(url).netloc
        wait = self.ready.get(host, 0.0) - self.clock()
        if wait > 0:
            self.sleep(wait)                # wait for this host's turn
        self.ready[host] = self.clock() + self.delay
        headers = {"User-Agent": self.agent}
        if url in self.etags:
            headers["If-None-Match"] = self.etags[url]
        status, reply, body = self.transport(url, headers)
        if status == 304:                   # not modified: reuse the stored copy
            body = self.store[url]
        elif status == 200 and "ETag" in reply:
            self.etags[url], self.store[url] = reply["ETag"], body
        return status, body

    def allowed(self, url):
        """Check robots.txt. The sketch neither follows redirects nor refreshes."""
        p = urlsplit(url)
        if p.netloc not in self.robots:
            status, body = self.get(f"{p.scheme}://{p.netloc}/robots.txt")
            rp = RobotFileParser()
            if status == 200:
                rp.parse(body.splitlines())
            elif 400 <= status < 500:  # unavailable: no restrictions
                rp.parse([])
            else:                      # 3xx, 5xx, anything else: unreachable
                rp.disallow_all = True
            self.robots[p.netloc] = rp
        return self.robots[p.netloc].can_fetch(self.agent, url)

    def crawl(self, seeds):
        frontier, seen, counts = {}, set(), Counter()

        def add(url):
            url = canonicalize(url)
            if url in seen:
                counts["duplicate"] += 1
            else:
                seen.add(url)
                frontier.setdefault(urlsplit(url).netloc, deque()).append(url)

        for url in seeds:
            add(url)
        while any(frontier.values()):
            waiting = (h for h, q in frontier.items() if q)
            host = min(waiting, key=lambda h: self.ready.get(h, 0.0))
            url = frontier[host].popleft()
            if not self.allowed(url):
                counts["blocked"] += 1
                continue
            status, body = self.get(url)
            counts[status] += 1
            for link in re.findall(r'href="([^"]+)"', body):
                add(urljoin(url, link))
        return counts


SITE = {  # a simulated web: URL -> (ETag, body)
    "https://example.com/robots.txt": ("r1", "User-agent: *\nDisallow: /private/\n"),
    "https://example.com/": ("e1", '<a href="/a">a</a> <a href="/private/x">x</a> '
                                   '<a href="https://example.org/">org</a>'),
    "https://example.com/a": ("e2", '<a href="/A/../a#part-2">this page</a> '
                                    '<a href="/b">b</a>'),
    "https://example.com/b": ("e3", "no links"),
    "https://example.org/": ("e4", '<a href="/c">c</a> '
                                   '<a href="HTTPS://EXAMPLE.ORG:443/c?">c again</a>'),
    "https://example.org/c": ("e5", "no links"),
}


def transport(url, headers):
    if url not in SITE:
        return 404, {}, ""
    etag, body = SITE[url]
    if headers.get("If-None-Match") == etag:
        return 304, {"ETag": etag}, ""
    return 200, {"ETag": etag}, body


print(canonicalize("HTTP://Example.COM:80/a/./b/../c/%7euser?b=2&a=1#top"))
print(canonicalize("http://user:secret@[2001:DB8::1]:8080/x"))  # userinfo dropped
print(canonicalize("https://[::1]:443/"))
print(canonicalize("http://example.com/a//b/./c"))   # the empty segment is kept
bot = PoliteCrawler(transport)
first = bot.crawl(["https://Example.com/"])
print("first crawl: ", dict(sorted(first.items(), key=str)), "in", bot.now,
      "simulated s")
start = bot.now
second = bot.crawl(["https://example.com/"])
print("second crawl:", dict(sorted(second.items(), key=str)), "in", bot.now - start,
      "simulated s")
```

% output

```text
http://example.com/a/c/~user?a=1&b=2
http://[2001:db8::1]:8080/x
https://[::1]/
http://example.com/a//b/c
first crawl:  {'blocked': 1, 'duplicate': 2, 200: 5} in 6.0 simulated s
second crawl: {'blocked': 1, 'duplicate': 2, 304: 5} in 6.0 simulated s
```

The first crawl fetches five pages and two robots.txt files in six simulated seconds, skips the disallowed `/private/x`, and recognizes two links (`/A/../a#part-2` and the upper-case `example.org` URL with an empty query, which canonicalization treats as the URL without one by a crawler convention and not by the RFC) as pages it already has. The re-crawl receives five `304` answers and transfers no page content. Against a real server the same exchange looks like this:

```python
# network
import os
import time
import urllib.error
import urllib.request

URL = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-39/warc.paths.gz"
UA = os.environ.get("LLMATLAS_USER_AGENT", "llmatlas-notebook/0.1")


def get(headers):
    req = urllib.request.Request(URL, headers={"User-Agent": UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers, len(r.read())
    except urllib.error.HTTPError as e:   # urllib raises on 304, not an error here
        return e.code, e.headers, len(e.read())


status, headers, size = get({})
print(status, size, "bytes, ETag", headers["ETag"].strip('"')[:8])
time.sleep(2)              # politeness: two seconds between requests to one host
status, _, size = get({"If-None-Match": headers["ETag"]})
print(status, size, "bytes")
```

% output-shape

```text
200 201289 bytes, ETag e621c46e
304 0 bytes
```

```{admonition} Reading robots.txt
:class: note
The notebook section {ref}`nb-acq-robots` parses with the standard library the
robots.txt files of Wikipedia (one of the most detailed on the web, with 428 Disallow
rules for every agent (`*`), which treats search and training crawlers alike) and of a
major news site, which allows Googlebot and disallows every AI crawler by name. It
counts the Disallow rules per agent in each.
```

(sec-ch3-economics)=

### 3.4 The economics

Order-of-magnitude arithmetic for a billion-page crawl: assume pages average about 100 KB of HTML (the page fetched in {ref}`sec-ch3-index-targeted-retrieval` was 34 KB), so the raw fetch is 100 TB of transfer, which is free as cloud ingress but costs a few thousand dollars in NAT-gateway or bandwidth charges, and far more if the crawl is then moved between regions or clouds. Assume gzip shrinks the WARC output to about a quarter of that (the record of {ref}`sec-ch3-index-targeted-retrieval` is 9,882 bytes compressed from 34,353 of HTML, 29%). At an aggregate 5,000 pages per second, a tenth of the ceiling above, a billion pages take $10^9 / (5{,}000 \times 86{,}400) \approx 2.3$ days of wall-clock time; politeness, not hardware, sets that rate. Extraction and filtering, by comparison, are a few CPU-milliseconds per page and run in hours. The dominant cost of a serious crawl is not any of these but the engineering of the frontier and the handling of the long tail of malformed responses, redirect loops, tar pits and hosts that respond slowly enough to tie up a fetcher without timing out.

(sec-ch3-curated-sources)=

## 4. Curated Sources

The web supplies volume; the sources in this section supply things the web is short of: long documents, reliable facts, technical reasoning, working code and natural conversation. Each is a small fraction of a corpus, and the smallest of them (Wikipedia, books) are typically seen two or more times.

(sec-ch3-wikipedia-wikimedia-projects)=

### 4.1 Wikipedia and the Wikimedia projects

Wikipedia is in every corpus. The Wikimedia Foundation publishes complete database dumps of every language edition twice a month (`dumps.wikimedia.org`), under CC BY-SA and the GFDL, as XML with the wikitext markup; converting wikitext to clean text (templates, infoboxes, references, tables) is the main work, and the `wikimedia/wikipedia` dataset on Hugging Face is a widely used pre-converted version. English Wikipedia is about 5.3 billion words in 7.3 million articles (the site's statistics, read on 13 November 2026 {cite}`wikipedia2026statistics`), roughly 6 to 7 billion tokens or about 0.04% of a 15T-token corpus, and models have seen it about two to three and a half times: 2.45 epochs in LLaMA and 3.4 in GPT-3 {cite}`touvron2023llama,brown2020language`. Wiktionary, Wikibooks, Wikisource and Wikidata are published the same way; Dolma includes Wikibooks alongside Wikipedia.

(sec-ch3-books)=

### 4.2 Books

Books are the best source of long-range structure (a plot, an argument, a chapter that refers to an earlier one), and the most legally fraught. The sources that have been used:

- **Project Gutenberg**: nearly 80,000 free books (79,802 on 13 November 2026 {cite}`gutenberg2026home`), mostly English works published before 1931 (the US public-domain cutoff as of 2026), downloadable in bulk. Legally clean and stylistically dated; it is the books portion of Dolma and part of LLaMA's.
- **BookCorpus**: 11,000 self-published novels scraped from Smashwords for a 2015 study {cite}`zhu2015aligning` that aligned books with their film adaptations, and later used to train BERT and the original GPT; a retrospective datasheet {cite}`bandy2021addressing` found duplicates, skewed genres and books whose licenses did not permit the use.
- **Books3**: a collection of books copied from a shadow library, included in The Pile {cite}`gao2020pile` and used by several models. Where copies came from is the question the books lawsuits of {ref}`sec-ch3-copyright-training-question` turned on, so a corpus that contains it carries that legal exposure.
- **Licensed collections**: publishers and aggregators now sell training licenses, and the large labs increasingly buy them, or buy and scan print books, rather than rely on shadow libraries. The US rulings of 2025 that made the difference between a lawful and a pirated copy matter are in {ref}`sec-ch3-copyright-training-question`.

The practical rule in 2026: use public-domain text, text whose license explicitly allows the use, or text you have licensed, and keep the receipts. Books3 in The Pile is the cautionary example of what happens otherwise.

(sec-ch3-scientific-technical-papers)=

### 4.3 Scientific and technical papers

Papers supply formal argument, mathematics and domain vocabulary. The accessible sources:

- **arXiv** publishes bulk access to its LaTeX sources and PDFs (on S3, requester pays). The LaTeX source, converted with macro expansion and with bibliographies and comments removed, is what The Pile and LLaMA used; it is also the main source of mathematical text in the open corpora.
- **S2ORC** {cite}`lo2020s2orc` and its successor **peS2o** {cite}`soldaini2023pes2o` provide full text for roughly eight million open-access papers from Semantic Scholar, plus titles and abstracts for tens of millions more (peS2o v2: 8.2M full-text papers and 30.6M abstracts), already converted from PDF, with licenses tracked per paper. Dolma's papers portion is peS2o.
- **PubMed Central** offers the open-access subset of biomedical literature in bulk XML, license by license.

PDF extraction is the technical problem: two-column layouts, equations, tables and footnotes come out scrambled from generic tools, which is why pipelines prefer LaTeX or publisher XML wherever it exists. Where only PDFs exist, document-understanding models trained for the job, Nougat {cite}`blecher2023nougat` for academic PDFs and olmOCR {cite}`poznanski2025olmocr` at corpus scale, now recover text, equations and tables well enough that scanned books and papers have become an acquisition channel in their own right.

(sec-ch3-source-code)=

### 4.4 Source code

Code is a few percent of most mixtures, 4.5% of LLaMA's and 8% of The Pile's, but 13% of Dolma's and none of FineWeb's, which is web text only; it is kept even in models not aimed at programming. LLaMA and Gopher both include a few percent of code {cite}`touvron2023llama,rae2021scaling`; Dolma's mixing study (Appendix M) {cite}`soldaini2024dolma` and the data-constrained scaling experiments {cite}`muennighoff2023scaling` test what adding code does for natural-language tasks. The sources:

- **The Stack** {cite}`kocetkov2022stack` is 3 TB of source files from public GitHub repositories, restricted to permissively licensed repositories (MIT, Apache, BSD and similar) determined from the repository's license file, near-deduplicated, with an opt-out form that lets developers remove their repositories from future versions. **The Stack v2** {cite}`lozhkov2024starcoder2` was built from Software Heritage's archive instead, which preserves provenance for every file.
- **GitHub archives** such as GH Archive (event streams) and the Software Heritage graph are the raw material behind both; building from them directly means doing the license detection yourself.
- **Notebooks, issues, commits and documentation** are included alongside code in StarCoder's training data because they pair code with natural-language explanation.

License filtering is the acquisition-time decision that matters most for code: a permissive-only corpus is smaller but can be released and used without the copyleft question that GPL-licensed code raises for model outputs.

(sec-ch3-forums-dialogue-question-answering)=

### 4.5 Forums, dialogue and question answering

Conversation teaches turn-taking, register and the question-answer form that instruction tuning ({ref}`roadmap-post-training-and-alignment`) later refines. The sources have become harder to obtain:

- **Reddit** was the backbone of WebText (GPT-2 {cite}`radford2019language` used links with at least three upvotes as a quality signal) and of the Pushshift archive used by many corpora, until Reddit restricted bulk access through its API. Its Data API Terms, in effect since 19 June 2023 and last revised in July 2026, grant no right to use Reddit content "for training a machine learning or AI model, without the express permission of rightsholders" {cite}`reddit2026dataapi`, and access for training is licensed instead: Google announced in February 2024 that it has access to the Data API to "display, train on, and otherwise use" Reddit content {cite}`google2024reddit`.
- **Stack Exchange** published a quarterly dump of every question and answer on its sites under CC BY-SA, which is in The Pile and LLaMA, and is the single best source of paired technical questions and answers; in July 2024 the download moved behind a login, with an agreement that the file is for "projects that do not include training a large language model" {cite}`stackexchange2024dump`.
- **Mailing lists, IRC logs and email archives**, the Ubuntu IRC logs and the Enron email corpus in The Pile, supply informal dialogue.
- **Subtitles and transcripts** (OpenSubtitles, YouTube captions) supply spoken-register dialogue; their licensing is unsettled and most recent open corpora omit them.

(sec-ch3-news-government-legal-text)=

### 4.6 News, government and legal text

News archives are mostly licensed commercially now; the open sources are the CC-NEWS subset of Common Crawl (news sites crawled continuously, used by RoBERTa {cite}`liu2019roberta`) and national-library collections with their own terms. Government text, court opinions (the FreeLaw project in The Pile), patents (USPTO), legislation and regulatory filings are public records in most jurisdictions, large, well edited and formal, and under-used relative to their quality because they have to be collected site by site.

```{admonition} Comparing Sources
:class: tip
The notebook section {ref}`nb-acq-sources` samples five hosted corpora through the
Hugging Face rows API: the first hundred rows each of FineWeb, FineWeb-Edu and C4, the
first fifty of the French portion of FineWeb-2, and fifty Wikipedia articles in five
pulls of ten at offsets drawn with a fixed seed (the first rows in page-id order are the
oldest and longest articles, so they would mislead). It compares document lengths,
language scores and the FineWeb-Edu educational score, and measures domain concentration
on 2,000 C4 host names from twenty pulls at seeded random offsets. The rows API serves
only the first 2,207,027 rows of C4's English split, so those offsets are random within
that range only. The samples are small: Wikipedia's median article (206.5 words in 50)
is only slightly longer than C4's median page (188.5 words in 100), a gap too narrow to
read as a general result.
```

(sec-ch3-licensed-news)=

### 4.7 Licensed news and publisher archives

A growing share of high-quality text is obtained by contract. A publisher that does not want its archive crawled can license it instead: in 2023 the Associated Press and OpenAI announced an agreement under which "OpenAI [is] licensing part of AP's text archive" {cite}`ap2023openai`. Such deals change what acquisition means. The text arrives as a delivered archive and not as a crawl, so there is no robots.txt to consult and no snapshot to cite; what the manifest of {ref}`sec-ch3-provenance-documentation` records is the contract, with its permitted uses, its term and any attribution or deletion duties. The scale and the price are generally not disclosed, and the press release above states neither, so a corpus builder cannot treat licensed news as a source of known size: its $U_s$ of {ref}`sec-ch3-epoch-cap-per-source` is whatever the contract delivers. The cost is also what makes it a mixture decision: a licensed archive is small, edited, recent and paid for by the token, so it is a candidate for a high weight $w_s$ only if the epoch cap leaves room for it.

(sec-ch3-first-party-conversations)=

### 4.8 First-party conversations

A provider that runs a deployed assistant holds a second kind of source: the conversations of its users, usable only under the terms those users accepted. Consent is therefore the acquisition step. In August 2025 one provider changed its consumer terms so that users choose whether their chats and coding sessions are used for model training, and tied the choice to retention: 30 days if they decline, five years if they allow it {cite}`anthropic2025consumer`. Five years is 1,825 days, a factor of 61 over 30 days, so the choice decides not only whether a conversation enters the corpus but how long it can. The same announcement says the updates do not apply to services under its commercial terms, which is the usual split: business and API traffic is excluded by default and consumer traffic is opt-in or opt-out by setting. Conversations are personal data of the kind treated in {ref}`sec-ch3-personal-data-sensitive-sources`, written by people who expected an answer and not publication, so the filtering of Chapter 4 and the consent record both have to be in place before such text is used, and the consent state has to travel with each record so that a later withdrawal can be honored.

(sec-ch3-rights-licenses-consent)=

## 5. Rights, Licenses and Consent

This section describes the landscape as of 2026 and is not legal advice. The questions are live in several courts, the answers differ by jurisdiction, and a team assembling a corpus for anything beyond research should have counsel involved from the start.

(sec-ch3-copyright-training-question)=

### 5.1 Copyright and the training question

Almost every document on the web is under copyright from the moment it is written; the open question is whether copying it into a training set, and training on it, is permitted without a license.

- In the **United States** the argument is fair use: that training is a transformative, non-expressive use that does not substitute for the work. Suits by authors, publishers and news organizations against model developers were filed from 2023 on, and three rulings of 2025 show the factors that have decided cases so far.
  - *How the copies were obtained.* In *Bartz v. Anthropic* (N.D. Cal., June 2025) training on lawfully acquired books was held fair use, while building a permanent central library from pirated copies was not and was sent to trial {cite}`bartz2025order`. The case then settled for a fund of 1.5 billion dollars, and the court "granted final approval" of the settlement on 20 July 2026 {cite}`bartz2026settlement`.
  - *Market harm.* In *Kadrey v. Meta* (N.D. Cal., June 2025) Meta won summary judgment on fair use although it had downloaded books from shadow libraries, because the plaintiffs offered no evidence that the models' outputs would dilute the market for their books. The judge wrote that the ruling "does not stand for the proposition that Meta's use of copyrighted materials to train its language models is lawful" and limited it to those thirteen authors and that record {cite}`kadrey2025order`.
  - *A competing, non-generative tool.* In *Thomson Reuters v. Ross Intelligence* (D. Del., February 2025) the court rejected fair use for a legal-research tool trained on Westlaw headnotes to compete with Westlaw {cite}`thomsonreuters2025ross`.

  The US Copyright Office's report on generative AI training (Part 3, pre-publication version of May 2025) reaches no single answer either: "some uses of copyrighted works for generative AI training will qualify as fair use, and some will not" {cite}`usco2025part3`.
- The **European Union**'s Digital Single Market directive (Directive 2019/790) created a text-and-data-mining exception for scientific research (Article 3) and a general one (Article 4) that applies only where use has not been "expressly reserved by their rightholders in an appropriate manner, such as machine-readable means in the case of content made publicly available online" {cite}`eu2019dsm`. The AI Act (Regulation 2024/1689) requires providers of general-purpose models to "put in place a policy to comply with Union law on copyright" that identifies and complies with those reservations (Article 53(1)(c)), and to publish "a sufficiently detailed summary about the content used for training" on a template from the AI Office (Article 53(1)(d)) {cite}`eu2024aiact`. A robots.txt directive is the most common machine-readable form, but the directive names none, and which forms count is contested.
- **European courts** have begun to apply these rules to models.
  - In *GEMA v. OpenAI* the Munich Regional Court I (judgment of 11 November 2025, 42 O 14139/24, under appeal) held that song lyrics memorized by a language model are reproduced in it, and that the text-and-data-mining exception covers copies made in assembling the training data but not the training of the model {cite}`gema2025openai`.
  - In England, *Getty Images v. Stability AI* ([2025] EWHC 2863 (Ch), 4 November 2025) went the other way on the model itself: the judge found that the model "does not store any of those Copyright Works" and that "the model weights are not themselves an infringing copy", so Getty's secondary-infringement claim failed {cite}`getty2025stability`. Getty had abandoned its claim about the training itself, for want of evidence that it took place in the United Kingdom, so the court did not rule on it.
  - In Germany, *Kneschke v. LAION* concerned a photograph that had been downloaded to check a caption against its image when a public image-text dataset was assembled. The Hamburg Regional Court (27 September 2024, 310 O 227/23) dismissed the photographer's claim: the copy was covered by the exception for text and data mining for scientific research (section 60d of the Copyright Act), and whether the general exception (section 44b) also applied "does not need to be conclusively assessed" {cite}`lghamburg2024laion`. On the section 44b point, which the court discussed anyway, it said it "tends to regard a reservation of use written solely in 'natural language' as 'machine understandable'", depending on the technology available when the work was used. The Hanseatic Higher Regional Court dismissed the appeal on 10 December 2025 (5 U 104/24). On the reservation it held that the claimant had not shown that his natural-language reservation was machine-readable at the time of use, and it allowed an appeal on points of law, since no highest-court ruling on sections 44b and 60d exists {cite}`olghamburg2025laion`.
- **Japan**'s Copyright Act, Article 30-4, permits exploiting a work in any case "in which it is not a person's purpose to personally enjoy or cause another person to enjoy the thoughts or sentiments expressed in that work", which covers training, unless the use "would unreasonably prejudice the interests of the copyright owner" {cite}`japan2024copyright`. The **United Kingdom** allows copying for text and data analysis only "for the sole purpose of research for a non-commercial purpose" (section 29A of the Copyright, Designs and Patents Act 1988) {cite}`uk1988cdpa29a`. Its government published a consultation on a broader exception on 17 December 2024 {cite}`ukipo2024consultation`, and the Data (Use and Access) Act 2025 then required it to publish, within nine months of the Act's passing, an economic impact assessment of the consultation's four policy options and a report on the use of copyright works in developing AI systems (sections 135 and 136) {cite}`uk2025duaa`. Both were published in March 2026, and the report states: "We will not introduce reforms to copyright law until we are confident that they will meet our objectives for the economy and UK citizens" {cite}`ukdsit2026report`. **Singapore**'s Copyright Act 2021 has a statutory exception for computational data analysis: in its Part 5 (permitted uses), Division 8, section 243 defines what computational data analysis is and section 244 permits "copying or communicating for computational data analysis" {cite}`singapore2021copyright`.

Copyright is not the only legal route. A site's terms of service raise a contract question, and in the United States anti-hacking law adds another: the Computer Fraud and Abuse Act forbids accessing a protected computer "without authorization". In *hiQ Labs v. LinkedIn* the Ninth Circuit, reconsidering the case after the Supreme Court's *Van Buren* decision, held that hiQ had raised serious questions whether access can be "without authorization" under the CFAA "where, as here, prior authorization is not generally required, but a particular person—or bot—is refused access", that is, to pages anyone can view, while noting that trespass to chattels and breach of contract claims may still be available {cite}`hiq2022linkedin`. Scraping public pages is therefore not thereby lawful: the risk moves from copyright to contract and tort, which is why the terms-of-service restrictions measured in {ref}`sec-ch3-robots-txt-opt-outs-commons` matter even where robots.txt admits a crawler.

The practical consequences for acquisition: keep the record of where each document came from and under what terms ({ref}`sec-ch3-provenance-documentation`); prefer sources that are public domain, openly licensed, or licensed to you; treat shadow libraries and scraped paywalled content as liabilities, whatever the fair-use argument; and build the machinery to honor opt-outs, because a document that was lawful to collect may not be lawful to keep.

(sec-ch3-dataset-licenses-not-content-licenses)=

### 5.2 Dataset licenses are not content licenses

A dataset on a hub carries a license (ODC-BY, CC BY, Apache 2.0, "other"), and it is tempting to read that license as permission to train on the contents. It is not. The dataset license covers the compilation and the curation work; the documents inside keep their own rights. FineWeb and Dolma are both released under ODC-BY with an explicit note that Common Crawl's terms of use apply to the underlying pages. The Data Provenance Initiative {cite}`longpre2023data` audited more than 1,800 widely used fine-tuning datasets and found that more than 70% of the licenses were omitted on the hosting platforms and more than half of those stated were wrong, frequently in the permissive direction, with the original sources often restricting commercial use where the hub listing did not. The lesson is to trace every dataset to its sources and read the sources' terms, which is tedious and is the job.

(sec-ch3-robots-txt-opt-outs-commons)=

### 5.3 robots.txt, opt-outs and the shrinking commons

Consent in Crisis {cite}`longpre2024consent` measured, across C4, RefinedWeb and Dolma, how the web's permissions for AI crawlers changed between 2023 and 2024: within a year, robots.txt restrictions had put off limits more than 5% of all of C4's tokens and more than 28% of the tokens from its most actively maintained, highest-quality domains (news organizations above all), while, measured separately, terms of service restricting crawling covered "a full 45% of C4" (the paper reports the two measures on their own, not as shares that add), and the restrictions were rising fastest exactly where the best text was. The restrictions are also inconsistent: many sites block OpenAI's crawler but not others, many block `CCBot` without realizing that this also removes them from the open corpora that research depends on, and many state a prohibition in their terms of service that their robots.txt does not express.

{numref}`fig-ch3-consent-restrictions` shows how abrupt the change was. From 2016, where the study's series begins, to mid-2023 the restricted share sat at a few percent; between August 2023 and April 2024 it rose to 20 to 33% of the head domains' tokens, the range the study states (the endpoints read off the plotted lines are a point lower), against 5 to 7% of each corpus as a whole.

:::{figure} figures/consent-restrictions.*
:name: fig-ch3-consent-restrictions
:alt: Line plot from January 2016 to April 2024 of the percentage of tokens fully restricted by robots.txt. Three dashed red lines for head domains stay between 2 and 5 percent until August 2023, then rise steeply to between 19 and 32 percent by April 2024. Three solid blue lines for the full corpora stay between about 1 and 4.5 percent until August 2023, highest in 2016 and 2017, and rise to between 5 and 7 percent.

Share of tokens fully restricted to AI crawlers by robots.txt, by domain tier, monthly
from January 2016 to April 2024; data read from Fig. 2 of {cite:t}`longpre2024consent`.
Each tier has three lines, one for each of the three corpora the paper studied (C4,
RefinedWeb and Dolma); the source figure does not label which line is which corpus, so
they are drawn alike. Head domains are the most-used, most actively maintained sources
in each corpus.
:::

Opt-out mechanisms beyond robots.txt:

- **Per-dataset opt-out forms**, as The Stack provides for GitHub repositories, honored in the next release.
- **Retroactive removal**: FineWeb and RefinedWeb state that they will remove pages whose owners ask, and Common Crawl removes content on request from future snapshots.
- **Reservations in the response or the page.** The TDM Reservation Protocol (TDMRep) of a W3C community group {cite}`w3c2024tdmrep` expresses the EU text-and-data-mining opt-out as two properties, `tdm-reservation` (1 means the rights are reserved) and `tdm-policy` (where to learn how to obtain a license), declared in a `tdmrep.json` file under `/.well-known`, in an HTTP response header or in HTML metadata. DeviantArt introduced `noai` and `noimageai` directives in November 2022, placed in an HTML tag on the page and sent in an HTTP header when an image is downloaded {cite}`deviantart2022noai`. Search engines read directives of the same kind from the robots meta tag and from its HTTP-header form, `X-Robots-Tag` {cite}`google2026robotsmeta`. A crawler that checks only robots.txt sees none of these; honoring them means parsing the headers and HTML it has already fetched.

A corpus built today should record, per document, the robots.txt state at fetch time and should be rebuilt against current opt-outs before each new training run. The cost of doing so is small; the cost of not doing so is a model that cannot be shipped.

(sec-ch3-open-corpora-decided)=

### 5.4 What the open corpora decided

| Corpus                                  | Content license position                                        | Opt-out                                   | Known problems                                          |
| --------------------------------------- | --------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| C4 {cite}`raffel2019exploring`          | Common Crawl terms; first released as a script that regenerates it from Common Crawl, hosted as text from 2021 | None | Removed benign minority text with its word list {cite}`dodge2021documenting` |
| The Pile {cite}`gao2020pile`            | Mixed; included Books3                                          | None                                      | Books3 withdrawn after takedown; litigation            |
| The Stack {cite}`kocetkov2022stack`     | Permissive licenses only, detected from license files           | Form per repository                       | License detection errors; copyleft-adjacent files       |
| Dolma {cite}`soldaini2024dolma`         | ODC-BY on the compilation; sources' terms apply                 | Via Common Crawl and source hubs          | Reddit portion predates API closure                     |
| FineWeb {cite}`penedo2024fineweb`       | ODC-BY on the compilation; Common Crawl terms apply             | Email removal requests honored           | Web only, so inherits every robots.txt gap              |
| RedPajama-V2 {cite}`weber2024redpajama` | Released with quality signals so users apply their own filters  | Via Common Crawl                          | 100T raw tokens, deliberately unfiltered                |

(sec-ch3-personal-data-sensitive-sources)=

### 5.5 Personal data and sensitive sources

Data-protection law (the GDPR and its relatives) applies to personal data regardless of copyright, which is why acquisition should avoid sources that are mostly personal data (people directories, leaked databases, social profiles) rather than relying on the redaction in {ref}`Chapter 4 <sec-ch4-chapter-overview>` to fix them later. Health records, children's content and content from regions with strict localization rules are usually excluded at the source level for the same reason.

In the European Union, processing personal data needs one of the lawful bases in Article 6 of the GDPR {cite}`gdpr2016`. The basis examined for model development is Article 6(1)(f), processing "necessary for the purposes of the legitimate interests pursued by the controller or by a third party, except where such interests are overridden by the interests or fundamental rights and freedoms of the data subject". The European Data Protection Board's Opinion 28/2024, adopted on 17 December 2024, considers "how controllers can demonstrate the appropriateness of legitimate interest as a legal basis" in the development and deployment of AI models. It recalls that "there is no hierarchy between the legal bases provided by the GDPR" {cite}`edpb2024opinion`, and it sets out the three-step test that "should be conducted when assessing the use of legitimate interest as a legal basis" {cite}`edpb2024opinion`: identify the legitimate interest pursued by the controller or a third party; show that the processing is necessary for that interest (the "necessity test"); and assess that the interest is not overridden by the interests or fundamental rights and freedoms of the data subjects (the "balancing test"). It also lists mitigating measures for the development phase, including for web scraping, while leaving their adequacy to the supervisory authorities case by case. Article 17 gives a person the right to obtain the erasure of their personal data "without undue delay", and Article 12(3) binds the controller to act on such a request "without undue delay and in any event within one month of receipt of the request", extendable by two further months for complex or numerous requests {cite}`gdpr2016`.

Erasure is where acquisition decisions come back. A request names a person, an address or a page; answering it means finding every document that matches, removing those documents from the shards, and making sure that the next rebuild does not bring them back. The provenance manifest of {ref}`sec-ch3-record-contents` turns that into a query:

```python
import hashlib
import re

# A manifest as in the provenance section: one record per document, with the
# shard it was written to.
docs = {
    "https://example.com/team":
        b"Contact Jane Roe at jane.roe@example.com or on extension 4417.",
    "https://example.com/blog/1": b"Jane Roe wrote about tokenizers.",
    "https://example.org/news": b"The council met on Tuesday.",
    "https://example.org/people/jroe":
        b"J. Roe (jane.roe@example.com) lives on Elm Street.",
}
manifest = [{"locator": u, "sha256": hashlib.sha256(b).hexdigest(), "shard": i % 2}
            for i, (u, b) in enumerate(docs.items())]

# An erasure request names the person and their address; the matching runs over
# the raw text.
request = re.compile(r"jane\.roe@example\.com|\bjane roe\b", re.IGNORECASE)
hits = [r for r in manifest if request.search(docs[r["locator"]].decode())]
blocklist = {r["sha256"] for r in hits}   # so the next rebuild drops the same bytes
print(len(hits), "of", len(manifest), "documents match; shards to rewrite:",
      sorted({r["shard"] for r in hits}))

# At corpus scale: k matching documents spread uniformly over S shards touch
# S(1 - (1 - 1/S)^k) shards.
S, k = 10_000, 200
print(f"{S * (1 - (1 - 1 / S) ** k):.1f} of {S:,} shards rewritten for {k} documents")
```

% output

```text
3 of 4 documents match; shards to rewrite: [0, 1]
198.0 of 10,000 shards rewritten for 200 documents
```

The fourth document is found only through the email address, which is why a request is matched on every identifier it gives. Removing documents is cheap; rewriting the shards they sit in is not. With documents spread uniformly over $n_{shard}$ shards (`S` in the code), $n_{match}$ matching documents (`k`) touch about $n_{shard}\left(1 - (1 - 1/n_{shard})^{n_{match}}\right)$ of them, nearly one shard per document while $n_{match}$ is small compared with $n_{shard}$, so 200 matches rewrite about 198 of 10,000 shards. The hash blocklist lets the next rebuild from raw data drop the same bytes without repeating the search. A model already trained on the removed text is a separate problem that the corpus cannot fix.

(sec-ch3-infrastructure-replacing-robots-txt)=

### 5.6 The infrastructure that is replacing robots.txt

Three developments in 2025 are turning the informal conventions above into machinery, and a corpus built in 2026 should expect to interact with all of them:

- **Default blocking and paid crawling.** Cloudflare, which fronts a large share of the web, announced on 1 July 2025 that it was "changing the default to block AI crawlers unless they pay creators for their content" {cite}`cloudflare2025content`. The IETF's AI Preferences working group is chartered to standardize a "vocabulary for expressing AI-related preferences" and the means of attaching those preferences to content in IETF protocols and formats {cite}`ietf2025aipref`.
- **Regulatory codes.** The EU's General-Purpose AI Code of Practice is "a voluntary tool" whose chapters on transparency and copyright "offer all providers of general-purpose AI models a way to demonstrate compliance with their obligations under Article 53 AI Act", which include honoring machine-readable opt-outs {cite}`ec2025gpaicode`. The training-content summary is not a Code commitment but a legal obligation under Article 53(1)(d), with a template the Commission published on 24 July 2025 {cite}`ec2025template`. Together they make the opt-out handling of {ref}`sec-ch3-robots-txt-opt-outs-commons` a compliance requirement rather than a courtesy.
- **Openly licensed corpora.** The Common Pile {cite}`kandpal2025common` (8 TB of public-domain and openly licensed text, with a model trained on it to show the approach is viable), Common Corpus and institutional collections of digitized public-domain books show what a corpus looks like when the advice of {ref}`sec-ch3-rights-licenses-consent` is followed to the letter: smaller, older on average, and free of the exposure that the web corpora carry.

One acquisition-time risk is specific to how corpora are distributed. Many image-text datasets are published as URL lists rather than content, and {cite:t}`carlini2024poisoning` showed two attacks. Buying expired domains on those lists is enough to insert chosen content into the copy of anyone who later re-downloads the dataset. And timing edits to land just before a Wikipedia snapshot is taken poisons the snapshot. Content hashes recorded at collection time ({ref}`sec-ch3-record-contents`) defend against the first attack, because a re-downloaded page whose bytes changed no longer matches its hash. They do not defend against the second, because the poisoned text is what was collected and the hash records it faithfully. For that one the authors propose timing defenses: "randomizing the order in which data is snapshotted", or delaying content before it enters a snapshot so that trusted moderators can revert malicious edits first.

(sec-ch3-multilingual-low-resource-sources)=

## 6. Multilingual and Low-Resource Sources

(sec-ch3-english-skew)=

### 6.1 The English skew

Roughly 42% of Common Crawl's pages are English (41.9% in the snapshot used in this chapter {cite}`commoncrawl2026stats`), which is already a large over-representation relative to the world's speakers, and every filtering stage makes the skew worse because the filters were tuned on English. A model trained on an English-dominant corpus can still speak other languages, but it does so with less data, fewer facts and a tokenizer that spends several tokens per word, which raises its cost and lowers its quality for most of the world's people.

(sec-ch3-multilingual-corpora)=

### 6.2 The multilingual corpora

```{tabularcolumns} \Y{0.23}\Y{0.16}\Y{0.19}\Y{0.42}
```

| Corpus                                         | Languages | Source               | Approach                                                                                 |
| ---------------------------------------------- | --------- | -------------------- | ---------------------------------------------------------------------------------------- |
| mC4 {cite}`xue2021mt5`                         | 101       | 71 Common Crawl WET  | cld3 language id, per-language C4 rules; known to contain mislabeled and machine-translated text |
| OSCAR {cite}`abadji2022oscar`                  | 150+      | Common Crawl WET     | fastText language id per line, then document level; releases new versions per snapshot  |
| ROOTS {cite}`laurenccon2022bigscience`         | 46 + 13 programming | Curated datasets + crawl, chosen by language communities | Community-sourced catalog of sources; per-language cleaning parameters |
| CulturaX {cite}`nguyen2023culturax`            | 167       | mC4 + OSCAR          | Merge, re-identify language, URL blocklist, per-language perplexity thresholds, MinHash |
| MADLAD-400 {cite}`kudugunta2023madlad`         | 419       | Common Crawl         | Manual audit of every language by inspection; drops the labels that were not real languages |
| FineWeb-2 {cite}`penedo2025fineweb2`           | 1,000+    | 96 Common Crawl snapshots | Per-language thresholds derived from reference-corpus statistics; duplicates "rehydrated" with cluster-size-dependent weights (1–10) derived from filter rates |

The recurring finding, first established by the audit by {cite:t}`kreutzer2022quality` across mC4, OSCAR and other crawled corpora and repeated in the MADLAD-400 audit, is that automatic language identification produces hundreds of "languages" that on inspection are noise, boilerplate, mislabeled majority-language text or machine translation. Below the top hundred or so languages, a human who reads the language has to look at the data. FineWeb-2's rehydration step adds a nuance: after global per-language deduplication it records each kept document's cluster size and up-samples by a weight between 1 and 10 chosen from how often the quality filters removed documents of that cluster size. Moderately duplicated pages get the highest weights; unique pages and the most-copied pages (often boilerplate) get the lowest.

{numref}`fig-ch3-fineweb2-rehydration` shows the U shape for French. The filters removed 70.4% of unique documents and 66.2% of the most-copied group, both above the global rate of 62.4%, but only 44.6% of documents with 19 copies; the sizes around that minimum get the weight 10, and the sizes whose removal rate is at or near the global rate get 1.

:::{figure} figures/fineweb2-rehydration.*
:name: fig-ch3-fineweb2-rehydration
:alt: Bar chart of the percentage of documents removed by filters against MinHash cluster size from 1 to 218, bars colored by upsampling weight from 1 to 10. Removal is 70.4 percent at size 1, falls to a minimum of about 45 percent near sizes 17 to 28 (weight 10), then rises slowly past 60 percent above size 175 (weight 1 to 2). A dashed line marks the global removal rate of 62.4 percent.

Filter removal rate by duplicate-cluster size for French in FineWeb-2, and the
upsampling weight each size receives; data read from Fig. 2 of
{cite:t}`penedo2025fineweb2`. The weight is 10 at the size with the lowest removal rate,
1 above the global removal rate, and interpolated between. The last bar groups the
most-copied 0.1% of documents.
:::

(sec-ch3-beyond-crawl)=

### 6.3 Beyond the crawl

For languages the web under-serves, acquisition reaches for sources that the web pipeline skips: religious texts and their translations (the Bible and Quran exist in more languages than any other document), government publications and legal codes in official languages, broadcaster transcripts, Wikipedia editions, OPUS and other parallel corpora assembled for machine translation, and community collections such as those gathered for ROOTS and by Masakhane for African languages. These are small, but a few million tokens of genuine text in a language is worth more than a billion tokens of mislabeled crawl.

How far such a source goes is set by the epoch cap of {ref}`sec-ch3-epoch-cap-per-source`: a source can supply at most four times its unique tokens before it passes four epochs. In a trillion-token run:

```python
budget = 1e12        # tokens in the training run
max_epochs = 4
for unique in [5e6, 50e6, 500e6]:
    share = max_epochs * unique / budget
    print(f"{unique / 1e6:5.0f}M unique tokens -> at most {share * 100:.1g}%",
          "of a 1T-token run")
needed = 0.01 * budget / max_epochs
print(f"for a 1% share at {max_epochs} epochs: {needed / 1e9:.1f}B unique tokens")
```

% output

```text
    5M unique tokens -> at most 0.002% of a 1T-token run
   50M unique tokens -> at most 0.02% of a 1T-token run
  500M unique tokens -> at most 0.2% of a 1T-token run
for a 1% share at 4 epochs: 2.5B unique tokens
```

Even half a billion tokens of curated text is a fifth of a percent of such a run, so a low-resource language gets most of its share of a multilingual model from the crawl, and the curated sources serve another purpose: they are the clean reference text from which language-identification models and per-language filter thresholds (as in FineWeb-2) are built and checked. Two cautions apply to all of them. Many modern translations of religious texts, and the publications of many governments, are under copyright or restrictive terms although they are free to read, so the license check of {ref}`sec-ch3-dataset-licenses-not-content-licenses` applies; and a parallel corpus built from translations can contain machine-translated text, the same contamination the audits found in the crawl.

(sec-ch3-synthetic-data)=

## 7. Synthetic Data

(sec-ch3-generating-text)=

### 7.1 Why generate text

If the stock of human-written text is the binding constraint, text generated by a model is the obvious supplement, and it is now a standard component rather than an experiment:

- **Instruction data**: Self-Instruct {cite}`wang2023selfinstruct` showed that a model can generate the instruction-response pairs used to fine-tune it, bootstrapping from 175 human-written seed tasks; this is the basis of most instruction datasets ({ref}`roadmap-post-training-and-alignment`).
- **Textbook-style pretraining data**: phi-1 {cite}`gunasekar2023textbooks`, a 1.3B-parameter code model, trained on under a billion tokens of synthetic textbook-style text plus about six billion tokens of classifier-filtered code and matched much larger models on the HumanEval and MBPP coding benchmarks. Cosmopedia {cite}`benallal2024cosmopedia` reproduced the idea openly, generating 25 billion tokens of textbooks, blog posts and stories from seed topics with an open model.
- **Rephrased web**: rewriting crawled pages in a cleaner style with a model, then training on the rewritten version alongside the original, gives a cleaner corpus of the same size; WRAP {cite}`maini2024rephrasing` paraphrased C4 into Wikipedia-like and question-answer styles and reached the same perplexity about three times faster.
- **Translation and augmentation** for low-resource languages, where a model translates high-resource text into a target language.

(sec-ch3-failure-modes)=

### 7.2 What can go wrong

Synthetic data inherits the generating model's errors, style and gaps, and feeding it back repeatedly compounds them. {cite:t}`shumailov2024curse` showed that models trained on the outputs of earlier models lose the tails of the distribution first, so that after a few generations rare content disappears and the model's outputs converge on a narrow, repetitive mode. The practical mitigations are to cap the synthetic share (a share that production mixtures set differently; see {ref}`sec-ch3-epoch-cap-per-source`), to keep human-written data in every generation, to anchor generation in real documents (rephrasing, grounded question writing) rather than free generation, and to mark synthetic data as such in provenance records. The last matters beyond your own pipeline: model-generated text is published on the open web too, so future crawls contain it whether or not it was labeled.

A toy version of the experiment shows the mechanism. Let training be the estimation of word frequencies from a sample and generation be sampling from the estimate. A word type that a finite sample happens to miss gets probability zero and can never return, so with each generation of purely synthetic data the rare types disappear; replacing half of each generation's data with fresh human text keeps most of them:

```python
import random

rng = random.Random(0)
V, N, GENERATIONS = 1000, 10_000, 10
human = [1 / r for r in range(1, V + 1)]   # Zipf-shaped frequencies over V word types


def train(sample):
    """'Training' here is estimating unigram frequencies; a type never sampled
    gets zero."""
    counts = [0] * V
    for t in sample:
        counts[t] += 1
    return counts


synthetic = mixed = human
print("generation  types left (all synthetic)  types left (half human)")
for g in range(1, GENERATIONS + 1):
    synthetic = train(rng.choices(range(V), weights=synthetic, k=N))
    mixed = train(rng.choices(range(V), weights=mixed, k=N // 2)
                  + rng.choices(range(V), weights=human, k=N // 2))
    if g in (1, 2, 5, 10):
        alive_s, alive_m = sum(c > 0 for c in synthetic), sum(c > 0 for c in mixed)
        print(f"{g:10}  {alive_s:25}  {alive_m:23}")
# the head survives in both; the tail survives only with human data
assert all(c > 0 for c in synthetic[:50]) and all(c > 0 for c in mixed[:50])
assert sum(c > 0 for c in synthetic) < sum(c > 0 for c in mixed)
```

% output

```text
generation  types left (all synthetic)  types left (half human)
         1                        901                      912
         2                        799                      880
         5                        620                      882
        10                        478                      898
```

After ten generations the all-synthetic chain keeps 478 of the 1,000 types and the half-human chain 898, and every one of the 50 most frequent types survives in both: the loss is in the tail, as {cite:t}`shumailov2024curse` report. The half-and-half ratio is not a recommendation (the toy says nothing about how much synthetic data helps a real model), but it shows why keeping human data in every generation is the mitigation that works.

(sec-ch3-mixture-design-token-budgeting)=

## 8. Mixture Design and Token Budgeting

(sec-ch3-compute-token-conversion)=

### 8.1 From compute to tokens

{ref}`Chapter 2 <sec-ch2-chapter-overview>`'s relation $C \approx 6ND$ fixes the token count once the budget and the model size are chosen. For an 8B-parameter model and $10^{24}$ FLOPs that is about 20.8T tokens; for the Chinchilla optimum at the same budget, $N \approx \sqrt{C/120} \approx 91$B and $D \approx 1.8$T. Production practice sits between: a model sized for inference cost, trained on as many tokens as the data allows.

(sec-ch3-epoch-cap-per-source)=

### 8.2 Epochs and unique tokens

Each source $s$ has $U_s$ unique tokens after cleaning. If the mixture assigns it a weight $w_s$ (its share of training tokens) and the run sees $D$ tokens in total, the source is repeated $e_s = w_s D / U_s$ times. By the data-constrained scaling result of {ref}`sec-ch3-epochs`, $e_s$ up to about 4 is nearly free, so the design constraint is

$$e_s = \frac{w_s D}{U_s} \le e_{\max} \approx 4 \quad \text{for every source } s, \qquad \sum_s w_s = 1$$

Small, high-quality sources (Wikipedia, textbooks, papers) hit the cap first and are the reason mixtures cannot simply weight sources by preference.

The cap is a guideline from controlled experiments, and published mixtures do exceed it. The phi-4 report {cite}`abdin2024phi4` lists, for each source of its pretraining mixture, the "fraction of training", the "unique token count" and the "number of epochs", which are $w_s$, $U_s$ and $e_s$ of this section: web 15%, 1.3T tokens, 1.2 epochs; web rewrites 15%, 290B, 5.2; synthetic 40%, 290B, 13.8; code 20%, 820B, 2.4; acquired sources 10%, 580B, 1.7 (its Table 5). It states that "the total number of epochs on each data source is determined using the ratio of allocated tokens in the mixture and the number of unique tokens in that source", the formula above. Synthetic text is the largest component, and the authors report that "prioritizing more epochs over our synthetic data led to better performance with respect to adding fresh web tokens", while models trained only on synthetic data "underperformed on the knowledge-heavy benchmarks". The listing sets the stated epochs against the decay law of {ref}`sec-ch3-epochs`:

```python
import math

R_STAR = 15.387756  # decay constant of the data-constrained scaling law


def effective(epochs):
    """Fresh-token equivalents of one unique token seen for `epochs` epochs."""
    return 1 + R_STAR * (1 - math.exp(-(epochs - 1) / R_STAR))


# phi-4 pretraining mixture (Table 5): share of training tokens, unique tokens
# in billions, and the epochs the report states.
PHI4 = {"web": (0.15, 1300, 1.2), "web rewrites": (0.15, 290, 5.2),
        "synthetic": (0.40, 290, 13.8), "code": (0.20, 820, 2.4),
        "acquired": (0.10, 580, 1.7)}
D = 10_000  # billions of training tokens, implied by e = w * D / U

for name, (w, u, e) in PHI4.items():
    assert abs(w * D / u - e) / e < 0.06  # the table's epochs follow from w, U, D
    print(f"{name:>12}: {e:4.1f} epochs, {effective(e):5.2f} fresh equivalents,",
          f"{effective(e) / e:.0%} per pass")
assert 9.6 < effective(13.8) < 9.8
```

% output

```text
         web:  1.2 epochs,  1.20 fresh equivalents, 100% per pass
web rewrites:  5.2 epochs,  4.68 fresh equivalents, 90% per pass
   synthetic: 13.8 epochs,  9.69 fresh equivalents, 70% per pass
        code:  2.4 epochs,  2.34 fresh equivalents, 97% per pass
    acquired:  1.7 epochs,  1.68 fresh equivalents, 99% per pass
```

The five epoch counts follow from the shares and a run of about 10T tokens, as the formula says. The decay law values the 13.8 passes over the synthetic data at 9.7 fresh passes: a loss, but a smaller one than the four-epoch rule would suggest, and a trade the authors made on purpose because each extra synthetic pass was worth more to them than a fresh web token. The law was fitted on natural text, and the report does not cite it, so the listing shows what the law would predict and not what the authors measured. For a mixture designer the lesson is that $e_{\max} \approx 4$ is a default for natural text, to be revisited source by source, and that the share of synthetic data, like the epoch cap, is a decision tested by ablation in the report and not a constant of nature.

(sec-ch3-setting-weights)=

### 8.3 Setting the weights

Three ways the weights get set:

- **By hand, from experience.** LLaMA's mixture (67% crawl, 15% C4, 4.5% each of GitHub, Wikipedia and books, 2.5% arXiv, 2% StackExchange, with Wikipedia and books at about two epochs) {cite}`touvron2023llama` is the template most open models started from.
- **By optimization.** DoReMi {cite}`xie2023doremi` trains a small proxy model with weights adjusted to minimize the worst-case excess loss across domains against a reference, then trains the large model with the proxy run's weights averaged over its steps; on The Pile it improved average downstream accuracy at the same token budget and reached the baseline's downstream accuracy 2.6 times faster. During the proxy run each domain has a weight, and every step multiplies it by the exponential of the domain's excess loss times a step size. The excess loss is the per-token loss of the proxy minus that of the reference, floored at zero, averaged over the domain's tokens in the batch. The weights are then renormalized and mixed with the uniform distribution with a small weight $c$ (the paper uses $c = 10^{-3}$), which keeps every domain's weight above zero. Domains that the proxy still learns slowly relative to the reference gain weight, which is the worst-case (minimax) objective carried out by exponentiated-gradient ascent on the weights. The update is written out below, with the averaging that the paper's Algorithm 1 returns: "DoReMi takes the averaged domain weights over DRO training steps" {cite}`xie2023doremi`.
- **By ablation.** The FineWeb {cite}`penedo2024fineweb` and DataComp-LM {cite}`li2024datacomp` protocol, training small models on candidate mixtures and comparing benchmarks, is the most expensive of the three, since every candidate costs a training run, and it is the one most teams use for the final decision.

With $\Delta_t[i]$ the excess loss of domain $i$ at step $t$, $\nu$ the step size, $c$ the smoothing weight and $u$ the uniform distribution over the $k$ domains, one step and the returned weights are

$$q_t' = q_{t-1} \odot \exp(\nu \Delta_t), \qquad q_t = (1 - c)\,\frac{q_t'}{\sum_{i=1}^k q_t'[i]} + c\,u, \qquad \bar{q} = \frac{1}{T} \sum_{t=1}^{T} q_t,$$

where $\odot$ is the entrywise product and $\bar{q}$ is what the large run uses. The weights at the last step are one noisy sample of a process that keeps moving, because each batch's excess loss is noisy; the average smooths that out. The toy below has two domains, one the proxy lags on by 0.5 nats per token and one by 0.1, with a gap that shrinks as the domain's weight grows and is multiplied by noise in each batch. It is not the paper's setup, only the update rule applied to a made-up loss:

```python
import math
import random

rng = random.Random(0)
NU, C = 4.0, 1e-3  # step size; smoothing with the uniform distribution
GAP = [0.5, 0.1]    # how far the proxy lags the reference on each domain
alpha = [0.5, 0.5]
history = []
for step in range(200):
    # the excess loss shrinks as a domain's weight grows, and each batch is noisy
    excess = [max(0.0, g * (1 - 1.6 * a) * rng.uniform(0.0, 2.0))
              for g, a in zip(GAP, alpha)]
    raw = [a * math.exp(NU * x) for a, x in zip(alpha, excess)]
    alpha = [(1 - C) * r / sum(raw) + C / 2 for r in raw]
    history.append(alpha)
mean = [sum(h[i] for h in history) / len(history) for i in range(2)]
print("final step weights:", [round(a, 3) for a in history[-1]])
print("averaged weights:  ", [round(a, 3) for a in mean])
assert abs(sum(mean) - 1) < 1e-9 and mean[0] > mean[1]
assert min(min(h) for h in history) >= C / 2
```

% output

```text
final step weights: [0.563, 0.437]
averaged weights:   [0.586, 0.414]
```

The domain the proxy lags on more gains weight, and the two ways of reading the weights differ by two points of share in the first domain; the paper uses the average.

A fourth, increasingly common choice is **curriculum**: training on the full mixture for most of the run and then "annealing" on a small, high-quality subset (FineWeb-Edu, textbooks, instruction-like data) in the final stretch of the run. The Llama 3 report {cite}`grattafiori2024llama` linearly anneals the learning rate to 0 over the final 40M tokens of pre-training, and runs its annealing experiments on a 50% trained 8B model with 40B tokens, 30% of them from the dataset under test.

(sec-ch3-budgeting-procedure)=

### 8.4 A budgeting procedure

The function below turns a token budget, a table of sources with unique-token counts and preferred weights, and an epoch cap into a feasible mixture. When a preferred weight would exceed the cap, the source is clipped at the cap and the surplus is redistributed proportionally to the sources that still have room.

```python
import math


def plan_mixture(total_tokens, sources, max_epochs=4.0):
    """Return {source: (weight, tokens, epochs)} obeying sum(weight) == 1 and
    epochs <= max_epochs.

    sources: {name: (unique_tokens, preferred_weight)}
    """
    caps = {s: min(1.0, max_epochs * u / total_tokens) for s, (u, _) in sources.items()}
    if sum(caps.values()) < 1.0:
        raise ValueError("not enough unique data for this budget at this epoch cap")
    weights = {s: w for s, (_, w) in sources.items() if w > 0}
    if sum(caps[s] for s in weights) < 1.0:
        raise ValueError(
            "the sources with a positive preferred weight cannot absorb the budget")
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
        s: (weights.get(s, 0.0), weights.get(s, 0.0) * total_tokens,
            weights.get(s, 0.0) * total_tokens / sources[s][0])
        for s in sources
    }


sources = {               # unique tokens, preferred weight
    "web":        (12e12, 0.82),
    "code":       (1.0e12, 0.08),
    "papers":     (0.15e12, 0.04),
    "wikipedia":  (0.0065e12, 0.03),
    "books":      (0.03e12, 0.03),
}
plan = plan_mixture(total_tokens=2e12, sources=sources)
assert abs(sum(w for w, _, _ in plan.values()) - 1) < 1e-9
assert all(e <= 4.0 + 1e-9 for _, _, e in plan.values())
assert math.isclose(plan["wikipedia"][2], 4.0)   # smallest source pinned at the cap
assert plan["web"][0] > 0.82              # its surplus went to the sources with room
for name, (w, t, e) in plan.items():
    print(f"{name:10} weight {w:6.3f}  tokens {t/1e12:5.2f}T  epochs {e:4.2f}")
```

% output

```text
web        weight  0.834  tokens  1.67T  epochs 0.14
code       weight  0.081  tokens  0.16T  epochs 0.16
papers     weight  0.041  tokens  0.08T  epochs 0.54
wikipedia  weight  0.013  tokens  0.03T  epochs 4.00
books      weight  0.031  tokens  0.06T  epochs 2.04
```

The procedure says nothing about whether the preferred weights are good; it only keeps them feasible. Choosing them is the work of {ref}`sec-ch3-setting-weights`, and documenting the choice is the subject of {ref}`sec-ch3-provenance-documentation`.

```{admonition} Planning a Mixture
:class: note
The notebook section {ref}`nb-acq-mixture` starts from a compute budget, derives the
token count with the relation of {ref}`sec-ch2-flops`, runs `plan_mixture` over a
realistic table of sources, plots the planned epochs per source against the epoch cap,
and shows the epochs the same plan implies when the web slices lose the share the
consent study measured.
```

{numref}`fig-ch3-epochs-per-source` is the notebook's plot. Its run is larger than the 2T-token example above: a budget of $C = 10^{24}$ FLOPs for an 8B-parameter model gives $D = C / 6N \approx 20.8$T tokens, so four small sources reach the cap of $e_{\max} = 4$. The opt-outs leave the plan unchanged: the same token counts are drawn from fewer unique tokens, so the two web slices are repeated more.

:::{figure} figures/epochs-per-source.*
:name: fig-ch3-epochs-per-source
:alt: Grouped bar chart of epochs per source for a 21T-token run, planned and after opt-outs. Web rises slightly, from about 1.17 to 1.23 epochs, quality web from 1.5 to 2.1, code stays at 1.8, and papers, wikipedia, books and dialogue all sit at the 4-epoch cap in both plans. A dashed line marks the cap at 4.

Epochs $e_s$ per source from the notebook's `plan_mixture` run ($D \approx 20.8$T
tokens, cap $e_{\max} = 4$), before and after removing 5% of the web and 28% of the
quality web, the losses measured by {cite:t}`longpre2024consent`. The weights $w_s$ are the same in both plans; only $U_s$ shrinks, for the web and quality web slices, so $e_s = w_s D / U_s$ rises by a factor $1/0.95$ and $1/0.72$ for those two and for no other source.
:::

(sec-ch3-provenance-documentation)=

## 9. Provenance and Documentation

(sec-ch3-record-contents)=

### 9.1 What to record

Every document that enters the corpus should carry, as metadata that travels with it through {ref}`Chapter 4 <sec-ch4-chapter-overview>`'s pipeline and into {ref}`Chapter 5 <sec-ch5-chapter-overview>`'s versioned dataset:

- **Source and locator**: the corpus or site, the URL or file path, and for Common Crawl the snapshot, WARC file and byte offset, which together are enough to re-fetch the exact bytes.
- **Fetch time and crawl time**, because the page may have changed, been removed or opted out since.
- **License or terms** as understood at acquisition, and the robots.txt decision for the crawler that fetched it.
- **A content hash** of the raw bytes, so that the document can be recognized in later versions and so that exact deduplication has something to work with.
- **Synthetic or not**, with the generating model and prompt if synthetic.

```python
import hashlib
import time


def provenance_record(source, locator, raw_bytes, license, robots_decision, fetched_at,
                      synthetic=False, **extra):
    """One manifest line per acquired document; stable field order,
    content-addressed."""
    return {
        "source": source,
        "locator": locator,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
        "fetched_at": fetched_at,      # when the bytes were fetched, not when recorded
        "license": license,
        "robots": robots_decision,
        "synthetic": synthetic,
        **extra,
    }


doc = b"<html><body><p>Example page.</p></body></html>"
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # stamp at fetch time
loc = "crawl-data/CC-MAIN-2026-39/.../00000.warc.gz#offset=0"
rec = provenance_record("commoncrawl", loc, doc, license="CC terms of use",
                        robots_decision="allowed:CCBot", fetched_at=now)
other = provenance_record("commoncrawl", "same locator", doc + b"\n", "CC terms of use",
                          "allowed:CCBot", now)
same = provenance_record("x", "y", doc, "l", "r", "2026-10-20")
assert same["sha256"] == rec["sha256"]          # same bytes, same id
assert other["sha256"] != rec["sha256"]         # one byte changed, different id
print(rec["sha256"][:16], rec["bytes"], "bytes")
print(other["sha256"][:16], other["bytes"], "bytes")
```

% output

```text
a146439479bc0c21 46 bytes
a06b0fa001efbf09 47 bytes
```

(sec-ch3-datasheets-data-cards)=

### 9.2 Datasheets and data cards

Datasheets for Datasets {cite}`gebru2021datasheets` is the standard template: motivation, composition, collection process, preprocessing, uses, distribution and maintenance, each as a list of questions with answers. Dolma's datasheet is a good complete example, and Hugging Face dataset cards are the lightweight version that every hosted dataset should have. The questions that acquisition has to answer, and that are most often left blank, are the collection ones: who collected it, over what period, with what mechanism, under what terms, and what was deliberately excluded.

A documentation practice that goes further is to publish measurements of the corpus itself. What's In My Big Data {cite}`elazar2024whats` ran a fixed battery of analyses (duplication, benchmark contamination, toxic and personal content, domain and language distributions, the most frequent documents) over ten open corpora and found, among other things, that C4 and the Pile contained benchmark test sets and that corpora differ in how concentrated their domains are: the most common domain accounts for less than 0.05% of C4's pages, but for over 5% of mC4-en's and about 6% of LAION-2B-en's. Running the same battery over your own corpus before training is cheaper than discovering the answers from the model.

Two other formats complement the datasheet. Data Statements {cite}`bender2018data`, which their authors propose "as a design solution and professional practice" for natural language processing, record which populations and language varieties a text collection comes from, because, in the authors' words, data "from certain populations" is used "in the development of technology for other populations". Croissant {cite}`akhtar2024croissant` is a machine-readable metadata format that gives a dataset "a shared representation across ML tools, frameworks, and platforms"; repositories that publish it let a dataset be found by search and loaded into common ML frameworks without code written for that dataset. A datasheet answers questions for a human reader; a Croissant file lets a pipeline check the same facts.

(sec-ch3-handing-off)=

### 9.3 Handing off

The output of acquisition is a set of raw shards plus a manifest: one provenance record per document, the robots and license decisions, the source table with unique-token counts, and the mixture plan. {ref}`Chapter 4 <sec-ch4-chapter-overview>` consumes the shards and must preserve the records through every filter; {ref}`Chapter 5 <sec-ch5-chapter-overview>` versions the result so that the model, the corpus and the manifest can be matched up later.

```{admonition} Provenance in Practice
:class: tip
The notebook section {ref}`nb-acq-provenance` writes a provenance record for every fetch
made earlier in the notebook, checks every committed sample against the checksums
recorded when it was collected, and prints the manifest that would accompany the samples
into Chapter 4.
```

(sec-ch3-summary)=

## Summary

- **Composition.** Most published mixtures are mostly web crawl, because nothing else is large enough; The Pile, at 18% web crawl, is the exception; curated sources are a few percent each in most mixtures (The Pile's papers, court opinions and patents, 37%, and Dolma's code, 13%, are the exceptions) and are upweighted. Training sets grew from GPT-3's 300B tokens to 15T in four years and are now within a factor of seven of the lower end (100T tokens) of the estimated effective stock of public text, and a factor of twenty below its central 300T; that stock is already adjusted for quality and for repeated epochs ({ref}`sec-ch3-epochs`).
- **Common Crawl.** A snapshot is 100 segments and 100,000 WARC files, about 90 TB compressed. Each record is its own gzip member, so an index line (file, offset, length) and one range request retrieve a single page; for many URLs the columnar index answers the same question by reading only the matching parts of Parquet files ({ref}`sec-ch3-index-targeted-retrieval`).
- **Crawling.** A crawler obeys robots.txt as RFC 9309 defines it (redirects followed, at least five in a row, 4xx permits, 5xx forbids, longest match wins), spaces its requests per host, takes its seeds from sitemaps, canonicalizes URLs and re-fetches with `ETag`s. Politeness, not bandwidth, sets its throughput ({ref}`sec-ch3-crawler-architecture`).
- **Curated sources.** Wikipedia, books, papers, code, forums and public records each supply something the web lacks, and each comes with its own terms, which for licensed news and for users' conversations are a contract and a consent setting ({ref}`sec-ch3-licensed-news`, {ref}`sec-ch3-first-party-conversations`); the books lawsuits turned on how the copies were obtained ({ref}`sec-ch3-books`).
- **Rights.** A dataset's license covers the compilation, not the documents. Opt-outs grew fastest on the best-maintained sites, and in the EU honoring them is a legal obligation. Personal data needs a lawful basis, and a provenance manifest turns an erasure request into a query over shards ({ref}`sec-ch3-personal-data-sensitive-sources`).
- **Languages and synthetic data.** Language identification produces many false languages, and below the largest languages a human reader has to check the data; curated text in a small language is better used as reference than as volume ({ref}`sec-ch3-beyond-crawl`). Synthetic data fills gaps but loses the tail of the distribution when it replaces human text ({ref}`sec-ch3-failure-modes`).
- **Mixture.** With $e_s = w_s D / U_s \le e_{\max} \approx 4$, the small sources reach the cap first and the web absorbs the rest, and the decay law behind the cap values a fortieth epoch at a small fraction of a fresh pass; a larger budget makes the mixture more web-heavy unless more unique data is acquired ({ref}`sec-ch3-budgeting-procedure`).
- **Provenance.** One record per document (source, locator, fetch time, terms, robots decision, content hash, synthetic or not) is what makes a corpus auditable, rebuildable and correctable ({ref}`sec-ch3-record-contents`).

(sec-ch3-exercises)=

## Exercises

1. **Read a snapshot.** Using the manifest and range-request code of {ref}`sec-ch3-snapshots-segments-files`, estimate the size of a current Common Crawl snapshot from the number of WARC files and the sizes reported by a `HEAD` request on ten of them. Compare with Common Crawl's published figure.

```{admonition} Solution 1
:class: dropdown
A `HEAD` request returns a file's headers without its body, and `Content-Length` is the
file's size in bytes. The manifest of `CC-MAIN-2026-39` lists 100,000 WARC files. Ten of
them, drawn with `random.Random(0).sample(paths, 10)` and measured on 12 November 2026,
were 910,707,757, 966,719,339, 903,957,552, 915,750,155, 911,469,743, 933,101,733,
896,086,136, 935,203,619, 918,424,076 and 939,744,129 bytes long.

- Mean size: 923,116,424 bytes, with a sample standard deviation of 20.8 MB.
- Estimate: $100{,}000 \times 923{,}116{,}424 \approx 9.231 \times 10^{13}$ bytes, which
  is 92.3 TB or 83.96 TiB ($1\ \text{TiB} = 2^{40}$ bytes).
- Uncertainty: the standard error of a ten-file mean is $20.8 / \sqrt{10} \approx 6.6$
  MB, 0.71% of the mean.

Common Crawl's announcement of the crawl gives the WARC files as 82.79 TiB
{cite}`commoncrawl2026september`. The estimate is 1.4% high, about two standard errors,
which a sample of ten can produce; a hundred files would cut the standard error to about
0.2%. Note the units: the announcement's 82.79 TiB is 91.0 TB, the "about 90 TB" of the
table in {ref}`sec-ch3-snapshots-segments-files`. The notebook makes the same estimate from the same ten seeded files, and
rounds it to about 92 TB.
```

2. **Who is blocked.** The notebook commits the two robots.txt files it reads, Wikipedia's and a news site's, under `examples/data/`. Parse both and tabulate which of `*`, `CCBot`, `GPTBot`, `ClaudeBot` and `Google-Extended` may fetch an article. Which site blocks Common Crawl, and did it likely intend to leave the open corpora? How would you extend the table to fifty news sites?

```{admonition} Solution 2
:class: dropdown
Parse each file with `urllib.robotparser`, as in the listing of
{ref}`sec-ch3-robots-txt-politeness-conventions`, and ask `can_fetch(agent, url)` for
the article URL the notebook uses on each site (`/wiki/Language_model` on Wikipedia, a
dated article path on the news site):

| Agent | Wikipedia | News site |
| --- | --- | --- |
| `*` | allowed | allowed |
| `CCBot` | allowed | disallowed |
| `GPTBot` | allowed | disallowed |
| `ClaudeBot` | allowed | disallowed |
| `Google-Extended` | allowed | disallowed |

The news site blocks Common Crawl; Wikipedia does not. How the block is written suggests
the intent. The news file has 69 `User-agent` lines, and 55 groups that consist of
`Disallow: /` alone, each naming one crawler or a few related ones. `CCBot` is one of
them, with the same single line as `GPTBot`, `ClaudeBot` and `Google-Extended`. That is
the form of a list of AI crawlers copied into the file, not a decision about research
corpora, although the effect is the same: the site is absent from every future Common
Crawl snapshot and from every corpus derived from one.

To extend the table to fifty sites, fetch each `https://<host>/robots.txt` with a named
user agent and record the status first, since a 4xx answer means no restrictions and a
5xx answer means everything is disallowed. Mark whether each answer came from a group
that names the agent or from the `*` fallback, because only named answers show intent,
and use an RFC 9309 parser such as `protego` for files with wildcards. Report the list
and the date with the count, because both change it. Terms of service often state
restrictions the robots file does not express
({ref}`sec-ch3-robots-txt-opt-outs-commons`), so the table is a lower bound on what the
sites forbid.
```

3. **Source comparison.** Before adding a sixth source to the notebook's comparison, find out how well its statistics separate the five it has. Using the committed samples under `examples/data/`, compute the median word count of each source and the ratio of medians for every pair. Which pair is closest and which farthest apart, and what kind of sixth source would median length separate clearly?

```{admonition} Solution 3
:class: dropdown
The committed samples store each document's full word count as `full_words`, and give
these medians: C4 188.5 words, Wikipedia 206.5, FineWeb-2 French 266, FineWeb 354.5 and
FineWeb-Edu 387. Of the ten pairs, the closest is FineWeb and FineWeb-Edu, with a ratio
of 1.09, followed by C4 and Wikipedia at 1.10. The farthest is FineWeb-Edu and C4, at
2.05.

So among the five, median length separates at most by a factor of about two, and two
pairs differ by less than 10%, which samples of fifty or a hundred documents cannot
resolve. FineWeb-Edu is told apart from FineWeb by its educational score, not by length.
A sixth source is separated clearly by length only if its documents are of a different
kind: books, which run to tens of thousands of words, or forum posts and question-answer
pairs, which are much shorter than a web page. For a sixth web corpus, use a statistic
tied to how it was built (a quality or language score the dataset ships, or the host
count if the rows carry URLs).
```

4. **License tracing.** For three kinds of hosted dataset, a filtered web corpus (FineWeb), a copy of the Stack Exchange dump, and an instruction dataset generated by a model, trace the license on the hub card to the terms of the original sources, and record every discrepancy in the style of the Data Provenance Initiative audit.

```{admonition} Solution 4
:class: dropdown
For each dataset, record four things in a table: the license on the hub card, the
sources the card or paper names, the terms of each source, and whether the two agree.

- **FineWeb.** The card states ODC-BY. ODC-BY covers the compilation, and the card
  itself says that Common Crawl's terms of use apply to the pages. The pages keep their
  authors' copyright. The entry records ODC-BY for the compilation and the pages' own
  rights for the content, which is not a contradiction but is often misread as one
  ({ref}`sec-ch3-dataset-licenses-not-content-licenses`).
- **A Stack Exchange dataset.** A hub copy of the Stack Exchange dump typically carries
  CC BY-SA, the license of the posts. Since July 2024 the official download requires a
  login and an agreement that the file is for projects that do not train a large
  language model ({ref}`sec-ch3-forums-dialogue-question-answering`). A copy made from a
  later dump therefore carries a restriction the card does not show. Record which dump
  the copy was made from.
- **An instruction dataset generated by a model.** The card's license covers the
  released text; the terms of the model that generated it may restrict using its outputs
  to train competing models. Record the generating model and its terms.

The discrepancies the audit of {cite:t}`longpre2023data` found most often are the ones
to look for: no license on the card, a license more permissive than the sources allow,
and a commercial-use restriction in a source that the card omits.
```

5. **Budget sensitivity.** Using `plan_mixture`, hold the mixture preferences fixed and sweep the total budget from 0.5T to 20T tokens. At what budget does each source hit the epoch cap, and what does the mixture look like at 20T?

```{admonition} Solution 5
:class: dropdown
Use the source table of {ref}`sec-ch3-budgeting-procedure`. A source $s$ reaches the cap
when its weight $w_s$ reaches $e_{\max} U_s / D$. Before any source is clipped the
weights are the preferred ones, so Wikipedia (preferred 0.03, $U_s = 6.5$B) is the first
to reach the cap, at

$$D = \frac{4 \times 6.5\text{B}}{0.03} \approx 0.867\text{T}.$$

From then on Wikipedia contributes a fixed 26B tokens, and the other sources share the
rest in proportion to their preferences. Books (preferred 0.03 of the remaining 0.97,
$U_s = 30$B) reach the cap when $0.03\,(D - 26\text{B}) / 0.97 = 4 \times 30\text{B}$,
that is at $D \approx 3.906$T. Papers (0.04 of the remaining 0.94, $U_s = 150$B) reach
it when $0.04\,(D - 26\text{B} - 120\text{B}) / 0.94 = 600\text{B}$, at
$D \approx 14.25$T. Web and code never reach it in the sweep.

At 20T the plan is:

| Source | Weight | Tokens | Epochs |
| --- | --- | --- | --- |
| web | 0.877 | 17.54T | 1.46 |
| code | 0.086 | 1.71T | 1.71 |
| papers | 0.030 | 0.60T | 4.00 |
| wikipedia | 0.0013 | 0.026T | 4.00 |
| books | 0.006 | 0.12T | 4.00 |

The three small sources are pinned at four epochs and together supply 3.7% of the tokens
instead of the preferred 10%; the web absorbs almost all of the difference. A larger
budget makes the mixture more web-heavy, not more diverse, unless more unique data is
acquired.
```

6. **Low-resource audit.** You plan to take fifty documents labeled as a language you read from mC4 or OSCAR and classify each by hand as genuine text, boilerplate, mislabeled or machine-translated. Write the classification rules you would fix before reading. If 12 of the 50 turn out not to be genuine, what is the 95% interval for the rate, and how many documents would you need to read for an interval of about plus or minus five percentage points? When does a comparison with the MADLAD-400 audit for that language mean something?

```{admonition} Solution 6
:class: dropdown
Draw the documents at random offsets, not from the start of the split, for the reason
the notebook gives for Wikipedia. Fix the rules before reading: a document is genuine if
most of its sentences are fluent text in the labeled language; boilerplate if it is
mostly navigation, lists or templates; mislabeled if most of it is another language;
machine-translated if it shows the signs of translation (calques, untranslated named
entities, sentences that read as a word-by-word rendering of English).

The result is a proportion with a wide interval at this sample size. With 12 of 50 not
genuine, the rate is 24%. The 95% Wilson score interval, which behaves better than the
normal approximation at small samples and rates far from one half, runs from 14.3% to
37.4%, an interval 23 points wide. For a half-width of five points at a rate near 24%,
the normal approximation gives the sample size as
$1.96^2 \times 0.24 \times 0.76 / 0.05^2 \approx 280.3$ documents, so 281. A comparison
with the audit of {cite:t}`kudugunta2023madlad` is meaningful only if its figure for the
language lies outside such an interval. The audits of {cite:t}`kreutzer2022quality` and
MADLAD-400 found the problem concentrated in the smaller languages, so expect a
high-resource language to come out mostly genuine and a low-resource one to show far
more mislabeled text and boilerplate.
```

7. **A datasheet.** Write the collection-process section of a datasheet for the samples the notebook fetched, answering every question in the Datasheets template that applies.

```{admonition} Solution 7
:class: dropdown
The collection-process questions of {cite:t}`gebru2021datasheets`, answered for the
notebook's samples:

- **How was the data acquired?** Directly from public services: the Common Crawl
  manifest and a byte range of one WARC file over HTTPS, two robots.txt files, and rows
  of five datasets from the Hugging Face dataset-viewer rows API. Nothing was inferred
  or derived at collection time.
- **What mechanisms were used?** Python's `urllib` with a named user agent, a fixed seed
  for every random offset, and a fallback to the committed copies under `examples/data/`
  when a source is unreachable. The checksums in `examples/data/checksums.json` validate
  the committed copies.
- **What was the sampling strategy?** The first rows of FineWeb, FineWeb-Edu, C4 and
  FineWeb-2 French, Wikipedia articles at seeded random offsets, and C4 host names from
  seeded random offsets within the rows the API serves. The samples are small, are not
  representative of any corpus, and are meant to illustrate.
- **Who collected the data, and over what time frame?** The person who ran the notebook,
  on the dates in the manifest's `fetched_at` fields: `fetch` stamps each live request
  with the time it was made, and a record served from a committed sample carries the
  date that sample was collected (the `collected` table of `checksums.json`).
- **Were the people the data is about notified or asked for consent?** No. The documents
  are public web pages and dataset rows. Their consent is expressed only through
  robots.txt and the hosting datasets' terms, which the manifest records.
- **Was an ethical review conducted?** No; the samples are small and used only for
  illustration.

The answer that most datasheets leave blank is the one that matters here: what was
deliberately excluded. For these samples, nothing was filtered at collection; the
content is whatever the first rows or the seeded offsets returned, including any
personal data they hold.
```

8. **Size a seen-set.** A crawl will see $n = 10^9$ distinct URLs. Using the Bloom-filter rule of {ref}`sec-ch3-crawler-architecture`, how many bits per URL and how many bytes in total does a filter with false-positive rate $p = 1\%$ need, and how many hash functions is optimal? Then give the `Range` header that fetches the record of 9,882 bytes at offset 48,374,322 of a WARC file.

```{admonition} Solution 8
:class: dropdown
The filter needs $-\ln p / (\ln 2)^2 = -\ln 0.01 / 0.4805 = 9.585$ bits per URL, so
$9.585 \times 10^9$ bits, which is $1.198 \times 10^9$ bytes, about 1.2 GB. The optimal
number of hash functions is the bits per URL times $\ln 2$, $9.585 \times 0.6931 = 6.64$,
so 7 in practice. The range of a record ends at its offset plus its length minus 1,
because both ends of a `Range` are inclusive:
`Range: bytes=48374322-48384203`.
```

(sec-ch3-further-reading)=

## Further Reading

(sec-ch3-further-reading-corpora-papers)=

### Corpora and their papers

- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) {cite}`penedo2024fineweb` - the most carefully ablated open web corpus, with a full account of how it was taken from Common Crawl
- [FineWeb2](https://arxiv.org/abs/2506.20920) {cite}`penedo2025fineweb2` - the same pipeline for a thousand languages, with the rehydration result
- [Dolma](https://arxiv.org/abs/2402.00159) {cite}`soldaini2024dolma` - a multi-source corpus whose datasheet is a model of documentation
- [The Pile](https://arxiv.org/abs/2101.00027) {cite}`gao2020pile` - the case for source diversity, and the Books3 cautionary tale
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) {cite}`penedo2023refinedweb` - the argument that filtered web alone is enough
- [RedPajama](https://arxiv.org/abs/2411.12372) {cite}`weber2024redpajama` - a 100T-token raw web corpus released with quality signals instead of filters
- [The Stack](https://arxiv.org/abs/2211.15533) {cite}`kocetkov2022stack` and [StarCoder 2 and The Stack v2](https://arxiv.org/abs/2402.19173) {cite}`lozhkov2024starcoder2` - license-filtered code with an opt-out process
- [S2ORC](https://arxiv.org/abs/1911.02782) {cite}`lo2020s2orc` - open-access papers as full text with licenses tracked
- [A Critical Analysis of Common Crawl](https://doi.org/10.1145/3630106.3659033) {cite}`baack2024critical` - what the crawl is and is not

(sec-ch3-further-reading-rights-consent-provenance)=

### Rights, consent and provenance

- [Consent in Crisis](https://arxiv.org/abs/2407.14933) {cite}`longpre2024consent` - the year the web started opting out
- [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787) {cite}`longpre2023data` - the licensing audit
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) {cite}`gebru2021datasheets` - the documentation template
- [What's In My Big Data?](https://arxiv.org/abs/2310.20707) {cite}`elazar2024whats` - measuring corpora before training on them
- [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) {cite}`dodge2021documenting` - what C4's acquisition and filtering removed
- [The Common Pile v0.1](https://arxiv.org/abs/2506.05209) {cite}`kandpal2025common` - an 8 TB openly licensed corpus and a model trained on it
- [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149) {cite}`carlini2024poisoning` - why URL-list datasets need content hashes

(sec-ch3-further-reading-scale-mixtures-synthetic-data)=

### Scale, mixtures and synthetic data

- [Will we run out of data?](https://arxiv.org/abs/2211.04325) {cite}`villalobos2024run` - the size of the stock of human text
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) {cite}`muennighoff2023scaling` - how many epochs are safe
- [DoReMi](https://arxiv.org/abs/2305.10429) {cite}`xie2023doremi` - learning the mixture weights
- [A Pretrainer's Guide to Training Data](https://arxiv.org/abs/2305.13169) {cite}`longpre2023pretrainers` - age, domain and quality effects measured
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) {cite}`gunasekar2023textbooks` and [Self-Instruct](https://arxiv.org/abs/2212.10560) {cite}`wang2023selfinstruct` - synthetic pretraining and instruction data
- [AI models collapse when trained on recursively generated data](https://doi.org/10.1038/s41586-024-07566-y) {cite}`shumailov2024curse` - the failure mode of synthetic data

(sec-ch3-further-reading-multilingual)=

### Multilingual

- [mT5 and mC4](https://arxiv.org/abs/2010.11934) {cite}`xue2021mt5`, [OSCAR](https://arxiv.org/abs/2201.06642) {cite}`abadji2022oscar`, [CulturaX](https://arxiv.org/abs/2309.09400) {cite}`nguyen2023culturax`
- [Quality at a Glance](https://arxiv.org/abs/2103.12028) {cite}`kreutzer2022quality` and [MADLAD-400](https://arxiv.org/abs/2309.04662) {cite}`kudugunta2023madlad` - the audits that found which "languages" in crawled corpora were real
- [The BigScience ROOTS Corpus](https://arxiv.org/abs/2303.03915) {cite}`laurenccon2022bigscience` - community-sourced multilingual acquisition

(sec-ch3-further-reading-tools-services)=

### Tools and services

- [Common Crawl](https://commoncrawl.org/get-started) - snapshots, the CDX index API and the columnar index
- [warcio](https://github.com/webrecorder/warcio) - streaming WARC reader and writer
- [Hugging Face dataset viewer API](https://huggingface.co/docs/dataset-viewer) - the rows endpoint the notebook uses to sample hosted corpora
- [datatrove](https://github.com/huggingface/datatrove) and the [Dolma toolkit](https://github.com/allenai/dolma) - pipeline libraries with Common Crawl readers
- [Wikimedia dumps](https://dumps.wikimedia.org/) - Wikipedia and sister projects

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
