# Data Preprocessing and Cleaning

A language model is a compressed description of its training data. Every property you want the model to have, from grammatical fluency to factual reliability to the absence of memorised phone numbers, is either present in the data that reaches the optimiser or absent from the model. This chapter covers the stage between acquiring raw text (Chapter 3) and managing a versioned dataset (Chapter 5): turning billions of crawled, scanned and scraped documents into a corpus that is worth spending compute on.

## Learning Objectives

By the end of this chapter, you will understand:

- Why a large fraction of raw web text is discarded before training, and what the retained fraction looks like
- How to extract, normalise and language-identify text so that later stages see consistent input
- The heuristic, perplexity-based and classifier-based quality filters used by the major open corpora, and how to measure whether a filter helps
- Exact, near-duplicate and substring deduplication, including a MinHash and locality-sensitive hashing implementation
- How personal data, toxic content and benchmark test sets are detected and removed
- How the stages fit into a reproducible, shard-parallel pipeline with auditable drop statistics

## Hands-On Learning

The accompanying notebook, [Preprocessing Pipeline Demo](examples/preprocessing_pipeline_demo.ipynb), runs every technique in this chapter on a small synthetic corpus with assertion-checked results:

- Unicode normalisation and whitespace clean-up ({ref}`nb-prep-normalize`)
- A character n-gram language identifier ({ref}`nb-prep-langid`)
- Gopher-style quality and repetition filters ({ref}`nb-prep-filters`)
- A smoothed n-gram perplexity filter ({ref}`nb-prep-perplexity`)
- Exact and MinHash near-duplicate detection, with the LSH probability curve ({ref}`nb-prep-dedup`)
- Personal-data redaction and benchmark decontamination ({ref}`nb-prep-pii`, {ref}`nb-prep-decontam`)
- A complete multi-stage pipeline with per-stage drop accounting ({ref}`nb-prep-pipeline`)

Everything runs on a CPU in well under a minute and uses only the standard library and NumPy.

## Chapter Overview

Think of a raw web crawl as the contents of every recycling bin in a city. Somewhere in there is a library's worth of good writing, but it is mixed with navigation menus, advertising, machine-translated spam, the same press release copied onto two thousand sites, scanned invoices with people's addresses on them, and the answer keys to the exams you were planning to grade the model on. Preprocessing is the sorting plant.

**Extraction and normalisation** is unpacking. HTML tags, encoding errors and the hundred visually identical ways Unicode can spell a character are removed so that every later stage sees plain, consistent text.

**Language identification** sorts the pile by language, because a filter tuned for English will throw away perfectly good Swahili, and a tokenizer trained on the mixture will be bad at both.

**Quality filtering** is the inspector who glances at each item and discards what is obviously not a document: a page that is ninety per cent bullet points, a wall of hashtags, a string of product codes. A few cheap rules remove most of the junk; a trained classifier catches what rules miss.

**Deduplication** finds the two thousand copies of the press release and keeps one. Duplicates waste compute, make the model memorise rather than generalise, and are the main route by which test sets leak into training.

**Privacy and safety filtering** removes what should never have been collected: email addresses, phone numbers, credit-card numbers and content that would make the model unsafe to deploy.

**Decontamination** checks the corpus against the benchmarks the model will be evaluated on, so that a high score means the model learned something rather than that it has seen the answers.

**Pipeline engineering** makes all of this run across thousands of machines, in a fixed order, with a count of what each stage dropped and why, so that the corpus can be rebuilt, audited and improved.

The theme throughout is that none of these decisions is neutral. Every filter encodes a view of what "good text" is, and that view is inherited by the model. The chapter therefore pairs each technique with the way its effect is measured.

## 1. Why Preprocessing Decides What a Model Learns

### 1.1 From raw crawl to training tokens

Common Crawl, the source of most open pretraining corpora, publishes a new snapshot of several billion web pages every month or two. Only a small fraction of that text ends up in any training set. The table below gives representative retention figures reported by the teams who built the main open English corpora; the exact numbers depend on the snapshots used, but the shape is always the same.

| Corpus                                      | Starting material                     | Retained after cleaning           | Main discards                                                        |
| ------------------------------------------- | ------------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| C4 {cite}`raffel2019exploring`              | One Common Crawl snapshot (~20 TB)    | ~750 GB (~156B tokens)            | Non-English, lines without terminal punctuation, bad-word pages      |
| MassiveText (Gopher) {cite}`rae2021scaling` | Web, books, news, code, Wikipedia     | ~10.5 TB after filtering and dedup | Repetitive pages, non-English, quality heuristics, near-duplicates  |
| RefinedWeb {cite}`penedo2023refinedweb`     | 90 Common Crawl snapshots (WARC)      | ~5T tokens (about a tenth of input documents) | Boilerplate, language, repetition, exact and fuzzy duplicates |
| FineWeb {cite}`penedo2024fineweb`           | 96 Common Crawl snapshots             | ~15T tokens                       | URL blocklist, language, Gopher and C4 rules, per-snapshot MinHash   |
| Dolma {cite}`soldaini2024dolma`             | Web, code, papers, books, Wikipedia   | ~3T tokens                        | Language, quality, toxicity, PII, deduplication, decontamination     |

Most of the text, usually well over 90%, is thrown away, and the discarded portion is not random. It is disproportionately boilerplate, spam, machine-generated text and duplicate content, so the filter's definition of those categories shapes the model.

### 1.2 Quality versus quantity

The scaling laws of Chapter 2 treat tokens as interchangeable, but they are not. Several results show that a smaller, cleaner corpus beats a larger, dirtier one at fixed compute:

- **Gopher** {cite}`rae2021scaling` found that the combination of heuristic quality filters and deduplication improved downstream performance across the board in ablations on a 1.4B-parameter model.
- **Deduplication alone** {cite}`lee2022deduplicating` lets a model reach the same validation perplexity in fewer steps and reduces the fraction of generated text that is copied verbatim from the training set by an order of magnitude.
- **Textbook-quality data** {cite}`gunasekar2023textbooks` allowed a 1.3B-parameter model trained on 7B tokens of filtered and synthetic code to outperform models trained on hundreds of times more unfiltered data on HumanEval.
- **Classifier-filtered web text** was the strongest single intervention in DataComp-LM's controlled comparison {cite}`li2024datacomp`, ahead of every heuristic rule set and deduplication strategy tested, and FineWeb-Edu {cite}`penedo2024fineweb` reports large gains on knowledge and reasoning benchmarks from the same idea.

The caveat, which Section 4.5 returns to, is that "better" is measured on a fixed set of benchmarks. A filter that keeps educational prose and drops forum conversation will raise scores on knowledge benchmarks and may lower the model's ability to handle informal text. Preprocessing is a design decision, not a neutral clean-up.

### 1.3 The stages of a pipeline

Most published pipelines are a variation of the following sequence. The order matters because the cheap stages run first and shrink the input for the expensive ones, and because some stages assume the output of others (deduplication over normalised text finds more duplicates than deduplication over raw HTML).

| Stage                   | Typical cost per document | Typical drop rate | Depends on                       |
| ----------------------- | ------------------------- | ----------------- | -------------------------------- |
| URL and domain blocking | Negligible                | 1 to 10%          | Nothing                          |
| Text extraction         | Milliseconds              | Reduces size, not count | Raw HTML                   |
| Normalisation           | Microseconds              | None              | Extracted text                   |
| Language identification | Microseconds              | 50 to 90%         | Normalised text                  |
| Repetition and quality rules | Microseconds         | 10 to 40%         | Language (thresholds per language) |
| Exact deduplication     | Microseconds              | 10 to 30%         | Normalised text                  |
| Near deduplication      | Milliseconds plus a global join | 10 to 50%   | Everything above                 |
| Classifier scoring      | Milliseconds (CPU) to GPU time | 50 to 90% if used as a hard filter | Everything above |
| PII and toxicity        | Microseconds to milliseconds | Modifies rather than drops | Final text            |
| Decontamination         | Microseconds              | Under 1%          | Benchmark set                    |

The percentages compound. A document that survives language identification with probability 0.3, quality rules with 0.7 and deduplication with 0.6 has about a one in eight chance of being trained on, which is why starting material measured in petabytes produces corpora measured in terabytes.

## 2. Text Extraction and Normalisation

### 2.1 From HTML to text

Common Crawl distributes each snapshot in two forms: WARC files containing the raw HTTP responses, and WET files containing text that Common Crawl's own extractor pulled from the HTML. The WET extraction keeps navigation menus, cookie banners, footers and repeated headers, and early corpora such as C4 built on top of it. RefinedWeb {cite}`penedo2023refinedweb` and FineWeb {cite}`penedo2024fineweb` both re-extract from WARC with a boilerplate-aware extractor (trafilatura in FineWeb's case) and report that this single change gives a measurable improvement in downstream performance, because the model no longer spends capacity learning that "Skip to main content" follows every title.

A boilerplate remover works from the structure of the page rather than its words. The usual signals are:

- **Text density**: the ratio of text characters to tag characters within a block. Article bodies are dense; menus are sparse.
- **Link density**: the fraction of a block's text that sits inside anchors. Navigation is almost all links.
- **Position and repetition**: content that appears in the same position on many pages of a site is a template, not an article.
- **Tag semantics**: `<article>`, `<main>` and `<p>` are more likely to hold content than `<nav>`, `<aside>` or `<footer>`.

The output of extraction is plain text with paragraph breaks preserved. Whether to keep anything else (headings as Markdown, tables as rows, lists as bullets) is a choice that propagates all the way to the model's output format. Pipelines that aim at code or mathematics keep more structure than pipelines that aim at prose.

### 2.2 Encoding and Unicode normalisation

Web text arrives in dozens of declared encodings, some of them wrong. The first job is to produce valid UTF-8; the second is to make equivalent strings byte-identical, because deduplication and tokenization both compare bytes.

Unicode defines four normalisation forms. The two that matter here:

- **NFC** composes characters with their combining marks: the two-code-point sequence `e` + U+0301 becomes the single code point `é`. This is lossless and almost always wanted.
- **NFKC** additionally replaces compatibility characters with their canonical equivalents: the ligature `ﬁ` becomes `fi`, full-width `Ａ` becomes `A`, superscript `²` becomes `2`. This loses information that matters in mathematics and some scripts, so it is applied selectively.

Beyond normalisation, the following are removed or replaced by essentially every pipeline:

- Control characters other than newline and tab, including the null byte, which breaks several downstream tools
- The byte-order mark U+FEFF when it appears mid-text
- Zero-width characters (U+200B to U+200D) that are invisible but change tokenization
- Private-use and unassigned code points
- Mojibake, the garbled sequences produced by decoding UTF-8 as Latin-1 (`Ã©` for `é`); the `ftfy` library repairs the common cases

```python
import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_text(text: str, form: str = "NFC") -> str:
    """Canonicalise Unicode and whitespace without changing the words.

    Keeps paragraph breaks (a blank line) but collapses runs of spaces and
    runs of three or more newlines; strips control and zero-width characters.
    """
    text = unicodedata.normalize(form, text)
    text = _CONTROL.sub("", text)
    text = _ZERO_WIDTH.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# The decomposed and composed spellings of "café" become identical:
decomposed = "café"
composed = "café"
assert decomposed != composed
assert normalize_text(decomposed) == normalize_text(composed) == "café"

# NFKC folds compatibility characters; NFC does not:
assert normalize_text("ﬁle Ａ", form="NFKC") == "file A"
assert normalize_text("ﬁle Ａ", form="NFC") == "ﬁle Ａ"
```

One rule that is easy to get wrong: do not lowercase. Case carries information (proper nouns, sentence boundaries, the difference between `US` and `us`), modern tokenizers handle it without vocabulary explosion, and it cannot be put back.

### 2.3 Lines, paragraphs and document boundaries

The unit of most later filters is the line, so the definition of a line should be fixed here and not changed downstream. The conventions used by C4 and inherited by most of its successors:

- A line is a run of text between newline characters after extraction. Lines, not sentences, are what the terminal-punctuation and bullet-point rules in Section 4.1 inspect.
- A paragraph break is one blank line. Runs of blank lines are collapsed.
- A document is one page, one file or one record. Its boundary is where the model will later see an end-of-document token, and it is the unit that deduplication compares.

Pipelines that split a document into chunks before filtering (for example, to score chunks with a classifier) must either filter at the document level afterwards or accept that a document can be partially kept.

## 3. Language Identification

### 3.1 Classifiers and thresholds

Language identification is a solved problem for long documents in well-resourced languages and an open one for short texts, closely related languages, and code-switched text. The standard tool is a fastText classifier {cite}`joulin2017bag` trained on character n-grams: the `lid.176` model distinguishes 176 languages at microsecond speed, and the later OpenLID and GlotLID models cover several hundred and well over a thousand languages respectively.

The classifier returns a distribution over languages. Pipelines keep a document if the top language is the target and its score exceeds a threshold. The threshold trades recall for precision:

- C4 kept pages scored at least 0.99 English by `langdetect`, which is strict enough to drop much multilingual and dialectal content.
- RefinedWeb and FineWeb keep pages scored at least 0.65 by fastText.
- Dolma keeps pages scored at least 0.5, trading some precision for recall on short pages.

Multilingual corpora run the same classifier and route each document to a per-language pipeline, because the quality thresholds in Section 4 (mean word length, stop-word counts, terminal punctuation) are language-specific. ROOTS {cite}`laurenccon2022bigscience` applies per-language parameter tables produced with native-speaker review.

### 3.2 A character n-gram identifier from scratch

The fastText model is a linear classifier over hashed character n-grams. The far older method of Cavnar and Trenkle, which ranks the most frequent n-grams of a text and compares the ranking against per-language profiles, still works well enough to illustrate why character n-grams are the right features: they capture orthography (`th`, `sch`, `ção`) without needing a dictionary, and they are robust to unknown words.

```python
from collections import Counter


def char_ngrams(text: str, n_values=(1, 2, 3)) -> Counter:
    padded = f" {text.lower()} "
    counts: Counter = Counter()
    for n in n_values:
        for i in range(len(padded) - n + 1):
            counts[padded[i : i + n]] += 1
    return counts


def build_profile(texts, top_k: int = 300):
    """Rank the most frequent n-grams of a language's training text."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(char_ngrams(text))
    return {gram: rank for rank, (gram, _) in enumerate(counts.most_common(top_k))}


def out_of_place_distance(doc_profile, lang_profile, top_k: int = 300) -> int:
    """Sum of rank differences; n-grams missing from the language get the max penalty."""
    return sum(
        abs(rank - lang_profile.get(gram, top_k)) for gram, rank in doc_profile.items()
    )


def identify(text: str, profiles: dict) -> tuple:
    doc = build_profile([text])
    scores = {lang: out_of_place_distance(doc, prof) for lang, prof in profiles.items()}
    best = min(scores, key=scores.get)
    return best, scores


profiles = {
    "en": build_profile(
        [
            "The quick brown fox jumps over the lazy dog.",
            "Language models are trained on large amounts of text.",
            "This is the house that Jack built.",
        ]
    ),
    "de": build_profile(
        [
            "Der schnelle braune Fuchs springt über den faulen Hund.",
            "Sprachmodelle werden mit großen Textmengen trainiert.",
            "Das ist das Haus, das Jack gebaut hat.",
        ]
    ),
}
assert identify("Which language is this sentence written in?", profiles)[0] == "en"
assert identify("In welcher Sprache ist dieser Satz geschrieben?", profiles)[0] == "de"
```

Real profiles are built from megabytes of text per language, and real classifiers learn weights for each n-gram rather than using ranks, but the failure modes are the same as the toy's: a document of a few words has too few n-grams to rank, a list of names has no language, and text in Bosnian, Croatian and Serbian shares most of its n-gram profile.

```{admonition} Normalisation and Language Identification
:class: tip
The notebook section {ref}`nb-prep-normalize` applies the normaliser to a corpus containing mojibake, zero-width characters and decomposed accents, and {ref}`nb-prep-langid` builds the n-gram identifier from a small parallel corpus and shows how its confidence collapses as documents get shorter.
```

## 4. Quality Filtering

### 4.1 Heuristic rules

Rule-based filters are cheap, interpretable and surprisingly effective. The two most widely copied sets are the C4 rules {cite}`raffel2019exploring` and the Gopher rules {cite}`rae2021scaling`; FineWeb {cite}`penedo2024fineweb` applies both and adds three of its own.

**C4 rules** operate on lines and pages:

- Keep only lines that end in a terminal punctuation mark (period, exclamation mark, question mark or closing quotation mark).
- Drop lines with fewer than five words.
- Drop pages with fewer than three sentences.
- Drop lines containing the word "javascript" (a proxy for "please enable JavaScript" notices).
- Drop pages containing any word from a blocklist, the string "lorem ipsum", or a curly brace (a proxy for code).

The terminal-punctuation rule is the most consequential: it removes almost every heading, list item and table cell, which is why C4 contains little structured text and why later corpora relaxed it to a page-level fraction.

**Gopher rules** operate on whole documents and were tuned by inspection of MassiveText:

- Between 50 and 100,000 words.
- Mean word length between 3 and 10 characters.
- Symbol-to-word ratio for `#` and `...` below 0.1.
- Fewer than 90% of lines starting with a bullet character, and fewer than 30% ending with an ellipsis.
- At least 80% of words containing at least one alphabetic character.
- At least two of a short list of English stop words (`the`, `be`, `to`, `of`, `and`, `that`, `have`, `with`).

**FineWeb additions**, found by comparing the rejected and retained halves of the Gopher filter on a held-out sample:

- Fraction of lines ending in punctuation at least 0.12 (a soft version of the C4 line rule).
- Fraction of characters in duplicated lines below 0.1.
- Fraction of lines shorter than 30 characters below 0.67.

The implementation below follows the Gopher rules. The function returns the first reason a document fails, which is what a pipeline needs to record.

```python
import re

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("-", "*", "•", "·", "◦", "▪", "▫", "‣", "⁃")
_WORD = re.compile(r"\S+")
_ALPHA = re.compile(r"[A-Za-z]")


def gopher_quality_filter(text: str):
    """Return (keep, reason). Thresholds follow Rae et al. (2021), Appendix A."""
    words = _WORD.findall(text)
    n_words = len(words)
    if n_words < 50:
        return False, "too_few_words"
    if n_words > 100_000:
        return False, "too_many_words"

    mean_len = sum(len(w) for w in words) / n_words
    if not 3 <= mean_len <= 10:
        return False, "mean_word_length"

    hashes = text.count("#")
    ellipses = text.count("...") + text.count("…")
    if (hashes + ellipses) / n_words > 0.1:
        return False, "symbol_ratio"

    lines = [line for line in text.split("\n") if line.strip()]
    if lines:
        bullet_frac = sum(line.lstrip().startswith(BULLETS) for line in lines) / len(lines)
        if bullet_frac > 0.9:
            return False, "bullet_lines"
        ellipsis_frac = sum(
            line.rstrip().endswith(("...", "…")) for line in lines
        ) / len(lines)
        if ellipsis_frac > 0.3:
            return False, "ellipsis_lines"

    alpha_frac = sum(bool(_ALPHA.search(w)) for w in words) / n_words
    if alpha_frac < 0.8:
        return False, "non_alpha_words"

    lowered = {w.lower().strip(".,;:!?\"'()") for w in words}
    if len(STOP_WORDS & lowered) < 2:
        return False, "stop_words"

    return True, "ok"


prose = (
    "The committee met on Tuesday to discuss the proposal. Members raised several "
    "concerns about the budget, and the chair agreed to circulate a revised draft "
    "before the next meeting. The draft will have to address the timeline as well "
    "as the cost, since both were questioned at length. Everyone present agreed "
    "that the work should continue, with a final decision expected in the spring."
)
assert gopher_quality_filter(prose) == (True, "ok")

listing = "\n".join(f"- item {i}" for i in range(60))
assert gopher_quality_filter(listing) == (False, "mean_word_length")

catalogue = " ".join(f"SKU-{i:05d}" for i in range(80))
assert gopher_quality_filter(catalogue)[1] in {"non_alpha_words", "stop_words"}
```

Every threshold above was set for English. The mean-word-length bounds exclude most Chinese and Japanese text (no spaces, so one "word" per line) and much German (compounds), and the stop-word list is meaningless outside English. Multilingual pipelines replace the word splitter with a language-appropriate tokenizer and the stop-word list with a per-language one.

### 4.2 Repetition filters

A page that repeats itself, whether because of a template, a scraper bug or a spam generator, is both low in information and dangerous: models trained on repetitive text learn to produce it. Gopher's repetition rules, which FineWeb applies before any other quality rule, measure repetition at three granularities and drop the document if any exceeds its threshold.

| Measure                                       | Threshold |
| --------------------------------------------- | --------- |
| Fraction of lines that are duplicates          | 0.30      |
| Fraction of paragraphs that are duplicates     | 0.30      |
| Fraction of characters in duplicate lines      | 0.20      |
| Fraction of characters in duplicate paragraphs | 0.20      |
| Fraction of characters in the most common 2-gram | 0.20    |
| Fraction of characters in the most common 3-gram | 0.18    |
| Fraction of characters in the most common 4-gram | 0.16    |
| Fraction of characters in duplicate 5-grams    | 0.15      |
| Fraction of characters in duplicate 6-grams    | 0.14      |
| Fraction of characters in duplicate 7-grams    | 0.13      |
| Fraction of characters in duplicate 8-grams    | 0.12      |
| Fraction of characters in duplicate 9-grams    | 0.11      |
| Fraction of characters in duplicate 10-grams   | 0.10      |

The n-gram measures are computed over words. "Fraction of characters in the most common n-gram" is the number of occurrences of the most frequent n-gram, times its character length, divided by the document's character count. "Fraction of characters in duplicate n-grams" scans the document left to right; when an n-gram has been seen before, its characters are counted and the scan skips ahead by $n$ words, so a repeated passage is counted once per repeat rather than once per overlapping window. This follows the public implementations in datatrove and Dolma.

```python
from collections import Counter


def repetition_stats(text: str) -> dict:
    """Line- and n-gram-level repetition fractions in the style of Rae et al. (2021)."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    words = text.split()
    total_chars = max(1, sum(len(w) for w in words))
    stats = {}

    if lines:
        line_counts = Counter(lines)
        dup_lines = sum(c - 1 for c in line_counts.values())
        dup_line_chars = sum(len(line) * (c - 1) for line, c in line_counts.items())
        stats["dup_line_frac"] = dup_lines / len(lines)
        stats["dup_line_char_frac"] = dup_line_chars / max(1, sum(map(len, lines)))

    for n in (2, 3, 4):
        grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
        if grams:
            gram, count = grams.most_common(1)[0]
            stats[f"top_{n}gram_char_frac"] = count * sum(map(len, gram)) / total_chars

    for n in range(5, 11):
        seen = set()
        dup_chars = 0
        i = 0
        while i < len(words) - n + 1:
            gram = tuple(words[i : i + n])
            if gram in seen:
                dup_chars += sum(map(len, gram))
                i += n  # count a repeat once, then move past it
            else:
                seen.add(gram)
                i += 1
        stats[f"dup_{n}gram_char_frac"] = dup_chars / total_chars

    return stats


THRESHOLDS = {
    "dup_line_frac": 0.30,
    "dup_line_char_frac": 0.20,
    "top_2gram_char_frac": 0.20,
    "top_3gram_char_frac": 0.18,
    "top_4gram_char_frac": 0.16,
    **{f"dup_{n}gram_char_frac": t for n, t in zip(range(5, 11), (0.15, 0.14, 0.13, 0.12, 0.11, 0.10))},
}


def passes_repetition_filter(text: str) -> bool:
    stats = repetition_stats(text)
    return all(stats.get(k, 0.0) <= v for k, v in THRESHOLDS.items())


spam = "buy now " * 40
assert not passes_repetition_filter(spam)

varied = (
    "The harbour was quiet at dawn. A single trawler moved out past the breakwater, "
    "its lights still on. By seven the market had opened and the first crates of fish "
    "were being carried up from the quay. Nobody mentioned the storm forecast for the "
    "afternoon, though everyone had read it."
)
assert passes_repetition_filter(varied)
```

The duplicate n-gram measure is deliberately simple: a passage repeated once contributes its characters once at every $n$ from 5 to 10, and the decreasing thresholds reflect that longer repeats are rarer in genuine prose. Changing the counting rule, for instance to count overlapping windows, changes what the thresholds mean, so a reimplementation should match the reference definition or re-tune them.

### 4.3 Perplexity-based filtering

A different idea, introduced by CCNet {cite}`wenzek2020ccnet`, is to let a language model decide. Train a small n-gram model (CCNet used 5-gram KenLM models) on a reference corpus such as Wikipedia, score every candidate document by its perplexity under that model, and keep the documents that look most like the reference. CCNet divided each language into head, middle and tail thirds by perplexity; Dolma, the Llama 1 pipeline and many others reuse its models.

The appeal is that one scoring function replaces dozens of rules. The problems are what it optimises for:

- Low perplexity under a Wikipedia model means "written like Wikipedia", which penalises dialogue, fiction, code and informal registers.
- Short, formulaic text (boilerplate again) scores well because it is predictable.
- The reference corpus's biases, including which topics and demographics Wikipedia covers, become the filter's biases.

Marion et al. {cite}`marion2023less` tested several pruning criteria and found that keeping the *middle* of the perplexity distribution, rather than the head, gave the best downstream models at a fixed token budget, for the reference models they tested. The extremes are both suspicious: the most predictable text is boilerplate and the least predictable is noise.

A smoothed unigram model is enough to see how the ranking behaves. Perplexity of a document with tokens $w_1, \dots, w_N$ under a model $q$ is

$$\mathrm{PPL}(w_{1:N}) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log q(w_i \mid w_{<i})\right)$$

and for an add-$\alpha$ smoothed unigram model, $q(w) = (c(w) + \alpha)/(C + \alpha |V|)$ where $c(w)$ is the reference count of $w$, $C$ the total reference count and $|V|$ the vocabulary size including an unknown-word slot.

```python
import math
from collections import Counter


class UnigramLM:
    """Add-alpha smoothed unigram model, a stand-in for CCNet's KenLM 5-gram."""

    def __init__(self, reference_texts, alpha: float = 0.5):
        self.counts = Counter(w for t in reference_texts for w in t.lower().split())
        self.total = sum(self.counts.values())
        self.alpha = alpha
        self.vocab_size = len(self.counts) + 1  # +1 for unknown words

    def log_prob(self, word: str) -> float:
        c = self.counts.get(word.lower(), 0)
        return math.log((c + self.alpha) / (self.total + self.alpha * self.vocab_size))

    def perplexity(self, text: str) -> float:
        words = text.split()
        if not words:
            return float("inf")
        return math.exp(-sum(self.log_prob(w) for w in words) / len(words))


reference = [
    "The river flows through the valley and into the sea.",
    "The city was founded in the twelfth century and grew around its port.",
    "Mathematics is the study of quantity, structure, space and change.",
    "The committee published its report on the state of the economy.",
]
lm = UnigramLM(reference)

encyclopedic = "The valley was settled in the twelfth century and the city grew around the river."
gibberish = "zxq vprt qqa lkjh mnbv asdf poiu ytre wqaz"
assert lm.perplexity(encyclopedic) < lm.perplexity(gibberish)

# The pathology: formulaic text that repeats common words scores *better* than real prose
boilerplate = "the the the and the of the and the the"
assert lm.perplexity(boilerplate) < lm.perplexity(encyclopedic)
```

The last assertion is the reason perplexity is rarely used as a hard filter on its own. In practice it is one feature among several, or it is used to bucket documents so that a later classifier or a human can look at each bucket.

### 4.4 Classifier-based filtering

The strongest results in recent corpus construction come from training a classifier to recognise the kind of text you want and keeping documents it scores highly. Three generations of this idea:

**GPT-3** {cite}`brown2020language` trained a logistic regression on hashed word features (Spark's tokenizer and HashingTF) to separate Common Crawl pages from a positive set of WebText, Wikipedia and books. Rather than thresholding, each page was kept if

$$\text{np.random.pareto}(9) > 1 - \text{score}$$

which keeps a document with a probability that rises steeply with its score, so that the corpus stays diverse without a hard cut-off. The same positive-versus-crawl approach, with the same Pareto sampling, was reused for several later models.

**FineWeb-Edu** {cite}`penedo2024fineweb` annotated 460,000 FineWeb documents with a 0 to 5 "educational value" score produced by Llama 3 70B with a rubric prompt, trained a small regression head on embeddings from a 110M-parameter encoder to predict the score, and kept documents scoring 3 or higher. The result was a 1.3T-token subset that outperformed the 15T-token parent on knowledge and reasoning benchmarks at every training budget tested.

**DataComp-LM** {cite}`li2024datacomp` compared many filters in a controlled setting and found that a fastText classifier trained to distinguish instruction-formatted and high-quality question-answer text (OpenHermes and ELI5) from the crawl, keeping the top 10%, was the strongest single intervention, beating both the GPT-3-style classifier and every heuristic set.

What these have in common is that the positive set defines the target distribution. A classifier trained on Wikipedia keeps encyclopaedic text; one trained on explanatory question-answer pairs keeps text that reads like a helpful answer. The choice of positive set is the single most important design decision in a modern pipeline, and it is where a team's view of what the model should be good at enters the data.

```python
import numpy as np


def pareto_keep(scores: np.ndarray, shape: float = 9.0, seed: int = 0) -> np.ndarray:
    """GPT-3's stochastic filter: keep a document when a Pareto draw exceeds 1 - score.

    NumPy's pareto is the Lomax distribution, so the keep probability at score s
    is (2 - s)**-9: 1.0 at s = 1, about 0.42 at 0.9, 0.026 at 0.5, 0.002 at 0.
    """
    rng = np.random.default_rng(seed)
    return rng.pareto(shape, size=scores.shape) > (1.0 - scores)


scores = np.linspace(0.0, 1.0, 11)
keep_rate = np.stack([pareto_keep(scores, seed=s) for s in range(2000)]).mean(axis=0)
assert keep_rate[-1] == 1.0  # score 1.0 is always kept
assert keep_rate[0] < 0.05  # score 0.0 almost never
assert np.all(np.diff(keep_rate) >= -0.02)  # monotone up to sampling noise
```

Because the expected keep rate at score $s$ is $P(X > 1 - s)$ for $X \sim \text{Pareto}(9)$, which is $(1 + 1 - s)^{-9}$, the curve is steep: a document at score 0.5 is kept with probability about $1.5^{-9} \approx 0.026$. Diversity comes from the sheer number of low-scoring documents, not from a generous acceptance rate.

### 4.5 Measuring whether a filter helps

Every filter in this section was validated the same way, and so should any filter you add: train small models on data with and without it, at the same token budget, and compare on a fixed benchmark suite. FineWeb's protocol, which is typical, trains a 1.8B-parameter model on about 28B tokens per variant and reports the average of a set of benchmarks that show a clear signal at that scale (HellaSwag, ARC, PIQA, MMLU, OpenBookQA, CommonsenseQA and a few others), using several seeds because the differences are often within noise.

Three things to watch for:

- **Keep-rate accounting.** A filter that improves benchmark scores by dropping 60% of tokens has to be compared against the alternative of simply training on 60% fewer tokens, or on 60% more compute per token. Report the retained fraction alongside the score.
- **Benchmark fit.** A classifier whose positive set resembles benchmark questions will raise benchmark scores whether or not the model is better. FineWeb-Edu's authors note that its gains are concentrated on knowledge benchmarks and that it loses some ground on others.
- **Scale extrapolation.** Filters validated at 1B parameters usually transfer to 10B, but the effect size shrinks, and a few interventions (notably aggressive deduplication) have been reported to reverse at scale when the remaining data is too small for the model.

```{admonition} Quality Filters in Practice
:class: note
The notebook section {ref}`nb-prep-filters` applies the Gopher quality and repetition filters to a synthetic corpus and tabulates the drop reasons, and {ref}`nb-prep-perplexity` ranks the same corpus by unigram perplexity and shows that the head of the ranking is boilerplate, the middle is prose and the tail is noise.
```

## 5. Deduplication

### 5.1 Why duplicates hurt

Web text is heavily duplicated: the same article is syndicated, the same product description appears on every reseller's site, the same licence text heads every source file. Lee et al. {cite}`lee2022deduplicating` found a single 61-word sentence repeated more than 60,000 times in C4, and that near-duplicate removal dropped about 3% of its documents while substring deduplication removed a further few per cent of its tokens. The consequences of leaving duplicates in:

- **Memorisation.** The fraction of generated text copied verbatim from training data dropped by roughly 10x after deduplication, and the probability that a sequence is memorised grows with its number of occurrences {cite}`carlini2021extracting`. This is the main privacy risk in a pretrained model, because the duplicated sequences include names and addresses.
- **Wasted compute.** A duplicated document is a training step spent on something the model has already seen. Lee et al. report reaching the same validation loss in fewer steps after deduplication.
- **Test-set leakage.** Benchmark questions on the web are duplicated like everything else, so decontamination (Section 7) is partly a deduplication problem.
- **Skewed mixtures.** Without deduplication, the effective mixture of the corpus is set by what was copied most, not by what was chosen.

The flip side, established by Muennighoff et al. {cite}`muennighoff2023scaling`, is that training for a few epochs on a deduplicated corpus costs almost nothing. Up to about four repetitions, repeated tokens are nearly as valuable as fresh ones. Deduplication removes the thousand-fold copies, not the second epoch.

### 5.2 Exact deduplication

Exact deduplication hashes each document after normalisation and keeps one document per hash. It catches syndicated content and mirrored sites, costs one hash per document and one global set, and is the first thing every pipeline does. The choice of hash is about speed and collision rate rather than security; a 64-bit non-cryptographic hash such as xxHash has a real chance of one or two collisions among a few billion documents (the birthday bound), each of which costs one wrongly dropped document; a 128-bit hash removes even that, and SHA-1 is common because it is in every standard library.

```python
import hashlib


def exact_dedup(docs, key=lambda d: d):
    """Keep the first document with each content hash; yields (doc, is_duplicate)."""
    seen = set()
    for doc in docs:
        digest = hashlib.sha1(key(doc).encode("utf-8")).hexdigest()
        yield doc, digest in seen
        seen.add(digest)


corpus = ["a b c", "d e f", "a b c", "g h i", "d e f"]
flags = [dup for _, dup in exact_dedup(corpus)]
assert flags == [False, False, True, False, True]
```

Exact matching over the whole document misses the common case where two copies differ by a date, a byline or a trailing menu. The usual mitigation is to hash a normalised key: lowercase, punctuation removed, digits replaced by a placeholder. Even then, near-duplicates need their own method.

### 5.3 Near-duplicate detection with MinHash and LSH

Two documents are near-duplicates if the sets of their $n$-gram "shingles" have high Jaccard similarity,

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Computing Jaccard similarity for every pair of a billion documents is impossible, and MinHash {cite}`broder1997resemblance` is the standard way around it. The key fact: if $h$ is a random hash function over shingles and $\min h(A)$ denotes the smallest hash value among $A$'s shingles, then

$$P\left[\min h(A) = \min h(B)\right] = J(A, B)$$

because the minimum over $A \cup B$ lands in $A \cap B$ with exactly that probability. With $k$ independent hash functions, the fraction of positions at which the two signatures agree is an unbiased estimate of $J$ with variance $J(1-J)/k$. A signature of 128 or 256 small integers replaces a document of any size.

Locality-sensitive hashing (LSH) then turns the signature into candidate pairs without comparing every pair. Split the $k$ positions into $b$ bands of $r$ rows each ($k = br$). Hash each band of the signature to a bucket; two documents are candidates if any band lands them in the same bucket. The probability that a pair with similarity $s$ becomes a candidate is

$$P(\text{candidate}) = 1 - (1 - s^r)^b$$

which is a steep S-curve with its midpoint near $(1/b)^{1/r}$. Choosing $b$ and $r$ sets the similarity threshold.

Gopher used 13-word shingles with 450 hash functions at a 0.8 threshold; RefinedWeb used 5-word shingles with 9,000 hash functions at the same threshold; FineWeb used 5-word shingles with 112 hash functions in 14 bands of 8 rows, which puts its threshold near 0.75. Dolma relied on exact matching of URLs, documents and paragraphs through a Bloom filter rather than MinHash. The implementation below uses a universal hash family $h_i(x) = (a_i x + b_i) \bmod p$ over a 32-bit shingle hash, which is what most production implementations do.

```python
import hashlib
import re

import numpy as np

_MERSENNE = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def shingles(text: str, n: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "little")


class MinHash:
    def __init__(self, num_perm: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.int64)
        self.b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.int64)
        self.num_perm = num_perm

    def signature(self, shingle_set: set) -> np.ndarray:
        if not shingle_set:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.int64)
        x = np.array([_hash32(s) for s in shingle_set], dtype=np.int64)
        # (a*x + b) mod p, then fold to 32 bits; object dtype avoids int64 overflow
        ax = (x[:, None].astype(object) * self.a.astype(object) + self.b.astype(object)) % _MERSENNE
        hashes = np.array(ax, dtype=np.int64) & _MAX_HASH
        return hashes.min(axis=0)

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Band the signature; documents sharing any band bucket are candidate pairs."""

    def __init__(self, num_perm: int = 128, bands: int = 16):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.buckets = [dict() for _ in range(bands)]

    def add(self, doc_id, sig: np.ndarray):
        for band, table in enumerate(self.buckets):
            key = sig[band * self.rows : (band + 1) * self.rows].tobytes()
            table.setdefault(key, []).append(doc_id)

    def candidate_pairs(self) -> set:
        pairs = set()
        for table in self.buckets:
            for ids in table.values():
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))
        return pairs


base = (
    "The town council voted on Monday to approve the new library building after "
    "months of public consultation about its design, location and projected cost."
)
docs = {
    "orig": base,
    "retitled": "Library approved. " + base + " More details to follow.",
    "unrelated": (
        "Quarterly earnings exceeded analyst expectations on strong demand for cloud "
        "services, sending shares higher in after-hours trading on Thursday evening."
    ),
}

mh = MinHash(num_perm=128)
sigs = {k: mh.signature(shingles(v)) for k, v in docs.items()}

true_j = len(shingles(base) & shingles(docs["retitled"])) / len(shingles(base) | shingles(docs["retitled"]))
est_j = MinHash.estimate_jaccard(sigs["orig"], sigs["retitled"])
assert abs(true_j - est_j) < 0.15, (true_j, est_j)
assert MinHash.estimate_jaccard(sigs["orig"], sigs["unrelated"]) < 0.1

index = LSHIndex(num_perm=128, bands=16)
for k, s in sigs.items():
    index.add(k, s)
assert ("orig", "retitled") in index.candidate_pairs()
assert ("orig", "unrelated") not in index.candidate_pairs()
```

Production systems differ from this toy in three ways. They compute signatures in a distributed map over shards and do the bucket join as a global shuffle; they then build the connected components of the candidate graph (union-find) and keep one document per component, usually the longest or the earliest; and they skip the exact Jaccard check on candidates because at the chosen threshold the false-positive rate is low enough to ignore.

Which document to keep from a cluster is not a detail. Keeping the shortest copy drops the version with the comment thread; keeping the longest keeps the version with the navigation menu. FineWeb's authors found that deduplicating each Common Crawl snapshot independently produced *better* models than deduplicating globally across all snapshots, because global deduplication preferentially removed the oldest copy of good content while leaving the long tail of low-quality pages that were never duplicated. The content that survives many crawls is, on average, the content worth keeping.

### 5.4 Substring deduplication with suffix arrays

Document-level methods miss the case where a long passage (a licence header, a quotation, a boilerplate disclaimer) is embedded in otherwise distinct documents. Lee et al. {cite}`lee2022deduplicating` address this with a suffix array over the concatenated, tokenized corpus: sort all suffixes, then any two adjacent suffixes that share a prefix of at least 50 tokens mark a repeated span, which is removed from all but one occurrence.

A suffix array over $N$ tokens needs about $8N$ bytes for the index alone, so the method is run over a few hundred gigabytes at a time on a large-memory machine rather than in a distributed map. The implementation below is the naive $O(N \log N)$-comparison version, enough to see the idea on a few thousand tokens.

```python
def repeated_spans(tokens, min_len: int = 50):
    """Return (start, end) of every span that occurs at least twice, via a suffix array.

    Naive construction (sorts full suffixes); fine for a few thousand tokens,
    hopeless for a corpus, which uses SA-IS or a parallel construction instead.
    """
    n = len(tokens)
    suffix_array = sorted(range(n), key=lambda i: tokens[i:])
    spans = []
    for k in range(1, n):
        i, j = suffix_array[k - 1], suffix_array[k]
        lcp = 0
        while i + lcp < n and j + lcp < n and tokens[i + lcp] == tokens[j + lcp]:
            lcp += 1
        if lcp >= min_len:
            later = max(i, j)  # remove the occurrence that comes later in the text
            spans.append((later, later + lcp))
    return sorted(set(spans))


licence = "permission is hereby granted free of charge to any person obtaining a copy".split()
doc_a = "alpha beta gamma".split() + licence + "delta".split()
doc_b = "one two".split() + licence + "three four five".split()
tokens = doc_a + doc_b
spans = repeated_spans(tokens, min_len=len(licence))
assert len(spans) == 1
start, end = spans[0]
assert tokens[start:end] == licence
assert start == len(doc_a) + 2  # the copy in doc_b, not the one in doc_a
```

Because substring removal cuts through the middle of documents, it is applied after document-level deduplication, and the resulting documents are re-checked against the minimum-length rule so that a document reduced to a fragment is dropped.

### 5.5 Semantic deduplication

MinHash finds documents that share words. SemDeDup {cite}`abbas2023semdedup` finds documents that share meaning: embed every document with a pretrained encoder, cluster the embeddings with $k$-means, and within each cluster remove all but one of any group of documents whose cosine similarity exceeds a threshold (0.95 or higher in the paper). On LAION and C4 the method removed a further 20 to 50% of data after conventional deduplication with little or no loss, and training on the smaller set was correspondingly faster.

The cost is an embedding pass over the corpus, which is a forward pass of a small model per document, and the risk is removing paraphrases that a model would benefit from seeing both of. It is applied last among the deduplication stages and, in most published pipelines, conservatively.

```{admonition} Deduplication Implementations
:class: tip
The notebook section {ref}`nb-prep-dedup` runs exact and MinHash deduplication over a corpus seeded with syndicated, lightly edited and unrelated documents, plots the LSH candidate-probability curve for several band settings against the empirical rate, and shows the suffix-array method finding a shared licence header.
```

## 6. Privacy, Toxicity and Safety Filtering

### 6.1 Personal data

Pretrained models reproduce personal data that was in their training set, and the risk grows with duplication {cite}`carlini2021extracting`. Pipelines therefore detect and either remove or mask personally identifiable information (PII) in the final text. The categories and methods used by Dolma {cite}`soldaini2024dolma` and The Stack {cite}`kocetkov2022stack` are representative:

- **Email addresses, phone numbers and IP addresses** are found with regular expressions and replaced with placeholder tokens (`|||EMAIL_ADDRESS|||` in Dolma). Precision is high, recall is good for well-formatted values.
- **Names, usernames, keys and passwords** in code need a named-entity model; StarPII, from the StarCoder pipeline that built on The Stack, is a small encoder fine-tuned for the purpose, run only over source code, where the regular expressions are weakest.
- **Documents dense with PII** (more than five matches in Dolma's pipeline) are dropped outright on the assumption that they are directories, contact lists or scraped forms rather than prose.

Replacement rather than deletion keeps the surrounding sentence grammatical, and a fixed placeholder lets the model learn that such a token exists without learning any value for it.

```python
import re

PII_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "PHONE": re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
    ),
    "IPV4": re.compile(
        r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)"
    ),
}


def redact_pii(text: str, max_matches: int = 5):
    """Replace PII with placeholders; return (text, counts, keep).

    keep is False when the document holds more than max_matches items, which
    marks it as a directory or form rather than prose.
    """
    counts = {}
    for label, pattern in PII_PATTERNS.items():
        text, n = pattern.subn(f"|||{label}|||", text)
        counts[label] = n
    return text, counts, sum(counts.values()) <= max_matches


sample = "Contact Jane at jane.doe@example.org or (555) 123-4567; server 192.168.1.20 is down."
redacted, counts, keep = redact_pii(sample)
assert counts == {"EMAIL": 1, "PHONE": 1, "IPV4": 1}
assert "example.org" not in redacted and "123-4567" not in redacted and "192.168" not in redacted
assert keep

directory = "\n".join(f"user{i}@example.com" for i in range(8))
assert redact_pii(directory)[2] is False
```

Two cautions. Regular expressions written for North American formats miss most of the world's phone numbers, and a pattern loose enough to catch them all will also catch ISBNs, timestamps and version numbers. And redaction is not anonymisation: a document that names a person and describes their circumstances is still identifying after its email address is removed. Pipelines that need stronger guarantees drop whole documents from sources known to contain personal records rather than trying to scrub them.

### 6.2 Toxicity and adult content

Three mechanisms, in increasing order of cost and accuracy:

- **URL and domain blocklists** remove whole sites known for adult content, gambling, malware and spam. FineWeb and RefinedWeb use lists of several million domains maintained for web filtering. This is the first stage of most pipelines because it costs nothing.
- **Word lists** flag documents containing any of a few hundred terms. C4's "bad words" filter is the canonical example, and Dodge et al. {cite}`dodge2021documenting` showed its canonical failure: the list removed a disproportionate amount of text written in African American English and text about LGBTQ identity, because the same words are slurs in one context and self-description in another.
- **Classifiers** trained on labelled toxic and non-toxic text (the Jigsaw datasets, for example) give a probability per document. Dolma applies fastText classifiers for hate speech and NSFW content and drops documents above a threshold tuned by inspection.

The lesson of the C4 analysis is that any filter aimed at toxic content will also remove text by and about the communities that toxic content targets, so the threshold is a trade-off between the two kinds of harm. A more recent practice is to filter less at the pretraining stage and rely on post-training (Chapters 12 and 13) to control behaviour, on the grounds that a model which has never seen a slur cannot recognise one either.

### 6.3 Licences, consent and opt-out

Whether a document may be used is a question for acquisition (Chapter 3), but preprocessing is where the decision is enforced. The Stack keeps only files whose repository licence is on a permissive allowlist and honours opt-out requests by repository; several web corpora respect `robots.txt` directives retroactively and remove sites that have added opt-out metadata since the crawl. These rules are implemented as filters on document metadata and run first, because a document that should not be in the corpus should not consume any further compute.

```{admonition} Redaction in Practice
:class: note
The notebook section {ref}`nb-prep-pii` runs the redaction patterns over documents with international phone formats and version strings to show where the regular expressions over- and under-match.
```

## 7. Benchmark Decontamination

### 7.1 n-gram overlap

An evaluation benchmark is useful only if the model has not seen it. Since benchmarks are published on the web and copied into forums, tutorials and GitHub repositories, they are in every crawl, and a pipeline has to look for them. The standard method is $n$-gram overlap:

- **GPT-3** {cite}`brown2020language` marked a training document as contaminated if it shared any 13-gram with a benchmark example, with special handling for short examples, and reported results on both the full and the "clean" subsets of each benchmark.
- **Llama 2** {cite}`touvron2023llama2` tokenized both sides and marked a token as contaminated if it fell in any 10-gram shared with a benchmark; an example was contaminated if more than 80% of its tokens were. This was an analysis of the evaluation sets rather than a training-data filter: the paper reports performance separately on clean, "not clean", "not dirty" and dirty subsets.
- **Dolma** matched at the paragraph level, using a Bloom filter over benchmark paragraphs, and removed training documents containing a matching paragraph.

The implementation is a set of hashed $n$-grams built from the benchmark and a scan over the training text.

```python
import re


def ngram_set(text: str, n: int) -> set:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def contamination_index(benchmark_examples, n: int = 13) -> dict:
    """Map n -> set of n-grams. An example shorter than n words is indexed whole,
    under its own length, so that it can still be matched."""
    index: dict = {}
    for example in benchmark_examples:
        words = re.findall(r"\w+", example.lower())
        size = min(n, len(words))
        if size:
            index.setdefault(size, set()).update(ngram_set(example, size))
    return index


def contaminated_fraction(doc: str, index: dict) -> float:
    """Largest fraction, over the index's n-gram sizes, of the document's n-grams
    that appear in the benchmark index."""
    best = 0.0
    for n, grams in index.items():
        doc_grams = ngram_set(doc, n)
        if doc_grams:
            best = max(best, len(doc_grams & grams) / len(doc_grams))
    return best


benchmark = [
    "What is the capital of the country whose largest city is Istanbul and whose currency is the lira?",
    "Which element has atomic number 26?",
]
index = contamination_index(benchmark, n=13)

leaked = (
    "Here is a question from the quiz: What is the capital of the country whose largest "
    "city is Istanbul and whose currency is the lira? The answer, of course, is Ankara."
)
innocent = "Istanbul is the largest city in Turkey, and the lira is its currency."
short_leak = "Pub quiz round two. Which element has atomic number 26? Iron, as most tables knew."
assert contaminated_fraction(leaked, index) > 0
assert contaminated_fraction(short_leak, index) > 0  # the 6-word example is indexed whole
assert contaminated_fraction(innocent, index) == 0.0
```

### 7.2 What overlap does not catch

Exact $n$-gram matching misses paraphrases, translations, reformatted versions (a table turned into prose) and benchmarks whose answers appear without their questions. Studies that rephrase benchmark items and retrain have found contamination effects that $n$-gram filters did not detect, and the current state of practice is to combine $n$-gram decontamination of the training set with evaluation on held-out or newly written test sets whose release postdates the training data. Decontamination should be reported as a procedure with its parameters ($n$, tokenization, threshold) and its effect (how many documents were removed per benchmark), not as a binary claim.

```{admonition} Decontamination
:class: tip
The notebook section {ref}`nb-prep-decontam` builds a 13-gram index from a toy benchmark, finds the leaked copies in a corpus, and shows a paraphrase slipping past.
```

## 8. Domain-Specific Preprocessing

### 8.1 Source code

Code corpora use the same stages with different rules. The Stack {cite}`kocetkov2022stack` and its successor StarCoder data pipelines:

- **Licence filtering** by repository, keeping permissive licences only.
- **Near-deduplication** at the file level with MinHash at a 0.85 threshold on 5-gram shingles, which removes roughly 40% of files, since forks and vendored dependencies dominate GitHub.
- **Quality heuristics** per file: drop files whose fraction of alphanumeric characters is below 0.25, whose maximum line length exceeds 1,000 characters, or whose mean line length exceeds 100 characters (minified or generated code), and files with "auto-generated" in the first few lines.
- **Data file removal**: JSON, YAML and CSV files are kept only below a size cap, since they are data rather than code.
- **Secrets and PII** via regular expressions for keys and the StarPII model for names and emails.

```python
import re

_AUTOGEN = re.compile(r"(auto[- ]?generated|do not edit|generated by)", re.IGNORECASE)


def code_quality_filter(source: str):
    """File-level rules in the style of The Stack / StarCoder; returns (keep, reason)."""
    if not source.strip():
        return False, "empty"
    lines = source.split("\n")
    if _AUTOGEN.search("\n".join(lines[:5])):
        return False, "autogenerated"
    alnum = sum(ch.isalnum() for ch in source) / len(source)
    if alnum < 0.25:
        return False, "low_alphanumeric"
    if max(len(line) for line in lines) > 1000:
        return False, "max_line_length"
    if sum(len(line) for line in lines) / len(lines) > 100:
        return False, "mean_line_length"
    return True, "ok"


good = "def add(a, b):\n    return a + b\n\n\nprint(add(2, 3))\n"
assert code_quality_filter(good) == (True, "ok")
assert code_quality_filter("# Auto-generated by protoc. DO NOT EDIT!\nx = 1\n")[1] == "autogenerated"
assert code_quality_filter("x = 1\n" + "y = 1; " * 400 + "\n")[1] == "max_line_length"
assert code_quality_filter("".join("=" * 60 + "\n" for _ in range(20)))[1] == "low_alphanumeric"
```

### 8.2 Mathematics and scientific text

Mathematical text is destroyed by the default prose pipeline: HTML extractors drop MathML and MathJax, the symbol-ratio rule in Section 4.1 rejects equations, NFKC normalisation flattens superscripts, and line-based filters remove display equations because they do not end in a period. OpenWebMath {cite}`paster2023openwebmath` rebuilt extraction from the raw HTML to recover LaTeX from the rendered page, trained a classifier on mathematical content to replace the symbol rules, and kept its own line conventions. Scientific papers from arXiv are handled by converting the LaTeX source rather than the PDF, with macro expansion and bibliography removal. The general point holds for any domain: inspect what the default rules reject before adopting them.

### 8.3 Multilingual text

Every threshold in Section 4 is a function of language. The practical approach, used by ROOTS {cite}`laurenccon2022bigscience` and the Common Crawl-derived multilingual corpora, is a per-language parameter table covering the word-splitting rule (character-based for Chinese, Japanese and Thai), stop-word lists, the mean-word-length bounds, and the perplexity model, set with help from speakers of each language. Scripts need their own treatment: a document in Cyrillic with a Latin-script navigation menu should not fail the "words with an alphabetic character" rule, and right-to-left text should not be reordered by extraction. Languages with little web presence suffer twice, because the language identifier is less accurate on them and the quality filters, tuned on majority languages, reject more of what it finds.

## 9. Assembling the Pipeline

### 9.1 Order of operations

Putting the stages together, this is the order FineWeb runs them in, with decontamination added where most pipelines place it:

1. URL filtering against a blocklist (metadata only; no text read).
1. Text extraction from WARC with a boilerplate-aware extractor.
1. Language identification; keep the target language above threshold.
1. Gopher repetition and quality rules.
1. MinHash deduplication, per snapshot.
1. C4 quality rules (with the line rule relaxed).
1. FineWeb's additional rules.
1. PII redaction.
1. Decontamination against the evaluation suite.
1. Optional classifier scoring (the FineWeb-Edu step).

The principles behind the order: cheap before expensive; rules that depend on a property (language) after the stage that establishes it; deduplication after the text has been normalised and the cheapest filters have shrunk it, but before classifier scoring, so the classifier is not run on copies; redaction last among the content stages so that placeholders do not affect filters; decontamination last of all, against the final text.

### 9.2 Sharding, streaming and idempotence

A corpus is thousands of shards, each a compressed JSONL file of documents with metadata (source, URL, timestamp, language score, per-stage flags). Every stage except deduplication is a pure function of one document, so the pipeline is a map over shards that can run on any number of machines with no coordination. Deduplication needs a global step: compute signatures in a map, group by bucket in a shuffle, resolve clusters, then filter in a second map. Frameworks such as datatrove, the Dolma toolkit and NeMo Curator provide exactly this structure.

Each stage writes its output to a new location rather than modifying its input, and records enough in each document's metadata to be rerun. A stage that fails halfway on a shard can be rerun on that shard alone, and a change to one filter requires rerunning that filter and everything after it, not everything before.

```python
from collections import Counter


class Pipeline:
    """Run document-level stages in order, counting what each one drops and why."""

    def __init__(self, stages):
        self.stages = stages  # list of (name, fn); fn(doc) -> (doc or None, reason)
        self.counts = Counter()
        self.reasons = Counter()

    def run(self, docs):
        for doc in docs:
            self.counts["input"] += 1
            for name, fn in self.stages:
                doc, reason = fn(doc)
                if doc is None:
                    self.counts[f"dropped_by:{name}"] += 1
                    self.reasons[f"{name}:{reason}"] += 1
                    break
            else:
                self.counts["output"] += 1
                yield doc

    def report(self) -> str:
        lines = [f"input   {self.counts['input']}"]
        for name, _ in self.stages:
            lines.append(f"  -{self.counts[f'dropped_by:{name}']:<4} {name}")
        lines.append(f"output  {self.counts['output']}")
        return "\n".join(lines)


def min_length(doc):
    return (doc, "ok") if len(doc["text"].split()) >= 5 else (None, "short")


def no_lorem(doc):
    return (None, "lorem_ipsum") if "lorem ipsum" in doc["text"].lower() else (doc, "ok")


def english_only(doc):
    return (doc, "ok") if doc.get("lang") == "en" else (None, "language")


docs = [
    {"text": "Short.", "lang": "en"},
    {"text": "Lorem ipsum dolor sit amet consectetur adipiscing elit.", "lang": "en"},
    {"text": "Dies ist ein vollständiger Satz in deutscher Sprache.", "lang": "de"},
    {"text": "This document is long enough and in the right language.", "lang": "en"},
]
pipe = Pipeline([("min_length", min_length), ("no_lorem", no_lorem), ("english_only", english_only)])
kept = list(pipe.run(docs))
assert len(kept) == 1
assert pipe.counts["dropped_by:min_length"] == 1
assert pipe.counts["dropped_by:no_lorem"] == 1
assert pipe.counts["dropped_by:english_only"] == 1
assert pipe.reasons["english_only:language"] == 1
```

### 9.3 Keeping the receipts

The output of a pipeline is not just the corpus. It is also:

- **Per-stage counts and drop reasons**, as the `Pipeline` class above records, aggregated over all shards. A stage whose drop rate changes between runs is the first place to look when model quality changes.
- **Samples of rejected documents** from each stage, which are the only way to find out that a rule is rejecting something it should not (equations, poetry, non-English names).
- **The exact code and configuration** that produced the corpus, pinned to a commit, including every threshold, blocklist and classifier checkpoint. A corpus that cannot be rebuilt cannot be fixed.
- **A data card** {cite}`gebru2021datasheets` describing sources, filters, known gaps and intended use, so that the people training on the corpus know what they are getting.

Dolma's release {cite}`soldaini2024dolma` is a good model: the toolkit, the per-stage statistics and the corpus were published together, and every number in the paper can be recomputed.

### 9.4 Preparing the output for training

The last step formats the corpus for the training loader, which Chapter 5 covers in depth. The decisions that belong to preprocessing:

- **Document boundaries** are marked with an end-of-text token, so that the model learns where documents end and attention does not need to cross them.
- **Shuffling** happens at the document level across all sources before sharding, so that every training batch is a mixture; a loader that reads shards in order would otherwise feed the model all of Wikipedia and then all of GitHub.
- **Mixture weights** per source are chosen here or in the loader. Upsampling a small, high-quality source (Wikipedia, textbooks) a few times is standard and, by the result in Section 5.1, costs little.
- **Tokenization** is deferred to the loader or done once and cached; either way the tokenizer is trained on a sample of the *cleaned* corpus, because a tokenizer trained on raw text wastes vocabulary on boilerplate.

```{admonition} The Complete Pipeline
:class: note
The notebook section {ref}`nb-prep-pipeline` assembles normalisation, language identification, the quality and repetition filters, exact and MinHash deduplication, redaction and decontamination into one `Pipeline`, runs it over the synthetic corpus and prints the per-stage drop report.
```

## Exercises

1. **Threshold sensitivity.** Take the Gopher quality filter and sweep the minimum word count from 10 to 200 on a sample of web text. Plot the fraction retained and inspect twenty rejected documents at each setting. At what threshold do you start rejecting documents you would want to keep?
1. **Repetition definitions.** Change `repetition_stats` to count non-overlapping n-gram occurrences and compare its decisions with the overlapping version on a sample. How many documents change category? Which version matches your own judgement better?
1. **LSH curve.** For 128 permutations, plot $1 - (1 - s^r)^b$ for $(b, r) \in \{(8, 16), (16, 8), (32, 4)\}$ and mark the similarity at which each crosses 0.5. Confirm the curve empirically with the notebook's MinHash implementation by generating pairs of documents with controlled overlap.
1. **Which copy to keep.** Build a cluster of near-duplicate documents (an article with and without its comment thread, with and without a navigation menu, truncated). For each "keep the longest", "keep the shortest" and "keep the earliest" policy, say what the model would see. Which policy would you choose for news? For documentation?
1. **Phone numbers of the world.** Extend the redaction patterns to handle UK, German and Indian phone formats. Measure precision on a sample of text containing version numbers, ISBNs and timestamps.
1. **Decontamination by paraphrase.** Paraphrase ten benchmark questions by hand and check whether the 13-gram index catches them. Then try 8-grams. What is the false-positive rate on innocent text at each $n$?
1. **A filter ablation.** Using a small model and a training budget you can afford (even a few million tokens on a CPU gives a signal on validation perplexity), compare training on a corpus with and without the repetition filter. Report retained fraction alongside the result.

## Further Reading

### Corpus construction papers

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) {cite}`raffel2019exploring` - the C4 pipeline, Section 2.2
- [Scaling Language Models: Methods, Analysis and Insights from Training Gopher](https://arxiv.org/abs/2112.11446) {cite}`rae2021scaling` - the MassiveText pipeline and the Gopher rules, Appendix A
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) {cite}`penedo2023refinedweb` - web data only, with extraction and deduplication at scale
- [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) {cite}`penedo2024fineweb` - the most thoroughly ablated open pipeline; read the data-processing section with the public `datatrove` code open
- [Dolma: an Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159) {cite}`soldaini2024dolma` - a fully documented multi-source pipeline with released tooling
- [DataComp-LM](https://arxiv.org/abs/2406.11794) {cite}`li2024datacomp` - a controlled testbed for comparing data curation strategies
- [The Pile](https://arxiv.org/abs/2101.00027) {cite}`gao2020pile` - a curated multi-source corpus and the arguments for source diversity

### Deduplication and memorisation

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) {cite}`lee2022deduplicating` - exact, MinHash and suffix-array deduplication with measured effects
- [On the Resemblance and Containment of Documents](https://doi.org/10.1109/SEQUEN.1997.666900) {cite}`broder1997resemblance` - the original MinHash paper
- [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) {cite}`carlini2021extracting` - why duplicated PII ends up in model output
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) {cite}`muennighoff2023scaling` - how many epochs are safe
- [SemDeDup](https://arxiv.org/abs/2303.09540) {cite}`abbas2023semdedup` - embedding-based deduplication

### Quality filtering and its effects

- [CCNet](https://arxiv.org/abs/1911.00359) {cite}`wenzek2020ccnet` - perplexity-based filtering with KenLM
- [When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale](https://arxiv.org/abs/2309.04564) {cite}`marion2023less` - why the middle of the perplexity distribution wins
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) {cite}`gunasekar2023textbooks` - quality over quantity for code
- [Documenting Large Webtext Corpora: A Case Study on C4](https://arxiv.org/abs/2104.08758) {cite}`dodge2021documenting` - what C4's filters removed and whom it affected
- [A Pretrainer's Guide to Training Data](https://arxiv.org/abs/2305.13169) {cite}`longpre2023pretrainers` - a controlled study of age, quality, toxicity and domain filters
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) {cite}`gebru2021datasheets` - the documentation template

### Tools

- [datatrove](https://github.com/huggingface/datatrove) - Hugging Face's pipeline library, used for FineWeb
- [Dolma toolkit](https://github.com/allenai/dolma) - the Allen Institute's taggers, deduplication and mixing tools
- [NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) - GPU-accelerated curation including fuzzy and semantic deduplication
- [text-dedup](https://github.com/ChenghaoMou/text-dedup) - reference implementations of MinHash, SimHash and suffix-array deduplication
- [trafilatura](https://trafilatura.readthedocs.io/) - boilerplate-aware text extraction from HTML
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html) - the `lid.176` models
- [ftfy](https://ftfy.readthedocs.io/) - mojibake repair

## Bibliography

```{bibliography}
:filter: docname in docnames
```

______________________________________________________________________

*Part of the LLM Atlas - A Comprehensive Guide to Large Language Models*
