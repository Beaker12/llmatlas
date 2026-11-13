"""Write figures/data/wet_stats.json: per-document statistics of English pages from one Common Crawl WET file.

Usage: python fetch_wet_stats.py <wet_part.gz> <wikipedia.json> <output.json>

Inputs, fetched once by hand with a generic User-Agent and kept out of the repository:
  wet_part.gz     the first 40 MiB (bytes 0-41943039) of
                  crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/wet/
                  CC-MAIN-20240220211055-20240221001055-00000.warc.wet.gz
                  (WET files are concatenated gzip members, so a prefix decodes up to the cut);
  wikipedia.json  a list of plain-text English Wikipedia articles (random articles from the
                  MediaWiki extracts API), the reference corpus of the unigram model.
For every page whose WARC-Identified-Content-Language is exactly "eng" and which has at least 50 words, the
output holds the page's word count, mean word length, fraction of words with an alphabetic character (two
of the Gopher statistics of section 4.1) and its perplexity under the page's add-0.5 unigram model trained
on the Wikipedia articles (section 4.3). The figures are drawn from the committed output, never from a
live call.
"""
import json
import math
import re
import sys
import zlib
from collections import Counter


def records(path):
    data, out = open(path, "rb").read(), []
    while data:
        d = zlib.decompressobj(31)
        try:
            chunk = d.decompress(data)
        except zlib.error:
            break
        if not d.eof:
            break                                   # the member cut by the byte range
        out.append(chunk.decode("utf-8", "replace"))
        data = d.unused_data
    return out


def main(wet, wiki, output):
    texts = json.load(open(wiki, encoding="utf-8"))
    counts = Counter(w for t in texts for w in t.lower().split())
    total, vocab = sum(counts.values()), len(counts) + 1
    rows = []
    for rec in records(wet):
        head, _, body = rec.partition("\r\n\r\n")
        if "WARC-Type: conversion" not in head:
            continue
        lang = re.search(r"WARC-Identified-Content-Language: (\S+)", head)
        if not lang or lang.group(1) != "eng":
            continue
        words = re.findall(r"\S+", body)
        if len(words) < 50:
            continue
        mean_len = sum(map(len, words)) / len(words)
        alpha = sum(any(c.isalpha() for c in w) for w in words) / len(words)
        nll = -sum(math.log((counts.get(w.lower(), 0) + 0.5) / (total + 0.5 * vocab)) for w in words) / len(words)
        rows.append([len(words), round(mean_len, 3), round(alpha, 3), round(math.exp(nll))])
    meta = {"wet": "crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/wet/"
                   "CC-MAIN-20240220211055-20240221001055-00000.warc.wet.gz, bytes 0-41943039",
            "reference": f"{len(texts)} random English Wikipedia articles, {total} words",
            "columns": ["words", "mean_word_length", "alpha_word_fraction", "unigram_perplexity"]}
    json.dump({"meta": meta, "rows": rows}, open(output, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"{len(rows)} English pages with at least 50 words")


if __name__ == "__main__":
    main(*sys.argv[1:4])
