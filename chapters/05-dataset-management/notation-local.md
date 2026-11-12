# Chapter 5 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $M$ | document mask ($M_{ij} = 1$ when position $i$ may attend to position $j$) | index.md:L385 |  |
| $n_{ranks}$ | world size (number of data-parallel ranks) | index.md:L532 | $W$ at L532, L534, L535 |
| $\Pi$ | keyed bijection on example indices that fixes the within-epoch order | reserved: C5-08 |  |
| $n_{ex}$ | number of examples in an epoch | reserved: C5-08 |  |
| $n_{loader}$ | DataLoader worker processes per rank | reserved: C5-11 |  |

The per-source count of documents drawn by step $t$ (C5-08) is the operator $\operatorname{drawn}_s(t)$, not a symbol.
