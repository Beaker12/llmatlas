# Chapter 4 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename |
|---|---|---|---|
| $J$ | Jaccard similarity of two shingle sets | index.md:L607 | $s$ at L615, L617, L1073 |
| $\mathcal{X}$ | shingle set of the first document | index.md:L607 | $A$ at L607-L613 |
| $\mathcal{Y}$ | shingle set of the second document | index.md:L607 | $B$ at L607-L613 |
| $h$ | MinHash hash function (h_i the i-th) | index.md:L609 |  |
| $n_{hash}$ | number of MinHash hash functions (signature length) | index.md:L613 | $k$ at L613, L615 |
| $b$ | number of LSH bands | index.md:L615 |  |
| $r$ | rows per LSH band | index.md:L615 |  |
| $a$ | coefficients of hash function $h_i$, written $a_{i,1}$ and $a_{i,0}$ | index.md:L621 | $a_i$, $b_i$ at L621 |
| $P$ | prime modulus of the hash family | index.md:L621 | $p$ at L621 |
| $\nu$ | integer id of a shingle, the input of a hash function | index.md:L621 | $x$ at L621 |
| $\alpha$ | additive-smoothing constant | index.md:L466 |  |
| $T_{ref}$ | tokens in the reference corpus of the n-gram model | index.md:L466 | $C$ at L466 |
| $\hat{p}$ | quality classifier's score of a document | index.md:L549 | $s$ at L549 |
| $n$ | order of an n-gram | index.md:L378 |  |
| $x_T$ | last token of a $T$-token text | index.md:L462 | $w_N$ at L462 |
| $T_{idx}$ | tokens indexed by a suffix array | index.md:L722 | $N$ at L722 |
| $n_{items}$ | items inserted into a Bloom filter | reserved: C4-22 |  |
| $n_{bits}$ | bits in a Bloom filter or a SimHash fingerprint | reserved: C4-22 |  |
| $d_{Ham}$ | Hamming distance between two SimHash fingerprints | reserved: C4-22 |  |
