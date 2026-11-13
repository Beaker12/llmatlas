# Chapter 3 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
first place the symbol appears on the page (by section, since line numbers move), or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $e_{\max}$ | largest number of epochs allowed for any source | index.md:L31 (figure caption); first in display math at the 8.2 equation | | |
| $n_{shard}$ | number of shards a corpus is stored in (`S` in the erasure fence of 5.5) | index.md, 5.5 erasure paragraph after the fence | | |
| $n_{match}$ | number of documents an erasure request matches (`k` in the same fence) | index.md, 5.5 erasure paragraph after the fence | | |
| $\bar{q}$ | the DoReMi domain weights $q_t$ averaged over the $T$ proxy steps, used for the large run | index.md, 8.3 display equation | | |
| $\nu$ | DoReMi step size on the domain weights (`NU` in its listing) | index.md, 8.3 display equation | | |
| $\Delta$ | DoReMi excess loss of a domain ($\Delta_t[i]$ at proxy step $t$) | index.md, 8.3 display equation | | |
