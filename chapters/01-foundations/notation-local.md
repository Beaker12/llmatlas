# Chapter 1 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $\mathcal{M}$ | set of masked positions in the masked-language-model objective | reserved: C1-13 |  |
| $H_{enc}$ | encoder output states, the keys and values of cross-attention | reserved: C1-13 |  |
| $n_{src}$ | source sequence length in an encoder-decoder model | reserved: C1-13 |  |
| $W_P$ | learned position-embedding matrix | reserved: C1-15 |  |
| $\mathcal{R}$ | RoPE rotation matrix ($\mathcal{R}_m$ at position $m$, block diagonal) | D2, `sec-ch1-rope` |  |
| $\varphi$ | rotation angle of one RoPE pair ($\varphi = m\omega_i$) in the derivation | D2, `sec-ch1-rope` |  |
| $\mathcal{N}$ | normal distribution | C1-03 |  |
| $\tilde{q}$ | RoPE-rotated query ($\tilde{q}_m = q_m \mathcal{R}_m^\top$) | D2, `sec-ch1-rope` |  |
| $\tilde{k}$ | RoPE-rotated key ($\tilde{k}_{m'} = k_{m'} \mathcal{R}_{m'}^\top$) | D2, `sec-ch1-rope` |  |
