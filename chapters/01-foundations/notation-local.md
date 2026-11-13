# Chapter 1 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $\mathcal{M}$ | set of masked positions in the masked-language-model objective | C1-13, `sec-ch1-masked-lm` |  |  |
| $H_{enc}$ | encoder output states, the keys and values of cross-attention | C1-13, `sec-ch1-encoder-decoder` |  |  |
| $n_{src}$ | source sequence length in an encoder-decoder model | C1-13, `sec-ch1-encoder-decoder` |  |  |
| $\mathcal{R}$ | RoPE rotation matrix ($\mathcal{R}_m$ at position $m$, block diagonal) | D2, `sec-ch1-rope` |  |  |
| $\varphi$ | rotation angle of one RoPE pair ($\varphi = m\omega_i$) in the derivation | D2, `sec-ch1-rope` |  |  |
| $\mathcal{N}$ | normal distribution | C1-03 |  |  |
| $\tilde{q}$ | RoPE-rotated query ($\tilde{q}_m = q_m \mathcal{R}_m^\top$) | D2, `sec-ch1-rope` |  |  |
| $\tilde{k}$ | RoPE-rotated key ($\tilde{k}_{m'} = k_{m'} \mathcal{R}_{m'}^\top$) | D2, `sec-ch1-rope` |  |  |
| $Y$ | decoder states that supply the queries of cross-attention | C1-13, `sec-ch1-encoder-decoder` |  |  |
| $\mathcal{L}_{MLM}$ | masked-language-model loss, mean over the masked positions | C1-13, `sec-ch1-masked-lm` |  |  |
| $\tilde{x}$ | corrupted input sequence of the masked-language-model objective | C1-13, `sec-ch1-masked-lm` |  |  |
| $n_{ctx}$ | context length (number of learned position embeddings) | C1-13, `sec-ch1-parameter-count` |  |  |
| $a_h$ | ALiBi slope of head $h$ | fix round 1, `sec-ch1-positional-encodings` |  |  |
| $\omega_0$ | highest RoPE frequency | index.md:L743, `sec-ch1-rope` |  |  |
| $\omega_{d_k/2-1}$ | lowest RoPE frequency | index.md:L746, `sec-ch1-rope` |  |  |
| $\mathcal{R}_1$ | RoPE rotation for a position offset of 1 | index.md:L1912, `sec-ch1-exercise-9` |  |  |

