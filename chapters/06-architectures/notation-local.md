# Chapter 6 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $\gamma_q$ | largest gain of the query RMSNorm (bounds a normalized logit) | index.md:L235 |  |
| $\gamma_k$ | largest gain of the key RMSNorm (bounds a normalized logit) | index.md:L235 |  |
| $f$ | fraction of tokens dispatched to expert i (f_i) | index.md:L352 |  |
| $\lambda_{aux}$ | weight of the auxiliary load-balancing loss | index.md:L352 |  |
| $v$ | value vector (v_i at position i) | index.md:L236 |  |
| $d_{rope}$ | per-head width of the decoupled rotary query and key in latent attention | index.md:L179 |  |
| $\omega_0$ | highest RoPE frequency | index.md:L261 |  |
| $\omega_{d_k/2-1}$ | lowest RoPE frequency | index.md:L261 |  |
| $\hat{q}$ | pair i of a query read as a complex number ($\hat{q}_{m,i}$ at position m; $\hat{k}$ likewise for a key) | ch6 acceptance item 10 |  |
| $\hat{k}$ | pair i of a key read as a complex number ($\hat{k}_{m',i}$ at position m') | ch6 acceptance item 10 |  |
| $\rho$ | router probability (book-wide; $\rho_i = \mathrm{softmax}(r)_i$) | index.md 6.1 |  | DeepSeek-V3's sigmoid router score $\rho_i = \sigma(r_i)$ in 6.3, stated there as a score that is not a probability |

In the complex-form RoPE proof (C6-12a) the complex conjugate is a superscript star, $k^{*}$, which is not a symbol ($k^{*}$ lists as $k$); never an overline, because the bar accent already means a batch mean ($\bar{\rho}$).
