# Chapter 6 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename |
|---|---|---|---|
| $f$ | fraction of tokens dispatched to expert i (f_i) | index.md:L352 |  |
| $\lambda_{aux}$ | weight of the auxiliary load-balancing loss | index.md:L352 | $\alpha$ at L352 |
| $v$ | value vector (v_i at position i) | index.md:L236 | $v_t$ at L236 (position $i$, not $t$) |
| $d_{rope}$ | per-head width of the decoupled rotary query and key in latent attention | index.md:L179 | $d_h^{R}$ at L179, L238, L539 |
| $\omega_0$ | highest RoPE frequency | index.md:L261 | $\theta_0$ at L261, L275, L293 |
| $\omega_{d_k/2-1}$ | lowest RoPE frequency | index.md:L261 | $\theta_{d/2-1}$ at L261, L293 |
