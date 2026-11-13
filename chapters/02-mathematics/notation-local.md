# Chapter 2 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
page and line at commit `4f4ee01`, or the review finding whose addition reserves the symbol. "Rename" is what the
chapter writes today and must change to the symbol in the first column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $N_c$ | Kaplan's parameter-scaling constant | index.md:L400 |  |
| $D_c$ | Kaplan's data-scaling constant | index.md:L408 |  |
| $C_c$ | Kaplan's compute-scaling constant (written $C_c^{\min}$) | index.md:L416 |  |
| $C_{\min}$ | minimum compute that reaches a given loss | index.md:L416 |  |
| $\alpha_N$ | Kaplan's parameter exponent | index.md:L400 |  |
| $\alpha_D$ | Kaplan's data exponent | index.md:L408 |  |
| $\alpha_C$ | Kaplan's compute exponent (written $\alpha_C^{\min}$) | index.md:L416 |  |
| $D_{KL}$ | Kullback-Leibler divergence | index.md:L110 |  |
| $\delta$ | Kronecker delta | index.md:L56 | the one-hot target $y$ at L58 is written $\delta_{j x_i}$ |
| $\delta_{jx_i}$ | the one-hot target: the Kronecker delta at the target token, 1 when $j = x_i$ and 0 otherwise (`tools/list_symbols.py` keeps the subscript because $x_i$ is not a running index) | index.md, sec-ch2-softmax-jacobian |  |
| $\eta$ | learning rate ($\eta_t$ at step $t$) | index.md:L172 |  |
| $\eta_{\max}$ | peak learning rate | index.md:L206 |  |
| $\eta_{\min}$ | final learning rate | index.md:L217 |  |
| $\hat{m}$ | Adam's bias-corrected first moment | index.md:L192 |  |
| $\hat{v}$ | Adam's bias-corrected second moment | index.md:L192 |  |
| $\hat{x}$ | LayerNorm's normalized input | index.md:L325 |  |
| $\mathbb{E}$ | expectation | index.md:L82 |  |
| $\mathcal{N}$ | normal distribution | index.md:L529 |  |
| $g_{clipped}$ | clipped gradient | index.md:L223 |  |
| $n_{acc}$ | number of micro-batches accumulated per optimizer step | index.md, sec-ch2-gradient-accumulation |  |
| $z_{\max}$ | largest logit, the shift in log-sum-exp | index.md, sec-ch2-log-sum-exp |  |
| $\hat{g}$ | mini-batch gradient estimate | index.md, sec-ch2-minibatch-noise |  |
| $\sigma_g$ | standard deviation of a per-example gradient (mini-batch noise) | index.md, sec-ch2-minibatch-noise |  |
| $n_{rank}$ | rank kept by a low-rank approximation | index.md, sec-ch2-svd-low-rank |  |
| $\theta_x$ | first of the two parameters of the Rosenbrock function (exercise 2) | index.md, sec-ch2-exercises |  |
| $\theta_y$ | second of the two parameters of the Rosenbrock function (exercise 2) | index.md, sec-ch2-exercises |  |
| $z_{ff}$ | pre-activation of the feed-forward hidden layer | index.md, sec-ch2-ffn-embedding-head-gradients |  |
| $x_i$ | feature $i$ of a layer's input row vector $x$ (LayerNorm and RMSNorm, sec-ch2-layer-normalization-mathematics and sec-ch2-epsilon-placement; the book-wide $x_i$ is the token at position $i$, and the page says which is meant) | index.md, sec-ch2-layer-normalization-mathematics |  |
| $\mathrm{X}$ | a random variable | index.md:L118 | $X$ at L118 (and at L96, where $H(X)$ becomes $H(p)$) |
| $\mathrm{Y}$ | a second random variable | index.md:L118 | $Y$ at L118 |

The generic activation in the feed-forward gradients (C2-12) is the operator $\operatorname{act}(\cdot)$, not a symbol. Jacobians are written $\partial y / \partial x$; $J$ is chapter 4's Jaccard similarity.
