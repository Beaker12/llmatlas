# Chapter 7 local notation

Symbols used only in this chapter, beside the book-wide table in `tests/notation_global.txt` (exceptions in
`tests/notation_exceptions.txt`). Symbols are in the normal form `tools/list_symbols.py` prints. "First used" is the
section label of the page. "Rename" is what the chapter writes today and must change to the symbol in the first
column. This file is not part of the build.

| Symbol | Meaning | First used | Rename | Also means |
|---|---|---|---|---|
| $n_{acc}$ | number of micro-batches accumulated per optimizer step | sec-ch7-what-loss-averages |  |  |
| $Z$ | softmax normalizer of the output logits ($\log Z$ the log-normalizer) | sec-ch7-auxiliary-terms |  |  |
| $z_v$ | output logit of token v | sec-ch7-auxiliary-terms |  |  |
| $p_v$ | predicted probability of token v | sec-ch7-auxiliary-terms |  |  |
| $\lambda_z$ | weight of the z-loss | sec-ch7-auxiliary-terms |  |  |
| $s_{cap}$ | soft-capping bound of the logits | sec-ch7-auxiliary-terms |  |  |
| $\eta$ | learning rate ($\eta_t$ at step $t$) | sec-ch7-adamw-practice |  |  |
| $\eta_{\max}$ | peak learning rate | sec-ch7-wsd |  |  |
| $t_{dec}$ | step at which the cooldown of a warm-up-stable-decay schedule starts | sec-ch7-wsd |  |  |
| $\eta_{opt}$ | fitted optimal learning rate at a compute budget | sec-ch7-choosing-hyperparameters |  |  |
| $B_{opt}$ | fitted optimal batch size in tokens at a compute budget | sec-ch7-choosing-hyperparameters |  |  |
| $\Omega$ | iterate of Muon's Newton-Schulz orthogonalization | sec-ch7-muon |  |  |
| $R_u$ | left singular vectors of the momentum matrix ($u_t = R_u \Sigma R_v^\top$) | sec-ch7-muon |  |  |
| $R_v$ | right singular vectors of the momentum matrix | sec-ch7-muon |  |  |
| $\mathcal{L}_{\infty}$ | irreducible loss of the Chinchilla parametric fit | sec-ch7-fitting-scaling-law |  |  |
| $c_N$ | scale of the parameter term of the Chinchilla fit | sec-ch7-fitting-scaling-law |  |  |
| $c_D$ | scale of the data term of the Chinchilla fit | sec-ch7-fitting-scaling-law |  |  |
| $\gamma_N$ | exponent of the parameter term of the Chinchilla fit | sec-ch7-fitting-scaling-law |  |  |
| $\gamma_D$ | exponent of the data term of the Chinchilla fit | sec-ch7-fitting-scaling-law |  |  |
| $N_{opt}$ | compute-optimal number of parameters under the fit | sec-ch7-fitting-scaling-law |  |  |
| $D_{opt}$ | compute-optimal number of tokens under the fit | sec-ch7-fitting-scaling-law |  |  |
| $K_{opt}$ | constant of the closed-form compute-optimal size | sec-ch7-fitting-scaling-law |  |  |
| $v_{old}$ | Adam's second moment before a jump in the gradient (exercise 4) | sec-ch7-exercises |  |  |
