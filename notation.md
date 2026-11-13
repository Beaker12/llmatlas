(sec-notation)=
# Notation

This page collects the symbols the book uses. The first table holds the symbols every chapter shares, with one
meaning each. A few letters knowingly carry two meanings; the second table lists them and says why the two never
meet in one formula. The chapter sections after that list the symbols that only one chapter uses, each with a link
to the section where it is introduced.

Symbols are given in a normal form: a subscript that is only a running index ($i$, $j$, $t$, ...) is dropped, so
$\eta$ stands for $\eta_t$ and $d_k$ keeps its $k$ because it is a name, not an index. Vectors are rows, and a
layer computes $xW$.

## Symbols used throughout the book

| Symbol | Meaning |
|---|---|
| $d$ | model (residual) width |
| $L$ | number of layers |
| $L_{enc}$ | number of encoder layers |
| $L_{dec}$ | number of decoder layers |
| $n_h$ | number of attention heads |
| $n_{kv}$ | number of key/value heads |
| $d_k$ | per-head query and key width |
| $d_v$ | per-head value width (equal to $d_k$) |
| $d_{attn}$ | attention width $n_h$ $d_k$ |
| $n$ | sequence or context length ($n'$ the target context length) |
| $n_{train}$ | trained context length |
| $m$ | position in a positional encoding or relative offset ($m'$ a second position) |
| $m$ | Adam's first moment ($m_t$) |
| $q$ | query vector ($q_m$ at position $m$) |
| $q$ | a distribution (generic pair $p$, $q$) |
| $k$ | key vector ($k_{m'}$ at position $m'$) |
| $n_{win}$ | sliding-window span |
| $B$ | batch size |
| $\mathcal{B}$ | a mini-batch ($\mathcal{B}_t$ at step $t$) |
| $\mathcal{V}$ | vocabulary ($\vert \mathcal{V}\vert $ its size) |
| $T$ | number of tokens in a loss, perplexity sum or batch |
| $\tau$ | temperature (sampling, source mixing, YaRN) |
| $g_{\max}$ | gradient-clipping threshold |
| $N$ | number of parameters (non-embedding where Kaplan's $N$ is meant) |
| $D$ | number of training tokens |
| $C$ | training compute |
| $\theta$ | model parameters as a vector |
| $\mathcal{L}$ | loss, mean per token, in nats |
| $h$ | hidden state ($h^{(l)}_i$ at layer $l$, position $i$) |
| $p_{data}$ | data distribution |
| $p_{\theta}$ | model distribution |
| $p$ | a distribution (generic pair $p$, $q$) |
| $p$ | sampling probability of source $s$ ($p_s$) |
| $Q$ | query matrix |
| $K$ | key matrix |
| $V$ | value matrix |
| $S$ | attention scores |
| $A$ | attention weights |
| $O$ | attention output |
| $W_Q$ | query projection matrix |
| $W_K$ | key projection matrix |
| $W_V$ | value projection matrix |
| $W_O$ | attention output projection matrix |
| $g$ | gradient vector (the upstream gradient in a backward pass; $g_t$ the loss gradient at step $t$) |
| $G$ | upstream gradient matrix |
| $\tilde{g}$ | LayerNorm's scaled upstream gradient |
| $W_{gate}$ | SwiGLU gate matrix |
| $W_{up}$ | feed-forward up-projection matrix |
| $W_{down}$ | feed-forward down-projection matrix |
| $d_{ff}$ | feed-forward hidden width |
| $W$ | weight matrix (generic; a label in its subscript or superscript names it) |
| $W_E$ | embedding matrix |
| $W_{out}$ | unembedding matrix |
| $E$ | number of experts |
| $\mathrm{i}$ | imaginary unit (the complex conjugate is a superscript star, $k^{*}$, never an overline: the bar accent is a batch mean) |
| $H$ | entropy $H(p)$ and cross-entropy $H(p, q)$ |
| $U$ | left singular vectors ($U \Sigma R^\top$) |
| $U$ | unique tokens of source $s$ ($U_s$) |
| $\Sigma$ | singular values (diagonal, $\Sigma_{jj}$) |
| $R$ | right singular vectors ($U \Sigma R^\top$) |
| $\epsilon_{ls}$ | label-smoothing mass |
| $\epsilon$ | numerical-stability constant |
| $a_N$ | compute-optimal parameter exponent ($N^*$ $\propto$ $C^{a_N}$) |
| $a_D$ | compute-optimal data exponent ($D^*$ $\propto$ $C^{a_D}$) |
| $\zeta$ | loss-scaling factor (mixed precision) |
| $w$ | given weight of source $s$ ($w_s$) |
| $e$ | epochs of source $s$ ($e_s$) |
| $e$ | Euler's number |
| $b$ | RoPE base ($b'$ a rescaled base) |
| $\omega$ | RoPE frequency ($\omega_i$ = $b^{-2i/d_k}$) |
| $s_{ctx}$ | context scale factor $n'/n_{train}$ |
| $r$ | router logit ($r_i$ for expert $i$) |
| $\rho$ | router probability |
| $\tilde{\rho}$ | router gate weight |
| $\bar{\rho}$ | batch-mean router probability |
| $n_{act}$ | experts active per token |
| $\kappa$ | expert capacity |
| $\phi$ | capacity factor |
| $\Phi$ | kernel feature map |
| $c$ | multi-head latent attention latent ($c_i$ at position $i$) |
| $d_c$ | multi-head latent attention latent width |
| $\lambda$ | weight decay |
| $\lambda_{\max}$ | largest Hessian eigenvalue |
| $\mu$ | normalization mean |
| $\sigma$ | normalization standard deviation |
| $\gamma$ | normalization gain |
| $\beta$ | normalization shift |
| $\beta_{mom}$ | momentum coefficient |
| $u$ | momentum buffer ($u_t$) |
| $\beta_1$ | Adam's first-moment coefficient |
| $\beta_2$ | Adam's second-moment coefficient |
| $v$ | Adam's second moment ($v_t$) |
| $t$ | optimizer or training step |
| $t_{warm}$ | warm-up steps |
| $t_{end}$ | total steps |
| $d_{in}$ | fan-in of a weight matrix |
| $d_{out}$ | fan-out of a weight matrix |
| $\mathcal{O}$ | asymptotic order |
| $k_{top}$ | top-k cutoff |
| $p_{top}$ | nucleus (top-p) cutoff |
| $\ell$ | YaRN wavelength ($\ell_i$ = $2\pi/\omega_i$) |
| $\xi$ | YaRN ratio ($\xi_i$ = $n_{train}/\ell_i$) |
| $\psi$ | YaRN ramp function |
| $\beta_{fast}$ | YaRN ramp bound $beta_fast$ |
| $\beta_{slow}$ | YaRN ramp bound $beta_slow$ |
| $i$ | running index (range stated where used) |
| $j$ | running index (range stated where used) |
| $l$ | layer index |
| $s$ | source index |
| $x$ | token ($x_i$ at position $i$, $x_{<i}$ the prefix) |
| $x$ | input row vector of a layer ($xW$) |
| $x_1$ | first token of a sequence |
| $x_2$ | second token of a sequence |
| $x_n$ | last token of a sequence of length n |
| $X$ | input matrix, one row per position |
| $F$ | residual-branch (sublayer) function |
| $y$ | output of a layer (LayerNorm, MoE layer) |
| $z$ | logits |
| $I$ | identity matrix |
| $\mathbb{R}$ | the real numbers |
| $\pi$ | the circle constant |
| $n_{bytes}$ | number of bytes in a text (bits per byte) |

## Letters with two meanings

| Symbol | Meanings | Why both are kept |
|---|---|---|
| $m$ | position in a positional encoding or relative offset ($m'$ a second position); Adam's first moment ($m_t$) | The plan's index rule: Adam's $m_t$ is a named quantity, always subscripted with the step, and never meets a positional encoding in the same formula. |
| $v$ | Adam's second moment ($v_t$); value vector ($v_i$ at position $i$) | Adam's $v_t$ is a named quantity (the plan's index rule); the value vector appears only in chapter 6's latent-attention paragraph, where no optimizer appears. |
| $p$ | a distribution (generic pair $p$, $q$); sampling probability of source $s$ ($p_s$) | $p_s$ is always subscripted with the source and is a probability over sources, the standard symbol in data-mixing papers. |
| $q$ | query vector ($q_m$ at position $m$); a distribution (generic pair $p$, $q$) | Both are standard; the query vector always carries a position subscript and the distribution an argument $q(x)$, and they never share a formula. |
| $U$ | left singular vectors ($U \Sigma R^\top$); unique tokens of source $s$ ($U_s$) | The SVD lives in chapter 2's primer; $U_s$ always carries its source subscript and lives in chapters 3 and 5. |
| $e$ | epochs of source $s$ ($e_s$); Euler's number | Euler's number is universal; $e_s$ always carries its source subscript. |
| $x$ | token ($x_i$ at position $i$, $x_{<i}$ the prefix); input row vector of a layer ($xW$) | Both are universal; a token is indexed by position inside a probability, a layer input appears as a function argument or in xW. |
| $b$ | RoPE base ($b'$ a rescaled base); number of LSH bands | Settled by the plan for chapter 4 (LSH bands $b$, rows $r$ are the standard symbols); chapter 4 has no RoPE. |
| $r$ | router logit ($r_i$ for expert $i$); rows per LSH band | Settled by the plan for chapter 4; chapter 4 has no router. |
| $h$ | hidden state ($h^{(l)}_i$ at layer $l$, position $i$); MinHash hash function ($h_i$ the $i$-th) | The MinHash literature writes $h$ for the hash function; chapter 4 has no hidden state in its math, and the hash function always takes an argument. |
| $n$ | sequence or context length ($n'$ the target context length); order of an n-gram | "n-gram" is the universal name; chapter 4 has no sequence length in its math once its string length is written $T_{idx}$. |

## Symbols used in one chapter

### Chapter 1: LLM Foundations

| Symbol | Meaning | Introduced in |
|---|---|---|
| $\mathcal{M}$ | set of masked positions in the masked-language-model objective | {ref}`sec-ch1-masked-lm` |
| $H_{enc}$ | encoder output states, the keys and values of cross-attention | {ref}`sec-ch1-encoder-decoder` |
| $n_{src}$ | source sequence length in an encoder-decoder model | {ref}`sec-ch1-encoder-decoder` |
| $\mathcal{R}$ | RoPE rotation matrix ($\mathcal{R}_m$ at position $m$, block diagonal) | {ref}`sec-ch1-rope` |
| $\varphi$ | rotation angle of one RoPE pair ($\varphi = m\omega_i$) in the derivation | {ref}`sec-ch1-rope` |
| $\mathcal{N}$ | normal distribution | {ref}`sec-ch1-embedding-initialization-strategies` |
| $\tilde{q}$ | RoPE-rotated query ($\tilde{q}_m = q_m \mathcal{R}_m^\top$) | {ref}`sec-ch1-rope` |
| $\tilde{k}$ | RoPE-rotated key ($\tilde{k}_{m'} = k_{m'} \mathcal{R}_{m'}^\top$) | {ref}`sec-ch1-rope` |
| $Y$ | decoder states that supply the queries of cross-attention | {ref}`sec-ch1-encoder-decoder` |
| $\mathcal{L}_{MLM}$ | masked-language-model loss, mean over the masked positions | {ref}`sec-ch1-masked-lm` |
| $\tilde{x}$ | corrupted input sequence of the masked-language-model objective | {ref}`sec-ch1-masked-lm` |
| $n_{ctx}$ | context length (number of learned position embeddings) | {ref}`sec-ch1-parameter-count` |
| $a_h$ | ALiBi slope of head $h$ | {ref}`sec-ch1-positional-encodings` |
| $\omega_0$ | highest RoPE frequency | {ref}`sec-ch1-rope` |
| $\omega_{d_k/2-1}$ | lowest RoPE frequency | {ref}`sec-ch1-rope` |
| $\mathcal{R}_1$ | RoPE rotation for a position offset of 1 | {ref}`sec-ch1-exercise-9` |

### Chapter 2: Mathematical Foundations

| Symbol | Meaning | Introduced in |
|---|---|---|
| $N_c$ | Kaplan's parameter-scaling constant | {ref}`sec-ch2-parameter-scaling` |
| $D_c$ | Kaplan's data-scaling constant | {ref}`sec-ch2-data-scaling` |
| $C_c$ | Kaplan's compute-scaling constant (written $C_c^{\min}$) | {ref}`sec-ch2-compute-scaling` |
| $C_{\min}$ | minimum compute that reaches a given loss | {ref}`sec-ch2-compute-scaling` |
| $\alpha_N$ | Kaplan's parameter exponent | {ref}`sec-ch2-parameter-scaling` |
| $\alpha_D$ | Kaplan's data exponent | {ref}`sec-ch2-data-scaling` |
| $\alpha_C$ | Kaplan's compute exponent (written $\alpha_C^{\min}$) | {ref}`sec-ch2-compute-scaling` |
| $D_{KL}$ | Kullback-Leibler divergence | {ref}`sec-ch2-cross-entropy-kl-divergence` |
| $\delta$ | Kronecker delta | {ref}`sec-ch2-softmax-jacobian` |
| $\delta_{jx_i}$ | the one-hot target: the Kronecker delta at the target token, 1 when $j = x_i$ and 0 otherwise (`tools/list_symbols.py` keeps the subscript because $x_i$ is not a running index) | {ref}`sec-ch2-softmax-jacobian` |
| $\eta$ | learning rate ($\eta_t$ at step $t$) | {ref}`sec-ch2-learning-objectives` |
| $\eta_{\max}$ | peak learning rate | {ref}`sec-ch2-linear-warmup` |
| $\eta_{\min}$ | final learning rate | {ref}`sec-ch2-cosine-annealing` |
| $\hat{m}$ | Adam's bias-corrected first moment | {ref}`sec-ch2-adam-optimizer` |
| $\hat{v}$ | Adam's bias-corrected second moment | {ref}`sec-ch2-adam-optimizer` |
| $\hat{x}$ | LayerNorm's normalized input | {ref}`sec-ch2-layer-normalization-mathematics` |
| $\mathbb{E}$ | expectation | {ref}`sec-ch2-cross-entropy-loss` |
| $\mathcal{N}$ | normal distribution | {ref}`sec-ch2-xavier-glorot-initialization` |
| $g_{clipped}$ | clipped gradient | {ref}`sec-ch2-gradient-clipping` |
| $n_{acc}$ | number of micro-batches accumulated per optimizer step | {ref}`sec-ch2-gradient-accumulation` |
| $z_{\max}$ | largest logit, the shift in log-sum-exp | {ref}`sec-ch2-log-sum-exp` |
| $\hat{g}$ | mini-batch gradient estimate | {ref}`sec-ch2-minibatch-noise` |
| $\sigma_g$ | standard deviation of a per-example gradient (mini-batch noise) | {ref}`sec-ch2-minibatch-noise` |
| $n_{rank}$ | rank kept by a low-rank approximation | {ref}`sec-ch2-svd-low-rank` |
| $\theta_x$ | first of the two parameters of the Rosenbrock function (exercise 2) | {ref}`sec-ch2-exercises` |
| $\theta_y$ | second of the two parameters of the Rosenbrock function (exercise 2) | {ref}`sec-ch2-exercises` |
| $z_{ff}$ | pre-activation of the feed-forward hidden layer | {ref}`sec-ch2-ffn-embedding-head-gradients` |
| $x_i$ | feature $i$ of a layer's input row vector $x$ (LayerNorm and RMSNorm, {ref}`sec-ch2-layer-normalization-mathematics` and {ref}`sec-ch2-epsilon-placement`; the book-wide $x_i$ is the token at position $i$, and the page says which is meant) | {ref}`sec-ch2-layer-normalization-mathematics` |
| $\mathrm{X}$ | a random variable | {ref}`sec-ch2-mutual-information` |
| $\mathrm{Y}$ | a second random variable | {ref}`sec-ch2-mutual-information` |

### Chapter 3: Data Acquisition and Sources

| Symbol | Meaning | Introduced in |
|---|---|---|
| $e_{\max}$ | largest number of epochs allowed for any source | {ref}`sec-ch3-chapter-overview` |
| $n_{shard}$ | number of shards a corpus is stored in (`S` in the erasure fence of 5.5) | {ref}`sec-ch3-personal-data-sensitive-sources` |
| $n_{match}$ | number of documents an erasure request matches (`k` in the same fence) | {ref}`sec-ch3-personal-data-sensitive-sources` |

### Chapter 4: Data Preprocessing and Cleaning

| Symbol | Meaning | Introduced in |
|---|---|---|
| $J$ | Jaccard similarity of two shingle sets | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $\mathcal{X}$ | shingle set of the first document | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $\mathcal{Y}$ | shingle set of the second document | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $h$ | MinHash hash function ($h_i$ the $i$-th) | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $n_{hash}$ | MinHash signature length (number of MinHash hash functions) | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $b$ | number of LSH bands | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $r$ | rows per LSH band | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $a$ | coefficients of hash function $h_i$, written $a_{i,1}$ and $a_{i,0}$ | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $P$ | prime modulus of the hash family | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $\chi$ | integer id of a shingle, the input of a hash function | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $\alpha$ | additive-smoothing constant | {ref}`sec-ch4-perplexity-based-filtering` |
| $T_{ref}$ | tokens in the reference corpus of the n-gram model | {ref}`sec-ch4-perplexity-based-filtering` |
| $\hat{p}$ | quality classifier's score of a document | {ref}`sec-ch4-classifier-based-filtering` |
| $n$ | order of an n-gram | {ref}`sec-ch4-learning-objectives` |
| $x_T$ | last token of a $T$-token text | {ref}`sec-ch4-perplexity-based-filtering` |
| $T_{idx}$ | tokens indexed by a suffix array | {ref}`sec-ch4-substring-deduplication-suffix-arrays` |
| $n_{items}$ | items inserted into a Bloom filter | {ref}`sec-ch4-exact-deduplication` |
| $n_{bits}$ | bits in a Bloom filter's array | {ref}`sec-ch4-exact-deduplication` |
| $n_{probe}$ | hash functions of a Bloom filter (bits set per inserted item) | {ref}`sec-ch4-exact-deduplication` |
| $n_{fp}$ | bits in a SimHash fingerprint | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $d_{Ham}$ | Hamming distance between two SimHash fingerprints | {ref}`sec-ch4-near-duplicate-detection-minhash-lsh` |
| $\epsilon_{dup}$ | SemDeDup's distance margin: documents at cosine similarity at least $1 - \epsilon_{dup}$ are duplicates | {ref}`sec-ch4-semantic-deduplication` |

### Chapter 5: Dataset Management and Versioning

| Symbol | Meaning | Introduced in |
|---|---|---|
| $M$ | document mask ($M_{ij} = 1$ when position $i$ may attend to position $j$) | {ref}`sec-ch5-cross-document-attention` |
| $n_{ranks}$ | world size (number of data-parallel ranks) | {ref}`sec-ch5-sharding-across-ranks` |
| $\Pi$ | keyed bijection on example indices that fixes the within-epoch order | {ref}`sec-ch5-determinism-resumption` |
| $n_{ex}$ | number of examples in an epoch | {ref}`sec-ch5-determinism-resumption` |
| $n_{loader}$ | DataLoader worker processes per rank | {ref}`sec-ch5-loader-workers` |

### Chapter 6: Modern Transformer Architectures

| Symbol | Meaning | Introduced in |
|---|---|---|
| $f$ | fraction of tokens dispatched to expert i ($f_i$) | {ref}`sec-ch6-keeping-experts-balanced` |
| $\lambda_{aux}$ | weight of the auxiliary load-balancing loss | {ref}`sec-ch6-keeping-experts-balanced` |
| $v$ | value vector ($v_i$ at position $i$) | {ref}`sec-ch6-multi-head-latent-attention` |
| $d_{rope}$ | per-head width of the decoupled rotary query and key in latent attention | {ref}`sec-ch6-inference-drives-design` |
| $\omega_0$ | highest RoPE frequency | {ref}`sec-ch6-rope-one-paragraph` |
| $\omega_{d_k/2-1}$ | lowest RoPE frequency | {ref}`sec-ch6-rope-one-paragraph` |
| $\hat{q}$ | pair i of a query read as a complex number ($\hat{q}_{m,i}$ at position $m$; $\hat{k}$ likewise for a key) | {ref}`sec-ch6-rope-one-paragraph` |
| $\hat{k}$ | pair i of a key read as a complex number ($\hat{k}_{m',i}$ at position $m'$) | {ref}`sec-ch6-rope-one-paragraph` |
| $\rho$ | router probability (book-wide; $\rho_i = \mathrm{softmax}(r)_i$); also DeepSeek-V3's sigmoid router score $\rho_i = \sigma(r_i)$ in 6.3, stated there as a score that is not a probability | {ref}`sec-ch6-moe-idea` |
