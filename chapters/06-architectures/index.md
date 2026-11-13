# Modern Transformer Architectures

Chapter 1 built the transformer as it was drawn in 2017, and listed in one section what changed ({ref}`sec-ch1-modern-block`). This chapter takes that list apart. A decoder-only model released in 2024 or 2025 is still recognizably the 2017 block, but nearly every part of it has been replaced for a reason: to keep a 70-billion-parameter run from diverging, to shrink the cache that dominates serving memory, to stretch the context window, or to buy more parameters per unit of compute. We go through those replacements one at a time, put numbers on each from the published configuration files of real models, and finish with the accounting that turns a configuration into parameters, FLOPs and bytes.

(sec-ch6-chapter-overview)=

## Chapter Overview

Read a modern model's `config.json` as a list of decisions. Each key is the answer to a question that the 2017 paper left open or answered differently.

**Normalization and the residual stream** decide whether a very deep stack trains at all. Normalizing inside the residual branch rather than after the addition keeps a clean path for the gradient; normalizing queries and keys, or normalizing the output of each sublayer, removes the instabilities that appear when activations grow.

**The feed-forward network** holds most of the parameters. Gated activations replaced the plain ReLU or GELU layer, and the hidden width is chosen so that three matrices cost about what two used to.

**Attention variants** are mostly about inference. The keys and values of every previous token are kept in memory while generating, and the number of key/value heads, or the size of a compressed latent, sets that memory.

**Positions and long context** are handled by rotating queries and keys. Making a model work at four times its training length is a matter of changing the rotation frequencies, then training a little at the new length.

**Mixture of experts** replaces one feed-forward network by many and sends each token to a few, so parameters grow without the compute per token growing with them.

**Accounting** is the discipline that keeps all of this honest: a function from a configuration to parameters, FLOPs and bytes, checked against the real checkpoints.

The theme throughout is that architecture choices are constraints from somewhere else in the system. The head count is chosen for the cache, the expert count for the interconnect, the norm placement for the learning rate. A model table is therefore worth reading column by column.

Every symbol the book uses is listed on the {ref}`notation page <sec-notation>`, with a link to the section that introduces each symbol used only in this chapter.

(sec-ch6-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- How the 2025 decoder block differs from the 2017 block: pre-norm, RMSNorm, post-sublayer and QK normalization, gated feed-forward networks, no biases
- How depth and width are chosen across a model family, and what the scaling-law evidence says about the shape of a model at fixed size
- How multi-head, multi-query, grouped-query and multi-head latent attention trade quality against the size of the KV cache, and how to compute that size for any configuration
- How sliding-window and interleaved local/global attention bound the cost of long sequences
- How rotary embeddings are extended to longer contexts by interpolation, NTK-aware scaling and YaRN, and what a context-length figure does and does not promise
- How mixture-of-experts layers route tokens, why they need a load-balancing mechanism, and the difference between total and active parameters
- How to count the parameters of a model from its `config.json`, exactly, why published parameter figures use different conventions, and how to turn the count into FLOPs per token and inference memory
- Where encoder and encoder-decoder models are still the right choice, and how their parameter counts differ from a decoder's

(sec-ch6-hands-on-learning)=

## Hands-On Learning

The accompanying notebook, [Modern Architectures Demo](examples/architecture_demo.ipynb), builds a small LLaMA-style model and checks it against independent references:

- Reading the downloadable configurations and those transcribed from papers, and reproducing the stored element counts of all 8 downloadable checkpoints and the hub totals of the gated ones exactly ({ref}`nb-arch-configs`)
- RMSNorm, grouped-query attention and a pre-norm block, each checked against a reference implementation ({ref}`nb-arch-block`)
- A KV cache whose incremental decoding matches the full forward pass, and a simplified latent cache ({ref}`nb-arch-kvcache`)
- RoPE, position interpolation and NTK-aware scaling, with the frequency identities asserted ({ref}`nb-arch-rope`)
- A top-2 mixture-of-experts router, its load-balancing loss and capacity dropping ({ref}`nb-arch-moe`)
- A seeded comparison of post-norm and pre-norm training on a small text ({ref}`nb-arch-train`)

Everything runs on a CPU. The configuration cell reads the model files from the Hugging Face hub when it can and from committed copies when it cannot, so the notebook gives the same results offline.

(sec-ch6-block-2017-2025)=

## 1. From the 2017 Block to the 2025 Block

(sec-ch6-models-this-chapter)=

### 1.1 The models in this chapter

Twelve models are used as running examples, among them the open mixture-of-experts model OLMoE {cite}`muennighoff2025olmoe` and two from 2025, Qwen3 {cite}`qwen2025qwen3` and Gemma 3 {cite}`gemma2025gemma3`. They span dense and sparse designs, three attention families and six organizations. Every figure below traces to a pinned `config.json` or to the model's paper, with the paper-derived rows expressed in the conventions of the Hugging Face configurations (for example vocabulary entries, and the width of one gate or up projection).

All twelve have the same outer structure: a token embedding, a stack of identical blocks, a final normalization, and an output projection to the vocabulary. Within a block, an attention sublayer and then a feed-forward sublayer each add their output to the residual stream. What differs is the contents of the boxes and where the normalizations sit: before each sublayer in most of the twelve, after it in OLMo 2, and on both sides in Gemma 2 and Gemma 3 (the design table below and {ref}`sec-ch6-normalizing-sublayer-outputs`). {numref}`fig-ch6-block-2017-2025` sets the two ends of that history side by side.

:::{figure} figures/block-2017-2025.*
:name: fig-ch6-block-2017-2025
:alt: Two block diagrams side by side. Left, the original transformer's block (2017): multi-head attention, add, LayerNorm, ReLU feed-forward network, add, LayerNorm, with the residual skips entering each add. Right, the Llama 3.1 8B block (2024): RMSNorm, grouped-query attention with RoPE, add, RMSNorm, SwiGLU network, add, with the residual skips bypassing the norms.
:width: 100%

The original transformer's block (2017) and Llama 3.1 8B's block (2024). The outer
structure is the same; what changed is the contents of each box (shaded blue) and where
the normalization sits (orange). On the right, the residual path (orange arrows) runs
from block to block without passing through a norm.
:::

**Dense models.**

```{tabularcolumns} \Y{0.30}\Y{0.12}\Y{0.12}\Y{0.26}\Y{0.20}
```

| Model | Layers | $d$ | Query / KV heads × head dim | FFN width |
| --- | --- | --- | --- | --- |
| Llama 3.1 8B | 32 | 4,096 | 32 / 8 × 128 | 14,336 |
| Llama 3.1 70B | 80 | 8,192 | 64 / 8 × 128 | 28,672 |
| Mistral 7B v0.1 | 32 | 4,096 | 32 / 8 × 128 | 14,336 |
| Qwen2.5 7B | 28 | 3,584 | 28 / 4 × 128 | 18,944 |
| Gemma 2 9B | 42 | 3,584 | 16 / 8 × 256 | 14,336 |
| OLMo 2 7B (1124) | 32 | 4,096 | 32 / 32 × 128 | 11,008 |
| Qwen3 8B (base) | 36 | 4,096 | 32 / 8 × 128 | 12,288 |
| Gemma 3 12B, text decoder only | 48 | 3,840 | 16 / 8 × 256 | 15,360 |

```{tabularcolumns} \Y{0.17}\Y{0.14}\Y{0.20}\Y{0.20}\Y{0.12}\Y{0.17}
```

| Model | Vocabulary | RoPE base | Context | Parameters | Source |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B | 128,256 | 500,000 | 128K | 8.03B | paper |
| Llama 3.1 70B | 128,256 | 500,000 | 128K | 70.55B | paper |
| Mistral 7B v0.1 | 32,000 | 10,000 | 8,192 (paper), 32,768 (config) | 7.24B | config |
| Qwen2.5 7B | 152,064 | 1,000,000 | 131,072 (config; pretrained at 4,096, extended to 32,768, longer with YaRN) | 7.62B | config |
| Gemma 2 9B | 256,000 | not stated (Transformers default 10,000) | 8,192 | 9.24B | paper |
| OLMo 2 7B (1124) | 100,352 | 500,000 | 4,096 | 7.30B | config |
| Qwen3 8B (base) | 151,936 | 1,000,000 | 32,768 (config); 128K (paper) | 8.19B | config |
| Gemma 3 12B, text decoder only | 262,208 | 10,000 local, 1,000,000 global | 128K | 11.77B | paper, reference code and conversion code (transcribed) |

**Sparse (mixture-of-experts) models.**

```{tabularcolumns} \Y{0.23}\Y{0.09}\Y{0.10}\Y{0.22}\Y{0.16}\Y{0.20}
```

| Model | Layers | $d$ | Attention | Dense layers | Experts per MoE layer |
| --- | --- | --- | --- | --- | --- |
| Mixtral 8x7B v0.1 | 32 | 4,096 | GQA, 32 / 8 × 128 | 0 | 8 routed, top 2, width 14,336 |
| OLMoE 1B-7B | 16 | 2,048 | MHA, 16 × 128 | 0 | 64 routed, top 8, width 1,024 |
| DeepSeek-V2-Lite | 27 | 2,048 | MLA, 16 heads | 1 | 64 routed + 2 shared, top 6, width 1,408 |
| DeepSeek-V3 | 61 | 7,168 | MLA, 128 heads | 3 | 256 routed + 1 shared, top 8, width 2,048 |

```{tabularcolumns} \Y{0.28}\Y{0.18}\Y{0.36}\Y{0.18}
```

| Model | Total | Active | Source |
| --- | --- | --- | --- |
| Mixtral 8x7B v0.1 | 46.70B | 12.88B (with embedding) | config |
| OLMoE 1B-7B | 6.92B | 1.28B (with embedding) | config |
| DeepSeek-V2-Lite | 15.71B | 2.45B (without embedding) | config |
| DeepSeek-V3 | 671.03B | 36.63B (without embedding) | config |

A few notes on reading the tables.

- **Source.** "config" means the value is read from the model's `config.json` at the pinned commit listed in `data/pins.json`. "paper" means the architecture is taken from the model's paper, because the repository is gated (it needs an account and a license acceptance). A paper-derived row is not unverified: the hub reports the exact parameter total of a gated checkpoint in its public metadata, and the configurations transcribed from the papers reproduce those totals to the last digit ({ref}`sec-ch6-validation`). Two values in the gated rows describe the released checkpoint rather than the paper: Gemma 2's and Gemma 3's vocabulary sizes, explained in the notes on vocabulary and on Gemma 3.
- **Parameters** are exact element counts divided by $10^9$, from the checkpoints' tensor shapes rather than from rounded marketing figures. {ref}`sec-ch6-accounting` explains the "with" and "without" embedding convention for active parameters.
- **Vocabulary.** The Llama 3 paper describes a 128K vocabulary; the configuration has 128,256 entries, the 128,000 of the tokenizer plus 256 reserved special tokens. The Gemma 2 paper's Table 1 lists a vocabulary of 256,128 and its Table 2 counts 917,962,752 embedding parameters accordingly ($256{,}128 \times 3{,}584$); the released checkpoint has 256,000 rows (917,504,000 parameters), and the table uses the checkpoint's 256,000 {cite}`gemma2024gemma2`. The paper's non-embedding count for 9B, 8,324,201,984, is reproduced exactly. Likewise Gemma 2's paper lists the feed-forward width as the combined 28,672 of the gate and up projections, which is twice the 14,336 of each in the configuration and the table.
- **Context.** Mistral 7B v0.1's paper reports training at 8,192 tokens with a 4,096-token sliding window; its configuration advertises 32,768 positions. Both are quoted because they answer different questions: what the model saw, and what the code accepts ({ref}`sec-ch6-long-context-training`).
- **Gemma 3.** The 12B model is multimodal, and its row is the text decoder only: the paper's Table 1 counts a 417M-parameter vision encoder separately {cite}`gemma2025gemma3`. The paper gives the parameter counts, the attention pattern and the RoPE bases but not the layer shapes, which are taken from Google's reference implementation, the `Gemma3_12B` configuration of the `gemma` package ([`gemma/gm/nn/_gemma.py`](https://github.com/google-deepmind/gemma/blob/be0e06ff698422272ea40a82b58fd293885ec8cb/gemma/gm/nn/_gemma.py)): 48 layers, width 3,840 and a feed-forward width of 15,360. The vocabulary has four figures, listed in the table below. The conversion script for Gemma 3 ([`convert_gemma3_weights.py`](https://github.com/huggingface/transformers/blob/f4410762062c274fb53471e551e62b377df44cc1/src/transformers/models/gemma3/convert_gemma3_weights.py)) appends 64 rows to the embedding of the multimodal model (`config.vocab_size += 64`) so that the image placeholder token, id 262,144, has a row, and the Transformers configuration for Gemma 3's text model defaults to `vocab_size` 262,208 accordingly. The table of models uses 262,208, because that is what the hub's parameter total counts; {ref}`sec-ch6-validation` shows that the total then decomposes exactly. The row's shapes are still transcribed, not read from a configuration, because the repository is gated.
- **Gemma 2's head dimension.** Sixteen heads of 256 give an attention width of 4,096, larger than the model width of 3,584. The head width is a free parameter, not $d/n_h$.
- **DeepSeek-V3's total.** The 671.03B counts the main model. The released checkpoint also contains a 13.46B-element multi-token-prediction layer ({ref}`sec-ch6-multi-token-prediction`), about 1.85B of which are copies of the embedding and output head that the paper shares with the main model, and 41.5M FP8 scale factors; the [model card](https://huggingface.co/deepseek-ai/DeepSeek-V3)'s "685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights" is the sum of those two rounded parts. The stored elements come to 684.49B, or 684.53B with the scale factors.

The four vocabulary figures for Gemma 3:

```{tabularcolumns} \Y{0.55}\Y{0.25}
```

| Source of the figure | Vocabulary |
| --- | --- |
| Paper, Table 1's caption ("Our vocabulary has 256k entries") | 256k |
| Paper, text ("The resulting vocabulary has 262k entries") | 262k |
| Reference implementation: one row per tokenizer entry | 262,144 |
| Released Hugging Face checkpoint | 262,208 |

Beyond shapes, the models make different design choices, summarized here and developed in the sections that follow.

```{tabularcolumns} \Y{0.22}\Y{0.17}\Y{0.15}\Y{0.12}\Y{0.15}\Y{0.19}
```

| Model | Norm placement | QK-norm | Attention biases | Tied embeddings | Attention pattern |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 | pre | no | no | no | full causal |
| Mistral 7B v0.1 | pre | no | no | no | sliding window of 4,096 |
| Qwen2.5 | pre | no | q, k, v | no | full causal |
| Gemma 2 | pre and post | no | no | yes | alternating local (4,096) and global |
| OLMo 2 | post-sublayer | yes | no | no | full causal |
| Qwen3 | pre | yes | no | no | full causal |
| Gemma 3 | pre and post | yes | no | yes | five local (1,024) per global |
| Mixtral | pre | no | no | no | full causal (config: no window) |
| OLMoE | pre | yes | no | no | full causal |
| DeepSeek-V2/V3 | pre | on the latent ({ref}`sec-ch6-multi-head-latent-attention`) | no | no | full causal |

```{admonition} The Tables, Computed
:class: tip
The notebook section {ref}`nb-arch-configs` loads the configurations of the ungated
models, rebuilds the parameter counts, and computes the derived columns (embedding
share, feed-forward share, KV-cache bytes per token) used throughout this chapter.
```

(sec-ch6-unchanged-parts)=

### 1.2 What did not change

It is worth stating what is *constant*, because it explains why the field moved by small edits rather than redesigns. Every model above is autoregressive, uses causal self-attention, a residual stream of width $d$ that every sublayer reads from and writes to, and a learned token embedding. There are no recurrent connections, no convolutions in the main path and no cross-attention. The residual stream is the shared bus, and almost every innovation in this chapter is a change to what is plugged into it or to how the plug is normalized.

(sec-ch6-depth-width)=

### 1.3 Depth and width

The first two columns of the tables, layers $L$ and width $d$, are the shape of the model, and the ratio $d / L$ describes it. Within a family the shape follows the size: from Llama 3.1 8B to 70B the width doubles (4,096 to 8,192) and the depth grows 2.5 times (32 to 80 layers), so $d / L$ falls from 128 to 102. Across families half of the twelve models sit at exactly 128 (Llama 3.1 8B, Mistral 7B, Qwen2.5 7B, OLMo 2 7B, Mixtral and OLMoE), and three sit far lower: Gemma 2 9B at 85, Gemma 3 12B at 80 and DeepSeek-V2-Lite at 76, which have more layers for their width. The notebook computes the ratio for all twelve models ({ref}`nb-arch-configs`).

How much this choice matters was measured at fixed size. {cite:t}`kaplan2020scaling` held the non-embedding parameter count $N \approx 12 L d^2$ fixed while varying the shape, and report that "performance depends very weakly" on depth against width: the aspect ratio "can vary by a factor of 40 while only slightly impacting performance", and a model with $(L, d) = (6, 4288)$ reaches a loss within 3% of one with $(48, 1600)$ (their Figure 5). The loss varies by a few percent over the range, which is small against what a doubling of $N$ buys, so the shape is chosen for reasons other than the loss. The Gemma 2 report gives one: in an ablation at 9B parameters, "a deeper 9B network is slightly better than a wider 9B for the same number of parameters. Although the gap is small, it is consistent across benchmarks and warrants the switch to a deeper architecture" {cite}`gemma2024gemma2`.

(sec-ch6-normalization-residual-stream)=

## 2. Normalization and the Residual Stream

(sec-ch6-pre-norm-post-norm)=

### 2.1 Pre-norm and post-norm

The original transformer {cite}`vaswani2017attention` applied layer normalization *after* adding the sublayer output to the residual (post-norm): $x \leftarrow \mathrm{Norm}(x + F(x))$. Every layer's output passes through a normalization on the way to the next, which means the gradient reaching early layers is rescaled at each step. At initialization the gradients near the output are large, and {cite:t}`xiong2020layer` trace post-norm's reliance on a learning-rate warm-up to them. How strict that reliance is depends on the learning rate: in the notebook section {ref}`nb-arch-train`, post-norm fails without warm-up only at the highest of three rates, and warm-up lets it train there in some seeds but not in all.

Pre-norm moves the normalization inside the branch: $x \leftarrow x + F(\mathrm{Norm}(x))$. The residual path from the embedding to the output is then an unnormalized identity, so gradients reach every layer directly. {cite:t}`xiong2020layer` show that this keeps the gradient scale stable at initialization and makes warm-up optional. The price is that the residual stream's magnitude grows with depth, so the representation of the last layers can be dominated by the accumulated sum; this is why every pre-norm model adds a final normalization before the output projection. The notebook section {ref}`nb-arch-train` trains both arrangements side by side at three learning rates, with and without warm-up. {numref}`fig-ch6-norm-sweep` plots the recorded runs behind its statements.

:::{figure} figures/norm-sweep.*
:name: fig-ch6-norm-sweep
:alt: Two panels of training loss after 400 steps against learning rate 0.0003, 0.001 and 0.003, for pre-norm in blue and post-norm in red, five seeds each. Without warm-up post-norm is lower at the two smaller rates and jumps to about 3.1 at 0.003 while pre-norm stays near 2.1. With warm-up post-norm at 0.003 is spread from about 1.9 to 3.1 across seeds.
:width: 100%

Training loss of the notebook's six-layer model after 400 steps (the mean over steps 381
to 400), five seeds per point, from the recorded sweep
(`examples/data/norm_sweep.json`). The runs are short, so the panels compare how fast
each arrangement learns, not where it would end. Without warm-up (left), post-norm ends
below pre-norm in every seed at the two lower rates and fails in every seed at 0.003,
where pre-norm still trains. With a 100-step warm-up (right), post-norm at 0.003 trains
in four seeds and fails in one.
:::

Chapter 2 follows the gradient through the residual path and notes that where the normalization sits matters ({ref}`sec-ch2-pre-post-ln`). Post-norm is not extinct, and the question is revisited every few years, but the practical argument favors pre-norm: a diverged run on a cluster costs far more than any small quality difference at a size where both train.

(sec-ch6-rmsnorm)=

### 2.2 RMSNorm

Layer normalization subtracts the mean and divides by the standard deviation. RMSNorm drops the mean subtraction and the bias {cite}`zhang2019root`:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\lVert x \rVert^2 / d + \epsilon}}$$

The argument is that the re-scaling is what matters and the re-centering is not. It has one parameter vector instead of two and a cheaper forward and backward pass; it is what every model in the table uses. Chapter 1 lists RMSNorm and SwiGLU among the changes to the block since 2017 ({ref}`sec-ch1-modern-block`). In the parameter count, each RMSNorm contributes $d$ weights.

(sec-ch6-normalizing-queries-keys)=

### 2.3 Normalizing queries and keys

Attention logits are dot products $q \cdot k / \sqrt{d_k}$. If a few directions in $q$ and $k$ grow large during training, the softmax saturates into a one-hot distribution, gradients through it vanish, and the loss spikes. Normalizing the queries and keys before the dot product (QK-norm) was proposed for translation models by {cite:t}`henry2020query`, and {cite:t}`dehghani2023scaling` used it to stabilize a 22-billion-parameter vision transformer. It bounds the logits regardless of how the projection weights drift, and the bound follows from the definition of RMSNorm. Without the constant $\epsilon$, a normalized vector $x / \sqrt{\lVert x \rVert^2 / d}$ has norm exactly $\sqrt{d}$, and the gain $\gamma$ multiplies each coordinate. When each head's query and key (width $d_k$) is normalized with largest gains $\gamma_q$ and $\gamma_k$, the Cauchy-Schwarz inequality gives

$$\frac{|q \cdot k|}{\sqrt{d_k}} \le \frac{\gamma_q \sqrt{d_k}\; \gamma_k \sqrt{d_k}}{\sqrt{d_k}} = \gamma_q \gamma_k \sqrt{d_k},$$

which is $\sqrt{128} = 11.3$ at $d_k = 128$ and unit gains. The learned gains set the bound, and the weights of the projections do not enter it: scaling $W_Q$ and $W_K$ by 100 multiplies the unnormalized logits by $10^4$ and leaves the normalized ones unchanged ({ref}`nb-arch-block` checks both).

OLMo 2 {cite}`olmo2024olmo2` and OLMoE {cite}`muennighoff2025olmoe` (whose configuration and model code show the same layout) apply an RMSNorm to the full query and key projections before they are split into heads; in the parameter count this adds $n_h d_k$ weights for the queries and $n_{kv} d_k$ for the keys per layer. Normalizing the whole projection fixes the norm of the concatenation, $\sqrt{n_h d_k}$ for the queries, and one head's slice can hold all of it, so the per-head bound loosens to $\gamma_q \gamma_k \sqrt{n_h n_{kv} d_k}$: 362 for OLMo 2 7B ($n_h = n_{kv} = 32$, $d_k = 128$, unit gains), against 11.3 per head. Qwen3 {cite}`qwen2025qwen3` and Gemma 3 {cite}`gemma2025gemma3` normalize each head separately, with one weight vector of width $d_k$ shared by all query heads and another by all key heads, so the count is $2 d_k$ per layer. Both reports name the purpose: Qwen3 introduces QK-norm "to ensure stable training", and Gemma 3 uses it to "replace the soft-capping of Gemma 2" ({ref}`sec-ch6-sliding-window-interleaved-attention`). DeepSeek's multi-head latent attention normalizes the compressed latents instead ({ref}`sec-ch6-multi-head-latent-attention`).

(sec-ch6-normalizing-sublayer-outputs)=

### 2.4 Normalizing sublayer outputs

A second family places the normalization on the *output* of each sublayer, before it is added to the residual. OLMo 2 does this in its "reordered" arrangement, $x \leftarrow x + \mathrm{Norm}(F(x))$ for both attention and feed-forward, and combines it with QK-norm; the authors report it as one of the changes that removed the loss spikes of their first-generation recipe {cite}`olmo2024olmo2`. Gemma 2 uses both: a normalization before each sublayer and another after it, four per layer {cite}`gemma2024gemma2`, and Gemma 3 keeps this "post-norm and pre-norm with RMSNorm" {cite}`gemma2025gemma3`. The sandwich arrangement was introduced for image-token transformers in CogView, again to control instabilities {cite}`ding2021cogview`.

All of these keep the *magnitude* of what each layer adds to the residual stream bounded. That is the common thread: the stream is a running sum of terms, and the stability techniques of the last five years are ways of controlling the size of the terms. {numref}`fig-ch6-norm-placements` draws the four arrangements of this section on one sublayer.

:::{figure} figures/norm-placements.*
:name: fig-ch6-norm-placements
:alt: Four small diagrams of one sublayer F with its residual addition. Pre-norm: Norm then F then add. Post-norm: F, add, then Norm on the sum. Reordered: F, Norm, then add. Sandwich: Norm, F, Norm, then add.
:width: 100%

Where the norm sits relative to one sublayer F and its residual addition. Only post-norm
puts a norm on the residual path itself; the other three leave the path from x to the
next layer an identity (orange) and differ in whether the sublayer's input, its output,
or both are normalized.
:::

(sec-ch6-initialization-residual-branches)=

### 2.5 Initialization of the residual branches

The residual stream is a sum of $2L$ terms. If each term has unit variance at initialization, the sum's variance grows as $2L$. GPT-2 scales the weights of the residual projections by $1/\sqrt{2L}$ at initialization, with $2L$ the number of residual layers {cite}`radford2019language`, and most codebases do something equivalent for the output projections of attention and the feed-forward network. Chapter 2 states this rule ({ref}`sec-ch2-residual-init`).

```{admonition} Norm Placement in Practice
:class: note
The notebook section {ref}`nb-arch-block` checks the chapter's RMSNorm against PyTorch's
reference, and {ref}`nb-arch-train` trains a six-layer post-norm and pre-norm model from
the same seeds at three learning rates, with and without warm-up: post-norm fails at the
highest rate without warm-up, where pre-norm still trains, and after 400 steps is lower
than pre-norm in every seed at the two lower rates.
```

(sec-ch6-feed-forward-network)=

## 3. The Feed-Forward Network

(sec-ch6-relu-gated-units)=

### 3.1 From ReLU to gated units

The 2017 feed-forward network is $\mathrm{ReLU}(x W_{up})\, W_{down}$ with a hidden width of $4d$; Chapter 1 sets it beside the gated form that replaced it ({ref}`sec-ch1-modern-block`). GPT-1, GPT-2 and BERT swapped ReLU for GELU {cite}`hendrycks2016gaussian`. {cite:t}`shazeer2020glu` then tested gated variants, in which the hidden activation is the elementwise product of two projections, one passed through a nonlinearity:

$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(x W_{gate}) \odot x W_{up}\big) W_{down}$$

With a GELU gate this is GeGLU, which Gemma uses; with SiLU it is SwiGLU, which every other model in the table uses. At matched parameter count the gated forms gave consistently lower loss, and the paper's own explanation is candid: "we offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." Since then the gated forms have become the default: every one of the twelve models in this chapter uses one.

(sec-ch6-width)=

### 3.2 Width

A gated unit has three matrices instead of two, so to keep the parameter count at that of a $4d$ feed-forward network the hidden width is reduced to $\tfrac{8}{3} d$. In practice the width is rounded to a hardware-friendly multiple, and then chosen freely. OLMo 2 7B's width of 11,008 is $\tfrac{8}{3} \times 4096$ rounded up to a multiple of 256, as in the first LLaMA {cite}`touvron2023llama`. The later designs go wider: 14,336 is $3.5 d$ for Llama 3.1 and Mistral, and Qwen2.5 7B uses $5.3 d$.

Each feed-forward network has $3\,d \cdot d_{ff}$ parameters. Across the dense models above, that is between 59% (OLMo 2 7B) and 80% (Llama 3.1 70B) of all parameters; the notebook computes each share. The feed-forward network is where most of a dense model's capacity, and most of its FLOPs, live, which is also why it is the part that mixture-of-experts models replace ({ref}`sec-ch6-mixture-of-experts`).

(sec-ch6-no-biases)=

### 3.3 No biases

Linear layers in these models have no bias terms, as the model code of each shows. {cite:t}`chowdhery2023palm` give a reason: removing the biases from the dense layers and layer norms "result[ed] in increased training stability for large models". The exception in the table is Qwen2.5, which keeps biases on the query, key and value projections only; this detail is fixed in the model code rather than in `config.json`, a pattern that matters for the parameter count in {ref}`sec-ch6-accounting`. Its successor drops them: Qwen3's report says "we remove QKV-bias used in Qwen2" and adds QK-norm in their place {cite}`qwen2025qwen3`. For Qwen3 8B that removes $36 \times (32 + 2 \times 8) \times 128 = 221{,}184$ bias parameters and adds $36 \times 2 \times 128 = 9{,}216$ norm weights.

(sec-ch6-attention-variants)=

## 4. Attention Variants

(sec-ch6-inference-drives-design)=

### 4.1 Why inference drives the design

During generation, each new token attends to all previous tokens. Recomputing their keys and values at every step would cost quadratic time, so every serving system stores them: the KV cache. For one sequence the cache holds, per layer, one key vector and one value vector per key/value head for every token seen so far. It is read in full at every decoding step, so it costs both memory and bandwidth; at long context and large batch it, not the weights, sets the number of concurrent users a GPU can serve.

In elements, the cache of one token in one layer is:

| Variant | Elements per token per layer |
| --- | --- |
| Multi-head attention (MHA) | $2\, n_h\, d_k$ |
| Grouped-query attention (GQA) with $n_{kv}$ groups | $2\, n_{kv}\, d_k$ |
| Multi-query attention (MQA) | $2\, d_k$ |
| Multi-head latent attention (MLA) | $d_c + d_{rope}$ |

Multiplying by the number of layers and the bytes per element (2 for bf16, bfloat16, a 16-bit floating-point format with float32's 8 exponent bits and 7 mantissa bits) gives the bytes per token. The following function implements the table for a Hugging Face configuration dictionary.

```python
def kv_cache_elements_per_token(cfg):
    """KV-cache elements one token occupies across all layers of a decoder model."""
    layers = cfg["num_hidden_layers"]
    if cfg["model_type"].startswith("deepseek"):  # MLA: one latent and one rope key
        return layers * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"])
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    return layers * 2 * n_kv * head_dim


print(kv_cache_elements_per_token(  # Llama 3.1 8B: 32 layers, 8 KV heads of 128
    {"model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32,
     "num_key_value_heads": 8, "hidden_size": 4096}), "elements per token")
```

% output

```text
65536 elements per token
```

(sec-ch6-multi-query-grouped-query-attention)=

### 4.2 Multi-query and grouped-query attention

Multi-query attention shares a single key head and a single value head across all query heads, shrinking the cache by a factor of $n_h$ {cite}`shazeer2019fast`. It speeds up decoding substantially, at some cost in quality and in training stability. Grouped-query attention interpolates: the $n_h$ query heads are divided into $n_{kv}$ groups, each sharing one key and one value head {cite}`ainslie2023gqa`.

How much quality the cache costs has been measured in two ways, and the results do not all point the same way. {cite:t}`ainslie2023gqa` convert T5-XXL and train it further on 5% of its pretraining steps; their Table 1 reports an inference time per sample of 1.51 s and an average score of 47.2 over seven summarization, translation and question-answering datasets for multi-head attention, 0.24 s and 46.6 for multi-query attention, and 0.28 s and 47.1 for grouped-query attention with 8 groups: grouped-query attention keeps most of multi-query attention's speed-up while losing a sixth as much of the score. Their conversion mean-pools the key and value heads of each group of a trained multi-head checkpoint and then trains briefly. {cite:t}`deepseek2024v2` train three 7B dense models from scratch on 1.33T tokens, identical except for the attention (Appendix D.1, Table 8), and find larger gaps on hard benchmarks: MMLU 37.9 with multi-query, 41.2 with grouped-query (8 groups) and 45.2 with multi-head attention; "MHA demonstrates significant advantages over GQA and MQA on hard benchmarks." The same appendix (D.2, Table 9) compares latent attention with multi-head attention in mixture-of-experts models of about 16B and 250B parameters and finds that "MLA shows better performance than MHA" while needing "14% for small MoE models and 4% for large MoE models" of its cache; for the small pair, 15.6K against 110.6K elements per token. Llama 2's two largest models adopted grouped-query attention, and every Llama since has {cite}`touvron2023llama2`. {numref}`fig-ch6-head-sharing` shows the three sharing patterns for eight query heads.

:::{figure} figures/head-sharing.*
:name: fig-ch6-head-sharing
:alt: Three rows of eight query heads. In multi-head attention each query head has its own key/value head. In grouped-query attention two key/value heads each serve four query heads. In multi-query attention one key/value head serves all eight.
:width: 100%

Head sharing for eight query heads. The number of key/value heads, eight, two or one, is
what the cache stores per token per layer, $2\, n_{kv}\, d_k$ elements; the query heads
and the attention computation are the same in all three.
:::

The choice of $n_{kv}$ is a direct dial on the cache. Llama 3.1 8B uses 8 key/value heads for 32 query heads, a 4× reduction, and Llama 3.1 70B uses 8 for 64, an 8× reduction. The cache per token (bf16, all layers) is:

| Model | KV elements per token | Bytes per token | Compared with MHA |
| --- | --- | --- | --- |
| OLMo 2 7B (MHA) | 262,144 | 524,288 | 1× |
| OLMoE 1B-7B (MHA) | 65,536 | 131,072 | 1× |
| Mistral 7B, Mixtral, Llama 3.1 8B (GQA, 8 KV heads) | 65,536 | 131,072 | 4× smaller |
| Qwen2.5 7B (GQA, 4 KV heads) | 28,672 | 57,344 | 7× smaller |
| Qwen3 8B (GQA, 8 KV heads) | 73,728 | 147,456 | 4× smaller |
| Gemma 2 9B (GQA, 8 KV heads of 256) | 172,032 | 344,064 | 2× smaller |
| Gemma 3 12B (GQA, 8 KV heads of 256) | 196,608 | 393,216 | 2× smaller |
| Llama 3.1 70B (GQA, 8 KV heads) | 163,840 | 327,680 | 8× smaller |
| DeepSeek-V2-Lite (MLA) | 15,552 | 31,104 | 8.9× smaller |
| DeepSeek-V3 (MLA) | 35,136 | 70,272 | 71× smaller |

"Compared with MHA" divides the cache the same model would need with one key/value head per query head by the actual cache. For the two latent-attention models that baseline keeps their own head shape, so each multi-head key is $128 + 64 = 192$ wide; DeepSeek-V2's Table 9 uses a multi-head model with 128-wide keys instead, $27 \times 2 \times 16 \times 128 = 110{,}592$ elements per token for the V2-Lite size, which makes the reduction a factor of 7.1 rather than 8.9. For Gemma 2 and Gemma 3, the full-cache figure assumes every layer keeps the entire context; its local layers can in fact discard tokens beyond the window ({ref}`sec-ch6-sliding-window-interleaved-attention`), so the real cache at long context is smaller. {numref}`fig-ch6-kv-bytes` plots the cache per token of each model on a logarithmic scale.

:::{figure} figures/kv-bytes.*
:name: fig-ch6-kv-bytes
:alt: Horizontal bar chart of KV cache per token in KiB on a log scale, colored by attention type. DeepSeek-V2-Lite (MLA) is smallest at about 30 KiB, then Qwen2.5 7B (GQA) at 56, DeepSeek-V3 (MLA) at about 69, then the other GQA models up to Gemma 3 12B at 384, and OLMo 2 7B (MHA) largest at 512.
:width: 100%

KV cache per token, all layers, bf16, in KiB (1 KiB = 1,024 bytes; the table above gives
bytes), from the notebook section {ref}`nb-arch-configs`. The color is the attention
type. Latent attention puts DeepSeek-V3, with 61 layers and 128 heads, below every
grouped-query model except Qwen2.5 7B, which keeps only 4 key/value heads; OLMo 2 7B's
multi-head cache is 4 times Llama 3.1 8B's at the same width and depth.
:::

At 128K tokens, Llama 3.1 8B's cache for a single sequence is $131{,}072 \times 131{,}072 = 17.2$ GB, more than its 16.1 GB of bf16 weights. {ref}`sec-ch6-accounting` works through this.

(sec-ch6-grouped-query-attention-computation)=

### 4.3 What grouped-query attention does to the computation

The attention scores themselves are unchanged; only the keys and values are shared. The following sketch shows the whole idea: expand the key and value heads to match the query heads, then run ordinary attention.

```python
import torch
import torch.nn.functional as F


def grouped_query_attention(q, k, v):
    """q: [batch, n_heads, seq, head_dim]; k, v: [batch, n_kv, seq, head_dim].

    n_heads must be a multiple of n_kv.
    """
    group = q.size(1) // k.size(1)
    k = k.repeat_interleave(group, dim=1)  # each KV head serves `group` adjacent heads
    v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


torch.manual_seed(0)
q = torch.randn(1, 8, 5, 16)  # 8 query heads
k, v = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)  # 2 KV heads
out = grouped_query_attention(q, k, v)
for head in range(8):  # query head h uses KV head h // 4, as single-head attention
    kv = head // 4
    alone = F.scaled_dot_product_attention(
        q[:, head], k[:, kv], v[:, kv], is_causal=True
    )
    assert torch.allclose(out[:, head], alone, atol=1e-6)
print("output shape:", tuple(out.shape))
print("heads 0-3 share KV head 0, heads 4-7 share KV head 1")
```

% output

```text
output shape: (1, 8, 5, 16)
heads 0-3 share KV head 0, heads 4-7 share KV head 1
```

The assertion checks each query head against plain attention with the key/value head of its group. Production kernels avoid materializing the repeated tensors, but the result is the same. That equivalence is what the notebook asserts ({ref}`nb-arch-block`): grouped-query attention equals multi-head attention whose key and value weights are the grouped weights repeated, to floating-point precision, and multi-query attention is the `n_kv = 1` case.

(sec-ch6-multi-head-latent-attention)=

### 4.4 Multi-head latent attention

DeepSeek-V2 reduces the cache by a different route: instead of sharing heads, it compresses them {cite}`deepseek2024v2`. The hidden state $h_m$ at position $m$ is projected down to a latent vector $c_m = h_m W^{DKV}$ of size $d_c$ (512 in both V2-Lite and V3). Keys and values for all heads are reconstructed from the latent by up-projections, $k_m = c_m W^{UK}$ and $v_m = c_m W^{UV}$, and only $c_m$ is cached. At inference the up-projection of the keys can be absorbed into the query projection and that of the values into the output projection $W^O$, so the full keys and values are never materialized.

There is a complication. RoPE rotates the key by a position-dependent matrix, and that matrix would sit between $W^{UK}$ and the query, preventing the absorption. DeepSeek therefore splits each head into a part without positional information (128 dimensions) and a small decoupled part that carries RoPE (64 dimensions). The rotary key is shared across heads and cached separately, which is the $d_{rope} = 64$ in the cache formula. DeepSeek-V3's cache per token per layer is $512 + 64 = 576$ elements; the same layer under multi-head attention with its 128 heads would hold $128 \times (128 + 64 + 128) = 40{,}960$ elements, a factor of 71.

Queries are also compressed through a low-rank bottleneck in V3 to save activation memory in training: a down-projection $W^{DQ}$ to rank 1,536 and an up-projection $W^{UQ}$ back to the heads' width. V2-Lite projects queries directly. RMSNorm layers on the two latents are part of the block. {ref}`sec-ch6-accounting` includes these in the parameter count. {numref}`fig-ch6-mla-dataflow` traces one layer's data flow with DeepSeek-V3's sizes.

:::{figure} figures/mla-dataflow.*
:name: fig-ch6-mla-dataflow
:alt: Data flow of multi-head latent attention. The hidden state is projected to a latent of 512 elements and to a rotary key of 64 elements, both cached. Dashed boxes, content keys and values reconstructed from the latent by up-projections, feed the attention scores and the output. Queries come from the hidden state through a rank-1,536 bottleneck.
:width: 100%

Multi-head latent attention in DeepSeek-V3. Only the two orange boxes are cached, 576
elements per token per layer. Queries pass through a rank-1,536 bottleneck of their own,
which is not cached. The dashed keys and values are what the up-projections would
produce; at inference they are never formed, because $W^{UK}$ is folded into the query
side and $W^{UV}$ into the output projection.
:::

DeepSeek-V3 caches 68.6 KiB per token against Llama 3.1 70B's 320 KiB, although it has 128 heads against 64. The design is a bet that the cache is the binding constraint and that the extra projection compute is cheap by comparison.

(sec-ch6-sliding-window-interleaved-attention)=

### 4.5 Sliding-window and interleaved attention

Full causal attention costs $\mathcal{O}(n^2)$ in time and the cache grows with $n$. Sliding-window attention lets each token attend only to the previous $n_{win}$ tokens {cite}`beltagy2020longformer`. Time becomes $\mathcal{O}(n\, n_{win})$ and the cache $\mathcal{O}(n_{win})$ tokens per layer, independent of $n$, and since each layer extends the receptive field by $n_{win}$, information can still travel $L\, n_{win}$ positions through the stack. Mistral 7B uses $n_{win} = 4096$ in every layer and a rolling-buffer cache of fixed size {cite}`jiang2023mistral`.

A pure window gives up exact long-range retrieval in any single layer. Gemma 2 alternates local layers (window 4,096) with global layers (8,192-token span) {cite}`gemma2024gemma2`: half the layers pay for the full span, half pay a constant. It also bounds the magnitudes of the attention logits and of the final logits with a soft-cap, which the paper writes as `logits ← soft_cap ∗ tanh(logits / soft_cap)`, with `soft_cap` set to 50.0 for the attention logits and 30.0 for the final ones {cite}`gemma2024gemma2`. Near zero the function is the identity, and its output stays strictly between `-soft_cap` and `soft_cap`.

Gemma 3 pushes the ratio further: "a pattern of 5 local layers for every global layer, starting with a local layer", with the local span cut to 1,024 tokens, and QK-norm in place of the soft-cap {cite}`gemma2025gemma3`. The authors report "minimal impact on perplexity" from the ratio. The cache arithmetic shows why it is worth having. Gemma 3 12B has 8 global and 40 local layers, each caching $2 \times 8 \times 256 = 4{,}096$ elements per token. At 128K tokens in bf16 the global layers hold $8 \times 4{,}096 \times 131{,}072 \times 2 = 8.59$ GB and the local layers only their last 1,024 tokens, $40 \times 4{,}096 \times 1{,}024 \times 2 = 0.34$ GB: 8.93 GB in all, against 51.5 GB if every layer were global. ModernBERT, an encoder, uses a similar alternation, with a global layer every third layer and a 128-token local window ({ref}`sec-ch6-encoders-encoder-decoders-today`).

The listing below checks both halves of that arithmetic on a toy stack in Gemma 3's pattern, five local layers and one global layer, one head of width 16 each. Decoding token by token, every local layer keeps a rolling buffer of its last $n_{win}$ keys and values and evicts the oldest; the result must equal the full pass with the window applied as a mask, and the cache must hold $n$ entries in the global layer and $\min(n, n_{win})$ in each local one.

```python
import torch
import torch.nn.functional as F


def attend(q, k, v, window=None):
    """Causal single-head attention; with a window, query t sees keys t-window+1..t."""
    pos = torch.arange(q.size(0))
    allowed = pos[None, :] <= pos[:, None]
    if window is not None:
        allowed &= pos[:, None] - pos[None, :] < window
    scores = (q @ k.T / q.size(-1) ** 0.5).masked_fill(~allowed, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


torch.manual_seed(0)
n, d_k, n_win = 64, 16, 8
kinds = ["local"] * 5 + ["global"]
# W_q, W_k, W_v per layer
weights = [torch.randn(3, d_k, d_k) / d_k ** 0.5 for _ in kinds]
x = torch.randn(n, d_k)

full = x  # every position at once, the window as a mask
for kind, w in zip(kinds, weights):
    window = n_win if kind == "local" else None
    full = full + attend(full @ w[0], full @ w[1], full @ w[2], window)

cache = [([], []) for _ in kinds]  # token by token; local layers evict past the window
steps = []
for t in range(n):
    h = x[t]
    for kind, w, (keys, values) in zip(kinds, weights, cache):
        keys.append(h @ w[1]); values.append(h @ w[2])
        if kind == "local" and len(keys) > n_win:
            keys.pop(0); values.pop(0)  # rolling buffer: drop the oldest token
        scores = torch.stack(keys) @ (h @ w[0]) / d_k ** 0.5
        h = h + F.softmax(scores, dim=-1) @ torch.stack(values)
    steps.append(h)

assert torch.allclose(torch.stack(steps), full, atol=1e-5)
cached = sum(len(keys) + len(values) for keys, values in cache) * d_k
assert cached == (1 * n + 5 * min(n, n_win)) * 2 * d_k
print("decoding with eviction equals the masked full pass")
print(f"cache: {cached:,} elements = (1 x {n} + 5 x {n_win}) x 2 x {d_k}")
print(f"with every layer global: {6 * n * 2 * d_k:,} elements")
```

% output

```text
decoding with eviction equals the masked full pass
cache: 3,328 elements = (1 x 64 + 5 x 8) x 2 x 16
with every layer global: 12,288 elements
```

Trained softmax attention has a property that surprises people and that windowed eviction exposes: tokens near the start of a sequence receive a disproportionate share of attention even when they are irrelevant, as "attention sinks", and removing them from a windowed cache breaks the model. {cite:t}`xiao2024efficient` exploit this to stream arbitrarily long text by always keeping the first few tokens plus a window. The streaming model does not remember what fell out of the window; it simply stays stable. {numref}`fig-ch6-attention-patterns` shows which keys each query scores under these patterns, and the order of local and global layers in Gemma 2 and Gemma 3.

:::{figure} figures/attention-patterns.*
:name: fig-ch6-attention-patterns
:alt: Three 16-by-16 masks of query position against key position: full causal is a lower triangle; a sliding window of 4 is a diagonal band; sink plus window adds the first two key columns to the band. Below, two strips of layers: Gemma 2 alternates local and global layers; Gemma 3 has five local layers before each global one, eight global layers in 48.
:width: 100%

Top: the keys each of 16 queries scores (blue) under full causal attention, a window of
$n_{win} = 4$, and two sink tokens plus the window. The window's cost grows as
$n\, n_{win}$ rather than $n^2/2$. Bottom: local and global layers in Gemma 2 9B
(alternating, as in the Transformers implementation) and Gemma 3 12B (five local per
global, starting with a local layer).
:::

```{admonition} Cache Sizes, Measured
:class: tip
The notebook section {ref}`nb-arch-kvcache` generates text with a cache and checks that
it equals the full forward pass, measures the cache of the small model in bytes against
the formulas above for MHA, GQA and MQA, and builds a simplified latent cache whose
output matches its own uncompressed expansion. The windowed rolling-buffer cache is
checked by the listing above.
```

(sec-ch6-positions-long-context)=

## 5. Positions and Long Context

(sec-ch6-rope-one-paragraph)=

### 5.1 RoPE and the relative offset

Chapter 1 implements rotary embeddings ({ref}`sec-ch1-rope`) {cite}`su2021roformer`; this section states the formulas. Each pair of dimensions $(2i, 2i+1)$ of a query or key at position $m$ is rotated by the angle $m\,\omega_i$, with $\omega_i = b^{-2i/d_k}$ for a base $b$ (10,000 in the original), $i = 0, \dots, d_k/2 - 1$ and $d_k$ the head width. The dot product between a query at position $m$ and a key at position $m'$ then depends only on $m - m'$.

In complex form the rotation is a single multiplication, and the relative-offset property takes three lines. Write pair $i$ of the query at position $m$ as the complex number $\hat{q}_{m,i} = q_{m,2i} + \mathrm{i}\, q_{m,2i+1}$, and pair $i$ of the key at position $m'$ likewise as $\hat{k}_{m',i}$. RoPE multiplies each pair by a unit complex number:

$$\hat{q}_{m,i} \mapsto \hat{q}_{m,i}\, e^{\mathrm{i}\, m\, \omega_i}, \qquad \hat{k}_{m',i} \mapsto \hat{k}_{m',i}\, e^{\mathrm{i}\, m'\, \omega_i}$$ (eq-ch6-rope-complex)

The dot product of two real 2-vectors equals the real part of the first, read as a complex number, times the complex conjugate of the second, which is written with a superscript star: for pair $i$ it is $\mathrm{Re}(\hat{q}_{m,i}\, \hat{k}_{m',i}^{*})$ before rotation. The contribution of pair $i$ to the attention score is therefore

$$\begin{aligned}
\mathrm{Re}\big[(\hat{q}_{m,i}\, e^{\mathrm{i} m \omega_i})(\hat{k}_{m',i}\, e^{\mathrm{i} m' \omega_i})^{*}\big]
&= \mathrm{Re}\big[\hat{q}_{m,i}\, \hat{k}_{m',i}^{*}\, e^{\mathrm{i} m \omega_i}\, e^{-\mathrm{i} m' \omega_i}\big] \\
&= \mathrm{Re}\big[\hat{q}_{m,i}\, \hat{k}_{m',i}^{*}\, e^{\mathrm{i} (m - m') \omega_i}\big].
\end{aligned}$$

The first step uses that the conjugate of a product is the product of the conjugates, and that conjugating $e^{\mathrm{i} m' \omega_i}$ negates its exponent; the second adds the exponents. The positions enter only through $m - m'$, and the score is the sum of these terms over the $d_k/2$ pairs, so it depends on the offset alone. Chapter 1 derives the same result with real $2 \times 2$ rotation matrices ({ref}`sec-ch1-rope`).

The frequencies range from $\omega_0 = 1$ radian per position, which completes a turn in about 6 tokens, down to $\omega_{d_k/2-1} = b^{-(d_k-2)/d_k}$, almost $1/b$, which turns once in $2\pi b^{(d_k-2)/d_k}$ tokens: about 54,400 for $b = 10{,}000$ and $d_k = 128$, or 0.87 of $2\pi b$. The base therefore sets the longest wavelength the model can represent.

(sec-ch6-trained-length-matters)=

### 5.2 Why trained length matters

A model trained on sequences up to $n_{train}$ tokens has only ever seen rotation angles $m\,\omega_i$ with $m < n_{train}$. For high-frequency pairs this covers many full turns, so every angle in $[0, 2\pi)$ has been seen. For the low-frequency pairs, those with $n_{train}\,\omega_i < 2\pi$, the training positions cover only part of a turn. At position $m \geq n_{train}$ these pairs rotate to angles never seen in training, and the model's behavior is unconstrained. The failures of naive extrapolation come from these low-frequency components.

Two families of fix change the frequencies, and both are used in the table:

**A larger base.** Raising $b$ lowers every frequency, so the angles at position $m$ are those of a smaller position. {cite:t}`xiong2024effective` proposed raising the base from 10,000 to 500,000 for long-context training, and Llama 3 uses that value {cite}`grattafiori2024llama`; Qwen2.5 and Mixtral use 1,000,000. It is the simplest approach: train from the start, or continue training, with the larger base.

**Rescaling an existing model.** Position interpolation divides every position by the length ratio $s_{ctx} = n'/n_{train}$ of a target context length $n'$ to the trained one, so positions up to the target length map into $[0, n_{train})$ and no angle exceeds what training saw {cite}`chen2023extending`. Its drawback is that it squeezes the high frequencies too, blurring the distinction between adjacent tokens. NTK-aware scaling, from a 2023 Reddit post by the pseudonymous {cite:t}`bloc972023ntk`, instead changes the base; the YaRN paper writes the rule out (its Eq. 22, arXiv v3) {cite}`peng2024yarn`:

$$b' = b \cdot s_{ctx}^{\,d_k/(d_k-2)}$$

This leaves the highest frequency $\omega_0 = 1$ unchanged and divides the lowest by exactly $s_{ctx}$, with the frequencies between spread geometrically. High-frequency components are preserved and low-frequency ones are interpolated. YaRN refines this: it interpolates the low-frequency dimensions, leaves the high-frequency ones, blends between them with a ramp, and multiplies the attention logits by a factor $1/\tau$ that grows slowly with the scale factor, so the temperature $\tau$ falls: $\sqrt{1/\tau} = 0.1 \ln s_{ctx} + 1$ in the YaRN paper's fit for LLaMA (its Eq. 15, arXiv v3) {cite}`peng2024yarn`, and DeepSeek-V2 uses $0.0707 \ln s_{ctx} + 1$ for the same factor {cite}`deepseek2024v2`. DeepSeek-V2 and V3 use YaRN with scale factor 40 over a 4,096-token pretraining length (163,840 positions in the configuration); the reports describe extension training at 32K for V2, evaluated up to 128K, and at 32K and then 128K for V3 {cite}`deepseek2024v2,deepseek2024v3`.

Written out, YaRN's interpolation (called "NTK-by-parts" in the paper; Eqs. 8, 10, 11 and 13 of arXiv v3) works pair by pair, from the wavelength $\ell_i = 2\pi/\omega_i$ and the number of turns the pair completes within the trained length, $\xi_i = n_{train}/\ell_i$ {cite}`peng2024yarn`:

$$\psi(\xi) = \begin{cases} 0 & \xi < \beta_{slow} \\ 1 & \xi > \beta_{fast} \\ \dfrac{\xi - \beta_{slow}}{\beta_{fast} - \beta_{slow}} & \text{otherwise} \end{cases} \qquad \omega_i' = \big(1 - \psi(\xi_i)\big)\, \frac{\omega_i}{s_{ctx}} + \psi(\xi_i)\, \omega_i$$

A pair that turns more than $\beta_{fast}$ times during training keeps its frequency; one that turns fewer than $\beta_{slow}$ times is divided by $s_{ctx}$, as in position interpolation; those between are blended linearly. The paper's values for the Llama family are $\beta_{slow} = 1$ and $\beta_{fast} = 32$ (the paper names them alpha and beta). The attention factor $\sqrt{1/\tau}$ above completes the method. For $d_k = 128$, $b = 10{,}000$ and $n_{train} = 4{,}096$, the function below keeps 21 of the 64 pairs, blends 25 and fully interpolates 18.

```python
import math


def yarn_frequencies(d, base, n_train, scale, beta_slow=1.0, beta_fast=32.0):
    """YaRN's per-pair frequencies, each with the ramp value psi of its pair."""
    out = []
    for i in range(d // 2):
        omega = base ** (-2 * i / d)
        xi = n_train * omega / (2 * math.pi)  # turns within the trained length
        psi = min(1.0, max(0.0, (xi - beta_slow) / (beta_fast - beta_slow)))
        out.append(((1 - psi) * omega / scale + psi * omega, psi))
    return out


pairs = yarn_frequencies(128, 10000.0, 4096, 8)
kept = sum(psi == 1.0 for _, psi in pairs)
interpolated = sum(psi == 0.0 for _, psi in pairs)
blended = len(pairs) - kept - interpolated
print(f"kept {kept}, blended {blended}, interpolated {interpolated}")
print(f"lowest frequency divided by {(10000.0 ** (-126 / 128)) / pairs[-1][0]:.4g}")
print(f"attention factor at s = 8: {0.1 * math.log(8) + 1:.4g}")
```

% output

```text
kept 21, blended 25, interpolated 18
lowest frequency divided by 8
attention factor at s = 8: 1.208
```

The paper defines the ramp on $\xi$, but DeepSeek-V3's released inference code (the function `precompute_freqs_cis` in its [`inference/model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py)) ramps in the pair index $i$ instead: it finds the pair indices at which $\xi$ equals $\beta_{fast}$ and $\beta_{slow}$, rounds the first down and the second up to whole pairs, and blends linearly in $i$ between them. Because $\xi_i$ falls geometrically with $i$, the two forms give the blended pairs different weights. They agree on which pairs are kept, blended and interpolated in this example: the index form keeps pairs 0 to 20 and fully interpolates pairs 46 to 63, again 21, 25 and 18.

{numref}`fig-ch6-rope-scaling` compares the three rescalings pair by pair for the same example.

:::{figure} figures/rope-scaling.*
:name: fig-ch6-rope-scaling
:alt: Line chart of the ratio of new to old frequency for each of 64 rotary pairs, log scale from 1 to 1/8. Position interpolation is flat at 1/8. NTK-aware scaling falls in a straight line from 1 at pair 0 to 1/8 at pair 63. YaRN stays at 1 for pairs 0 to 20, falls between, and is 1/8 from pair 46 on.
:width: 100%

How each method changes the 64 frequencies when 4,096 trained positions are extended 8
times ($d_k = 128$, $b = 10{,}000$). Position interpolation divides all of them by 8,
including the high frequencies that tell neighboring tokens apart. NTK-aware scaling
divides pair $i$ by $8^{2i/(d_k-2)}$, a straight line on this log scale. YaRN keeps the
21 pairs that turn more than 32 times within the trained length, divides the 18 that
turn less than once by 8, and blends the 25 between.
:::

Two short functions capture the frequency arithmetic.

```python
import torch


def rope_frequencies(d, base=10000.0):
    """omega_i = base^(-2i/d) for i = 0 .. d/2 - 1, in float64."""
    return 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))


def ntk_aware_base(base, scale, d):
    """Base that divides the lowest frequency by `scale` and keeps the highest."""
    return base * scale ** (d / (d - 2))


print(f"lowest frequency at d = 128: {rope_frequencies(128)[-1].item():.4g}")
print(f"NTK-aware base, 8x longer context: {ntk_aware_base(10000.0, 8, 128):,.0f}")
```

% output

```text
lowest frequency at d = 128: 0.0001155
NTK-aware base, 8x longer context: 82,685
```

The notebook section {ref}`nb-arch-rope` asserts the identities above to $10^{-9}$: after NTK-aware scaling, $\omega_0$ is unchanged and $\omega_{d_k/2-1}$ is exactly $1/s_{ctx}$ of its old value; with position interpolation, the largest angle at the low-frequency components over the extended length stays within one position's rotation of the largest angle seen in training.

(sec-ch6-alternatives-rope)=

### 5.3 Alternatives to RoPE

ALiBi adds a penalty to attention scores proportional to the distance between query and key, with a different slope per head, and no positional signal elsewhere {cite}`press2022train`. It extrapolates to longer sequences with no change, which was its selling point, but it has been displaced by RoPE with rescaling in most recent models. T5's bucketed biases {cite}`raffel2019exploring`, which Chapter 1 implements with T5's 32 buckets and maximum distance of 128 ({ref}`sec-ch1-relative-bias`), belong to the same family: a learned additive bias indexed by distance. {cite:t}`shaw2018self` introduced relative positions earlier, as learned vectors added to the keys and values.

(sec-ch6-long-context-training)=

### 5.4 Long-context training and what a number promises

The configuration's `max_position_embeddings` says what the code accepts, not what the model does well. A long-context model is made in stages. Pretraining is done at a short length because attention cost grows with the square of it. A short final stage, a small fraction of total tokens, continues at the long length with a larger RoPE base or rescaled frequencies. Llama 3's 405B model does this in six stages from 8K to 128K, and the authors check at each that short-context quality is unchanged and that needle-in-a-haystack retrieval succeeds at the new length before moving on {cite}`grattafiori2024llama`.

Three different statements get blurred under "context length":

1. *The model accepts that many positions* (the configuration value).
2. *The model was trained on sequences that long*, which is what the training report says.
3. *The model uses information anywhere in them*, which only evaluations on long-context tasks show.

Mistral 7B v0.1 is an example: its configuration says 32,768 positions, its paper reports training at 8,192 with a 4,096 window. The first number is a statement about code, the second about training. Treat a context-length figure as the first claim unless the report supports the others. {numref}`fig-ch6-context-claims` sets the stated numbers of four models against each other.

:::{figure} figures/context-claims.*
:name: fig-ch6-context-claims
:alt: Dot chart on a log scale from 4K to 160K tokens. Mistral 7B v0.1: window 4K, pretraining 8K, configuration 32K. Qwen2.5 7B: pretraining 4K, long-context stage at 32K, configuration 128K. Llama 3 405B: pretraining 8K, long-context stage at 128K, configuration 128K. DeepSeek-V3: pretraining 4K, stages at 32K and 128K, configuration 160K.
:width: 100%

Context numbers from the sources cited in this chapter. The red bar is what the
configuration accepts (claim 1); the markers are the lengths the reports say training
used (claim 2). Where a long-context stage reaches the red bar, as for Llama 3, the two
claims meet; for Mistral 7B v0.1 and Qwen2.5 7B the configuration accepts four times the
longest trained length.
:::

```{admonition} Frequencies and Scaling
:class: note
The notebook section {ref}`nb-arch-rope` checks that RoPE preserves norms and makes
query-key products depend on relative offset, asserts the NTK-aware identities in
float64, compares interpolation and extrapolation on the low-frequency components, and
plots the rotation angles across the extended range.
```

(sec-ch6-mixture-of-experts)=

## 6. Mixture of Experts

(sec-ch6-moe-idea)=

### 6.1 The idea

A dense model uses all of its parameters for every token. A mixture-of-experts (MoE) layer replaces the feed-forward network by $E$ separate feed-forward networks (experts) and a router that chooses $n_{act}$ of them for each token. The layer's parameters grow with $E$; its compute per token grows with $n_{act}$. Capacity is decoupled from cost.

A router computes a logit $r_i$ for every expert $i$ from the token's hidden state $x$ and a probability $\rho_i = \mathrm{softmax}(r)_i$, keeps the $n_{act}$ experts with the largest logits, $\mathrm{TopK}(r)$, and combines the chosen experts' outputs weighted by their probabilities, renormalized over the chosen experts in this form:

$$y = \sum_{i \in \mathrm{TopK}(r)} \tilde{\rho}_i\, \mathrm{FFN}_i(x), \qquad \tilde{\rho}_i = \frac{\rho_i}{\sum_{j \in \mathrm{TopK}(r)} \rho_j}$$

GShard introduced top-2 routing for translation transformers {cite}`lepikhin2021gshard`; Switch Transformers showed that routing to a single expert works at trillion-parameter scale {cite}`fedus2022switch`. Mixtral uses top-2 of 8 experts in every layer, and the softmax over the two selected logits is the same as renormalizing the full softmax over the selected pair {cite}`jiang2024mixtral`. Renormalization is a per-model choice: Mixtral and DeepSeek-V3 renormalize over the selected experts, while OLMoE and DeepSeek-V2-Lite weight by the raw probabilities $\rho_i$ (their configurations set `norm_topk_prob: false`). A sketch of the router:

```python
import torch


def route_top_k(x, router_weight, k):
    """x: [tokens, d]; router_weight: [experts, d].

    Returns the chosen expert ids [tokens, k], their gate weights [tokens, k], and the
    load-balancing loss of Fedus et al., n_experts * sum_i f_i * P_i.
    """
    n_experts = router_weight.size(0)
    probs = (x @ router_weight.T).softmax(dim=-1)
    top_p, top_i = probs.topk(k, dim=-1)
    gates = top_p / top_p.sum(dim=-1, keepdim=True)
    counts = torch.bincount(top_i.flatten(), minlength=n_experts).float()
    f = counts / counts.sum()  # fraction of routed slots sent to each expert; sums to 1
    p = probs.mean(dim=0)  # mean router probability of each expert
    return top_i, gates, n_experts * (f * p).sum()


def fractions(ids, n_experts=8):
    return torch.bincount(ids.flatten(), minlength=n_experts) / ids.numel()


torch.manual_seed(0)
# independent random logits
ids, _, aux = route_top_k(torch.randn(4096, 8), torch.eye(8), 2)
f = fractions(ids)
shares = " ".join(f"{v:.3f}" for v in f.tolist())
print("random logits: fractions", shares, f"loss {aux:.3f}")
assert (f - 1 / 8).abs().max() < 0.01

ids, _, aux = route_top_k(torch.ones(64, 16), torch.zeros(8, 16), 2)  # all logits tied
f = fractions(ids).sort(descending=True).values
shares = " ".join(f"{v:.3f}" for v in f.tolist())
print("tied logits:   fractions", shares, f"loss {aux:.3f}")
assert (f > 0).sum() == 2
```

% output

```text
random logits: fractions 0.121 0.131 0.120 0.125 0.127 0.123 0.126 0.127 loss 1.000
tied logits:   fractions 0.500 0.500 0.000 0.000 0.000 0.000 0.000 0.000 loss 1.000
```

With independent random logits over 4,096 tokens every expert receives within 0.01 of an eighth of the routed slots, and the loss is 1. With all logits tied, `topk` sends every token to the same two experts and leaves six idle, yet the loss is still 1, because the mean probabilities stay uniform. That second router is the collapsed case {ref}`sec-ch6-keeping-experts-balanced` warns about: a loss near 1 does not certify balance.

(sec-ch6-keeping-experts-balanced)=

### 6.2 Keeping the experts balanced

Left alone, routers collapse: an expert that gets slightly more tokens trains faster, gets better, and receives still more. The remaining experts starve and their parameters are wasted. The listing below shows it on a toy layer of 8 linear experts with top-1 routing, trained to reproduce a random linear map. Each expert's output is weighted by its router probability, so the gradient raises the probability of whichever expert currently does best, and only the experts that are chosen are trained. It trains three seeds without and with the Switch loss defined below, at $\lambda_{aux} = 0.1$, and reports how many experts receive no tokens in the last batch.

```python
import torch
import torch.nn.functional as F


def train_moe(aux_weight, seed, n_experts=8, d=16, steps=400):
    torch.manual_seed(seed)
    target = torch.randn(d, d) / d ** 0.5
    router = torch.nn.Linear(d, n_experts, bias=False)
    experts = torch.nn.Parameter(torch.randn(n_experts, d, d) * 0.02)
    opt = torch.optim.Adam([
        {"params": router.parameters(), "lr": 0.05},
        {"params": [experts], "lr": 0.01},
    ])
    for _ in range(steps):
        x = torch.randn(256, d)
        probs = F.softmax(router(x), dim=-1)
        top_p, ids = probs.max(dim=-1)  # top-1 routing
        y = top_p[:, None] * torch.einsum("bd,bde->be", x, experts[ids])
        f = torch.bincount(ids, minlength=n_experts).float() / len(ids)
        aux = n_experts * (f * probs.mean(0)).sum()  # Switch loss
        loss = F.mse_loss(y, x @ target) + aux_weight * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
    return int((f == 0).sum())


for aux_weight in (0.0, 0.1):
    idle = [train_moe(aux_weight, seed) for seed in range(3)]
    print(f"aux weight {aux_weight}: idle experts per seed {idle}")
    assert all(i >= 3 for i in idle) if aux_weight == 0 else all(i == 0 for i in idle)
```

% output

```text
aux weight 0.0: idle experts per seed [4, 3, 4]
aux weight 0.1: idle experts per seed [0, 0, 0]
```

Without the balancing term, three or four of the eight experts receive no tokens by the end of every run: they lost the early competition, so they stopped being trained, so they never caught up. With it, every expert stays in use. Three mechanisms counter this collapse in practice.

**An auxiliary loss.** The Switch loss is $\lambda_{aux}\, E \sum_{i=1}^{E} f_i \bar{\rho}_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$ and $\bar{\rho}_i$ is the mean router probability assigned to it {cite}`fedus2022switch`. With $\sum_i f_i = 1$ (defined over the routed slots when $n_{act} > 1$), the loss equals 1 under uniform routing. It is not a minimum over all routers: with two experts, 30% of tokens at $\rho = (1, 0)$ and 70% at $(0.49, 0.51)$ give 0.886. If $f$ tracked $\bar{\rho}$ exactly, the loss would be $E \sum_i \bar{\rho}_i^2 \geq 1$, with equality only for uniform routing, which is the sense in which {cite:t}`fedus2022switch` call it minimized. A value near 1 therefore does not certify balance. Because $f_i$ is not differentiable, the gradient flows through $\bar{\rho}_i$. The coefficient trades balance against quality: Mixtral's configuration has 0.02 and OLMoE's 0.01. ST-MoE adds a *router z-loss*, a penalty on the squared log-sum-exp of the router logits that keeps them small and the routing computation numerically stable in bf16 {cite}`zoph2022st`.

**A capacity factor.** On parallel hardware each expert processes a fixed-size buffer of $\kappa = \lceil \phi\, n_{act}\, T / E \rceil$ tokens for $T$ tokens in a batch and capacity factor $\phi$. Tokens beyond an expert's capacity are dropped from that expert (they pass through the residual connection unchanged). A factor of 1.0 is fully balanced and drops any overload; larger values waste compute on padding. "Dropless" implementations such as MegaBlocks {cite}`gale2023megablocks`, used for OLMoE {cite}`muennighoff2025olmoe`, avoid the choice at the cost of irregular shapes.

**Expert-choice routing.** {cite:t}`zhou2022mixture` invert the direction: each expert chooses its top tokens, which makes balance exact by construction, though it complicates autoregressive decoding, where the future tokens that would compete for an expert are not yet known.

(sec-ch6-fine-grained-shared-experts)=

### 6.3 Fine-grained and shared experts

DeepSeekMoE makes two changes to the conventional layout of GShard-style top-$n_{act}$ routing {cite}`dai2024deepseekmoe`. *Fine-grained experts* splits each expert into several smaller ones and routes to more of them: OLMoE uses 64 experts of width 1,024 with top-8, DeepSeek-V3 uses 256 with width 2,048 and top-8, instead of 8 experts with width 14,336 and top-2. The number of combinations of experts a token can use grows combinatorially, which should let experts specialize more finely. *Shared experts* are always active: DeepSeek keeps one or two that process every token, so common knowledge need not be duplicated across the routed experts. {numref}`fig-ch6-moe-layer` shows one token passing through such a layer.

:::{figure} figures/moe-layer.*
:name: fig-ch6-moe-layer
:alt: Diagram of a mixture-of-experts layer. The token's hidden state x goes to a router and to the shared experts. The router computes logits and probabilities and selects the top-scoring experts; two selected experts, FFN 1 and FFN 3, are drawn bold with their gate weights, the others faint. The output y is the shared experts' output plus the weighted sum of the selected experts.
:width: 100%

One token through a mixture-of-experts layer with shared experts, drawn in the
renormalized form of {ref}`sec-ch6-moe-idea`. The router scores all $E$ routed experts
but only the $n_{act}$ selected (bold) run, each weighted by its renormalized
probability $\tilde{\rho}_i$ (models that set `norm_topk_prob: false`, such as OLMoE and
DeepSeek-V2-Lite, weight by $\rho_i$ instead); the shared experts run for every token.
Compute per token follows the bold boxes, memory follows all of them.
:::

DeepSeek-V3 adds three more refinements {cite}`deepseek2024v3`. Its router uses sigmoid scores rather than softmax, normalizes them over the selected experts and multiplies them by 2.5 (Eqs. 12 to 15 of the paper, with the factor from the configuration):

$$\rho_i = \mathrm{sigmoid}(r_i) = \frac{1}{1 + e^{-r_i}}, \qquad \tilde{\rho}_i = 2.5\, \frac{\rho_i}{\sum_{j \in \mathrm{TopK}} \rho_j} \quad \text{for } i \in \mathrm{TopK},$$

where $\mathrm{TopK}$ is the set of $n_{act}$ selected experts (chosen with the bias described below) and the unselected experts get no weight. Unlike softmax probabilities, the sigmoid scores do not compete: each lies in $(0, 1)$ on its own, and only the normalization couples them. Routing is node-limited, as DeepSeek-V2 already did with devices: the 256 experts are divided into 8 groups, a token chooses its best 4 groups and then the top 8 experts among them, which bounds cross-node communication. And it replaces the auxiliary loss with *auxiliary-loss-free* balancing {cite}`wang2024auxiliary`: a per-expert bias is added to the scores for the *selection* of the top experts only (not for the gate weights), and after each step the bias of overloaded experts is lowered and of underloaded ones raised. No gradient from a balancing objective competes with the language-modeling loss; a very small sequence-level auxiliary loss remains as a safeguard. The bias is stored per layer, one value per expert, which is why V3's router has $E$ extra parameters per layer in {ref}`sec-ch6-accounting`.

The first layers of both DeepSeek models are dense (one in V2-Lite, three in V3); the configuration's `first_k_dense_replace` says how many.

(sec-ch6-total-active-parameters)=

### 6.4 Total and active parameters

A sparse model has two sizes. Its *total* parameters set the memory needed to hold it; its *active* parameters, those a single token touches, set the compute per token. Mixtral 8x7B holds 46.7B parameters and uses 12.9B for each token, so it costs per token about what a 13B dense model does while requiring memory for a 47B one. DeepSeek-V3 holds 671B and activates 36.6B, 5.5% of them. {numref}`fig-ch6-moe-parameters` breaks each sparse model's total into its parts.

:::{figure} figures/moe-parameters.*
:name: fig-ch6-moe-parameters
:alt: Stacked horizontal bars, each 100 percent of a model's parameters, split into input embedding, output head, attention with norms and routers, dense and shared feed-forward, active routed experts and idle routed experts. The idle share dominates: active excluding the input embedding is 27.3 percent for Mixtral, 17.0 for OLMoE, 15.6 for DeepSeek-V2-Lite and 5.5 for DeepSeek-V3.
:width: 100%

Where the parameters of the four sparse models are, as shares of each total, computed
from the committed configurations with the counting rules of {ref}`sec-ch6-accounting`
(the totals match {ref}`sec-ch6-validation`). Everything except the pale idle experts
and the input-embedding lookup is used for a token; finer-grained routing pushes the
active share down from Mixtral's 27.3% to DeepSeek-V3's 5.5%.
:::

This changes what "size" means when comparing models. For compute, FLOPs per token, and training cost, the active count is what matters. For serving memory, and the number of GPUs needed to hold the weights, the total counts. Because a batch of tokens activates many different experts, the whole parameter set is touched at any reasonable batch size, so the memory bandwidth required per decoding step is also closer to that of the total than of the active count. Sparse models are cheaper to *train* than dense models of equal quality: Switch Transformers reached the quality of a dense T5 model up to seven times faster "while still using the same FLOPS per token" {cite}`fedus2022switch`. Their serving advantage is smaller and depends on batch size and the interconnect.

Vendors disagree on whether "active" includes the input embedding table, which contributes parameters but no matrix multiplications (a lookup is not a multiplication). DeepSeek's 2.4B and 37B match a count that excludes it; OLMoE's 1.3B and Mixtral's 12.9B match a count that includes it. {ref}`sec-ch6-accounting` computes both and states which each vendor's figure matches.

```{admonition} A Router You Can Run
:class: tip
The notebook section {ref}`nb-arch-moe` routes 4,096 random tokens through a top-2
router over 8 experts, checks that the dispatch fractions are near uniform and that the
auxiliary loss is near 1 for a balanced router and well above it for one biased toward a
single expert, and verifies the capacity-dropping count against a plain loop.
```

(sec-ch6-efficiency-built-into-architecture)=

## 7. Efficiency Built into the Architecture

(sec-ch6-exact-attention-faster)=

### 7.1 Exact attention, faster

Before FlashAttention, the usual answer to the quadratic cost of attention was to approximate it. Linformer observes that the attention matrix is approximately low-rank and projects the keys and values along the sequence to a fixed length, which makes time and memory linear in $n$ {cite}`wang2020linformer`. Performers approximate the softmax kernel with random features, also in linear time and space {cite}`choromanski2021rethinking`. Sparse patterns such as Longformer's window restrict which pairs are scored at all {cite}`beltagy2020longformer`. None of the twelve models in this chapter uses an approximation; the sliding windows of Mistral and Gemma are exact attention over a shorter span. {cite:t}`dao2022flashattention` name the reason: approximate methods trade "model quality to reduce the compute complexity, but often do not achieve wall-clock speedup". On a GPU the cost of attention was the memory traffic of the score matrix, which for one head at $n = 131{,}072$ would be $131{,}072^2 \times 2$ bytes $= 34.4$ GB in bf16, not the arithmetic, and an exact algorithm that avoids that traffic removed the reason to approximate.

FlashAttention computes exactly the same attention output as the textbook implementation but never writes the $n \times n$ score matrix to GPU main memory. It processes blocks that fit in on-chip SRAM, keeping running softmax statistics, so the extra memory is linear rather than quadratic in sequence length and the traffic to main memory falls by a large constant factor {cite}`dao2022flashattention`. FlashAttention-2 improves the work partition across thread blocks and warps {cite}`dao2024flashattention2`. Because the result is identical up to floating-point rounding, these are kernel choices, not architectural ones, but they changed what is affordable: attention memory stopped limiting sequence length, and the architectural cost that remains is the KV cache. PyTorch's `scaled_dot_product_attention`, used in this chapter's notebook, dispatches to such a kernel when one is available.

(sec-ch6-kv-cache-serving)=

### 7.2 The KV cache in serving

Paged attention manages the KV cache like virtual memory, in fixed-size blocks allocated on demand, which removes the fragmentation and over-reservation that wasted most of the cache memory in earlier serving systems {cite}`kwon2023efficient`. Serving systems, including the cache-aware scheduling that makes this work, are the subject of the planned chapters on inference; see the {doc}`/roadmap`. For this chapter the point is that every architecture decision in {ref}`sec-ch6-attention-variants` changes the number of bytes that system must manage.

(sec-ch6-multi-token-prediction)=

### 7.3 Multi-token prediction

Standard training predicts one next token per position. {cite:t}`gloeckle2024better` add extra heads that predict tokens 2, 3 and 4 positions ahead from the same hidden state, which improved code benchmarks at larger scales and allows speculative decoding without a separate draft model. DeepSeek-V3 uses a variant in which the additional prediction is made by a one-layer transformer module that takes the main model's hidden state and the next token's embedding, keeping the causal chain {cite}`deepseek2024v3`. The module is used as a training objective; it is not needed at inference (it can be dropped, or used for speculative decoding). Its layer accounts for the 13.46B elements that are in V3's checkpoint but not in its 671B (about 11.6B of them are the module's own weights; the rest are copies of the shared embedding and output head).

(sec-ch6-embeddings-vocabulary)=

### 7.4 Embeddings and vocabulary

The embedding table and the output projection each hold $|\mathcal{V}| \cdot d$ parameters. Tying them, using one matrix for both, halves that cost; Gemma 2 does so, and many small models do. At large scale the embedding is a small part of the total and untying gives the output projection freedom, so Llama, Qwen, Mistral, OLMo and DeepSeek do not tie. The share is large for small models with large vocabularies: Gemma 2 9B's tied table has 917M parameters, 9.9% of the model, while DeepSeek-V3's two tables hold 1.9B of 671B, 0.3%.

Larger vocabularies shorten sequences (chapter 1's tokenization companion measures this in {ref}`sec-tdd-choosing-vocabulary-size`) at the cost of a larger table and a softmax over more classes. The progression from 32,000 entries in the original Llama and Mistral to 128,256 in Llama 3 and 256,000 in Gemma 2 follows from multilingual coverage and from the fact that the table became a smaller part of larger models.

(sec-ch6-accounting)=

## 8. Accounting

(sec-ch6-counting-parameters-configuration)=

### 8.1 Counting parameters from a configuration

The following function turns a Hugging Face `config.json` into exact parameter counts. It dispatches on `model_type` and refuses model types it does not know, because several details are fixed in a model's code rather than in its configuration: Qwen2's attention biases, the QK-norm weights of OLMo 2 and OLMoE (over the full projection) and of Qwen3 and Gemma 3 (one head's width, shared by all heads), the four norms of a Gemma 2 or Gemma 3 layer, the per-expert bias of DeepSeek-V3's router. The same function is run by the repository's tests and by the notebook.

```python
SUPPORTED = ("llama", "qwen2", "qwen3", "mistral", "mixtral", "olmo2", "olmoe",
             "gemma2", "gemma3_text", "deepseek_v2", "deepseek_v3")


def count_params(cfg):
    """Parameter counts from a Hugging Face config.json dictionary.

    Returns exact integers: total (what the checkpoint stores, excluding DeepSeek's
    multi-token prediction layer), and the parameters one token touches, with and
    without the input embedding (vendors differ on which they call "active").
    """
    kind = cfg["model_type"]
    if kind not in SUPPORTED:
        raise ValueError(f"unsupported model_type {kind!r}; supported: {SUPPORTED}")
    H, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    nh = cfg["num_attention_heads"]
    nkv = cfg.get("num_key_value_heads") or nh
    dh = cfg.get("head_dim") or H // nh

    # Attention, per layer
    if kind.startswith("deepseek"):  # multi-head latent attention
        dn, dr, dv = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        kv_rank, q_rank = cfg["kv_lora_rank"], cfg["q_lora_rank"]
        if q_rank is None:
            q = H * nh * (dn + dr)
        else:  # low-rank queries: down-projection, its norm, up-projection
            q = H * q_rank + q_rank + q_rank * nh * (dn + dr)
        kv = H * (kv_rank + dr) + kv_rank + kv_rank * nh * (dn + dv)
        attn = q + kv + nh * dv * H
    else:
        attn = H * nh * dh + 2 * H * nkv * dh + nh * dh * H
        if kind == "qwen2":
            attn += nh * dh + 2 * nkv * dh  # q, k and v biases (set in the model code)
        if kind in ("olmo2", "olmoe"):
            attn += nh * dh + nkv * dh  # QK-norm weights over the full q and k
        if kind in ("qwen3", "gemma3_text"):
            attn += 2 * dh  # QK-norm weights of one head, shared by all heads
    # RMSNorm weights per layer: two (before or after each sublayer), or four for
    # Gemma 2 and 3, which normalize both before and after each sublayer
    norms = (4 if kind in ("gemma2", "gemma3_text") else 2) * H

    def gated(width):  # SwiGLU feed-forward: gate, up and down projections
        return 3 * H * width

    def ffn(layer):  # (total, active) feed-forward parameters of one layer
        dense = ("llama", "qwen2", "qwen3", "mistral", "olmo2", "gemma2", "gemma3_text")
        if kind in dense:
            return gated(cfg["intermediate_size"]), gated(cfg["intermediate_size"])
        if kind in ("mixtral", "olmoe"):
            experts = cfg.get("num_local_experts") or cfg["num_experts"]
            top_k, width = cfg["num_experts_per_tok"], cfg["intermediate_size"]
            router = H * experts
            return router + experts * gated(width), router + top_k * gated(width)
        if layer < cfg["first_k_dense_replace"]:  # DeepSeek: dense layers come first
            return gated(cfg["intermediate_size"]), gated(cfg["intermediate_size"])
        assert cfg.get("moe_layer_freq", 1) == 1
        experts, top_k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
        width, shared = cfg["moe_intermediate_size"], cfg["n_shared_experts"]
        bias = experts if kind == "deepseek_v3" else 0  # V3: one bias per expert
        router = H * experts + bias
        return (router + shared * gated(width) + experts * gated(width),
                router + shared * gated(width) + top_k * gated(width))

    tied = bool(cfg.get("tie_word_embeddings"))
    body = L * (attn + norms)
    ffn_total = sum(ffn(i)[0] for i in range(L))
    ffn_active = sum(ffn(i)[1] for i in range(L))
    total = V * H + (0 if tied else V * H) + H + body + ffn_total
    # output head, final norm and layers; no input embedding
    active_without_embedding = V * H + H + body + ffn_active
    active_with_embedding = active_without_embedding + (0 if tied else V * H)
    return {"total": total, "active_with_embedding": active_with_embedding,
            "active_without_embedding": active_without_embedding}


print("Llama 3.1 8B:", format(count_params(  # the paper's architecture
    {"model_type": "llama", "hidden_size": 4096, "num_hidden_layers": 32,
     "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
     "vocab_size": 128256, "tie_word_embeddings": False})["total"], ","))
```

% output

```text
Llama 3.1 8B: 8,030,261,248
```

Everything outside the transformer blocks is the embedding, the output projection (absent when tied) and one final norm. Inside a block, the terms follow the sections above: four attention projections (with $n_{kv}$ sharing keys and values), the norms, and the feed-forward networks.

(sec-ch6-validation)=

### 8.2 Validation

A parameter counter is only useful if it is right, and "right" has three different meanings, each checked.

1. **Against the checkpoint.** `tools/fetch_arch_samples.py` reads the header of every `.safetensors` shard of each ungated model with two HTTP range requests (eight bytes for the header length, then the JSON header) and sums the tensor element counts. Counting elements rather than bytes makes the check independent of the storage type: OLMo 2 is stored in float32, DeepSeek-V3 in a mix of BF16, FP8 and float32. The counts are committed in `data/checkpoint_counts.json`. The function reproduces **all eight** to the last digit: for DeepSeek-V3, the main model's 671,026,419,200 elements, with the multi-token-prediction layer (`model.layers.61.*`) and the FP8 scale tensors (`*_scale_inv`) summed separately and excluded.
2. **Against the hub's totals for gated models.** The paper-derived configurations of Llama 3.1 8B and 70B and of Gemma 2 9B (with Gemma 2's vocabulary of 256,000 taken from the hub, not the paper's 256,128) are not downloadable, but the hub's public metadata reports each checkpoint's parameter total. The function applied to the transcribed configurations gives 8,030,261,248, 70,553,706,496 and 9,241,705,984, matching the metadata exactly. That checks the architecture description of each paper, including Gemma 2's four norms and tied embeddings. Gemma 3 12B's hub total of 12,187,325,040 also includes the vision encoder, so its check has three parts, set out after this list {cite}`gemma2025gemma3`.
3. **Against the published figures.** Each vendor's quoted total, and active count in its convention, is compared with a tolerance of half a unit in the last published digit, except where noted. In the table, "Total" and "Active" are computed; the last column gives the convention the vendor's active count matches.

```{tabularcolumns} \Y{0.21}\Y{0.17}\Y{0.09}\Y{0.18}\Y{0.16}\Y{0.19}
```

| Model | Total | Quoted | Active | Quoted active | Input embedding in active |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5 7B | 7,615,616,512 | 7.61B | — | — | — |
| Qwen3 8B | 8,190,735,360 | 8.2B | — | — | — |
| Mistral 7B v0.1 | 7,241,732,096 | 7.3B | — | — | — |
| Mixtral 8x7B | 46,702,792,704 | 46.7B | 12,879,925,248 | 12.9B | yes |
| OLMoE 1B-7B | 6,919,161,856 | 6.9B | 1,282,017,280 | 1.3B | yes |
| DeepSeek-V2-Lite | 15,706,484,224 | 15.7B | 2,451,435,008 | 2.4B | no |
| DeepSeek-V3 | 671,026,419,200 | 671B | 36,625,618,432 | 37B | no |

Three entries need a comment. Qwen's 7.61B is consistent with truncating 7.6156B, and Mistral's 7.3B, from its release announcement, with rounding 7.2417B up to the next tenth of a billion; the checks for these two use correspondingly wider tolerances (0.01B and 0.1B). DeepSeek-V2-Lite's "2.4B active" is consistent with truncating 2.451B. Qwen3 8B's model card also gives 6.95B non-embedding parameters, which is the total less both untied tables, $8{,}190{,}735{,}360 - 2 \times 151{,}936 \times 4{,}096 = 6{,}946{,}075{,}648$. OLMo 2 7B is not in the table: its model card names it 7B and quotes no more precise total, so the 7.30B of {ref}`sec-ch6-models-this-chapter` is the hub's count, which the checkpoint check already covers. These are the vendors' choices and the exact counts are in the committed regression test.

The published figures come from: the model cards of [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) ("Number of Parameters: 7.61B"), [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B-Base) ("8.2B" and "(Non-Embedding): 6.95B") and [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) ("671B" and "37B"); the OLMoE paper ("6.9B total parameters, of which only 1.3B are activated" {cite}`muennighoff2025olmoe`; the [model card](https://huggingface.co/allenai/OLMoE-1B-7B-0924) rounds these to "1B active and 7B total"); Mistral's announcements of [Mistral 7B](https://mistral.ai/news/announcing-mistral-7b) ("a 7.3B parameter model") and [Mixtral](https://mistral.ai/news/mixtral-of-experts) ("46.7B total parameters but only uses 12.9B parameters per token"); and the DeepSeek-V2 paper for V2-Lite ("15.7B total parameters, of which 2.4B are activated") {cite}`deepseek2024v2`.

**Gemma 3's total.** The decomposition of the hub's total of 12,187,325,040 parameters for Gemma 3 12B:

```{tabularcolumns} \Y{0.36}\Y{0.24}\Y{0.30}
```

| Part | Parameters | Paper |
| --- | --- | --- |
| Text decoder, 262,208 embedding rows | 11,766,034,176 | 10,759M non-embedding (Table 1) |
| SigLIP vision encoder | 416,866,032 | 417M |
| Projection into the text width, with its norm | 4,424,832 | not given |
| Sum | 12,187,325,040 | hub total, exactly |

The text decoder's non-embedding count is $11{,}766{,}034{,}176 - 262{,}208 \times 3{,}840 = 10{,}759{,}155{,}456$. The vision encoder is a SigLIP at 896 pixels without its pooling head: 27 layers of width 1,152 with a feed-forward width of 4,304, 14-pixel patches and 4,096 learned positions. The projection holds $1{,}152 \times 3{,}840 + 1{,}152 = 4{,}424{,}832$ parameters. With the reference implementation's 262,144 embedding rows the sum would fall 245,760 short. The paper's embedding count for the 12B model, 1,012M, is reproduced by neither row count ($1{,}006.6$M or $1{,}006.9$M), and the check does not use it.

The three checks catch different mistakes. The checkpoint comparison finds wrong formulas, including a forgotten bias or norm. The hub-totals comparison verifies the descriptions transcribed from papers. The published-figure comparison finds convention mismatches (the embedding in "active"), which the first two cannot, because they only concern the total.

(sec-ch6-flops-per-token)=

### 8.3 FLOPs per token

For a dense model with $N$ parameters in matrix multiplications, the forward pass costs about $2N$ FLOPs per token (one multiply and one add per weight) and the backward pass twice that, so training costs $6N$ per token {cite}`kaplan2020scaling`. Chapter 2 counts the multiply-adds of each layer from which this follows ({ref}`sec-ch2-flops`). Two refinements matter for modern models.

First, the embedding lookup is not a multiplication, so $N$ should exclude the input embedding table. The output projection is a multiplication and stays in. For a sparse model, $N$ is the *active* count, with the router included.

Second, attention adds a term proportional to the context length $n$, which the $6N$ rule omits. A token that attends to $n$ positions scores them against its query ($n\,d_{attn}$ multiply-adds per layer, with $d_{attn} = n_h d_k$ the attention width) and then takes the weighted sum of their values (another $n\,d_{attn}$). Over $L$ layers that is $4\,L\,n\,d_{attn}$ FLOPs, the count of {ref}`sec-ch2-flops`. {cite:t}`kaplan2020scaling` write the forward pass per token as $2N + 2\,L\,n\,d_{attn}$, half of this term, which is also the average over a causal sequence of $n$ tokens, where the mean token attends to $n/2$ positions; training is three times the forward pass. For Llama 3.1 8B the last token of a 131,072-token sequence attends to 131,072 earlier positions and costs $4 \times 32 \times 131{,}072 \times 4{,}096 = 68.7$ GFLOPs in the attention term of the forward pass, against $2N = 2 \times 7.50\text{B} = 15.0$ GFLOPs for the weights, 4.6 times as much. Averaged over the sequence the term is 34.4 GFLOPs, still 2.3 times $2N$. At $n = d_{attn} = 4{,}096$ the attention term is 12.5% of the total (6.7% for the average), so at short context the $6N$ rule is a good approximation and at 128K it is not. {numref}`fig-ch6-flops-context` follows both terms from short to long context.

:::{figure} figures/flops-context.*
:name: fig-ch6-flops-context
:alt: Line chart of forward GFLOPs per token against context length from 512 to 128K on a log scale. The weights' 2N is flat at 15.0. The attention term for a token that attends to n positions rises from near zero, crosses 15.0 at about 29K and reaches 68.7 at 128K; a dashed line at half of it, the causal average, crosses 15.0 at about 57K and reaches 34.4.
:width: 100%

Llama 3.1 8B's forward cost per token: the weights' $2N = 15.0$ GFLOPs against the
attention term $4\,L\,n\,d_{attn}$ of a token that attends to $n$ positions. At
$n = d_{attn} = 4{,}096$ attention is 12.5% of the total; it equals the weights near
29K tokens and is 4.6 times the weights at 128K. The dashed line, $2\,L\,n\,d_{attn}$,
is the average over a causal sequence of $n$ tokens (the term in Kaplan et al.'s count);
it equals the weights near 57K tokens.
:::

| Model | Parameters in the $6N$ rule | $6N$ (GFLOPs per token) |
| --- | --- | --- |
| Llama 3.1 8B | 8.03B, less 0.53B input embedding | 45.0 |
| Llama 3.1 70B | 70.55B, less 1.05B input embedding | 417.0 |
| Mixtral 8x7B | 12.75B active, no input embedding | 76.5 |
| DeepSeek-V3 | 36.63B active, no input embedding | 219.8 |

Training a dense model of DeepSeek-V3's size would cost $6 \times 671\text{B} = 4.0$ TFLOPs per token, 18 times as much as the 0.22 TFLOPs that the sparse model needs.

(sec-ch6-inference-memory)=

### 8.4 Memory at inference

Weights take the parameter count times the bytes per parameter: 2 for bf16, 1 for FP8 (DeepSeek-V3's native format), roughly 0.5 for 4-bit quantization (compression is a planned chapter; see the {doc}`/roadmap`). Add the KV cache from {ref}`sec-ch6-attention-variants`: bytes per token times the number of tokens in all sequences being served.

**A dense example.** Llama 3.1 8B in bf16 needs $8{,}030{,}261{,}248 \times 2 = 16.06$ GB for the weights. One sequence of 128K tokens needs $131{,}072$ bytes $\times$ $131{,}072$ tokens $= 17.18$ GB for the cache, so serving a single full-length request needs about 33 GB before activations, and a second concurrent request adds another 17 GB. The same cache with 32 key/value heads (multi-head attention) would be 69 GB per sequence, and the model would not fit at that length on a 40 GB card at all. This is the practical content of grouped-query attention.

**A sparse example.** DeepSeek-V3 in FP8 needs about 671 GB for its weights, which is why it is served on a cluster, not a card. The attention cache is the small part: $70{,}272 \times 131{,}072 = 9.2$ GB for a 128K-token sequence, compared with 655 GB had it used multi-head attention with its head shape. Compute per token is that of a 37B model; weight memory is that of a 671B one, and the batch must be large enough for every expert to receive tokens for the compute advantage to be realized.

```{admonition} The Accounting Function
:class: tip
The notebook section {ref}`nb-arch-configs` runs the function above on every
configuration, compares it with the checkpoint element counts, the hub totals and the
published figures, and prints the derived tables of this chapter, with an assertion
behind each check.
```

(sec-ch6-encoders-encoder-decoders-today)=

## 9. Encoders and Encoder-Decoders Today

Decoder-only models dominate generation, but two other designs persist where their properties fit the task.

**Encoder-only models** (BERT {cite}`devlin2018bert`) apply bidirectional attention and are trained to fill in masked tokens. They produce a representation of each input token that can see both sides, which is what classification, retrieval and embedding need, at a size and cost far below a generative model. ModernBERT brings the decoder block's improvements to the form: RoPE, GeGLU, bias-free layers, alternating local and global attention, an 8,192-token context and a modern training mix {cite}`warner2024smarter`.

**Encoder-decoder models** (T5 {cite}`raffel2019exploring`) read the whole input with a bidirectional encoder and generate with a decoder that cross-attends to it. For tasks with a clear input and output, such as translation, summarization and structured extraction, the input is encoded once and need not be re-attended with a causal mask. At the frontier scale the field has standardized on decoders because one architecture and one objective serve every use through prompting.

**Counting them.** The method of {ref}`sec-ch6-counting-parameters-configuration` carries over, with differences that the two functions below make explicit. BERT-base, from its configuration (vocabulary 30,522, 512 positions, 2 segment types, width 768, 12 layers, feed-forward width 3,072, which is $4d$), has learned position and segment embedding tables, a bias on every projection, layer norms with both a gain and a shift, a two-matrix feed-forward network and a pooler over the first token. It comes to 109,482,240 parameters, the paper's "Total Parameters=110M" {cite}`devlin2018bert`. The hub's checkpoint holds 110,106,428: the extra 624,188 are the two pretraining heads, 622,650 for masked-token prediction and 1,538 for next-sentence prediction.

An encoder-decoder gives every decoder layer a cross-attention sublayer, with its own four projections and norm, so a decoder layer is a third larger than an encoder layer of the same width. T5 drops the biases, uses norms whose "activations are only rescaled and no additive bias is applied", shares one embedding matrix between the input and the output layer, and learns 32 relative-position embeddings per head, shared across the layers of each stack {cite}`raffel2019exploring`. T5-base (width 768, 12 encoder and 12 decoder layers, 12 heads of 64, feed-forward width 3,072, and the configuration's 32,128 vocabulary entries) has 222,903,552 parameters, the paper's "about 220 million", which it describes as "roughly twice the number of parameters of BERT$_{\text{BASE}}$ since our baseline model contains two layer stacks instead of one".

```python
def encoder_params(vocab, d, layers, d_ff, positions, segments):
    """BERT-style encoder: learned position and segment tables, biases, and
    LayerNorm with gain and shift."""
    embeddings = (vocab + positions + segments) * d + 2 * d
    attention = 4 * (d * d + d)
    ffn = d * d_ff + d_ff + d_ff * d + d
    pooler = d * d + d
    return embeddings + layers * (attention + ffn + 2 * 2 * d) + pooler


def encoder_decoder_params(vocab, d, layers, d_ff, heads, head_dim, buckets=32):
    """T5-style: no biases, scale-only norms, one shared embedding, and a
    relative-position table per stack."""
    attention = 4 * d * heads * head_dim
    ffn = 2 * d * d_ff
    encoder = layers * (attention + ffn + 2 * d) + buckets * heads + d
    decoder = layers * (2 * attention + ffn + 3 * d) + buckets * heads + d
    return vocab * d + encoder + decoder, encoder, decoder


bert = encoder_params(30522, 768, 12, 3072, 512, 2)
t5, encoder, decoder = encoder_decoder_params(32128, 768, 12, 3072, 12, 64)
print(f"BERT-base: {bert:,}")
print(f"T5-base: {t5:,} (encoder {encoder:,}, decoder {decoder:,}, "
      f"ratio to BERT-base {t5 / bert:.3g})")
```

% output

```text
BERT-base: 109,482,240
T5-base: 222,903,552 (encoder 84,954,240, decoder 113,275,008, ratio to BERT-base 2.04)
```

**State-space and hybrid models** replace some or all attention layers with recurrent layers whose state does not grow with sequence length; Mamba is the best known {cite}`gu2023mamba`. They remove the KV cache's linear growth rather than shrinking it. They are outside the scope of this chapter and are planned for the roadmap's later chapters; see the {doc}`/roadmap`.

(sec-ch6-summary)=

## Summary

- A 2025 decoder block is the 2017 block with the normalization moved or added so that residual contributions stay bounded, a gated feed-forward network without biases, rotary positions, and grouped or latent key/value sharing.
- Depth and width vary with size and between families (width over layers is 128 for half of the twelve models and 80 to 85 for the Gemma models), and at fixed non-embedding parameter count the loss depends only weakly on the shape. QK-norm bounds each attention logit by $\gamma_q \gamma_k \sqrt{d_k}$ when each head is normalized, whatever the projection weights.
- In the notebook's six-layer model, post-norm fails at the highest of three learning rates without warm-up while pre-norm trains, warm-up rescues post-norm there in some seeds but not all, and at the two lower rates post-norm's loss after 400 steps is lower than pre-norm's in every seed, a comparison of early training speed rather than of where either ends.
- The key/value structure sets the KV cache: $2 n_{kv} d_k$ per token per layer for GQA and $d_c + d_{rope}$ for MLA, ranging in bf16 from 512 KiB per token (OLMo 2 7B) to 30.4 KiB (DeepSeek-V2-Lite) and 68.6 KiB (DeepSeek-V3).
- Long context is a frequency change plus a short stage of training at the new length. A context-length number from a configuration is a statement about the code.
- A mixture-of-experts layer buys parameters at constant compute; it needs a balancing mechanism, and "active parameters" depends on whether the input embedding is counted.
- Parameter counts can be computed exactly from a configuration, and should be checked three ways: against the checkpoint, against the vendor's own total, and against the vendor's own convention.
- Training costs about $6N$ FLOPs per token, with $N$ the parameters in matrix multiplications (the active ones for a sparse model, without the input embedding), plus an attention term of $4\,L\,n\,d_{attn}$ for a token that attends to $n$ positions, which for Llama 3.1 8B overtakes the weights near 29K tokens (near 57K on average over a causal sequence). Serving memory is the total parameter count times the bytes per parameter, plus the KV cache.
- Encoders and encoder-decoders remain the choice for representations and for tasks with a clear input and output; the same counting method, with their biases, learned positions and cross-attention added, gives BERT-base's 109,482,240 and T5-base's 222,903,552 parameters.

(sec-ch6-exercises)=

## Exercises

1. **Which head count?** For a model with $d = 4{,}096$, $L = 32$ layers and $d_k = 128$, tabulate the bf16 KV cache per token for $n_{kv} \in \{32, 16, 8, 4, 2, 1\}$. A 24 GB GPU holds 8 GB of weights and serves sequences of 32,768 tokens. How many such sequences fit in the remaining memory at each setting?

````{admonition} Solution 1
:class: dropdown

By the table of {ref}`sec-ch6-inference-drives-design`, one token occupies
$2\, n_{kv}\, d_k$ elements per layer, so
$2 \times n_{kv} \times 128 \times 32 \times 2 = 16{,}384\, n_{kv}$ bytes across all
layers in bf16. A sequence of 32,768 tokens needs $0.537\, n_{kv}$ GB, and 16 GB are
free after the weights.

```python
layers, head_dim, context = 32, 128, 32_768
free = 24e9 - 8e9  # bytes left for the cache after the weights
for n_kv in (32, 16, 8, 4, 2, 1):
    per_token = 2 * n_kv * head_dim * layers * 2  # K and V, all layers, 2 bytes each
    per_sequence = per_token * context
    fit = int(free // per_sequence)
    print(f"n_kv = {n_kv:2d}: {per_token:7,} bytes per token, "
          f"{per_sequence / 1e9:5.2f} GB per sequence, {fit:2d} fit")
```

% output

```text
n_kv = 32: 524,288 bytes per token, 17.18 GB per sequence,  0 fit
n_kv = 16: 262,144 bytes per token,  8.59 GB per sequence,  1 fit
n_kv =  8: 131,072 bytes per token,  4.29 GB per sequence,  3 fit
n_kv =  4:  65,536 bytes per token,  2.15 GB per sequence,  7 fit
n_kv =  2:  32,768 bytes per token,  1.07 GB per sequence, 14 fit
n_kv =  1:  16,384 bytes per token,  0.54 GB per sequence, 29 fit
```

With 32 key/value heads (multi-head attention) not even one sequence fits: its 17.18 GB
cache exceeds the 16 GB left. Each halving of $n_{kv}$ roughly doubles the number of
sequences, from 1 at $n_{kv} = 16$ to 29 at $n_{kv} = 1$; the counts are not exact
doublings because a fraction of a sequence does not fit. The common choice of 8
serves 3.
````

2. **Converting to grouped-query attention.** {cite:t}`ainslie2023gqa` convert a
   multi-head checkpoint by mean-pooling the key and value projection weights of the
   heads in each group. (a) For Llama 3.1 70B's shape ($d = 8{,}192$, $L = 80$,
   $n_h = 64$ heads of $d_k = 128$), how many key and value weights does converting to
   $n_{kv} = 8$ remove, and how does the cache per token change? (b) Show that the
   conversion leaves the attention scores unchanged when the key heads within each group
   are already equal, and explain why the paper still trains the converted model
   briefly.

````{admonition} Solution 2
:class: dropdown

(a) Each layer's key and value projections have $2\, d\, n_h\, d_k$ weights under
multi-head attention and $2\, d\, n_{kv}\, d_k$ after the conversion. (b) A query head's
scores are $x W_Q^{(j)} \big(x W_K^{(j)}\big)^{\top}$ for head $j$; if every head of a
group has the same $W_K^{(j)}$, their mean is that same matrix, and every score is
unchanged. The listing builds such a layer and also one whose key heads are independent.

```python
import numpy as np

d, layers, n_h, head_dim, n_kv = 8192, 80, 64, 128, 8
removed = layers * 2 * d * (n_h - n_kv) * head_dim  # K and V weights of all layers
print(f"key and value weights removed: {removed:,}")
before, after = (2 * n * head_dim * layers * 2 for n in (n_h, n_kv))
print(f"cache per token: {before:,} bytes before, {after:,} after")

rng = np.random.default_rng(0)
width, heads, groups, head_width, seq = 32, 8, 2, 4, 6  # a small layer
x = rng.standard_normal((seq, width))
w_q = rng.standard_normal((heads, width, head_width))


def scores(w_k):
    """Attention scores of every head, [heads, seq, seq]."""
    return np.einsum("sw,hwc,tv,hvc->hst", x, w_q, x, w_k)


def pooled(w_k):
    """Mean-pool the key heads of each group; repeat it for the group's query heads."""
    mean = w_k.reshape(groups, heads // groups, width, head_width).mean(axis=1)
    return np.repeat(mean, heads // groups, axis=0)


shared = rng.standard_normal((groups, width, head_width))
equal = np.repeat(shared, heads // groups, axis=0)
independent = rng.standard_normal((heads, width, head_width))
same = np.allclose(scores(equal), scores(pooled(equal)), rtol=1e-12, atol=1e-12)
print("equal heads, scores unchanged:", same)
original = scores(independent)
difference = original - scores(pooled(independent))
change = np.linalg.norm(difference) / np.linalg.norm(original)
print(f"independent heads, relative change of the scores: {change:.2f}")
```

% output

```text
key and value weights removed: 9,395,240,960
cache per token: 2,621,440 bytes before, 327,680 after
equal heads, scores unchanged: True
independent heads, relative change of the scores: 0.93
```

The conversion removes 9.40 billion of the 79.95 billion parameters of the multi-head
model of the same shape, leaving Llama 3.1 70B's 70.55 billion, and divides the cache
per token by 8, from the 2,621,440 bytes of multi-head attention to the 327,680 of the
table in {ref}`sec-ch6-multi-query-grouped-query-attention`. The pooled layer reproduces
the original exactly only when the heads of a group were equal. Heads trained
independently are not, and with fully independent random heads the scores move by about
as much as their own size. Mean-pooling is the starting point that keeps as much as an
average can, and the paper's short uptraining, 5% of the original pretraining compute,
lets the model adapt to the shared heads.
````

3. **A forgotten bias.** Suppose you count Qwen2.5 7B ($d = 3{,}584$, $L = 28$,
   $n_{kv} = 4$ key/value heads of $d_k = 128$) without the biases on its query, key and
   value projections ({ref}`sec-ch6-no-biases`). By how much is the count short, and
   which of the three checks of {ref}`sec-ch6-validation` notices?

````{admonition} Solution 3
:class: dropdown

The query bias has one value per output, $d$; the key and value biases have $n_{kv} d_k$
each. The listing subtracts their sum over the layers from the correct total of
{ref}`sec-ch6-validation` and applies the vendor's convention, a truncation to
hundredths of a billion.

```python
layers, d, n_kv, head_dim = 28, 3584, 4, 128
total = 7_615_616_512  # the count with the biases, which matches the checkpoint
biases = layers * (d + 2 * n_kv * head_dim)
wrong = total - biases
print(f"biases: {biases:,}; count without them: {wrong:,}")
print(f"truncated to the published precision: {int(total / 1e7) / 100}B with, "
      f"{int(wrong / 1e7) / 100}B without")
```

% output

```text
biases: 129,024; count without them: 7,615,487,488
truncated to the published precision: 7.61B with, 7.61B without
```

The count is 129,024 short. The published figure cannot see the difference: both counts
truncate to the 7.61B the vendor publishes. Qwen2.5 7B is ungated, so the hub-totals
check, which is for gated models, does not apply. The checkpoint check catches it: the
stored tensors hold exactly 7,615,616,512 elements, and the wrong count misses by
129,024. This is the kind of mistake the checkpoint comparison exists for.
````

4. **Absorbing the up-projection.** Show that for multi-head latent attention without
   RoPE, $q_m \cdot k_{m'} = \big(q_m W^{UK\top}\big) \cdot c_{m'}$, where
   $k_{m'} = c_{m'} W^{UK}$ and $W^{UK}$ is one head's key up-projection. Count the
   multiplications one head needs per decoded token over $n = 4{,}096$ cached tokens
   with and without this absorption, for DeepSeek-V3's $d_c = 512$ and content-key
   width 128. Why would RoPE applied to the reconstructed key prevent the absorption,
   and how does the decoupled rotary key restore it?

````{admonition} Solution 4
:class: dropdown

With row vectors,
$q_m \cdot k_{m'} = q_m \big(c_{m'} W^{UK}\big)^{\top}$
$= q_m W^{UK\top} c_{m'}^{\top} = \big(q_m W^{UK\top}\big) \cdot c_{m'}$.
Without absorption, every cached latent is first expanded into a key, $d_c \times 128$
multiplications per cached token, and then dotted with the query, 128 more. With
absorption the query is mapped into the latent space once, $128 \times d_c$
multiplications, and each cached latent costs one dot product of length $d_c$.

```python
import numpy as np

# latent width, content-key width of one head, cached tokens
d_c, width, n = 512, 128, 4096
rng = np.random.default_rng(0)
q = rng.standard_normal(width)
w_uk = rng.standard_normal((d_c, width)) / np.sqrt(d_c)
c = rng.standard_normal((16, d_c))  # sixteen cached latents
agree = np.allclose((c @ w_uk) @ q, c @ (w_uk @ q), rtol=1e-12, atol=1e-12)
print("the two orders agree:", agree)
without = n * (d_c * width + width)
absorbed = width * d_c + n * d_c
ratio = without / absorbed
print(f"without absorption {without:,}, with {absorbed:,}: {ratio:.0f} times fewer")
```

% output

```text
the two orders agree: True
without absorption 268,959,744, with 2,162,688: 124 times fewer
```

Absorption cuts the work of one head from 268,959,744 to 2,162,688 multiplications per
decoded token, 124 times fewer, and it is what lets the cache hold only the latent:
caching the reconstructed keys instead would give back the memory MLA saves. RoPE
rotates the query by an angle set by $m$ and the reconstructed key $c_{m'} W^{UK}$ by
one set by $m'$. Moved to the query side as above, the two rotations combine into a
rotation by the offset $m - m'$ that sits between $q_m$ and $W^{UK\top}$, so the matrix
that would be folded into the query differs for every cached token: there is no single
absorbed matrix. DeepSeek therefore leaves the 128 content dimensions unrotated, so that
they absorb as above, and adds a separate 64-dimensional key, computed from the hidden
state, rotated and cached as it is. The score is the sum of the two dot products, and
only the small rotary part needs positions.
````

5. **YaRN for DeepSeek-V3.** DeepSeek-V3's configuration applies YaRN to its rotary
   dimensions with $d_{rope} = 64$, $b = 10{,}000$, $n_{train} = 4{,}096$,
   $s_{ctx} = 40$, $\beta_{fast} = 32$ and $\beta_{slow} = 1$. Using the ramp $\psi$ of
   {ref}`sec-ch6-trained-length-matters`, how many of the 32 pairs are kept, blended and
   interpolated? By what factor is the lowest frequency divided, and how does its
   largest angle over the 163,840 configured positions compare with the largest angle it
   reached in training?

````{admonition} Solution 5
:class: dropdown

The listing repeats the chapter's `yarn_frequencies` with the configuration's values;
the pair index runs over the 64 rotary dimensions, so $\omega_i = b^{-2i/64}$.

```python
import math


def yarn_frequencies(d, base, n_train, scale, beta_slow=1.0, beta_fast=32.0):
    """YaRN's per-pair frequencies, each with the ramp value psi of its pair."""
    out = []
    for i in range(d // 2):
        omega = base ** (-2 * i / d)
        xi = n_train * omega / (2 * math.pi)  # turns within the trained length
        psi = min(1.0, max(0.0, (xi - beta_slow) / (beta_fast - beta_slow)))
        out.append(((1 - psi) * omega / scale + psi * omega, psi))
    return out


d, base, n_train, scale = 64, 10000.0, 4096, 40
pairs = yarn_frequencies(d, base, n_train, scale)
kept = sum(psi == 1.0 for _, psi in pairs)
interpolated = sum(psi == 0.0 for _, psi in pairs)
blended = len(pairs) - kept - interpolated
print(f"kept {kept}, blended {blended}, interpolated {interpolated}")
lowest_old, lowest_new = base ** (-(d - 2) / d), pairs[-1][0]
print(f"lowest frequency divided by {lowest_old / lowest_new:.4g}")
print(f"largest angle, lowest pair: {n_train * lowest_old:.4g} rad in training, "
      f"{n_train * scale * lowest_new:.4g} rad at {n_train * scale:,} positions")
```

% output

```text
kept 11, blended 12, interpolated 9
lowest frequency divided by 40
largest angle, lowest pair: 0.5462 rad in training, 0.5462 rad at 163,840 positions
```

Eleven pairs keep their frequency, 12 are blended and 9 are fully interpolated: fewer
than in the chapter's $d_k = 128$ example because there are half as many pairs. The
lowest frequency is divided by exactly 40, so over the 163,840 configured positions it
turns through 0.5462 rad, the same largest angle it reached in the 4,096 trained
positions. The low-frequency pairs therefore see no angle outside training, while the 11
kept pairs, each completing more than 32 turns within the trained length, had already
seen every angle.
````

6. **Capacity and the auxiliary loss.** A batch of $T = 4{,}096$ tokens is routed top-2
   over $E = 8$ experts, and the router sends a quarter of all routed slots to expert 1
   and spreads the rest evenly. Using the capacity formula of
   {ref}`sec-ch6-keeping-experts-balanced`, how many of expert 1's tokens are dropped at
   capacity factors $\phi = 1.0$ and $\phi = 1.25$? What is the Switch loss
   $E \sum_i f_i \bar{\rho}_i$ (without $\lambda_{aux}$) if the mean probabilities
   $\bar{\rho}_i$ equal the dispatch fractions $f_i$, and what is it under uniform
   routing?

````{admonition} Solution 6
:class: dropdown

There are $n_{act} T = 8{,}192$ routed slots, so expert 1 receives 2,048 and each of the
other seven 878 (to the nearest token). The capacity is
$\kappa = \lceil \phi\, n_{act}\, T / E \rceil$.

```python
import math

tokens, experts, k = 4096, 8, 2
slots = k * tokens
f = [0.25] + [0.75 / 7] * 7  # dispatch fractions over the routed slots; they sum to 1
for phi in (1.0, 1.25):
    capacity = math.ceil(phi * k * tokens / experts)
    dropped = max(0, round(f[0] * slots) - capacity)
    print(f"phi = {phi}: capacity {capacity:,}, expert 1 drops {dropped:,} tokens")
uniform = [1 / experts] * experts
print(f"Switch loss, skewed router: {experts * sum(fi * fi for fi in f):.4g}")
print(f"Switch loss, uniform router: {experts * sum(fi * fi for fi in uniform):.4g}")
```

% output

```text
phi = 1.0: capacity 1,024, expert 1 drops 1,024 tokens
phi = 1.25: capacity 1,280, expert 1 drops 768 tokens
Switch loss, skewed router: 1.143
Switch loss, uniform router: 1
```

At $\phi = 1.0$ each expert holds 1,024 tokens, and expert 1 drops 1,024 of its 2,048,
half of them. At $\phi = 1.25$ it holds 1,280 and drops 768, while each of the other
seven experts receives 878 tokens and pads the remaining 402 places of its buffer. If
$\bar{\rho}$ tracks $f$, the loss is $E \sum_i f_i^2$, 1.143 here against 1 for uniform
routing; it exceeds 1 for any uneven routing, so its gradient lowers expert 1's
probability. As {ref}`sec-ch6-keeping-experts-balanced` shows, the converse fails when
$\bar{\rho}$ does not track $f$.
````

7. **Auxiliary-loss-free balancing.** Implement the bias update of
   {cite:t}`wang2024auxiliary` for a router whose sigmoid affinities favor some of its
   16 experts, with top-2 selection: after each batch, lower the selection bias of each
   overloaded expert by a fixed step and raise that of each underloaded one. How loaded
   are the most and least used experts, relative to a perfectly even share, before the
   update and after 200 batches?

````{admonition} Solution 7
:class: dropdown

The affinity of each token to each expert is a sigmoid of a random logit plus a fixed
preference that grows from 0 for the first expert to 1 for the last. The bias enters
only the choice of experts, as in {ref}`sec-ch6-fine-grained-shared-experts`; the gate
weights would still come from the unbiased affinities.

```python
import numpy as np

tokens, experts, k, batches, step = 4096, 16, 2, 200, 0.01
rng = np.random.default_rng(0)
preference = np.linspace(0.0, 1.0, experts)
bias = np.zeros(experts)


def load(bias):
    """Tokens each expert receives in a fresh batch, as multiples of the even share."""
    affinity = 1 / (1 + np.exp(-(rng.standard_normal((tokens, experts)) + preference)))
    # the bias changes the selection only
    chosen = np.argsort(-(affinity + bias), axis=1)[:, :k]
    return np.bincount(chosen.ravel(), minlength=experts) / (tokens * k / experts)


before = load(bias)
for _ in range(batches):
    bias -= step * np.sign(load(bias) - 1)  # lower overloaded, raise underloaded
after = load(bias)
print(f"before: most loaded {before.max():.3g}, least loaded {before.min():.3g} "
      "times the even share")
print(f"after {batches} batches: most loaded {after.max():.3g}, "
      f"least loaded {after.min():.3g}")
```

% output

```text
before: most loaded 1.97, least loaded 0.348 times the even share
after 200 batches: most loaded 1.16, least loaded 0.857
```

Before any update the most favored expert receives 1.97 times an even share and the
least favored 0.348 times. After 200 batches the spread is 1.16 to 0.857: the bias has
offset the preference without any loss term, so no gradient competes with the
language-modeling objective. The load does not settle at exactly 1, because each batch
is a fresh sample and the bias moves by a fixed step, so it keeps oscillating around the
balancing values; a smaller step settles closer but takes more batches.
````

8. **Active parameters.** From the totals and active counts of
   {ref}`sec-ch6-validation`, and the input-embedding sizes (vocabulary times width:
   32,000 × 4,096 for Mixtral, 50,304 × 2,048 for OLMoE, 102,400 × 2,048 for
   DeepSeek-V2-Lite and 129,280 × 7,168 for DeepSeek-V3), compute the share of active
   parameters under both conventions for each sparse model. Which count goes into the
   $6N$ rule of {ref}`sec-ch6-flops-per-token`, and which number sets the memory needed
   to serve the model?

````{admonition} Solution 8
:class: dropdown

The table gives one convention for each model; adding or subtracting the input embedding
gives the other.

```python
# total, active, whether the active count includes the input embedding, embedding size
models = {
    "Mixtral 8x7B": (46_702_792_704, 12_879_925_248, True, 32_000 * 4_096),
    "OLMoE 1B-7B": (6_919_161_856, 1_282_017_280, True, 50_304 * 2_048),
    "DeepSeek-V2-Lite": (15_706_484_224, 2_451_435_008, False, 102_400 * 2_048),
    "DeepSeek-V3": (671_026_419_200, 36_625_618_432, False, 129_280 * 7_168),
}
for name, (total, active, includes, embedding) in models.items():
    with_embedding = active if includes else active + embedding
    without = with_embedding - embedding
    share_with, share_without = 100 * with_embedding / total, 100 * without / total
    print(f"{name:16s} with embedding {share_with:5.2f}%, "
          f"without {share_without:5.2f}%")
```

% output

```text
Mixtral 8x7B     with embedding 27.58%, without 27.30%
OLMoE 1B-7B      with embedding 18.53%, without 17.04%
DeepSeek-V2-Lite with embedding 16.94%, without 15.61%
DeepSeek-V3      with embedding  5.60%, without  5.46%
```

The $6N$ rule counts parameters in matrix multiplications, so it takes the count without
the input embedding (the output head stays in), the 12.75B for Mixtral in the table of
{ref}`sec-ch6-flops-per-token`. Serving memory is set by neither active count: every
expert must be held, so the total is what multiplies the bytes per parameter, 46.7B for
Mixtral and 671B for DeepSeek-V3. The two conventions differ least for the largest
models (5.60% against 5.46% for DeepSeek-V3) and most for OLMoE (18.53% against 17.04%),
whose input embedding is a larger share of its total (1.5%) than in any of the other
three.
````

(sec-ch6-further-reading)=

## Further Reading

(sec-ch6-further-reading-models-technical-reports)=

### Models and technical reports

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
  {cite}`grattafiori2024llama` - the architecture table, long-context staging and
  scaling choices of Llama 3 and 3.1
- [Mistral 7B](https://arxiv.org/abs/2310.06825) {cite}`jiang2023mistral` and [Mixtral
  of Experts](https://arxiv.org/abs/2401.04088) {cite}`jiang2024mixtral` -
  sliding-window attention, the rolling cache, and a small sparse model
- [Gemma 2](https://arxiv.org/abs/2408.00118) {cite}`gemma2024gemma2` - interleaved
  local/global attention, soft-capping and sandwich norms
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) {cite}`qwen2024qwen25` -
  a dense family with YaRN and long-context extension
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) {cite}`olmo2024olmo2` - reordered
  norms, QK-norm and the stability changes behind them, with open data and code
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) {cite}`qwen2025qwen3` and
  [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) {cite}`gemma2025gemma3` -
  two 2025 families: QK-norm in place of QKV biases and soft-capping, and five local
  layers per global one
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) {cite}`deepseek2024v2` and
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437) {cite}`deepseek2024v3` - multi-head
  latent attention, fine-grained MoE, multi-token prediction and FP8 training

(sec-ch6-further-reading-attention-positions)=

### Attention and positions

- [Fast Transformer Decoding: One Write-Head is All You
  Need](https://arxiv.org/abs/1911.02150) {cite}`shazeer2019fast` and
  [GQA](https://arxiv.org/abs/2305.13245) {cite}`ainslie2023gqa` - multi-query and
  grouped-query attention
- [Longformer](https://arxiv.org/abs/2004.05150) {cite}`beltagy2020longformer` and
  [Efficient Streaming Language Models with Attention
  Sinks](https://arxiv.org/abs/2309.17453) {cite}`xiao2024efficient` - windows and sinks
- [Extending Context Window of Large Language Models via Positional
  Interpolation](https://arxiv.org/abs/2306.15595) {cite}`chen2023extending` and
  [YaRN](https://arxiv.org/abs/2309.00071) {cite}`peng2024yarn` - extending RoPE
- [FlashAttention](https://arxiv.org/abs/2205.14135) {cite}`dao2022flashattention`,
  [FlashAttention-2](https://arxiv.org/abs/2307.08691) {cite}`dao2024flashattention2`
  and [PagedAttention](https://arxiv.org/abs/2309.06180) {cite}`kwon2023efficient` - the
  kernels and the cache manager

(sec-ch6-further-reading-normalization-activations)=

### Normalization and activations

- [On Layer Normalization in the Transformer
  Architecture](https://arxiv.org/abs/2002.04745) {cite}`xiong2020layer` - why pre-norm
  needs no warm-up
- [Scaling Vision Transformers to 22 Billion
  Parameters](https://arxiv.org/abs/2302.05442) {cite}`dehghani2023scaling` - QK-norm
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
  {cite}`shazeer2020glu` - the gated feed-forward network

(sec-ch6-further-reading-mixture-of-experts)=

### Mixture of experts

- [GShard](https://arxiv.org/abs/2006.16668) {cite}`lepikhin2021gshard`, [Switch
  Transformers](https://arxiv.org/abs/2101.03961) {cite}`fedus2022switch` and
  [ST-MoE](https://arxiv.org/abs/2202.08906) {cite}`zoph2022st` - routing,
  load-balancing loss, capacity and router z-loss
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368)
  {cite}`zhou2022mixture` - experts choose tokens
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) {cite}`dai2024deepseekmoe` and
  [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)
  {cite}`wang2024auxiliary` - fine-grained, shared and bias-balanced experts
- [OLMoE](https://arxiv.org/abs/2409.02060) {cite}`muennighoff2025olmoe` - a fully open
  mixture-of-experts model with 64 experts per layer, its training data and ablations of
  the routing choices

(sec-ch6-further-reading-other-designs)=

### Other designs

- [Better & Faster Large Language Models via Multi-token
  Prediction](https://arxiv.org/abs/2404.19737) {cite}`gloeckle2024better` - extra heads
  that predict several tokens ahead, the idea behind DeepSeek-V3's multi-token module
- [Smarter, Better, Faster, Longer (ModernBERT)](https://arxiv.org/abs/2412.13663)
  {cite}`warner2024smarter` - an encoder rebuilt with RoPE, GeGLU and alternating local
  and global attention
- [Mamba](https://arxiv.org/abs/2312.00752) {cite}`gu2023mamba` - a selective
  state-space model whose recurrent state replaces the KV cache

(sec-ch6-further-reading-tools)=

### Tools

- [Hugging Face Transformers model implementations](
  https://github.com/huggingface/transformers/tree/main/src/transformers/models) -
  the reference code for every configuration key discussed here
- [Safetensors format](https://huggingface.co/docs/safetensors) - the header layout read
  by this chapter's fetch script

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
