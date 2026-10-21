# Modern Transformer Architectures

Chapter 1 built the transformer as it was drawn in 2017, and listed in one section what changed. This chapter takes that list apart. A decoder-only model released in 2024 or 2025 is still recognisably the 2017 block, but nearly every part of it has been replaced for a reason: to keep a 70-billion-parameter run from diverging, to shrink the cache that dominates serving memory, to stretch the context window, or to buy more parameters per unit of compute. We go through those replacements one at a time, put numbers on each from the published configuration files of real models, and finish with the accounting that turns a configuration into parameters, FLOPs and bytes.

## Learning Objectives

By the end of this chapter, you will understand:

- How the 2025 decoder block differs from the 2017 block: pre-norm, RMSNorm, post-sublayer and QK normalization, gated feed-forward networks, no biases
- How multi-head, multi-query, grouped-query and multi-head latent attention trade quality against the size of the KV cache, and how to compute that size for any configuration
- How sliding-window and interleaved local/global attention bound the cost of long sequences
- How rotary embeddings are extended to longer contexts by interpolation, NTK-aware scaling and YaRN, and what a context-length figure does and does not promise
- How mixture-of-experts layers route tokens, why they need a load-balancing mechanism, and the difference between total and active parameters
- How to count the parameters of a model from its `config.json`, exactly, and why published parameter figures use different conventions
- Where encoder and encoder-decoder models are still the right choice

## Hands-On Learning

The accompanying notebook, [Modern Architectures Demo](examples/architecture_demo.ipynb), builds a small LLaMA-style model and checks it against independent references:

- Reading seven real configurations, adding three transcribed from papers, and reproducing all ten parameter counts exactly ({ref}`nb-arch-configs`)
- RMSNorm, grouped-query attention and a pre-norm block, each checked against a reference implementation ({ref}`nb-arch-block`)
- A KV cache whose incremental decoding matches the full forward pass, and a simplified latent cache ({ref}`nb-arch-kvcache`)
- RoPE, position interpolation and NTK-aware scaling, with the frequency identities asserted ({ref}`nb-arch-rope`)
- A top-2 mixture-of-experts router, its load-balancing loss and capacity dropping ({ref}`nb-arch-moe`)
- A seeded comparison of post-norm and pre-norm training on a small text ({ref}`nb-arch-train`)

Everything runs on a CPU. The configuration cell reads the model files from the Hugging Face hub when it can and from committed copies when it cannot, so the notebook gives the same results offline.

## Chapter Overview

Read a modern model's `config.json` as a list of decisions. Each key is the answer to a question that the 2017 paper left open or answered differently.

**Normalization and the residual stream** decide whether a very deep stack trains at all. Normalizing inside the residual branch rather than after the addition keeps a clean path for the gradient; normalizing queries and keys, or normalizing the output of each sublayer, removes the instabilities that appear when activations grow.

**The feed-forward network** holds most of the parameters. Gated activations replaced the plain ReLU or GELU layer, and the hidden width is chosen so that three matrices cost about what two used to.

**Attention variants** are mostly about inference. The keys and values of every previous token are kept in memory while generating, and the number of key/value heads, or the size of a compressed latent, sets that memory.

**Positions and long context** are handled by rotating queries and keys. Making a model work at four times its training length is a matter of changing the rotation frequencies, then training a little at the new length.

**Mixture of experts** replaces one feed-forward network by many and sends each token to a few, so parameters grow without the compute per token growing with them.

**Accounting** is the discipline that keeps all of this honest: a function from a configuration to parameters, FLOPs and bytes, checked against the real checkpoints.

The theme throughout is that architecture choices are constraints from somewhere else in the system. The head count is chosen for the cache, the expert count for the interconnect, the norm placement for the learning rate. A model table is therefore worth reading column by column.

## 1. From the 2017 Block to the 2025 Block

### 1.1 The models in this chapter

Ten models are used as running examples. They span dense and sparse designs, three attention families and six organisations. Every figure below traces to a pinned `config.json` or to the model's paper, with the paper-derived rows expressed in the conventions of the Hugging Face configurations (for example vocabulary entries, and the width of one gate or up projection).

All ten have the same outer structure: a token embedding, a stack of identical blocks, a final normalization, and an output projection to the vocabulary. Within a block, a normalization, an attention sublayer and a residual addition are followed by a normalization, a feed-forward sublayer and a second residual addition. Only the contents of the boxes differ.

**Dense models.**

| Model | Layers | $d_{model}$ | Query / KV heads × head dim | FFN width | Vocabulary | RoPE base | Context | Parameters | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama 3.1 8B | 32 | 4,096 | 32 / 8 × 128 | 14,336 | 128,256 | 500,000 | 128K | 8.03B | paper |
| Llama 3.1 70B | 80 | 8,192 | 64 / 8 × 128 | 28,672 | 128,256 | 500,000 | 128K | 70.55B | paper |
| Mistral 7B v0.1 | 32 | 4,096 | 32 / 8 × 128 | 14,336 | 32,000 | 10,000 | 8,192 (paper), 32,768 (config) | 7.24B | config |
| Qwen2.5 7B | 28 | 3,584 | 28 / 4 × 128 | 18,944 | 152,064 | 1,000,000 | 131,072 (config; 32,768 trained, longer with YaRN) | 7.62B | config |
| Gemma 2 9B | 42 | 3,584 | 16 / 8 × 256 | 14,336 | 256,000 | not stated | 8,192 | 9.24B | paper |
| OLMo 2 7B (1124) | 32 | 4,096 | 32 / 32 × 128 | 11,008 | 100,352 | 500,000 | 4,096 | 7.30B | config |

**Sparse (mixture-of-experts) models.**

| Model | Layers | $d_{model}$ | Attention | Experts per MoE layer | Dense layers | Total | Active | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mixtral 8x7B v0.1 | 32 | 4,096 | GQA, 32 / 8 × 128 | 8 routed, top 2, width 14,336 | 0 | 46.70B | 12.88B (with embedding) | config |
| OLMoE 1B-7B | 16 | 2,048 | MHA, 16 × 128 | 64 routed, top 8, width 1,024 | 0 | 6.92B | 1.28B (with embedding) | config |
| DeepSeek-V2-Lite | 27 | 2,048 | MLA, 16 heads | 64 routed + 2 shared, top 6, width 1,408 | 1 | 15.71B | 2.45B (without embedding) | config |
| DeepSeek-V3 | 61 | 7,168 | MLA, 128 heads | 256 routed + 1 shared, top 8, width 2,048 | 3 | 671.03B | 36.63B (without embedding) | config |

A few notes on reading the tables.

- **Source.** "config" means the value is read from the model's `config.json` at the pinned commit listed in `data/pins.json`. "paper" means the architecture is taken from the model's paper, because the repository is gated (it needs an account and a licence acceptance). A paper-derived row is not unverified: the hub reports the exact parameter total of a gated checkpoint in its public metadata, and the configurations transcribed from the papers reproduce those totals to the last digit (Section 8.2).
- **Parameters** are exact element counts divided by $10^9$, from the checkpoints' tensor shapes rather than from rounded marketing figures. Section 8 explains the "with" and "without" embedding convention for active parameters.
- **Vocabulary.** The Llama 3 paper describes a 128K vocabulary; the configuration has 128,256 entries, the 128,000 of the tokenizer plus 256 reserved special tokens. Gemma 2's configuration has 256,000 entries and its paper rounds this to 256K. Likewise Gemma 2's paper lists the feed-forward width as the combined 28,672 of the gate and up projections, which is twice the 14,336 of each in the configuration and the table.
- **Context.** Mistral 7B v0.1's paper reports training at 8,192 tokens with a 4,096-token sliding window; its configuration advertises 32,768 positions. Both are quoted because they answer different questions: what the model saw, and what the code accepts (Section 5.4).
- **Gemma 2's head dimension.** Sixteen heads of 256 give an attention width of 4,096, larger than the model width of 3,584. The head width is a free parameter, not $d_{model}/h$.
- **DeepSeek-V3's total.** The 671.03B counts the main model. The released checkpoint also contains a 13.46B-element multi-token-prediction layer (Section 7.3), about 1.85B of which are copies of the embedding and output head that the paper shares with the main model, and 41.5M FP8 scale factors; the model card's "685B" includes both.

Beyond shapes, the models make different design choices, summarised here and developed in the sections that follow.

| Model | Norm placement | QK-norm | Attention biases | Tied embeddings | Attention pattern |
| --- | --- | --- | --- | --- | --- |
| Llama 3.1 | pre | no | no | no | full causal |
| Mistral 7B v0.1 | pre | no | no | no | sliding window of 4,096 |
| Qwen2.5 | pre | no | q, k, v | no | full causal |
| Gemma 2 | pre and post | no | no | yes | alternating local (4,096) and global |
| OLMo 2 | post-sublayer | yes | no | no | full causal |
| Mixtral | pre | no | no | no | full causal (config: no window) |
| OLMoE | pre | yes | no | no | full causal |
| DeepSeek-V2/V3 | pre | on the latent (Section 4.4) | no | no | full causal |

```{admonition} The Tables, Computed
:class: tip
The notebook section {ref}`nb-arch-configs` loads the configurations of the ungated models, rebuilds the parameter counts, and computes the derived columns (embedding share, feed-forward share, KV-cache bytes per token) used throughout this chapter.
```

### 1.2 What did not change

It is worth stating what is *constant*, because it explains why the field moved by small edits rather than redesigns. Every model above is autoregressive, uses causal self-attention, a residual stream of width $d_{model}$ that every sublayer reads from and writes to, and a learned token embedding. There are no recurrent connections, no convolutions in the main path and no cross-attention. The residual stream is the shared bus, and almost every innovation in this chapter is a change to what is plugged into it or to how the plug is normalized.

## 2. Normalization and the Residual Stream

### 2.1 Pre-norm and post-norm

The original transformer applied layer normalization *after* adding the sublayer output to the residual (post-norm): $x \leftarrow \mathrm{Norm}(x + F(x))$. Every layer's output passes through a normalization on the way to the next, which means the gradient reaching early layers is rescaled at each step. At initialization the gradients near the output are large, and the model needs a learning-rate warm-up to avoid diverging {cite}`xiong2020layer`.

Pre-norm moves the normalization inside the branch: $x \leftarrow x + F(\mathrm{Norm}(x))$. The residual path from the embedding to the output is then an unnormalized identity, so gradients reach every layer directly. Xiong et al. show that this keeps the gradient scale stable at initialization and makes warm-up optional. The price is that the residual stream's magnitude grows with depth, so the representation of the last layers can be dominated by the accumulated sum; this is why every pre-norm model adds a final normalization before the output projection. The notebook section {ref}`nb-arch-train` trains both arrangements side by side.

Post-norm is not extinct, and the question is revisited every few years, but the practical argument favours pre-norm: a diverged run on a cluster costs far more than any small quality difference at a size where both train.

### 2.2 RMSNorm

Layer normalization subtracts the mean and divides by the standard deviation. RMSNorm drops the mean subtraction and the bias {cite}`zhang2019root`:

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}$$

The argument is that the re-scaling is what matters and the re-centring is not. It has one parameter vector instead of two and a cheaper forward and backward pass; it is what every model in the table uses. Chapter 1 gives the implementation, and Chapter 2 derives the backward pass of layer normalization. In the parameter count, each RMSNorm contributes $d_{model}$ weights.

### 2.3 Normalizing queries and keys

Attention logits are dot products $q \cdot k / \sqrt{d_h}$. If a few directions in $q$ and $k$ grow large during training, the softmax saturates into a one-hot distribution, gradients through it vanish, and the loss spikes. Normalizing the queries and keys before the dot product (QK-norm) was proposed for translation models by Henry et al. in 2020, and Dehghani et al. used it to stabilise a 22-billion-parameter vision transformer {cite}`dehghani2023scaling`. It bounds the logits regardless of how the projection weights drift.

OLMo 2 {cite}`olmo2024olmo2` and OLMoE (whose configuration and model code show the same layout) apply an RMSNorm to the full query and key projections before they are split into heads; in the parameter count this adds $n_h d_h$ weights for the queries and $n_{kv} d_h$ for the keys per layer. DeepSeek's multi-head latent attention normalizes the compressed latents instead (Section 4.4).

### 2.4 Normalizing sublayer outputs

A second family places the normalization on the *output* of each sublayer, before it is added to the residual. OLMo 2 does this in its "reordered" arrangement, $x \leftarrow x + \mathrm{Norm}(F(x))$ for both attention and feed-forward, and combines it with QK-norm; the authors report it as one of the changes that removed the loss spikes of their first-generation recipe {cite}`olmo2024olmo2`. Gemma 2 uses both: a normalization before each sublayer and another after it, four per layer {cite}`gemma2024gemma2`. The sandwich arrangement was introduced for image-token transformers in CogView, again to control instabilities {cite}`ding2021cogview`.

All of these keep the *magnitude* of what each layer adds to the residual stream bounded. That is the common thread: the stream is a running sum of terms, and the stability techniques of the last five years are ways of controlling the size of the terms.

### 2.5 Initialization of the residual branches

The residual stream is a sum of $2L$ terms. If each term has unit variance at initialization, the sum's variance grows as $2L$. GPT-2 scales the weights of the residual projections by $1/\sqrt{N}$ at initialization, with $N$ the number of residual layers {cite}`radford2019language`, and most codebases do something equivalent for the output projections of attention and the feed-forward network. Chapter 2's section on initialization derives why this keeps the signal and gradient variance constant with depth.

```{admonition} Norm Placement in Practice
:class: note
The notebook section {ref}`nb-arch-block` checks the chapter's RMSNorm against PyTorch's reference, and {ref}`nb-arch-train` trains a six-layer post-norm and pre-norm model from the same seeds and learning rate, reporting the loss of each over five seeds.
```

## 3. The Feed-Forward Network

### 3.1 From ReLU to gated units

The 2017 feed-forward network is $W_2\,\mathrm{ReLU}(W_1 x)$ with a hidden width of $4 d_{model}$. GPT-1, GPT-2 and BERT swapped ReLU for GELU {cite}`hendrycks2016gaussian`. Shazeer then tested gated variants, in which the hidden activation is the elementwise product of two projections, one passed through a nonlinearity {cite}`shazeer2020glu`:

$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(x W_{gate}) \odot x W_{up}\big) W_{down}$$

With a GELU gate this is GeGLU, which Gemma uses; with SiLU it is SwiGLU, which every other model in the table uses. At matched parameter count the gated forms gave consistently lower loss, and the paper's own explanation is candid: "we offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence." The gain has held up in every large-scale replication since.

### 3.2 Width

A gated unit has three matrices instead of two, so to keep the parameter count at that of a $4d$ feed-forward network the hidden width is reduced to $\tfrac{8}{3} d_{model}$. In practice the width is rounded to a hardware-friendly multiple, and then chosen freely. OLMo 2 7B's width of 11,008 is $\tfrac{8}{3} \times 4096$ rounded up to a multiple of 256, as in the first LLaMA. The later designs go wider: 14,336 is $3.5 d_{model}$ for Llama 3.1 and Mistral, and Qwen2.5 7B uses $5.3 d_{model}$.

Each feed-forward network has $3\,d_{model} \cdot d_{ffn}$ parameters. Across the dense models above, that is between 59% (OLMo 2 7B) and 80% (Llama 3.1 70B) of all parameters; the notebook computes each share. The feed-forward network is where most of a dense model's capacity, and most of its FLOPs, live, which is also why it is the part that mixture-of-experts models replace (Section 6).

### 3.3 No biases

Linear layers in these models have no bias terms. The biases add few parameters but cost memory traffic and, with some parallelism schemes, extra communication, and the models that dropped them report no loss from doing so. The exception in the table is Qwen2.5, which keeps biases on the query, key and value projections only; this detail is fixed in the model code rather than in `config.json`, a pattern that matters for the parameter count in Section 8.

## 4. Attention Variants

### 4.1 Why inference drives the design

During generation, each new token attends to all previous tokens. Recomputing their keys and values at every step would cost quadratic time, so every serving system stores them: the KV cache. For one sequence the cache holds, per layer, one key vector and one value vector per key/value head for every token seen so far. It is read in full at every decoding step, so it costs both memory and bandwidth; at long context and large batch it, not the weights, sets the number of concurrent users a GPU can serve.

In elements, the cache of one token in one layer is:

| Variant | Elements per token per layer |
| --- | --- |
| Multi-head attention (MHA) | $2\, n_h\, d_h$ |
| Grouped-query attention (GQA) with $n_{kv}$ groups | $2\, n_{kv}\, d_h$ |
| Multi-query attention (MQA) | $2\, d_h$ |
| Multi-head latent attention (MLA) | $d_c + d_h^{R}$ |

Multiplying by the number of layers and the bytes per element (2 for bf16) gives the bytes per token. The following function implements the table for a Hugging Face configuration dictionary.

```python
def kv_cache_elements_per_token(cfg):
    """KV-cache elements one token occupies across all layers of a decoder model."""
    layers = cfg["num_hidden_layers"]
    if cfg["model_type"].startswith("deepseek"):  # MLA caches one latent and one rope key
        return layers * (cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"])
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads") or n_heads
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    return layers * 2 * n_kv * head_dim
```

### 4.2 Multi-query and grouped-query attention

Multi-query attention shares a single key head and a single value head across all query heads, shrinking the cache by a factor of $n_h$ {cite}`shazeer2019fast`. It speeds up decoding substantially, at some cost in quality and in training stability. Grouped-query attention interpolates: the $n_h$ query heads are divided into $n_{kv}$ groups, each sharing one key and one value head {cite}`ainslie2023gqa`. The paper also shows that an existing multi-head checkpoint can be converted to grouped-query attention by mean-pooling the key and value heads of each group and training briefly. Llama 2's two largest models adopted it, and every Llama since has {cite}`touvron2023llama2`.

The choice of $n_{kv}$ is a direct dial on the cache. Llama 3.1 8B uses 8 key/value heads for 32 query heads, a 4× reduction, and Llama 3.1 70B uses 8 for 64, an 8× reduction. The cache per token (bf16, all layers) is:

| Model | KV elements per token | Bytes per token | Compared with MHA |
| --- | --- | --- | --- |
| OLMo 2 7B (MHA) | 262,144 | 524,288 | 1× |
| Mistral 7B, Mixtral, Llama 3.1 8B (GQA, 8 KV heads) | 65,536 | 131,072 | 4× smaller |
| Qwen2.5 7B (GQA, 4 KV heads) | 28,672 | 57,344 | 7× smaller |
| Gemma 2 9B (GQA, 8 KV heads of 256) | 172,032 | 344,064 | 2× smaller |
| Llama 3.1 70B (GQA, 8 KV heads) | 163,840 | 327,680 | 8× smaller |
| DeepSeek-V2-Lite (MLA) | 15,552 | 31,104 | 8.9× smaller |
| DeepSeek-V3 (MLA) | 35,136 | 70,272 | 71× smaller |

"Compared with MHA" divides the cache the same model would need with one key/value head per query head by the actual cache. For Gemma 2, the full-cache figure assumes every layer keeps the entire context; its local layers can in fact discard tokens beyond the window (Section 4.5), so the real cache at long context is smaller.

At 128K tokens, Llama 3.1 8B's cache for a single sequence is $131{,}072 \times 131{,}072 = 17.2$ GB, more than its 16.1 GB of bf16 weights. Section 8 works through this.

### 4.3 What grouped-query attention does to the computation

The attention scores themselves are unchanged; only the keys and values are shared. The following sketch shows the whole idea: expand the key and value heads to match the query heads, then run ordinary attention.

```python
import torch
import torch.nn.functional as F


def grouped_query_attention(q, k, v):
    """q: [batch, n_heads, seq, d_h]; k, v: [batch, n_kv, seq, d_h], n_heads % n_kv == 0."""
    group = q.size(1) // k.size(1)
    k = k.repeat_interleave(group, dim=1)  # each KV head serves `group` adjacent query heads
    v = v.repeat_interleave(group, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Production kernels avoid materialising the repeated tensors, but the result is the same. That equivalence is what the notebook asserts ({ref}`nb-arch-block`): grouped-query attention equals multi-head attention whose key and value weights are the grouped weights repeated, to floating-point precision, and multi-query attention is the `n_kv = 1` case.

### 4.4 Multi-head latent attention

DeepSeek-V2 reduces the cache by a different route: instead of sharing heads, it compresses them {cite}`deepseek2024v2`. The hidden state $h_t$ is projected down to a latent vector $c_t = W^{DKV} h_t$ of size $d_c$ (512 in both V2-Lite and V3). Keys and values for all heads are reconstructed from the latent by up-projections, $k_t = W^{UK} c_t$ and $v_t = W^{UV} c_t$, and only $c_t$ is cached. At inference the up-projection of the keys can be absorbed into the query projection and that of the values into the output projection, so the full keys and values are never materialised.

There is a complication. RoPE rotates the key by a position-dependent matrix, and that matrix would sit between $W^{UK}$ and the query, preventing the absorption. DeepSeek therefore splits each head into a part without positional information (128 dimensions) and a small decoupled part that carries RoPE (64 dimensions). The rotary key is shared across heads and cached separately, which is the $d_h^{R} = 64$ in the cache formula. DeepSeek-V3's cache per token per layer is $512 + 64 = 576$ elements; the same layer under multi-head attention with its 128 heads would hold $128 \times (128 + 64 + 128) = 40{,}960$ elements, a factor of 71.

Queries are also compressed through a low-rank bottleneck in V3 (rank 1,536) to save activation memory in training; V2-Lite projects queries directly. RMSNorm layers on the two latents are part of the block. Section 8 includes these in the parameter count.

DeepSeek-V3 caches 70 KB per token against Llama 3.1 70B's 328 KB, although it has 128 heads against 64. The design is a bet that the cache is the binding constraint and that the extra projection compute is cheap by comparison.

### 4.5 Sliding-window and interleaved attention

Full causal attention costs $O(n^2)$ in time and the cache grows with $n$. Sliding-window attention lets each token attend only to the previous $W$ tokens {cite}`beltagy2020longformer`. Cost and cache become $O(nW)$, and since each layer extends the receptive field by $W$, information can still travel $L \cdot W$ positions through the stack. Mistral 7B uses $W = 4096$ in every layer and a rolling-buffer cache of fixed size {cite}`jiang2023mistral`.

A pure window gives up exact long-range retrieval in any single layer. Gemma 2 alternates local layers (window 4,096) with global layers (8,192-token span) {cite}`gemma2024gemma2`: half the layers pay for the full span, half pay a constant. It also caps the attention logits with a $\tanh$ soft-cap (at 50.0) and the final logits (at 30.0) as another bound on magnitudes. ModernBERT, an encoder, uses a similar alternation, with a global layer every third layer and a 128-token local window (Section 9).

A property of window attention that surprises people: tokens near the start of a sequence receive a disproportionate share of attention even when they are irrelevant, as "attention sinks", and removing them from a windowed cache breaks the model. Xiao et al. exploit this to stream arbitrarily long text by always keeping the first few tokens plus a window {cite}`xiao2024efficient`. The streaming model does not remember what fell out of the window; it simply stays stable.

```{admonition} Cache Sizes, Measured
:class: tip
The notebook section {ref}`nb-arch-kvcache` generates text with a cache and checks that it equals the full forward pass, measures the cache of the small model in bytes against the formulas above for MHA, GQA and MQA, and builds a simplified latent cache whose output matches its own uncompressed expansion.
```

## 5. Positions and Long Context

### 5.1 RoPE in one paragraph

Chapter 1 derived rotary embeddings {cite}`su2021roformer`. Each pair of dimensions $(2i, 2i+1)$ of a query or key at position $m$ is rotated by the angle $m\,\theta_i$, with $\theta_i = b^{-2i/d}$ for a base $b$ (10,000 in the original) and $i = 0, \dots, d/2 - 1$. The dot product between a query at position $m$ and a key at position $n$ then depends only on $m - n$. The frequencies range from $\theta_0 = 1$ radian per position, which completes a turn in about 6 tokens, down to $\theta_{d/2-1} = b^{-(d-2)/d}$, almost $1/b$, which turns once in about $2\pi b$ tokens. The base therefore sets the longest wavelength the model can represent.

### 5.2 Why trained length matters

A model trained on sequences up to $L$ has only ever seen rotation angles $m\,\theta_i$ with $m < L$. For high-frequency pairs this covers many full turns, so every angle in $[0, 2\pi)$ has been seen. For the low-frequency pairs, those with $L\,\theta_i < 2\pi$, the training positions cover only part of a turn. At position $m > L$ these pairs rotate to angles never seen in training, and the model's behaviour is unconstrained. The failures of naive extrapolation come from these low-frequency components.

Two families of fix change the frequencies, and both are used in the table:

**A larger base.** Raising $b$ lowers every frequency, so the angles at position $m$ are those of a smaller position. Llama 3 raised the base from 10,000 to 500,000 {cite}`grattafiori2024llama`, Qwen2.5 and Mixtral use 1,000,000. It is the simplest approach: train from the start, or continue training, with the larger base.

**Rescaling an existing model.** Position interpolation divides every position by a scale factor $s = L'/L$, so positions up to $L'$ map into $[0, L)$ and no angle exceeds what training saw {cite}`chen2023extending`. Its drawback is that it squeezes the high frequencies too, blurring the distinction between adjacent tokens. NTK-aware scaling, from a pseudonymous 2023 post that appeared shortly after position interpolation and before YaRN {cite}`bloc972023ntk`, instead changes the base:

$$b' = b \cdot s^{\,d/(d-2)}$$

This leaves the highest frequency $\theta_0 = 1$ unchanged and divides the lowest by exactly $s$, with the frequencies between spread geometrically. High-frequency components are preserved and low-frequency ones are interpolated. YaRN refines this: it interpolates the low-frequency dimensions, leaves the high-frequency ones, blends between them with a ramp, and rescales the attention logits by a temperature that grows with the scale factor {cite}`peng2024yarn`. DeepSeek-V2 and V3 use YaRN with scale factor 40, from a trained length of 4,096 to 163,840 positions.

Two short functions capture the frequency arithmetic.

```python
import torch


def rope_frequencies(d, base=10000.0):
    """theta_i = base^(-2i/d) for i = 0 .. d/2 - 1, in float64."""
    return 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float64) / d))


def ntk_aware_base(base, scale, d):
    """Base that divides the lowest frequency by `scale` and keeps the highest."""
    return base * scale ** (d / (d - 2))
```

The notebook section {ref}`nb-arch-rope` asserts the identities above to $10^{-9}$: after NTK-aware scaling, $\theta_0$ is unchanged and $\theta_{d/2-1}$ is exactly $1/s$ of its old value; with position interpolation, the largest angle at the low-frequency components over the extended length stays below the largest angle seen in training.

### 5.3 Alternatives to RoPE

ALiBi adds a penalty to attention scores proportional to the distance between query and key, with a different slope per head, and no positional signal elsewhere {cite}`press2022train`. It extrapolates to longer sequences with no change, which was its selling point, but it has been displaced by RoPE with rescaling in most recent models. T5's bucketed biases {cite}`raffel2019exploring`, implemented in Chapter 1, belong to the same family: a learned additive bias indexed by distance. Shaw et al. introduced relative positions earlier, as learned vectors added to the keys and values {cite}`shaw2018self`.

### 5.4 Long-context training and what a number promises

The configuration's `max_position_embeddings` says what the code accepts, not what the model does well. A long-context model is made in stages. Pretraining is done at a short length because attention cost grows with the square of it. A short final stage, a small fraction of total tokens, continues at the long length with a larger RoPE base or rescaled frequencies. Llama 3.1 does this in six increments up to 128K, and checks at each that short-context quality is unchanged and that needle-in-a-haystack retrieval succeeds at the new length before moving on {cite}`grattafiori2024llama`.

Three different statements get blurred under "context length":

1. *The model accepts that many positions* (the configuration value).
2. *The model was trained on sequences that long*, which is what the training report says.
3. *The model uses information anywhere in them*, which only evaluations on long-context tasks show.

Mistral 7B v0.1 is an example: its configuration says 32,768 positions, its paper reports training at 8,192 with a 4,096 window. The first number is a statement about code, the second about training. Treat a context-length figure as the first claim unless the report supports the others.

```{admonition} Frequencies and Scaling
:class: note
The notebook section {ref}`nb-arch-rope` checks that RoPE preserves norms and makes query-key products depend on relative offset, asserts the NTK-aware identities in float64, compares interpolation and extrapolation on the low-frequency components, and plots the rotation angles across the extended range.
```

## 6. Mixture of Experts

### 6.1 The idea

A dense model uses all of its parameters for every token. A mixture-of-experts (MoE) layer replaces the feed-forward network by $E$ separate feed-forward networks (experts) and a router that chooses $k$ of them for each token. The layer's parameters grow with $E$; its compute per token grows with $k$. Capacity is decoupled from cost.

A router computes a score for every expert from the token's hidden state, keeps the top $k$, and combines the chosen experts' outputs weighted by their (renormalised) scores:

$$y = \sum_{i \in \mathrm{TopK}(s)} g_i\, E_i(x), \qquad g_i = \frac{s_i}{\sum_{j \in \mathrm{TopK}(s)} s_j}$$

GShard introduced top-2 routing for translation transformers {cite}`lepikhin2021gshard`; Switch Transformers showed that routing to a single expert works at trillion-parameter scale {cite}`fedus2022switch`. Mixtral uses top-2 of 8 experts in every layer, and the softmax over the two selected logits is the same as renormalising the full softmax over the selected pair {cite}`jiang2024mixtral`. A sketch of the router:

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
```

### 6.2 Keeping the experts balanced

Left alone, routers collapse: an expert that gets slightly more tokens trains faster, gets better, and receives still more. The remaining experts starve and their parameters are wasted. Three mechanisms counter this.

**An auxiliary loss.** The Switch loss is $\alpha\, E \sum_{i=1}^{E} f_i P_i$, where $f_i$ is the fraction of tokens dispatched to expert $i$ and $P_i$ is the mean router probability assigned to it {cite}`fedus2022switch`. With $\sum_i f_i = 1$ (defined over the routed slots when $k > 1$), the loss is minimised, at the value 1, by uniform routing. Because $f_i$ is not differentiable, the gradient flows through $P_i$. The coefficient trades balance against quality: Mixtral's configuration has 0.02 and OLMoE's 0.01. ST-MoE adds a *router z-loss*, a penalty on the squared log-sum-exp of the router logits that keeps them small and the routing computation numerically stable in bf16 {cite}`zoph2022st`.

**A capacity factor.** On parallel hardware each expert processes a fixed-size buffer of $C = \lceil c\, k\, T / E \rceil$ tokens for $T$ tokens in a batch and capacity factor $c$. Tokens beyond an expert's capacity are dropped from that expert (they pass through the residual connection unchanged). A factor of 1.0 is fully balanced and drops any overload; larger values waste compute on padding. "Dropless" implementations, such as the one used for OLMoE, avoid the choice at the cost of irregular shapes.

**Expert-choice routing.** Zhou et al. invert the direction: each expert chooses its top tokens, which makes balance exact by construction, though it complicates autoregressive decoding, where the future tokens that would compete for an expert are not yet known {cite}`zhou2022mixture`.

### 6.3 Fine-grained and shared experts

DeepSeekMoE makes two changes to the Mixtral layout {cite}`dai2024deepseekmoe`. *Fine-grained experts* splits each expert into several smaller ones and routes to more of them: OLMoE uses 64 experts of width 1,024 with top-8, DeepSeek-V3 uses 256 with width 2,048 and top-8, instead of 8 experts with width 14,336 and top-2. The number of combinations of experts a token can use grows combinatorially, which should let experts specialise more finely. *Shared experts* are always active: DeepSeek keeps one or two that process every token, so common knowledge need not be duplicated across the routed experts.

DeepSeek-V3 adds three more refinements {cite}`deepseek2024v3`. Its router uses sigmoid scores (normalised over the selected experts, then multiplied by 2.5), rather than softmax. Routing is node-limited, as DeepSeek-V2 already did with devices: the 256 experts are divided into 8 groups, a token chooses its best 4 groups and then the top 8 experts among them, which bounds cross-node communication. And it replaces the auxiliary loss with *auxiliary-loss-free* balancing {cite}`wang2024auxiliary`: a per-expert bias is added to the scores for the top-$k$ *selection* only (not for the gate weights), and after each step the bias of overloaded experts is lowered and of underloaded ones raised. No gradient from a balancing objective competes with the language-modelling loss; a very small sequence-level auxiliary loss remains as a safeguard. The bias is stored per layer, one value per expert, which is why V3's router has $E$ extra parameters per layer in Section 8.

The first layers of both DeepSeek models are dense (one in V2-Lite, three in V3); the configuration's `first_k_dense_replace` says how many.

### 6.4 Total and active parameters

A sparse model has two sizes. Its *total* parameters set the memory needed to hold it; its *active* parameters, those a single token touches, set the compute per token. Mixtral 8x7B holds 46.7B parameters and uses 12.9B for each token, so it costs per token about what a 13B dense model does while requiring memory for a 47B one. DeepSeek-V3 holds 671B and activates 36.6B, 5.5% of them.

This changes what "size" means when comparing models. For compute, FLOPs per token, and training cost, the active count is what matters. For serving memory, and the number of GPUs needed to hold the weights, the total counts. Because a batch of tokens activates many different experts, the whole parameter set is touched at any reasonable batch size, so the memory bandwidth required per decoding step is also closer to that of the total than of the active count. Sparse models are cheaper to *train* than dense models of equal quality, and their serving advantage is smaller and depends on batch size and the interconnect.

Vendors disagree on whether "active" includes the input embedding table, which contributes parameters but no matrix multiplications (a lookup is not a multiplication). DeepSeek's 2.4B and 37B match a count that excludes it; OLMoE's 1.3B and Mixtral's 12.9B match a count that includes it. Section 8 computes both and states which each vendor's figure matches.

```{admonition} A Router You Can Run
:class: tip
The notebook section {ref}`nb-arch-moe` routes 4,096 random tokens through a top-2 router over 8 experts, checks that the dispatch fractions are near uniform and that the auxiliary loss is near 1 for a balanced router and well above it for one biased toward a single expert, and verifies the capacity-dropping count against a plain loop.
```

## 7. Efficiency Built into the Architecture

### 7.1 Exact attention, faster

FlashAttention computes exactly the same attention output as the textbook implementation but never writes the $n \times n$ score matrix to GPU main memory. It processes blocks that fit in on-chip SRAM, keeping running softmax statistics, so the extra memory is linear rather than quadratic in sequence length and the traffic to main memory falls by a large constant factor {cite}`dao2022flashattention`. FlashAttention-2 improves the work partition across thread blocks and warps {cite}`dao2024flashattention2`. Because the result is identical up to floating-point rounding, these are kernel choices, not architectural ones, but they changed what is affordable: attention memory stopped limiting sequence length, and the architectural cost that remains is the KV cache. PyTorch's `scaled_dot_product_attention`, used in this chapter's notebook, dispatches to such a kernel when one is available.

### 7.2 The KV cache in serving

Paged attention manages the KV cache like virtual memory, in fixed-size blocks allocated on demand, which removes the fragmentation and over-reservation that wasted most of the cache memory in earlier serving systems {cite}`kwon2023efficient`. Serving systems, including the cache-aware scheduling that makes this work, are the subject of the planned chapters on inference; see the {doc}`/roadmap`. For this chapter the point is that every architecture decision in Section 4 changes the number of bytes that system must manage.

### 7.3 Multi-token prediction

Standard training predicts one next token per position. Gloeckle et al. add extra heads that predict tokens 2, 3 and 4 positions ahead from the same hidden state, which improved code benchmarks at larger scales and allows speculative decoding without a separate draft model {cite}`gloeckle2024better`. DeepSeek-V3 uses a variant in which the additional prediction is made by a one-layer transformer module that takes the main model's hidden state and the next token's embedding, keeping the causal chain {cite}`deepseek2024v3`. The module is used as a training objective; it is not needed at inference (it can be dropped, or used for speculative decoding). Its layer accounts for the 13.46B elements that are in V3's checkpoint but not in its 671B (about 11.6B of them are the module's own weights; the rest are copies of the shared embedding and output head).

### 7.4 Embeddings and vocabulary

The embedding table and the output projection each hold $V \cdot d_{model}$ parameters. Tying them, using one matrix for both, halves that cost; Gemma 2 does so, and many small models do. At large scale the embedding is a small part of the total and untying gives the output projection freedom, so Llama, Qwen, Mistral, OLMo and DeepSeek do not tie. The share is large for small models with large vocabularies: Gemma 2 9B's tied table has 917M parameters, 9.9% of the model, while DeepSeek-V3's two tables hold 1.9B of 671B, 0.3%.

Larger vocabularies shorten sequences (Chapter 1 and its tokenization companion measure this) at the cost of a larger table and a softmax over more classes. The progression from 32,000 entries in the original Llama and Mistral to 128,256 in Llama 3 and 256,000 in Gemma 2 follows from multilingual coverage and from the fact that the table became a smaller part of larger models.

## 8. Accounting

### 8.1 Counting parameters from a configuration

The following function turns a Hugging Face `config.json` into exact parameter counts. It dispatches on `model_type` and refuses model types it does not know, because several details are fixed in a model's code rather than in its configuration: Qwen2's attention biases, the QK-norm weights of OLMo 2 and OLMoE, the four norms of a Gemma 2 layer, the per-expert bias of DeepSeek-V3's router. The same function is run by the repository's tests and by the notebook.

```python
SUPPORTED = ("llama", "qwen2", "mistral", "mixtral", "olmo2", "olmoe", "gemma2",
             "deepseek_v2", "deepseek_v3")


def count_params(cfg):
    """Parameter counts from a Hugging Face config.json dictionary.

    Returns exact integers: total (what the checkpoint stores, excluding DeepSeek's multi-token
    prediction layer), and the parameters one token touches, with and without the input
    embedding (vendors differ on which they call "active").
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
        q = H * nh * (dn + dr) if q_rank is None else H * q_rank + q_rank + q_rank * nh * (dn + dr)
        kv = H * (kv_rank + dr) + kv_rank + kv_rank * nh * (dn + dv)
        attn = q + kv + nh * dv * H
    else:
        attn = H * nh * dh + 2 * H * nkv * dh + nh * dh * H
        if kind == "qwen2":
            attn += nh * dh + 2 * nkv * dh  # biases on q, k and v (set in the model code)
        if kind in ("olmo2", "olmoe"):
            attn += nh * dh + nkv * dh  # QK-norm weights over the full q and k projections
    # RMSNorm weights per layer: two (before or after each sublayer), or four for Gemma 2,
    # which normalises both before and after each sublayer
    norms = (4 if kind == "gemma2" else 2) * H

    def gated(width):  # SwiGLU feed-forward: gate, up and down projections
        return 3 * H * width

    def ffn(layer):  # (total, active) feed-forward parameters of one layer
        if kind in ("llama", "qwen2", "mistral", "olmo2", "gemma2"):
            return gated(cfg["intermediate_size"]), gated(cfg["intermediate_size"])
        if kind in ("mixtral", "olmoe"):
            experts = cfg.get("num_local_experts") or cfg["num_experts"]
            top_k, width = cfg["num_experts_per_tok"], cfg["intermediate_size"]
            return H * experts + experts * gated(width), H * experts + top_k * gated(width)
        if layer < cfg["first_k_dense_replace"]:  # DeepSeek: dense layers come first
            return gated(cfg["intermediate_size"]), gated(cfg["intermediate_size"])
        assert cfg.get("moe_layer_freq", 1) == 1
        experts, top_k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
        width, shared = cfg["moe_intermediate_size"], cfg["n_shared_experts"]
        router = H * experts + (experts if kind == "deepseek_v3" else 0)  # V3: per-expert bias
        return (router + shared * gated(width) + experts * gated(width),
                router + shared * gated(width) + top_k * gated(width))

    tied = bool(cfg.get("tie_word_embeddings"))
    body = L * (attn + norms)
    ffn_total = sum(ffn(i)[0] for i in range(L))
    ffn_active = sum(ffn(i)[1] for i in range(L))
    total = V * H + (0 if tied else V * H) + H + body + ffn_total
    active_without_embedding = V * H + H + body + ffn_active  # output head, no input embedding
    active_with_embedding = active_without_embedding + (0 if tied else V * H)
    return {"total": total, "active_with_embedding": active_with_embedding,
            "active_without_embedding": active_without_embedding}
```

Everything outside the transformer blocks is the embedding, the output projection (absent when tied) and one final norm. Inside a block, the terms follow the sections above: four attention projections (with $n_{kv}$ sharing keys and values), the norms, and the feed-forward networks.

### 8.2 Validation

A parameter counter is only useful if it is right, and "right" has three different meanings, each checked.

1. **Against the checkpoint.** `tools/fetch_arch_samples.py` reads the header of every `.safetensors` shard of each ungated model with two HTTP range requests (eight bytes for the header length, then the JSON header) and sums the tensor element counts. Counting elements rather than bytes makes the check independent of the storage type: OLMo 2 is stored in float32, DeepSeek-V3 in a mix of BF16, FP8 and float32. The counts are committed in `data/checkpoint_counts.json`. The function reproduces **all seven** to the last digit: for DeepSeek-V3, the main model's 671,026,419,200 elements, with the multi-token-prediction layer (`model.layers.61.*`) and the FP8 scale tensors (`*_scale_inv`) summed separately and excluded.
2. **Against the hub's totals for gated models.** The paper-derived configurations of Llama 3.1 8B and 70B and of Gemma 2 9B are not downloadable, but the hub's public metadata reports each checkpoint's parameter total. The function applied to the transcribed configurations gives 8,030,261,248, 70,553,706,496 and 9,241,705,984, matching the metadata exactly. That checks the architecture description of each paper, including Gemma 2's four norms and tied embeddings.
3. **Against the published figures.** Each vendor's quoted total, and active count in its convention, is compared with a tolerance of half a unit in the last published digit, except where noted.

| Model | Computed total | Published | Computed active | Published active (convention) |
| --- | --- | --- | --- | --- |
| Qwen2.5 7B | 7,615,616,512 | 7.61B | — | — |
| OLMo 2 7B | 7,298,617,344 | 7.3B | — | — |
| Mistral 7B v0.1 | 7,241,732,096 | 7.3B | — | — |
| Mixtral 8x7B | 46,702,792,704 | 46.7B | 12,879,925,248 (with) | 12.9B (with) |
| OLMoE 1B-7B | 6,919,161,856 | 6.9B | 1,282,017,280 (with) | 1.3B (with) |
| DeepSeek-V2-Lite | 15,706,484,224 | 15.7B | 2,451,435,008 (without) | 2.4B (without) |
| DeepSeek-V3 | 671,026,419,200 | 671B | 36,625,618,432 (without) | 37B (without) |

Two entries need a comment. Qwen's 7.61B is a truncation of 7.6156B, and Mistral's 7.3B is a rounding of 7.2417B up to the next tenth of a billion; the checks for these two use correspondingly wider tolerances (0.01B and 0.1B). DeepSeek-V2-Lite's "2.4B active" is a truncation of 2.451B. These are the vendors' choices and the exact counts are in the committed regression test.

The three checks catch different mistakes. The checkpoint comparison finds wrong formulas, including a forgotten bias or norm. The hub-totals comparison verifies the descriptions transcribed from papers. The published-figure comparison finds convention mismatches (the embedding in "active"), which the first two cannot, because they only concern the total.

### 8.3 FLOPs per token

For a dense model with $N$ parameters in matrix multiplications, the forward pass costs about $2N$ FLOPs per token (one multiply and one add per weight) and the backward pass twice that, so training costs $6N$ per token {cite}`kaplan2020scaling`. Chapter 2 gives the derivation. Two refinements matter for modern models.

First, the embedding lookup is not a multiplication, so $N$ should exclude the input embedding table. The output projection is a multiplication and stays in. For a sparse model, $N$ is the *active* count, with the router included.

Second, attention scores and their application to the values add a term proportional to the context length $s$, which the $6N$ rule omits. In Kaplan et al.'s accounting, the forward pass per token is $2N + 2\,L\,s\,d_{attn}$, where $d_{attn} = n_h d_h$ is the attention width, and training is three times that (up to a factor of two lower under causal masking, which skips half of the scores). The attention term is below about 10% of the total while the context is no longer than the model width, and grows past it.

| Model | Parameters in the $6N$ rule | $6N$ (GFLOPs per token) |
| --- | --- | --- |
| Llama 3.1 8B | 8.03B, less 0.53B input embedding | 45.0 |
| Llama 3.1 70B | 70.55B, less 1.05B input embedding | 417.0 |
| Mixtral 8x7B | 12.75B active, no input embedding | 76.5 |
| DeepSeek-V3 | 36.63B active, no input embedding | 219.8 |

Training a dense model of DeepSeek-V3's size would cost $6 \times 671\text{B} = 4.0$ TFLOPs per token, 18 times as much as the 0.22 TFLOPs that the sparse model needs.

### 8.4 Memory at inference

Weights take the parameter count times the bytes per parameter: 2 for bf16, 1 for FP8 (DeepSeek-V3's native format), roughly 0.5 for 4-bit quantization (Chapter on compression, planned). Add the KV cache from Section 4: bytes per token times the number of tokens in all sequences being served.

**A dense example.** Llama 3.1 8B in bf16 needs $8{,}030{,}261{,}248 \times 2 = 16.06$ GB for the weights. One sequence of 128K tokens needs $131{,}072$ bytes $\times$ $131{,}072$ tokens $= 17.18$ GB for the cache, so serving a single full-length request needs about 33 GB before activations, and a second concurrent request adds another 17 GB. The same cache with 32 key/value heads (multi-head attention) would be 69 GB per sequence, and the model would not fit at that length on a 40 GB card at all. This is the practical content of grouped-query attention.

**A sparse example.** DeepSeek-V3 in FP8 needs about 671 GB for its weights, which is why it is served on a cluster, not a card. The attention cache is the small part: $70{,}272 \times 131{,}072 = 9.2$ GB for a 128K-token sequence, compared with 655 GB had it used multi-head attention with its head shape. Compute per token is that of a 37B model; weight memory is that of a 671B one, and the batch must be large enough for every expert to receive tokens for the compute advantage to be realised.

```{admonition} The Accounting Function
:class: tip
The notebook section {ref}`nb-arch-configs` runs the function above on every configuration, compares it with the checkpoint element counts, the hub totals and the published figures, and prints the derived tables of this chapter, with an assertion behind each check.
```

## 9. Encoders and Encoder-Decoders Today

Decoder-only models dominate generation, but two other designs persist where their properties fit the task.

**Encoder-only models** (BERT {cite}`devlin2018bert`) apply bidirectional attention and are trained to fill in masked tokens. They produce a representation of each input token that can see both sides, which is what classification, retrieval and embedding need, at a size and cost far below a generative model. Most production embedding and reranking models are still encoders. ModernBERT brings the decoder block's improvements to the form: RoPE, GeGLU, bias-free layers, alternating local and global attention, an 8,192-token context and a modern training mix {cite}`warner2024smarter`.

**Encoder-decoder models** (T5 {cite}`raffel2019exploring`) read the whole input with a bidirectional encoder and generate with a decoder that cross-attends to it. For tasks with a clear input and output, such as translation, summarisation and structured extraction, the input is encoded once and need not be re-attended with a causal mask; at small and medium sizes these models are competitive with decoder-only models. At the frontier scale the field has standardised on decoders because one architecture and one objective serve every use through prompting.

**State-space and hybrid models** replace some or all attention layers with recurrent layers whose state does not grow with sequence length; Mamba is the best known {cite}`gu2023mamba`. They remove the KV cache's linear growth rather than shrinking it. They are outside the scope of this chapter and are planned for the roadmap's later chapters; see the {doc}`/roadmap`.

## 10. Summary

- A 2025 decoder block is the 2017 block with the normalization moved or added so that residual contributions stay bounded, a gated feed-forward network without biases, rotary positions, and grouped or latent key/value sharing.
- The key/value structure sets the KV cache: $2 n_{kv} d_h$ per token per layer for GQA and $d_c + d_h^R$ for MLA, ranging in bf16 from 524 KB per token (OLMo 2 7B) to 31 KB (DeepSeek-V2-Lite) and 70 KB (DeepSeek-V3).
- Long context is a frequency change plus a short stage of training at the new length. A context-length number from a configuration is a statement about the code.
- A mixture-of-experts layer buys parameters at constant compute; it needs a balancing mechanism, and "active parameters" depends on whether the input embedding is counted.
- Parameter counts can be computed exactly from a configuration, and should be checked three ways: against the checkpoint, against the vendor's own total, and against the vendor's own convention.

## Exercises

1. **Which head count?** For a model with $d_{model} = 4096$, 32 layers and $d_h = 128$, tabulate the KV cache per token for $n_{kv} \in \{32, 16, 8, 4, 2, 1\}$. For a 24 GB GPU holding 8 GB of weights and serving 32K-token sequences, how many concurrent sequences fit at each setting?
1. **Converting to GQA.** Take the notebook's multi-head attention layer, mean-pool its key and value projection weights into groups as Ainslie et al. describe, and measure how far the layer's output moves from the original for $n_{kv} = 16, 8, 4, 2, 1$. What does the shape of the curve suggest about the choice of 8?
1. **Counting a new model.** Fetch the `config.json` of a model type `count_params` does not support (for example a recent Qwen3 or Llama variant), work out which details its model code fixes, extend the function, and validate it against the checkpoint's safetensors header. Which of the three checks of Section 8.2 caught your first mistake?
1. **Absorbing the up-projection.** Show algebraically that for MLA without RoPE, $q_t^\top k_s = (W^{UK\top} q_t)^\top c_s$ and compute the number of multiplications per decoded token with and without the absorption. Why does the decoupled rotary key prevent it for the RoPE part?
1. **A better interpolation.** Implement YaRN's per-dimension blend of interpolation and extrapolation, with the ramp defined by the `beta_fast` and `beta_slow` values in DeepSeek's configuration. Plot the resulting frequencies against the original and the NTK-aware ones.
1. **Router collapse.** Train the notebook's small mixture-of-experts layer on a text task without the auxiliary loss and with coefficients 0.001, 0.01 and 0.1. Plot the fraction of tokens each expert receives over training. When does the loss improve with balance, and when does it get worse?
1. **Auxiliary-loss-free balancing.** Implement the bias update of Wang et al. and compare its expert load and loss with the auxiliary-loss router of the previous exercise.
1. **Active parameters.** Using `count_params`, compute the ratio of active to total parameters for each MoE model in Section 1. Which of the two conventions for "active" would you use to estimate the cost of a training run, and which to estimate serving memory?

## Further Reading

### Models and technical reports

- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) {cite}`grattafiori2024llama` - the architecture table, long-context staging and scaling choices of Llama 3 and 3.1
- [Mistral 7B](https://arxiv.org/abs/2310.06825) {cite}`jiang2023mistral` and [Mixtral of Experts](https://arxiv.org/abs/2401.04088) {cite}`jiang2024mixtral` - sliding-window attention, the rolling cache, and a small sparse model
- [Gemma 2](https://arxiv.org/abs/2408.00118) {cite}`gemma2024gemma2` - interleaved local/global attention, soft-capping and sandwich norms
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) {cite}`qwen2024qwen25` - a dense family with YaRN and long-context extension
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) {cite}`olmo2024olmo2` - reordered norms, QK-norm and the stability changes behind them, with open data and code
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) {cite}`deepseek2024v2` and [DeepSeek-V3](https://arxiv.org/abs/2412.19437) {cite}`deepseek2024v3` - multi-head latent attention, fine-grained MoE, multi-token prediction and FP8 training

### Attention and positions

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) {cite}`shazeer2019fast` and [GQA](https://arxiv.org/abs/2305.13245) {cite}`ainslie2023gqa` - multi-query and grouped-query attention
- [Longformer](https://arxiv.org/abs/2004.05150) {cite}`beltagy2020longformer` and [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) {cite}`xiao2024efficient` - windows and sinks
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) {cite}`chen2023extending` and [YaRN](https://arxiv.org/abs/2309.00071) {cite}`peng2024yarn` - extending RoPE
- [FlashAttention](https://arxiv.org/abs/2205.14135) {cite}`dao2022flashattention`, [FlashAttention-2](https://arxiv.org/abs/2307.08691) {cite}`dao2024flashattention2` and [PagedAttention](https://arxiv.org/abs/2309.06180) {cite}`kwon2023efficient` - the kernels and the cache manager

### Normalization and activations

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) {cite}`xiong2020layer` - why pre-norm needs no warm-up
- [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442) {cite}`dehghani2023scaling` - QK-norm
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) {cite}`shazeer2020glu` - the gated feed-forward network

### Mixture of experts

- [GShard](https://arxiv.org/abs/2006.16668) {cite}`lepikhin2021gshard`, [Switch Transformers](https://arxiv.org/abs/2101.03961) {cite}`fedus2022switch` and [ST-MoE](https://arxiv.org/abs/2202.08906) {cite}`zoph2022st` - routing, load-balancing loss, capacity and router z-loss
- [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368) {cite}`zhou2022mixture` - experts choose tokens
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) {cite}`dai2024deepseekmoe` and [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) {cite}`wang2024auxiliary` - fine-grained, shared and bias-balanced experts

### Other designs

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) {cite}`gloeckle2024better`
- [Smarter, Better, Faster, Longer (ModernBERT)](https://arxiv.org/abs/2412.13663) {cite}`warner2024smarter`
- [Mamba](https://arxiv.org/abs/2312.00752) {cite}`gu2023mamba`

### Tools

- [Hugging Face Transformers model implementations](https://github.com/huggingface/transformers/tree/main/src/transformers/models) - the reference code for every configuration key discussed here
- [Safetensors format](https://huggingface.co/docs/safetensors) - the header layout read by this chapter's fetch script

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
