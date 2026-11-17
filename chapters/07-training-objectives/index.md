# Training Objectives and Methodologies

Chapter 2 derived the pieces of a training step: the cross-entropy loss, Adam and AdamW, a warm-up followed by a cosine decay, gradient clipping, and the Chinchilla rule that a compute budget should be split about equally between parameters and tokens. This chapter is about how those pieces are set for a real pretraining run, and what the published runs of 2020 to 2025 changed. Which tokens the loss averages over, and which objective produces them. Which optimizer, with which constants. How the learning rate rises and falls, and why the shape of its end matters more than its middle. How many tokens to train on when the model will be served for a year, and how to keep a run from diverging two months in. We put numbers on each from the technical reports of GPT-3, Llama 2 and 3, OLMo 2 and DeepSeek-V3, and check the mechanisms on small models that train on a CPU.

(sec-ch7-chapter-overview)=

## Chapter Overview

A pretraining run is a long list of decisions, most of which are invisible in the released checkpoint. This chapter goes through them in the order they are made.

**The objective** is next-token prediction for every model in this book, but the loss is an average, and how it is averaged (over tokens, sequences or micro-batches) changes the gradient. Small auxiliary terms keep the output logits in range.

**Other objectives** predict masked spans, the middle of a document, or several tokens at once. They decide which tokens produce a training signal and what the model can do at inference, such as completing code between a prefix and a suffix.

**The optimizer** is AdamW in nearly every report, with $\beta_2 = 0.95$ rather than the textbook 0.999. Optimizers that keep less state, or that orthogonalize the update of each weight matrix, are the first serious alternatives in years.

**The schedule** of the learning rate and the batch size sets how much of the run's potential is realized. The final decay produces a large share of the improvement, which is why a schedule with a long constant phase and a short decay can stop at any length.

**Scaling** decides the model size and the token count from the budget. The compute-optimal ratio is the starting point; models that will serve many requests are trained far past it.

**Stability** is the difference between a run that finishes and one that is restarted from a checkpoint. Most instabilities have a known mechanism and a known guard.

**The stages** of a run, from the main pretraining mixture to an annealing phase on high-quality data and checkpoint averaging, end in a base model; the post-training that follows is the subject of later parts of the book ({ref}`roadmap-post-training-and-alignment`).

Every symbol the book uses is listed on the {ref}`notation page <sec-notation>`, with a link to the section that introduces each symbol used only in this chapter.

(sec-ch7-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- How the loss of a batch is averaged over tokens, why padding, packing and gradient accumulation change that average, and how to accumulate gradients so the result equals one large batch
- What the z-loss and logit soft-capping add to the objective, and why neither changes the model's predictions
- How span corruption, prefix language modeling, fill-in-the-middle and multi-token prediction differ from next-token prediction in their inputs, their targets and the share of tokens that carry a loss
- The AdamW settings of published runs, why $\beta_2$ is 0.95 and which parameters are exempt from weight decay
- How Adafactor, Lion and Muon reduce optimizer memory or change the shape of the update, and how many bytes each needs per parameter
- Why the decay at the end of a learning-rate schedule matters, how warm-up-stable-decay schedules work, and how batch sizes are ramped
- How compute-optimal model sizes are fitted, why the two published fits of the Chinchilla data disagree, and what training past the optimum costs and buys
- The mechanisms behind loss spikes and logit divergence, and the guards against them
- How a run ends: annealing on high-quality data, checkpoint averaging, and the conversion of a FLOP budget into GPU-hours

(sec-ch7-hands-on-learning)=

## Hands-On Learning

The accompanying notebook, [Training Objectives Demo](examples/training_objectives_demo.ipynb), checks the mechanisms of this chapter on small models:

- Gradient accumulation that matches one large batch to rounding error, and the mean of means that does not ({ref}`nb-train-loss`)
- The z-loss leaving the predictions unchanged while pulling the log-normalizer to zero ({ref}`nb-train-loss`)
- Span corruption, a prefix-LM mask and the fill-in-the-middle transformation, with the share of tokens that carry a loss in each ({ref}`nb-train-objectives`)
- An AdamW step written out by hand and checked against PyTorch's, and the Newton-Schulz iteration of Muon ({ref}`nb-train-optimizers`)
- A small byte-level model trained with a cosine schedule and with a warm-up-stable-decay schedule, including decay branches taken from one constant-rate run ({ref}`nb-train-schedules`)
- Compute-optimal sizes from both fits of the Chinchilla data, and the cost of training past the optimum ({ref}`nb-train-scaling`)

Everything runs on a CPU. The training cells take about two minutes per seed on a laptop.

(sec-ch7-objective)=

## 1. The Objective Behind Pretraining

(sec-ch7-what-loss-averages)=

### 1.1 What the loss averages

The pretraining objective of a decoder is the cross-entropy of each token given its prefix ({ref}`sec-ch1-next-token-prediction`), averaged over the tokens of a batch:

$$\mathcal{L} = -\frac{1}{T} \sum_{i=1}^{T} \log p_\theta(x_i \mid x_{<i})$$

where $T$ counts the tokens that carry a loss. That count is not the batch size times the sequence length. Padding positions carry no loss, and in a packed sequence ({ref}`sec-ch5-tokenization-packing`) the first token of each document has no prefix from its own document to predict it from, so some pipelines drop it as well. In PyTorch, a position whose target is `-100` (the `ignore_index` of `cross_entropy`) is left out of both the sum and the count.

The average over tokens is a choice. A mean of per-sequence means gives every sequence the same weight, so a token in a short sequence counts more than a token in a long one. In a pretraining batch every sequence is full, and the two agree. They part as soon as sequences have different numbers of loss-bearing tokens, which is the normal case in fine-tuning and in any batch with padding.

**Gradient accumulation** is where this goes wrong silently. Chapter 2 split a batch into $n_{acc}$ micro-batches and divided each micro-batch loss by $n_{acc}$ ({ref}`sec-ch2-gradient-accumulation`). That reproduces the large batch only when every micro-batch has the same number of loss-bearing tokens. When they differ, the sum of micro-batch means divided by $n_{acc}$ is a mean of means: a micro-batch with few tokens gets the same weight as one with many. The correct version divides each micro-batch's *summed* loss by the token count of the whole accumulated batch, which has to be known before the first backward pass. In October 2024 the Hugging Face Transformers trainer was found to compute the mean of means; the fix normalizes "the total loss across all batches in a gradient accumulation step divided by the total number of all non padding tokens in those batches" {cite}`hf2024gradaccum`. The listing compares the three on a small linear model with variable-length sequences.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab, width, n_micro = 50, 16, 4
model = torch.nn.Linear(width, vocab)
# 4 micro-batches of 2 sequences of 32 positions, padded by different amounts
x = torch.randn(n_micro, 2, 32, width)
targets = torch.randint(vocab, (n_micro, 2, 32))
for i, kept in enumerate((32, 20, 8, 3)):
    targets[i, :, kept:] = -100  # padding: no loss


def grad_of(losses):
    model.zero_grad()
    sum(losses).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def token_losses(i, reduction):
    logits = model(x[i]).reshape(-1, vocab)
    return F.cross_entropy(logits, targets[i].reshape(-1), ignore_index=-100,
                           reduction=reduction)


full = grad_of([token_losses(slice(None), "mean")])  # one batch, mean over its tokens
mean_of_means = grad_of([token_losses(i, "mean") / n_micro for i in range(n_micro)])
n_tokens = (targets != -100).sum()
token_mean = grad_of([token_losses(i, "sum") / n_tokens for i in range(n_micro)])
for name, g in (("mean of means", mean_of_means), ("sum / all tokens", token_mean)):
    error = ((g - full).norm() / full.norm()).item()
    print(f"{name:17s} relative error against one batch: {error:.2e}")
print(f"loss-bearing tokens per micro-batch: {(targets != -100).sum((1, 2)).tolist()}")
assert torch.allclose(token_mean, full, atol=1e-6)
```

% output

```text
mean of means     relative error against one batch: 1.08e+00
sum / all tokens  relative error against one batch: 1.42e-07
loss-bearing tokens per micro-batch: [64, 40, 16, 6]
```

The mean of means is off by more than the gradient's own length here, because the micro-batches hold 64, 40, 16 and 6 loss-bearing tokens: the last micro-batch's 6 tokens carry a quarter of the weight instead of 6 in 126. Dividing the sums by the total gives the one-batch gradient to rounding error. The same reasoning applies across data-parallel ranks, which the next chapter takes up: each rank must divide by the global token count, not its own.

(sec-ch7-auxiliary-terms)=

### 1.2 Auxiliary terms on the logits

The softmax is unchanged when a constant is added to every logit (the stable softmax of chapter 2 relies on this), so nothing in the cross-entropy fixes the level of the logits. During long runs in low precision the level can drift, and with it the log-normalizer $\log Z = \log \sum_{v} e^{z_v}$, where $z$ is the vector of output logits over the vocabulary. PaLM added an auxiliary loss of $10^{-4} (\log Z)^2$ "to encourage the softmax normalizer log(Z) to be close to 0", which they found increased the stability of training {cite}`chowdhery2023palm`. With the weight written $\lambda_z$, the per-token loss becomes

$$\ell = -\log p_\theta(x_i \mid x_{<i}) + \lambda_z (\log Z)^2$$

The extra term changes the logits but not the prediction: adding the same constant to every logit leaves $p_\theta$ unchanged and moves $\log Z$ by that constant, so the term has a direction to push in that costs the cross-entropy nothing. Its gradient with respect to the logit $z_v$ is $2 \lambda_z \log Z\, p_v$, where $p_v$ is the predicted probability of token $v$: every logit is pulled in the same direction, toward $\log Z = 0$, in proportion to its probability. OLMo 2 uses the z-loss with weight $10^{-5}$ (its Table 1 and released configuration; one sentence in the text says $10^{-4}$) {cite}`olmo2024olmo2`, and {cite:t}`wortsman2024smallscale` show that without it, small models trained at high learning rates reproduce the output-logit divergence seen in large ones.

```python
import torch


def z_loss(logits, weight=1e-4):
    """PaLM's auxiliary loss: weight * (log Z)^2, averaged over positions."""
    return weight * torch.logsumexp(logits, dim=-1).pow(2).mean()


torch.manual_seed(0)
logits = torch.randn(4, 1000) + 30.0  # every logit far from zero: log Z is large
shifted = logits - torch.logsumexp(logits, dim=-1, keepdim=True)  # log Z = 0
same = torch.allclose(logits.softmax(-1), shifted.softmax(-1), atol=1e-6)
print(f"log Z before {torch.logsumexp(logits, -1).mean():.2f}, "
      f"after {torch.logsumexp(shifted, -1).mean():.2f}")
print(f"same probabilities: {same}; "
      f"z-loss {z_loss(logits):.4f} -> {z_loss(shifted):.1e}")
logits.requires_grad_(True)
z_loss(logits).backward()
row_sums = logits.grad.sum(-1)
print("gradient of each row sums to 2 * weight * log Z / rows:",
      torch.allclose(row_sums, 2e-4 * torch.logsumexp(logits, -1).detach() / 4))
assert same
```

% output

```text
log Z before 37.42, after 0.00
same probabilities: True; z-loss 0.1400 -> 1.0e-16
gradient of each row sums to 2 * weight * log Z / rows: True
```

Gemma 2 takes a different route and bounds the logits themselves. *Soft-capping* replaces each logit $z_v$ by $s_{cap} \tanh(z_v / s_{cap})$, which is close to $z_v$ for small values and never exceeds $s_{cap}$ in size; Gemma 2 sets $s_{cap}$ to 50.0 for the attention logits and 30.0 for the final layer {cite}`gemma2024gemma2`. Unlike the z-loss, soft-capping does change the predicted distribution when logits are large, and it is part of the model rather than of the loss, so it must also be applied at inference. Gemma 3 replaced it with QK normalization ({ref}`sec-ch6-normalizing-queries-keys`).

(sec-ch7-reading-loss)=

### 1.3 Reading a loss curve

A training loss is measured on data the model has not yet seen, since a pretraining run rarely repeats a token, so for most of a run it is an honest estimate of the held-out loss. Runs still keep separate validation sets, for three reasons. The training loss is averaged over a few batches and is noisy; a fixed validation set measured at intervals is not. The training mixture changes during a run ({ref}`sec-ch7-stages`), and a fixed set measures the model rather than the mixture. And a validation set split by domain (web, code, mathematics, several languages) shows where the improvement comes from.

Two losses are comparable only when they are computed on the same text with the same tokenizer: perplexity per token depends on how many tokens a text is cut into, and bits per byte does not ({ref}`sec-ch1-perplexity`). The models in the notebook are byte-level, so their loss in nats per token is already a loss per byte; divided by $\ln 2$ it is bits per byte.

(sec-ch7-other-objectives)=

## 2. Objectives Beyond the Next Token

(sec-ch7-denoising)=

### 2.1 Denoising objectives

An encoder cannot be trained on the next token, because it sees the whole sequence. BERT's masked-language-model objective hides 15% of the tokens and predicts them from both sides ({ref}`sec-ch1-masked-lm`). Only the hidden positions carry a loss, so each sequence produces about a seventh of the training signal of a causal sequence of the same length.

T5 turned denoising into a sequence-to-sequence task {cite}`raffel2019exploring`. *Span corruption* removes spans of tokens from the input, replaces each span by one sentinel token unique within the sequence, and asks the decoder to output the sentinels each followed by the tokens it replaced. The final T5 setup uses "a mean span length of 3 and corrupt[s] 15% of the original sequence". Because each span is written once in the target with a sentinel in front, the targets are short: with 15% of 512 tokens corrupted in spans of mean length 3, the target holds the 77 removed tokens, 26 sentinels in front of their spans and one more that ends it, and the input shrinks by the removed tokens minus the sentinels. The listing implements it.

```python
import numpy as np


def span_corrupt(tokens, rate=0.15, mean_span=3.0, sentinel_base=32_000, seed=0):
    """T5-style span corruption: returns (inputs, targets).

    Picks n_spans = round(rate * len / mean_span) spans whose lengths sum to
    round(rate * len), placed at random non-overlapping positions; each span is
    replaced in the input by a sentinel, and the target lists sentinel, span, ...
    """
    rng = np.random.default_rng(seed)
    n = len(tokens)
    n_noise = round(rate * n)
    n_spans = max(1, round(n_noise / mean_span))
    # random split of n_noise into n_spans positive lengths, and of the rest into gaps
    cuts = np.sort(rng.choice(np.arange(1, n_noise), n_spans - 1, replace=False))
    span_lengths = np.diff(np.concatenate([[0], cuts, [n_noise]]))
    cuts = np.sort(rng.choice(np.arange(1, n - n_noise), n_spans - 1, replace=False))
    gap_lengths = np.diff(np.concatenate([[0], cuts, [n - n_noise]]))
    inputs, targets, position = [], [], 0
    for k, (gap, span) in enumerate(zip(gap_lengths, span_lengths)):
        inputs += list(tokens[position:position + gap]) + [sentinel_base + k]
        position += gap
        targets += [sentinel_base + k] + list(tokens[position:position + span])
        position += span
    return inputs, targets + [sentinel_base + n_spans]  # a last sentinel ends it


tokens = list(range(512))
inputs, targets = span_corrupt(tokens)
n_sentinels = sum(t >= 32_000 for t in targets)
print(f"input {len(inputs)} tokens, target {len(targets)} tokens "
      f"({len(targets) - n_sentinels} original + {n_sentinels} sentinels)")
restored = []
spans = {}
current = None
for t in targets:
    if t >= 32_000:
        current = t
        spans[current] = []
    else:
        spans[current].append(t)
for t in inputs:
    restored += spans[t] if t >= 32_000 else [t]
print("inputs and targets restore the original:", restored == tokens)
assert restored == tokens and len(targets) - n_sentinels == 77
```

% output

```text
input 461 tokens, target 104 tokens (77 original + 27 sentinels)
inputs and targets restore the original: True
```

A *prefix language model* sits between the two. The sequence is split at a point; attention over the prefix is bidirectional, attention from the rest is causal, and only the tokens after the split carry a loss. UL2 trained one model on a mixture of all three families {cite}`tay2023ul2`: R-denoisers (span corruption with mean span 3 or 8 at a 15% rate), X-denoisers (spans of 64 tokens, or corruption rates of 50%), and an S-denoiser that is a prefix language model, seven configurations in all, with a paradigm token ([R], [S] or [X]) prepended so the model knows which task it faces. {numref}`fig-ch7-attention-masks` shows the three attention patterns.

:::{figure} figures/attention-masks.*
:name: fig-ch7-attention-masks
:alt: Three 8 by 8 grids of which keys each query may attend to. Causal is lower-triangular. Prefix LM with a prefix of 3 has a full 3 by 3 block in the top left and is lower-triangular elsewhere. Bidirectional is fully filled. Below each grid a row marks the positions that carry a loss: all eight for causal, the five after the prefix for prefix LM, and two masked positions for bidirectional.
:width: 100%

Which keys (columns) each query (row) attends to, and which positions carry a loss (the
strip below each grid). A causal model learns from every position; a prefix LM with a
3-token prefix learns from the 5 positions after it; a masked model sees everything but
learns only from the positions it masked, here 2 of 8.
:::

Decoder-only models won for generation ({ref}`sec-ch1-masked-lm` gives the history), and among the twelve models of Chapter 6 none is trained with a denoising objective. The reason is partly the training signal per token and partly that a causal model can be used for any task by writing the task as a prefix. Denoising survives in encoders, and in a form adapted to causal models: fill-in-the-middle.

(sec-ch7-fill-in-the-middle)=

### 2.2 Fill-in-the-middle

A code editor needs completions that depend on what comes after the cursor as well as before it. A causal model sees only the left context. {cite:t}`bavarian2022efficient` showed that a causal model learns to fill in the middle if some training documents are rearranged. A document is cut at two random character positions into a prefix, a middle and a suffix, and written as

```text
<PRE> prefix <SUF> suffix <MID> middle <EOT>
```

(the *PSM* order; *SPM* puts the suffix first). The model is trained on the rearranged document with the ordinary next-token loss, so it learns to generate the middle after seeing both ends. The rearrangement is applied to a fraction of the documents, the *FIM rate*. The paper's main models use a rate of 0.5 and show that this does not measurably hurt ordinary left-to-right generation, which they call "FIM-for-free"; they recommend rates between 50% and 90%, and splitting after tokenization and packing ("context-level") rather than before. StarCoder 2 trains with fill-in-the-middle in this way {cite}`lozhkov2024starcoder2`.

```python
import random


def fim_transform(document, rate=0.5, spm_share=0.5, rng=random.Random(0)):
    """Rearrange a document for fill-in-the-middle with probability `rate`."""
    if rng.random() >= rate:
        return document
    i, j = sorted(rng.sample(range(len(document) + 1), 2))
    prefix, middle, suffix = document[:i], document[i:j], document[j:]
    if rng.random() < spm_share:
        return f"<PRE><SUF>{suffix}<MID>{prefix}{middle}<EOT>"
    return f"<PRE>{prefix}<SUF>{suffix}<MID>{middle}<EOT>"


code = "def area(r):\n    return 3.14159 * r * r\n"
for _ in range(3):
    print(repr(fim_transform(code)))
documents = [fim_transform(code) for _ in range(10_000)]
share = sum(d.startswith("<PRE>") for d in documents) / len(documents)
print(f"share transformed: {share:.3f}")
assert abs(share - 0.5) < 0.02
```

% output

```text
'def area(r):\n    return 3.14159 * r * r\n'
'def area(r):\n    return 3.14159 * r * r\n'
'<PRE><SUF>* r * r\n<MID>def area(r):\n    return 3.14159 <EOT>'
share transformed: 0.498
```

The SPM branch follows the paper's variant in which the prefix and middle are written together after `<MID>`, which keeps the tokenization of the joint at the cursor the same as in an ordinary document. At inference, an editor writes the prefix and the suffix in the trained format and generates until `<EOT>`.

(sec-ch7-multi-token)=

### 2.3 Multi-token prediction

Chapter 6 described DeepSeek-V3's multi-token prediction module, which predicts the token after next from the main model's hidden state and the embedding of the next token ({ref}`sec-ch6-multi-token-prediction`). As an objective it adds a second cross-entropy term with its own weight. DeepSeek-V3 uses one extra depth and sets the weight "to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens" {cite}`deepseek2024v3`. {cite:t}`gloeckle2024better` found that predicting four tokens ahead with independent heads helped larger models on code benchmarks more than smaller ones. The extra heads can be discarded after training, or used to propose tokens for speculative decoding.

(sec-ch7-signal-per-token)=

### 2.4 The training signal per token

{numref}`fig-ch7-objectives` sets the objectives side by side for one sequence of 16 tokens. The share of positions that carry a loss is the clearest difference: all of them for causal and fill-in-the-middle training, the positions after the split for a prefix language model, and the corrupted positions (plus sentinels) for the denoisers.

:::{figure} figures/objectives.*
:name: fig-ch7-objectives
:alt: Five rows of token boxes, one row per objective, showing the model's input and which positions carry a loss. Causal: all sixteen positions are dark. Masked LM: positions 5 and 12 are replaced by a mask token and only they are dark. Span corruption: tokens 5 and 6 and token 12 are replaced by sentinels S0 and S1 in the input, and after an arrow the dark target reads S0, x5, x6, S1, x12, S2. Prefix LM: the first six positions are light context and the remaining ten are dark. Fill-in-the-middle: the sequence is reordered as PRE, x1 to x5, SUF, x11 to x16, MID, x6 to x10, all dark.
:width: 100%

Five pretraining objectives on the same 16 tokens. Dark boxes carry a loss; pink boxes
are sentinel tokens in the input. Masked and span-corruption objectives learn
from the corrupted positions only (here 2 and 3 of the 16 tokens, plus the sentinels of
the span-corruption target after the arrow); causal and fill-in-the-middle training
learn from every position; a prefix LM learns from the positions after its prefix.
:::

The objective also decides what the model can do without further training. A causal model can continue text; a fill-in-the-middle model can also complete a gap; a span-corruption model needs its sentinels in the input. A run can mix objectives, as UL2 does and as code models do with fill-in-the-middle, at the cost of more complicated data loading and a share of the batch that teaches something other than continuation.

```{admonition} Objectives You Can Run
:class: tip
The notebook section {ref}`nb-train-objectives` applies span corruption to a passage of
text and restores it, builds a prefix-LM attention mask and checks it against the causal
mask, applies the fill-in-the-middle transformation to many documents, and counts the
share of positions that carry a loss under each objective.
```

(sec-ch7-optimizers)=

## 3. Optimizers

(sec-ch7-adamw-practice)=

### 3.1 AdamW as published

Chapter 2 derived Adam and its decoupled-weight-decay variant AdamW ({ref}`sec-ch2-adam-optimizer`, {ref}`sec-ch2-adamw-implementation`). Chinchilla switched from Adam to AdamW because it "improves the language modelling loss and the downstream task performance after finetuning" {cite}`hoffmann2022training`, and every run in the table below uses it.

```{tabularcolumns} \Y{0.24}\Y{0.10}\Y{0.10}\Y{0.13}\Y{0.27}\Y{0.16}
```

| Run | $\beta_1$ | $\beta_2$ | $\epsilon$ | Weight decay | Clipping |
| --- | --- | --- | --- | --- | --- |
| GPT-3 (2020) | 0.9 | 0.95 | $10^{-8}$ | 0.1 | 1.0 |
| Llama 2 (2023) | 0.9 | 0.95 | $10^{-5}$ | 0.1 | 1.0 |
| OLMo 2 (2024) | 0.9 | 0.95 | $10^{-8}$ | 0.1, none on embeddings | 1.0 |
| DeepSeek-V3 (2024) | 0.9 | 0.95 | not stated | 0.1 | 1.0 |

Sources: GPT-3's Appendix B {cite}`brown2020language`, Llama 2's hyperparameter description {cite}`touvron2023llama2`, OLMo 2's Table 1 and its released stage-1 configuration (the paper does not state $\beta_2$ or the clipping threshold; the configuration does) {cite}`olmo2024olmo2`, and DeepSeek-V3's training hyperparameters {cite}`deepseek2024v3`. "Clipping" is the threshold on the global gradient norm ({ref}`sec-ch2-gradient-clipping`).

Two of these settings differ from the defaults in PyTorch and in Adam's paper, and a third differs in one run.

**$\beta_2 = 0.95$, not 0.999.** The second moment $v_t$ is an exponential average with a memory of about $1/(1 - \beta_2)$ steps: 1,000 steps at 0.999, 20 at 0.95. Adam divides the first moment by $\sqrt{v_t}$, so a parameter whose gradients have been small for a long time and then become large gets a step far larger than usual until $v_t$ catches up; with a long memory, the catching up takes over a thousand steps (Exercise 4). Chapter 2 bounded a single Adam step ({ref}`sec-ch2-adam-step-bound`); the listing shows how close the steps come to that bound after a sudden change. Rare tokens are the usual case in a language model: an embedding row receives a gradient only when its token appears in the batch.

```python
import math


def adam_steps(grads, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam's update m_hat / (sqrt(v_hat) + eps) for a scalar parameter, per step."""
    m = v = 0.0
    steps = []
    for t, g in enumerate(grads, start=1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        steps.append(m_hat / (math.sqrt(v_hat) + eps))
    return steps


# 2,000 steps of gradient 0.01, then the gradient jumps to 1 for 100 steps
grads = [0.01] * 2000 + [1.0] * 100
for beta2 in (0.999, 0.95):
    after = adam_steps(grads, beta2=beta2)[2000:]
    above = sum(abs(s) > 1.5 for s in after)
    print(f"beta2 = {beta2}: largest step {max(after):.2f}, "
          f"steps above 1.5 after the jump: {above}")
```

% output

```text
beta2 = 0.999: largest step 6.12, steps above 1.5 after the jump: 100
beta2 = 0.95: largest step 1.10, steps above 1.5 after the jump: 0
```

Before the jump both settings take steps of about 1, since a constant gradient gives $m/\sqrt{v} = 1$. After it, with $\beta_2 = 0.999$ every one of the next 100 steps stays above 1.5 and the largest is about 6; with $\beta_2 = 0.95$ the largest is 1.1. Every parameter moves by $\eta$ times this ratio, so a long run of large ratios is a large, coherent move of the parameters that saw the change.

**$\epsilon$** (the default $10^{-8}$ in all but Llama 2) is added to $\sqrt{v_t}$ to avoid division by zero, and it matters when gradients are small. When the root-mean-square gradient of a layer approaches $\epsilon$, the update shrinks below its intended size. {cite:t}`wortsman2024smallscale` found that in a 4.8B model, lowering $\epsilon$ from $10^{-8}$ to $10^{-15}$ improved the loss, and OLMo 2 lowered it from $10^{-5}$ to $10^{-8}$ {cite}`olmo2024olmo2`. Llama 2's $10^{-5}$ is large by comparison.

**Weight decay** in AdamW multiplies the weights by $1 - \eta \lambda$ each step, so in the absence of gradients a weight forgets its value over about $1/(\eta\lambda)$ steps: 33,333 steps at $\eta = 3 \times 10^{-4}$ and $\lambda = 0.1$. Not every parameter is decayed. A norm gain $\gamma$ multiplies a normalized vector, so decaying it toward zero shrinks the sublayer's output rather than regularizing a map; the usual rule decays the two-dimensional tensors and exempts the gains and biases. nanoGPT's `configure_optimizers` states it as "Any parameters that is 2D will be weight decayed, otherwise no", which decays the embedding table too {cite}`karpathy2022nanogpt`. OLMo 2 turns off decay for the embeddings {cite}`olmo2024olmo2`: a row of the embedding table receives a gradient only when its token is in the batch, but decay shrinks it at every step.

```python
import torch


def param_groups(model, weight_decay=0.1, decay_embeddings=True):
    """AdamW groups: decay matrices, not vectors (norm gains, biases)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        is_embedding = isinstance(model.get_submodule(name.rsplit(".", 1)[0]),
                                  torch.nn.Embedding)
        if p.ndim < 2 or (is_embedding and not decay_embeddings):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


model = torch.nn.Sequential(torch.nn.Embedding(1000, 64), torch.nn.RMSNorm(64),
                            torch.nn.Linear(64, 64, bias=True))
for flag in (True, False):
    groups = param_groups(model, decay_embeddings=flag)
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    print(f"decay embeddings {flag!s:5}: "
          f"decayed {sizes[0]:,}, not decayed {sizes[1]:,}")
optimizer = torch.optim.AdamW(param_groups(model), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
print(f"forgetting time 1 / (lr * decay) = {1 / (3e-4 * 0.1):,.0f} steps")
```

% output

```text
decay embeddings True : decayed 68,096, not decayed 128
decay embeddings False: decayed 4,096, not decayed 64,128
forgetting time 1 / (lr * decay) = 33,333 steps
```

(sec-ch7-memory-light-optimizers)=

### 3.2 Optimizers with less state

Adam keeps two numbers per parameter beside the weight. Two older alternatives keep fewer.

**Adafactor** stores the second moment of a weight matrix in factored form, "maintaining only the per-row and per-column sums of these moving averages, and estimating the per-parameter second moments based on these sums" {cite}`shazeer2018adafactor`. For a matrix of $d_{in} \times d_{out}$ weights that is $d_{in} + d_{out}$ numbers instead of $d_{in} d_{out}$, and with the first moment turned off ($\beta_1 = 0$) the optimizer state nearly vanishes. T5 was trained with it {cite}`raffel2019exploring`. PaLM used Adafactor "without factorization", which "is effectively equivalent to Adam" with a learning rate scaled by the size of each parameter matrix, together with Adafactor's increasing second-moment coefficient $\beta_2 = 1 - t^{-0.8}$ {cite}`chowdhery2023palm`.

**Lion** was found by a program search over update rules {cite}`chen2023symbolic`. It keeps only a momentum buffer $u_t$ and moves every parameter by the same amount, in the direction of the sign of an interpolation between momentum and gradient:

$$\theta_t = \theta_{t-1} - \eta \left(\mathrm{sign}\big(\beta_1 u_{t-1} + (1 - \beta_1) g_t\big) + \lambda \theta_{t-1}\right), \qquad u_t = \beta_2 u_{t-1} + (1 - \beta_2) g_t$$

with $\beta_1 = 0.9$ and $\beta_2 = 0.99$ by default. Because a sign has size 1 in every coordinate, its update is larger than Adam's on average, and the paper uses a learning rate 3 to 10 times smaller than AdamW's and a weight decay 3 to 10 times larger.

(sec-ch7-muon)=

### 3.3 Muon: orthogonalized updates

Adam and Lion treat each parameter as a separate number. A weight matrix $W$ is a map, and an update to it can be judged by how much it changes the map in each direction. {cite:t}`jordan2024muon` observe that the updates of SGD with momentum and of Adam for the weight matrices of a transformer "typically have very high condition number. That is, they are almost low-rank matrices", so an update moves the map a lot along a few directions and hardly at all along the rest. *Muon* replaces the momentum $u_t$ of each weight matrix by the nearest semi-orthogonal matrix, $R_u R_v^\top$ for the singular value decomposition $u_t = R_u \Sigma R_v^\top$, which keeps the directions and sets every singular value to 1. (Chapter 2 writes the decomposition $U \Sigma R^\top$; the left factor is renamed here because $u$ is the momentum.)

Computing an SVD of every matrix at every step is too slow, so Muon approximates the result with five steps of a Newton-Schulz iteration that uses only matrix products. Starting from $\Omega = u_t / \lVert u_t \rVert_F$, each step applies

$$\Omega \leftarrow 3.4445\, \Omega - 4.7750\, (\Omega \Omega^\top) \Omega + 2.0315\, (\Omega \Omega^\top)^2 \Omega$$

which leaves the singular vectors of $\Omega$ alone and maps each singular value $\Sigma_{jj}$ to $3.4445\,\Sigma_{jj} - 4.7750\,\Sigma_{jj}^3 + 2.0315\,\Sigma_{jj}^5$. The coefficients were tuned to make small singular values grow as fast as possible while the iterates stay in a band around 1 (the blog's design target is 0.7 to 1.3), accepting that the result is not exactly 1. After five steps every singular value above about 0.0015 ends between 0.68 and 1.21. {numref}`fig-ch7-newton-schulz` plots the polynomial and its iterates.

```python
import torch


def newton_schulz(u, steps=5, coefficients=(3.4445, -4.7750, 2.0315)):
    """Muon's approximate orthogonalization of a matrix, in float64."""
    a, b, c = coefficients
    p = u / (u.norm() + 1e-7)  # Frobenius norm 1: every singular value <= 1
    transpose = p.size(0) > p.size(1)
    if transpose:
        p = p.T
    for _ in range(steps):
        gram = p @ p.T
        p = a * p + (b * gram + c * gram @ gram) @ p
    return p.T if transpose else p


torch.manual_seed(0)
# a momentum matrix whose singular values span three orders of magnitude
left, _ = torch.linalg.qr(torch.randn(256, 128, dtype=torch.float64))
right, _ = torch.linalg.qr(torch.randn(128, 128, dtype=torch.float64))
spectrum = torch.logspace(0, -3, 128, dtype=torch.float64)
u = left @ torch.diag(spectrum) @ right.T
before = torch.linalg.svdvals(u)
after = torch.linalg.svdvals(newton_schulz(u))
print(f"singular values before normalization: {before.max():.3f} to {before.min():.5f}")
print(f"after five steps:       {after.max():.3f} to {after.min():.3f}")
exact = left @ right.T
close = (newton_schulz(u) - exact).norm() / exact.norm()
print(f"distance to the exact orthogonal factor: {close:.3f}")
```

% output

```text
singular values before normalization: 1.000 to 0.00100
after five steps:       1.202 to 0.155
distance to the exact orthogonal factor: 0.356
```

:::{figure} figures/newton-schulz.*
:name: fig-ch7-newton-schulz
:alt: A plot of output against input singular value, the input on a logarithmic axis from 0.0001 to 1. The dotted identity curve rises to 1 at the right. The curve for one step of the quintic polynomial rises from zero, peaks at about 1.2 near an input of 0.55 and comes back to 0.7 at 1. Each further step moves the rise to inputs about three times smaller and adds oscillations inside a shaded band from 0.68 to 1.21; after five steps every input above about 0.0015 lands in the band.
:width: 100%

What Muon's Newton-Schulz steps do to one singular value. Each step applies the
quintic polynomial of the text. Its slope at zero is 3.4445, so small singular values
grow fast; large ones are folded back into a band around 1 (shaded, 0.68 to 1.21).
After five steps every input above about 0.0015 is in the band; the logarithmic axis
shows how far below 1 the inputs start.
:::

Muon is applied only to the weight matrices inside the blocks. The embedding, the output head and the vector parameters keep AdamW. {cite:t}`liu2025muon` made Muon work at scale with two changes: weight decay, and a rescaling of the update by $0.2\sqrt{\max(d_{in}, d_{out})}$ so that its root-mean-square size matches AdamW's typical 0.2 to 0.4 and AdamW's learning rate and decay can be reused. They report that Muon "achieves ∼ 2× computational efficiency compared to AdamW" in their scaling experiments. Kimi K2, a mixture-of-experts model with a trillion parameters, was trained with Muon on 15.5 trillion tokens "with zero loss spike", using a *QK-clip* that rescales a head's query and key weights whenever its largest attention logit exceeds 100 {cite}`kimi2025k2`. Whether the advantage holds across architectures and at the largest scales is still being measured; as of 2025 AdamW remains the default of the major open reports.

(sec-ch7-optimizer-memory)=

### 3.4 Optimizer state in bytes

Training memory per parameter is the weight, its gradient and the optimizer state. With bf16 weights and gradients and an fp32 master copy of the weights, as in mixed-precision training ({ref}`sec-ch2-mixed-precision-training`), AdamW needs 16 bytes per parameter ({ref}`sec-ch2-memory-requirements`). The other optimizers save their state's share.

```python
STATE_BYTES = {  # fp32 optimizer state per parameter
    "AdamW": 8,           # first and second moment
    "Lion": 4,            # momentum only
    "Muon": 4,            # momentum (vectors and embeddings keep AdamW's 8)
    "Adafactor": 0,       # factored second moment, beta1 = 0: about 0 per parameter
    "SGD": 0,
}
BASE = 2 + 2 + 4  # bf16 weight, bf16 gradient, fp32 master weight


def training_bytes(n_params, optimizer):
    return n_params * (BASE + STATE_BYTES[optimizer])


for name in STATE_BYTES:
    total = training_bytes(8.03e9, name)  # Llama 3.1 8B
    print(f"{name:9s} {BASE + STATE_BYTES[name]:2d} bytes per parameter, "
          f"{total / 2**30:5.0f} GiB for 8.03B parameters")
```

% output

```text
AdamW     16 bytes per parameter,   120 GiB for 8.03B parameters
Lion      12 bytes per parameter,    90 GiB for 8.03B parameters
Muon      12 bytes per parameter,    90 GiB for 8.03B parameters
Adafactor  8 bytes per parameter,    60 GiB for 8.03B parameters
SGD        8 bytes per parameter,    60 GiB for 8.03B parameters
```

At 16 bytes per parameter even an 8-billion-parameter model needs 120 GiB before any activation is stored, more than one 80 GB accelerator holds. The next chapter, on distributed training, divides this state across devices; the choice of optimizer decides how much there is to divide. {numref}`fig-ch7-optimizer-memory` shows the breakdown.

:::{figure} figures/optimizer-memory.*
:name: fig-ch7-optimizer-memory
:alt: Horizontal stacked bars of bytes per parameter for five optimizers. Each bar has 2 bytes of bf16 weight, 2 of bf16 gradient and 4 of fp32 master weight; AdamW adds 8 bytes of state for 16 in total, Lion and Muon add 4 for 12, Adafactor without momentum and SGD add none for 8.
:width: 100%

Bytes per parameter in mixed-precision training, by optimizer. The weight, gradient
and master copy (8 bytes) are the same for all; the optimizer state is what differs.
Muon's 4 bytes apply to the matrices it handles; the embeddings, output head and vector
parameters it leaves to AdamW keep 8. Activations are not included.
:::

```{admonition} An Optimizer Step You Can Check
:class: tip
The notebook section {ref}`nb-train-optimizers` writes one AdamW step by hand, with
bias correction and decoupled weight decay, and checks it against `torch.optim.AdamW`;
checks the weight-decay parameter groups; and runs Muon's Newton-Schulz iteration on
matrices of several shapes.
```

(sec-ch7-schedules)=

## 4. Learning-Rate and Batch-Size Schedules

(sec-ch7-published-schedules)=

### 4.1 Schedules in published runs

Chapter 2 introduced linear warm-up and cosine decay ({ref}`sec-ch2-linear-warmup`, {ref}`sec-ch2-cosine-annealing`). The runs below show how much variety there is around that template.

```{tabularcolumns} \Y{0.17}\Y{0.13}\Y{0.17}\Y{0.53}
```

| Run | Peak rate | Warm-up | Decay |
| --- | --- | --- | --- |
| GPT-3 175B | $6 \times 10^{-5}$ | 375M tokens | cosine to 10% of the peak over 260B tokens, then constant |
| Llama 2 70B | $1.5 \times 10^{-4}$ | 2,000 steps | cosine to 10% of the peak |
| Llama 3 405B | $8 \times 10^{-5}$ | 8,000 steps | cosine to $8 \times 10^{-7}$ over 1.2M steps; the final 40M tokens linearly to 0 |
| OLMo 2 7B | $3 \times 10^{-4}$ | 2,000 steps | cosine aimed at 10% at 5T tokens, stopped at 4T; then linear to 0 during mid-training |
| DeepSeek LLM 67B | fitted | 2,000 steps | constant to 80% of the tokens, 31.6% of the peak to 90%, then 10% |
| DeepSeek-V3 | $2.2 \times 10^{-4}$ | 2K steps | constant to 10T tokens, cosine to $2.2 \times 10^{-5}$ over 4.3T, then $2.2 \times 10^{-5}$ for 333B and $7.3 \times 10^{-6}$ for 167B |

Sources: the training details of each report: {cite:t}`brown2020language` (Table 2.1 and Appendix B), {cite:t}`touvron2023llama2` (Table 1), {cite:t}`grattafiori2024llama` (the pretraining recipe), {cite:t}`olmo2024olmo2` (Table 3), {cite:t}`deepseek2024llm` and {cite:t}`deepseek2024v3`.

Warm-up is short in every case: 2,000 steps is well under 1% of a run of several hundred thousand steps. The decay is where the runs differ, and {numref}`fig-ch7-schedules` draws four of the shapes on a common axis.

:::{figure} figures/schedules.*
:name: fig-ch7-schedules
:alt: Learning rate as a fraction of the peak against the fraction of training completed, for four schedules. Cosine to 10 percent falls smoothly from 1 to 0.1. The DeepSeek LLM multi-step schedule stays at 1 until 0.8, drops to 0.316, and drops to 0.1 at 0.9. Warm-up-stable-decay stays at 1 until 0.8 and falls to 0 along a 1 minus square root curve. DeepSeek-V3's stays at 1 until about 0.68, follows a cosine to 0.1 until about 0.97, and ends in two short constant steps at 0.1 and 0.033.
:width: 100%

Four decay shapes, each as a fraction of its peak rate against the fraction of training
completed (warm-up omitted). DeepSeek-V3's breakpoints are its token counts (10T, 4.3T,
333B and 167B of 14.8T). The schedules that hold the peak longest decay fastest at the
end.
:::

(sec-ch7-end-of-schedule)=

### 4.2 Why the end of the schedule matters

With a constant learning rate, the parameters do not settle at a minimum: each step moves them by an amount set by $\eta$ and by the noise in the mini-batch gradient ({ref}`sec-ch2-minibatch-noise`), so they keep moving in a region around it. Lowering $\eta$ shrinks the region, and the loss drops as it does. A large share of the improvement in a run therefore happens during the decay, whenever the decay happens.

This is why the cosine schedule needs to know the length of the run in advance. Chinchilla's authors found that "when the cosine cycle overshoots the number of training steps by more than 25%, performance is noticeably degraded" {cite}`hoffmann2022training`: a run that is stopped before its schedule has decayed has not had its drop. It is also a pitfall in scaling studies: comparing checkpoints from the middle of one long cosine run, instead of separate runs each decayed at its own length, understates what the shorter runs would reach, and this was one of the differences between the Kaplan and Chinchilla fits.

(sec-ch7-wsd)=

### 4.3 Warm-up, stable, decay

If most of the gain comes from the decay, the decay can be separated from the rest. A *warm-up-stable-decay* (WSD) schedule holds the peak rate after warm-up and decays only in a short final phase. MiniCPM introduced the name and found that "a decay of 10% of the total tokens is sufficient to achieve the best results", with the loss falling during the decay "to be equal to or lower than" a cosine schedule's {cite}`hu2024minicpm`. {cite:t}`hagele2024scaling` showed that a constant rate followed by a short cooldown "scales predictably and reliably similar to cosine", and that a cooldown shaped as $1 - \sqrt{\cdot}$, which drops the rate quickly at first, "consistently performs better than linear decay". In this shape the rate at step $t$ of a cooldown from step $t_{dec}$ to $t_{end}$ is

$$\eta_t = \eta_{\max}\left(1 - \sqrt{\frac{t - t_{dec}}{t_{end} - t_{dec}}}\right)$$

The practical advantage is that the stable phase does not depend on the total length. A run can be continued from its last stable checkpoint when more data or compute arrives, and decays of several lengths can be branched off one stable run, each costing only its decay steps. A scaling study then needs one stable run per model size instead of one run per size and length. DeepSeek LLM's multi-step schedule has the same property with a staircase in place of a smooth decay: the first stage, at the peak rate, can be reused {cite}`deepseek2024llm`.

```python
import math


def wsd(t, peak, t_warm, t_end, decay_fraction=0.2):
    """Warm-up, stable, then a 1 - sqrt cooldown to zero over the final fraction."""
    t_decay = round(t_end * (1 - decay_fraction))
    if t < t_warm:
        return peak * (t + 1) / t_warm
    if t < t_decay:
        return peak
    return peak * (1 - math.sqrt((t - t_decay) / (t_end - t_decay)))


def cosine(t, peak, t_warm, t_end, floor=0.1):
    """Warm-up, then cosine decay to floor * peak at t_end."""
    if t < t_warm:
        return peak * (t + 1) / t_warm
    progress = (t - t_warm) / (t_end - t_warm)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


t_end, t_warm, peak = 1200, 50, 1e-2
for t in (0, 49, 500, 959, 960, 1100, 1199):
    print(f"step {t:4d}: wsd {wsd(t, peak, t_warm, t_end):.5f}  "
          f"cosine {cosine(t, peak, t_warm, t_end):.5f}")
area = {f.__name__: sum(f(t, peak, t_warm, t_end) for t in range(t_end))
        / (peak * t_end) for f in (wsd, cosine)}
print(f"mean rate as a share of the peak: wsd {area['wsd']:.3f}, "
      f"cosine {area['cosine']:.3f}")
```

% output

```text
step    0: wsd 0.00020  cosine 0.00020
step   49: wsd 0.01000  cosine 0.01000
step  500: wsd 0.01000  cosine 0.00701
step  959: wsd 0.01000  cosine 0.00194
step  960: wsd 0.01000  cosine 0.00193
step 1100: wsd 0.00236  cosine 0.00117
step 1199: wsd 0.00002  cosine 0.00100
mean rate as a share of the peak: wsd 0.847, cosine 0.549
```

The notebook trains a 135,488-parameter byte-level model on *Pride and Prejudice* for 1,200 steps with both schedules at a peak rate of $10^{-2}$, three seeds each, and branches 20% decays off one constant-rate run's stable phase to end at 800 and 1,200 steps. {numref}`fig-ch7-schedule-runs` shows the validation loss.

:::{figure} figures/schedule-runs.*
:name: fig-ch7-schedule-runs
:alt: Validation loss against training step from 0 to 1200 for a small byte-level model, mean of three seeds. The constant-rate stable run falls to about 1.91. The 1,200-step cosine run falls below it from around step 650 and ends near 1.81. The WSD branch decayed from step 960 drops steeply to about 1.75, below cosine. A second branch decayed from step 640 drops to about 1.92 at step 800, while a separate cosine run of 800 steps ends near 2.09.
:width: 100%

Validation loss of a 135,488-parameter byte-level model (mean of three seeds), recorded
by the notebook's training cell. The stable run holds the peak rate throughout; each WSD
branch copies it at 80% of its length (thin vertical lines) and decays to zero along
$1 - \sqrt{\cdot}$. Both branches end below the stable run and below the cosine runs of
the same length.
:::

In this small setting the WSD branches end below the cosine runs of the same length in every seed, by more than the spread between seeds. The 800-step cosine run even ends above the stable run at step 800: at that point the small model is still improving fast at the peak rate, so the cosine's early lowering of the rate costs more than its decay gains back, while the WSD branch keeps the peak until its last 160 steps. The decay does much of the work: in its 240 decay steps the 1,200-step branch drops by about 0.2, more than the stable run gains in its last 500 steps. The comparison is between schedules at one peak rate; at a peak rate chosen separately for each schedule the reported differences in large runs are smaller, and the robust result is that WSD matches cosine while letting the run be extended or branched.

(sec-ch7-batch-ramps)=

### 4.4 Batch-size ramps

The critical batch size grows as the loss falls ({ref}`sec-ch2-critical-batch`), so a batch that is efficient late in a run is wasteful early in it. Several runs therefore start small and grow the batch.

- GPT-3 increased the batch "linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens", reaching 3.2M tokens for the 175B model {cite}`brown2020language`.
- PaLM used 512 sequences of 2,048 tokens (1M tokens) until step 50k, 1,024 until step 115k, and 2,048 (4M tokens) for the rest {cite}`chowdhery2023palm`.
- Llama 3 405B began with 4M tokens in sequences of 4,096, doubled to 8M tokens in sequences of 8,192 after 252M tokens, and doubled again to 16M after 2.87T tokens {cite}`grattafiori2024llama`.
- DeepSeek-V3 grew from 3,072 to 15,360 sequences "in the training of the first 469B tokens" {cite}`deepseek2024v3`.

A ramp changes the gradient noise as well as the throughput, and the learning rate interacts with it: a larger batch averages away noise, so it tolerates a higher rate.

(sec-ch7-choosing-hyperparameters)=

### 4.5 Choosing the peak rate and the batch

The peak learning rate cannot be tuned on the full run. Two approaches make the small-scale optimum carry over.

**Fit the optimum as a function of compute.** DeepSeek LLM swept learning rates and batch sizes at small budgets and fitted power laws in the compute $C$, finding $\eta_{opt} = 0.3118\, C^{-0.1250}$ and $B_{opt} = 0.2920\, C^{0.3271}$, with $B_{opt}$ in tokens {cite}`deepseek2024llm`. The rate falls slowly and the batch grows quickly with the budget. The paper counts compute with its own measure of FLOPs per token, which includes attention and excludes the vocabulary, so the fit is only approximately applicable to $C = 6ND$.

```python
def deepseek_llm_hyperparameters(compute):
    """DeepSeek LLM's fitted optimal learning rate and batch size (tokens)."""
    return 0.3118 * compute ** -0.1250, 0.2920 * compute ** 0.3271


for compute in (1e20, 1e22, 1e24):
    lr, batch = deepseek_llm_hyperparameters(compute)
    print(f"C = {compute:.0e} FLOPs: learning rate {lr:.2e}, "
          f"batch {batch / 1e6:5.2f}M tokens")
```

% output

```text
C = 1e+20 FLOPs: learning rate 9.86e-04, batch  1.02M tokens
C = 1e+22 FLOPs: learning rate 5.54e-04, batch  4.59M tokens
C = 1e+24 FLOPs: learning rate 3.12e-04, batch 20.69M tokens
```

**Make the optimum independent of width.** In standard parameterization, the best learning rate shrinks as a model gets wider, because the update to a wider matrix changes its output more. The *maximal update parameterization* (μP) scales the initialization and the per-layer learning rates with width so that the optimal hyperparameters stay the same, and tunes them on a narrow proxy {cite}`yang2021tensor`. Its authors tuned a 40M-parameter proxy and transferred the result to GPT-3 6.7B, outperforming the published model "with tuning cost only 7% of total pretraining cost". μP needs changes to the model code and the optimizer setup, and only some of the published runs use it; others fit the optimum as above or extrapolate from a sweep.

(sec-ch7-scaling)=

## 5. Compute-Optimal Training and Beyond

(sec-ch7-fitting-scaling-law)=

### 5.1 Fitting the law

Chapter 2 gave the Chinchilla result: at a fixed compute $C \approx 6ND$, parameters and tokens should grow in equal proportion, about 20 tokens per parameter ({ref}`sec-ch2-chinchilla-scaling-laws`). {cite:t}`hoffmann2022training` reached it three ways. The first fixes several model sizes, trains each on many token counts with the schedule matched to each length, and takes the lowest loss reached at each compute. The second, the *isoFLOP* method, fixes several budgets, trains models of several sizes at each, fits a parabola to the loss against $\log N$ and takes its minimum. The third fits one formula to all runs,

$$\mathcal{L}(N, D) = \mathcal{L}_{\infty} + \frac{c_N}{N^{\gamma_N}} + \frac{c_D}{D^{\gamma_D}}$$

where $\mathcal{L}_{\infty}$ is the loss no model removes, and $c_N$, $\gamma_N$, $c_D$, $\gamma_D$ are the fitted scale and exponent of the parameter and data terms. Minimizing it subject to $6ND = C$ gives closed forms,

$$N_{opt} = K_{opt} \left(\frac{C}{6}\right)^{\frac{\gamma_D}{\gamma_N + \gamma_D}}, \qquad D_{opt} = \frac{1}{K_{opt}}\left(\frac{C}{6}\right)^{\frac{\gamma_N}{\gamma_N + \gamma_D}}, \qquad K_{opt} = \left(\frac{\gamma_N c_N}{\gamma_D c_D}\right)^{\frac{1}{\gamma_N + \gamma_D}}$$

The first two methods agreed with each other; the third, as published ($\mathcal{L}_{\infty} = 1.69$, $c_N = 406.4$, $c_D = 410.7$, $\gamma_N = 0.34$, $\gamma_D = 0.28$), gives smaller models. {cite:t}`besiroglu2024chinchilla` extracted the data points from the paper's figure and refitted the formula. They found that the published estimates "are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals", and their refit, $\mathcal{L}_{\infty} = 1.8172$, $c_N = 482.01$, $c_D = 2085.43$, $\gamma_N = 0.3478$, $\gamma_D = 0.3658$, agrees with the other two methods. The listing compares the two fits.

```python
FITS = {  # L_inf, c_N, c_D, gamma_N, gamma_D
    "Hoffmann et al.": (1.69, 406.4, 410.7, 0.34, 0.28),
    "Besiroglu et al.": (1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}


def chinchilla_loss(n, d, fit):
    l_inf, c_n, c_d, g_n, g_d = fit
    return l_inf + c_n / n ** g_n + c_d / d ** g_d


def compute_optimal(compute, fit):
    """Parameters and tokens that minimize the fitted loss at 6 N D = compute."""
    _, c_n, c_d, g_n, g_d = fit
    k = (g_n * c_n / (g_d * c_d)) ** (1 / (g_n + g_d))
    n = k * (compute / 6) ** (g_d / (g_n + g_d))
    return n, compute / (6 * n)


for name, fit in FITS.items():
    print(name)
    for compute in (5.88e23, 1e25, 3.8e25):
        n, d = compute_optimal(compute, fit)
        loss = chinchilla_loss(n, d, fit)
        print(f"  C = {compute:.2e}: N = {n / 1e9:6.1f}B, D = {d / 1e12:5.2f}T, "
              f"{d / n:5.1f} tokens per parameter, loss {loss:.3f}")
```

% output

```text
Hoffmann et al.
  C = 5.88e+23: N =   32.5B, D =  3.02T,  92.8 tokens per parameter, loss 1.930
  C = 1.00e+25: N =  116.8B, D = 14.27T, 122.1 tokens per parameter, loss 1.845
  C = 3.80e+25: N =  213.5B, D = 29.67T, 139.0 tokens per parameter, loss 1.817
Besiroglu et al.
  C = 5.88e+23: N =   73.0B, D =  1.34T,  18.4 tokens per parameter, loss 1.974
  C = 1.00e+25: N =  312.1B, D =  5.34T,  17.1 tokens per parameter, loss 1.912
  C = 3.80e+25: N =  618.7B, D = 10.24T,  16.5 tokens per parameter, loss 1.892
```

At Chinchilla's own budget ($6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 5.88 \times 10^{23}$ FLOPs), the published fit recommends a model of under half Chinchilla's size trained on more than twice its tokens, while the refit lands near Chinchilla's 70B and 1.4T. The exponents also differ in how the ratio moves with the budget: the published fit's $\gamma_N > \gamma_D$ makes the ratio grow with compute, while in the refit $\gamma_D > \gamma_N$ and the ratio shrinks slowly. Llama 3's own isoFLOP study, over budgets from $6 \times 10^{18}$ to $10^{22}$ FLOPs, predicted a compute-optimal "402B parameter model on 16.55T tokens" at its budget {cite}`grattafiori2024llama`; Meta trained 405B on 15.6T. {numref}`fig-ch7-isoflop` draws the refit's isoFLOP curves.

:::{figure} figures/isoflop.*
:name: fig-ch7-isoflop
:alt: Loss against model size on a log axis from ten million to ten trillion parameters, one U-shaped curve per compute budget from 1e18 to 1e26 FLOPs, each with a dot at its minimum. The minima move to larger models and lower losses as compute grows. Chinchilla's size at its budget sits close to the minimum of its curve.
:width: 100%

IsoFLOP curves of the Besiroglu et al. refit: for each budget $C$, the fitted loss of a
model of $N$ parameters trained on $D = C/(6N)$ tokens. The dots are the compute-optimal
sizes; Chinchilla (70B parameters, 1.4T tokens) is marked on its own budget's curve. To
the left of a minimum, a model is trained past the optimum; to the right, it is
undertrained.
:::

The curves are flat near their minima: a model half or twice the optimal size, trained on the tokens the budget then allows, loses little. That flatness is what makes the next section affordable.

(sec-ch7-overtraining)=

### 5.2 Training past the optimum

A compute-optimal model is the cheapest way to reach a loss in training. It is not the cheapest model to *use*. Generating a token costs about $2N$ FLOPs, so a model served for billions of requests can cost more in inference than in training, and a smaller model trained on more tokens can reach the same quality more cheaply over its life. {cite:t}`sardana2024beyond` include inference in the optimization and conclude that "LLM researchers expecting reasonably large inference demand (~1B requests) should train models smaller and longer than Chinchilla-optimal". The open models of 2023 to 2025 do this by large factors. {numref}`fig-ch7-tokens-per-parameter` places them against the 20-token line.

:::{figure} figures/tokens-per-parameter.*
:name: fig-ch7-tokens-per-parameter
:alt: Log-log scatter of training tokens against parameters for published models, with a line at 20 tokens per parameter. GPT-3 175B lies below the line at under 2 tokens per parameter. Chinchilla and Llama 1 65B lie on it. Llama 2 and later models lie above it, Llama 3 8B near 1,900 tokens per parameter and Qwen3 8B above 4,000.
:width: 100%

Training tokens against parameters for published models, with the 20-token
compute-optimal ratio as a line. Token counts are each report's stated pretraining
total; parameter counts are exact where the chapter 6 tables give them. DeepSeek-V3 is
plotted at its 37B active parameters, which set its compute per token. The trend since
2023 is to train far past the compute-optimal point.
:::

What overtraining costs can be read off the fit. The listing takes Llama 3 8B's shape ($8.03 \times 10^9$ parameters, about $15 \times 10^{12}$ tokens), computes its fitted loss under the refit, and finds the compute a compute-optimal model would need to reach the same loss.

```python
from scipy.optimize import brentq


def chinchilla_loss(n, d, fit):
    l_inf, c_n, c_d, g_n, g_d = fit
    return l_inf + c_n / n ** g_n + c_d / d ** g_d


def compute_optimal(compute, fit):
    """Parameters and tokens that minimize the fitted loss at 6 N D = compute."""
    _, c_n, c_d, g_n, g_d = fit
    k = (g_n * c_n / (g_d * c_d)) ** (1 / (g_n + g_d))
    n = k * (compute / 6) ** (g_d / (g_n + g_d))
    return n, compute / (6 * n)


fit = (1.8172, 482.01, 2085.43, 0.3478, 0.3658)  # Besiroglu et al.
n, d = 8.03e9, 15e12
compute = 6 * n * d
loss = chinchilla_loss(n, d, fit)
n_opt, d_opt = compute_optimal(compute, fit)
print(f"Llama 3 8B: {d / n:,.0f} tokens per parameter, C = {compute:.2e}, "
      f"fitted loss {loss:.4f}")
print(f"compute-optimal at that C: N = {n_opt / 1e9:.0f}B, "
      f"loss {chinchilla_loss(n_opt, d_opt, fit):.4f}")
same_loss = brentq(lambda c: chinchilla_loss(*compute_optimal(c, fit), fit) - loss,
                   1e21, compute)
n_small, d_small = compute_optimal(same_loss, fit)
print(f"same loss at the optimum: C = {same_loss:.2e} "
      f"({compute / same_loss:.1f}x less), N = {n_small / 1e9:.0f}B")
print(f"inference FLOPs per token: {2 * n / 1e9:.0f}G for 8B, "
      f"{2 * n_small / 1e9:.0f}G for the compute-optimal model")
```

% output

```text
Llama 3 8B: 1,868 tokens per parameter, C = 7.23e+23, fitted loss 2.0218
compute-optimal at that C: N = 81B, loss 1.9682
same loss at the optimum: C = 1.31e+23 (5.5x less), N = 34B
inference FLOPs per token: 16G for 8B, 68G for the compute-optimal model
```

Under the fit, a compute-optimal model would reach Llama 3 8B's loss with several times less training compute, but it would be several times larger, and every generated token would cost proportionally more. Once the model has been asked for enough tokens, the 8B is the cheaper choice. The extrapolation should not be read too precisely: the fit was made on models trained near 20 tokens per parameter, and it is applied here 90 times beyond that.

(sec-ch7-repeating-data)=

### 5.3 Repeating data

Training past the optimum needs tokens, and high-quality text is finite ({ref}`sec-ch5-weights-temperature-epochs`). {cite:t}`muennighoff2023scaling` trained models on repeated data and found that a few epochs are almost as good as fresh data: up to about four epochs the loss changes little compared with unique tokens, and the value of a repeated token falls off after that. In practice a mixture repeats its best sources a few times and takes the rest once; chapter 5's epoch cap is the implementation.

(sec-ch7-predicting)=

### 5.4 Predicting the large run

A scaling law is also a forecast. Llama 3 fitted isoFLOP curves on models from 40M to 16B parameters, then mapped the predicted loss to downstream accuracy with a second fit, so that the 405B model's benchmark scores were predicted before it was trained {cite}`grattafiori2024llama`. The same small runs that set the hyperparameters ({ref}`sec-ch7-choosing-hyperparameters`) set the size. A forecast of this kind assumes the large run behaves like the small ones, which is the subject of the next section.

```{admonition} Scaling Numbers You Can Run
:class: tip
The notebook section {ref}`nb-train-scaling` computes the compute-optimal sizes under
both fits at several budgets, checks the closed form against a numerical minimization
along each isoFLOP curve, and evaluates the cost of training a small model far past the
optimum.
```

(sec-ch7-stability)=

## 6. Stability

(sec-ch7-loss-spikes)=

### 6.1 Loss spikes

A loss spike is a sudden rise in the training loss, sometimes recovering within a few hundred steps, sometimes leading to divergence. PaLM 540B saw about 20 of them "despite the fact that gradient clipping was enabled". The remedy was operational: "We re-started training from a checkpoint roughly 100 steps before the spike started, and skipped roughly 200-500 data batches" {cite}`chowdhery2023palm`. The authors replayed the surrounding batches from a different, earlier checkpoint and saw no spike, so the cause was the combination of particular batches with a particular state of the model, not bad data.

Restart-and-skip is the standard response and needs three things from the infrastructure: checkpoints frequent enough to roll back a few hundred steps, a data loader that can skip batches deterministically ({ref}`sec-ch5-mixing-sampling` covers resumable loaders), and monitoring that sees the spike quickly.

(sec-ch7-logit-growth)=

### 6.2 Growing logits

Two instabilities have known mechanisms, and {cite:t}`wortsman2024smallscale` reproduced both in small models by training at high learning rates.

**Attention logits grow.** If the query and key projections grow, the attention scores grow with them, the softmax saturates to one-hot weights, and the gradient through it vanishes for the other positions. Normalizing the queries and keys bounds the logits ({ref}`sec-ch6-normalizing-queries-keys`), and OLMo 2, Qwen3 and Gemma 3 do so. Kimi K2's QK-clip does it without a normalization layer, by rescaling the weights after any step that pushes a head's largest logit above 100 {cite}`kimi2025k2`.

**Output logits diverge.** The log-normalizer drifts away from zero until the loss becomes unstable; the z-loss of {ref}`sec-ch7-auxiliary-terms` holds it in place.

Both are guards against growth that the loss alone does not penalize. A related setting to get right is $\epsilon$ in Adam ({ref}`sec-ch7-adamw-practice`): it is not a guard against growth, but when gradients become very small a large $\epsilon$ shrinks the updates of exactly the layers whose gradients are smallest, and {cite:t}`wortsman2024smallscale` found that this, too, shows up in small proxies.

(sec-ch7-monitoring)=

### 6.3 What to monitor

The loss is a lagging signal: by the time it spikes, the cause is some steps in the past. Runs log quantities that move first.

- The global gradient norm, and how often clipping triggers. A clipping rate that rises from rare to frequent precedes many spikes.
- The ratio of each layer's update norm to its weight norm. AdamW keeps it near $\eta$ times a constant; a layer whose ratio departs from the others is changing faster than it should.
- The largest attention logit per layer, and the mean of $\log Z$ at the output.
- The activation scale of the residual stream at a few depths.

```python
import torch


def step_statistics(model, update_from, lr):
    """Global gradient norm and per-matrix update/weight ratios after optimizer.step().

    `update_from` holds copies of the parameters taken before the step.
    """
    grad_norm = torch.cat([p.grad.flatten() for p in model.parameters()]).norm()
    ratios = {}
    for (name, p), before in zip(model.named_parameters(), update_from):
        if p.ndim == 2:
            ratios[name] = ((p - before).norm() / before.norm()).item() / lr
    return grad_norm.item(), ratios


torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(),
                            torch.nn.Linear(256, 64))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
for step in range(3):
    x = torch.randn(32, 64)
    loss = (model(x) - x).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    norm, ratios = step_statistics(model, before, lr=1e-3)
    shown = ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
    print(f"step {step}: grad norm {norm:.3f}; update/weight per unit rate: {shown}")
```

% output

```text
step 0: grad norm 0.336; update/weight per unit rate: 0.weight 13.84, 2.weight 27.82
step 1: grad norm 0.347; update/weight per unit rate: 0.weight 10.43, 2.weight 21.14
step 2: grad norm 0.344; update/weight per unit rate: 0.weight 9.20, 2.weight 18.61
```

On the first step every Adam update has size close to $\eta$ in every coordinate (the first step is a sign step), so the ratio for each matrix is about $\sqrt{d_{in} d_{out}}$ divided by the matrix's norm; it then drifts with the gradients. What matters in a real run is the ratio's history: a sudden change in one layer is the alarm.

(sec-ch7-precision)=

### 6.4 Precision

Mixed precision in bf16 with fp32 master weights and fp32 reductions is the norm ({ref}`sec-ch2-mixed-precision-training`), and its stability record is good. DeepSeek-V3 went further and trained in FP8, with scaling factors computed per tile rather than per tensor, "on a 1x128 tile basis (i.e., per token per 128 channels)" for activations and "on a 128x128 block basis" for weights, so that one outlier affects only its tile {cite}`deepseek2024v3`. The next chapter takes up low-precision formats with the rest of efficient training.

(sec-ch7-stages)=

## 7. The Stages of a Run

(sec-ch7-annealing)=

### 7.1 Pretraining, annealing and long context

The final phase of the learning-rate decay is the part of training with the most influence per token on the final model, so runs put their best data there. Llama 3 annealed "on the final 40M tokens" with the learning rate going linearly to zero at a context of 128K tokens, upsampling high-quality sources {cite}`grattafiori2024llama`. OLMo 2 made it an explicit second stage, *mid-training*: after 3.90T tokens of the main mixture, the 7B model was trained on 50B tokens of a separate high-quality mix (Dolmino), with the rate decaying linearly to zero {cite}`olmo2024olmo2`. Long-context extension ({ref}`sec-ch6-long-context-training`) is another short stage near the end; Llama 3 placed it before the anneal, which therefore ran at the full 128K context. {numref}`fig-ch7-run-stages` shows the sequence.

:::{figure} figures/run-stages.*
:name: fig-ch7-run-stages
:alt: A left-to-right pipeline of the stages of a pretraining run. Warm-up leads to the main pretraining stage on the full mixture at a stable or slowly decaying rate, then to an optional long-context extension, then to annealing or mid-training on the best data with the rate decaying to zero, then checkpoint averaging, producing the base model. Notes under two stages give OLMo 2 7B's token counts and Llama 3 405B's 40M-token anneal at 128K context.
:width: 100%

The stages of a pretraining run. The main stage holds most of the tokens; the annealing
stage, with the learning rate going to zero, holds the best data; checkpoint averaging
finishes the base model, which post-training then adapts. Where a run extends its
context, Llama 3 among them, the extension comes before the anneal.
:::

(sec-ch7-averaging)=

### 7.2 Averaging checkpoints

At a constant learning rate the parameters move around a minimum ({ref}`sec-ch7-end-of-schedule`), and their average is closer to it than any one of them. Averaging the iterates, *Polyak averaging*, is an old remedy, and Llama 3 averaged checkpoints during annealing {cite}`grattafiori2024llama`. OLMo 2 averages whole runs: its 7B model is the average of three mid-training runs of 50B tokens each, and its 13B model of three runs of 100B tokens and one of 300B, each started from the same pretraining checkpoint with a different data order {cite}`olmo2024olmo2`. Runs that share a starting point stay close enough that their average is a good model; runs from different initializations are not, because their hidden units are permuted relative to each other.

```python
import torch

torch.manual_seed(0)
truth = torch.randn(20, dtype=torch.float64)
features = torch.randn(4000, 20, dtype=torch.float64)
labels = features @ truth + 0.5 * torch.randn(4000, dtype=torch.float64)


def loss(w):
    return ((features @ w - labels) ** 2).mean().item()


w = torch.zeros(20, dtype=torch.float64)
iterates = []
for step in range(2000):  # SGD at a constant rate, batches of 8
    rows = torch.randint(0, 4000, (8,))
    grad = 2 * features[rows].T @ (features[rows] @ w - labels[rows]) / 8
    w = w - 0.02 * grad
    iterates.append(w)
average = torch.stack(iterates[1000:]).mean(0)  # mean of the last 1,000 iterates
best = torch.linalg.lstsq(features, labels).solution
print(f"loss of the last iterate {loss(w):.4f}, of the average {loss(average):.4f}, "
      f"of the exact minimum {loss(best):.4f}")
assert loss(average) < loss(w)
```

% output

```text
loss of the last iterate 0.2540, of the average 0.2476, of the exact minimum 0.2469
```

(sec-ch7-budget)=

### 7.3 From FLOPs to GPU-hours

A compute budget becomes a schedule once a throughput is assumed. *Model FLOPs utilization* (MFU) is the share of the hardware's peak arithmetic that goes into the model's useful FLOPs: $6N$ per token, plus attention. Llama 3 405B reports a bf16 MFU of 38% to 43% on up to 16K H100 GPUs, and $3.8 \times 10^{25}$ FLOPs in total {cite}`grattafiori2024llama`. DeepSeek-V3 reports 2.664M H800 GPU-hours for pretraining on 14.8T tokens with 37B active parameters {cite}`deepseek2024v3`. The listing converts both, with 989 TFLOP/s as the dense bf16 peak of either GPU.

```python
PEAK = 989e12  # dense bf16 FLOP/s of an H100 or H800 (SXM)


def gpu_hours(flops, mfu, peak=PEAK):
    return flops / (mfu * peak) / 3600


llama = 6 * 405e9 * 15.6e12
print(f"Llama 3 405B: 6ND = {llama:.2e} FLOPs (reported 3.8e25)")
for mfu in (0.38, 0.43):
    hours = gpu_hours(llama, mfu)
    print(f"  at {mfu:.0%} MFU: {hours / 1e6:.1f}M GPU-hours, "
          f"{hours / 16_384 / 24:.0f} days on 16,384 GPUs")
deepseek = 6 * 37e9 * 14.8e12
mfu = deepseek / (2.664e6 * 3600 * PEAK)
print(f"DeepSeek-V3: 6ND = {deepseek:.2e} FLOPs in 2.664M GPU-hours: "
      f"{mfu:.0%} of the bf16 peak")
```

% output

```text
Llama 3 405B: 6ND = 3.79e+25 FLOPs (reported 3.8e25)
  at 38% MFU: 28.0M GPU-hours, 71 days on 16,384 GPUs
  at 43% MFU: 24.8M GPU-hours, 63 days on 16,384 GPUs
DeepSeek-V3: 6ND = 3.29e+24 FLOPs in 2.664M GPU-hours: 35% of the bf16 peak
```

The two runs differ by an order of magnitude in compute. Llama 3 405B needed tens of millions of GPU-hours, about two months on 16,384 GPUs; DeepSeek-V3, with a twelfth of the training FLOPs thanks to its sparse layers, needed under three million. DeepSeek-V3's implied utilization is a lower bound in one sense and an upper bound in another: $6N$ leaves out attention, which raises the true count, and much of its arithmetic ran in FP8, whose peak is twice bf16's. Neither figure includes failed runs, ablations or the hardware faults that a run of this length meets (Llama 3 reports 466 job interruptions in a 54-day period), which is why a budget needs a margin.

(sec-ch7-summary)=

## Summary

- The loss is a mean over loss-bearing tokens. Gradient accumulation and data parallelism reproduce one large batch only when each piece divides its summed loss by the token count of the whole batch.
- The z-loss pulls the log-normalizer toward zero without changing the predicted distribution; soft-capping bounds the logits and is part of the model.
- Span corruption, prefix language modeling and fill-in-the-middle differ in what the model sees and which positions carry a loss; fill-in-the-middle at a rate of 0.5 gives a causal model infilling without measurably hurting left-to-right generation.
- Published runs use AdamW with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1 on the matrices and clipping at 1.0. The short second-moment memory limits the large steps that follow a sudden change in a gradient.
- Adafactor, Lion and Muon cut optimizer state from 8 bytes per parameter to about 0 or 4; Muon also replaces each matrix update by an orthogonalized one, computed with five Newton-Schulz steps.
- Much of a run's improvement happens during the final learning-rate decay. Warm-up-stable-decay schedules move the decay to a short final phase, so one stable run can be extended or branched into decays of several lengths.
- Batch sizes are ramped as the critical batch grows; peak rates and batch sizes are fitted as power laws in compute or transferred from narrow models with μP.
- The two published fits of the Chinchilla data disagree; the refit agrees with the paper's other methods and puts the compute-optimal ratio near 20 tokens per parameter. Models meant for heavy use are trained tens to hundreds of times past it.
- Loss spikes are handled by rolling back and skipping batches; attention-logit growth and output-logit divergence are prevented by QK normalization or clipping and by the z-loss.
- A run ends with long-context training where it has one, an annealing stage on its best data and checkpoint averaging; its cost in GPU-hours is its FLOPs divided by the utilization times the peak.

(sec-ch7-exercises)=

## Exercises

1. **Accumulating by hand.** Four micro-batches hold 100, 60, 30 and 10 loss-bearing tokens, and their mean losses are 2.0, 2.4, 3.0 and 4.0. (a) What is the loss of the accumulated batch, as one batch would compute it? (b) What does the mean of the micro-batch means give? (c) By what factor does the mean of means over- or under-weight a token of each micro-batch?

````{admonition} Solution 1
:class: dropdown

(a) The batch loss is the summed loss over all tokens divided by their number. (b) The
mean of means averages the four means. (c) Under the mean of means each micro-batch has
weight $1/4$, so a token of micro-batch $i$ has weight $1/(4 n_i)$ for $n_i$ tokens,
against $1/200$ in the batch mean.

```python
tokens = [100, 60, 30, 10]
means = [2.0, 2.4, 3.0, 4.0]
batch = sum(n * m for n, m in zip(tokens, means)) / sum(tokens)
print(f"batch loss {batch:.3f}, mean of means {sum(means) / len(means):.3f}")
for n in tokens:
    print(f"{n:3d} tokens: weight per token x {sum(tokens) / (len(tokens) * n):.3f}")
```

% output

```text
batch loss 2.370, mean of means 2.850
100 tokens: weight per token x 0.500
 60 tokens: weight per token x 0.833
 30 tokens: weight per token x 1.667
 10 tokens: weight per token x 5.000
```

The mean of means reports 2.85 instead of 2.37, because the 10 tokens of the last
micro-batch, which has the highest loss, count five times as much as they should and
the 100 tokens of the first count half.
````

2. **Where the z-loss pulls.** Add the same constant to every logit in the per-token loss $\ell$ of {ref}`sec-ch7-auxiliary-terms`. (a) Show that the cross-entropy part does not depend on the constant, and find the constant that minimizes the total. (b) Show that the gradient of $\lambda_z (\log Z)^2$ with respect to $z_v$ is $2\lambda_z \log Z\, p_v$, and that it sums to $2 \lambda_z \log Z$ over the vocabulary.

````{admonition} Solution 2
:class: dropdown

(a) Adding a constant to every logit multiplies every $e^{z_v}$ by the exponential of
the constant, so $Z$ is multiplied by it too and $\log Z$ grows by the constant. The
probability of the target token is its exponentiated logit divided by $Z$; both are
multiplied by the same factor, so the probability, and with it the cross-entropy, does
not change. The total is the cross-entropy plus $\lambda_z$ times the square of the
shifted $\log Z$, smallest when the constant is $-\log Z$, where the z-loss is zero: the z-loss selects the one shift of the
logits that makes $\log Z = 0$, and the model's predictions are the same for every
shift.

(b) $\partial \log Z / \partial z_v = e^{z_v}/Z = p_v$, so by the chain rule
$\partial\, \lambda_z (\log Z)^2 / \partial z_v = 2 \lambda_z \log Z\, p_v$. Summed over
$v$, the probabilities sum to 1, which leaves $2\lambda_z \log Z$: every logit moves in
the same direction, by an amount proportional to its probability.
````

3. **Span-corruption lengths.** Use `span_corrupt` from {ref}`sec-ch7-denoising` on a sequence of 512 tokens with (a) T5's setting (rate 0.15, mean span 3) and (b) one of UL2's X-denoisers (rate 0.5, mean span 64). For each, give the number of spans, the input length and the target length, and the share of the original tokens that appear in the target.

````{admonition} Solution 3
:class: dropdown

The function corrupts $\mathrm{round}(0.15 \times 512) = 77$ tokens in
$\mathrm{round}(77/3) = 26$ spans, or $256$ tokens in $\mathrm{round}(256/64) = 4$ spans.
The input keeps the other tokens and gains one sentinel per span; the target holds every
span with its sentinel and one final sentinel.

```python
import numpy as np


def span_corrupt(tokens, rate=0.15, mean_span=3.0, sentinel_base=32_000, seed=0):
    """T5-style span corruption: returns (inputs, targets).

    Picks n_spans = round(rate * len / mean_span) spans whose lengths sum to
    round(rate * len), placed at random non-overlapping positions; each span is
    replaced in the input by a sentinel, and the target lists sentinel, span, ...
    """
    rng = np.random.default_rng(seed)
    n = len(tokens)
    n_noise = round(rate * n)
    n_spans = max(1, round(n_noise / mean_span))
    # random split of n_noise into n_spans positive lengths, and of the rest into gaps
    cuts = np.sort(rng.choice(np.arange(1, n_noise), n_spans - 1, replace=False))
    span_lengths = np.diff(np.concatenate([[0], cuts, [n_noise]]))
    cuts = np.sort(rng.choice(np.arange(1, n - n_noise), n_spans - 1, replace=False))
    gap_lengths = np.diff(np.concatenate([[0], cuts, [n - n_noise]]))
    inputs, targets, position = [], [], 0
    for k, (gap, span) in enumerate(zip(gap_lengths, span_lengths)):
        inputs += list(tokens[position:position + gap]) + [sentinel_base + k]
        position += gap
        targets += [sentinel_base + k] + list(tokens[position:position + span])
        position += span
    return inputs, targets + [sentinel_base + n_spans]  # a last sentinel ends it


for rate, span in ((0.15, 3.0), (0.5, 64.0)):
    inputs, targets = span_corrupt(list(range(512)), rate=rate, mean_span=span)
    spans = sum(t >= 32_000 for t in inputs)
    original = sum(t < 32_000 for t in targets)
    print(f"rate {rate}, mean span {span:g}: {spans} spans, input {len(inputs)}, "
          f"target {len(targets)}, {original / 512:.0%} of the tokens in the target")
```

% output

```text
rate 0.15, mean span 3: 26 spans, input 461, target 104, 15% of the tokens in the target
rate 0.5, mean span 64: 4 spans, input 260, target 261, 50% of the tokens in the target
```

T5's setting turns a 512-token sequence into an input of 461 tokens and a target of
104, so 15% of the tokens carry a loss. The X-denoiser halves the input and moves half
the sequence into the target, a task close to generating long text from a partial
context, which is what UL2 meant it to teach.
````

4. **How fast the second moment catches up.** A parameter's gradient has been constant at a small value for a long time and jumps to a constant $g$. Ignoring the old value of $v_t$, how many steps does $\sqrt{v_t}$ take to reach 90% of $|g|$ for $\beta_2 = 0.999$ and for $\beta_2 = 0.95$? How many does the first moment take with $\beta_1 = 0.9$?

````{admonition} Solution 4
:class: dropdown

Counting $t$ from the jump, after $t$ steps at the new gradient
$v_t = \beta_2^t v_{old} + (1 - \beta_2^t) g^2$, where $v_{old}$ is the second moment
before the jump. Ignoring $v_{old}$, $\sqrt{v_t} \geq 0.9 |g|$ needs
$1 - \beta_2^t \geq 0.81$, so $t \geq \ln 0.19 / \ln \beta_2$. The first moment reaches
90% of $g$ when $1 - \beta_1^t \geq 0.9$.

```python
import math

for beta2 in (0.999, 0.95):
    print(f"beta2 = {beta2}: {math.ceil(math.log(0.19) / math.log(beta2))} steps")
print(f"beta1 = 0.9: {math.ceil(math.log(0.1) / math.log(0.9))} steps")
```

% output

```text
beta2 = 0.999: 1660 steps
beta2 = 0.95: 33 steps
beta1 = 0.9: 22 steps
```

With $\beta_2 = 0.999$ the denominator needs about 1,660 steps while the numerator needs
22, so for well over a thousand steps Adam divides an up-to-date first moment by an
out-of-date second moment, and the step is larger than its usual size. With
$\beta_2 = 0.95$ the two catch up within about the same time.
````

5. **Memory for a 70B model.** For a dense model of $70 \times 10^9$ parameters trained in mixed precision, how many 80 GB accelerators are needed just to hold the weights, gradients, master weights and optimizer state with AdamW, and with Muon, if 98% of the parameters are in matrices that Muon handles and the rest keep AdamW? Activations are extra.

````{admonition} Solution 5
:class: dropdown

AdamW needs 16 bytes per parameter; under Muon, the 98% in matrices need 12 and the 2%
left to AdamW need 16.

```python
import math

n = 70e9
adamw = 16 * n
muon = (0.98 * 12 + 0.02 * 16) * n
for name, total in (("AdamW", adamw), ("Muon", muon)):
    print(f"{name}: {total / 1e9:,.0f} GB, at least {math.ceil(total / 80e9)} "
          f"accelerators of 80 GB")
```

% output

```text
AdamW: 1,120 GB, at least 14 accelerators of 80 GB
Muon: 846 GB, at least 11 accelerators of 80 GB
```

Muon halves the optimizer state and saves about a quarter of the total, three of
fourteen accelerators. In
practice the state is divided across many more devices than this minimum, together
with the activations, which the next chapter covers.
````

6. **What branching saves.** A team wants models trained for 25,000, 50,000 and 100,000 steps. Compare the total number of training steps for (a) three separate cosine runs and (b) one warm-up-stable-decay run whose stable phase is branched into 20% decays for each length.

````{admonition} Solution 6
:class: dropdown

(a) Separate runs cost $25{,}000 + 50{,}000 + 100{,}000 = 175{,}000$ steps. (b) The
stable run must reach the start of the longest decay, 80,000 steps; the three decays
cost 5,000, 10,000 and 20,000 steps, for 115,000 in total, 34% fewer. The saving grows
with the number of lengths: each extra length costs only its decay. For scaling-law
studies, which need many lengths per model size, this is the main attraction of the
schedule.
````

7. **The closed form, checked.** Derive $N_{opt}$ in {ref}`sec-ch7-fitting-scaling-law` by substituting $D = C/(6N)$ into the fitted loss and setting the derivative with respect to $N$ to zero. Then check the closed form numerically at $C = 10^{24}$ FLOPs for both fits.

````{admonition} Solution 7
:class: dropdown

With $D = C/(6N)$ the loss is
$\mathcal{L}_{\infty} + c_N N^{-\gamma_N} + c_D (6N/C)^{\gamma_D}$. Its derivative in $N$
is $-\gamma_N c_N N^{-\gamma_N - 1} + \gamma_D c_D (6/C)^{\gamma_D} N^{\gamma_D - 1}$,
which is zero when
$N^{\gamma_N + \gamma_D} = \frac{\gamma_N c_N}{\gamma_D c_D} \left(\frac{C}{6}\right)^{\gamma_D}$,
the closed form with $K_{opt} = (\gamma_N c_N / (\gamma_D c_D))^{1/(\gamma_N + \gamma_D)}$.

```python
from scipy.optimize import minimize_scalar

FITS = {  # L_inf, c_N, c_D, gamma_N, gamma_D
    "Hoffmann et al.": (1.69, 406.4, 410.7, 0.34, 0.28),
    "Besiroglu et al.": (1.8172, 482.01, 2085.43, 0.3478, 0.3658),
}


def chinchilla_loss(n, d, fit):
    l_inf, c_n, c_d, g_n, g_d = fit
    return l_inf + c_n / n ** g_n + c_d / d ** g_d


def compute_optimal(compute, fit):
    """Parameters and tokens that minimize the fitted loss at 6 N D = compute."""
    _, c_n, c_d, g_n, g_d = fit
    k = (g_n * c_n / (g_d * c_d)) ** (1 / (g_n + g_d))
    n = k * (compute / 6) ** (g_d / (g_n + g_d))
    return n, compute / (6 * n)


def isoflop_loss(log_n, compute, fit):
    """Fitted loss of a model of 10**log_n parameters trained with `compute` FLOPs."""
    return chinchilla_loss(10 ** log_n, compute / (6 * 10 ** log_n), fit)


compute = 1e24
for name, fit in FITS.items():
    closed, _ = compute_optimal(compute, fit)
    result = minimize_scalar(isoflop_loss, args=(compute, fit), bounds=(8, 13),
                             method="bounded", options={"xatol": 1e-9})
    print(f"{name}: closed form {closed / 1e9:.2f}B, "
          f"numerical {10 ** result.x / 1e9:.2f}B")
```

% output

```text
Hoffmann et al.: closed form 41.30B, numerical 41.30B
Besiroglu et al.: closed form 95.86B, numerical 95.86B
```
````

8. **When overtraining pays off.** {ref}`sec-ch7-overtraining` found that under the refit a compute-optimal model of about 34B parameters reaches Llama 3 8B's fitted loss with $1.31 \times 10^{23}$ training FLOPs instead of $7.23 \times 10^{23}$. Counting $2N$ FLOPs per generated or processed token at inference, after how many tokens of inference does the 8B model become the cheaper of the two in total compute?

````{admonition} Solution 8
:class: dropdown

The compute-optimal model saves training compute and spends more on every inference
token; the break-even point is the saving divided by the extra cost per token.

```python
train_8b, train_opt = 7.23e23, 1.31e23
n_8b, n_opt = 8.03e9, 34e9
extra_per_token = 2 * (n_opt - n_8b)
tokens = (train_8b - train_opt) / extra_per_token
print(f"saving {train_8b - train_opt:.2e} FLOPs; "
      f"extra {extra_per_token:.2e} FLOPs per inference token")
print(f"break-even after {tokens:.2e} tokens")
```

% output

```text
saving 5.92e+23 FLOPs; extra 5.19e+10 FLOPs per inference token
break-even after 1.14e+13 tokens
```

About eleven trillion tokens of inference, less than the model's own training set. A
model served to many users passes that quickly, which is the argument of
{cite:t}`sardana2024beyond`. The comparison ignores that a smaller model also needs
less memory per request (a smaller KV cache, {ref}`sec-ch6-attention-variants`), which
favors it further.
````

9. **A budget in GPU-hours.** OLMo 2 7B has $7.30 \times 10^9$ parameters and was trained on $4.05 \times 10^{12}$ tokens in all (3.90T of pretraining and three anneals of 50B). Estimate its training compute with $6ND$, and the GPU-hours and days it needs on 1,024 accelerators with a dense bf16 peak of 989 TFLOP/s at 40% MFU.

````{admonition} Solution 9
:class: dropdown

```python
flops = 6 * 7.30e9 * 4.05e12
hours = flops / (0.40 * 989e12) / 3600
print(f"{flops:.2e} FLOPs, {hours:,.0f} GPU-hours, {hours / 1024 / 24:.1f} days "
      f"on 1,024 GPUs")
```

% output

```text
1.77e+23 FLOPs, 124,558 GPU-hours, 5.1 days on 1,024 GPUs
```

About $1.8 \times 10^{23}$ FLOPs, roughly 125,000 GPU-hours, five days on a thousand
accelerators, about two hundred times less than Llama 3 405B. The estimate leaves out
attention FLOPs, evaluation, restarts and the ablations that chose the recipe.
````

(sec-ch7-further-reading)=

## Further Reading

(sec-ch7-further-reading-objectives)=

### Objectives

- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
  Transformer](https://arxiv.org/abs/1910.10683) {cite}`raffel2019exploring` - span
  corruption and a systematic comparison of objectives
- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)
  {cite}`tay2023ul2` - a mixture of denoisers and the prefix language model
- [Efficient Training of Language Models to Fill in the
  Middle](https://arxiv.org/abs/2207.14255) {cite}`bavarian2022efficient` -
  fill-in-the-middle, its formats and the FIM rate
- [Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)
  {cite}`hf2024gradaccum` - the loss normalization bug and its fix

(sec-ch7-further-reading-optimizers)=

### Optimizers and hyperparameters

- [Adafactor](https://arxiv.org/abs/1804.04235) {cite}`shazeer2018adafactor` and
  [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)
  {cite}`chen2023symbolic` - factored second moments, and Lion
- [Muon](https://kellerjordan.github.io/posts/muon/) {cite}`jordan2024muon` and [Muon is
  Scalable for LLM Training](https://arxiv.org/abs/2502.16982) {cite}`liu2025muon` -
  orthogonalized updates and how to scale them
- [Tensor Programs V](https://arxiv.org/abs/2203.03466) {cite}`yang2021tensor` - μP and
  hyperparameter transfer
- [Small-scale proxies for large-scale Transformer training
  instabilities](https://arxiv.org/abs/2309.14322) {cite}`wortsman2024smallscale` -
  logit growth, the z-loss, QK normalization and $\epsilon$

(sec-ch7-further-reading-schedules-scaling)=

### Schedules and scaling

- [MiniCPM](https://arxiv.org/abs/2404.06395) {cite}`hu2024minicpm` and [Scaling Laws and
  Compute-Optimal Training Beyond Fixed Training
  Durations](https://arxiv.org/abs/2405.18392) {cite}`hagele2024scaling` -
  warm-up-stable-decay schedules and cooldowns
- [DeepSeek LLM](https://arxiv.org/abs/2401.02954) {cite}`deepseek2024llm` - fitted
  learning rates and batch sizes, and the multi-step schedule
- [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102)
  {cite}`besiroglu2024chinchilla` - refitting the parametric law
- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448)
  {cite}`sardana2024beyond` - scaling laws with inference included

(sec-ch7-further-reading-reports)=

### Technical reports

- [PaLM](https://arxiv.org/abs/2204.02311) {cite}`chowdhery2023palm` - the z-loss, loss
  spikes and the restart-and-skip remedy
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
  {cite}`grattafiori2024llama` - isoFLOP forecasting, batch ramps, annealing and
  checkpoint averaging at 405B
- [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) {cite}`olmo2024olmo2` - stability
  changes, mid-training and model souping, with open code and configurations
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) {cite}`deepseek2024v3` - a
  constant-then-cosine schedule, FP8 training and multi-token prediction
- [Kimi K2](https://arxiv.org/abs/2507.20534) {cite}`kimi2025k2` - Muon with QK-clip at a
  trillion parameters

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
