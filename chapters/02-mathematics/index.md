# Mathematical Foundations

A language model is a probability distribution over text, trained by gradient descent on a loss from information theory, and the way its loss falls with size and data is described by empirical power laws. This chapter derives the mathematics behind each of these steps: the linear algebra and calculus of the transformer's gradients, the information theory of the loss, the optimizers and schedules that minimize it, the scaling laws and compute counts that plan a training run, and the floating-point arithmetic that limits it.

(sec-ch2-chapter-overview)=

## Chapter Overview

The chapter has six parts, each needed to read the rest of the book. A short linear-algebra primer ({ref}`sec-ch2-linear-algebra-you-need`) fixes notation and derives the softmax Jacobian and vector-Jacobian products, which every later gradient uses.

**Probability and information theory.** A language model assigns a probability to each possible next token, and it is trained to minimize the cross-entropy between the data and those probabilities. This part defines entropy, cross-entropy, KL divergence, perplexity and bits per byte, the quantities in which every loss and evaluation in this book is reported.

**Optimization.** Training changes the parameters step by step against the gradient of the loss. This part derives SGD, momentum and Adam, the learning-rate limit beyond which a step diverges, and the schedules, clipping and accumulation used in every large training run.

**Backpropagation.** The gradient of the loss with respect to billions of parameters is computed in one backward pass. This part works through the backward pass of attention, layer normalization, the feed-forward layer, the embedding and the output head, and shows how residual connections and pre-LN keep gradients from vanishing with depth.

**Scaling laws and compute.** Loss falls as a power law in parameters, data and compute. This part derives the $6N$ FLOPs-per-token rule, uses the Kaplan and Chinchilla laws to split a compute budget between model size and data, and counts the memory a training run needs.

**Numerical stability.** Training runs in 16-bit formats with a limited range and few significant digits. This part shows where values overflow or underflow, and how log-sum-exp, loss scaling, bf16 and initialization scaled to width and depth keep them finite.

Every symbol the book uses is listed on the {ref}`notation page <sec-notation>`, with a link to the section that introduces each symbol used only in this chapter.

(sec-ch2-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will be able to:

- Treat a weight matrix as a map, derive the softmax Jacobian and the $\sqrt{d_k}$ scaling of attention scores, and compute gradients as vector-Jacobian products
- Compute entropy, cross-entropy, KL divergence, perplexity and bits per byte, and explain how temperature and label smoothing change a distribution
- Derive the updates of SGD, momentum and Adam, the stability bound $\eta < 2/\lambda_{\max}$, warm-up and cosine schedules, global-norm clipping and gradient accumulation
- Backpropagate by hand through attention, layer normalization, the feed-forward layer, the embedding and the output head, and explain how residual connections and pre-LN keep gradients alive with depth
- Derive the $6N$ FLOPs-per-token rule, apply the Kaplan and Chinchilla scaling laws to a compute budget, and estimate the compute and memory of a training run
- Choose a floating-point format, keep a softmax and fp16 gradients finite with log-sum-exp and loss scaling, and scale initialization to width and depth

(sec-ch2-hands-on-learning)=

## Hands-On Learning

For interactive implementations and visualizations of the mathematical concepts in this chapter, see the accompanying Jupyter notebook: [Mathematical Foundations Demo](examples/mathematical_foundations_demo.ipynb)

The notebook provides executable, assertion-checked code for:

- Entropy, cross-entropy and KL divergence in bits ({ref}`nb-math-information-theory`)
- SGD, momentum and Adam from scratch (Adam checked against `torch.optim.Adam`) on the Rosenbrock function ({ref}`nb-math-optimizers`)
- Warm-up and decay schedules ({ref}`nb-math-schedules`)
- Attention gradients, layer normalization, residual connections and initialization scale ({ref}`nb-math-backprop`)
- Kaplan and Chinchilla scaling laws, parameter counts and training memory ({ref}`nb-math-scaling`)
- fp16/bf16 precision, loss scaling, gradient clipping and initialization ({ref}`nb-math-numerics`)

(sec-ch2-linear-algebra-you-need)=

## 0. Linear Algebra You Need

Everything a transformer does is a sequence of matrix products followed by simple element-wise functions. Three facts carry most of the weight.

(sec-ch2-matrix-map)=

### 0.1 A matrix is a map between spaces

A weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ turns a row vector $x \in \mathbb{R}^{d_{\text{in}}}$ into $xW \in \mathbb{R}^{d_{\text{out}}}$. Stacking $n$ inputs as the rows of $X \in \mathbb{R}^{n \times d_{\text{in}}}$ applies the same map to all of them at once, $XW$, which is why a batch of tokens costs one matrix product rather than $n$ separate ones. A query-key score matrix $QK^\top \in \mathbb{R}^{n \times n}$ is the table of all pairwise dot products $q_i \cdot k_j$.

(sec-ch2-sqrt-dk)=

#### Why attention scores are divided by the key dimension's square root

Suppose the entries of a query $q$ and a key $k$ are independent, with mean zero and variance one. Each product $q_j k_j$ then has mean zero and variance one, and the dot product $q \cdot k$, a sum of $d_k$ independent such products, has variance $d_k$. Dividing the scores by $\sqrt{d_k}$ brings them back to unit variance. Without it, at $d_k = 64$ the scores would have standard deviation 8; the softmax would put almost all weight on one position, and by the softmax Jacobian below ({ref}`sec-ch2-softmax-jacobian`) almost no gradient would flow through it.

(sec-ch2-softmax-jacobian)=

### 0.2 The softmax Jacobian

For $p = \mathrm{softmax}(z)$, $p_i = e^{z_i}/\sum_j e^{z_j}$,

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \qquad \text{so for an upstream gradient } g: \quad \frac{\partial \mathcal{L}}{\partial z} = p \odot \left(g - \langle g, p\rangle \mathbf{1}\right)$$

This is the identity behind the attention gradient in {ref}`sec-ch2-attention-gradient-analysis` and behind cross-entropy's gradient $p_j - \delta_{j x_i}$ (take $g_j = -\delta_{j x_i}/p_j$ for the target token $x_i$). It also explains saturation: if one $p_i \approx 1$ then every factor $p_i(\delta_{ij} - p_j)$ is near zero, and no gradient flows through that softmax.

{numref}`fig-ch2-softmax-saturation` shows how fast the gradient through one softmax output vanishes as its logit pulls ahead.

```{figure} figures/softmax-saturation.*
:name: fig-ch2-softmax-saturation
:alt: Log-scale plot of the softmax derivative p1 times one minus p1 against the logit gap, for 2 and 64 logits; at gap 8 it is 0.00034 for 2 logits and 0.02 for 64.

The diagonal softmax derivative $\partial p_i/\partial z_i = p_i(1 - p_i)$ when logit
$z_i$ leads all the others by the same gap. It peaks at 0.25 where $p_i = 0.5$ and then
falls by about a factor $e$ per unit of gap. At a gap of 8, the standard deviation of a
single unscaled score at $d_k = 64$ (the gap between two such scores has standard
deviation $8\sqrt{2} = 11.3$, Solution 4 of {ref}`sec-ch2-exercise-4`), it is 0.00034
with 2 logits and 0.02 with 64. Computed from the formula above by
`figures/make_figures.py`.
```

(sec-ch2-norms-conditioning-scale-matters)=

### 0.3 Norms, conditioning and why scale matters

The operator norm $\|W\|_2$ is the largest factor by which $W$ can stretch a vector. A stack of $L$ layers multiplies these factors, which is why gradients vanish or explode geometrically with depth unless each layer is kept near norm 1, by initialization scale ({ref}`sec-ch2-initialization-strategies`), by normalization ({ref}`sec-ch2-layer-normalization-mathematics`) or by residual connections ({ref}`sec-ch2-residual-connections`). The notebook section {ref}`nb-math-layernorm` measures the effect of each on a stack of linear layers.

How unevenly a matrix stretches is measured by its **condition number**, the ratio of its largest to its smallest stretch, $\operatorname{cond} = \sigma_{\max}/\sigma_{\min}$ in terms of singular values ({ref}`sec-ch2-svd-low-rank`). For a symmetric positive definite matrix, such as the Hessian of a loss near a minimum, the singular values are the eigenvalues and $\operatorname{cond} = \lambda_{\max}/\lambda_{\min}$. A large $\operatorname{cond}$ means the loss is steep in some directions and nearly flat in others, and {ref}`sec-ch2-stability-condition` shows that gradient descent then needs about $\operatorname{cond}$ steps to make progress along the flat ones.

(sec-ch2-svd-low-rank)=

### 0.4 Singular value decomposition and low rank

Every real matrix factors as $W = U \Sigma R^\top$, where $U$ and $R$ have orthonormal columns and $\Sigma$ is diagonal, holding the singular values $\Sigma_{jj} \ge 0$ in decreasing order. Read as a map, $W$ rotates or reflects (by $R^\top$), stretches each axis by a singular value, and rotates or reflects again (by $U$); the operator norm of {ref}`sec-ch2-norms-conditioning-scale-matters` is the largest stretch, so $\|W\|_2$ is the largest singular value.

Keeping only the $n_{rank}$ largest singular values, $\sum_{j=1}^{n_{rank}} \Sigma_{jj}\, U_{:,j} R_{:,j}^\top$, gives a matrix of rank $n_{rank}$. {cite:t}`eckart1936approximation` showed that no matrix of that rank is closer to $W$ in the Frobenius norm, and that the error is the root of the sum of the discarded squares, $\sqrt{\sum_{j > n_{rank}} \Sigma_{jj}^2}$; in the operator norm the error is the largest discarded singular value. A rank-$n_{rank}$ matrix stored as two factors costs $n_{rank}(d_{\text{in}} + d_{\text{out}})$ numbers instead of $d_{\text{in}} d_{\text{out}}$: for a $4096 \times 4096$ matrix and $n_{rank} = 8$ that is 65,536 instead of 16,777,216, or 0.39%. LoRA {cite}`hu2022lora` uses exactly this count, training the update to a frozen weight matrix as a product of two such thin factors.

```python
import torch

torch.manual_seed(0)
# a 64 x 64 matrix that is rank 4 plus a little noise
W = torch.randn(64, 4) @ torch.randn(4, 64) + 0.01 * torch.randn(64, 64)
U, S, Rh = torch.linalg.svd(W)
for n_rank in (2, 4, 8):
    W_r = U[:, :n_rank] @ torch.diag(S[:n_rank]) @ Rh[:n_rank]
    err = torch.linalg.norm(W - W_r).item()
    predicted = S[n_rank:].square().sum().sqrt().item()
    print(f"rank {n_rank}: Frobenius error {err:.4g} "
          f"(from the singular values {predicted:.4g})")
print(f"parameters of a rank-8 factorization of 4096 x 4096: "
      f"{8 * (4096 + 4096):,} of {4096 * 4096:,}")
```

% output

```text
rank 2: Frobenius error 78.64 (from the singular values 78.64)
rank 4: Frobenius error 0.5906 (from the singular values 0.5906)
rank 8: Frobenius error 0.5197 (from the singular values 0.5197)
parameters of a rank-8 factorization of 4096 x 4096: 65,536 of 16,777,216
```

(sec-ch2-vjp)=

### 0.5 Vector-Jacobian products

For a layer that maps a row vector $x \in \mathbb{R}^{d_{\text{in}}}$ to a row vector $y \in \mathbb{R}^{d_{\text{out}}}$, the Jacobian $\partial y / \partial x$ is a $d_{\text{out}} \times d_{\text{in}}$ matrix. Backpropagation never builds it. It needs only the vector-Jacobian product (VJP): given the gradient $g = \partial \mathcal{L} / \partial y$ of a scalar loss, the chain rule gives

$$\frac{\partial \mathcal{L}}{\partial x} = g \, \frac{\partial y}{\partial x},$$

and for the operations a transformer uses this product has a closed form far cheaper than the matrix. For a linear map $y = xW$ the VJP is $g W^\top$ (and the weight gradient is $x^\top g$); for an element-wise activation $y = \operatorname{act}(x)$ it is $g \odot \operatorname{act}'(x)$; for the softmax it is the formula of {ref}`sec-ch2-softmax-jacobian`, whose cost grows linearly with the number of logits while the Jacobian has the square of that number of entries. The softmax over a 50,257-token vocabulary has a Jacobian of about 2.5 billion entries, while its VJP touches 50,257 numbers twice.

```python
import torch

torch.manual_seed(0)
z = torch.randn(6, dtype=torch.float64, requires_grad=True)
g = torch.randn(6, dtype=torch.float64)
p = torch.softmax(z, dim=0)
(vjp_autograd,) = torch.autograd.grad(p, z, grad_outputs=g)
jacobian = torch.autograd.functional.jacobian(
    lambda t: torch.softmax(t, dim=0), z.detach()
)
vjp_closed_form = p.detach() * (g - (g * p.detach()).sum())
print("g^T (dp/dz) equals autograd's VJP:", torch.allclose(g @ jacobian, vjp_autograd))
print("closed form p * (g - <g, p>) equals it:",
      torch.allclose(vjp_closed_form, vjp_autograd))
```

% output

```text
g^T (dp/dz) equals autograd's VJP: True
closed form p * (g - <g, p>) equals it: True
```

(sec-ch2-probability-information-theory)=

## 1. Probability and Information Theory

(sec-ch2-foundations-language-modeling)=

### 1.1 Foundations of Language Modeling

(sec-ch2-probability-distributions-over-sequences)=

#### Probability Distributions Over Sequences

Given a vocabulary $\mathcal{V}$ and sequence $x = (x_1, x_2, ..., x_n)$, language models estimate:

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_{<i})$$

This autoregressive factorization is the chain rule of probability, $p(x_1, x_2) = p(x_1)\,p(x_2 \mid x_1)$, applied $n - 1$ times, so it is exact and loses nothing. What it buys is tractability: instead of one distribution over $|\mathcal{V}|^n$ sequences, the model needs only a distribution over $|\mathcal{V}|$ next tokens for each context.

(sec-ch2-cross-entropy-loss)=

#### Cross-Entropy Loss

The standard training objective minimizes cross-entropy between the true data distribution $p_{\text{data}}$ and model distribution $p_{\theta}$:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_{\theta}(x)]$$

Estimated on a training corpus of $T$ tokens, this becomes:

$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{i=1}^{T} \log p_{\theta}(x_i \mid x_{<i})$$

Cross-entropy is convex in the predicted probabilities, because $-\log q$ is a convex function of $q$, and its gradient with respect to the logits, $p_j - \delta_{j x_i}$ ({ref}`sec-ch2-softmax-jacobian`), stays large for as long as the model puts little probability on the correct token.

(sec-ch2-perplexity-bpb)=

#### Perplexity and Bits per Byte

**Perplexity** is the exponential of the mean loss in nats, $\operatorname{PPL} = e^{\mathcal{L}}$. It is the size of a uniform choice that would be as uncertain as the model: a model with $\mathcal{L} = \ln 50$ is, on average, as unsure of the next token as a fair 50-sided die. Perplexity depends on the tokenizer, because the same text cut into fewer, longer tokens has fewer, harder predictions. **Bits per byte** removes that dependence by spreading the total loss over the bytes of the text instead of its tokens. The total loss of $T$ tokens is $T\mathcal{L}$ nats, which is $T\mathcal{L}/\ln 2$ bits, so for a text of $n_{bytes}$ bytes

$$\text{bits per byte} = \frac{T\,\mathcal{L}}{n_{bytes} \ln 2}.$$

The Pile's authors prefer bits per byte to perplexity "due to its invariance to different tokenization schemes" {cite}`gao2020pile`. With $\mathcal{L} = 2.0$ nats per token and 4 bytes per token on average, perplexity is $e^{2} = 7.389$ and the model spends $2.0/(4 \ln 2) = 0.7213$ bits per byte; a tokenizer that averaged 3 bytes per token at the same bits per byte would show a loss of 1.5 nats and a perplexity of 4.482, so perplexities are comparable only across models with the same tokenizer.

(sec-ch2-information-theory-fundamentals)=

### 1.2 Information Theory Fundamentals

(sec-ch2-entropy-language)=

#### Entropy and Language

**Shannon entropy** quantifies the uncertainty in a probability distribution:

$$H(p) = -\sum_{x} p(x) \log p(x)$$

Using $\log_2$ gives entropy in bits; the natural logarithm gives nats. The entropy and KL functions below use $\log_2$ and return bits; the cross-entropy loss uses the natural logarithm, as PyTorch does, so it and the perplexity built on it are in nats.

For language, entropy measures the average information content per symbol. Shannon's experiments bound the entropy of printed English, for a reader who knows the preceding 100 letters, between 0.6 and 1.3 bits per letter, far below the $\log_2 27 = 4.75$ bits of a uniform choice over the 27 symbols of his alphabet, the 26 letters and the space, which is what makes language predictable {cite}`shannon1951prediction`.

(sec-ch2-cross-entropy-kl-divergence)=

#### Cross-Entropy and KL Divergence

**Cross-entropy** between distributions $p$ and $q$ measures the average number of bits needed to encode data from $p$ using code optimized for $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

**Kullback-Leibler (KL) divergence** measures the difference between distributions:

$$D_{KL}(p \,\|\, q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Training by maximum likelihood is the same as minimizing this divergence from the data distribution. The training loss of {ref}`sec-ch2-cross-entropy-loss` is the cross-entropy $H(p_{\text{data}}, p_\theta)$, so

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)] = H(p_{\text{data}}) + D_{KL}(p_{\text{data}} \,\|\, p_\theta).$$

The entropy of the data does not depend on $\theta$, so any change of $\theta$ that lowers the loss lowers the KL divergence by exactly the same amount, and the loss can never fall below $H(p_{\text{data}})$, which is reached only when $p_\theta = p_{\text{data}}$. For a worked case take $p = (0.5, 0.25, 0.25)$ and $q = (0.25, 0.5, 0.25)$. In bits, $H(p) = 0.5 \cdot 1 + 0.25 \cdot 2 + 0.25 \cdot 2 = 1.5$, $H(p, q) = 0.5 \cdot 2 + 0.25 \cdot 1 + 0.25 \cdot 2 = 1.75$, and $D_{KL}(p \,\|\, q) = 0.5 \log_2 2 + 0.25 \log_2 0.5 + 0 = 0.25 = 1.75 - 1.5$.

This pair happens to give the same value both ways, $D_{KL}(q \,\|\, p) = 0.25$ too, but KL divergence is not symmetric in general. For $p = (0.9, 0.1)$ and the uniform $q = (0.5, 0.5)$, $D_{KL}(p \,\|\, q) = 0.9 \log_2 1.8 + 0.1 \log_2 0.2 = 0.531$ bits while $D_{KL}(q \,\|\, p) = 0.5 \log_2 (5/9) + 0.5 \log_2 5 = 0.737$ bits. The direction matters for training: the loss is $D_{KL}(p_{\text{data}} \,\|\, p_\theta)$, which weights each outcome by how often the data produces it, and it is infinite if the model gives probability zero to an outcome that occurs.

{numref}`fig-ch2-cross-entropy-decomposition` draws the worked example, with two more choices of $q$ for comparison.

```{figure} figures/cross-entropy-decomposition.*
:name: fig-ch2-cross-entropy-decomposition
:alt: Stacked horizontal bars: each cross-entropy H(p, q) is the entropy H(p) = 1.5 bits plus a KL segment of 0, 0.085 and 0.25 bits for three choices of q.

Cross-entropy splits into the entropy of $p$, the same for every model, and the KL
divergence, the only part a model can reduce. The distributions are the worked example
above, $p = (0.5, 0.25, 0.25)$, with $q = p$, a uniform $q$ and $q = (0.25, 0.5, 0.25)$;
values in bits computed by `figures/make_figures.py`.
```

(sec-ch2-mutual-information)=

#### Mutual Information

**Mutual information** measures the amount of information shared between variables:

$$\operatorname{MI}(\mathrm{X}; \mathrm{Y}) = \sum_{x, x'} p(x, x') \log \frac{p(x, x')}{p(x)\,p(x')}$$

where $x$ ranges over the values of $\mathrm{X}$ and $x'$ over the values of $\mathrm{Y}$. It is zero exactly when the two variables are independent.

The **conditional entropy** $H(\mathrm{Y} \mid \mathrm{X}) = -\sum_{x, x'} p(x, x') \log p(x' \mid x)$ is the uncertainty left in $\mathrm{Y}$ once $\mathrm{X}$ is known, averaged over $\mathrm{X}$. Writing $p(x, x')/(p(x)\,p(x')) = p(x' \mid x)/p(x')$ in the definition above splits mutual information into

$$\operatorname{MI}(\mathrm{X}; \mathrm{Y}) = H(\mathrm{Y}) - H(\mathrm{Y} \mid \mathrm{X}),$$

the reduction in uncertainty about $\mathrm{Y}$ that knowing $\mathrm{X}$ brings. Take two binary variables that agree 80% of the time, $p(0, 0) = p(1, 1) = 0.4$ and $p(0, 1) = p(1, 0) = 0.1$. Each is uniform, so $H(\mathrm{Y}) = 1$ bit; given $\mathrm{X}$, $\mathrm{Y}$ equals it with probability 0.8, so $H(\mathrm{Y} \mid \mathrm{X}) = -0.8 \log_2 0.8 - 0.2 \log_2 0.2 = 0.7219$ bits, and $\operatorname{MI} = 0.2781$ bits, the same value the double sum gives.

This is the sense in which language modeling exploits mutual information. With $\mathrm{Y}$ the next token and $\mathrm{X}$ its context, the lowest per-token loss any model can reach is the conditional entropy $H(\mathrm{Y} \mid \mathrm{X})$, the data's own uncertainty given the context, and the loss of a real model is the cross-entropy, which exceeds it by a KL divergence ({ref}`sec-ch2-cross-entropy-kl-divergence`). A context that carries more information about the next token, a larger $\operatorname{MI}$, lowers that floor below the $H(\mathrm{Y})$ of a model that ignores the context.

(sec-ch2-temperature)=

#### Temperature

Dividing the logits by a **temperature** $\tau > 0$ before the softmax, $p_i = e^{z_i/\tau} / \sum_j e^{z_j/\tau}$, changes how peaked the distribution is without changing the order of the tokens. As $\tau \to 0$ the distribution approaches a one-hot vector on the largest logit; as $\tau \to \infty$ it approaches uniform. A probability ratio $p_i/p_j = e^{(z_i - z_j)/\tau}$ shows why: the temperature divides every logit gap. {cite:t}`hinton2015distilling` raise the temperature of a teacher's softmax to produce soft targets for distillation; at sampling time a temperature below one makes generation more conservative. The entropy of {ref}`sec-ch2-entropy-language` measures the effect.

```python
import torch

z = torch.tensor([2.0, 1.0, 0.0])
for tau in (0.5, 1.0, 2.0):
    p = torch.softmax(z / tau, dim=0)
    entropy_bits = -(p * torch.log2(p)).sum().item()
    rounded = [round(v, 4) for v in p.tolist()]
    print(f"tau={tau}: p = {rounded}, entropy {entropy_bits:.4g} bits")
```

% output

```text
tau=0.5: p = [0.8668, 0.1173, 0.0159], entropy 0.6363 bits
tau=1.0: p = [0.6652, 0.2447, 0.09], entropy 1.201 bits
tau=2.0: p = [0.5065, 0.3072, 0.1863], entropy 1.472 bits
```

(sec-ch2-label-smoothing)=

#### Label Smoothing

**Label smoothing** {cite}`szegedy2016rethinking` replaces the one-hot target with a mixture of the one-hot vector and the uniform distribution over the vocabulary, $y_j = (1 - \epsilon_{ls})\,\delta_{j x_i} + \epsilon_{ls}/|\mathcal{V}|$. The gradient with respect to the logits becomes $p_j - y_j$, so the loss is minimized not by a one-hot prediction but by $p = y$: the largest logit stops being pushed away from the others once the correct token holds probability $1 - \epsilon_{ls} + \epsilon_{ls}/|\mathcal{V}|$. Szegedy et al. used $\epsilon_{ls} = 0.1$, and so did the original transformer {cite}`vaswani2017attention`. With $|\mathcal{V}| = 4$ and $\epsilon_{ls} = 0.1$ the target is $(0.925, 0.025, 0.025, 0.025)$ for the correct token first, and the smallest loss the model can reach is the entropy of that target, $H(y) = 0.3488$ nats rather than zero. PyTorch implements this as the `label_smoothing` argument of `F.cross_entropy`.

(sec-ch2-practical-applications)=

### 1.3 Practical Applications

```python
import torch
import torch.nn.functional as F


def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Shannon entropy of probability distribution, in bits.
    
    H(p) = -∑ p(x) log₂ p(x)
    
    Args:
        probs: Probability distribution [batch_size, num_classes]
        eps: Small constant for numerical stability
    
    Returns:
        Entropy values [batch_size]
    """
    return -torch.sum(probs * torch.log2(probs + eps), dim=-1)


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss between predictions and targets.
    
    H(p, q) = -∑ p(x) log q(x), in nats
    
    Args:
        logits: Model outputs [batch_size, vocab_size]
        targets: True token indices [batch_size]
    
    Returns:
        Cross-entropy loss (scalar)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(log_probs, targets)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence between distributions.

    D_KL(p || q) = ∑ p(x) log₂(p(x) / q(x)), in bits. Outcomes with p(x) = 0
    contribute 0;
    where q(x) = 0 < p(x) the divergence is infinite, and the function returns inf.

    Args:
        p: True distribution [batch_size, num_classes]
        q: Approximate distribution [batch_size, num_classes]

    Returns:
        KL divergence values [batch_size]
    """
    support = p > 0
    # p / 0 = inf where q has no mass
    ratio = torch.where(support, p / q, torch.ones_like(p))
    terms = torch.where(support, p * torch.log2(ratio), torch.zeros_like(p))
    return torch.sum(terms, dim=-1)


def perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Convert cross-entropy loss to perplexity.
    
    Perplexity = exp(H(p, q)), with H in nats
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        Perplexity value
    """
    return torch.exp(loss)


p = torch.tensor([[0.5, 0.25, 0.25]])
q = torch.tensor([[0.25, 0.5, 0.25]])
print(f"H(p) = {entropy(p).item():.4g} bits, "
      f"D_KL(p || q) = {kl_divergence(p, q).item():.4g} bits")
# a model with uniform output
loss = cross_entropy_loss(torch.zeros(1, 50257), torch.tensor([0]))
print(f"uniform over 50,257 tokens: loss {loss.item():.4g} nats, "
      f"perplexity {perplexity(loss).item():,.0f}")
```

% output

```text
H(p) = 1.5 bits, D_KL(p || q) = 0.25 bits
uniform over 50,257 tokens: loss 10.82 nats, perplexity 50,257
```

```{admonition} Hands-On Practice
:class: tip
The notebook section {ref}`nb-math-information-theory` implements entropy, cross-entropy
and KL divergence in bits, shows how entropy falls as a softmax sharpens, and computes
the perplexity of an untrained model.
```

(sec-ch2-optimization-fundamentals)=

## 2. Optimization Fundamentals

(sec-ch2-gradient-descent-variants)=

### 2.1 Gradient Descent and Variants

(sec-ch2-stochastic-gradient-descent)=

#### Stochastic Gradient Descent

The basic SGD update rule:

$$\theta_{t+1} = \theta_t - \eta\, \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)$$

where $\eta$ is the learning rate and $\mathcal{B}_t$ is a mini-batch drawn at random at step $t$, over which the loss is averaged. Its gradient is an unbiased estimate of the gradient of the loss over the whole training set.

(sec-ch2-minibatch-noise)=

#### Mini-batch Noise

Write $g_i$ for the gradient of the loss of example $i$ and $\hat{g} = \frac{1}{B}\sum_{i \in \mathcal{B}_t} g_i$ for the mini-batch gradient. If the $B$ examples are drawn independently, $\mathbb{E}[\hat{g}]$ is the full gradient, and each coordinate of $\hat{g}$ has variance $\sigma_g^2 / B$, where $\sigma_g$ is the standard deviation of that coordinate of $g_i$ across examples: the variance of a mean of $B$ independent terms is the single-term variance divided by $B$. The noise therefore falls only as $1/\sqrt{B}$: four times the batch buys half the noise. {cite:t}`mccandlish2018empirical` build their gradient noise scale, the batch size beyond which a larger batch stops paying for itself, on this $1/B$ term.

```python
import torch

torch.manual_seed(0)
sigma_g = 2.0
# 20,000 draws of 64 per-example gradients
per_example = 1.0 + sigma_g * torch.randn(20_000, 64)
for B in (1, 4, 16, 64):
    g_hat = per_example[:, :B].mean(dim=1)
    print(f"B={B:2d}: std of g_hat {g_hat.std().item():.3f}, "
          f"predicted sigma_g/sqrt(B) {sigma_g / B ** 0.5:.3f}")
```

% output

```text
B= 1: std of g_hat 1.996, predicted sigma_g/sqrt(B) 2.000
B= 4: std of g_hat 1.000, predicted sigma_g/sqrt(B) 1.000
B=16: std of g_hat 0.498, predicted sigma_g/sqrt(B) 0.500
B=64: std of g_hat 0.249, predicted sigma_g/sqrt(B) 0.250
```

(sec-ch2-stability-condition)=

#### The Stability Condition

Why can the learning rate not simply be made large? Near a minimum the loss is close to a quadratic, and along the direction of largest curvature it behaves like $\frac{1}{2}\lambda_{\max}\theta^2$, whose gradient is $\lambda_{\max}\theta$. One gradient-descent step gives $\theta_{t+1} = (1 - \eta\lambda_{\max})\,\theta_t$, so the iterates shrink exactly when $|1 - \eta\lambda_{\max}| < 1$, that is when

$$0 < \eta < \frac{2}{\lambda_{\max}}.$$

At $\eta = 1/\lambda_{\max}$ one step lands on the minimum; between $1/\lambda_{\max}$ and $2/\lambda_{\max}$ the iterates overshoot and oscillate but still shrink; beyond $2/\lambda_{\max}$ each step multiplies the error by more than one in magnitude and training diverges. For $\lambda_{\max} = 100$ the limit is $\eta < 0.02$.

Directions of smaller curvature converge more slowly at the same $\eta$. Along the flattest direction, with curvature $\lambda_{\min}$, the same step multiplies the error by $1 - \eta\lambda_{\min}$, which is close to 1 when $\eta$ must stay below $2/\lambda_{\max}$. The best single $\eta$ balances the two extreme directions, $1 - \eta\lambda_{\min} = -(1 - \eta\lambda_{\max})$, so $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, and both then shrink by

$$\frac{\lambda_{\max} - \lambda_{\min}}{\lambda_{\max} + \lambda_{\min}} = \frac{\operatorname{cond} - 1}{\operatorname{cond} + 1}$$

per step, where $\operatorname{cond} = \lambda_{\max}/\lambda_{\min}$ is the condition number of the Hessian ({ref}`sec-ch2-norms-conditioning-scale-matters`). For $\operatorname{cond} = 100$ the factor is $99/101 = 0.9802$, and reducing the error tenfold takes $\ln 10/\ln(101/99) = 115$ steps; for $\operatorname{cond} = 2$ it takes 2.1 steps. The Rosenbrock function of {numref}`fig-ch2-optimizer-trajectories` has the Hessian $\begin{pmatrix} 802 & -400 \\ -400 & 200 \end{pmatrix}$ at its minimum $(1, 1)$, with eigenvalues 1001.6 and 0.399 and $\operatorname{cond} = 2508$: at the best fixed step the error shrinks by a factor of 0.9992 per step, about 2,900 steps per factor of ten, which is why plain SGD creeps along the valley floor. Momentum builds up speed along the direction that stays downhill from step to step, and Adam rescales each coordinate's step by its own gradient history, so neither is held to this rate.

(sec-ch2-momentum)=

#### Momentum

Momentum accelerates optimization by accumulating gradients:

$$u_{t+1} = \beta_{\text{mom}} u_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta u_{t+1}$$

The momentum coefficient $\beta_{\text{mom}}$ (typically 0.9) controls how much previous gradients influence the current update {cite}`polyak1964some`. On a quadratic whose Hessian has largest eigenvalue $\lambda_{\max}$, plain gradient descent converges for $\eta < 2/\lambda_{\max}$ and momentum for $\eta < 2(1+\beta_{\text{mom}})/\lambda_{\max}$ {cite}`goh2017momentum`; at $\beta_{\text{mom}} = 0.9$ the second limit is 1.9 times the first.

The momentum bound follows the same way. On $\frac{1}{2}\lambda_{\max}\theta^2$ the two update lines combine, using $\eta u_t = \theta_{t-1} - \theta_t$, into one recurrence,

$$\theta_{t+1} = (1 + \beta_{\text{mom}} - \eta\lambda_{\max})\,\theta_t - \beta_{\text{mom}}\,\theta_{t-1}.$$

As $\eta$ grows, the first instability is a solution that flips sign every step, $\theta_t \propto (-1)^t$. Substituting it gives $1 + (1 + \beta_{\text{mom}} - \eta\lambda_{\max}) + \beta_{\text{mom}} = 0$, so the boundary is $\eta\lambda_{\max} = 2(1 + \beta_{\text{mom}})$ {cite}`goh2017momentum`. With $\lambda_{\max} = 100$ and $\beta_{\text{mom}} = 0.9$ the limits are 0.02 without momentum and 0.038 with it, and the run below lands on either side of each.

```python
lam_max = 100.0


def final_size(eta, beta_mom=0.0, steps=200):
    theta, u = 1.0, 0.0
    for _ in range(steps):
        u = beta_mom * u + lam_max * theta
        theta = theta - eta * u
    return abs(theta)


for eta, beta_mom in ((0.019, 0.0), (0.021, 0.0), (0.037, 0.9), (0.039, 0.9)):
    size = final_size(eta, beta_mom)
    print(f"eta={eta}, beta_mom={beta_mom}: |theta| after 200 steps = {size:.3g}")
```

% output

```text
eta=0.019, beta_mom=0.0: |theta| after 200 steps = 7.06e-10
eta=0.021, beta_mom=0.0: |theta| after 200 steps = 1.9e+08
eta=0.037, beta_mom=0.9: |theta| after 200 steps = 0.000161
eta=0.039, beta_mom=0.9: |theta| after 200 steps = 2.58e+24
```

{numref}`fig-ch2-optimizer-trajectories` shows the three update rules on the Rosenbrock function, with momentum's learning rate inside the bound above along its whole path.

```{figure} figures/optimizer-trajectories.*
:name: fig-ch2-optimizer-trajectories
:alt: Left: paths of SGD (dashed), SGD with momentum and Adam on Rosenbrock contours from (-1, -1) to the minimum at (1, 1); momentum runs past the minimum along the valley to about (1.96, 3.69). Right: loss against iteration on a log scale.

SGD ($\eta = 10^{-3}$), SGD with momentum ($\eta = 5 \times 10^{-4}$,
$\beta_{\text{mom}} = 0.9$) and Adam ($\eta = 10^{-2}$) on the Rosenbrock function,
10,000 steps from $(-1, -1)$. Momentum runs along the valley past the minimum to about
(1.96, 3.69), just below the valley floor $\theta_y = \theta_x^2$, then turns back and
settles at (1, 1); its loss levels off near $2 \times 10^{-11}$ because the run is in
float32, whose spacing near 1 is about $10^{-7}$, and the same run in float64 reaches
$10^{-16}$; plain SGD (dashed, drawn on top) creeps along the valley floor and still has
a loss of about $3 \times 10^{-5}$ at the end. From the notebook section
{ref}`nb-math-rosenbrock`, by executing its cell tagged `figure:optimizer-trajectories`.
```

(sec-ch2-adam-optimizer)=

#### Adam Optimizer

Adam {cite}`kingma2014adam` adapts learning rates per parameter using first and second moment estimates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The bias correction undoes the zero start of the two averages. Unrolling the first recurrence from its zero start gives $m_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i$. If the gradients have a constant mean, $\mathbb{E}[g_i] = \mathbb{E}[g]$, then

$$\mathbb{E}[m_t] = (1-\beta_1)\,\mathbb{E}[g]\sum_{i=1}^{t}\beta_1^{t-i} = (1-\beta_1^t)\,\mathbb{E}[g],$$

because the geometric sum is $(1-\beta_1^t)/(1-\beta_1)$. Dividing by $1-\beta_1^t$ makes $\hat{m}_t$ an unbiased estimate of the mean gradient, and the same argument applies to $v_t$ and $\mathbb{E}[g^2]$ {cite}`kingma2014adam`. The correction matters most early: at the first step, $t = 1$, with $\beta_1 = 0.9$ and $\beta_2 = 0.999$, $m_t = 0.1\,g_t$ and $v_t = 0.001\,g_t^2$, so the uncorrected ratio $m_t/\sqrt{v_t}$ has magnitude 3.162, while the corrected $\hat{m}_t/\sqrt{\hat{v}_t}$ is exactly one in magnitude.

AdamW {cite}`loshchilov2017decoupled`, the variant implemented below, decouples weight decay $\lambda$ from this update. Its step, as PyTorch's `AdamW` and the class below take it, is

$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta_t\right).$$

The alternative, L2 regularization, adds $\lambda\theta_t$ to the gradient $g_t$ before the moments are formed, and so the decay is divided by $\sqrt{\hat{v}_t}$ like everything else. A weight whose gradients have been large is then decayed less than one whose gradients have been small. Take two weights both at $\theta = 1$ with $\lambda = 0.01$, one with $\sqrt{\hat{v}_t} = 0.001$ and one with $\sqrt{\hat{v}_t} = 1$: once the moment has absorbed the decay term, L2 decays them by about $\eta \cdot 10$ and $\eta \cdot 0.01$ per step, a thousandfold difference, while AdamW decays both by $\eta \cdot 0.01$. {cite:t}`loshchilov2017decoupled` show that this decoupling is what makes weight decay effective with Adam.

{cite:t}`kingma2014adam` designed Adam for "problems with very noisy and/or sparse gradients". A language model's embedding table is one source of sparse gradients: the row of a token that does not occur in a batch receives a zero gradient from that batch, so a rare token's row is updated only now and then, and Adam's per-parameter scaling gives those infrequent updates a usable size. They can also be larger than $\eta$: the first gradient after a long absence moves the row by about $3.15\,\eta$, and if the token keeps occurring the step grows to about $6.5\,\eta$ before it settles, below a general limit of $7.27\,\eta$ for these $\beta$s ({ref}`sec-ch2-epsilon-placement`).

(sec-ch2-learning-rate-scheduling)=

### 2.2 Learning Rate Scheduling

(sec-ch2-linear-warmup)=

#### Linear Warmup

Many LLM training schedules use linear warmup followed by decay:

$$\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t+1}{t_{\text{warm}}} & \text{if } t < t_{\text{warm}} \\
\eta_{\max} \cdot \text{decay}(t) & \text{otherwise}
\end{cases}$$

Warm-up keeps the first updates small while the weights are random and Adam's moment estimates rest on only a few gradients. For the original post-LN transformer, {cite:t}`xiong2020layer` show that at initialization the expected gradients of the parameters near the output layer are large, so a full learning rate from the first step makes training unstable. With $\eta_{\max} = 3 \times 10^{-4}$ and $t_{\text{warm}} = 2000$, the first step uses $1.5 \times 10^{-7}$ and the rate reaches $\eta_{\max}$ at step 1999.

(sec-ch2-cosine-annealing)=

#### Cosine Annealing

Cosine schedules {cite}`loshchilov2017sgdr` provide smooth learning rate decay:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warm}}}{t_{\text{end}} - t_{\text{warm}}} \pi\right)\right)$$

(sec-ch2-inverse-sqrt)=

#### Inverse Square Root

The original transformer {cite}`vaswani2017attention` warms up linearly and then decays in proportion to the inverse square root of the step, with no end point:

$$\eta_t = d^{-0.5} \cdot \min\left(t^{-0.5},\; t \cdot t_{\text{warm}}^{-1.5}\right),$$

counting steps from $t = 1$. The two arguments of the minimum are equal at $t = t_{\text{warm}}$, so the peak is $\eta_{\max} = (d\,t_{\text{warm}})^{-0.5}$, and after it the rate is $\eta_{\max}\sqrt{t_{\text{warm}}/t}$: four times as many steps, half the rate. With the paper's $d = 512$ and $t_{\text{warm}} = 4000$ the peak is $6.988 \times 10^{-4}$. Unlike cosine decay, the schedule does not need the length of training in advance, which suits runs that may be extended.

```python
def inverse_sqrt_lr(t, d=512, t_warm=4000):
    return d ** -0.5 * min(t ** -0.5, t * t_warm ** -1.5)


for t in (1, 1000, 4000, 16000, 100000):
    print(f"step {t:6d}: lr {inverse_sqrt_lr(t):.4g}")
```

% output

```text
step      1: lr 1.747e-07
step   1000: lr 0.0001747
step   4000: lr 0.0006988
step  16000: lr 0.0003494
step 100000: lr 0.0001398
```

{numref}`fig-ch2-lr-schedules` compares the schedules on one set of settings.

```{figure} figures/lr-schedules.*
:name: fig-ch2-lr-schedules
:alt: Log-scale learning rate against step for linear warm-up with cosine decay, inverse square root decay and constant after warm-up; all three rise linearly for 1,000 steps to 0.001.

The three schedules with the notebook's settings: peak $\eta_{\max} = 10^{-3}$ after
1,000 warm-up steps, cosine decay to $\eta_{\min} = 10^{-5}$ at step 10,000. On the
logarithmic axis the linear warm-up looks steep at the start and flat near its end. From
the notebook section {ref}`nb-math-schedules`, by executing its cell tagged
`figure:lr-schedules`.
```

(sec-ch2-gradient-clipping)=

### 2.3 Gradient Clipping

Large language models are susceptible to exploding gradients. **Gradient clipping** prevents this by rescaling gradients:

$$g_{\text{clipped}} = \min\left(1, \frac{g_{\max}}{\|g\|}\right) g$$

where $g_{\max}$ is the clipping threshold (typically 1.0) {cite}`pascanu2013difficulty`.

```python
import math

import torch


class AdamWWithWarmup:
    """Adam with decoupled weight decay (AdamW), linear warm-up, cosine decay to
    lr_min, and global-norm gradient clipping. Written out in full to mirror the
    equations above; in practice use torch.optim.AdamW with a scheduler."""

    def __init__(self, params, lr=1e-3, lr_min=0.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, warmup_steps=1000, max_steps=10000,
                 max_grad_norm=1.0):
        self.params = list(params)
        self.lr, self.lr_min = lr, lr_min
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps, self.max_steps = warmup_steps, max_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def get_lr(self, step=None):
        step = self.step_count if step is None else step
        if step < self.warmup_steps:
            # (step + 1) so that the very first update is not a no-op
            return self.lr * (step + 1) / self.warmup_steps
        decay_steps = self.max_steps - self.warmup_steps
        progress = min((step - self.warmup_steps) / decay_steps, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr - self.lr_min) * cosine

    @torch.no_grad()
    def step(self):
        lr = self.get_lr()
        self.step_count += 1
        t = self.step_count

        # Global-norm clipping across ALL parameters, as in the equation above
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            # Decoupled weight decay: shrink the weights directly, not through
            # the gradient
            p.mul_(1 - lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-lr)


# The schedule of the warm-up example: peak 3e-4, 2,000 warm-up steps,
# 100,000 steps in all
opt = AdamWWithWarmup([torch.zeros(1, requires_grad=True)],
                      lr=3e-4, warmup_steps=2000, max_steps=100_000)
for step in (0, 1999, 51_000, 100_000):
    print(f"step {step:6d}: lr {opt.get_lr(step):.4g}")
```

% output

```text
step      0: lr 1.5e-07
step   1999: lr 0.0003
step  51000: lr 0.00015
step 100000: lr 0
```

```{admonition} Interactive Exploration
:class: note
The notebook section {ref}`nb-math-optimizers` implements SGD, momentum and Adam from
scratch (Adam is asserted to match `torch.optim.Adam`), compares them on the Rosenbrock
function ({ref}`nb-math-rosenbrock`), and plots warm-up with cosine, inverse-square-root
and constant schedules ({ref}`nb-math-schedules`).
```

(sec-ch2-gradient-accumulation)=

### 2.4 Gradient Accumulation

When memory limits the batch size, **gradient accumulation** runs $n_{acc}$ micro-batches of $B$ examples through forward and backward passes before one optimizer step, dividing each micro-batch loss by $n_{acc}$. Gradients add, so the accumulated gradient is the mean over all $n_{acc} B$ examples, the same $\hat{g}$ as one batch of that size ({ref}`sec-ch2-minibatch-noise`), with the noise of the larger batch and the memory of the smaller one. The equality is exact for a loss that is a mean over examples and a model with no batch-dependent layers; batch normalization, which transformers do not use, would break it. Four micro-batches of 8 sequences of 2048 tokens give an update over 65,536 tokens. The demonstration at the end of the listing checks the equality on a one-parameter model whose gradient, 0.625, is below the clipping threshold of 1.0, so clipping cannot hide a difference; it also runs the common bug, summing the micro-batch losses without dividing by $n_{acc}$, which gives a gradient $n_{acc} = 4$ times too large and a different weight. When the number of micro-batches is not a multiple of $n_{acc}$, the last group is smaller, and dividing its losses by $n_{acc}$ would shrink that update by the same proportion; the function divides each loss by the size of its own group, and the last part of the demonstration checks that five micro-batches with $n_{acc} = 4$ give the same weight as a batch of eight followed by a batch of two.

```python
import torch


def train_with_gradient_accumulation(model, data_loader, optimizer,
                                     accumulation_steps=4):
    """Average gradients over `accumulation_steps` micro-batches before each update."""
    model.train()
    optimizer.zero_grad()

    running_loss, num_batches, n = 0.0, 0, len(data_loader)
    for i, batch in enumerate(data_loader, start=1):
        # a Hugging Face causal LM returns .loss only when it is given labels
        outputs = model(batch["input_ids"], labels=batch["input_ids"])
        group_start = (i - 1) // accumulation_steps * accumulation_steps
        # smaller for a final partial group
        group_size = min(accumulation_steps, n - group_start)
        # scale so the accumulated gradient is the group's mean
        loss = outputs.loss / group_size
        loss.backward()
        running_loss += outputs.loss.item()
        num_batches += 1

        if i % accumulation_steps == 0 or i == n:  # do not drop a final partial group
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return running_loss / max(num_batches, 1)  # mean per-batch loss, unscaled


# Demonstration: four micro-batches of two examples against one batch of eight
from types import SimpleNamespace


class OneWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.4]))

    def forward(self, input_ids, labels):
        x, y = input_ids.float(), 0.3 * labels.float()
        return SimpleNamespace(loss=((self.w * x - y) ** 2).mean())


ids = torch.tensor([1, 2, 1, 3, 2, 1, 2, 1])
micro_batches = [{"input_ids": ids[i:i + 2]} for i in range(0, 8, 2)]

# the full-batch gradient, below max_norm = 1.0 so clipping leaves it alone
probe = OneWeight()
probe(ids, ids).loss.backward()
print(f"full-batch gradient:                 {probe.w.grad.item():.4f}")

accumulated, full = OneWeight(), OneWeight()
train_with_gradient_accumulation(
    accumulated, micro_batches, torch.optim.SGD(accumulated.parameters(), lr=0.1)
)
train_with_gradient_accumulation(
    full, [{"input_ids": ids}], torch.optim.SGD(full.parameters(), lr=0.1),
    accumulation_steps=1,
)
print(f"w after 4 accumulated micro-batches: {accumulated.w.item():.4f}")
print(f"w after one batch of 8:              {full.w.item():.4f}")

# the bug the division prevents: micro-batch losses summed, not averaged
wrong = OneWeight()
for batch in micro_batches:
    wrong(batch["input_ids"], batch["input_ids"]).loss.backward()
print(f"gradient without the division:       {wrong.w.grad.item():.4f}")
torch.nn.utils.clip_grad_norm_(wrong.parameters(), max_norm=1.0)
torch.optim.SGD(wrong.parameters(), lr=0.1).step()
print(f"w after the same step without it:    {wrong.w.item():.4f}")
assert torch.allclose(accumulated.w, full.w) and not torch.allclose(wrong.w, full.w)

# A fifth micro-batch makes a final group of one: it must count as a whole batch,
# not a quarter of one
extra = {"input_ids": torch.tensor([3, 2])}
accumulated, two_batches = OneWeight(), OneWeight()
train_with_gradient_accumulation(
    accumulated, micro_batches + [extra],
    torch.optim.SGD(accumulated.parameters(), lr=0.1),
)
train_with_gradient_accumulation(
    two_batches, [{"input_ids": ids}, extra],
    torch.optim.SGD(two_batches.parameters(), lr=0.1),
    accumulation_steps=1,
)
print(f"w after 4 + 1 accumulated micro-batches: {accumulated.w.item():.4f}")
print(f"w after a batch of 8, then one of 2:     {two_batches.w.item():.4f}")
assert torch.allclose(accumulated.w, two_batches.w)
```

% output

```text
full-batch gradient:                 0.6250
w after 4 accumulated micro-batches: 0.3375
w after one batch of 8:              0.3375
gradient without the division:       2.5000
w after the same step without it:    0.3000
w after 4 + 1 accumulated micro-batches: 0.2887
w after a batch of 8, then one of 2:     0.2887
```

(sec-ch2-backpropagation-through-transformers)=

## 3. Backpropagation Through Transformers

(sec-ch2-gradient-flow-deep-networks)=

### 3.1 Gradient Flow in Deep Networks

(sec-ch2-reverse-mode)=

#### Reverse-Mode Automatic Differentiation

Backpropagation is reverse-mode automatic differentiation {cite}`baydin2018automatic`. The forward pass evaluates the network operation by operation and stores every intermediate value it will need later. The backward pass starts from $\partial \mathcal{L} / \partial \mathcal{L} = 1$ and visits the operations in reverse order; each one turns the gradient of its output into gradients of its inputs with its vector-Jacobian product ({ref}`sec-ch2-vjp`), and an input used by several operations adds up the gradients it receives. One backward pass yields the gradient with respect to every parameter, and, in Baydin et al.'s words, "the amount of arithmetic goes up by no more than a small constant factor" over the forward pass. Forward-mode differentiation would instead need one pass per input, which for a model with billions of parameters is out of the question. The price of reverse mode is memory: the stored intermediates are the activations counted in {ref}`sec-ch2-memory-requirements`.

The listing below differentiates $\mathcal{L} = (\tanh(\theta x) - 0.5)^2$ with respect to a single parameter $\theta$ at $x = 2$, $\theta = 0.3$ by hand, one stored intermediate at a time, and compares the result with PyTorch.

```python
import math

import torch

x, theta_value, target = 2.0, 0.3, 0.5
# forward pass: store every intermediate
a = theta_value * x            # 0.6
h = math.tanh(a)               # 0.5370
loss = (h - target) ** 2
# backward pass: one VJP per operation, in reverse order
dL_dh = 2 * (h - target)
dL_da = dL_dh * (1 - h ** 2)   # tanh'(a) = 1 - tanh(a)^2
dL_dtheta = dL_da * x
theta = torch.tensor(theta_value, requires_grad=True)
((torch.tanh(theta * x) - target) ** 2).backward()
print(f"loss {loss:.4g}; dL/dtheta by hand {dL_dtheta:.4g}, "
      f"by autograd {theta.grad.item():.4g}")
```

% output

```text
loss 0.001373; dL/dtheta by hand 0.1055, by autograd 0.1055
```

(sec-ch2-vanishing-gradient-problem)=

#### The Vanishing Gradient Problem

In deep networks, gradients can become exponentially small:

$$\frac{\partial \mathcal{L}}{\partial \theta^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(2)}}{\partial h^{(1)}} \frac{\partial h^{(1)}}{\partial \theta^{(1)}}$$

The factors are applied in that order, the last layer's Jacobian first. If every factor's norm is at most some $c < 1$, the norm of the product is at most $c^{L-1}$, so the gradient reaching the first layer shrinks exponentially with depth. A norm below 1 in every layer is not enough on its own: norms of $1 - 1/(l+1)^2$, for example, keep the product of the norms above one half at any depth, so this bound no longer forces the gradient to vanish.

(sec-ch2-residual-connections)=

#### Residual Connections

Residual connections provide direct gradient pathways:

$$h^{(l+1)} = h^{(l)} + F(h^{(l)})$$

The gradient becomes:

$$\frac{\partial h^{(l+1)}}{\partial h^{(l)}} = I + \frac{\partial F(h^{(l)})}{\partial h^{(l)}}$$

The identity term gives every layer a direct path for the gradient, so a deep stack no longer multiplies a long chain of possibly small Jacobians {cite}`he2016deep`. It does not by itself bound the gradient's magnitude: where normalization sits relative to the residual branch matters, and the original post-LN transformer needs learning-rate warm-up precisely because its gradients at initialization are not well behaved, which is why most later models normalize before each sublayer (pre-LN) {cite}`xiong2020layer`.

{numref}`fig-ch2-gradient-flow-depth` measures the effect: the same stack with and without the fixes.

```{figure} figures/gradient-flow-depth.*
:name: fig-ch2-gradient-flow-depth
:alt: Gradient norm against depth through 12 linear layers on a log scale: without LayerNorm at std 0.02 it falls from about 100 at the output to about 0.000002 at the input; the Xavier and LayerNorm variants stay near 100, and the residual variant rises to about 330 at the input.

Gradient norm at each depth of a stack of 12 linear layers ($d = 128$). With weights of
standard deviation 0.02 and nothing else, each layer shrinks the gradient and it falls
by nearly eight orders of magnitude; Xavier scaling or LayerNorm keeps it near its
output value, and with residual connections (plus LayerNorm) it grows by less than a
factor of four toward the input instead of shrinking; the Xavier and LayerNorm lines
coincide and are told apart by their markers. From the notebook section
{ref}`nb-math-layernorm`, by executing its cell tagged `figure:gradient-flow-depth`.
```

(sec-ch2-pre-post-ln)=

#### Pre-LN versus Post-LN

The original transformer normalizes after adding the branch (post-LN); most later models normalize the branch's input instead (pre-LN):

$$\text{post-LN: } h^{(l+1)} = \operatorname{LN}\left(h^{(l)} + F(h^{(l)})\right), \qquad \text{pre-LN: } h^{(l+1)} = h^{(l)} + F\left(\operatorname{LN}(h^{(l)})\right).$$

In pre-LN the residual stream is never normalized on its way through the stack, so $\partial h^{(l+1)} / \partial h^{(l)} = I + \partial F(\operatorname{LN}(h^{(l)}))/\partial h^{(l)}$ keeps the identity term of {ref}`sec-ch2-residual-connections`, and the gradient reaching layer $l$ contains the output gradient unchanged. In post-LN every block's output passes through a LayerNorm Jacobian ({ref}`sec-ch2-layer-normalization-mathematics`), which divides by the standard deviation of $h^{(l)} + F(h^{(l)})$, so there is no path on which the gradient passes untouched. {cite:t}`xiong2020layer` make the difference quantitative at initialization: for post-LN "the scale of the gradients to the last FFN layer is of order $\mathcal{O}(d\sqrt{\ln d})$ which is independent of $L$", while for pre-LN it is $\mathcal{O}(d\sqrt{\ln d / L})$, because the residual stream entering the final LayerNorm grows with depth and the normalization divides the gradients of all parameters by about $\sqrt{L}$. For a 24-layer model that factor is $\sqrt{24} = 4.899$: the post-LN gradients near the output are about five times larger relative to pre-LN, which is why post-LN needs warm-up ({ref}`sec-ch2-linear-warmup`) and pre-LN can often do without it.

{numref}`fig-ch2-pre-post-ln` puts the two arrangements side by side.

```{figure} figures/pre-post-ln.*
:name: fig-ch2-pre-post-ln
:alt: Two block diagrams. Pre-LN: h(l) goes through LN and F to an addition, and a green identity path goes straight to the addition. Post-LN: h(l) and F(h(l)) are added and the sum, including the red residual path, passes through LN.

One block in each arrangement, drawn forward; the gradient runs back along the same
edges. In pre-LN (top) the identity path reaches the addition untouched, so its Jacobian
is $I$; in post-LN (bottom) the residual path also passes through $\operatorname{LN}$
and its Jacobian. Schematic of the two equations above.
```

(sec-ch2-layer-normalization-mathematics)=

### 3.2 Layer Normalization Mathematics

Layer normalization is defined in {ref}`sec-ch1-layernorm`: $y = \gamma \odot \frac{x - \mu}{\sigma} + \beta$, with $\mu$ the mean of the $d$ features of the input row vector $x$ and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}$. In this section and in {ref}`sec-ch2-epsilon-placement`, $x_i$ is feature $i$ of that vector, not the token at position $i$ as elsewhere in the book. What matters here is its gradient with respect to the inputs:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(\tilde{g}_i - \frac{1}{d}\sum_j \tilde{g}_j - \hat{x}_i \frac{1}{d}\sum_j \tilde{g}_j \hat{x}_j\right)$$

where $\tilde{g}_j = \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}$ carries the per-feature scale and $\hat{x}_j = (x_j-\mu)/\sigma$ is the normalized input.

The formula follows from the Jacobian of $\hat{x}$. Since $\mu = \frac{1}{d}\sum_i x_i$, $\partial \mu / \partial x_j = 1/d$. Since $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2 + \epsilon$, and the terms that come from differentiating $\mu$ sum to zero because $\sum_i (x_i - \mu) = 0$, $\partial \sigma^2/\partial x_j = \frac{2}{d}(x_j - \mu)$ and so $\partial \sigma / \partial x_j = (x_j - \mu)/(d\sigma) = \hat{x}_j / d$. The quotient rule on $\hat{x}_i = (x_i - \mu)/\sigma$ then gives

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{\delta_{ij} - 1/d}{\sigma} - \frac{x_i - \mu}{\sigma^2}\cdot\frac{\hat{x}_j}{d} = \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i \hat{x}_j}{d}\right).$$

Contracting with $\tilde{g}$, $\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \tilde{g}_i \frac{\partial \hat{x}_i}{\partial x_j}$, gives the gradient above term by term. Each row of the Jacobian sums to zero, since $\sum_j (\delta_{ij} - 1/d) = 0$ and $\sum_j \hat{x}_j = 0$: adding the same constant to every feature does not change the output, so it receives no gradient.

Most recent models replace LayerNorm with RMSNorm ({ref}`sec-ch1-modern-block`), which drops the mean subtraction and the bias and divides by the root mean square $\operatorname{rms}(x) = \sqrt{\frac{1}{d}\sum_j x_j^2 + \epsilon}$. With $\hat{x}_i = x_i / \operatorname{rms}(x)$, the same quotient rule without the $\partial \mu / \partial x_j$ term gives $\partial \hat{x}_i / \partial x_j = \left(\delta_{ij} - \hat{x}_i \hat{x}_j / d\right) / \operatorname{rms}(x)$: the $-1/d$ term is gone, so the rows no longer sum to zero and a constant added to every feature does receive a gradient. The test suite checks this Jacobian against autograd as well.

The listing checks the LayerNorm formula against PyTorch.

```python
import torch

torch.manual_seed(0)
d, eps = 5, 1e-5
x = torch.randn(d, dtype=torch.float64)


def normalize(t):
    mu = t.mean()
    return (t - mu) / torch.sqrt(((t - mu) ** 2).mean() + eps)


sigma = torch.sqrt(x.var(unbiased=False) + eps)
x_hat = normalize(x)
identity = torch.eye(d, dtype=torch.float64)
formula = (identity - 1 / d - torch.outer(x_hat, x_hat) / d) / sigma
autograd = torch.autograd.functional.jacobian(normalize, x)
print("Jacobian formula matches autograd:", torch.allclose(formula, autograd))
row_sums = formula.sum(dim=1)
print("every row of the Jacobian sums to zero:",
      torch.allclose(row_sums, torch.zeros(d, dtype=torch.float64)))
```

% output

```text
Jacobian formula matches autograd: True
every row of the Jacobian sums to zero: True
```

{cite:t}`ba2016layer` introduced layer normalization because batch normalization depends "on the mini-batch size and it is not obvious how to apply it to recurrent neural networks". Layer normalization computes its statistics within each token's feature vector, so it behaves the same at any batch size and at every position of a sequence.

(sec-ch2-attention-gradient-analysis)=

### 3.3 Attention Gradient Analysis

For scaled dot-product attention write the scores, weights and output as

$$S = QK^\top/\sqrt{d_k}, \qquad A = \mathrm{softmax}(S), \qquad O = AV$$

Given the upstream gradient $\frac{\partial \mathcal{L}}{\partial O}$, let $G = \frac{\partial \mathcal{L}}{\partial O} V^\top$ be the gradient with respect to the weights $A$. Backpropagating through the row-wise softmax gives

$$\frac{\partial \mathcal{L}}{\partial S} = A \odot \left(G - (G \odot A)\mathbf{1}\mathbf{1}^\top\right)$$

where $(G \odot A)\mathbf{1}\mathbf{1}^\top$ broadcasts each row's sum of $G \odot A$ across that row. The gradients with respect to the three inputs are then

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S} K, \qquad \frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}}\left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top Q, \qquad \frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}$$

The $A \odot (\cdot)$ factor is why saturated attention (one weight near 1) passes almost no gradient to the other positions, and why the scores are divided by $\sqrt{d_k}$ ({ref}`sec-ch2-sqrt-dk`).

{numref}`fig-ch2-attention-graph` lays the forward values and the backward products out as one graph.

```{figure} figures/attention-graph.*
:name: fig-ch2-attention-graph
:alt: Computation graph: Q and K feed S, S feeds A, A and V feed O. Below each forward node, dashed red arrows run from dL/dO back to G and dL/dV, from G to dL/dS, and from dL/dS to dL/dQ and dL/dK.

Scaled dot-product attention as a computation graph. The top nodes are the forward
values with their shapes; below each is the vector-Jacobian product the backward pass
computes there, and the dashed arrows run the way the gradient flows. Schematic of the
equations above.
```

```python
import math

import torch
import torch.nn.functional as F


def attention_gradients(Q, K, V, grad_output):
    """
    Compute gradients for scaled dot-product attention.

    Args:
        Q, K, V: Query, Key, Value matrices
        grad_output: Gradient w.r.t. attention output
    """
    d_k = Q.size(-1)

    # Forward pass
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)

    # Backward pass
    grad_V = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
    grad_attn_weights = torch.matmul(grad_output, V.transpose(-2, -1))

    # Softmax gradient
    grad_scores = attn_weights * (
        grad_attn_weights
        - torch.sum(grad_attn_weights * attn_weights, dim=-1, keepdim=True)
    )

    grad_Q = torch.matmul(grad_scores, K) / math.sqrt(d_k)
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q) / math.sqrt(d_k)

    return grad_Q, grad_K, grad_V


torch.manual_seed(0)
Q, K, V = (torch.randn(5, 4, dtype=torch.float64, requires_grad=True) for _ in range(3))
grad_output = torch.randn(5, 4, dtype=torch.float64)
O = F.softmax(Q @ K.T / math.sqrt(4), dim=-1) @ V
reference = torch.autograd.grad(O, (Q, K, V), grad_outputs=grad_output)
ours = attention_gradients(Q.detach(), K.detach(), V.detach(), grad_output)
print("dQ, dK, dV match autograd:",
      all(torch.allclose(a, b) for a, b in zip(ours, reference)))
```

% output

```text
dQ, dK, dV match autograd: True
```

(sec-ch2-ffn-embedding-head-gradients)=

### 3.4 Gradients of the Feed-Forward Layer, Embedding and Output Head

The remaining pieces of a transformer are a feed-forward layer, the embedding lookup and the output head, and each has a short backward rule built from the VJPs of {ref}`sec-ch2-vjp`.

**Feed-forward layer.** For one position, $z_{\text{ff}} = x W_{\text{up}}$ and $y = \operatorname{act}(z_{\text{ff}})\,W_{\text{down}}$. Given $g = \partial \mathcal{L}/\partial y$, the backward pass runs the forward pass in reverse:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{down}}} = \operatorname{act}(z_{\text{ff}})^\top g, \qquad \frac{\partial \mathcal{L}}{\partial z_{\text{ff}}} = \left(g W_{\text{down}}^\top\right) \odot \operatorname{act}'(z_{\text{ff}}), \qquad \frac{\partial \mathcal{L}}{\partial W_{\text{up}}} = x^\top \frac{\partial \mathcal{L}}{\partial z_{\text{ff}}}, \qquad \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z_{\text{ff}}} W_{\text{up}}^\top.$$

For a sequence the same products are taken with one row per position, and the weight gradients sum over the positions.

**Embedding.** The embedding of token $x_i$ is row $x_i$ of $W_E$, so the gradient of $W_E$ is zero except in the rows of tokens that occur, and the row of a token that occurs at several positions is the sum of $\partial \mathcal{L} / \partial h^{(0)}_i$ over those positions. This is the sparse gradient of {ref}`sec-ch2-adam-optimizer`.

**Output head.** The logits at position $i$ are $z_i = h_i W_{\text{out}}$ (a row vector of $|\mathcal{V}|$ entries), and the loss is the mean cross-entropy over $T$ positions. By {ref}`sec-ch2-softmax-jacobian`, $\partial \mathcal{L}/\partial z_{ij} = (p_{ij} - \delta_{j x_i})/T$ for entry $j$ of $z_i$, where $x_i$ here stands for the target, the token to be predicted at position $i$, and $\partial \mathcal{L}/\partial W_{\text{out}} = \sum_i h_i^\top \partial \mathcal{L}/\partial z_i$, a sum of outer products over positions. With $T = 3$ and a model that gives the target probability 0.2, that logit's gradient is $(0.2 - 1)/3 = -0.2667$.

The listing builds a three-token sequence through an embedding, a feed-forward layer with $\operatorname{act} = \tanh$ and an output head, applies these rules by hand and compares them with autograd.

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab, d, d_ff = 10, 4, 16
tokens, targets = torch.tensor([3, 7, 3]), torch.tensor([7, 3, 1])
W_E = torch.randn(vocab, d, dtype=torch.float64, requires_grad=True)
W_up = torch.randn(d, d_ff, dtype=torch.float64, requires_grad=True)
W_down = torch.randn(d_ff, d, dtype=torch.float64, requires_grad=True)
W_out = torch.randn(d, vocab, dtype=torch.float64, requires_grad=True)

# forward pass
x = W_E[tokens]                     # embedding lookup, one row per position
z_ff = x @ W_up
a = torch.tanh(z_ff)
h = x + a @ W_down                  # a residual feed-forward block
logits = h @ W_out
loss = F.cross_entropy(logits, targets)
loss.backward()

# backward pass by hand
T = len(tokens)
p = torch.softmax(logits.detach(), dim=-1)
dz = (p - F.one_hot(targets, vocab)) / T
g_out = h.detach().T @ dz
dh = dz @ W_out.detach().T
g_down = a.detach().T @ dh
dz_ff = (dh @ W_down.detach().T) * (1 - a.detach() ** 2)
g_up = x.detach().T @ dz_ff
dx = dh + dz_ff @ W_up.detach().T
g_E = torch.zeros_like(W_E).index_add_(0, tokens, dx)

for name, ours, ref in (("W_out", g_out, W_out.grad), ("W_down", g_down, W_down.grad),
                        ("W_up", g_up, W_up.grad), ("W_E", g_E, W_E.grad)):
    print(f"{name}: matches autograd {torch.allclose(ours, ref)}")
nonzero_rows = g_E.abs().sum(dim=1).nonzero().flatten().tolist()
print("rows of W_E with a nonzero gradient:", nonzero_rows)
```

% output

```text
W_out: matches autograd True
W_down: matches autograd True
W_up: matches autograd True
W_E: matches autograd True
rows of W_E with a nonzero gradient: [3, 7]
```

```{admonition} Gradient Flow Analysis
:class: tip
The notebook section {ref}`nb-math-backprop` implements scaled dot-product attention and
reports its input gradients, and {ref}`nb-math-layernorm` compares gradient norms
through a stack of linear layers with and without layer normalization, with residual
connections, and with a Xavier-scaled initialization, so you can see which of these
actually prevents vanishing. With weights of standard deviation 0.02 and nothing else,
the gradient at the input is a fraction 1.912e-08 of the gradient at the last layer;
Xavier scaling alone keeps that ratio near one.
```

(sec-ch2-scaling-laws-compute)=

## 4. Scaling Laws and Compute

(sec-ch2-scaling-hypothesis)=

### 4.1 The Scaling Hypothesis

**Scaling laws** are empirical fits of a model's loss as a power law in its parameter count, its training tokens or its training compute, each measured with the other two not limiting.

(sec-ch2-parameter-scaling)=

#### Parameter Scaling

For models trained to convergence on a dataset large enough not to limit them, the relationship between model parameters $N$ and loss $\mathcal{L}$ follows a power law {cite}`kaplan2020scaling`:

$$\mathcal{L}(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

where $N$ counts non-embedding parameters, $N_c \approx 8.8 \times 10^{13}$ is a fitted scale and $\alpha_N \approx 0.076$ for language modeling.

(sec-ch2-data-scaling)=

#### Data Scaling

For large models trained on a limited dataset with early stopping, the loss falls with dataset size $D$ in tokens:

$$\mathcal{L}(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

with $\alpha_D \approx 0.095$ and $D_c \approx 5.4 \times 10^{13}$ tokens. Kaplan et al. fit these laws on models from 768 to 1.5 billion non-embedding parameters and datasets from 22 million to 23 billion tokens, so applying them to larger models is an extrapolation.

{numref}`fig-ch2-kaplan-power-laws` plots both laws and marks how far each fit reaches.

```{figure} figures/kaplan-power-laws.*
:name: fig-ch2-kaplan-power-laws
:alt: Two log-log plots of loss against N and against D, straight lines; the fitted ranges 768 to 1.5 billion parameters and 22 million to 23 billion tokens are shaded, with dashed extrapolation beyond.

The parameter and data laws with the constants above: $N_c = 8.8 \times 10^{13}$,
$\alpha_N = 0.076$ and $D_c = 5.4 \times 10^{13}$, $\alpha_D = 0.095$. A power law is a
straight line on log-log axes; the shaded bands are the ranges Kaplan et al. fit, and
the dashed lines are extrapolation. Constants from {cite:t}`kaplan2020scaling`, curves
computed by `figures/make_figures.py`.
```

(sec-ch2-flops)=

#### Training Compute per Token

The compute axis of the scaling laws counts floating-point operations, and nearly all of them are matrix products. A weight matrix with $d_{\text{in}} d_{\text{out}}$ entries, applied to one token's row vector $x$, costs $d_{\text{in}} d_{\text{out}}$ multiply-adds, which is 2 FLOPs per parameter. Summed over every weight matrix, the forward pass costs about $2N$ FLOPs per token. The backward pass ({ref}`sec-ch2-vjp`) needs two products of the same size for each matrix: $g W^\top$, the gradient passed down to the layer's input, and $x^\top g$, the gradient of the weights themselves. That is another $4N$ FLOPs, so one training token costs

$$2N + 4N = 6N \text{ FLOPs}, \qquad C \approx 6ND$$

for $D$ training tokens. {cite:t}`kaplan2020scaling` count the forward pass as $2N + 2 L n d$ per token, the second term being the attention scores. They call these "add-multiply operations, where the factor of two comes from the multiply-accumulate operation used in matrix multiplication": each multiply-add is counted as two operations, so both terms are FLOPs in this page's sense. They drop the second term because, when $d > n/12$, it is "a relatively small fraction of the total compute"; {ref}`sec-ch2-training-complexity` derives where the crossover lies. They also leave the embedding out of $N$. Biases, normalization, the softmax and the optimizer step add work proportional to $N$ or smaller, not to $N$ times a width, and are left out as well.

For Chinchilla, $N = 70 \times 10^9$ and $D = 1.4 \times 10^{12}$ give $C \approx 5.88 \times 10^{23}$ FLOPs. Turning the rule around answers the planning question: with $D = 20N$ ({ref}`sec-ch2-training-implications`), $C = 6N \cdot 20N = 120N^2$, so a budget $C$ buys a compute-optimal model of $N = \sqrt{C/120}$ parameters.

```python
N, D = 70e9, 1.4e12
C = 6 * N * D
print(f"C = 6ND = {C:.3g} FLOPs = {C / 8.64e19:,.0f} PF-days")
for budget in (1e21, 5.88e23, 1e25):
    N_opt = (budget / 120) ** 0.5
    print(f"budget {budget:.3g} FLOPs: N = {N_opt:.3g} parameters, "
          f"D = 20N = {20 * N_opt:.3g} tokens")
```

% output

```text
C = 6ND = 5.88e+23 FLOPs = 6,806 PF-days
budget 1e+21 FLOPs: N = 2.89e+09 parameters, D = 20N = 5.77e+10 tokens
budget 5.88e+23 FLOPs: N = 7e+10 parameters, D = 20N = 1.4e+12 tokens
budget 1e+25 FLOPs: N = 2.89e+11 parameters, D = 20N = 5.77e+12 tokens
```

(sec-ch2-compute-scaling)=

#### Compute Scaling

Training compute also follows a power law. On the compute-efficient frontier (the best loss reachable with a given budget), Kaplan et al. report

$$\mathcal{L}(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \qquad C_c^{\min} \approx 3.1 \times 10^{8}\ \text{PF-days},\quad \alpha_C^{\min} \approx 0.050$$

where compute is measured in petaflop/s-days (one PF-day is $8.64 \times 10^{19}$ FLOPs).

(sec-ch2-chinchilla-scaling-laws)=

### 4.2 Chinchilla Scaling Laws

For a given compute budget $C$, the optimal model size $N^*$ and dataset size $D^*$ grow as

$$N^* \propto C^{a_N}, \quad D^* \propto C^{a_D}$$

{cite:t}`kaplan2020scaling` found $N^* \propto C^{0.73}$ and $D^* \propto C^{0.27}$: most of a larger budget should go into a larger model. {cite:t}`hoffmann2022training` revisited the question with three independent methods and found $a_N = 0.50, a_D = 0.50$; $a_N = 0.49, a_D = 0.51$; and $a_N = 0.46, a_D = 0.54$, so parameters and data should grow in equal proportion. Their third method fits the loss as a function of both,

$$\mathcal{L}(N, D) = 1.69 + \frac{406.4}{N^{0.34}} + \frac{410.7}{D^{0.28}}$$

where 1.69 is the loss no model or dataset size removes. At $N = 70 \times 10^9$ and $D = 1.4 \times 10^{12}$ (Chinchilla's own size) the fit gives $1.69 + 0.0835 + 0.1632 = 1.937$.

(sec-ch2-training-implications)=

#### Implications for Training

{cite:t}`hoffmann2022training` conclude that "current large language models are significantly undertrained": at the same compute, a smaller model trained on more tokens reaches a lower loss. A smaller model is also cheaper to run, so the shift lowers inference cost as well.

Because $a_N \approx a_D \approx 0.5$, the compute-optimal ratio $D^*/N^*$ is roughly the same at every budget, and Hoffmann et al.'s Table 3, which gives the projections of their first approach, puts it near 20 tokens per parameter: 20.2 billion tokens for 1 billion parameters, 205.1 billion for 10 billion, 1.5 trillion for 67 billion and 3.7 trillion for 175 billion {cite}`hoffmann2022training`, ratios of 20.2, 20.5, 22.4 and 21.1. Chinchilla itself, 70 billion parameters on 1.4 trillion tokens, sits at 20. With $C \approx 6ND$ ({ref}`sec-ch2-flops`) the rule fixes both sizes from the budget alone. GPT-3's 175 billion parameters would on this rule call for about 3.5 trillion training tokens (Table 3's own projection is 3.7 trillion).

{numref}`fig-ch2-allocation-lines` shows how far the two prescriptions part once the budget moves away from Chinchilla's.

```{figure} figures/allocation-lines.*
:name: fig-ch2-allocation-lines
:alt: Two log-log plots against compute: optimal parameters and optimal tokens for Kaplan's exponents 0.73 and 0.27 and Chinchilla's 0.50 and 0.50, crossing at Chinchilla's size, with the four Table 3 points on the Chinchilla lines; the 67 billion point sits inside the hollow star.

Compute-optimal parameters $N^*$ (left) and tokens $D^*$ (right) under Kaplan's
exponents and Chinchilla's. Both lines are drawn through Chinchilla's own size (70
billion parameters, 1.4 trillion tokens, $C = 6ND$); the dots are the four rows of Table
3 quoted above, the 67 billion row inside the hollow star that marks Chinchilla. Each
panel's legend gives that panel's exponents. Exponents and points from
{cite:t}`kaplan2020scaling` and {cite:t}`hoffmann2022training`; the common anchor is
this figure's choice.
```

(sec-ch2-computational-complexity)=

### 4.3 Computational Complexity

(sec-ch2-training-complexity)=

#### Training Complexity

Training a transformer requires $\mathcal{O}(n \cdot d^2 + n^2 \cdot d)$ operations per layer:

- $\mathcal{O}(n \cdot d^2)$: the four attention projections (Q, K, V, output), $4nd^2$ multiply-adds
- $\mathcal{O}(n \cdot d^2)$: the feed-forward network with a $4d$ hidden width, $8nd^2$ multiply-adds, about two thirds of the linear-layer cost
- $\mathcal{O}(n^2 \cdot d)$: the attention scores and their application to the values

For long sequences, attention dominates; for large models, linear layers dominate. The crossover follows from the two counts. Per layer and sequence the linear layers cost $12nd^2$ multiply-adds and the attention scores and their application to the values $2n^2d$ ($n^2 d$ each for $QK^\top$ and $AV$). The two are equal when

$$12nd^2 = 2n^2 d \quad\Longleftrightarrow\quad n = 6d.$$

At $d = 4096$ attention costs as much as the linear layers only at $n = 24{,}576$ tokens; for GPT-3's $d = 12{,}288$ at its context of 2048 tokens, attention costs $n/(6d) = 2.78\%$ as much as the linear layers, 2.7% of the layer's work. In FLOPs per token the $2n^2d$ multiply-adds above are $4nd$ per layer, $4 L n d$ over $L$ layers; Kaplan et al. count the context term as $2 L n d$ FLOPs per token, half of that, so in their accounting the terms meet at $n = 12d$, which is their condition $d > n/12$ ({ref}`sec-ch2-flops`).

```python
def attention_share(n, d):
    # multiply-adds per layer and sequence
    linear, attention = 12 * n * d ** 2, 2 * n ** 2 * d
    return attention / (linear + attention)


for n, d in ((2048, 12288), (8192, 4096), (24576, 4096), (131072, 4096)):
    share = attention_share(n, d)
    print(f"n={n:6d}, d={d:5d}: attention is {share:.1%} of the per-layer work")
```

% output

```text
n=  2048, d=12288: attention is 2.7% of the per-layer work
n=  8192, d= 4096: attention is 25.0% of the per-layer work
n= 24576, d= 4096: attention is 50.0% of the per-layer work
n=131072, d= 4096: attention is 84.2% of the per-layer work
```

{numref}`fig-ch2-attention-crossover` plots the share over the whole range of sequence lengths.

```{figure} figures/attention-crossover.*
:name: fig-ch2-attention-crossover
:alt: Attention's share of per-layer work against sequence length for d = 4,096 and d = 12,288; each crosses 50 percent at n = 6d, and GPT-3 at n = 2,048 is at 2.7 percent.

Attention's share of the per-layer multiply-adds, $2n^2d/(12nd^2 + 2n^2d)$, against
sequence length. Each curve crosses one half at $n = 6d$; GPT-3 at its context of 2,048
tokens sits at 2.7%. Computed by `figures/make_figures.py` from the `attention_share`
listing above.
```

(sec-ch2-memory-requirements)=

#### Memory Requirements

Training memory has three parts:

- **Parameters**: $\approx 12 L d^2 + |\mathcal{V}| d$ where $L$ is the number of layers (each layer holds $4d^2$ attention and $8d^2$ feed-forward weights) and $|\mathcal{V}| d$ is the embedding table. The $8d^2$ assumes the dense $4d$-wide feed-forward layer; a gated layer such as SwiGLU ({ref}`sec-ch1-modern-block`) has three $d \times d_{\text{ff}}$ matrices, so it keeps the same $8d^2$ parameters with $d_{\text{ff}} = 8d/3$, 10,923 at $d = 4096$
- **Activations**: $\mathcal{O}(B \cdot n \cdot d \cdot L)$ for the residual stream, plus the $\mathcal{O}(B \cdot n_h \cdot n^2 \cdot L)$ attention scores and the $4d$-wide feed-forward intermediates, where $B$ is batch size and $n_h$ the number of heads; without activation recomputation (below) each layer stores about $B\,n\,d\,(34 + 5\,n_h\,n/d)$ bytes {cite}`korthikanti2022reducing`
- **Optimizer and gradient state**: with mixed precision and Adam, each parameter costs about 16 bytes in total: a 2-byte weight, a 2-byte gradient, a 4-byte fp32 master copy and two 4-byte Adam moments {cite}`rajbhandari2020zero`

**Activation recomputation** (also called activation checkpointing) trades time for memory. Instead of storing every intermediate of the forward pass for the backward pass ({ref}`sec-ch2-reverse-mode`), it stores only the input of each layer and, during the backward pass, runs that layer's forward pass again to rebuild the rest. {cite:t}`korthikanti2022reducing` show that this "full activation recomputation" cuts the activation memory to $2\,B\,n\,d\,L$ bytes, the fp16 layer inputs alone, at the price of one extra forward pass, about $2N$ more FLOPs per token on top of $6N$; they "observe 30 − 40% execution time overhead" in their runs. For the GPT-3 configuration below at batch 8 that is $2 \times 8 \times 2048 \times 12{,}288 \times 96$ bytes $= 38.7$ GB instead of 2,203 GB.

The function below also counts compute: about $6N$ FLOPs per training token (2 per parameter for the forward pass, 4 for the backward), so $C \approx 6ND$ for $D$ tokens {cite}`kaplan2020scaling`, as derived in {ref}`sec-ch2-flops`. Kaplan et al. count only non-embedding parameters; the function uses the total, which for GPT-3 is 0.36% more.

```python
def estimate_memory_usage(num_layers, d_model, seq_length, batch_size, vocab_size,
                          num_heads=None):
    """Estimate parameters, memory and compute for mixed-precision Adam training.

    Parameters per layer: 4*d^2 (attention projections) + 8*d^2 (feed-forward, 4d wide).
    Memory per parameter: 2 (fp16 weight) + 2 (fp16 grad) + 4 (fp32 master)
        + 8 (Adam m, v).
    Activations per layer without recomputation (Korthikanti et al., 2022):
        s*b*h*(34 + 5*a*s/h) bytes, in the paper's symbols: sequence s (n in this
        chapter), batch b (B), hidden size h (d) and heads a (n_h).
    """
    num_heads = num_heads or max(1, d_model // 128)
    layer_params = num_layers * 12 * d_model**2
    embedding_params = vocab_size * d_model
    total_params = layer_params + embedding_params

    bytes_per_param = 2 + 2 + 4 + 8
    param_state_gb = total_params * bytes_per_param / 1e9

    s, b, h, a = seq_length, batch_size, d_model, num_heads
    activation_gb = num_layers * s * b * h * (34 + 5 * a * s / h) / 1e9

    return {
        "parameters_B": total_params / 1e9,
        "param_and_optimizer_GB": param_state_gb,
        "activation_GB": activation_gb,
        "total_memory_GB": param_state_gb + activation_gb,
        "flops_per_step": 6 * total_params * batch_size * seq_length,
    }


# Example: the GPT-3 175B configuration (96 layers, d_model 12288,
# 50257-token vocabulary)
gpt3 = estimate_memory_usage(num_layers=96, d_model=12288, seq_length=2048,
                             batch_size=8, vocab_size=50257)
print(f"parameters: {gpt3['parameters_B']:.1f} B")               # ~174.6 B
print(f"weights + optimizer state: {gpt3['param_and_optimizer_GB']:.0f} GB")  # ~2800 GB
print(f"activations (batch 8, no recomputation): {gpt3['activation_GB']:.0f} GB")
```

% output

```text
parameters: 174.6 B
weights + optimizer state: 2793 GB
activations (batch 8, no recomputation): 2203 GB
```

Weights and optimizer state alone are close to three terabytes, which is why a 175B-parameter model is trained across hundreds of accelerators with the parameters, gradients and optimizer state sharded between them.

{numref}`fig-ch2-memory-stack` breaks the total down.

```{figure} figures/memory-stack.*
:name: fig-ch2-memory-stack
:alt: Stacked bar of GPT-3 training memory: fp16 weights 349 GB, fp16 gradients 349 GB, fp32 master weights 698 GB, Adam moments 1,397 GB and activations 2,203 GB, 4,996 GB in total.
:width: 90%

Training memory of the GPT-3 175B configuration with mixed precision and Adam, batch 8
of 2,048 tokens, no activation recomputation. The Adam moments alone take as much memory
as the weights, gradients and master copy together, 8 bytes per parameter each. Computed
by `figures/make_figures.py` from the `estimate_memory_usage` listing above.
```

```{admonition} Scaling Laws Visualization
:class: note
The notebook section {ref}`nb-math-scaling` plots the Kaplan parameter, data and compute
laws (with the sizes of GPT-2, GPT-3 and Llama 2 70B marked on the parameter curve as
extrapolations of the fit, not measured losses) and the Chinchilla allocation derived
from $C \approx 6ND$ {cite}`kaplan2020scaling` and $D = 20N$, and {ref}`nb-math-memory`
computes parameter counts, training memory and FLOPs per step for the GPT-2 and GPT-3
model sizes, asserting that the GPT-3 configuration comes out within 2% of 175 billion
parameters (it gives 174.56 billion).
```

(sec-ch2-numerical-stability-considerations)=

## 5. Numerical Stability Considerations

(sec-ch2-floating-point-precision)=

### 5.1 Floating Point Precision

The three floating-point formats used in training differ in how they split their bits between the exponent, which sets the range, and the mantissa, which sets the precision:

| Format | Sign | Exponent | Mantissa | Largest value | Smallest normal value | Machine epsilon |
|---|---|---|---|---|---|---|
| fp32 | 1 | 8 | 23 | $3.403 \times 10^{38}$ | $1.175 \times 10^{-38}$ | $1.192 \times 10^{-7}$ |
| fp16 | 1 | 5 | 10 | 65,504 | $6.104 \times 10^{-5}$ | $9.766 \times 10^{-4}$ |
| bf16 | 1 | 8 | 7 | $3.390 \times 10^{38}$ | $1.175 \times 10^{-38}$ | $7.812 \times 10^{-3}$ |

Machine epsilon is the gap between 1 and the next representable number, $2^{-\text{mantissa bits}}$. bf16 has fp32's range with fewer than three significant decimal digits; fp16 has about three digits but overflows above 65,504 and loses precision below $6.104 \times 10^{-5}$. The table is what PyTorch reports:

```python
import torch

for dtype in (torch.float32, torch.float16, torch.bfloat16):
    info = torch.finfo(dtype)
    print(f"{str(dtype):15s} bits {info.bits}, max {info.max:.4g}, "
          f"smallest normal {info.smallest_normal:.4g}, eps {info.eps:.4g}")
```

% output

```text
torch.float32   bits 32, max 3.403e+38, smallest normal 1.175e-38, eps 1.192e-07
torch.float16   bits 16, max 6.55e+04, smallest normal 6.104e-05, eps 0.0009766
torch.bfloat16  bits 16, max 3.39e+38, smallest normal 1.175e-38, eps 0.007812
```

{numref}`fig-ch2-float-formats` draws the table.

```{figure} figures/float-formats.*
:name: fig-ch2-float-formats
:alt: Top: bit layouts of fp32 (1 sign, 8 exponent, 23 mantissa bits), fp16 (1, 5, 10) and bf16 (1, 8, 7). Bottom: on a log axis, fp32 and bf16 span 1e-38 to 3e38 while fp16 spans only 6.1e-5 to 65,504.

Bit layouts (top) and normal ranges (bottom) of the three formats. bf16 keeps fp32's 8
exponent bits and so its range, and pays with mantissa bits; fp16 spends more bits on
the mantissa and covers about 9 decades instead of about 76. Values from the table
above, as reported by `torch.finfo`.
```

(sec-ch2-log-sum-exp)=

#### Log-Sum-Exp and a Stable Softmax

The softmax and the cross-entropy loss both need $\log \sum_j e^{z_j}$. Computed as written, $e^{z_j}$ overflows fp32 once a logit exceeds about 88.72, and in fp16 once it exceeds about 11.09. Subtracting the largest logit $z_{\max}$ changes nothing mathematically,

$$\log \sum_j e^{z_j} = z_{\max} + \log \sum_j e^{z_j - z_{\max}},$$

because $e^{z_{\max}}$ factors out of the sum, but now every exponent is at most zero, the largest term is exactly 1, and the sum lies between 1 and the number of logits. The softmax is shifted the same way, $p_i = e^{z_i - z_{\max}} / \sum_j e^{z_j - z_{\max}}$, which is why `F.log_softmax` and `F.cross_entropy` take logits rather than probabilities. For $z = (100, 99)$ the shifted form gives $100 + \ln(1 + e^{-1}) = 100.3$, while the direct form returns infinity.

```python
import torch

z = torch.tensor([100.0, 99.0])
naive = torch.log(torch.exp(z).sum())
z_max = z.max()
stable = z_max + torch.log(torch.exp(z - z_max).sum())
library = torch.logsumexp(z, 0).item()
print(f"direct: {naive.item()}, shifted: {stable.item():.4g}, "
      f"torch.logsumexp: {library:.4g}")
print("softmax:", [round(v, 4) for v in torch.softmax(z, 0).tolist()])
```

% output

```text
direct: inf, shifted: 100.3, torch.logsumexp: 100.3
softmax: [0.7311, 0.2689]
```

(sec-ch2-mixed-precision-training)=

#### Mixed Precision Training

**fp16** (half precision) halves the memory of fp32 and runs faster on hardware with fp16 arithmetic, but its narrow range lets small gradients underflow to zero and large values overflow. **Mixed precision** {cite}`micikevicius2017mixed` uses fp16 for most operations but fp32 for loss computation and gradient updates.

(sec-ch2-loss-scaling)=

#### Loss Scaling

To prevent gradient underflow in fp16, gradients are scaled up during backpropagation and scaled down before parameter updates:

$$\mathrm{loss}_{\text{scaled}} = \mathrm{loss} \times \zeta$$

where $\zeta$ is the scale factor. By the chain rule every gradient is multiplied by $\zeta$ as well, and the optimizer divides them by $\zeta$ in fp32 before the update, so the update itself is unchanged. A gradient of $10^{-8}$ is below fp16's smallest subnormal, $2^{-24} = 5.96 \times 10^{-8}$, and would be stored as zero; with $\zeta = 2^{16}$ it becomes $6.55 \times 10^{-4}$, which fp16 stores with its full ten mantissa bits ({ref}`sec-ch2-exercise-5`).

The right $\zeta$ depends on the largest gradient: {cite:t}`micikevicius2017mixed` trained with constant factors between 8 and 32K, chosen so that the factor times "the maximum absolute gradient value is below 65,504", and noted that an overflow can be detected in the gradients and that update skipped. They proposed, as future work, that the factor "could be dynamically increased or decreased by inspecting the weight gradients for overflow". **Dynamic loss scaling** is that rule as frameworks implement it; PyTorch's `GradScaler` uses these defaults {cite}`pytorch2026gradscaler`: it starts at $\zeta = 2^{16}$; when any scaled gradient overflows to infinity or NaN it skips that optimizer step and multiplies $\zeta$ by 0.5; and after 2,000 consecutive steps without overflow it multiplies $\zeta$ by 2. The scale therefore settles just below the largest value that does not overflow, halving quickly when it is too large and probing upward again every 2,000 steps.

(sec-ch2-bfloat16)=

#### bfloat16

**bf16** keeps fp32's 8 exponent bits and gives up mantissa bits instead, so it covers the same range as fp32 and needs no loss scaling. {cite:t}`kalamkar2019study` give that range as the reason bf16 suits training ("the range of values it can represent is the same as that of IEEE 754 floating-point format (FP32)") and report that training in bf16 reaches the same results as fp32 "in the same number of iterations and with no changes to hyper-parameters" across image, speech, language-modeling and recommendation workloads. Their training is mixed precision, converting tensors between fp32 and bf16, and the rounding used for that conversion is one of the operations they study. On hardware without bf16 arithmetic, fp16 with loss scaling is the alternative.

(sec-ch2-epsilon-placement)=

#### Where the Stability Constant Goes

Adam and LayerNorm both add a small $\epsilon$ to a denominator, in different places. Adam adds it outside the square root, $\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ ({ref}`sec-ch2-adam-optimizer`), as in Kingma and Ba's algorithm and PyTorch's `Adam` {cite}`pytorch2026adam`. Without it the ratio $|\hat{m}_t|/\sqrt{\hat{v}_t}$ is still bounded, though not by the value usually quoted. {cite:t}`kingma2014adam` give $(1 - \beta_1)/\sqrt{1 - \beta_2} = 0.1/\sqrt{0.001} = 3.16$ as the bound when $1 - \beta_1 > \sqrt{1 - \beta_2}$, a case that "only happens in the most severe case of sparsity: when a gradient has been zero at all timesteps except at the current timestep", and add that "for less sparse cases, the effective stepsize will be smaller". The first statement holds: a rare token's embedding row ({ref}`sec-ch2-adam-optimizer`) whose gradient has been zero for 5,000 steps and is then nonzero moves by $3.15\,\eta$. The second does not. If the token then occurs in every batch with the same gradient, $\hat{m}_t$ approaches the gradient within tens of steps while $\sqrt{\hat{v}_t}$ grows over thousands, and the step rises to $4.24\,\eta$ at the second occurrence and $6.55\,\eta$ at the twelfth before it falls back toward $\eta$, the same curve as the uncorrected steps of Solution 2 in {ref}`sec-ch2-exercise-2`.

The bound that does hold for every gradient sequence follows from the Cauchy-Schwarz inequality, which bounds a sum of products by the square roots of the two sums of squares. Unrolled from their zero start, $m_t = (1 - \beta_1)\sum_{i=1}^{t} \beta_1^{t-i} g_i$ and $v_t = (1 - \beta_2)\sum_{i=1}^{t} \beta_2^{t-i} g_i^2$ ({ref}`sec-ch2-adam-optimizer`). Writing each term of the first sum as the product $(\beta_1/\sqrt{\beta_2})^{t-i} \cdot \beta_2^{(t-i)/2} g_i$ gives

$$\left|\sum_{i=1}^{t} \beta_1^{t-i} g_i\right| \le \sqrt{\sum_{i=1}^{t} \left(\frac{\beta_1^2}{\beta_2}\right)^{t-i}}\;\sqrt{\sum_{i=1}^{t} \beta_2^{t-i} g_i^2} = \sqrt{\frac{1 - (\beta_1^2/\beta_2)^t}{1 - \beta_1^2/\beta_2}}\;\sqrt{\frac{v_t}{1 - \beta_2}},$$

where the first sum is geometric with ratio $\beta_1^2/\beta_2 = 0.8108$. Multiplying by $1 - \beta_1$ and dividing $m_t$ and $v_t$ by their bias corrections gives

$$\frac{|\hat{m}_t|}{\sqrt{\hat{v}_t}} \le \frac{1 - \beta_1}{1 - \beta_1^t}\sqrt{\frac{1 - \beta_2^t}{1 - \beta_2}\cdot\frac{1 - (\beta_1^2/\beta_2)^t}{1 - \beta_1^2/\beta_2}} \;\longrightarrow\; \frac{1 - \beta_1}{\sqrt{(1 - \beta_2)(1 - \beta_1^2/\beta_2)}} = \frac{0.1}{\sqrt{0.001 \times 0.1892}} = 7.27.$$

The right-hand side is 1 at $t = 1$ and rises with $t$ toward its limit. Equality in Cauchy-Schwarz needs the two factors to be proportional, $\beta_2^{(t-i)/2} g_i \propto (\beta_1/\sqrt{\beta_2})^{t-i}$, that is $g_i \propto (\beta_2/\beta_1)^{i}$: a gradient that grows by the factor $\beta_2/\beta_1 = 1.11$ at every step. Such a gradient reaches $7.09\,\eta$ after 3,000 steps, and 7.27 is approached but never reached. For dense gradients of steady size the ratio is about 1 or less. Either way $\epsilon$ is not what keeps the step finite. It does two other things: it avoids $0/0$ for a weight that has so far received only zero gradients, and it damps the steps of weights whose gradients are tiny, which without it would move by about $\eta$ whatever their gradient's size. With constant gradients of $10^{-10}$, $\hat{m}_t = \sqrt{\hat{v}_t} = 10^{-10}$ and the step is $\eta \cdot 10^{-10}/(10^{-10} + 10^{-8}) = 0.0099\,\eta$ instead of $\eta$. LayerNorm adds it inside, $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2 + \epsilon}$ ({ref}`sec-ch2-layer-normalization-mathematics`), as PyTorch's `LayerNorm` documents {cite}`pytorch2026layernorm`, so the factor $1/\sigma$ can never exceed $1/\sqrt{\epsilon}$: 316.2 at PyTorch's default $\epsilon = 10^{-5}$. The size of $\epsilon$ has to suit the format. Adam's usual $10^{-8}$ is below fp16's smallest positive number, about $5.96 \times 10^{-8}$, and rounds to zero, which is one reason optimizer state is kept in fp32 ({ref}`sec-ch2-memory-requirements`).

```python
import math

import torch

beta1, beta2 = 0.9, 0.999
m = v = 0.0
steps = []  # |m_hat| / sqrt(v_hat): the step in units of eta, without eps
# a rare token: absent, then in every batch
for t, g in enumerate([0.0] * 5000 + [1.0] * 20, start=1):
    m, v = beta1 * m + (1 - beta1) * g, beta2 * v + (1 - beta2) * g * g
    if g:
        steps.append((m / (1 - beta1**t)) / math.sqrt(v / (1 - beta2**t)))
print("step / eta at the 1st, 2nd and 12th occurrence:",
      [round(steps[k], 2) for k in (0, 1, 11)])
bound = (1 - beta1) / math.sqrt((1 - beta2) * (1 - beta1**2 / beta2))
print("Cauchy-Schwarz bound:", round(bound, 2))
print("1e-8 in fp16:", torch.tensor(1e-8, dtype=torch.float16).item())
print("1e-8 in bf16:", f"{torch.tensor(1e-8, dtype=torch.bfloat16).item():.4g}")
print("largest LayerNorm factor 1/sqrt(eps) at eps = 1e-5:", f"{1e-5 ** -0.5:.4g}")
constant = torch.full((1, 8), 3.0)
print("LayerNorm of a constant vector:",
      torch.nn.functional.layer_norm(constant, (8,)).tolist()[0])
```

% output

```text
step / eta at the 1st, 2nd and 12th occurrence: [3.15, 4.24, 6.55]
Cauchy-Schwarz bound: 7.27
1e-8 in fp16: 0.0
1e-8 in bf16: 1.001e-08
largest LayerNorm factor 1/sqrt(eps) at eps = 1e-5: 316.2
LayerNorm of a constant vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

(sec-ch2-initialization-strategies)=

### 5.2 Initialization Strategies

(sec-ch2-xavier-glorot-initialization)=

#### Xavier/Glorot Initialization

For layers with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs, {cite:t}`glorot2010understanding` choose the variance that keeps activations and gradients at the same scale in both directions:

$$W \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$$

(sec-ch2-he-initialization)=

#### He Initialization

For ReLU activations, which set about half of their inputs to zero, {cite:t}`he2015delving` use variance $2/d_{\text{in}}$, twice the $1/d_{\text{in}}$ that would keep a linear layer's output scale:

$$W \sim \mathcal{N}\left(0, \frac{2}{d_{\text{in}}}\right)$$

(sec-ch2-residual-init)=

#### Transformer-Specific Initialization

Deep residual networks also scale their initialization with depth: Fixup {cite}`zhang2019fixup` shows that with the right depth-dependent scaling of the residual branches a transformer can be trained with no normalization layers at all, and GPT-2 scales the weights of its residual layers at initialization by one over the square root of the number of residual layers {cite}`radford2019language`. A block has two residual layers, attention and feed-forward, so $L$ blocks give $2L$ of them and the factor is $1/\sqrt{2L}$, or 0.1021 for GPT-2's 48 blocks.

The factor comes from how variances add along the residual stream. After all $2L$ residual layers the stream is the input plus every branch output, $h^{(0)} + \sum_{l=1}^{2L} F_l$. If the branch outputs at initialization have mean zero and are uncorrelated with each other and with the input, their variances add:

$$\operatorname{Var}\left(h^{(0)} + \sum_{l=1}^{2L} F_l\right) = \operatorname{Var}(h^{(0)}) + 2L \operatorname{Var}(F).$$

With an ordinary initialization each branch contributes about the same variance, so the stream's variance grows linearly with depth: 96 branch variances for GPT-2's 48 blocks. Multiplying the weights of the matrix that produces each branch's output by $1/\sqrt{2L}$ multiplies the branch output by the same factor and its variance by $1/(2L)$, and the sum of all $2L$ of them is again one branch variance, whatever the depth. The listing builds pre-LN-style stacks of random linear branches and measures the variance of the final stream with and without the factor.

```python
import torch

d = 256


def final_variance(blocks, scaled, seed):
    torch.manual_seed(seed)
    h = torch.randn(64, d)
    factor = (2 * blocks) ** -0.5 if scaled else 1.0
    for _ in range(2 * blocks):   # two residual layers per block
        W = torch.randn(d, d) / d ** 0.5 * factor
        h = h + torch.nn.functional.layer_norm(h, (d,)) @ W
    return h.var().item()


for blocks in (12, 48):
    unscaled = [final_variance(blocks, False, seed) for seed in range(12)]
    scaled = [final_variance(blocks, True, seed) for seed in range(12)]
    print(f"{blocks} blocks, 12 seeds: unscaled mean {sum(unscaled) / 12:.3g} "
          f"(single seeds {min(unscaled):.3g} to {max(unscaled):.3g}),\n"
          f"    with 1/sqrt(2L) mean {sum(scaled) / 12:.3g}")
```

% output

```text
12 blocks, 12 seeds: unscaled mean 25.1 (single seeds 24.6 to 25.9),
    with 1/sqrt(2L) mean 2.01
48 blocks, 12 seeds: unscaled mean 97.2 (single seeds 94.9 to 99.8),
    with 1/sqrt(2L) mean 2
```

```{admonition} Numerical Stability Demonstrations
:class: tip
The notebook section {ref}`nb-math-numerics` shows fp16 underflow (and bf16 surviving
it), clips a global gradient norm and plots before/after, compares Xavier and He
initialization statistics, and {ref}`nb-math-loss-scaling` simulates dynamic loss
scaling.
```

(sec-ch2-summary)=

## Summary

- **Linear algebra** ({ref}`sec-ch2-linear-algebra-you-need`). A weight matrix maps one space to another; dividing attention scores by $\sqrt{d_k}$ keeps their variance at 1 when query and key entries have unit variance. The softmax Jacobian is $\operatorname{diag}(p) - p p^\top$, which vanishes as one probability approaches 1. Reverse-mode differentiation never forms a Jacobian: it passes a vector-Jacobian product back through each operation.
- **Probability and information theory** ({ref}`sec-ch2-probability-information-theory`). The training loss is the cross-entropy $H(p, q) = H(p) + D_{KL}(p\,\|\,q)$; perplexity is its exponential, and bits per byte divides the loss in bits by the bytes of text, so it compares models with different tokenizers. KL divergence is asymmetric and infinite where the model gives zero probability to an outcome that occurs.
- **Optimization** ({ref}`sec-ch2-optimization-fundamentals`). On a quadratic, gradient descent is stable for $\eta < 2/\lambda_{\max}$ and heavy-ball momentum for $\eta < 2(1 + \beta_{\text{mom}})/\lambda_{\max}$. Adam divides the bias-corrected first moment by the root of the bias-corrected second moment, so its first step has size about $\eta$ whatever the scale of the gradient. Warm-up, cosine decay, global-norm clipping and gradient accumulation (which reproduces a large batch's gradient from micro-batches) control the step.
- **Backpropagation through transformers** ({ref}`sec-ch2-backpropagation-through-transformers`). A residual connection adds the identity to every layer's Jacobian, so a gradient path that skips every layer always exists; pre-LN keeps that path free of normalization. The attention gradient is the softmax Jacobian applied row by row, and the layer-norm gradient removes the components along the mean and along the normalized input.
- **Scaling laws and compute** ({ref}`sec-ch2-scaling-laws-compute`). Training costs about $6N$ FLOPs per token, so $C \approx 6ND$: Chinchilla's 70 billion parameters and 1.4 trillion tokens cost $5.88 \times 10^{23}$ FLOPs. Kaplan's power laws have exponents 0.076 in parameters and 0.095 in data; the projections of Hoffmann et al.'s first approach (their Table 3) put the compute-optimal ratio near 20 tokens per parameter, the ratio Chinchilla was trained at, and their parametric loss fit gives 1.937 at Chinchilla's own size. Per layer, attention scores cost $2n^2 d$ multiply-adds against $12 n d^2$ for the weight matrices, so they dominate only beyond $n = 6d$. Mixed-precision Adam holds 16 bytes per parameter before activations.
- **Numerical stability** ({ref}`sec-ch2-numerical-stability-considerations`). fp16 overflows above 65,504 and underflows below its smallest subnormal, so its gradients need loss scaling; bf16 has fp32's exponent range and fewer than three significant decimal digits. Subtracting the largest logit makes softmax and log-sum-exp safe in any format. Xavier and He initialization scale the weight variance to the layer's width so that activations neither grow nor shrink from layer to layer, and scaling each residual branch's output weights by $1/\sqrt{2L}$ keeps the residual stream's variance from growing with depth.

(sec-ch2-exercises)=

## Exercises

The exercises are graded: level 1 applies one definition by hand, level 2 combines two ideas, level 3 asks for a short derivation. Every exercise is worked with paper, pencil and a calculator; the numbers in the solutions are checked in the book's test suite.

(sec-ch2-exercise-1)=

### Exercise 1: Entropy, cross-entropy and coding (level 1)

A four-symbol source emits its symbols with probabilities $p = (1/2, 1/4, 1/8, 1/8)$. Compute $H(p)$ in bits and the perplexity $2^{H(p)}$. A model predicts the uniform distribution $q = (1/4, 1/4, 1/4, 1/4)$; compute $H(p, q)$ and $D_{KL}(p\,\|\,q)$. Relate the three numbers to the average length of a binary code for the source.

```{admonition} Solution 1
:class: dropdown

$$
H(p) = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3
= 1.75$$

bits, so the perplexity is $2^{1.75} = 3.364$: the source is as uncertain as a fair
choice among about 3.36 symbols. Every $q$ entry is $1/4$, so
$H(p, q) = -\sum_x p(x) \log_2 \tfrac14 = 2$ bits whatever $p$ is, and
$D_{KL}(p\,\|\,q) = H(p, q) - H(p) = 0.25$ bits.

A code with lengths $-\log_2 p(x)$, here 1, 2, 3 and 3 bits (for example 0, 10, 110,
111), averages exactly $H(p) = 1.75$ bits per symbol, the least any code can achieve. A
code built for $q$ spends 2 bits on every symbol, and the extra 0.25 bits per symbol is
the KL divergence: the price of coding with the wrong distribution. Training a language
model to lower cross-entropy is the same thing as making it a better compressor of its
training text.
```

(sec-ch2-exercise-2)=

### Exercise 2: Adam's first step and the stability bound (level 2)

(a) Adam starts with both moments at zero and uses $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$, $\epsilon = 10^{-8}$. For a single parameter whose first gradient is $g = 0.003$, and again for $g = 300$, compute the first update with bias correction and without it. (b) The notebook's test problem is the Rosenbrock function of two parameters, $\mathcal{L}(\theta) = (1 - \theta_x)^2 + 100\,(\theta_y - \theta_x^2)^2$. Compute its Hessian at the starting point $\theta = (-1, -1)$, its largest eigenvalue $\lambda_{\max}$, and the largest stable learning rates for gradient descent and for heavy-ball momentum with $\beta_{\text{mom}} = 0.9$ on a quadratic with that curvature.

```{admonition} Solution 2
:class: dropdown
(a) After one step the moments are $(1 - \beta_1) g = 0.1\,g$ and
$(1 - \beta_2) g^2 = 0.001\,g^2$. Bias correction divides them by $1 - \beta_1 = 0.1$
and $1 - \beta_2 = 0.001$, giving $\hat{m} = g$ and $\hat{v} = g^2$, so the update is
$\eta\, g / (|g| + \epsilon)$. For $g = 0.003$ that is
$10^{-3} \times 0.003 / 0.00300001 = 9.99997 \times 10^{-4}$, and for $g = 300$ it is
$10^{-3}$ to nine digits: the first step has size $\eta$ whatever the gradient's scale,
a gradient a hundred thousand times larger moves the parameter by the same amount.
Without the correction the update is

$$
\eta \cdot 0.1\,g / (\sqrt{0.001}\,|g| + \epsilon) = 0.1/\sqrt{0.001}\;\eta = 3.162
\times 10^{-3}$$

for both gradients, more than three times too large. It does not shrink at once: for a
constant gradient the uncorrected step is $\eta\,(1 - \beta_1^t)/\sqrt{1 - \beta_2^t}$,
because $m_t$ approaches $g$ in tens of steps while $v_t$ approaches $g^2$ only over
thousands. That ratio is 3.16 at $t = 1$, peaks at 6.57 at $t = 12$, and is still 2.35
at $t = 200$ and 1.26 at $t = 1000$ before it settles to 1. Bias correction removes this
overshoot at the start of training, but not later: an embedding row whose gradient has
been zero for thousands of steps has $m_t$ and $v_t$ near zero again while
$1 - \beta_1^t$ and $1 - \beta_2^t$ are close to 1, so when its token starts to occur in
every batch its corrected step follows almost the same curve, peaking at $6.55\,\eta$
({ref}`sec-ch2-epsilon-placement`). The peak is above the $3.16\,\eta$ that Kingma and
Ba give as the largest step, and below the $7.27\,\eta$ that no gradient sequence can
exceed.

(b) The second derivatives are

$$
\partial^2\mathcal{L}/\partial\theta_x^2 = 2 - 400\,(\theta_y - \theta_x^2)
+ 800\,\theta_x^2,$$

$\partial^2\mathcal{L}/\partial\theta_x\partial\theta_y = -400\,\theta_x$ and
$\partial^2\mathcal{L}/\partial\theta_y^2 = 200$. At $(-1, -1)$ they are
$2 + 800 + 800 = 1602$, $400$ and $200$, so the Hessian is
$\begin{pmatrix} 1602 & 400 \\ 400 & 200 \end{pmatrix}$. The eigenvalues of a symmetric
$2 \times 2$ matrix are the mean of its diagonal entries plus or minus
$\sqrt{(\text{half their difference})^2 + (\text{off-diagonal entry})^2}$:
$901 \pm \sqrt{701^2 + 400^2} = 901 \pm 807.1$, so $\lambda_{\max} = 1708$. Gradient
descent is stable for $\eta < 2/1708 = 1.171 \times 10^{-3}$ and momentum for
$\eta < 2 \times 1.9/1708 = 2.225 \times 10^{-3}$. Both bounds lie above $10^{-3}$, yet
the notebook's momentum run at $10^{-3}$ diverges: the bound holds only for the
curvature at the start, and momentum's overshoot carries the iterate into parts of the
valley where $\lambda_{\max}$ is larger ({ref}`sec-ch2-stability-condition`).
```

(sec-ch2-exercise-3)=

### Exercise 3: Planning a training run with the scaling laws (level 3)

Use Kaplan's parameter law $\mathcal{L}(N) = (N_c/N)^{\alpha_N}$ with $N_c = 8.8 \times 10^{13}$ and $\alpha_N = 0.076$, and the Chinchilla fit $\mathcal{L}(N, D) = 1.69 + 406.4/N^{0.34} + 410.7/D^{0.28}$. (a) By what factor must $N$ grow to lower Kaplan's loss by 10%? (b) Evaluate Kaplan's law at $N = 10^9$ and $N = 10^{10}$. (c) A budget of $C = 5.88 \times 10^{23}$ FLOPs, Chinchilla's own, can train 70 billion parameters on 1.4 trillion tokens, 280 billion on 350 billion, or 17.5 billion on 5.6 trillion; check that each costs $C$ under $C \approx 6ND$ and compare their Chinchilla losses. (d) Minimize the Chinchilla fit subject to $6ND = C$ and find the model size the fit itself prefers at this budget.

```{admonition} Solution 3
:class: dropdown
(a) Multiplying $N$ by some factor multiplies $\mathcal{L}(N)$ by that factor raised to
the power $-0.076$, so the loss falls by 10% when the factor is
$0.9^{-1/0.076} = e^{0.10536/0.076} = e^{1.3863} = 4.000$. Each 10% off the loss costs
four times the parameters; a power law with a small exponent is a law of steeply
diminishing returns.

(b) $\mathcal{L}(10^9) = (8.8 \times 10^4)^{0.076} = e^{0.076 \times 11.385} = 2.376$
nats per token and $\mathcal{L}(10^{10}) = (8.8 \times 10^3)^{0.076} = 1.994$. Tenfold
more parameters cut the loss by $1 - 10^{-0.076} = 16\%$. Here $N$ counts non-embedding
parameters and the loss is on Kaplan's WebText2 data; Chinchilla's fit counts all
parameters on a different dataset and tokenizer, so the two fits' losses are not
comparable with each other, only within each fit.

(c)

$$
6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 6 \times 280 \times 10^9
\times 350 \times 10^9 = 6 \times 17.5 \times 10^9 \times 5.6 \times 10^{12} = 5.88
\times 10^{23}.$$

The two terms of the fit are $0.0835 + 0.1632$ for 70 billion, $0.0521 + 0.2405$ for 280
billion and $0.1338 + 0.1107$ for 17.5 billion, giving losses of 1.937, 1.983 and 1.934.
The large model on few tokens is clearly worse: its data term dominates. Moving from 70
to 17.5 billion parameters changes the loss by only 0.002, so the minimum is flat, and
it lies below 70 billion.

(d) Substitute $D = C/(6N)$:
$\mathcal{L} = 1.69 + 406.4\,N^{-0.34} + 410.7\,(C/6)^{-0.28} N^{0.28}$. Setting the
derivative in $N$ to zero gives
$0.34 \times 406.4\,N^{-1.34} = 0.28 \times 410.7\,(C/6)^{-0.28} N^{-0.72}$, so

$$
N^* = \left(\frac{0.34 \times 406.4}{0.28 \times 410.7}\right)^{1/0.62}
\left(\frac{C}{6}\right)^{0.28/0.62} = 1.345 \left(\frac{C}{6}\right)^{0.4516}.$$

At $C = 5.88 \times 10^{23}$, $C/6 = 9.8 \times 10^{22}$ and $N^* = 3.25 \times 10^{10}$
parameters with $D^* = C/(6N^*) = 3.02 \times 10^{12}$ tokens, about 93 tokens per
parameter, at a loss of 1.930. This is not the paper's own prescription: the projections
of Hoffmann et al.'s Table 3, which come from their first approach, and Chinchilla
itself sit near 20 tokens per parameter ({ref}`sec-ch2-training-implications`).
{cite:t}`besiroglu2024chinchilla` found that the published constants of the parametric
fit are inconsistent with the paper's first two approaches and with the ratio Chinchilla
was trained at; refitting the same functional form to data reconstructed from the
paper's figures, they obtained an optimum of about 20 tokens per parameter (25.6 at
their point estimate). They report that Hoffmann et al.'s authors confirmed that the
original fit stopped its minimizer early because it averaged the Huber loss over data
points instead of summing it. The exponent $0.28/(0.34 + 0.28) = 0.4516$ rounds to 0.45,
not to the $a_N = 0.46$ the paper reports for this fit. The fit's two exponents are
published to two decimal places, which fixes the ratio only to between
$0.275/(0.345 + 0.275) = 0.444$ and $0.285/(0.335 + 0.285) = 0.460$, so the paper's 0.46
is consistent with the rounded constants. Either value is below the $a_N = 0.50$ and
$0.49$ from its first two approaches ({ref}`sec-ch2-chinchilla-scaling-laws`). Because
the fitted loss differs between 32.5 and 70 billion parameters by only 0.007, small
errors in the constants move the optimum a long way, while the conclusion of part (c),
that the 280-billion-parameter model is clearly worse, does not change.
```

(sec-ch2-exercise-4)=

### Exercise 4: Saturation and depth (level 2)

(a) A softmax over two logits $z_i \ge z_j$ gives the larger one the probability $p_i = 1/(1 + e^{-(z_i - z_j)})$, and $\partial p_i / \partial z_i = p_i(1 - p_i)$. Evaluate the probability and the derivative for gaps $z_i - z_j$ of 0, 1 and 8. Query and key entries are independent with mean 0 and variance 1, and $d_k = 64$: what is the standard deviation of a raw score $q \cdot k$, and of the scaled score $q \cdot k / \sqrt{d_k}$? (b) The residual stream starts as $h^{(0)}$ with variance 1 and has $2L$ branches added to it, each with variance 1 and uncorrelated with the rest. Compute its variance after $L = 12$ and $L = 48$ blocks, and again when each branch is multiplied by $1/\sqrt{2L}$.

```{admonition} Solution 4
:class: dropdown
(a) At gap 0, $p_i = 0.5$ and the derivative is $0.25$, its largest value. At gap 1,
$p_i = 0.7311$ and the derivative is $0.1966$. At gap 8, $p_i = 0.99966$ and the
derivative is $3.352 \times 10^{-4}$, 746 times smaller than at gap 0: a saturated
softmax passes almost no gradient back to its logits
({numref}`fig-ch2-softmax-saturation`). A raw score is a sum of 64 independent products,
each with variance $1 \times 1 = 1$, so its variance is 64 and its standard deviation 8.
Two scores $q \cdot k_i$ and $q \cdot k_j$ share the query, so they are not independent,
but they are uncorrelated: the two keys are independent of the query and of each other
and have mean zero, so $\mathbb{E}[(q \cdot k_i)(q \cdot k_j)] = 0$. Their variances
therefore still add, and their difference has standard deviation $8\sqrt{2} = 11.3$, so
typical gaps sit deep in saturation. Dividing by $\sqrt{d_k} = 8$ brings the standard
deviation of a score to 1 and of a gap to $\sqrt{2} = 1.41$; at that gap the derivative
is 0.157, about 470 times its value at gap 8 ({ref}`sec-ch2-sqrt-dk`).

(b) Variances of uncorrelated terms add, so the stream's variance is $1 + 2L$: 25 after
12 blocks and 97 after 48. Multiplying each branch by $1/\sqrt{2L}$ multiplies its
variance by $1/(2L)$, so the branches add $2L \times 1/(2L) = 1$ in total and the
variance is 2 at any depth. The simulation in {ref}`sec-ch2-residual-init`, averaged
over 12 seeds, measures 25.1 and 97.2 unscaled and 2 with the scaling, matching $1 + 2L$
and 2. A single seed can land a few percent to either side (94.9 to 99.8 at 48 blocks):
that spread is sampling noise from the random weights and inputs, and it averages out.
```

(sec-ch2-exercise-5)=

### Exercise 5: Formats, underflow and loss scaling (level 1)

Use the table in {ref}`sec-ch2-floating-point-precision`; fp16's smallest positive subnormal is $2^{-24}$. (a) What does a gradient of $10^{-8}$ become when stored in fp16? What does it become if the loss is first multiplied by $\zeta = 2^{16}$? (b) The largest gradient in a step is 3.0 before scaling. What is the largest power of two that $\zeta$ can be without overflowing fp16? (c) An optimizer adds an update of $10^{-4}$ to a weight equal to 1.0. What is the result in fp16, bf16 and fp32, and what does that imply for where the weights are kept in mixed-precision training?

```{admonition} Solution 5
:class: dropdown
(a) $2^{-24} = 5.96 \times 10^{-8}$, and a value below half of it,
$2.98 \times 10^{-8}$, rounds to zero, so $10^{-8}$ is stored as 0 and the gradient is
lost. Scaled by $2^{16} = 65{,}536$ it becomes $6.554 \times 10^{-4}$, above fp16's
smallest normal number $6.104 \times 10^{-5}$, and is stored as $6.552 \times 10^{-4}$
with fp16's ten mantissa bits; dividing by $\zeta$ in fp32 afterwards recovers $10^{-8}$
to three digits.

(b) The scaled gradient $3\zeta$ must stay at or below 65,504, so
$\zeta \le 65{,}504/3 = 21{,}834.67$, and the largest power of two is
$2^{14} = 16{,}384$. Dynamic loss scaling finds this by halving $\zeta$ whenever a step
overflows ({ref}`sec-ch2-loss-scaling`): from PyTorch's starting value $2^{16}$ the
scaled gradient $3 \times 65{,}536 = 196{,}608$ overflows, $2^{15}$ gives 98,304 and
overflows again, and $2^{14}$ gives 49,152, which fits, after two skipped steps.

(c) Next to 1.0 the representable numbers are $2^{-10} = 9.766 \times 10^{-4}$ apart in
fp16 and $2^{-7} = 7.812 \times 10^{-3}$ apart in bf16. An update smaller than half the
gap rounds away, so in both formats $1.0 + 10^{-4} = 1.0$ and the weight never changes.
In fp32 the gap is $1.192 \times 10^{-7}$ and the result is $1.0001$. This is why
mixed-precision training keeps a master copy of the weights in fp32 and applies the
update there, and why that copy is part of the 16 bytes per parameter in
{ref}`sec-ch2-memory-requirements`.
```

(sec-ch2-exercise-6)=

### Exercise 6: Attention backward by hand (level 2)

Take two positions, $d_k = d_v = 1$, $Q = (q, 0)^\top$ with $q = \ln 3$, $K = (1, 0)^\top$ and $V = (2, 0)^\top$, and let the loss be the first entry of $O$, so $\partial\mathcal{L}/\partial O = (1, 0)^\top$. (a) Compute $S$, $A$ and $O$. (b) Using {ref}`sec-ch2-attention-gradient-analysis`, compute $G$, $\partial\mathcal{L}/\partial S$, $\partial\mathcal{L}/\partial Q$, $\partial\mathcal{L}/\partial K$ and $\partial\mathcal{L}/\partial V$. Check $\partial\mathcal{L}/\partial q$ by differentiating the first entry of $O$ directly. (c) Repeat $\partial\mathcal{L}/\partial q$ with $q = 8$. What does the comparison show?

```{admonition} Solution 6
:class: dropdown
(a) $S = QK^\top = \begin{pmatrix} \ln 3 & 0 \\ 0 & 0 \end{pmatrix}$. Row 1 of the
softmax is $(3, 1)/4 = (0.75, 0.25)$ and row 2 is $(0.5, 0.5)$, so
$O = AV = (0.75 \cdot 2,\ 0.5 \cdot 2)^\top = (1.5, 1.0)^\top$.

(b)

$$
G = \frac{\partial \mathcal{L}}{\partial O} V^\top
= \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}.$$

In row 1 the sum of $G \odot A$ is $2 \times 0.75 = 1.5$, so
$\partial\mathcal{L}/\partial S$ has row 1
$= (0.75 \times (2 - 1.5),\ 0.25 \times (0 - 1.5)) = (0.375, -0.375)$ and row 2
$= (0, 0)$; each row sums to zero, as every softmax backward row does. Then

$$
\partial\mathcal{L}/\partial Q = \frac{\partial\mathcal{L}}{\partial S} K
= (0.375, 0)^\top,$$


$$
\partial\mathcal{L}/\partial K
= \left(\frac{\partial\mathcal{L}}{\partial S}\right)^\top Q
= (0.375 \ln 3,\ -0.375 \ln 3)^\top = (0.4120, -0.4120)^\top$$

and

$$
\partial\mathcal{L}/\partial V = A^\top \frac{\partial\mathcal{L}}{\partial O}
= (0.75, 0.25)^\top.$$

Directly, the first entry of $O$ is $2\,e^{q}/(e^{q} + 1)$, whose derivative in $q$ is
$2 p (1 - p)$ with $p = 0.75$: $2 \times 0.75 \times 0.25 = 0.375$.

(c) At $q = 8$, $p = 0.99966$ and
$\partial\mathcal{L}/\partial q = 2p(1 - p) = 6.705 \times 10^{-4}$, 559 times smaller.
The $A \odot (\cdot)$ factor scales the gradient by the small weight $1 - p$: a
saturated row passes almost no gradient back to its scores ({ref}`sec-ch2-sqrt-dk`).
```

(sec-ch2-exercise-7)=

### Exercise 7: A LayerNorm gradient (level 2)

A LayerNorm with $\gamma = 1$, $\beta = 0$ and $\epsilon = 0$ receives $x = (1, 2, 3)$, and the loss is the first feature of its output $y$. Compute $\mu$, $\sigma$, $\hat{x}$ and $\partial\mathcal{L}/\partial x$ with the formula of {ref}`sec-ch2-layer-normalization-mathematics`, and check that the gradient's entries sum to zero. Why must they?

```{admonition} Solution 7
:class: dropdown
$\mu = 2$, $\sigma^2 = (1 + 0 + 1)/3 = 2/3$, $\sigma = 0.8165$ and
$\hat{x} = (-1.2247, 0, 1.2247)$. Here $\tilde{g} = (1, 0, 0)$, so
$\frac{1}{d}\sum_j \tilde{g}_j = 1/3$ and
$\frac{1}{d}\sum_j \tilde{g}_j \hat{x}_j = -1.2247/3 = -0.4082$. Feature by feature the
formula gives

$$
\partial\mathcal{L}/\partial x
= \big((1 - 0.3333 - 0.5),\ (0 - 0.3333 - 0),\ (0 - 0.3333 + 0.5)\big)/0.8165
= (0.2041, -0.4082, 0.2041),$$

that is $(1, -2, 1)/(2\sqrt{6})$. The entries sum to zero because adding the same
constant to all three inputs leaves the output unchanged, so the gradient can have no
component along $(1, 1, 1)$.
```

(sec-ch2-exercise-8)=

### Exercise 8: Schedule, clipping and accumulation (level 1)

(a) A run uses linear warm-up and cosine decay ({ref}`sec-ch2-learning-rate-scheduling`) with $\eta_{\max} = 3 \times 10^{-4}$, $\eta_{\min} = 3 \times 10^{-5}$, $t_{\text{warm}} = 2000$ and $t_{\text{end}} = 100{,}000$. Compute $\eta_t$ at steps 0, 999, 26,500, 51,000 and 100,000. (b) A gradient $(3, 4)$ is clipped to norm 1.0 ({ref}`sec-ch2-gradient-clipping`). What is the clipped gradient? (c) An epoch has 10 micro-batches and $n_{acc} = 4$ ({ref}`sec-ch2-gradient-accumulation`). How many optimizer steps are taken, and by what must each micro-batch loss be divided?

```{admonition} Solution 8
:class: dropdown
(a) Step 0: $3 \times 10^{-4} \times 1/2000 = 1.5 \times 10^{-7}$; step 999:
$3 \times 10^{-4} \times 1000/2000 = 1.5 \times 10^{-4}$. After warm-up the cosine runs
over $98{,}000$ steps: step 26,500 is a quarter of the way, $\cos(\pi/4) = 0.7071$, so

$$
\eta = 3 \times 10^{-5} + \tfrac12 \times 2.7 \times 10^{-4} \times 1.7071 = 2.605
\times 10^{-4};$$

step 51,000 is halfway, $\cos(\pi/2) = 0$, so
$\eta = 3 \times 10^{-5} + 1.35 \times 10^{-4} = 1.65 \times 10^{-4}$; step 100,000 is
the end, $\cos \pi = -1$, so $\eta = \eta_{\min} = 3 \times 10^{-5}$.

(b) $\|g\| = 5$, so the gradient is multiplied by $1/5$: $(0.6, 0.8)$, with norm 1 and
the same direction.

(c) Three steps, after micro-batches 4, 8 and 10. The first two groups have four
micro-batches and divide each loss by 4; the last group has two and divides by 2, so
that each update uses the mean gradient of its own group.
```

(sec-ch2-further-reading)=

## Further Reading

(sec-ch2-further-reading-learning-path-beginners)=

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) {cite}`sanderson2026neural` - Visual explanations of backpropagation and gradient descent
1. [Distill: Momentum](https://distill.pub/2017/momentum/) - Interactive visualization of optimization algorithms
1. [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) {cite}`parr2018matrix` - Essential calculus for understanding backpropagation

**Mathematical Foundations:**

4. [Mathematics for Machine Learning](https://mml-book.github.io/) {cite}`deisenroth2020mathematics` - Free textbook covering linear algebra, calculus, and probability
1. [MIT 18.065: Matrix Methods in Data Analysis](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) - Gilbert Strang's course "Matrix Methods in Data Analysis, Signal Processing, and Machine Learning" {cite}`strang2018matrix`

(sec-ch2-further-reading-foundational-papers)=

### Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite}`vaswani2017attention`: The original transformer paper with mathematical details ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) {cite}`kaplan2020scaling`: Mathematical analysis of scaling behavior ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361))
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) {cite}`hoffmann2022training`: Chinchilla scaling laws ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556))
- {cite}`shazeer2020glu`: GLU variants of the transformer feed-forward layer ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202))

(sec-ch2-further-reading-optimization-theory)=

### Optimization Theory

**Optimizer Papers:**

- {cite}`kingma2014adam`: Adam optimizer derivation and analysis ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))
- {cite}`loshchilov2017decoupled`: AdamW and weight decay regularization ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101))
- {cite}`polyak1964some`: Classical momentum optimization ([DOI](<https://doi.org/10.1016/0041-5553(64)90137-5>))

**Learning Rate and Initialization:**

- {cite}`zhang2019fixup`: Advanced initialization strategies without normalization ([arXiv:1901.09321](https://arxiv.org/abs/1901.09321))
- {cite}`smith2017cyclical`: Cyclical learning rates, a practical learning-rate schedule ([arXiv:1506.01186](https://arxiv.org/abs/1506.01186))
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) {cite}`pascanu2013difficulty` - Gradient flow analysis

(sec-ch2-further-reading-numerical-methods-efficiency)=

### Numerical Methods and Efficiency

- {cite}`micikevicius2017mixed`: Mixed precision training techniques ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740))
- {cite}`you2019large`: Large batch training and scaling ([arXiv:1904.00962](https://arxiv.org/abs/1904.00962))

(sec-ch2-further-reading-mathematical-textbooks-references)=

### Mathematical Textbooks and References

- {cite}`goodfellow2016deep`: Deep learning mathematical foundations - [Online Book](https://www.deeplearningbook.org/)
- {cite}`murphy2012machine`: Comprehensive machine learning mathematics textbook
- {cite}`mackay2003information`: Information theory perspective on learning - [Online Book](http://www.inference.org.uk/mackay/itila/)

(sec-ch2-further-reading-scaling-laws-empirical-analysis)=

### Scaling Laws and Empirical Analysis

- {cite}`hestness2017deep`: Early observations of scaling behavior ([arXiv:1712.00409](https://arxiv.org/abs/1712.00409))
- {cite}`henighan2020scaling`: Scaling laws beyond language, for images, video, multimodal data and mathematics ([arXiv:2010.14701](https://arxiv.org/abs/2010.14701))
- {cite}`sorscher2022beyond`: Data pruning that beats power-law scaling in data ([arXiv:2206.14486](https://arxiv.org/abs/2206.14486))

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
