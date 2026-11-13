# Mathematical Foundations

The remarkable capabilities of Large Language Models emerge from sophisticated mathematical foundations spanning linear algebra, probability theory, information theory, and optimization. This chapter establishes the mathematical framework necessary to understand transformer architectures, training dynamics, and the scaling laws that govern LLM performance.

(sec-ch2-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- Probability and information theory concepts underlying language modeling
- Optimization fundamentals and the mathematics of gradient descent
- How backpropagation flows through transformer architectures
- Scaling laws that predict LLM performance and compute requirements
- Numerical stability considerations in large-scale training

(sec-ch2-chapter-overview)=

## Chapter Overview

The mathematics underlying Large Language Models can be understood through several interconnected concepts that build upon each other. A short linear-algebra primer (Section 0) fixes notation first.

**Probability and Information Theory** form the foundation of how LLMs understand and generate text. Think of it like predicting the next word in a sentence - the model assigns probabilities to different possibilities based on what it has learned. Information theory measures how "surprising" or "informative" each word choice is, helping the model learn which predictions are better.

**Optimization** is the process by which LLMs learn from data. Imagine adjusting thousands of knobs to minimize errors - that's essentially what gradient descent does. The model makes predictions, measures how wrong they are, and gradually adjusts its internal parameters to improve. Advanced optimizers like Adam are like smart assistants that remember which knobs to turn faster and which to adjust more carefully.

**Backpropagation** is the mechanism that tells the model how to improve. When the model makes a mistake, backpropagation traces the error backwards through all the layers, determining exactly how much each parameter contributed to the error. It's like analyzing a recipe gone wrong by working backwards from the taste to identify which ingredient amounts need adjustment.

**Scaling Laws** reveal predictable relationships between model size, training data, and performance. These mathematical patterns help researchers determine whether building a bigger model or gathering more data will be more effective. They're like economic laws that govern the "returns on investment" for computational resources.

**Numerical Stability** addresses practical challenges in training massive models with billions of parameters. Just as engineers must account for physical limitations when building bridges, ML practitioners must handle the computer's finite precision when performing trillions of calculations. Techniques like mixed precision training and gradient clipping prevent the mathematical equivalent of structural failure.

These mathematical principles work together to enable LLMs to learn patterns from vast amounts of text and generate coherent, contextually appropriate responses.

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

This is the identity behind the attention gradient in Section 3.3 and behind cross-entropy's gradient $p_j - \delta_{j x_i}$ (take $g_j = -\delta_{j x_i}/p_j$ for the target token $x_i$). It also explains saturation: if one $p_i \approx 1$ then every factor $p_i(\delta_{ij} - p_j)$ is near zero, and no gradient flows through that softmax.

(sec-ch2-norms-conditioning-scale-matters)=

### 0.3 Norms, conditioning and why scale matters

The operator norm $\|W\|_2$ is the largest factor by which $W$ can stretch a vector. A stack of $L$ layers multiplies these factors, which is why gradients vanish or explode geometrically with depth unless each layer is kept near norm 1, by initialization scale (Section 5.2), by normalization (Section 3.2) or by residual connections (Section 3.1). The Chapter 2 notebook measures all three on a twelve-layer stack.

(sec-ch2-svd-low-rank)=

### 0.4 Singular value decomposition and low rank

Every real matrix factors as $W = U \Sigma R^\top$, where $U$ and $R$ have orthonormal columns and $\Sigma$ is diagonal, holding the singular values $\Sigma_{jj} \ge 0$ in decreasing order. Read as a map, $W$ rotates (by $R^\top$), stretches each axis by a singular value, and rotates again (by $U$); the operator norm of {ref}`sec-ch2-norms-conditioning-scale-matters` is the largest stretch, so $\|W\|_2$ is the largest singular value.

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
    print(f"rank {n_rank}: Frobenius error {err:.4g} (from the singular values {predicted:.4g})")
print(f"parameters of a rank-8 factorization of 4096 x 4096: {8 * (4096 + 4096):,} of {4096 * 4096:,}")
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

For a layer that maps $x \in \mathbb{R}^{d_{\text{in}}}$ to $y \in \mathbb{R}^{d_{\text{out}}}$, the Jacobian $\partial y / \partial x$ is a $d_{\text{out}} \times d_{\text{in}}$ matrix. Backpropagation never builds it. It needs only the vector-Jacobian product (VJP): given the gradient $g = \partial \mathcal{L} / \partial y$ of a scalar loss, the chain rule gives

$$\frac{\partial \mathcal{L}}{\partial x} = g^\top \frac{\partial y}{\partial x},$$

and for the operations a transformer uses this product has a closed form far cheaper than the matrix. For a linear map $y = xW$ the VJP is $g W^\top$ (and the weight gradient is $x^\top g$); for an element-wise activation $y = \operatorname{act}(x)$ it is $g \odot \operatorname{act}'(x)$; for the softmax it is the formula of {ref}`sec-ch2-softmax-jacobian`, which costs $\mathcal{O}(d)$ instead of the $d^2$ entries of the Jacobian. The softmax over a 50,257-token vocabulary has a Jacobian of about 2.5 billion entries, while its VJP touches 50,257 numbers twice.

```python
import torch

torch.manual_seed(0)
z = torch.randn(6, dtype=torch.float64, requires_grad=True)
g = torch.randn(6, dtype=torch.float64)
p = torch.softmax(z, dim=0)
(vjp_autograd,) = torch.autograd.grad(p, z, grad_outputs=g)
jacobian = torch.autograd.functional.jacobian(lambda t: torch.softmax(t, dim=0), z.detach())
vjp_closed_form = p.detach() * (g - (g * p.detach()).sum())
print("g^T (dp/dz) equals autograd's VJP:", torch.allclose(g @ jacobian, vjp_autograd))
print("closed form p * (g - <g, p>) equals it:", torch.allclose(vjp_closed_form, vjp_autograd))
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

Language modeling is fundamentally a probabilistic task—predicting the likelihood of word sequences. Understanding the mathematical foundations provides insight into why LLMs work and how to improve them.

(sec-ch2-probability-distributions-over-sequences)=

#### Probability Distributions Over Sequences

Given a vocabulary $\mathcal{V}$ and sequence $x = (x_1, x_2, ..., x_n)$, language models estimate:

$$p(x) = \prod_{i=1}^{n} p(x_i \mid x_{<i})$$

This autoregressive factorization enables tractable modeling of exponentially large sequence spaces.

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

For language, entropy measures the average information content per symbol. Shannon's experiments bound the entropy of printed English, for a reader who knows the preceding 100 letters, between 0.6 and 1.3 bits per letter, far below the 4.7 bits of a uniform choice over the 26 letters, which is what makes language predictable {cite}`shannon1951prediction`.

(sec-ch2-cross-entropy-kl-divergence)=

#### Cross-Entropy and KL Divergence

**Cross-entropy** between distributions $p$ and $q$ measures the average number of bits needed to encode data from $p$ using code optimized for $q$:

$$H(p, q) = -\sum_{x} p(x) \log q(x)$$

**Kullback-Leibler (KL) divergence** measures the difference between distributions:

$$D_{KL}(p \,\|\, q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

Training by maximum likelihood is the same as minimizing this divergence from the data distribution. The training loss of {ref}`sec-ch2-cross-entropy-loss` is the cross-entropy $H(p_{\text{data}}, p_\theta)$, so

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim p_{\text{data}}}[\log p_\theta(x)] = H(p_{\text{data}}) + D_{KL}(p_{\text{data}} \,\|\, p_\theta).$$

The entropy of the data does not depend on $\theta$, so any change of $\theta$ that lowers the loss lowers the KL divergence by exactly the same amount, and the loss can never fall below $H(p_{\text{data}})$, which is reached only when $p_\theta = p_{\text{data}}$. For a worked case take $p = (0.5, 0.25, 0.25)$ and $q = (0.25, 0.5, 0.25)$. In bits, $H(p) = 0.5 \cdot 1 + 0.25 \cdot 2 + 0.25 \cdot 2 = 1.5$, $H(p, q) = 0.5 \cdot 2 + 0.25 \cdot 1 + 0.25 \cdot 2 = 1.75$, and $D_{KL}(p \,\|\, q) = 0.5 \log_2 2 + 0.25 \log_2 0.5 + 0 = 0.25 = 1.75 - 1.5$.

(sec-ch2-mutual-information)=

#### Mutual Information

**Mutual information** measures the amount of information shared between variables:

$$\operatorname{MI}(\mathrm{X}; \mathrm{Y}) = \sum_{x, x'} p(x, x') \log \frac{p(x, x')}{p(x)\,p(x')}$$

where $x$ ranges over the values of $\mathrm{X}$ and $x'$ over the values of $\mathrm{Y}$.

Mutual information is the quantity that language modeling implicitly exploits: a context that shares a lot of information with the next token makes that token predictable.

(sec-ch2-temperature)=

#### Temperature

Dividing the logits by a **temperature** $\tau > 0$ before the softmax, $p_i = e^{z_i/\tau} / \sum_j e^{z_j/\tau}$, changes how peaked the distribution is without changing the order of the tokens. As $\tau \to 0$ the distribution approaches a one-hot vector on the largest logit; as $\tau \to \infty$ it approaches uniform. A probability ratio $p_i/p_j = e^{(z_i - z_j)/\tau}$ shows why: the temperature divides every logit gap. {cite:t}`hinton2015distilling` raise the temperature of a teacher's softmax to produce soft targets for distillation; at sampling time a temperature below one makes generation more conservative. The entropy of {ref}`sec-ch2-entropy-language` measures the effect.

```python
import torch

z = torch.tensor([2.0, 1.0, 0.0])
for tau in (0.5, 1.0, 2.0):
    p = torch.softmax(z / tau, dim=0)
    entropy_bits = -(p * torch.log2(p)).sum().item()
    print(f"tau={tau}: p = {[round(v, 4) for v in p.tolist()]}, entropy {entropy_bits:.4g} bits")
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

    D_KL(p || q) = ∑ p(x) log₂(p(x) / q(x)), in bits. Outcomes with p(x) = 0 contribute 0;
    where q(x) = 0 < p(x) the divergence is infinite, and the function returns inf.

    Args:
        p: True distribution [batch_size, num_classes]
        q: Approximate distribution [batch_size, num_classes]

    Returns:
        KL divergence values [batch_size]
    """
    support = p > 0
    ratio = torch.where(support, p / q, torch.ones_like(p))  # p / 0 = inf where q has no mass
    return torch.sum(torch.where(support, p * torch.log2(ratio), torch.zeros_like(p)), dim=-1)


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
print(f"H(p) = {entropy(p).item():.4g} bits, D_KL(p || q) = {kl_divergence(p, q).item():.4g} bits")
loss = cross_entropy_loss(torch.zeros(1, 50257), torch.tensor([0]))  # a model with uniform output
print(f"uniform over 50,257 tokens: loss {loss.item():.4g} nats, perplexity {perplexity(loss).item():,.0f}")
```

% output

```text
H(p) = 1.5 bits, D_KL(p || q) = 0.25 bits
uniform over 50,257 tokens: loss 10.82 nats, perplexity 50,257
```

```{admonition} Hands-On Practice
:class: tip
The notebook section {ref}`nb-math-information-theory` implements entropy, cross-entropy and KL divergence in bits, shows how entropy falls as a softmax sharpens, and computes the perplexity of an untrained model.
```

(sec-ch2-optimization-fundamentals)=

## 2. Optimization Fundamentals

(sec-ch2-gradient-descent-variants)=

### 2.1 Gradient Descent and Variants

Optimization lies at the heart of LLM training. Understanding the mathematical properties of different optimizers explains their behavior during training.

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
per_example = 1.0 + sigma_g * torch.randn(20_000, 64)  # 20,000 draws of 64 per-example gradients
for B in (1, 4, 16, 64):
    g_hat = per_example[:, :B].mean(dim=1)
    print(f"B={B:2d}: std of g_hat {g_hat.std().item():.3f}, predicted sigma_g/sqrt(B) {sigma_g / B ** 0.5:.3f}")
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

At $\eta = 1/\lambda_{\max}$ one step lands on the minimum; between $1/\lambda_{\max}$ and $2/\lambda_{\max}$ the iterates overshoot and oscillate but still shrink; beyond $2/\lambda_{\max}$ each step multiplies the error by more than one in magnitude and training diverges. Directions of smaller curvature converge more slowly at the same $\eta$, which is the conditioning problem of {ref}`sec-ch2-norms-conditioning-scale-matters`. For $\lambda_{\max} = 100$ the limit is $\eta < 0.02$.

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
    print(f"eta={eta}, beta_mom={beta_mom}: |theta| after 200 steps = {final_size(eta, beta_mom):.3g}")
```

% output

```text
eta=0.019, beta_mom=0.0: |theta| after 200 steps = 7.06e-10
eta=0.021, beta_mom=0.0: |theta| after 200 steps = 1.9e+08
eta=0.037, beta_mom=0.9: |theta| after 200 steps = 0.000161
eta=0.039, beta_mom=0.9: |theta| after 200 steps = 2.58e+24
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

{cite:t}`kingma2014adam` designed Adam for "problems with very noisy and/or sparse gradients". A language model's embedding table is one source of sparse gradients: the row of a token that does not occur in a batch receives a zero gradient from that batch, so a rare token's row is updated only now and then, and Adam's per-parameter scaling gives those infrequent updates a usable size.

(sec-ch2-learning-rate-scheduling)=

### 2.2 Learning Rate Scheduling

Effective training requires careful learning rate management:

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

(sec-ch2-gradient-clipping)=

### 2.3 Gradient Clipping

Large language models are susceptible to exploding gradients. **Gradient clipping** prevents this by rescaling gradients:

$$g_{\text{clipped}} = \min\left(1, \frac{g_{\max}}{\|g\|}\right) g$$

where $g_{\max}$ is the clipping threshold (typically 1.0) {cite}`pascanu2013difficulty`.

```python
import math

import torch


class AdamWWithWarmup:
    """Adam with decoupled weight decay (AdamW), linear warm-up, cosine decay to lr_min,
    and global-norm gradient clipping. Written out in full to mirror the equations above;
    in practice use torch.optim.AdamW with a scheduler."""

    def __init__(self, params, lr=1e-3, lr_min=0.0, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, warmup_steps=1000, max_steps=10000, max_grad_norm=1.0):
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
        progress = min((step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 1.0)
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def step(self):
        lr = self.get_lr()
        self.step_count += 1
        t = self.step_count

        # Global-norm clipping across ALL parameters, as in Section 2.3
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)
            # Decoupled weight decay: shrink the weights directly, not through the gradient
            p.mul_(1 - lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-lr)


# The schedule of the warm-up example: peak 3e-4, 2,000 warm-up steps, 100,000 steps in all
opt = AdamWWithWarmup([torch.zeros(1, requires_grad=True)], lr=3e-4, warmup_steps=2000, max_steps=100_000)
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
The notebook section {ref}`nb-math-optimizers` implements SGD, momentum and Adam from scratch (Adam is asserted to match `torch.optim.Adam`), compares them on the Rosenbrock function ({ref}`nb-math-rosenbrock`), and plots warm-up with cosine, inverse-square-root and constant schedules ({ref}`nb-math-schedules`).
```

(sec-ch2-gradient-accumulation)=

### 2.4 Gradient Accumulation

When memory limits the batch size, **gradient accumulation** runs $n_{acc}$ micro-batches of $B$ examples through forward and backward passes before one optimizer step, dividing each micro-batch loss by $n_{acc}$. Gradients add, so the accumulated gradient is the mean over all $n_{acc} B$ examples, the same $\hat{g}$ as one batch of that size ({ref}`sec-ch2-minibatch-noise`), with the noise of the larger batch and the memory of the smaller one. The equality is exact for a loss that is a mean over examples and a model with no batch-dependent layers; batch normalization, which transformers do not use, would break it. Four micro-batches of 8 sequences of 2048 tokens give an update over 65,536 tokens. The demonstration at the end of the listing checks the equality on a one-parameter model.

```python
import torch


def train_with_gradient_accumulation(model, data_loader, optimizer, accumulation_steps=4):
    """Average gradients over `accumulation_steps` micro-batches before each update."""
    model.train()
    optimizer.zero_grad()

    running_loss, num_batches = 0.0, 0
    for i, batch in enumerate(data_loader, start=1):
        # a Hugging Face causal LM returns .loss only when it is given labels
        outputs = model(batch["input_ids"], labels=batch["input_ids"])
        loss = outputs.loss / accumulation_steps  # scale so the accumulated gradient is a mean
        loss.backward()
        running_loss += outputs.loss.item()
        num_batches += 1

        last = i == len(data_loader)
        if i % accumulation_steps == 0 or last:  # do not drop a final partial accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return running_loss / max(num_batches, 1)  # mean per-batch loss, unscaled


# Demonstration: four micro-batches of two examples against one batch of eight
from types import SimpleNamespace


class OneWeight(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.5]))

    def forward(self, input_ids, labels):
        x, y = input_ids.float(), 0.3 * labels.float()
        return SimpleNamespace(loss=((self.w * x - y) ** 2).mean())


ids = torch.tensor([1, 2, 1, 3, 2, 1, 2, 1])
accumulated, full = OneWeight(), OneWeight()
train_with_gradient_accumulation(accumulated, [{"input_ids": ids[i:i + 2]} for i in range(0, 8, 2)],
                                 torch.optim.SGD(accumulated.parameters(), lr=0.1))
train_with_gradient_accumulation(full, [{"input_ids": ids}], torch.optim.SGD(full.parameters(), lr=0.1),
                                 accumulation_steps=1)
print(f"w after 4 accumulated micro-batches: {accumulated.w.item():.4f}")
print(f"w after one batch of 8:              {full.w.item():.4f}")
```

% output

```text
w after 4 accumulated micro-batches: 0.4000
w after one batch of 8:              0.4000
```

(sec-ch2-backpropagation-through-transformers)=

## 3. Backpropagation Through Transformers

(sec-ch2-gradient-flow-deep-networks)=

### 3.1 Gradient Flow in Deep Networks

Understanding how gradients flow through transformer architectures reveals why certain design choices (layer normalization, residual connections) are crucial for training stability.

(sec-ch2-reverse-mode)=

#### Reverse-Mode Automatic Differentiation

Backpropagation is reverse-mode automatic differentiation {cite}`baydin2018automatic`. The forward pass evaluates the network operation by operation and stores every intermediate value it will need later. The backward pass starts from $\partial \mathcal{L} / \partial \mathcal{L} = 1$ and visits the operations in reverse order; each one turns the gradient of its output into gradients of its inputs with its vector-Jacobian product ({ref}`sec-ch2-vjp`), and an input used by several operations adds up the gradients it receives. One backward pass yields the gradient with respect to every parameter, and, in Baydin et al.'s words, "the amount of arithmetic goes up by no more than a small constant factor" over the forward pass. Forward-mode differentiation would instead need one pass per input, which for a model with billions of parameters is out of the question. The price of reverse mode is memory: the stored intermediates are the activations counted in {ref}`sec-ch2-memory-requirements`.

The listing below differentiates $\mathcal{L} = (\tanh(wx) - 0.5)^2$ at $x = 2$, $w = 0.3$ by hand, one stored intermediate at a time, and compares the result with PyTorch.

```python
import math

import torch

x, w_value, target = 2.0, 0.3, 0.5
# forward pass: store every intermediate
a = w_value * x                # 0.6
h = math.tanh(a)               # 0.5370
loss = (h - target) ** 2
# backward pass: one VJP per operation, in reverse order
dL_dh = 2 * (h - target)
dL_da = dL_dh * (1 - h ** 2)   # tanh'(a) = 1 - tanh(a)^2
dL_dw = dL_da * x
w = torch.tensor(w_value, requires_grad=True)
((torch.tanh(w * x) - target) ** 2).backward()
print(f"loss {loss:.4g}; dL/dw by hand {dL_dw:.4g}, by autograd {w.grad.item():.4g}")
```

% output

```text
loss 0.001373; dL/dw by hand 0.1055, by autograd 0.1055
```

(sec-ch2-vanishing-gradient-problem)=

#### The Vanishing Gradient Problem

In deep networks, gradients can become exponentially small:

$$\frac{\partial \mathcal{L}}{\partial \theta^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \prod_{l=2}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}} \frac{\partial h^{(1)}}{\partial \theta^{(1)}}$$

If $\left\|\frac{\partial h^{(l)}}{\partial h^{(l-1)}}\right\| < 1$, gradients vanish as depth increases.

(sec-ch2-residual-connections)=

#### Residual Connections

Residual connections provide direct gradient pathways:

$$h^{(l+1)} = h^{(l)} + F(h^{(l)})$$

The gradient becomes:

$$\frac{\partial h^{(l+1)}}{\partial h^{(l)}} = I + \frac{\partial F(h^{(l)})}{\partial h^{(l)}}$$

The identity term gives every layer a direct path for the gradient, so a deep stack no longer multiplies a long chain of possibly small Jacobians {cite}`he2016deep`. It does not by itself bound the gradient's magnitude: where normalization sits relative to the residual branch matters, and the original post-LN transformer needs learning-rate warm-up precisely because its gradients at initialization are not well behaved, which is why most later models normalize before each sublayer (pre-LN) {cite}`xiong2020layer`.

(sec-ch2-pre-post-ln)=

#### Pre-LN versus Post-LN

The original transformer normalizes after adding the branch (post-LN); most later models normalize the branch's input instead (pre-LN):

$$\text{post-LN: } h^{(l+1)} = \operatorname{LN}\left(h^{(l)} + F(h^{(l)})\right), \qquad \text{pre-LN: } h^{(l+1)} = h^{(l)} + F\left(\operatorname{LN}(h^{(l)})\right).$$

In pre-LN the residual stream is never normalized on its way through the stack, so $\partial h^{(l+1)} / \partial h^{(l)} = I + \partial F(\operatorname{LN}(h^{(l)}))/\partial h^{(l)}$ keeps the identity term of {ref}`sec-ch2-residual-connections`, and the gradient reaching layer $l$ contains the output gradient unchanged. In post-LN every block's output passes through a LayerNorm Jacobian ({ref}`sec-ch2-layer-normalization-mathematics`), which divides by the standard deviation of $h^{(l)} + F(h^{(l)})$, so there is no path on which the gradient passes untouched. {cite:t}`xiong2020layer` make the difference quantitative at initialization: for post-LN "the scale of the gradients to the last FFN layer is of order $\mathcal{O}(d\sqrt{\ln d})$ which is independent of $L$", while for pre-LN it is $\mathcal{O}(d\sqrt{\ln d / L})$, because the residual stream entering the final LayerNorm grows with depth and the normalization divides the gradients of all parameters by about $\sqrt{L}$. For a 24-layer model that factor is $\sqrt{24} = 4.899$: the post-LN gradients near the output are about five times larger relative to pre-LN, which is why post-LN needs warm-up ({ref}`sec-ch2-linear-warmup`) and pre-LN can often do without it.

(sec-ch2-layer-normalization-mathematics)=

### 3.2 Layer Normalization Mathematics

Layer normalization is defined in {ref}`sec-ch1-layernorm`: $y = \gamma \odot \frac{x - \mu}{\sigma} + \beta$, with $\mu$ the mean of the $d$ features and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}$. What matters here is its gradient with respect to the inputs:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(\tilde{g}_i - \frac{1}{d}\sum_j \tilde{g}_j - \hat{x}_i \frac{1}{d}\sum_j \tilde{g}_j \hat{x}_j\right)$$

where $\tilde{g}_j = \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}$ carries the per-feature scale and $\hat{x}_j = (x_j-\mu)/\sigma$ is the normalized input.

The formula follows from the Jacobian of $\hat{x}$. Since $\mu = \frac{1}{d}\sum_i x_i$, $\partial \mu / \partial x_j = 1/d$. Since $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2 + \epsilon$, and the terms that come from differentiating $\mu$ sum to zero because $\sum_i (x_i - \mu) = 0$, $\partial \sigma^2/\partial x_j = \frac{2}{d}(x_j - \mu)$ and so $\partial \sigma / \partial x_j = (x_j - \mu)/(d\sigma) = \hat{x}_j / d$. The quotient rule on $\hat{x}_i = (x_i - \mu)/\sigma$ then gives

$$\frac{\partial \hat{x}_i}{\partial x_j} = \frac{\delta_{ij} - 1/d}{\sigma} - \frac{x_i - \mu}{\sigma^2}\cdot\frac{\hat{x}_j}{d} = \frac{1}{\sigma}\left(\delta_{ij} - \frac{1}{d} - \frac{\hat{x}_i \hat{x}_j}{d}\right).$$

Contracting with $\tilde{g}$, $\frac{\partial \mathcal{L}}{\partial x_j} = \sum_i \tilde{g}_i \frac{\partial \hat{x}_i}{\partial x_j}$, gives the gradient above term by term. Each row of the Jacobian sums to zero, since $\sum_j (\delta_{ij} - 1/d) = 0$ and $\sum_j \hat{x}_j = 0$: adding the same constant to every feature does not change the output, so it receives no gradient. The listing checks the formula against PyTorch.

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
formula = (torch.eye(d, dtype=torch.float64) - 1 / d - torch.outer(x_hat, x_hat) / d) / sigma
autograd = torch.autograd.functional.jacobian(normalize, x)
print("Jacobian formula matches autograd:", torch.allclose(formula, autograd))
print("every row of the Jacobian sums to zero:", torch.allclose(formula.sum(dim=1), torch.zeros(d, dtype=torch.float64)))
```

% output

```text
Jacobian formula matches autograd: True
every row of the Jacobian sums to zero: True
```

{cite:t}`ba2016layer` introduced layer normalization because batch normalization depends "on the mini-batch size and it is not obvious how to apply it to recurrent neural networks". Layer normalization computes its statistics within each token's feature vector, so it behaves the same at any batch size and at every position of a sequence.

(sec-ch2-attention-gradient-analysis)=

### 3.3 Attention Gradient Analysis

The attention mechanism's gradients reveal how learning occurs:

For scaled dot-product attention write the scores, weights and output as

$$S = QK^\top/\sqrt{d_k}, \qquad A = \mathrm{softmax}(S), \qquad O = AV$$

Given the upstream gradient $\frac{\partial \mathcal{L}}{\partial O}$, let $G = \frac{\partial \mathcal{L}}{\partial O} V^\top$ be the gradient with respect to the weights $A$. Backpropagating through the row-wise softmax gives

$$\frac{\partial \mathcal{L}}{\partial S} = A \odot \left(G - (G \odot A)\mathbf{1}\mathbf{1}^\top\right)$$

where $(G \odot A)\mathbf{1}\mathbf{1}^\top$ broadcasts each row's sum of $G \odot A$ across that row. The gradients with respect to the three inputs are then

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S} K, \qquad \frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}}\left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top Q, \qquad \frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}$$

The $A \odot (\cdot)$ factor is why saturated attention (one weight near 1) passes almost no gradient to the other positions, and why the scores are divided by $\sqrt{d_k}$ ({ref}`sec-ch2-sqrt-dk`).

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
print("dQ, dK, dV match autograd:", all(torch.allclose(a, b) for a, b in zip(ours, reference)))
```

% output

```text
dQ, dK, dV match autograd: True
```

(sec-ch2-ffn-embedding-head-gradients)=

### 3.4 Gradients of the Feed-Forward Layer, Embedding and Output Head

The remaining pieces of a transformer are a feed-forward layer, the embedding lookup and the output head, and each has a short backward rule built from the VJPs of {ref}`sec-ch2-vjp`.

**Feed-forward layer.** For one position, $z_{ff} = x W_{up}$ and $y = \operatorname{act}(z_{ff})\,W_{down}$. Given $g = \partial \mathcal{L}/\partial y$, the backward pass runs the forward pass in reverse:

$$\frac{\partial \mathcal{L}}{\partial W_{down}} = \operatorname{act}(z_{ff})^\top g, \qquad \frac{\partial \mathcal{L}}{\partial z_{ff}} = \left(g W_{down}^\top\right) \odot \operatorname{act}'(z_{ff}), \qquad \frac{\partial \mathcal{L}}{\partial W_{up}} = x^\top \frac{\partial \mathcal{L}}{\partial z_{ff}}, \qquad \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial z_{ff}} W_{up}^\top.$$

For a sequence the same products are taken with one row per position, and the weight gradients sum over the positions.

**Embedding.** The embedding of token $x_i$ is row $x_i$ of $W_E$, so the gradient of $W_E$ is zero except in the rows of tokens that occur, and the row of a token that occurs at several positions is the sum of $\partial \mathcal{L} / \partial h^{(0)}_i$ over those positions. This is the sparse gradient of {ref}`sec-ch2-adam-optimizer`.

**Output head.** The logits at position $i$ are $z = h_i W_{out}$, and the loss is the mean cross-entropy over $T$ positions. By {ref}`sec-ch2-softmax-jacobian`, $\partial \mathcal{L}/\partial z_j = (p_j - \delta_{j x_i})/T$, where $x_i$ here stands for the target, the token to be predicted at position $i$, and $\partial \mathcal{L}/\partial W_{out} = \sum_i h_i^\top \partial \mathcal{L}/\partial z$, a sum of outer products over positions. With $T = 3$ and a model that gives the target probability 0.2, that logit's gradient is $(0.2 - 1)/3 = -0.2667$.

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
print("rows of W_E with a nonzero gradient:", g_E.abs().sum(dim=1).nonzero().flatten().tolist())
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
The notebook section {ref}`nb-math-backprop` implements scaled dot-product attention and reports its input gradients, and {ref}`nb-math-layernorm` compares gradient norms through twelve linear layers with and without layer normalization, with residual connections, and with a Xavier-scaled initialization, so you can see which of these actually prevents vanishing.
```

(sec-ch2-scaling-laws-compute)=

## 4. Scaling Laws and Compute

(sec-ch2-scaling-hypothesis)=

### 4.1 The Scaling Hypothesis

Empirical observations reveal predictable relationships between model size, compute, and performance. These **scaling laws** guide resource allocation and model design decisions.

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

(sec-ch2-flops)=

#### Training Compute per Token

The compute axis of the scaling laws counts floating-point operations, and nearly all of them are matrix products. A weight matrix with $d_{\text{in}} d_{\text{out}}$ entries, applied to one token's row vector $x$, costs $d_{\text{in}} d_{\text{out}}$ multiply-adds, which is 2 FLOPs per parameter. Summed over every weight matrix, the forward pass costs about $2N$ FLOPs per token. The backward pass ({ref}`sec-ch2-vjp`) needs two products of the same size for each matrix: $g W^\top$, the gradient passed down to the layer's input, and $x^\top g$, the gradient of the weights themselves. That is another $4N$ FLOPs, so one training token costs

$$2N + 4N = 6N \text{ FLOPs}, \qquad C \approx 6ND$$

for $D$ training tokens. {cite:t}`kaplan2020scaling` count the forward pass as $2N + 2 L n d$ add-multiply operations per token, the second term being the attention scores, and drop it because, when $d > n/12$, it is "a relatively small fraction of the total compute"; {ref}`sec-ch2-training-complexity` derives where the crossover lies. They also leave the embedding out of $N$. Biases, normalization, the softmax and the optimizer step add work proportional to $N$ or smaller, not to $N$ times a width, and are left out as well.

For Chinchilla, $N = 70 \times 10^9$ and $D = 1.4 \times 10^{12}$ give $C \approx 5.88 \times 10^{23}$ FLOPs. Turning the rule around answers the planning question: with $D = 20N$ ({ref}`sec-ch2-training-implications`), $C = 6N \cdot 20N = 120N^2$, so a budget $C$ buys a compute-optimal model of $N = \sqrt{C/120}$ parameters.

```python
N, D = 70e9, 1.4e12
C = 6 * N * D
print(f"C = 6ND = {C:.3g} FLOPs = {C / 8.64e19:,.0f} PF-days")
for budget in (1e21, 5.88e23, 1e25):
    N_opt = (budget / 120) ** 0.5
    print(f"budget {budget:.3g} FLOPs: N = {N_opt:.3g} parameters, D = 20N = {20 * N_opt:.3g} tokens")
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

Chinchilla scaling implies that many large models are "undertrained"—better performance could be achieved with smaller models trained on more data. This has significant implications for training efficiency and inference costs.

Because $a_N \approx a_D \approx 0.5$, the compute-optimal ratio $D^*/N^*$ is roughly the same at every budget, and Hoffmann et al.'s Table 3 puts it near 20 tokens per parameter: 20.2 billion tokens for 1 billion parameters, 205.1 billion for 10 billion, 1.5 trillion for 67 billion and 3.7 trillion for 175 billion {cite}`hoffmann2022training`, ratios of 20.2, 20.5, 22.4 and 21.1. Chinchilla itself, 70 billion parameters on 1.4 trillion tokens, sits at 20. With $C \approx 6ND$ ({ref}`sec-ch2-flops`) the rule fixes both sizes from the budget alone. GPT-3's 175 billion parameters would on this rule call for about 3.5 trillion training tokens (Table 3's own projection is 3.7 trillion).

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

At $d = 4096$ attention costs as much as the linear layers only at $n = 24{,}576$ tokens; for GPT-3's $d = 12{,}288$ at its context of 2048 tokens, attention costs $n/(6d) = 2.78\%$ as much as the linear layers, 2.7% of the layer's work. Kaplan et al. count the context term as $2 L n d$ FLOPs per token, half of the $4 L n d$ above, so in their accounting the terms meet at $n = 12d$, which is their condition $d > n/12$ ({ref}`sec-ch2-flops`).

```python
def attention_share(n, d):
    linear, attention = 12 * n * d ** 2, 2 * n ** 2 * d   # multiply-adds per layer and sequence
    return attention / (linear + attention)


for n, d in ((2048, 12288), (8192, 4096), (24576, 4096), (131072, 4096)):
    print(f"n={n:6d}, d={d:5d}: attention is {attention_share(n, d):.1%} of the per-layer work")
```

% output

```text
n=  2048, d=12288: attention is 2.7% of the per-layer work
n=  8192, d= 4096: attention is 25.0% of the per-layer work
n= 24576, d= 4096: attention is 50.0% of the per-layer work
n=131072, d= 4096: attention is 84.2% of the per-layer work
```

(sec-ch2-memory-requirements)=

#### Memory Requirements

Memory usage includes:

- **Parameters**: $\approx 12 L d^2 + |\mathcal{V}| d$ where $L$ is the number of layers (each layer holds $4d^2$ attention and $8d^2$ feed-forward weights) and $|\mathcal{V}| d$ is the embedding table
- **Activations**: $\mathcal{O}(B \cdot n \cdot d \cdot L)$ for the residual stream, plus the $\mathcal{O}(B \cdot n_h \cdot n^2 \cdot L)$ attention scores and the $4d$-wide feed-forward intermediates, where $B$ is batch size and $n_h$ the number of heads; without recomputation each layer stores about $B\,n\,d\,(34 + 5\,n_h\,n/d)$ bytes {cite}`korthikanti2022reducing`
- **Optimizer and gradient state**: with mixed precision and Adam, each parameter costs about 16 bytes in total: a 2-byte weight, a 2-byte gradient, a 4-byte fp32 master copy and two 4-byte Adam moments {cite}`rajbhandari2020zero`
- **Compute**: about $6N$ FLOPs per training token (2 per parameter for the forward pass, 4 for the backward), so $C \approx 6ND$ for $D$ tokens {cite}`kaplan2020scaling`, as derived in {ref}`sec-ch2-flops`. Kaplan et al. count only non-embedding parameters; the function below uses the total, which for GPT-3 is 0.36% more

```python
def estimate_memory_usage(num_layers, d_model, seq_length, batch_size, vocab_size, num_heads=None):
    """Estimate parameters, memory and compute for mixed-precision Adam training.

    Parameters per layer: 4*d^2 (attention projections) + 8*d^2 (feed-forward, 4d wide).
    Memory per parameter: 2 (fp16 weight) + 2 (fp16 grad) + 4 (fp32 master) + 8 (Adam m, v).
    Activations per layer without recomputation (Korthikanti et al., 2022):
        s*b*h*(34 + 5*a*s/h) bytes, in the paper's symbols: sequence s (n in this chapter),
        batch b (B), hidden size h (d) and heads a (n_h).
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


# Example: the GPT-3 175B configuration (96 layers, d_model 12288, 50257-token vocabulary)
gpt3 = estimate_memory_usage(num_layers=96, d_model=12288, seq_length=2048, batch_size=8, vocab_size=50257)
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

```{admonition} Scaling Laws Visualization
:class: note
The notebook section {ref}`nb-math-scaling` plots the Kaplan parameter, data and compute laws (with the sizes of GPT-2, GPT-3 and Llama 2 70B marked on the parameter curve as extrapolations of the fit, not measured losses) and the Chinchilla allocation derived from $C \approx 6ND$ {cite}`kaplan2020scaling` and $D = 20N$, and {ref}`nb-math-memory` computes parameter counts, training memory and FLOPs per step for the GPT-2 and GPT-3 model sizes, asserting that the 175B configuration comes out at 175B.
```

(sec-ch2-numerical-stability-considerations)=

## 5. Numerical Stability Considerations

(sec-ch2-floating-point-precision)=

### 5.1 Floating Point Precision

Training large models requires careful attention to numerical precision:

(sec-ch2-mixed-precision-training)=

#### Mixed Precision Training

**FP16** (half precision) reduces memory and increases speed but can cause numerical instability. **Mixed precision** {cite}`micikevicius2017mixed` uses FP16 for most operations but FP32 for loss computation and gradient updates.

(sec-ch2-loss-scaling)=

#### Loss Scaling

To prevent gradient underflow in FP16, gradients are scaled up during backpropagation and scaled down before parameter updates:

$$\mathrm{loss}_{\text{scaled}} = \mathrm{loss} \times \zeta$$

where $\zeta$ is the scale factor.

Dynamic loss scaling adjusts the scale factor to maximize gradient precision without overflow.

(sec-ch2-bfloat16)=

#### bfloat16

**bf16** keeps fp32's 8 exponent bits and gives up mantissa bits instead, so it covers the same range as fp32 and needs no loss scaling at all. bf16 is widely used for training on accelerators that support it, usually with an fp32 master copy of the weights; fp16 with loss scaling remains the choice on hardware without bf16 support.

(sec-ch2-initialization-strategies)=

### 5.2 Initialization Strategies

Proper initialization is crucial for stable training:

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

torch.manual_seed(0)
d = 256
x0 = torch.randn(64, d)


def final_variance(blocks, scaled):
    h = x0.clone()
    factor = (2 * blocks) ** -0.5 if scaled else 1.0
    for _ in range(2 * blocks):   # two residual layers per block
        W = torch.randn(d, d) / d ** 0.5 * factor
        h = h + torch.nn.functional.layer_norm(h, (d,)) @ W
    return h.var().item()


for blocks in (12, 48):
    print(f"{blocks} blocks: variance {final_variance(blocks, False):.3g} unscaled, "
          f"{final_variance(blocks, True):.3g} with 1/sqrt(2L)")
```

% output

```text
12 blocks: variance 24.8 unscaled, 2 with 1/sqrt(2L)
48 blocks: variance 94.9 unscaled, 2.02 with 1/sqrt(2L)
```

```{admonition} Numerical Stability Demonstrations
:class: tip
The notebook section {ref}`nb-math-numerics` shows fp16 underflow (and bf16 surviving it), clips a global gradient norm and plots before/after, compares Xavier and He initialization statistics, and {ref}`nb-math-loss-scaling` simulates dynamic loss scaling.
```

(sec-ch2-exercises)=

## Exercises

1. **Information Theory**: Calculate the entropy of character distributions in different languages. How does this relate to compression ratios?

1. **Optimization Comparison**: Implement SGD, Adam, and AdamW optimizers from scratch. Compare their convergence on a small language modeling task.

1. **Scaling Law Verification**: Train models of different sizes and verify parameter scaling laws on a simplified dataset.

1. **Gradient Analysis**: Implement gradient flow visualization through a small transformer to understand how residual connections affect learning.

1. **Numerical Stability**: Experiment with different initialization schemes and observe their effects on training stability.

(sec-ch2-further-reading)=

## Further Reading

(sec-ch2-further-reading-learning-path-beginners)=

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/?topic=neural-networks) - Visual explanations of backpropagation and gradient descent
1. [Distill: Momentum](https://distill.pub/2017/momentum/) - Interactive visualization of optimization algorithms
1. [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) - Essential calculus for understanding backpropagation

**Mathematical Foundations:**

4. [Mathematics for Machine Learning](https://mml-book.github.io/) - Free textbook covering linear algebra, calculus, and probability
1. [MIT 18.065: Matrix Methods in Data Analysis](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) - Gilbert Strang's course "Matrix Methods in Data Analysis, Signal Processing, and Machine Learning\"

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
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) - Practical learning rate scheduling
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
- [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701) - Extensions beyond language
- [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486) - Data efficiency perspectives

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
