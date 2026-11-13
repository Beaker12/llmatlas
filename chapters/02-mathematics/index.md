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

Minimizing cross-entropy loss is equivalent to minimizing KL divergence between data and model distributions.

(sec-ch2-mutual-information)=

#### Mutual Information

**Mutual information** measures the amount of information shared between variables:

$$\operatorname{MI}(\mathrm{X}; \mathrm{Y}) = \sum_{x, x'} p(x, x') \log \frac{p(x, x')}{p(x)\,p(x')}$$

where $x$ ranges over the values of $\mathrm{X}$ and $x'$ over the values of $\mathrm{Y}$.

Mutual information is the quantity that language modeling implicitly exploits: a context that shares a lot of information with the next token makes that token predictable.

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

(sec-ch2-momentum)=

#### Momentum

Momentum accelerates optimization by accumulating gradients:

$$u_{t+1} = \beta_{\text{mom}} u_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta u_{t+1}$$

The momentum coefficient $\beta_{\text{mom}}$ (typically 0.9) controls how much previous gradients influence the current update {cite}`polyak1964some`. On a quadratic whose Hessian has largest eigenvalue $\lambda_{\max}$, plain gradient descent converges for $\eta < 2/\lambda_{\max}$ and momentum for $\eta < 2(1+\beta_{\text{mom}})/\lambda_{\max}$ {cite}`goh2017momentum`; at $\beta_{\text{mom}} = 0.9$ the second limit is 1.9 times the first.

(sec-ch2-adam-optimizer)=

#### Adam Optimizer

Adam {cite}`kingma2014adam` adapts learning rates per parameter using first and second moment estimates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

AdamW {cite}`loshchilov2017decoupled`, the variant implemented below, decouples weight decay from this update: it shrinks the weights directly by $\eta \lambda \theta_t$ instead of adding $\lambda \theta_t$ to the gradient, where it would be divided by $\sqrt{\hat{v}_t}$ like everything else.

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
```

```{admonition} Interactive Exploration
:class: note
The notebook section {ref}`nb-math-optimizers` implements SGD, momentum and Adam from scratch (Adam is asserted to match `torch.optim.Adam`), compares them on the Rosenbrock function ({ref}`nb-math-rosenbrock`), and plots warm-up with cosine, inverse-square-root and constant schedules ({ref}`nb-math-schedules`).
```

(sec-ch2-backpropagation-through-transformers)=

## 3. Backpropagation Through Transformers

(sec-ch2-gradient-flow-deep-networks)=

### 3.1 Gradient Flow in Deep Networks

Understanding how gradients flow through transformer architectures reveals why certain design choices (layer normalization, residual connections) are crucial for training stability.

(sec-ch2-vanishing-gradient-problem)=

#### The Vanishing Gradient Problem

In deep networks, gradients can become exponentially small:

$$\frac{\partial \mathcal{L}}{\partial \theta^{(1)}} = \frac{\partial \mathcal{L}}{\partial h^{(L)}} \prod_{l=2}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}} \frac{\partial h^{(1)}}{\partial \theta^{(1)}}$$

If $\left\|\frac{\partial h^{(l)}}{\partial h^{(l-1)}}\right\| < 1$, gradients vanish as depth increases.

(sec-ch2-pre-post-ln)=

#### Residual Connections

Residual connections provide direct gradient pathways:

$$h^{(l+1)} = h^{(l)} + F(h^{(l)})$$

The gradient becomes:

$$\frac{\partial h^{(l+1)}}{\partial h^{(l)}} = I + \frac{\partial F(h^{(l)})}{\partial h^{(l)}}$$

The identity term gives every layer a direct path for the gradient, so a deep stack no longer multiplies a long chain of possibly small Jacobians {cite}`he2016deep`. It does not by itself bound the gradient's magnitude: where normalization sits relative to the residual branch matters, and the original post-LN transformer needs learning-rate warm-up precisely because its gradients at initialization are not well behaved, which is why most later models normalize before each sublayer (pre-LN) {cite}`xiong2020layer`.

(sec-ch2-layer-normalization-mathematics)=

### 3.2 Layer Normalization Mathematics

Layer normalization is defined in {ref}`sec-ch1-layernorm`: $y = \gamma \odot \frac{x - \mu}{\sigma} + \beta$, with $\mu$ the mean of the $d$ features and $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}$. What matters here is its gradient with respect to the inputs:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(\tilde{g}_i - \frac{1}{d}\sum_j \tilde{g}_j - \hat{x}_i \frac{1}{d}\sum_j \tilde{g}_j \hat{x}_j\right)$$

where $\tilde{g}_j = \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}$ carries the per-feature scale and $\hat{x}_j = (x_j-\mu)/\sigma$ is the normalized input.

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

(sec-ch2-computational-complexity)=

### 4.3 Computational Complexity

(sec-ch2-flops)=

#### Training Complexity

Training a transformer requires $\mathcal{O}(n \cdot d^2 + n^2 \cdot d)$ operations per layer:

- $\mathcal{O}(n \cdot d^2)$: the four attention projections (Q, K, V, output), $4nd^2$ multiply-adds
- $\mathcal{O}(n \cdot d^2)$: the feed-forward network with a $4d$ hidden width, $8nd^2$ multiply-adds, about two thirds of the linear-layer cost
- $\mathcal{O}(n^2 \cdot d)$: the attention scores and their application to the values

For long sequences, attention dominates; for large models, linear layers dominate.

(sec-ch2-memory-requirements)=

#### Memory Requirements

Memory usage includes:

- **Parameters**: $\approx 12 L d^2 + |\mathcal{V}| d$ where $L$ is the number of layers (each layer holds $4d^2$ attention and $8d^2$ feed-forward weights) and $|\mathcal{V}| d$ is the embedding table
- **Activations**: $\mathcal{O}(B \cdot n \cdot d \cdot L)$ for the residual stream, plus the $\mathcal{O}(B \cdot n_h \cdot n^2 \cdot L)$ attention scores and the $4d$-wide feed-forward intermediates, where $B$ is batch size and $n_h$ the number of heads; without recomputation each layer stores about $B\,n\,d\,(34 + 5\,n_h\,n/d)$ bytes {cite}`korthikanti2022reducing`
- **Optimizer and gradient state**: with mixed precision and Adam, each parameter costs about 16 bytes in total: a 2-byte weight, a 2-byte gradient, a 4-byte fp32 master copy and two 4-byte Adam moments {cite}`rajbhandari2020zero`
- **Compute**: about $6N$ FLOPs per training token (2 per parameter for the forward pass, 4 for the backward), so $C \approx 6ND$ for $D$ tokens {cite}`kaplan2020scaling`. Kaplan et al. count only non-embedding parameters; the function below uses the total, which for GPT-3 is 0.36% more

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

(sec-ch2-gradient-accumulation)=

### 5.3 Gradient Accumulation

When memory constraints limit batch size, **gradient accumulation** simulates larger batches:

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
