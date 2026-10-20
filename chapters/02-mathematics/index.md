# Mathematical Foundations

The remarkable capabilities of Large Language Models emerge from sophisticated mathematical foundations spanning linear algebra, probability theory, information theory, and optimization. This chapter establishes the mathematical framework necessary to understand transformer architectures, training dynamics, and the scaling laws that govern LLM performance.

## Learning Objectives

By the end of this chapter, you will understand:

- Probability and information theory concepts underlying language modeling
- Optimization fundamentals and the mathematics of gradient descent
- How backpropagation flows through transformer architectures
- Scaling laws that predict LLM performance and compute requirements
- Numerical stability considerations in large-scale training

## Chapter Overview

The mathematics underlying Large Language Models can be understood through several interconnected concepts that build upon each other. A short linear-algebra primer (Section 0) fixes notation first.

**Probability and Information Theory** form the foundation of how LLMs understand and generate text. Think of it like predicting the next word in a sentence - the model assigns probabilities to different possibilities based on what it has learned. Information theory measures how "surprising" or "informative" each word choice is, helping the model learn which predictions are better.

**Optimization** is the process by which LLMs learn from data. Imagine adjusting thousands of knobs to minimize errors - that's essentially what gradient descent does. The model makes predictions, measures how wrong they are, and gradually adjusts its internal parameters to improve. Advanced optimizers like Adam are like smart assistants that remember which knobs to turn faster and which to adjust more carefully.

**Backpropagation** is the mechanism that tells the model how to improve. When the model makes a mistake, backpropagation traces the error backwards through all the layers, determining exactly how much each parameter contributed to the error. It's like analyzing a recipe gone wrong by working backwards from the taste to identify which ingredient amounts need adjustment.

**Scaling Laws** reveal predictable relationships between model size, training data, and performance. These mathematical patterns help researchers determine whether building a bigger model or gathering more data will be more effective. They're like economic laws that govern the "returns on investment" for computational resources.

**Numerical Stability** addresses practical challenges in training massive models with billions of parameters. Just as engineers must account for physical limitations when building bridges, ML practitioners must handle the computer's finite precision when performing trillions of calculations. Techniques like mixed precision training and gradient clipping prevent the mathematical equivalent of structural failure.

These mathematical principles work together to enable LLMs to learn patterns from vast amounts of text and generate coherent, contextually appropriate responses.

## Hands-On Learning

For interactive implementations and visualizations of the mathematical concepts in this chapter, see the accompanying Jupyter notebook: [Mathematical Foundations Demo](examples/mathematical_foundations_demo.ipynb)

The notebook provides executable, assertion-checked code for:

- Entropy, cross-entropy and KL divergence in bits ({ref}`nb-math-information-theory`)
- SGD, momentum and Adam from scratch, checked against `torch.optim`, on the Rosenbrock function ({ref}`nb-math-optimizers`)
- Warm-up and decay schedules ({ref}`nb-math-schedules`)
- Attention gradients, layer normalization, residual connections and initialisation scale ({ref}`nb-math-backprop`)
- Kaplan and Chinchilla scaling laws, parameter counts and training memory ({ref}`nb-math-scaling`)
- fp16/bf16 precision, loss scaling, gradient clipping and initialisation ({ref}`nb-math-numerics`)

## 0. Linear Algebra You Need

Everything a transformer does is a sequence of matrix products followed by simple element-wise functions. Three facts carry most of the weight.

### 0.1 A matrix is a map between spaces

A weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ turns a row vector $x \in \mathbb{R}^{d_{\text{in}}}$ into $xW \in \mathbb{R}^{d_{\text{out}}}$. Stacking $n$ inputs as the rows of $X \in \mathbb{R}^{n \times d_{\text{in}}}$ applies the same map to all of them at once, $XW$, which is why a batch of tokens costs one matrix product rather than $n$ separate ones. A query-key score matrix $QK^T \in \mathbb{R}^{n \times n}$ is the table of all pairwise dot products $q_i \cdot k_j$; the $\sqrt{d_k}$ scaling keeps those dot products at unit variance when $q$ and $k$ have unit-variance entries, since a sum of $d_k$ such products has variance $d_k$.

### 0.2 The softmax Jacobian

For $p = \mathrm{softmax}(z)$, $p_i = e^{z_i}/\sum_j e^{z_j}$,

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \qquad \text{so for an upstream gradient } g: \quad \frac{\partial \mathcal{L}}{\partial z} = p \odot \left(g - \langle g, p\rangle \mathbf{1}\right)$$

This is the identity behind the attention gradient in Section 3.3 and behind cross-entropy's gradient $p - y$ (take $g = -y/p$ for a one-hot target $y$). It also explains saturation: if one $p_i \approx 1$ then every factor $p_i(\delta_{ij} - p_j)$ is near zero, and no gradient flows through that softmax.

### 0.3 Norms, conditioning and why scale matters

The operator norm $\|W\|_2$ is the largest factor by which $W$ can stretch a vector. A stack of $L$ layers multiplies these factors, which is why gradients vanish or explode geometrically with depth unless each layer is kept near norm 1, by initialization scale (Section 5.2), by normalization (Section 3.2) or by residual connections (Section 3.1). The Chapter 2 notebook measures all three on a twelve-layer stack.

## 1. Probability and Information Theory

### 1.1 Foundations of Language Modeling

Language modeling is fundamentally a probabilistic task—predicting the likelihood of word sequences. Understanding the mathematical foundations provides insight into why LLMs work and how to improve them.

#### Probability Distributions Over Sequences

Given a vocabulary $\mathcal{V}$ and sequence $x = (x_1, x_2, ..., x_n)$, language models estimate:

$$P(x) = \prod_{i=1}^{n} P(x_i | x_{<i})$$

This autoregressive factorization enables tractable modeling of exponentially large sequence spaces.

#### Cross-Entropy Loss

The standard training objective minimizes cross-entropy between the true data distribution $P_{\text{data}}$ and model distribution $P_{\theta}$:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim P_{\text{data}}}[\log P_{\theta}(x)]$$

For discrete sequences, this becomes:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log P_{\theta}(x_i | x_{<i})$$

Cross-entropy loss has favorable properties: it's convex in the output probabilities and provides strong gradients for incorrect predictions {cite}`murphy2012machine`.

### 1.2 Information Theory Fundamentals

#### Entropy and Language

**Shannon entropy** quantifies the uncertainty in a probability distribution:

$$H(X) = -\sum_{x} P(x) \log P(x)$$

Using $\log_2$ gives entropy in bits; the natural logarithm gives nats. The code in this chapter uses natural logarithms, so its values are in nats.

For language, entropy measures the average information content per symbol. Shannon's experiments put the entropy of printed English at roughly 0.6 to 1.3 bits per character, far below the ~4.7 bits of a uniform choice over the alphabet, which is what makes language predictable {cite}`shannon1951prediction`.

#### Cross-Entropy and KL Divergence

**Cross-entropy** between distributions $P$ and $Q$ measures the average number of bits needed to encode data from $P$ using code optimized for $Q$:

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

**Kullback-Leibler (KL) divergence** measures the difference between distributions:

$$D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

Minimizing cross-entropy loss is equivalent to minimizing KL divergence between data and model distributions.

#### Mutual Information

**Mutual information** measures the amount of information shared between variables:

$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

Mutual information is the quantity that language modeling implicitly exploits: a context that shares a lot of information with the next token makes that token predictable.

### 1.3 Practical Applications

```python
import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss between model predictions and targets.

    Args:
        logits: Model outputs [batch_size, vocab_size]
        targets: True token indices [batch_size]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return F.nll_loss(log_probs, targets)


def perplexity(loss):
    """Convert cross-entropy loss to perplexity."""
    return torch.exp(loss)


def entropy(probs):
    """Compute entropy of probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)


def kl_divergence(p, q):
    """Compute KL divergence between distributions p and q."""
    return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=-1)
```

```{admonition} Hands-On Practice
:class: tip
The notebook section {ref}`nb-math-information-theory` implements entropy, cross-entropy and KL divergence in bits, shows how entropy falls as a softmax sharpens, and computes the perplexity of an untrained model.
```

## 2. Optimization Fundamentals

### 2.1 Gradient Descent and Variants

Optimization lies at the heart of LLM training. Understanding the mathematical properties of different optimizers explains their behavior during training.

#### Stochastic Gradient Descent

The basic SGD update rule:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

where $\eta$ is the learning rate and $\nabla_\theta \mathcal{L}$ is the gradient of the loss function.

#### Momentum

Momentum accelerates optimization by accumulating gradients:

$$v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t+1}$$

The momentum parameter $\beta$ (typically 0.9) controls how much previous gradients influence the current update {cite}`polyak1964some`.

#### Adam Optimizer

Adam {cite}`kingma2014adam` adapts learning rates per parameter using first and second moment estimates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Adam's adaptive learning rates work particularly well for sparse gradients common in NLP.

### 2.2 Learning Rate Scheduling

Effective training requires careful learning rate management:

#### Linear Warmup

Many LLM training schedules use linear warmup followed by decay:

$$\eta_t = \begin{cases}
\eta_{\max} \cdot \frac{t+1}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\max} \cdot \text{decay}(t) & \text{otherwise}
\end{cases}$$

Warmup prevents large gradient updates early in training when parameters are randomly initialized.

#### Cosine Annealing

Cosine schedules provide smooth learning rate decay:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}} \pi\right)\right)$$

### 2.3 Gradient Clipping

Large language models are susceptible to exploding gradients. **Gradient clipping** prevents this by rescaling gradients:

$$g_{\text{clipped}} = \min\left(1, \frac{\tau}{||g||}\right) g$$

where $\tau$ is the clipping threshold (typically 1.0) {cite}`pascanu2013difficulty`.

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

## 3. Backpropagation Through Transformers

### 3.1 Gradient Flow in Deep Networks

Understanding how gradients flow through transformer architectures reveals why certain design choices (layer normalization, residual connections) are crucial for training stability.

#### The Vanishing Gradient Problem

In deep networks, gradients can become exponentially small:

$$\frac{\partial \mathcal{L}}{\partial \theta_1} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=2}^{L} \frac{\partial h_l}{\partial h_{l-1}} \frac{\partial h_1}{\partial \theta_1}$$

If $||\frac{\partial h_l}{\partial h_{l-1}}|| < 1$, gradients vanish as depth increases.

#### Residual Connections

Residual connections provide direct gradient pathways:

$$h_{l+1} = h_l + F(h_l)$$

The gradient becomes:

$$\frac{\partial h_{l+1}}{\partial h_l} = I + \frac{\partial F(h_l)}{\partial h_l}$$

The identity term gives every layer a direct path for the gradient, so a deep stack no longer multiplies a long chain of possibly small Jacobians {cite}`he2016deep`. It does not by itself bound the gradient's magnitude: where normalization sits relative to the residual branch matters, and the original post-LN transformer needs learning-rate warm-up precisely because its gradients at initialization are not well behaved, which is why most later models normalize before each sublayer (pre-LN) {cite}`xiong2020layer`.

### 3.2 Layer Normalization Mathematics

Layer normalization {cite}`ba2016layer` normalizes activations across the feature dimension:

$$y = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where:

- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$ (mean)
- $\sigma = \sqrt{\frac{1}{d}\sum_i (x_i-\mu)^2 + \epsilon}$ (standard deviation, with a small $\epsilon$ added for numerical stability)

The gradients with respect to inputs are:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma}\left(g_i - \frac{1}{d}\sum_j g_j - \hat{x}_i \frac{1}{d}\sum_j g_j \hat{x}_j\right)$$

where $g_j = \gamma_j \frac{\partial \mathcal{L}}{\partial y_j}$ carries the per-feature scale and $\hat{x}_j = (x_j-\mu)/\sigma$ is the normalized input.

Layer normalization provides more stable gradients than batch normalization in sequence models.

### 3.3 Attention Gradient Analysis

The attention mechanism's gradients reveal how learning occurs:

For scaled dot-product attention write the scores, weights and output as

$$S = QK^T/\sqrt{d_k}, \qquad P = \mathrm{softmax}(S), \qquad O = PV$$

Given the upstream gradient $\frac{\partial \mathcal{L}}{\partial O}$, let $G = \frac{\partial \mathcal{L}}{\partial O} V^T$ be the gradient with respect to the weights $P$. Backpropagating through the row-wise softmax gives

$$\frac{\partial \mathcal{L}}{\partial S} = P \odot \left(G - (G \odot P)\mathbf{1}\mathbf{1}^T\right)$$

where $(G \odot P)\mathbf{1}\mathbf{1}^T$ broadcasts each row's sum of $G \odot P$ across that row. The gradients with respect to the three inputs are then

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}}\frac{\partial \mathcal{L}}{\partial S} K, \qquad \frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}}\left(\frac{\partial \mathcal{L}}{\partial S}\right)^T Q, \qquad \frac{\partial \mathcal{L}}{\partial V} = P^T \frac{\partial \mathcal{L}}{\partial O}$$

The $P \odot (\cdot)$ factor is why saturated attention (one weight near 1) passes almost no gradient to the other positions; it is also the reason for the $\sqrt{d_k}$ scaling.

```python
import numpy as np
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
    scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
    attn_weights = F.softmax(scores, dim=-1)

    # Backward pass
    grad_V = torch.matmul(attn_weights.transpose(-2, -1), grad_output)
    grad_attn_weights = torch.matmul(grad_output, V.transpose(-2, -1))

    # Softmax gradient
    grad_scores = attn_weights * (
        grad_attn_weights
        - torch.sum(grad_attn_weights * attn_weights, dim=-1, keepdim=True)
    )

    grad_Q = torch.matmul(grad_scores, K) / np.sqrt(d_k)
    grad_K = torch.matmul(grad_scores.transpose(-2, -1), Q) / np.sqrt(d_k)

    return grad_Q, grad_K, grad_V
```

```{admonition} Gradient Flow Analysis
:class: tip
The notebook section {ref}`nb-math-backprop` implements scaled dot-product attention and reports its input gradients, and {ref}`nb-math-layernorm` compares gradient norms through twelve linear layers with and without layer normalization, with residual connections, and with a Xavier-scaled initialisation, so you can see which of these actually prevents vanishing.
```

## 4. Scaling Laws and Compute

### 4.1 The Scaling Hypothesis

Empirical observations reveal predictable relationships between model size, compute, and performance. These **scaling laws** guide resource allocation and model design decisions.

#### Parameter Scaling

The relationship between model parameters $N$ and loss $L$ follows a power law {cite}`kaplan2020scaling`:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

where $N$ counts non-embedding parameters, $N_c \approx 8.8 \times 10^{13}$ is a fitted scale and $\alpha_N \approx 0.076$ for language modeling.

#### Data Scaling

Similarly, performance improves with dataset size $D$:

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

with $\alpha_D \approx 0.095$.

#### Compute Scaling

Training compute also follows a power law. On the compute-efficient frontier (the best loss reachable with a given budget), Kaplan et al. report

$$L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \qquad C_c^{\min} \approx 3.1 \times 10^{8}\ \text{PF-days},\quad \alpha_C^{\min} \approx 0.050$$

where compute is measured in petaflop/s-days (one PF-day is $8.64 \times 10^{19}$ FLOPs).

### 4.2 Chinchilla Scaling Laws

Recent work {cite}`hoffmann2022training` refined scaling laws to find **compute-optimal** training:

For a given compute budget $C$, the optimal model size $N^*$ and dataset size $D^*$ satisfy:

$$N^* \propto C^a, \quad D^* \propto C^b$$

where $a \approx 0.50$ and $b \approx 0.50$, suggesting equal scaling of parameters and data.

#### Implications for Training

Chinchilla scaling implies that many large models are "undertrained"—better performance could be achieved with smaller models trained on more data. This has significant implications for training efficiency and inference costs.

### 4.3 Computational Complexity

#### Training Complexity

Training a transformer requires $O(n \cdot d^2 + n^2 \cdot d)$ operations per layer:

- $O(n \cdot d^2)$: the four attention projections (Q, K, V, output), $4nd^2$ multiply-adds
- $O(n \cdot d^2)$: the feed-forward network with a $4d$ hidden width, $8nd^2$ multiply-adds, about two thirds of the linear-layer cost
- $O(n^2 \cdot d)$: the attention scores and their application to the values

For long sequences, attention dominates; for large models, linear layers dominate.

#### Memory Requirements

Memory usage includes:

- **Parameters**: $\approx 12 L d^2 + V d$ where $L$ is the number of layers (each layer holds $4d^2$ attention and $8d^2$ feed-forward weights) and $V d$ is the embedding table
- **Activations**: $O(B \cdot n \cdot d \cdot L)$ for the residual stream, plus the $O(B \cdot h \cdot n^2 \cdot L)$ attention scores and the $4d$-wide feed-forward intermediates, where $B$ is batch size and $h$ the number of heads; without recomputation each layer stores about $s\,b\,d\,(34 + 5\,h\,s/d)$ bytes for sequence length $s$ {cite}`korthikanti2022reducing`
- **Optimizer and gradient state**: with mixed precision and Adam, each parameter costs about 16 bytes in total: a 2-byte weight, a 2-byte gradient, a 4-byte fp32 master copy and two 4-byte Adam moments

```python
def estimate_memory_usage(num_layers, d_model, seq_length, batch_size, vocab_size, num_heads=None):
    """Estimate parameters, memory and compute for mixed-precision Adam training.

    Parameters per layer: 4*d^2 (attention projections) + 8*d^2 (feed-forward, 4d wide).
    Memory per parameter: 2 (fp16 weight) + 2 (fp16 grad) + 4 (fp32 master) + 8 (Adam m, v).
    Activations per layer without recomputation (Korthikanti et al., 2022):
        s*b*h*(34 + 5*a*s/h) bytes for sequence s, batch b, hidden h, heads a.
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
The notebook section {ref}`nb-math-scaling` plots the Kaplan parameter, data and compute laws (with GPT-2, GPT-3 and LLaMA-70B placed on the parameter curve) and the Chinchilla allocation derived from C = 6ND and D = 20N, and {ref}`nb-math-memory` computes parameter counts, training memory and FLOPs per step for the GPT-2 and GPT-3 model sizes, asserting that the 175B configuration comes out at 175B.
```

## 5. Numerical Stability Considerations

### 5.1 Floating Point Precision

Training large models requires careful attention to numerical precision:

#### Mixed Precision Training

**FP16** (half precision) reduces memory and increases speed but can cause numerical instability. **Mixed precision** {cite}`micikevicius2017mixed` uses FP16 for most operations but FP32 for loss computation and gradient updates.

#### Loss Scaling

To prevent gradient underflow in FP16, gradients are scaled up during backpropagation and scaled down before parameter updates:

$$\mathrm{loss}_{\text{scaled}} = \mathrm{loss} \times s$$

where $s$ is the scale factor.

Dynamic loss scaling adjusts the scale factor to maximize gradient precision without overflow.

#### bfloat16

**bf16** keeps fp32's 8 exponent bits and gives up mantissa bits instead, so it covers the same range as fp32 and needs no loss scaling at all. Nearly every large model trained since 2022 (PaLM, the LLaMA family and their successors) uses bf16 for activations and gradients with an fp32 master copy of the weights; fp16 with loss scaling remains the choice on hardware without bf16 support.

### 5.2 Initialization Strategies

Proper initialization is crucial for stable training:

#### Xavier/Glorot Initialization

For layers with $n_{\text{in}}$ inputs and $n_{\text{out}}$ outputs:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

#### He Initialization

For ReLU activations:

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

#### Transformer-Specific Initialization

Deep residual networks also scale their initialization with depth: Fixup {cite}`zhang2019fixup` shows that with the right depth-dependent scaling of the residual branches a transformer can be trained with no normalization layers at all, and GPT-2 scales the residual-branch weights by $1/\sqrt{2L}$ for the same reason.

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
        outputs = model(batch["input_ids"])
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
The notebook section {ref}`nb-math-numerics` shows fp16 underflow (and bf16 surviving it), clips a global gradient norm and plots before/after, compares Xavier and He initialisation statistics, and {ref}`nb-math-loss-scaling` simulates dynamic loss scaling.
```

## Exercises

1. **Information Theory**: Calculate the entropy of character distributions in different languages. How does this relate to compression ratios?

1. **Optimization Comparison**: Implement SGD, Adam, and AdamW optimizers from scratch. Compare their convergence on a small language modeling task.

1. **Scaling Law Verification**: Train models of different sizes and verify parameter scaling laws on a simplified dataset.

1. **Gradient Analysis**: Implement gradient flow visualization through a small transformer to understand how residual connections affect learning.

1. **Numerical Stability**: Experiment with different initialization schemes and observe their effects on training stability.

## Further Reading

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/?topic=neural-networks) - Visual explanations of backpropagation and gradient descent
1. [Distill: Momentum](https://distill.pub/2017/momentum/) - Interactive visualization of optimization algorithms
1. [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) - Essential calculus for understanding backpropagation

**Mathematical Foundations:**

4. [Mathematics for Machine Learning](https://mml-book.github.io/) - Free textbook covering linear algebra, calculus, and probability
1. [MIT 18.065: Matrix Methods in Data Analysis](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) - Gilbert Strang's course on mathematical foundations

### Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite}`vaswani2017attention`: The original transformer paper with mathematical details ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762))
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) {cite}`kaplan2020scaling`: Mathematical analysis of scaling behavior ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361))
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) {cite}`hoffmann2022training`: Chinchilla scaling laws ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556))

### Optimization Theory

**Optimizer Papers:**

- {cite}`kingma2014adam`: Adam optimizer derivation and analysis ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980))
- {cite}`loshchilov2017decoupled`: AdamW and weight decay regularization ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101))
- {cite}`polyak1964some`: Classical momentum optimization ([DOI](<https://doi.org/10.1016/0041-5553(64)90137-5>))

**Learning Rate and Initialization:**

- {cite}`zhang2019fixup`: Advanced initialization strategies without normalization ([arXiv:1901.09321](https://arxiv.org/abs/1901.09321))
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) - Practical learning rate scheduling
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) {cite}`pascanu2013difficulty` - Gradient flow analysis

### Numerical Methods and Efficiency

- {cite}`micikevicius2017mixed`: Mixed precision training techniques ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740))
- {cite}`you2019large`: Large batch training and scaling ([arXiv:1904.00962](https://arxiv.org/abs/1904.00962))
- {cite}`shazeer2020glu`: Activation function analysis and GLU variants ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202))

### Mathematical Textbooks and References

- {cite}`goodfellow2016deep`: Deep learning mathematical foundations - [Online Book](https://www.deeplearningbook.org/)
- {cite}`murphy2012machine`: Comprehensive machine learning mathematics textbook
- {cite}`mackay2003information`: Information theory perspective on learning - [Online Book](http://www.inference.org.uk/mackay/itila/)

### Scaling Laws and Empirical Analysis

- {cite}`hestness2017deep`: Early observations of scaling behavior ([arXiv:1712.00409](https://arxiv.org/abs/1712.00409))
- [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701) - Extensions beyond language
- [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486) - Data efficiency perspectives

______________________________________________________________________

*Part of the LLM Atlas - A Comprehensive Guide to Large Language Models*
