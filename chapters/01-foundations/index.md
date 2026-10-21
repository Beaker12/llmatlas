# LLM Foundations

Large Language Models represent a paradigm shift in natural language processing, achieving unprecedented capabilities through the transformer architecture and massive scale training. This chapter establishes the foundational concepts necessary to understand how these models work, their historical development, and the core principles that enable their remarkable performance.

## Learning Objectives

By the end of this chapter, you will understand:

- How tokenization transforms text into numerical representations
- The mechanics of attention mechanisms and their role in language understanding
- The transformer architecture and its key components
- The language modeling objective and evaluation metrics
- The historical evolution from RNNs to modern transformers

## Hands-On Learning

For interactive implementations and visualizations of the foundational concepts in this chapter, see the accompanying Jupyter notebook: [Attention Mechanisms Demo](examples/attention_mechanisms_demo.ipynb)

The notebook provides executable, assertion-checked code for:

- A word-level tokenizer and the BPE merge loop ({ref}`nb-attn-tokenization`)
- Token embeddings with sinusoidal, learned, relative-bias and rotary positions ({ref}`nb-attn-positional`)
- Scaled dot-product and multi-head attention with attention maps ({ref}`nb-attn-attention`)
- A post-LN transformer block ({ref}`nb-attn-transformer`)
- The causal mask and the next-token objective with perplexity ({ref}`nb-attn-lm`)

## Chapter Overview

Large Language Models (LLMs) like ChatGPT work by understanding and generating text through several key processes that can be understood through everyday analogies:

**Tokenization** is like breaking down a sentence into meaningful chunks - imagine reading a book where you recognize not just individual letters, but common word parts, whole words, and phrases. LLMs break text into "tokens" (pieces like "un-", "happy", "ness") that help them understand meaning more efficiently than processing character by character.

**Embeddings** convert these text pieces into numerical representations - think of it like giving each word or word-part a unique mathematical "fingerprint" that captures its meaning. Similar words get similar fingerprints, allowing the model to understand that "happy" and "joyful" are related concepts.

**Attention Mechanisms** are like a sophisticated highlighting system that helps the model focus on the most relevant parts of text when making predictions. When processing "The cat sat on the mat," the attention mechanism helps the model understand that "cat" is the subject performing the action "sat," similar to how you naturally emphasize important words when reading.

**Transformers** are the overall architecture that combines all these elements - like a highly efficient assembly line where multiple "workers" (attention heads) simultaneously analyze different aspects of the text (grammar, meaning, context) and combine their insights to understand and generate language.

**Language Modeling** is the training process where the model learns to predict the next word in a sentence by reading millions of examples. It's like learning to complete sentences by reading vast amounts of text and developing an intuition for how language flows naturally.

These components work together to create models that can understand context, maintain coherent conversations, and generate human-like text by learning statistical patterns from enormous amounts of text data.

## 1. Tokenization and Embeddings

### 1.1 Understanding Tokenization

Large Language Models cannot directly process raw text—they require numerical representations. The process begins with **tokenization**, which breaks text into discrete units called tokens. This seemingly simple step profoundly affects model performance, computational efficiency, and language understanding capabilities.

#### What Are Tokens?

A **token** is the fundamental unit of text that a language model processes. Tokens can represent:

- **Complete words**: "hello", "world", "artificial"
- **Subwords**: "hell", "o", "artific", "ial"
- **Characters**: "h", "e", "l", "l", "o"
- **Bytes**: Individual byte values (0-255)
- **Special symbols**: `<BOS>`, `<EOS>`, `<UNK>`, `<PAD>`

The choice of tokenization granularity creates fundamental trade-offs:

| Granularity | Vocabulary Size    | Sequence Length | OOV Handling | Semantic Coherence |
| ----------- | ------------------ | --------------- | ------------ | ------------------ |
| Word-level  | Very Large (>100k) | Short           | Poor         | High               |
| Subword     | 32k-256k           | Medium          | Good         | Medium             |
| Character   | Small (\<100)      | Very Long       | Perfect      | Low                |
| Byte-level  | Fixed (256)        | Longest         | Perfect      | Lowest             |

#### The Vocabulary Size Dilemma

Vocabulary size directly impacts:

1. **Model parameters**: Embedding matrix size = vocab_size × embedding_dim
1. **Computational cost**: Output projection scales with vocabulary size
1. **Memory requirements**: Larger vocabularies require more memory
1. **Out-of-vocabulary handling**: Smaller vocabularies struggle with rare words

**Example**: A 50k vocabulary with 4096-dimensional embeddings requires 200M parameters just for embeddings—before any transformer layers!

### 1.2 Tokenization Algorithms

#### Word-Level Tokenization

The simplest approach splits text on whitespace and punctuation:

```python
def word_tokenize(text):
    import re

    # Split on whitespace and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens


# Example
text = "Hello, world! How are you?"
tokens = word_tokenize(text)
# Output: ['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
```

**Limitations**:

- Enormous vocabularies for real-world text
- Poor handling of morphologically rich languages
- Out-of-vocabulary (OOV) words become `<UNK>` tokens
- Inefficient for languages without clear word boundaries (Chinese, Japanese)

#### Character-Level Tokenization

Processes text character by character:

```python
def char_tokenize(text):
    return list(text)


# Example
text = "Hello"
tokens = char_tokenize(text)
# Output: ['H', 'e', 'l', 'l', 'o']
```

**Advantages**:

- Small, fixed vocabulary
- No OOV problems
- Language-agnostic

**Disadvantages**:

- Very long sequences
- Loss of semantic coherence
- Difficult to learn word-level patterns

#### Byte Pair Encoding (BPE)

BPE {cite}`sennrich2015neural` strikes a balance by iteratively merging the most frequent character pairs:

```python
class SimpleBPE:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def get_pairs(self, word):
        """Count adjacent symbol pairs in a word (a repeated pair counts each time)."""
        pairs = {}
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
        return pairs

    def train(self, corpus, num_merges):
        """Train BPE on a corpus."""
        # Initialize vocabulary with characters
        words = {}
        for word in corpus:
            word_tokens = tuple(word) + ("</w>",)  # End-of-word marker
            words[word_tokens] = words.get(word_tokens, 0) + 1

        vocab = set()
        for word in words:
            vocab.update(word)

        # Iteratively merge most frequent pairs
        for i in range(num_merges):
            pairs = {}
            for word, freq in words.items():
                for pair, n in self.get_pairs(word).items():
                    pairs[pair] = pairs.get(pair, 0) + n * freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            # Merge the best pair in all words
            new_words = {}
            for word, freq in words.items():
                new_word = self.merge_word(word, best_pair)
                new_words[new_word] = freq
            words = new_words

            vocab.add("".join(best_pair))

        self.vocab = vocab

    def merge_word(self, word, pair):
        """Merge a specific pair in a word."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append("".join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def encode(self, text):
        """Encode text using learned BPE."""
        tokens = []
        for word in text.split():
            word_tokens = tuple(word) + ("</w>",)

            # Apply merges in order
            for pair in self.merges:
                word_tokens = self.merge_word(word_tokens, pair)

            tokens.extend(word_tokens)
        return tokens


# Example usage
corpus = ["hello", "world", "hello", "help", "held"]
bpe = SimpleBPE()
bpe.train(corpus, num_merges=10)
tokens = bpe.encode("hello world")
print(f"Tokens: {tokens}")
```

**BPE Evolution**:

1. **Original BPE** {cite}`sennrich2015neural`: merges the most frequent adjacent pair
1. **WordPiece**: Google's variant that merges the pair which most increases the training corpus likelihood
1. **Unigram LM** {cite}`kudo2018subword`: starts from a large candidate vocabulary and prunes it, keeping the segmentation with the highest probability
1. **SentencePiece** {cite}`kudo2018sentencepiece` is a *toolkit*, not an algorithm: it implements both BPE and unigram on raw text without pre-tokenization

#### SentencePiece

SentencePiece {cite}`kudo2018sentencepiece` treats text as raw Unicode sequences without pre-tokenization:

```python
# SentencePiece example (requires sentencepiece library)
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tokenizer",
    vocab_size=32000,
    model_type="bpe",  # or 'unigram'
    character_coverage=0.9995,
    normalization_rule_name="identity",
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

text = "Hello, world! 你好世界"
tokens = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")
```

**Key advantages**:

- Language-agnostic (no pre-tokenization assumptions)
- Handles any Unicode text
- Consistent treatment of whitespace
- Reversible tokenization

```{admonition} Tokenization Implementation
:class: tip
The notebook's tokenization section ({ref}`nb-attn-tokenization`) builds a tiny word-level vocabulary and runs the BPE merge loop step by step, printing the splits after every merge. For WordPiece, unigram and SentencePiece in practice, see {doc}`tokenization-deep-dive`.
```

### 1.3 Embeddings and Positional Encoding

For vocabulary construction, special tokens, cross-lingual issues and a full tokenizer implementation, see {doc}`tokenization-deep-dive`.

#### Embedding Spaces

Once tokenized, each token is mapped to a high-dimensional vector through an **embedding matrix** $E \in \mathbb{R}^{V \times d}$, where $V$ is the vocabulary size and $d$ is the embedding dimension.

$$\mathbf{x}_i = E[\text{token}_i]$$

These embeddings are learned during training to capture semantic relationships. Similar tokens develop similar vector representations, enabling the model to generalize across related concepts {cite}`mikolov2013efficient`.

#### Understanding Embedding Dimensions

The embedding dimension $d$ is a crucial hyperparameter that affects:

**Representational Capacity**: Higher dimensions can encode more information

- 128-256d: Small models, limited expressiveness
- 512-1024d: Medium models, good balance
- 2048-4096d: Large models, rich representations
- 8192d+: Very large models, potential diminishing returns

**Computational Cost**: Embedding operations scale with dimension

- Matrix multiplications: O(d × other_dim)
- Memory usage: V × d parameters just for embeddings
- Attention computation: Affects all transformer operations

#### Embedding Initialization Strategies

Different initialization schemes affect training dynamics:

```python
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, init_strategy="xavier"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Initialize embeddings
        if init_strategy == "xavier":
            nn.init.xavier_uniform_(self.embedding.weight)
        elif init_strategy == "normal":
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        elif init_strategy == "scaled_normal":
            nn.init.normal_(
                self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model)
            )

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) following original transformer
        return self.embedding(x) * math.sqrt(self.d_model)
```

#### Embedding Sharing and Tying

Many architectures share the input embedding matrix with the output projection (GPT-2, T5, Gemma do; the LLaMA and Mistral families keep them separate):

```python
# pseudo-code: `TransformerStack` stands for the model body; illustrates weight tying only
class TransformerWithTiedEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Shared embedding matrix
        self.embeddings = nn.Embedding(vocab_size, d_model)

        # Transformer layers...
        self.transformer = TransformerStack(d_model, num_layers, num_heads)

        # Output projection using transposed embedding matrix
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.output_projection.weight = self.embeddings.weight

    def forward(self, input_ids):
        # Input embeddings
        x = self.embeddings(input_ids)

        # Transform through model
        x = self.transformer(x)

        # Output logits using tied weights
        logits = self.output_projection(x)
        return logits
```

**Benefits of weight tying**:

- Reduces parameters by ~V×d
- Enforces consistency between input and output representations
- Often improves performance on language modeling tasks

#### Subword Embedding Composition

For subword tokens, embeddings must compose meaningfully:

```python
# pseudo-code: `embedding_matrix` and `token_to_id` come from a trained model
# Example: How "unhappiness" might be tokenized and embedded
text = "unhappiness"
tokens = ["un", "happiness"]  # Example BPE tokenization

# Each token gets its own embedding
un_embedding = embedding_matrix[token_to_id["un"]]  # [d_model]
happiness_embedding = embedding_matrix[token_to_id["happiness"]]  # [d_model]

# The model sees these as separate tokens in sequence
# It must learn that "un" is a negation prefix
# This happens through attention and context
```

#### Embedding Visualization and Analysis

```python
# pseudo-code: needs a trained `model`, its `tokenizer`, and scikit-learn
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def visualize_embeddings(embedding_matrix, token_to_id, tokens_to_plot):
    """Visualize token embeddings in 2D using PCA."""

    # Extract embeddings for specified tokens
    embeddings = []
    labels = []

    for token in tokens_to_plot:
        if token in token_to_id:
            token_id = token_to_id[token]
            embedding = embedding_matrix[token_id].detach().numpy()
            embeddings.append(embedding)
            labels.append(token)

    embeddings = np.array(embeddings)

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

    for i, label in enumerate(labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title("Token Embeddings Visualization (PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.show()


# Example usage
semantic_groups = [
    # Colors
    ["red", "blue", "green", "yellow", "orange"],
    # Animals
    ["cat", "dog", "bird", "fish", "horse"],
    # Numbers
    ["one", "two", "three", "four", "five"],
]

for group in semantic_groups:
    visualize_embeddings(model.embeddings.weight, tokenizer.token_to_id, group)
```

#### Positional Encodings

Since transformers lack inherent sequence ordering, positional information must be explicitly encoded. This is crucial because language is inherently sequential—word order fundamentally affects meaning.

**The Position Problem**: Consider these sentences:

- "The cat sat on the mat"
- "The mat sat on the cat"

Without positional information, a transformer would see these as identical sets of tokens.

##### Sinusoidal Positional Encoding

The original transformer uses sinusoidal positional encodings {cite}`vaswani2017attention`:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

where:

- $pos$ is the position in the sequence
- $i$ is the dimension index
- $d$ is the embedding dimension

**Key properties**:

- Deterministic (same encoding for same position)
- Bounded values (between -1 and 1)
- Different frequencies for different dimensions
- Defined for any position, so a model *can* be run past its training length, although in practice quality degrades quickly there

```python
import torch
import math
import matplotlib.pyplot as plt


def create_sinusoidal_encoding(max_seq_len, d_model):
    """Create sinusoidal positional encodings."""
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()

    # Create dimension indices
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )

    # Apply sin to even dimensions
    pe[:, 0::2] = torch.sin(position * div_term)
    # Apply cos to odd dimensions
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def visualize_positional_encoding(max_seq_len=100, d_model=64):
    """Visualize positional encodings."""
    pe = create_sinusoidal_encoding(max_seq_len, d_model)

    plt.figure(figsize=(12, 8))
    plt.imshow(pe.T, cmap="RdBu", aspect="auto")
    plt.colorbar()
    plt.title("Sinusoidal Positional Encodings")
    plt.xlabel("Position")
    plt.ylabel("Dimension")
    plt.show()

    # Show how encoding changes with position for first few dimensions
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.plot(pe[:50, i].numpy(), label=f"dim {i}")
    plt.legend()
    plt.title("Positional Encoding Values by Dimension")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.show()


# Visualize the encodings
visualize_positional_encoding()
```

##### Learned Positional Embeddings

Many models use learned positional embeddings instead:

```python
import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_length, d_model):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        self.max_seq_length = max_seq_length

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings
```


**Learned vs. Sinusoidal**:

| Aspect         | Sinusoidal    | Learned                     |
| -------------- | ------------- | --------------------------- |
| Parameters     | 0 additional  | max_seq_len × d_model       |
| Beyond training length | Defined, but degrades in practice | Undefined (no embedding) |
| Used by        | Original transformer | BERT, GPT-2, GPT-3 |
| Flexibility    | Fixed pattern | Adaptive to data            |

##### Advanced Positional Encodings

**Relative Positional Encoding**: instead of adding absolute positions to the input, make attention depend on the offset $i-j$. Shaw et al. {cite}`shaw2018self` add learned relative-position *vectors* to the keys and values; T5 {cite}`raffel2019exploring` simplifies this to one learned scalar bias $b_{i-j}$ per head added to the scores, which is what the module below implements:

```python
import torch
import torch.nn as nn


class RelativePositionBias(nn.Module):
    """Learned scalar bias b[i-j] per head, added to the attention scores (T5-style).

    One scalar per (head, clipped offset). The table is a module parameter, so it is
    learned; building a fresh nn.Embedding on every call would give random, unlearnable values.
    """

    def __init__(self, num_heads, max_relative_distance=128):
        super().__init__()
        self.max_relative_distance = max_relative_distance
        self.table = nn.Embedding(2 * max_relative_distance + 1, num_heads)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        relative = positions.unsqueeze(1) - positions.unsqueeze(0)           # [seq, seq], entry (i, j) = i - j
        relative = relative.clamp(-self.max_relative_distance, self.max_relative_distance)
        bias = self.table(relative + self.max_relative_distance)             # [seq, seq, heads]
        return bias.permute(2, 0, 1)                                         # [heads, seq, seq]


# Inside attention: scores = Q @ K.transpose(-2, -1) / sqrt(d_k) + bias
bias = RelativePositionBias(num_heads=8)(seq_len=16)
print(bias.shape)  # torch.Size([8, 16, 16])
```


**Rotary Position Embedding (RoPE)** {cite}`su2021roformer`: used by LLaMA, Mistral, Qwen and Gemma. It is applied to the queries and keys *inside* attention, not to the input embeddings:

```python
import torch


def apply_rope(x, positions, base=10000.0):
    """Rotary position embedding (Su et al., 2021), applied to queries and keys.

    x: [..., seq, d]; positions: [seq]. Consecutive pairs of features are rotated by an
    angle m * theta_i, where m is the position and theta_i = base^(-2i/d). Because a rotation
    preserves norms and q_m . k_n then depends only on m - n, RoPE encodes relative position
    without any parameters.
    """
    d = x.size(-1)
    inv_freq = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))   # [d/2]
    angles = positions.float()[:, None] * inv_freq[None, :]           # [seq, d/2]
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# Usage inside attention: rotate q and k, never v, then take the dot products as usual
q = torch.randn(16, 64)
k = torch.randn(16, 64)
positions = torch.arange(16)
scores = apply_rope(q, positions) @ apply_rope(k, positions).T / 64**0.5
```

##### Position Encoding Trade-offs

```python
import matplotlib.pyplot as plt


def compare_position_encodings():
    """Extra parameters each positional scheme adds, as a function of maximum length."""
    seq_lens = [128, 512, 1024, 2048, 4096]
    d_model, num_heads, max_relative_distance = 512, 8, 128

    results = {
        "Sinusoidal": [0 for _ in seq_lens],                                        # no parameters
        "Learned absolute": [n * d_model for n in seq_lens],                        # one vector per position
        "Relative bias": [(2 * max_relative_distance + 1) * num_heads for _ in seq_lens],  # per (offset, head)
        "RoPE": [0 for _ in seq_lens],                                              # no parameters
    }

    plt.figure(figsize=(10, 6))
    for method, params in results.items():
        plt.plot(seq_lens, params, marker="o", label=method)
    plt.xlabel("Maximum sequence length")
    plt.ylabel("Additional parameters")
    plt.title("Positional encoding parameter cost (sinusoidal and RoPE add none)")
    plt.legend()
    plt.grid(True)
    plt.show()


compare_position_encodings()
```


Models since 2022 almost all use RoPE {cite}`su2021roformer` or ALiBi {cite}`press2022train`, both of which encode relative position inside attention without per-position parameters; learned absolute embeddings are now mostly found in BERT- and GPT-2-era models.

```{admonition} Embeddings and Positional Encoding
:class: note
The notebook ({ref}`nb-attn-positional`) visualises sinusoidal and learned encodings, implements a learned relative-position bias and a rotary (RoPE) transform, and asserts RoPE's two defining properties: rotation preserves vector norms, and query-key dot products depend only on the relative offset.
```

#### Advanced Embedding Implementation

```python
import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with multiple strategies."""

    def __init__(self, d_model, max_seq_length=5000, encoding_type="sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            pe = torch.zeros(max_seq_length, d_model)
            position = torch.arange(0, max_seq_length).unsqueeze(1).float()

            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            self.register_buffer("pe", pe.unsqueeze(0))

        elif encoding_type == "learned":
            self.position_embeddings = nn.Embedding(max_seq_length, d_model)

        elif encoding_type == "relative":
            raise ValueError(
                "relative position bias is added to the attention scores, not to the input; "
                "see RelativePositionBias above"
            )

    def forward(self, x, position_ids=None):
        if self.encoding_type == "sinusoidal":
            return x + self.pe[:, : x.size(1)]

        elif self.encoding_type == "learned":
            if position_ids is None:
                position_ids = torch.arange(x.size(1), device=x.device)
            return x + self.position_embeddings(position_ids)

        return x


class TokenEmbeddingLayer(nn.Module):
    """Complete token embedding layer with multiple features."""

    def __init__(
        self,
        vocab_size,
        d_model,
        max_seq_length=5000,
        dropout=0.1,
        position_encoding="sinusoidal",
        layer_norm=True,
        scale_embeddings=True,
    ):
        super().__init__()

        self.d_model = d_model
        self.scale_embeddings = scale_embeddings

        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.position_encoding = PositionalEncoding(
            d_model, max_seq_length, position_encoding
        )

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.token_embeddings.weight)

    def forward(self, input_ids, position_ids=None):
        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Scale embeddings (following original transformer)
        if self.scale_embeddings:
            token_embeds = token_embeds * math.sqrt(self.d_model)

        # Add positional encoding
        embeddings = self.position_encoding(token_embeds, position_ids)

        # Apply layer normalization
        if self.layer_norm:
            embeddings = self.layer_norm(embeddings)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings


# Example usage
vocab_size = 50000
d_model = 512
max_seq_length = 2048

embedding_layer = TokenEmbeddingLayer(
    vocab_size=vocab_size,
    d_model=d_model,
    max_seq_length=max_seq_length,
    position_encoding="sinusoidal",
    layer_norm=True,
    scale_embeddings=True,
)

# Test with sample input
batch_size = 4
seq_length = 256
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

embeddings = embedding_layer(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {embeddings.shape}")
print(f"Embedding norm: {embeddings.norm(dim=-1).mean():.3f}")
```

```{admonition} Hands-On Practice
:class: tip
For runnable versions of the ideas in this section, see the accompanying notebook: [Attention Mechanisms Demo](examples/attention_mechanisms_demo.ipynb). Its sections are linked from each part of this chapter.
```

## 2. Attention Mechanisms

### 2.1 The Attention Revolution

Attention mechanisms revolutionized sequence modeling by allowing models to focus on relevant parts of the input when making predictions. Unlike RNNs that process sequences sequentially, attention enables parallel computation and direct connections between any two positions.

```{admonition} Interactive Exploration
:class: note
The notebook implements scaled dot-product and multi-head attention in eval mode, asserts that every row of attention weights sums to 1, and plots per-head attention maps and entropies ({ref}`nb-attn-attention`).
```

#### Self-Attention Fundamentals

Self-attention computes a weighted representation of the input sequence, where weights indicate the relevance of each position to the current position. Given input embeddings $X \in \mathbb{R}^{n \times d}$:

1. **Query, Key, Value projections**:

   $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. **Attention scores**:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $\sqrt{d_k}$, where $d_k$ is the per-head key dimension, keeps the dot products at unit variance so the softmax does not saturate {cite}`vaswani2017attention`.

#### Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to attend to different types of relationships:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 2.2 Attention Patterns and Interpretability

Different attention heads learn to capture various linguistic phenomena:

- **Syntactic heads**: Focus on grammatical relationships (subject-verb, modifier-noun)
- **Semantic heads**: Capture semantic similarity and coreference
- **Positional heads**: Attend to specific relative positions

Visualization tools like attention maps provide insights into model behavior, though interpreting attention patterns requires caution {cite}`jain2019attention`.

```python
import math

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.W_o(attn_output)
```

## 3. Transformer Architecture

### 3.1 The Complete Architecture

The transformer architecture {cite}`vaswani2017attention` combines attention mechanisms with several key components to create a powerful sequence-to-sequence model.

#### Layer Normalization

Layer normalization stabilizes training by normalizing inputs across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed across the feature dimension (with a small $\epsilon$ added inside the square root of $\sigma$ for numerical stability), and $\gamma$, $\beta$ are learned per-feature scale and shift.

#### Residual Connections

Residual connections help gradient flow in deep networks:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

#### Feed-Forward Networks

Each transformer layer includes a position-wise feed-forward network:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This provides non-linear transformations and typically uses a hidden dimension 4× larger than the model dimension.

### 3.2 Architectural Variants

While the original transformer was designed for machine translation, different variants emerged for language modeling:

- **GPT** {cite}`radford2018improving`: Decoder-only architecture with causal masking
- **BERT** {cite}`devlin2018bert`: Encoder-only architecture with bidirectional attention
- **T5** {cite}`raffel2019exploring`: Encoder-decoder architecture treating all tasks as text-to-text

```python
# pseudo-code: uses MultiHeadAttention from Section 2.2
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

```{admonition} Transformer Architecture Implementation
:class: tip
The notebook ({ref}`nb-attn-transformer`) assembles one post-LN transformer block from multi-head attention, a position-wise feed-forward network, layer normalization and residual connections, and shows what layer normalization does to activations of different scales. It does not train a model; that is left as an exercise.
```

### 3.3 What Changed Since 2017

The block above is the 2017 design. Every widely used LLM since GPT-2 differs from it in a handful of ways, all aimed at training stability or inference cost:

- **Pre-LN instead of post-LN.** Normalization is applied *before* each sublayer, so the residual stream is never normalized: $x \leftarrow x + \mathrm{MHA}(\mathrm{Norm}(x))$. Post-LN transformers have large gradients at initialization and need learning-rate warm-up to train at all; pre-LN removes most of that sensitivity {cite}`xiong2020layer`.
- **RMSNorm instead of LayerNorm.** Dropping the mean subtraction and the bias, $\mathrm{RMSNorm}(x) = \gamma \odot x / \sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}$, is cheaper and trains just as well {cite}`zhang2019root`. LLaMA, Mistral, Qwen and Gemma all use it.
- **Gated activations.** The feed-forward network becomes $\mathrm{FFN}(x) = (\mathrm{SiLU}(xW_1) \odot xW_3)W_2$ (SwiGLU), with the hidden width reduced to about $\tfrac{8}{3}d$ to keep the parameter count unchanged {cite}`shazeer2020glu`.
- **Rotary position embeddings.** Positions enter through RoPE on the queries and keys (Section 1.3), not through vectors added to the input {cite}`su2021roformer`.
- **Grouped-query attention.** Several query heads share one key/value head, shrinking the KV cache that dominates inference memory at long context {cite}`ainslie2023gqa`.
- **No biases, tied or untied embeddings, larger vocabularies.** Linear layers usually drop their bias terms; vocabularies grew from 32k to 128k-256k (see {doc}`tokenization-deep-dive`, T.4).

Chapter 6 ({doc}`../06-architectures/index`) takes each of these in turn, adds multi-head latent attention, sliding-window attention and mixture-of-experts layers, and counts the parameters and cache of real models.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or int(8 * d_model / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PreLNBlock(nn.Module):
    """A LLaMA-style block: pre-norm, RMSNorm, SwiGLU. Attention is any module with
    forward(x, x, x) -> output (RoPE and grouped-query attention live inside it)."""

    def __init__(self, d_model, attention):
        super().__init__()
        self.attention = attention
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attention(h, h, h)
        return x + self.ffn(self.norm2(x))
```

## 4. Language Modeling Objective

### 4.1 Next-Token Prediction

Modern LLMs are trained on the **causal language modeling** objective, predicting the next token given previous tokens:

$$P(x_{t+1} | x_1, x_2, ..., x_t)$$

This autoregressive formulation enables text generation and captures the statistical structure of language.

#### Causal Masking

To prevent the model from "seeing the future," causal masking ensures attention only flows from earlier to later positions:

```python
import torch

def create_causal_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0  # True for allowed positions
```

### 4.2 Training Process

The training objective maximizes the likelihood of the training data:

$$\mathcal{L} = -\sum_{i=1}^{N} \log P(x_i | x_{<i})$$

Where the model outputs probability distributions over the vocabulary:

$$P(x_i | x_{<i}) = \text{softmax}(W_{\text{out}} h_{i-1} + b_{\text{out}})$$

where $h_{i-1}$ is the final-layer hidden state at position $i-1$, which under causal masking has seen exactly $x_{<i}$.

### 4.3 Evaluation Metrics

**Perplexity** measures how well the model predicts the test data:

$$\text{Perplexity} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

Perplexity was introduced as the effective branching factor of a language model {cite}`jelinek1977perplexity`; lower is better. It correlates only loosely with downstream task accuracy, so large-scale evaluations report both {cite}`liang2022holistic`.

```{admonition} Language Modeling in Practice
:class: note
The notebook ({ref}`nb-attn-lm`) builds the causal mask, applies it inside attention, adds a language-model head, and computes cross-entropy and perplexity for an untrained model so you can see what "no better than guessing" looks like. Optimization, schedules and gradient flow are in the Chapter 2 notebook ({ref}`nb-math-optimizers`, {ref}`nb-math-backprop`).
```

## 5. Historical Context

### 5.1 Evolution from RNNs

The path to transformers involved several key developments:

**Recurrent Neural Networks** initially dominated sequence modeling but suffered from:

- Vanishing gradient problems in long sequences
- Sequential processing preventing parallelization
- Limited ability to capture long-range dependencies

**Long Short-Term Memory (LSTM)** {cite}`hochreiter1997long` networks addressed vanishing gradients but remained sequential.

**Attention mechanisms** {cite}`bahdanau2014neural` first appeared in machine translation, allowing models to focus on relevant source positions when generating target words.

### 5.2 The Transformer Breakthrough

"Attention Is All You Need" {cite}`vaswani2017attention` demonstrated that attention alone could achieve state-of-the-art results without recurrence or convolution. Key innovations included:

- Self-attention for capturing dependencies regardless of distance
- Parallel processing enabling efficient training
- Multi-head attention for capturing diverse relationships

### 5.3 Scaling to Language Models

The transformer's success in machine translation led to its adoption for language modeling:

- **GPT** (2018) showed the power of unsupervised pre-training
- **BERT** (2018) demonstrated bidirectional representations
- **GPT-2** (2019) showed zero-shot transfer to many tasks from language modeling alone {cite}`radford2019language`
- **GPT-3** (2020) achieved few-shot learning with 175B parameters {cite}`brown2020language`

This progression established the foundation for today's LLMs and their remarkable capabilities.

## Exercises

1. **Tokenization Analysis**: Compare BPE and unigram tokenization (both available in SentencePiece) on multilingual text. How do vocabulary sizes affect model performance?

1. **Attention Visualization**: Implement attention map visualization and analyze patterns in a small transformer model.

1. **Positional Encoding Experiment**: Compare sinusoidal, learned, and relative positional encodings on a language modeling task.

1. **Architecture Comparison**: Implement encoder-only, decoder-only, and encoder-decoder variants. When might each be preferred?

## Further Reading

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar - Visual explanations of transformer concepts
1. [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng - Comprehensive attention mechanisms guide
1. [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Video series building transformers from scratch

**Mathematical Foundations:**

4. [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka
5. [Linear Algebra for Machine Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) - Deep Learning Book Chapter 2

### Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite}`vaswani2017attention`: The original transformer paper
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) {cite}`radford2018improving`: GPT and the decoder-only architecture
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) {cite}`devlin2018bert`: BERT and bidirectional representations

### Technical Implementation Resources

**Code Implementations:**

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation with explanations
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Production-ready transformer implementations
- [MinGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy - Minimal GPT implementation for learning

**Interactive Learning:**

- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive visualization tool
- [Tiktokenizer](https://tiktokenizer.vercel.app/) - Explore how GPT and other tokenizers split text
- [Attention Patterns](https://github.com/jessevig/bertviz) - Visualize attention in BERT and GPT models

### Advanced Topics and Recent Research

**Scaling Laws and Training:**

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) {cite}`kaplan2020scaling` - Understanding model scaling behavior
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) {cite}`hoffmann2022training` - Chinchilla scaling laws

**Technical Deep Dives:**

- [A Primer in BERTology](https://arxiv.org/abs/2002.12327) {cite}`rogers2020primer` - What probing studies found about how BERT works
- [Pre-trained Models for Natural Language Processing](https://arxiv.org/abs/2003.08271) {cite}`qiu2020pre` - Survey of pre-trained models
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) {cite}`tenney2019bert` - Probing which layers of BERT encode which linguistic structure

**Optimization and Efficiency:**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) {cite}`su2021roformer` - RoPE positional encodings
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) {cite}`dao2022flashattention` - Efficient attention computation
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) {cite}`touvron2023llama` - Modern transformer design choices

### Books and Comprehensive Resources

1. **[Natural Language Processing with Transformers](https://github.com/nlp-with-transformers/notebooks)** by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
1. **[Deep Learning](https://www.deeplearningbook.org/)** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Mathematical foundations
1. **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** by Dan Jurafsky and James H. Martin - NLP fundamentals

### Practical Learning Projects

1. **Build a Miniature GPT** - Implement transformer from scratch using the notebook examples
1. **Tokenization Comparison** - Compare different tokenization strategies on multilingual text
1. **Attention Visualization** - Create tools to visualize attention patterns in pre-trained models
1. **Scaling Experiments** - Study how model performance changes with size using smaller versions

### Documentation and APIs

- [PyTorch Transformer Tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)

### Research Communities and Updates

- [Hugging Face Papers](https://huggingface.co/papers) - Curated ML research papers

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
