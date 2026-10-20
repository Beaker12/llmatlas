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

The notebook provides executable code examples for:

- Tokenization algorithms (BPE, WordPiece, SentencePiece)
- Token embedding and positional encoding implementations
- Scaled dot-product attention mechanisms
- Multi-head attention with visualization
- Self-attention, cross-attention, and masked attention patterns
- Complete transformer encoder and decoder layers

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
| Subword     | Moderate (32k-50k) | Medium          | Good         | Medium             |
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
        """Get all adjacent character pairs in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
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
                word_pairs = self.get_pairs(word)
                for pair in word_pairs:
                    pairs[pair] = pairs.get(pair, 0) + freq

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

1. **Original BPE**: Merges frequent character pairs
1. **WordPiece**: Google's variant using likelihood maximization
1. **SentencePiece**: Unigram language model approach

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
The accompanying notebook (Sections 1-2) demonstrates practical tokenization implementations:
- BPE (Byte Pair Encoding) algorithm step-by-step
- WordPiece and SentencePiece tokenizers
- Comparing different tokenization strategies
- Building custom vocabularies from scratch
- Handling special tokens and edge cases

Experiment with different tokenization approaches to see how they affect sequence length and representation quality.
```

### 1.3 Modern Tokenization Strategies

#### Tiktoken (OpenAI's Tokenizer)

GPT models use a sophisticated tokenizer that:

- Handles regex patterns for different text types
- Uses byte-level BPE
- Includes special tokens for different contexts

```python
# Example with tiktoken (conceptual)
import tiktoken

# Load GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello, world! Here's some code: `print('hi')`"
tokens = enc.encode(text)
decoded = enc.decode(tokens)

print(f"Original: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Decoded: {decoded}")
```

#### Multi-Modal Tokenization

Modern models handle multiple modalities:

- **Text tokens**: Standard subword tokens
- **Image tokens**: Patch embeddings or discrete visual tokens
- **Audio tokens**: Spectogram or audio codebook tokens
- **Special tokens**: Task indicators, separators, padding

#### Vocabulary Construction Strategies

**Frequency-based**: Include most common tokens

```python
def build_vocab_frequency(corpus, vocab_size):
    from collections import Counter

    # Count token frequencies
    token_counts = Counter()
    for text in corpus:
        tokens = tokenize(text)  # Your tokenization function
        token_counts.update(tokens)

    # Select top tokens
    vocab = [token for token, count in token_counts.most_common(vocab_size)]
    return vocab
```

**Coverage-based**: Ensure vocabulary covers percentage of corpus

```python
def build_vocab_coverage(corpus, coverage_threshold=0.95):
    from collections import Counter

    token_counts = Counter()
    total_tokens = 0

    for text in corpus:
        tokens = tokenize(text)
        token_counts.update(tokens)
        total_tokens += len(tokens)

    # Sort by frequency
    sorted_tokens = token_counts.most_common()

    vocab = []
    covered_tokens = 0

    for token, count in sorted_tokens:
        vocab.append(token)
        covered_tokens += count

        if covered_tokens / total_tokens >= coverage_threshold:
            break

    return vocab
```

### 1.4 Tokenization Impact on Model Behavior

#### Length and Efficiency

Different tokenization schemes produce different sequence lengths:

```python
# Example text
text = "The transformer architecture revolutionized natural language processing."

# Word-level (hypothetical)
word_tokens = text.lower().split()
print(f"Word tokens ({len(word_tokens)}): {word_tokens}")

# Character-level
char_tokens = list(text)
print(f"Character tokens ({len(char_tokens)}): {char_tokens[:20]}...")

# Subword (GPT-style, approximate)
# This would typically be: ["The", " transform", "er", " architect", "ure", ...]
subword_tokens = [
    "The",
    " transform",
    "er",
    " architect",
    "ure",
    " revolution",
    "ized",
    " natural",
    " language",
    " processing",
    ".",
]
print(f"Subword tokens ({len(subword_tokens)}): {subword_tokens}")
```

**Computational implications**:

- Sequence length affects attention computation: O(n²) for self-attention
- Longer sequences require more memory
- Processing time scales with sequence length

#### Linguistic Bias

Tokenization can introduce linguistic biases:

```python
# Example showing tokenization bias
texts = [
    "Hello world",  # English
    "Bonjour le monde",  # French
    "Hola mundo",  # Spanish
    "こんにちは世界",  # Japanese
    "مرحبا بالعالم",  # Arabic
]

# English-centric tokenizer might produce:
# English: ["Hello", " world"] (2 tokens)
# Japanese: ["こ", "ん", "に", "ち", "は", "世", "界"] (7 tokens)
# This creates unfair computational costs for non-English text
```

#### Out-of-Vocabulary Handling

Different strategies for unknown tokens:

1. **UNK token**: Replace unknown words with `<UNK>`
1. **Subword fallback**: Break unknown words into subwords
1. **Character fallback**: Fall back to character-level for unknown segments
1. **Byte fallback**: Ultimate fallback to byte representation

### 1.5 Special Tokens and Their Roles

#### Common Special Tokens

```python
SPECIAL_TOKENS = {
    "<BOS>": "Beginning of sequence",
    "<EOS>": "End of sequence",
    "<UNK>": "Unknown token",
    "<PAD>": "Padding token",
    "<MASK>": "Masked token (for BERT-style training)",
    "<SEP>": "Separator between segments",
    "<CLS>": "Classification token",
    "<|endoftext|>": "Document boundary (GPT)",
    "<|im_start|>": "Start of instruction/message",
    "<|im_end|>": "End of instruction/message",
}
```

#### Token IDs and Embeddings

Each token maps to a unique integer ID:

```python
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def encode(self, text):
        tokens = self.tokenize(text)  # Your tokenization logic
        return [
            self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens
        ]

    def decode(self, ids):
        tokens = [self.id_to_token[id] for id in ids]
        return self.detokenize(tokens)  # Your detokenization logic
```

### 1.6 Practical Considerations

#### Choosing Vocabulary Size

Common vocabulary sizes and their trade-offs:

- **Small (1K-8K)**: Fast, memory-efficient, poor representation
- **Medium (16K-32K)**: Balanced performance, most common choice
- **Large (50K-100K)**: Rich representation, expensive computation
- **Very Large (100K+)**: Diminishing returns, rarely used

#### Tokenization Efficiency

```python
import time


def benchmark_tokenization(texts, tokenizer):
    start_time = time.time()

    total_tokens = 0
    for text in texts:
        tokens = tokenizer.encode(text)
        total_tokens += len(tokens)

    end_time = time.time()

    return {
        "total_time": end_time - start_time,
        "texts_processed": len(texts),
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / (end_time - start_time),
        "texts_per_second": len(texts) / (end_time - start_time),
    }
```

#### Reversibility and Information Loss

Good tokenizers should be reversible:

```python
def test_reversibility(tokenizer, texts):
    """Test if tokenization is reversible."""
    failures = []

    for text in texts:
        tokens = tokenizer.encode(text)
        reconstructed = tokenizer.decode(tokens)

        if text != reconstructed:
            failures.append(
                {"original": text, "reconstructed": reconstructed, "tokens": tokens}
            )

    return failures
```

#### Embedding Spaces

Once tokenized, each token is mapped to a high-dimensional vector through an **embedding matrix** $E \\in \\mathbb{R}^{V \\times d}$, where $V$ is the vocabulary size and $d$ is the embedding dimension.

$$\\mathbf{x}\_i = E\[\\text{token}\_i\]$$

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

Modern architectures often share embeddings between input and output:

```python
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

$$PE\_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)$$
$$PE\_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

where:

- $pos$ is the position in the sequence
- $i$ is the dimension index
- $d$ is the embedding dimension

**Key properties**:

- Deterministic (same encoding for same position)
- Bounded values (between -1 and 1)
- Different frequencies for different dimensions
- Enables extrapolation to longer sequences

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
| Extrapolation  | Good          | Poor beyond training length |
| Training Speed | Fast          | Slower (more parameters)    |
| Flexibility    | Fixed pattern | Adaptive to data            |

##### Advanced Positional Encodings

**Relative Positional Encoding**: Instead of absolute positions, encode relative distances:

```python
def relative_position_encoding(seq_len, d_model, max_relative_distance=128):
    """Create relative positional encodings."""
    # Create relative position matrix
    positions = torch.arange(seq_len)
    relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

    # Clip to maximum distance
    relative_positions = torch.clamp(
        relative_positions, -max_relative_distance, max_relative_distance
    )

    # Convert to positive indices
    relative_positions += max_relative_distance

    # Create embedding table for relative positions
    relative_embedding = nn.Embedding(2 * max_relative_distance + 1, d_model)

    return relative_embedding(relative_positions)
```

**Rotary Position Embedding (RoPE)**: Used in modern models like LLaMA:

```python
def apply_rotary_position_embedding(x, position_ids):
    """Apply RoPE to input embeddings."""

    def get_rotation_matrix(seq_len, dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv_freq)

        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        return cos_vals, sin_vals

    # Split features into pairs for rotation
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos_vals, sin_vals = get_rotation_matrix(x.size(-2), x.size(-1))

    # Apply rotation
    rotated_x1 = x1 * cos_vals - x2 * sin_vals
    rotated_x2 = x1 * sin_vals + x2 * cos_vals

    # Recombine
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
```

##### Position Encoding Trade-offs

```python
def compare_position_encodings():
    """Compare different positional encoding strategies."""
    seq_lens = [128, 512, 1024, 2048, 4096]
    d_model = 512

    results = {"Sinusoidal": [], "Learned": [], "Relative": [], "RoPE": []}

    for seq_len in seq_lens:
        # Calculate parameter counts
        results["Sinusoidal"].append(0)  # No additional parameters
        results["Learned"].append(seq_len * d_model)  # One embedding per position
        results["Relative"].append(256 * d_model)  # Fixed relative distance vocabulary
        results["RoPE"].append(0)  # No additional parameters

    # Plot parameter growth
    plt.figure(figsize=(10, 6))
    for method, params in results.items():
        plt.plot(seq_lens, params, marker="o", label=method)

    plt.xlabel("Maximum Sequence Length")
    plt.ylabel("Additional Parameters")
    plt.title("Positional Encoding Parameter Requirements")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.show()


compare_position_encodings()
```

Modern variants often use learned positional embeddings or relative position encodings for improved performance on longer sequences.

```{admonition} Embeddings and Positional Encoding
:class: note
The notebook (Sections 3-4) provides interactive demonstrations of:
- Token embedding initialization strategies (Xavier, He, scaled)
- Positional encoding visualization (sinusoidal vs learned)
- Embedding similarity analysis and semantic relationships
- Effect of embedding dimension on model capacity
- Combining token embeddings with positional information

Understanding these concepts is crucial for transformer model design and debugging training issues.
```

### 1.7 Tokenization Challenges and Solutions

#### Cross-Lingual Tokenization

Different languages present unique tokenization challenges:

**Morphologically Rich Languages** (German, Turkish, Finnish):

```python
# German compound words
examples = {
    "Donaudampfschifffahrtsgesellschaftskapitän": "Danube steamship company captain",
    "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz": "Beef labeling supervision task transfer law",
}

# Traditional word tokenization would treat these as single rare tokens
# Subword tokenization can break them into meaningful components:
# ["Donau", "dampf", "schiff", "fahrt", "s", "gesellschaft", "s", "kapitän"]
```

**Agglutinative Languages** (Turkish, Finnish):

```python
# Turkish example: "evlerimizde" = "in our houses"
# Morphological breakdown:
# ev (house) + ler (plural) + imiz (our) + de (in/at)

# Good tokenization preserves morphological structure:
turkish_tokens = ["ev", "ler", "imiz", "de"]  # Ideal
suboptimal_tokens = ["evl", "erimiz", "de"]  # Suboptimal
```

**Languages Without Spaces** (Chinese, Japanese, Thai):

```python
# Chinese doesn't use spaces between words
chinese_text = "我喜欢学习自然语言处理"
# Meaning: "I like learning natural language processing"

# Character-level: ["我", "喜", "欢", "学", "习", "自", "然", "语", "言", "处", "理"]
# Word-level (with segmentation): ["我", "喜欢", "学习", "自然语言处理"]
# Subword: ["我", "喜欢", "学", "习", "自然", "语言", "处理"]
```

#### Handling Special Cases

**Code and Technical Text**:

```python
code_example = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Challenges:
# - Preserving code structure
# - Handling indentation
# - Function/variable names
# - Special characters (:, (), +, -, etc.)

# Good tokenization for code:
# ["def", " fibonacci", "(", "n", "):", "\n", "    if", " n", " <=", " 1", ":", ...]
```

**URLs and Email Addresses**:

```python
web_examples = [
    "https://www.example.com/path/to/page?param=value",
    "user.name+tag@example.co.uk",
    "file:///home/user/document.pdf",
]

# Challenges:
# - Preserving URL structure
# - Handling domain names
# - Special characters (., @, /, ?, =, etc.)
# - Email format preservation
```

**Numbers and Dates**:

```python
numerical_examples = [
    "3.14159",  # Pi
    "1,234,567.89",  # Large number with commas
    "2023-12-25",  # ISO date
    "12/25/2023",  # US date format
    "25th December",  # Written date
    "$1,250.00",  # Currency
    "42%",  # Percentage
    "1.2e-5",  # Scientific notation
]

# Different tokenization strategies:
# Option 1: Preserve whole numbers
# ["3.14159", "1,234,567.89", "2023-12-25"]

# Option 2: Split on punctuation
# ["3", ".", "14159", "1", ",", "234", ",", "567", ".", "89"]

# Option 3: Smart number handling
# ["3.14159", "1,234,567.89", "2023", "-", "12", "-", "25"]
```

#### Domain Adaptation

**Medical Text**:

```python
medical_text = (
    "Patient presents with myocardial infarction and diabetes mellitus type 2"
)

# Domain-specific vocabulary:
medical_tokens = [
    "Patient",
    " presents",
    " with",
    " myocardial",
    " infarction",
    " and",
    " diabetes",
    " mellitus",
    " type",
    " 2",
]

# Generic tokenizer might split medical terms poorly:
generic_tokens = [
    "Patient",
    " presents",
    " with",
    " my",
    "ocard",
    "ial",
    " inf",
    "arc",
    "tion",
    " and",
    " diab",
    "etes",
    " mell",
    "itus",
    " type",
    " 2",
]
```

**Legal Text**:

```python
legal_text = "The aforementioned clause notwithstanding the heretofore provisions"

# Legal language has specific terminology patterns
# Need specialized vocabulary for legal corpus
```

#### Quality Metrics for Tokenization

```python
def evaluate_tokenization_quality(tokenizer, corpus, language="en"):
    """Evaluate tokenization quality using multiple metrics."""

    metrics = {
        "compression_ratio": 0,
        "oov_rate": 0,
        "morphological_consistency": 0,
        "reversibility_errors": 0,
        "cross_lingual_equity": 0,
    }

    total_chars = 0
    total_tokens = 0
    oov_count = 0
    reversibility_errors = 0

    for text in corpus:
        # Compression ratio
        tokens = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(tokens)

        # OOV rate
        oov_count += sum(1 for token in tokens if token == tokenizer.unk_token_id)

        # Reversibility
        reconstructed = tokenizer.decode(tokens)
        if text != reconstructed:
            reversibility_errors += 1

    metrics["compression_ratio"] = total_chars / total_tokens
    metrics["oov_rate"] = oov_count / total_tokens
    metrics["reversibility_errors"] = reversibility_errors / len(corpus)

    return metrics


def compare_tokenizers(tokenizers, test_corpus):
    """Compare multiple tokenizers on the same corpus."""
    results = {}

    for name, tokenizer in tokenizers.items():
        results[name] = evaluate_tokenization_quality(tokenizer, test_corpus)

    # Create comparison table
    import pandas as pd

    df = pd.DataFrame(results).T
    print("Tokenizer Comparison:")
    print(df.round(3))

    return df
```

#### Best Practices for Tokenization

**1. Vocabulary Size Selection**:

```python
def find_optimal_vocab_size(corpus, vocab_sizes, validation_corpus=None):
    """Find optimal vocabulary size through validation."""
    results = {}

    for vocab_size in vocab_sizes:
        # Train tokenizer
        tokenizer = train_bpe_tokenizer(corpus, vocab_size)

        # Evaluate on validation set
        if validation_corpus:
            metrics = evaluate_tokenization_quality(tokenizer, validation_corpus)
            results[vocab_size] = metrics

    return results


# Typical ranges to test:
vocab_sizes = [1000, 2000, 4000, 8000, 16000, 32000, 50000, 64000]
```

**2. Preprocessing Strategies**:

```python
def preprocess_for_tokenization(text, strategy="conservative"):
    """Apply preprocessing before tokenization."""

    if strategy == "conservative":
        # Minimal preprocessing
        return text.strip()

    elif strategy == "aggressive":
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Handle special cases
        text = re.sub(r"([.!?])\s+", r"\1 ", text)  # Normalize punctuation spacing
        text = re.sub(r"\s+([,.!?])", r"\1", text)  # Remove space before punctuation

        return text.strip()

    elif strategy == "code_aware":
        # Preserve code structure
        # Don't normalize indentation
        # Keep special programming characters
        return text

    return text
```

**3. Handling Rare Tokens**:

```python
def analyze_token_frequency(corpus, tokenizer):
    """Analyze token frequency distribution."""
    from collections import Counter

    all_tokens = []
    for text in corpus:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    token_counts = Counter(all_tokens)

    # Calculate statistics
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counts)

    # Find rare tokens (appear < 5 times)
    rare_threshold = 5
    rare_tokens = sum(1 for count in token_counts.values() if count < rare_threshold)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print(
        f"Rare tokens (< {rare_threshold}): {rare_tokens:,} ({rare_tokens/unique_tokens:.1%})"
    )

    # Plot frequency distribution
    frequencies = sorted(token_counts.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.loglog(range(1, len(frequencies) + 1), frequencies)
    plt.xlabel("Token Rank")
    plt.ylabel("Frequency")
    plt.title("Token Frequency Distribution (Zipf's Law)")
    plt.grid(True)
    plt.show()

    return token_counts
```

**4. Multi-Domain Tokenization**:

```python
def create_multi_domain_tokenizer(domain_corpora, vocab_size=32000):
    """Create tokenizer that works well across multiple domains."""

    # Combine corpora with domain balancing
    combined_corpus = []
    min_size = min(len(corpus) for corpus in domain_corpora.values())

    for domain, corpus in domain_corpora.items():
        # Sample equally from each domain
        sampled = random.sample(corpus, min_size)
        combined_corpus.extend(sampled)

    # Train on balanced corpus
    tokenizer = train_tokenizer(combined_corpus, vocab_size)

    # Evaluate on each domain
    domain_performance = {}
    for domain, corpus in domain_corpora.items():
        metrics = evaluate_tokenization_quality(tokenizer, corpus[:1000])
        domain_performance[domain] = metrics

    return tokenizer, domain_performance
```

### 1.8 Practical Implementation

This section provides hands-on implementation of tokenization and embedding concepts:

#### Complete Tokenization Pipeline

```python
import torch
import torch.nn as nn
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class ComprehensiveTokenizer:
    """A complete tokenizer implementation with multiple strategies."""

    def __init__(self, vocab_size=32000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<MASK>": 4,
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization."""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Add spaces around punctuation for better splitting
        text = re.sub(r"([.!?,:;])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def get_word_frequencies(self, corpus: List[str]) -> Dict[str, int]:
        """Count word frequencies in corpus."""
        word_freq = Counter()

        for text in corpus:
            text = self.preprocess_text(text)
            words = text.lower().split()
            word_freq.update(words)

        return word_freq

    def get_character_level_vocab(self, corpus: List[str]) -> set:
        """Build character-level vocabulary."""
        chars = set()
        for text in corpus:
            chars.update(text)
        return chars

    def train_bpe(self, corpus: List[str]) -> None:
        """Train Byte Pair Encoding tokenizer."""
        print("Training BPE tokenizer...")

        # Get word frequencies
        word_freq = self.get_word_frequencies(corpus)

        # Filter by frequency
        word_freq = {
            word: freq for word, freq in word_freq.items() if freq >= self.min_frequency
        }

        # Initialize vocabulary with characters + special tokens
        vocab = set()
        for text in corpus:
            vocab.update(text)

        # Add special tokens
        vocab.update(self.special_tokens.keys())

        # Convert words to character sequences
        word_splits = {}
        for word, freq in word_freq.items():
            word_splits[tuple(word) + ("</w>",)] = freq

        # Iteratively merge most frequent pairs
        target_vocab_size = self.vocab_size - len(self.special_tokens)

        while len(vocab) < target_vocab_size:
            # Count all adjacent pairs
            pairs = defaultdict(int)
            for word, freq in word_splits.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge the pair in all words
            new_word_splits = {}
            for word, freq in word_splits.items():
                new_word = self._merge_pair(word, best_pair)
                new_word_splits[new_word] = freq

            word_splits = new_word_splits

            # Add merged token to vocabulary
            merged_token = "".join(best_pair)
            vocab.add(merged_token)
            self.merges.append(best_pair)

            if len(vocab) % 1000 == 0:
                print(f"Vocabulary size: {len(vocab)}")

        # Create final vocabulary mapping
        vocab_list = list(self.special_tokens.keys()) + sorted(list(vocab))
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        self.reverse_vocab = {i: token for token, i in self.vocab.items()}

        print(f"Training complete. Final vocabulary size: {len(self.vocab)}")

    def _merge_pair(
        self, word: Tuple[str, ...], pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
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

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        text = self.preprocess_text(text)
        words = text.lower().split()

        token_ids = [self.special_tokens["<BOS>"]]

        for word in words:
            # Convert word to character tuple
            word_tokens = tuple(word) + ("</w>",)

            # Apply BPE merges
            for pair in self.merges:
                word_tokens = self._merge_pair(word_tokens, pair)

            # Convert to IDs
            for token in word_tokens:
                token_id = self.vocab.get(token, self.special_tokens["<UNK>"])
                token_ids.append(token_id)

        token_ids.append(self.special_tokens["<EOS>"])
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)

        # Join tokens and clean up
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, filepath: str):
        """Save tokenizer to file."""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "merges": self.merges,
                    "special_tokens": self.special_tokens,
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str):
        """Load tokenizer from file."""
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = data["merges"]
        tokenizer.special_tokens = data["special_tokens"]
        tokenizer.reverse_vocab = {i: token for token, i in tokenizer.vocab.items()}

        return tokenizer


# Example usage
sample_corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "The only thing we have to fear is fear itself",
    "Ask not what your country can do for you",
] * 100  # Repeat for better training

# Train tokenizer
tokenizer = ComprehensiveTokenizer(vocab_size=1000)
tokenizer.train_bpe(sample_corpus)

# Test encoding/decoding
test_text = "The quick brown fox"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print(f"Original: {test_text}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

#### Advanced Embedding Implementation

```python
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
            self.max_relative_distance = 128
            self.relative_embeddings = nn.Embedding(
                2 * self.max_relative_distance + 1, d_model
            )

    def forward(self, x, position_ids=None):
        if self.encoding_type == "sinusoidal":
            return x + self.pe[:, : x.size(1)]

        elif self.encoding_type == "learned":
            if position_ids is None:
                position_ids = torch.arange(x.size(1), device=x.device)
            return x + self.position_embeddings(position_ids)

        elif self.encoding_type == "relative":
            # Implement relative positioning
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device)
            relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
            relative_positions = (
                torch.clamp(
                    relative_positions,
                    -self.max_relative_distance,
                    self.max_relative_distance,
                )
                + self.max_relative_distance
            )

            relative_encodings = self.relative_embeddings(relative_positions)
            # This is simplified - in practice, relative encodings are applied in attention
            return x

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

#### Tokenization Analysis Tools

```python
def analyze_tokenization_efficiency(tokenizer, corpus):
    """Analyze tokenization efficiency and patterns."""

    results = {
        "compression_ratios": [],
        "token_lengths": [],
        "oov_rates": [],
        "subword_patterns": Counter(),
    }

    for text in corpus:
        # Encode text
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.reverse_vocab.get(tid, "<UNK>") for tid in token_ids]

        # Calculate compression ratio
        char_count = len(text)
        token_count = len(token_ids)
        compression_ratio = char_count / token_count if token_count > 0 else 0
        results["compression_ratios"].append(compression_ratio)

        # Track token lengths
        for token in tokens:
            if token not in tokenizer.special_tokens:
                results["token_lengths"].append(len(token))

                # Analyze subword patterns
                if token.endswith("</w>"):
                    results["subword_patterns"]["word_final"] += 1
                elif len(token) == 1:
                    results["subword_patterns"]["character"] += 1
                else:
                    results["subword_patterns"]["subword"] += 1

        # Calculate OOV rate
        oov_count = sum(
            1 for tid in token_ids if tid == tokenizer.special_tokens["<UNK>"]
        )
        oov_rate = oov_count / len(token_ids) if len(token_ids) > 0 else 0
        results["oov_rates"].append(oov_rate)

    # Summary statistics
    print("Tokenization Analysis Results:")
    print(f"Average compression ratio: {np.mean(results['compression_ratios']):.2f}")
    print(f"Average token length: {np.mean(results['token_lengths']):.2f}")
    print(f"Average OOV rate: {np.mean(results['oov_rates']):.3f}")
    print(f"Subword pattern distribution: {dict(results['subword_patterns'])}")

    return results


# Run analysis
analysis_results = analyze_tokenization_efficiency(tokenizer, sample_corpus)
```

This practical implementation provides a complete foundation for understanding and working with tokenization and embeddings in transformer models.

```{admonition} Hands-On Practice
:class: tip
For interactive examples and visualizations of these concepts, see the accompanying Jupyter notebook: [Attention Mechanisms Demo](examples/attention_mechanisms_demo.ipynb)

The notebook provides:
- Working tokenization implementations
- Interactive attention visualizations
- Embedding similarity analysis
- Complete transformer components
- Mathematical concept demonstrations
```

## 2. Attention Mechanisms

### 2.1 The Attention Revolution

Attention mechanisms revolutionized sequence modeling by allowing models to focus on relevant parts of the input when making predictions. Unlike RNNs that process sequences sequentially, attention enables parallel computation and direct connections between any two positions.

```{admonition} Interactive Exploration
:class: note
The accompanying notebook demonstrates attention mechanisms with working code and visualizations. See cells 8-11 for multi-head attention implementation and attention pattern analysis.
```

#### Self-Attention Fundamentals

Self-attention computes a weighted representation of the input sequence, where weights indicate the relevance of each position to the current position. Given input embeddings $X \\in \\mathbb{R}^{n \\times d}$:

1. **Query, Key, Value projections**:
   $$Q = XW_Q, \\quad K = XW_K, \\quad V = XW_V$$

1. **Attention scores**:
   $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

The scaling factor $\\sqrt{d_k}$ prevents the softmax from saturating for large embedding dimensions {cite}`vaswani2017attention`.

#### Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to attend to different types of relationships:

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}\_1, ..., \\text{head}\_h)W_O$$

where each head is:
$$\\text{head}\_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 2.2 Attention Patterns and Interpretability

Different attention heads learn to capture various linguistic phenomena:

- **Syntactic heads**: Focus on grammatical relationships (subject-verb, modifier-noun)
- **Semantic heads**: Capture semantic similarity and coreference
- **Positional heads**: Attend to specific relative positions

Visualization tools like attention maps provide insights into model behavior, though interpreting attention patterns requires caution {cite}`jain2019attention`.

```python
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

$$\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sigma} + \\beta$$

where $\\mu$ and $\\sigma$ are the mean and standard deviation computed across the feature dimension.

#### Residual Connections

Residual connections help gradient flow in deep networks:

$$\\text{output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))$$

#### Feed-Forward Networks

Each transformer layer includes a position-wise feed-forward network:

$$\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2$$

This provides non-linear transformations and typically uses a hidden dimension 4× larger than the model dimension.

### 3.2 Architectural Variants

While the original transformer was designed for machine translation, different variants emerged for language modeling:

- **GPT** {cite}`radford2018improving`: Decoder-only architecture with causal masking
- **BERT** {cite}`devlin2018bert`: Encoder-only architecture with bidirectional attention
- **T5** {cite}`raffel2019exploring`: Encoder-decoder architecture treating all tasks as text-to-text

```python
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
The notebook (Sections 12-13) walks through building a complete transformer:
- Layer normalization and residual connections
- Position-wise feed-forward networks
- Complete encoder and decoder blocks
- Training a small transformer on a simple task
- Visualizing attention patterns across layers

This hands-on implementation solidifies understanding of how all transformer components work together.
```

## 4. Language Modeling Objective

### 4.1 Next-Token Prediction

Modern LLMs are trained on the **causal language modeling** objective, predicting the next token given previous tokens:

$$P(x\_{t+1} | x_1, x_2, ..., x_t)$$

This autoregressive formulation enables text generation and captures the statistical structure of language.

#### Causal Masking

To prevent the model from "seeing the future," causal masking ensures attention only flows from earlier to later positions:

```python
def create_causal_mask(seq_length):
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0  # True for allowed positions
```

### 4.2 Training Process

The training objective maximizes the likelihood of the training data:

$$\\mathcal{L} = -\\sum\_{i=1}^{N} \\log P(x_i | x\_{\<i})$$

Where the model outputs probability distributions over the vocabulary:

$$P(x_i | x\_{\<i}) = \\text{softmax}(W\_{\\text{out}} h_i + b\_{\\text{out}})$$

### 4.3 Evaluation Metrics

**Perplexity** measures how well the model predicts the test data:

$$\\text{Perplexity} = \\exp\\left(-\\frac{1}{N} \\sum\_{i=1}^{N} \\log P(x_i | x\_{\<i})\\right)$$

Lower perplexity indicates better modeling of the data distribution. However, perplexity doesn't always correlate with downstream task performance {cite}`jelinek1977perplexity`.

```{admonition} Language Modeling in Practice
:class: note
The notebook (Section 14) demonstrates language modeling concepts:
- Implementing the causal language modeling objective
- Computing cross-entropy loss and perplexity
- Masked self-attention for autoregressive generation
- Comparing teacher forcing vs autoregressive sampling
- Evaluating model predictions with perplexity metrics

These implementations connect theory to practical language model training and evaluation.
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
- **GPT-2** (2019) revealed emergent capabilities at scale
- **GPT-3** (2020) achieved few-shot learning with 175B parameters

This progression established the foundation for today's LLMs and their remarkable capabilities.

## Exercises

1. **Tokenization Analysis**: Compare BPE and SentencePiece tokenization on multilingual text. How do vocabulary sizes affect model performance?

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
4\. [Understanding the Transformer](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka
5\. [Linear Algebra for Machine Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) - Deep Learning Book Chapter 2

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
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) - Explore tokenization with GPT models
- [Attention Patterns](https://github.com/jessevig/bertviz) - Visualize attention in BERT and GPT models

### Advanced Topics and Recent Research

**Scaling Laws and Training:**

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) {cite}`kaplan2020scaling` - Understanding model scaling behavior
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) - Chinchilla scaling laws

**Technical Deep Dives:**

- [A Primer on Neural Network Architectures](https://arxiv.org/abs/1807.10854) {cite}`rogers2020primer` - Comprehensive BERT analysis
- [Pre-trained Models for Natural Language Processing](https://arxiv.org/abs/2003.08271) {cite}`qiu2020pre` - Survey of pre-trained models
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) {cite}`kenton2019bert` - BERT implementation details

**Optimization and Efficiency:**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) {cite}`su2021roformer` - RoPE positional encodings
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) {cite}`dao2022flashattention` - Efficient attention computation
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) {cite}`touvron2023llama` - Modern transformer design choices

### Books and Comprehensive Resources

1. **[Natural Language Processing with Transformers](https://transformersbook.com/)** by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
1. **[Deep Learning](https://www.deeplearningbook.org/)** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Mathematical foundations
1. **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** by Dan Jurafsky and James H. Martin - NLP fundamentals

### Practical Learning Projects

1. **Build a Miniature GPT** - Implement transformer from scratch using the notebook examples
1. **Tokenization Comparison** - Compare different tokenization strategies on multilingual text
1. **Attention Visualization** - Create tools to visualize attention patterns in pre-trained models
1. **Scaling Experiments** - Study how model performance changes with size using smaller versions

### Documentation and APIs

- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)
- [Weights & Biases Model Training Guides](https://docs.wandb.ai/guides/models)

### Research Communities and Updates

- [Papers With Code - Transformers](https://paperswithcode.com/methods/category/transformers) - Latest research with code
- [Hugging Face Papers](https://huggingface.co/papers) - Curated ML research papers
- [Towards Data Science](https://towardsdatascience.com/tagged/transformers) - Medium articles on transformers

## Bibliography

```{bibliography}
:filter: docname in docnames
```

______________________________________________________________________

*Part of the LLM Atlas - A Comprehensive Guide to Large Language Models*
