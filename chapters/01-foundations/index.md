# LLM Foundations

Large Language Models represent a paradigm shift in natural language processing, achieving unprecedented capabilities through the transformer architecture and massive scale training. This chapter establishes the foundational concepts necessary to understand how these models work, their historical development, and the core principles that enable their remarkable performance.

(sec-ch1-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- How tokenization transforms text into numerical representations
- The mechanics of attention mechanisms and their role in language understanding
- The transformer architecture and its key components
- The language modeling objective and evaluation metrics
- The historical evolution from RNNs to modern transformers

(sec-ch1-hands-on-learning)=

## Hands-On Learning

For interactive implementations and visualizations of the foundational concepts in this chapter, see the accompanying Jupyter notebook: [Attention Mechanisms Demo](examples/attention_mechanisms_demo.ipynb)

The notebook provides executable, assertion-checked code for:

- A word-level tokenizer and the BPE merge loop ({ref}`nb-attn-tokenization`)
- Token embeddings with sinusoidal, learned, relative-bias and rotary positions ({ref}`nb-attn-positional`)
- Scaled dot-product and multi-head attention with attention maps ({ref}`nb-attn-attention`)
- A post-LN transformer block ({ref}`nb-attn-transformer`)
- The causal mask and the next-token objective with perplexity ({ref}`nb-attn-lm`)

(sec-ch1-chapter-overview)=

## Chapter Overview

Large Language Models (LLMs) like ChatGPT work by understanding and generating text through several key processes that can be understood through everyday analogies:

**Tokenization** is like breaking down a sentence into meaningful chunks - imagine reading a book where you recognize not just individual letters, but common word parts, whole words, and phrases. LLMs break text into "tokens" (pieces like "un-", "happy", "ness") that help them understand meaning more efficiently than processing character by character.

**Embeddings** convert these text pieces into numerical representations - think of it like giving each word or word-part a unique mathematical "fingerprint" that captures its meaning. Similar words get similar fingerprints, allowing the model to understand that "happy" and "joyful" are related concepts.

**Attention Mechanisms** are like a sophisticated highlighting system that helps the model focus on the most relevant parts of text when making predictions. When processing "The cat sat on the mat," the attention mechanism helps the model understand that "cat" is the subject performing the action "sat," similar to how you naturally emphasize important words when reading.

**Transformers** are the overall architecture that combines all these elements - like a highly efficient assembly line where multiple "workers" (attention heads) simultaneously analyze different aspects of the text (grammar, meaning, context) and combine their insights to understand and generate language.

**Language Modeling** is the training process where the model learns to predict the next word in a sentence by reading millions of examples. It's like learning to complete sentences by reading vast amounts of text and developing an intuition for how language flows naturally.

These components work together to create models that can understand context, maintain coherent conversations, and generate human-like text by learning statistical patterns from enormous amounts of text data.

(sec-ch1-tokenization-embeddings)=

## 1. Tokenization and Embeddings

(sec-ch1-understanding-tokenization)=

### 1.1 Understanding Tokenization

Large Language Models cannot directly process raw text—they require numerical representations. The process begins with **tokenization**, which breaks text into discrete units called tokens. This seemingly simple step profoundly affects model performance, computational efficiency, and language understanding capabilities.

(sec-ch1-token-definition)=

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
| Character   | ~10²–10⁴ (script dependent) | Very Long | Good (unseen characters are still OOV) | Low |
| Byte-level  | Fixed (256)        | Longest         | Perfect      | Lowest             |

(sec-ch1-vocabulary-size-dilemma)=

#### The Vocabulary Size Dilemma

Vocabulary size directly impacts:

1. **Model parameters**: Embedding matrix size = vocab_size × embedding_dim
1. **Computational cost**: Output projection scales with vocabulary size
1. **Memory requirements**: Larger vocabularies require more memory
1. **Out-of-vocabulary handling**: Smaller vocabularies struggle with rare words

**Example**: A 50k vocabulary with 4096-dimensional embeddings requires 200M parameters just for embeddings—before any transformer layers!

(sec-ch1-tokenization-algorithms)=

### 1.2 Tokenization Algorithms

(sec-ch1-word-level-tokenization)=

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

(sec-ch1-character-level-tokenization)=

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

(sec-ch1-byte-pair-encoding-bpe)=

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
bpe.train(corpus, num_merges=4)
print(f"Merges: {bpe.merges}")
print(f"Tokens: {bpe.encode('helper')}")  # an unseen word splits into learned subwords and characters
```

% output

```text
Merges: [('h', 'e'), ('he', 'l'), ('hel', 'l'), ('hell', 'o')]
Tokens: ['hel', 'p', 'e', 'r', '</w>']
```

**BPE Evolution**:

1. **Original BPE** {cite}`sennrich2015neural`: merges the most frequent adjacent pair
1. **WordPiece**: Google's variant that merges the pair which most increases the training corpus likelihood
1. **Unigram LM** {cite}`kudo2018subword`: starts from a large candidate vocabulary and prunes it, keeping the segmentation with the highest probability
1. **SentencePiece** {cite}`kudo2018sentencepiece` is a *toolkit*, not an algorithm: it implements both BPE and unigram on raw text without pre-tokenization

(sec-ch1-sentencepiece)=

#### SentencePiece

SentencePiece {cite}`kudo2018sentencepiece` treats text as raw Unicode sequences without pre-tokenization:

```python
import os
import tempfile

import sentencepiece as spm

# A small corpus: a few English sentences repeated, plus one line of Chinese the model will rarely see.
sentences = [
    "Hello, world! Tokenizers turn text into integers.",
    "The quick brown fox jumps over the lazy dog.",
    "Language models predict the next token from the previous ones.",
    "Byte fallback lets a tokenizer encode characters it never learned.",
]
with tempfile.TemporaryDirectory() as tmp:
    corpus_path = os.path.join(tmp, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences * 200 + ["你好"]))

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=os.path.join(tmp, "tokenizer"),
        vocab_size=400,  # includes the 256 byte pieces that byte_fallback reserves
        model_type="bpe",  # or "unigram"
        character_coverage=0.9995,
        byte_fallback=True,  # characters outside the learned set become UTF-8 byte pieces, not <unk>
        normalization_rule_name="identity",
        num_threads=1,
        minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=os.path.join(tmp, "tokenizer.model"))

text = "Hello, world! 你好世界"
pieces = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)
assert sp.decode_ids(ids) == text  # lossless, because unseen characters fall back to bytes
assert sp.unk_id() not in ids
print(f"Pieces: {pieces}")
print(f"{len(ids)} ids; decoded text equals the input: {sp.decode_ids(ids) == text}")
```

% output

```text
Pieces: ['▁He', 'llo', ',', '▁w', 'orld', '!', '▁', '<0xE4>', '<0xBD>', '<0xA0>', '<0xE5>', '<0xA5>', '<0xBD>', '<0xE4>', '<0xB8>', '<0x96>', '<0xE7>', '<0x95>', '<0x8C>']
19 ids; decoded text equals the input: True
```

With byte fallback, the four Chinese characters, absent from the learned pieces, become twelve UTF-8 byte pieces instead of `<unk>`, and decoding returns the input. Without it, the same model would map them to the unknown id. The toy corpus supports a vocabulary of a few hundred pieces; asking for the common 32,000 needs a training corpus of tens of megabytes, and SentencePiece refuses a vocabulary size larger than the corpus can fill.

**Key advantages**:

- Language-agnostic (no pre-tokenization assumptions)
- Handles any Unicode text, losslessly only with byte fallback or full character coverage
- Consistent treatment of whitespace, written as the visible symbol `▁`
- Reversible tokenization under the same condition

```{admonition} Tokenization Implementation
:class: tip
The notebook's tokenization section ({ref}`nb-attn-tokenization`) builds a tiny word-level vocabulary and runs the BPE merge loop step by step, printing the splits after every merge. For WordPiece, unigram and SentencePiece in practice, see {doc}`tokenization-deep-dive`.
```

(sec-ch1-embeddings-positional-encoding)=

### 1.3 Embeddings and Positional Encoding

For vocabulary construction, special tokens, cross-lingual issues and a full tokenizer implementation, see {doc}`tokenization-deep-dive`.

(sec-ch1-embedding-spaces)=

#### Embedding Spaces

Once tokenized, each token is mapped to a high-dimensional vector through an **embedding matrix** $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$, where $\mathcal{V}$ is the vocabulary, $|\mathcal{V}|$ its size and $d$ the embedding dimension. The vector for token $x_i$ is row $x_i$ of $W_E$:

$$W_E[x_i] \in \mathbb{R}^{d}$$

These embeddings are learned during training to capture semantic relationships. Similar tokens develop similar vector representations, enabling the model to generalize across related concepts {cite}`mikolov2013efficient`.

(sec-ch1-understanding-embedding-dimensions)=

#### Understanding Embedding Dimensions

The embedding dimension $d$ is a crucial hyperparameter that affects:

**Representational Capacity**: Higher dimensions can encode more information

- 128-256d: Small models, limited expressiveness
- 512-1024d: Medium models, good balance
- 2048-4096d: Large models, rich representations
- 8192d+: Very large models, potential diminishing returns

**Computational Cost**: Embedding operations scale with dimension

- Matrix multiplications: the cost of each projection grows linearly with $d$ when its other dimension is fixed
- Memory usage: $|\mathcal{V}| \times d$ parameters just for embeddings
- Attention computation: Affects all transformer operations

(sec-ch1-embedding-initialization-strategies)=

#### Embedding Initialization Strategies

The initialization and any scaling in the forward pass have to be chosen together. The original transformer multiplies the embedding by $\sqrt{d}$ before adding the sinusoidal encoding {cite}`vaswani2017attention`. If the table is drawn from $\mathcal{N}(0, 1/d)$, each coordinate of the scaled embedding has variance $1/d \cdot d = 1$, the same order as the sinusoidal encoding, whose coordinates average $\sin^2 = \cos^2 = 1/2$. Drawing the table with Xavier's rule instead, whose variance $2/(|\mathcal{V}| + d)$ is tiny for a 50,000-row table, would leave the position signal about 25 times stronger than the token signal; drawing it from $\mathcal{N}(0, 1)$ and then scaling would make the token signal over a hundred times stronger.

```python
import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        # N(0, 1/d): after the sqrt(d) scaling below, every coordinate has unit variance
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)

    def forward(self, x):
        # Scale embeddings by sqrt(d_model) following the original transformer
        return self.embedding(x) * math.sqrt(self.d_model)


torch.manual_seed(0)
emb = TokenEmbedding(vocab_size=50_000, d_model=512)
token_rms = emb(torch.arange(50_000)).pow(2).mean().sqrt()
print(f"RMS of a scaled token embedding: {token_rms:.3f}")
print(f"RMS of a sinusoidal encoding:    {0.5 ** 0.5:.3f}")
```

% output

```text
RMS of a scaled token embedding: 1.000
RMS of a sinusoidal encoding:    0.707
```

(sec-ch1-embedding-sharing-tying)=

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

(sec-ch1-subword-embedding-composition)=

#### Subword Embedding Composition

For subword tokens, embeddings must compose meaningfully:

```python
# pseudo-code: `embedding_matrix` and `token_to_id` come from a trained model
# Example: How "unhappiness" might be tokenized and embedded
text = "unhappiness"
tokens = ["un", "h", "appiness"]  # tiktoken's gpt2, cl100k_base and o200k_base all give this split

# Each token gets its own embedding
token_embeddings = [embedding_matrix[token_to_id[t]] for t in tokens]  # 3 vectors of size [d_model]

# The model sees these as separate tokens in sequence
# The split does not follow the morphemes un + happi + ness; the model must still learn that "un" negates
# This happens through attention and context
```

(sec-ch1-embedding-visualization-analysis)=

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

(sec-ch1-positional-encodings)=

#### Positional Encodings

Attention by itself does not see the order of its inputs, so positional information must be supplied. Word order matters in language:

- "The cat sat on the mat"
- "The mat sat on the cat"

Without positional information and without a mask, self-attention treats each sentence as a multiset of tokens: permuting the inputs permutes the outputs in the same way and changes nothing else, so these two sentences, which hold the same tokens, get the same set of output vectors. A causal decoder is not completely order-blind even without positions, because its mask lets position $m$ attend to exactly $m$ earlier-or-equal tokens; but that is a weak signal, and every model below supplies positions explicitly.

##### Sinusoidal Positional Encoding

The original transformer uses sinusoidal positional encodings {cite}`vaswani2017attention`:

$$PE_{(m, 2i)} = \sin\left(\frac{m}{10000^{2i/d}}\right)$$
$$PE_{(m, 2i+1)} = \cos\left(\frac{m}{10000^{2i/d}}\right)$$

where:

- $m$ is the position in the sequence
- $i$ indexes the sine-cosine pairs, $0 \le i < d/2$
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

(sec-ch1-relative-bias)=

#### Relative Position Bias (T5)

Instead of adding absolute positions to the input, a relative scheme makes attention depend on the offset $m - m'$ between the query position $m$ and the key position $m'$. {cite:t}`shaw2018self` add learned relative-position *vectors* to the keys and values. T5 {cite}`raffel2019exploring` simplifies this to one learned scalar per head, added to the attention score of every query-key pair:

$$S_{mm'} = \frac{q_m \cdot k_{m'}}{\sqrt{d_k}} + \operatorname{rb}(\operatorname{bucket}(m - m'))$$

The function $\operatorname{bucket}$ maps an offset to one of 32 buckets. Small offsets get a bucket each; larger ones share buckets whose width grows logarithmically, up to an offset of 128, beyond which all offsets share the last bucket. In the encoder, which attends in both directions, half of the buckets are for keys before or at the query and half for keys after it. For keys at or before the query, offsets 0 to 7 have their own buckets, and an offset $8 \le m - m' < 128$ goes to bucket

$$8 + \left\lfloor 8 \, \frac{\ln\big((m - m')/8\big)}{\ln 16} \right\rfloor,$$

so offset 12 lands in bucket $8 + \lfloor 8 \times 0.146 \rfloor = 9$, offset 16 in bucket 10, offset 64 in bucket 14, and every offset from 128 on in bucket 15. A key after the query uses the same rule on $m' - m$ plus 16. A causal decoder never sees keys after the query, so it spends all 32 buckets on the past: offsets 0 to 15 exactly, then 16 log-spaced buckets up to 128. The table $\operatorname{rb}$ holds one scalar per bucket and head, and it is *shared across all layers*: T5 computes the bias once, in the first layer, and adds the same $n_h \times n \times n$ tensor in every layer {cite}`raffel2019exploring`. With $n_h = 8$ heads the whole model spends $32 \times 8 = 256$ parameters on position.

```python
import math

import torch
import torch.nn as nn


def relative_position_bucket(offset, bidirectional=True, num_buckets=32, max_distance=128):
    """T5's bucket for the offset m - m' between query position m and key position m'."""
    buckets = torch.zeros_like(offset)
    n = offset
    if bidirectional:
        num_buckets //= 2
        buckets = buckets + (n < 0).long() * num_buckets  # keys after the query use the upper half
        n = n.abs()
    else:
        n = n.clamp(min=0)  # a causal model only sees keys at or before the query
    max_exact = num_buckets // 2
    is_small = n < max_exact
    log_spaced = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()
    log_spaced = log_spaced.clamp(max=num_buckets - 1)
    return buckets + torch.where(is_small, n, log_spaced)


class RelativePositionBias(nn.Module):
    """T5 relative position bias: one learned scalar per (bucket, head), added to the attention scores.

    T5-style scalar bias (Shaw et al. use vectors). A model builds one instance and passes its output to
    every layer, so the table is shared across layers.
    """

    def __init__(self, num_heads, num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.table = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        offset = positions[:, None] - positions[None, :]  # [seq, seq], entry (m, m') = m - m'
        buckets = relative_position_bucket(offset, self.bidirectional, self.num_buckets, self.max_distance)
        return self.table(buckets).permute(2, 0, 1)  # [heads, seq, seq]


offsets = torch.tensor([0, 1, 7, 8, 12, 16, 64, 127, 128, 1000, -1, -12])
print("offset m - m':      ", offsets.tolist())
print("encoder bucket:     ", relative_position_bucket(offsets).tolist())
print("decoder bucket:     ", relative_position_bucket(offsets, bidirectional=False).tolist())
rpb = RelativePositionBias(num_heads=8)
bias = rpb(seq_len=16)  # inside attention: scores = Q @ K.transpose(-2, -1) / sqrt(d_k) + bias
print("bias shape:", tuple(bias.shape), "| parameters:", rpb.table.weight.numel())
```

% output

```text
offset m - m':       [0, 1, 7, 8, 12, 16, 64, 127, 128, 1000, -1, -12]
encoder bucket:      [0, 1, 7, 8, 9, 10, 14, 15, 15, 15, 17, 25]
decoder bucket:      [0, 1, 7, 8, 12, 16, 26, 31, 31, 31, 0, 0]
bias shape: (8, 16, 16) | parameters: 256
```

(sec-ch1-rope)=

#### Rotary Position Embedding (RoPE)

Rotary position embedding {cite}`su2021roformer` is used by LLaMA, Mistral, Qwen and Gemma. It has no parameters, and like the relative bias it acts *inside* attention: it rotates the queries and keys, never the values or the input embeddings. Split a query or key row vector of width $d_k$ into $d_k/2$ consecutive pairs $(x_{2i}, x_{2i+1})$, $0 \le i < d_k/2$, and rotate pair $i$ at position $m$ by the angle $m\,\omega_i$, with the frequencies $\omega_i = b^{-2i/d_k}$ of the sinusoidal encoding and base $b = 10000$. As a matrix, the rotation at position $m$ is block diagonal:

```{math}
:label: eq-ch1-rope
\mathcal{R}_m = \begin{pmatrix}
\cos m\omega_0 & -\sin m\omega_0 & & & \\
\sin m\omega_0 & \cos m\omega_0 & & & \\
& & \ddots & & \\
& & & \cos m\omega_{d_k/2-1} & -\sin m\omega_{d_k/2-1} \\
& & & \sin m\omega_{d_k/2-1} & \cos m\omega_{d_k/2-1}
\end{pmatrix},
\qquad \tilde{q}_m = q_m \mathcal{R}_m^\top, \quad \tilde{k}_{m'} = k_{m'} \mathcal{R}_{m'}^\top
```

The score between the rotated query and key is $\tilde{q}_m \tilde{k}_{m'}^\top = q_m \mathcal{R}_m^\top \mathcal{R}_{m'} k_{m'}^\top$. Block-diagonal matrices multiply block by block, so it is enough to look at one $2 \times 2$ block. Write $\varphi = m\omega_i$ and $\varphi' = m'\omega_i$. The transpose of a rotation by $\varphi$ is the rotation by $-\varphi$, and the angle-difference formulas $\cos\varphi\cos\varphi' + \sin\varphi\sin\varphi' = \cos(\varphi' - \varphi)$ and $\cos\varphi\sin\varphi' - \sin\varphi\cos\varphi' = \sin(\varphi' - \varphi)$ give

$$\begin{pmatrix} \cos\varphi & \sin\varphi \\ -\sin\varphi & \cos\varphi \end{pmatrix}
\begin{pmatrix} \cos\varphi' & -\sin\varphi' \\ \sin\varphi' & \cos\varphi' \end{pmatrix}
= \begin{pmatrix} \cos(\varphi' - \varphi) & -\sin(\varphi' - \varphi) \\ \sin(\varphi' - \varphi) & \cos(\varphi' - \varphi) \end{pmatrix}.$$

Since $\varphi' - \varphi = (m' - m)\,\omega_i$, every block of $\mathcal{R}_m^\top \mathcal{R}_{m'}$ is the corresponding block of $\mathcal{R}_{m' - m}$, and

$$\tilde{q}_m \tilde{k}_{m'}^\top = q_m \, \mathcal{R}_{m' - m} \, k_{m'}^\top .$$

The score depends on the two positions only through the offset $m' - m$: shifting the query and the key by the same amount leaves it unchanged. Taking $m' = m$ gives a rotation by angle zero, $\mathcal{R}_m^\top \mathcal{R}_m = I$, so each $\mathcal{R}_m$ is orthogonal and the rotation preserves the norms of the queries and keys {cite}`su2021roformer`. For a worked number, take $d_k = 2$, so the single pair has frequency $b^0 = 1$, and $q_m = k_{m'} = (1, 0)$: the rotated score is $\cos(m' - m)$, which is $\cos 2 = -0.416$ both for $m = 3, m' = 5$ and for $m = 10, m' = 12$. The function below applies {eq}`eq-ch1-rope` without forming the matrix, pair by pair:

```python
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary position embedding (Su et al., 2021), applied to queries and keys.

    x: [..., seq, d]; positions: [seq]. Consecutive pairs of features are rotated by an
    angle m * omega_i, where m is the position and omega_i = base^(-2i/d). Because a rotation
    preserves norms and q_m . k_m' then depends only on m - m', RoPE encodes relative position
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
torch.manual_seed(0)
q, k = torch.randn(64), torch.randn(64)


def rotated_score(m, m_prime):
    return (apply_rope(q[None], torch.tensor([m])) @ apply_rope(k[None], torch.tensor([m_prime])).T).item()


print(f"score at (3, 5):   {rotated_score(3, 5):.2f}")
print(f"score at (10, 12): {rotated_score(10, 12):.2f}")
print(f"score at (3, 6):   {rotated_score(3, 6):.2f}")
print(f"norm before and after rotation: {q.norm():.3f} {apply_rope(q[None], torch.tensor([7])).norm():.3f}")
```

% output

```text
score at (3, 5):   -10.28
score at (10, 12): -10.28
score at (3, 6):   -10.75
norm before and after rotation: 8.371 8.371
```

(sec-ch1-position-encoding-trade-offs)=

#### Position Encoding Trade-offs

```python
import matplotlib.pyplot as plt


def compare_position_encodings():
    """Extra parameters each positional scheme adds, as a function of maximum length."""
    seq_lens = [128, 512, 1024, 2048, 4096]
    d_model, num_heads, num_buckets = 512, 8, 32

    results = {
        "Sinusoidal": [0 for _ in seq_lens],                                        # no parameters
        "Learned absolute": [n * d_model for n in seq_lens],                        # one vector per position
        "Relative bias (T5)": [num_buckets * num_heads for _ in seq_lens],  # per (bucket, head), all layers
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


Most open-weight models since 2022 use RoPE {cite}`su2021roformer`; a few use ALiBi {cite}`press2022train`. Both encode relative position inside attention without per-position parameters; learned absolute embeddings are now mostly found in BERT- and GPT-2-era models.

```{admonition} Embeddings and Positional Encoding
:class: note
The notebook ({ref}`nb-attn-positional`) visualises sinusoidal and learned encodings, implements a learned relative-position bias and a rotary (RoPE) transform, and asserts RoPE's two defining properties: rotation preserves vector norms, and query-key dot products depend only on the relative offset.
```

(sec-ch1-advanced-embedding-implementation)=

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
        """N(0, 1/d), so the sqrt(d)-scaled embedding has unit variance, like the positional encoding."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=self.d_model**-0.5)

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

(sec-ch1-attention-mechanisms)=

## 2. Attention Mechanisms

(sec-ch1-attention-revolution)=

### 2.1 The Attention Revolution

Attention mechanisms revolutionized sequence modeling by allowing models to focus on relevant parts of the input when making predictions. Unlike RNNs that process sequences sequentially, attention enables parallel computation and direct connections between any two positions.

```{admonition} Interactive Exploration
:class: note
The notebook implements scaled dot-product and multi-head attention in eval mode, asserts that every row of attention weights sums to 1, and plots per-head attention maps and entropies ({ref}`nb-attn-attention`).
```

(sec-ch1-self-attention-fundamentals)=

#### Self-Attention Fundamentals

Self-attention computes a weighted representation of the input sequence, where weights indicate the relevance of each position to the current position. Given input embeddings $X \in \mathbb{R}^{n \times d}$:

1. **Query, Key, Value projections**:

   $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. **Attention output**:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The scaling factor $\sqrt{d_k}$, where $d_k$ is the per-head key dimension, keeps the dot products at unit variance so the softmax does not saturate {cite}`vaswani2017attention`.

(sec-ch1-multi-head-attention)=

#### Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to attend to different types of relationships:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_{n_h})W_O, \qquad W_O \in \mathbb{R}^{n_h d_v \times d}$$

where each head projects the input $X$ with its own matrices:

$$\text{head}_i = \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}), \qquad W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d \times d_k}, \quad W_V^{(i)} \in \mathbb{R}^{d \times d_v}, \quad 1 \le i \le n_h$$

with $d_k = d_v = d/n_h$, so the heads together cost as much as one head of width $d$. The single projection $W_Q$ of the previous equations is the $n_h$ matrices $W_Q^{(i)}$ placed side by side, $W_Q = [W_Q^{(1)} \cdots W_Q^{(n_h)}] \in \mathbb{R}^{d \times d}$, and likewise for $W_K$ and $W_V$; that is how the code below computes all heads with one matrix product and then splits the columns into heads. For cross-attention the queries come from one sequence and the keys and values from another, so the key length may differ from the query length; the listing below shows both cases.

(sec-ch1-attention-patterns-interpretability)=

### 2.2 Attention Patterns and Interpretability

Different attention heads learn to capture various linguistic phenomena:

- **Syntactic heads**: Focus on grammatical relationships (subject-verb, modifier-noun)
- **Semantic heads**: Capture semantic similarity and coreference
- **Positional heads**: Attend to specific relative positions

Visualization tools like attention maps provide insights into model behavior, though interpreting attention patterns requires caution {cite}`jain2019attention`.

```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_k]
            key: [batch_size, seq_len, d_k]  
            value: [batch_size, seq_len, d_v]
            mask: [batch_size, seq_len, seq_len] or None
            
        Returns:
            output: [batch_size, seq_len, d_v]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with n_h heads of width d_k = d_model / n_h.

    w_q holds the per-head matrices W_Q^(i) side by side as its columns (likewise w_k and w_v), so
    splitting X W_Q into n_h blocks of d_k columns gives X W_Q^(i) for every head at once.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V (all heads at once) and the output projection W_O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """query: [batch, n_q, d_model]; key and value: [batch, n_k, d_model].

        mask: True where attention is allowed, broadcastable to [batch, heads, n_q, n_k]; a causal
        mask is [n_q, n_k], a padding mask [batch, 1, 1, n_k].
        Returns the output [batch, n_q, d_model] and the weights [batch, heads, n_q, n_k].
        """
        batch_size, n_q, n_k = query.size(0), query.size(1), key.size(1)

        # 1. Project, then split the d_model columns into heads: [batch, heads, length, d_k]
        Q = self.w_q(query).view(batch_size, n_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, n_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, n_k, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Attention in every head at once
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 3. Concatenate the heads and apply W_O
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, n_q, self.d_model)
        return self.w_o(attn_output), attn_weights


torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, num_heads=8).eval()
x, memory = torch.randn(2, 5, 64), torch.randn(2, 7, 64)
out, weights = mha(x, x, x)  # self-attention
print("self-attention: ", tuple(out.shape), tuple(weights.shape))
out, weights = mha(x, memory, memory)  # queries from x, keys and values from a longer sequence
print("cross-attention:", tuple(out.shape), tuple(weights.shape))
```

% output

```text
self-attention:  (2, 5, 64) (2, 8, 5, 5)
cross-attention: (2, 5, 64) (2, 8, 5, 7)
```

(sec-ch1-transformer-architecture)=

## 3. Transformer Architecture

(sec-ch1-complete-architecture)=

### 3.1 The Complete Architecture

The transformer architecture {cite}`vaswani2017attention` combines attention mechanisms with several key components to create a powerful sequence-to-sequence model.

(sec-ch1-layernorm)=

#### Layer Normalization

Layer normalization stabilizes training by normalizing inputs across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed across the feature dimension (with a small $\epsilon$ added inside the square root of $\sigma$ for numerical stability), and $\gamma$, $\beta$ are learned per-feature scale and shift.

(sec-ch1-residual-connections)=

#### Residual Connections

Residual connections help gradient flow in deep networks:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

(sec-ch1-feed-forward-networks)=

#### Feed-Forward Networks

Each transformer layer includes a position-wise feed-forward network:

$$\text{FFN}(x) = \max(0, xW_{\text{up}})W_{\text{down}}$$

with $W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$. Bias terms are left out of the formulas in this chapter. The network provides non-linear transformations, and the original transformer sets the hidden width $d_{\text{ff}} = 4d$.

(sec-ch1-architectural-variants)=

### 3.2 Architectural Variants

While the original transformer was designed for machine translation, different variants emerged for language modeling:

- **GPT** {cite}`radford2018improving`: Decoder-only architecture with causal masking
- **BERT** {cite}`devlin2018bert`: Encoder-only architecture with bidirectional attention
- **T5** {cite}`raffel2019exploring`: Encoder-decoder architecture treating all tasks as text-to-text

```python
# pseudo-code: uses MultiHeadAttention from the attention listing above
from typing import Optional

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Complete transformer block with attention and feed-forward layers."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

```{admonition} Transformer Architecture Implementation
:class: tip
The notebook ({ref}`nb-attn-transformer`) assembles one post-LN transformer block from multi-head attention, a position-wise feed-forward network, layer normalization and residual connections, and shows what layer normalization does to activations of different scales. It does not train a model; that is left as an exercise.
```

(sec-ch1-modern-block)=

### 3.3 What Changed Since 2017

The block above is the 2017 design. Every widely used LLM since GPT-2 differs from it in a handful of ways, all aimed at training stability or inference cost:

- **Pre-LN instead of post-LN.** Normalization is applied *before* each sublayer, so the residual stream is never normalized: $x \leftarrow x + \mathrm{MHA}(\mathrm{Norm}(x))$. Post-LN transformers have large gradients at initialization and need learning-rate warm-up to train at all; pre-LN removes most of that sensitivity {cite}`xiong2020layer`.
- **RMSNorm instead of LayerNorm.** Dropping the mean subtraction and the bias, $\mathrm{RMSNorm}(x) = \gamma \odot x / \sqrt{\lVert x \rVert_2^2 / d + \epsilon}$, is cheaper and trains just as well {cite}`zhang2019root`. LLaMA, Mistral, Qwen and Gemma all use it.
- **Gated activations.** The feed-forward network becomes $\mathrm{FFN}(x) = (\mathrm{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ (SwiGLU), with the hidden width reduced to about $\tfrac{8}{3}d$ to keep the parameter count unchanged {cite}`shazeer2020glu`.
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

(sec-ch1-language-modeling-objective)=

## 4. Language Modeling Objective

(sec-ch1-next-token-prediction)=

### 4.1 Next-Token Prediction

Modern LLMs are trained on the **causal language modeling** objective, predicting the next token given previous tokens:

$$p_\theta(x_i \mid x_1, x_2, \dots, x_{i-1}) = p_\theta(x_i \mid x_{<i})$$

This autoregressive formulation enables text generation and captures the statistical structure of language.

(sec-ch1-causal-masking)=

#### Causal Masking

To prevent the model from "seeing the future," causal masking ensures attention only flows from earlier to later positions:

```python
import torch

def create_causal_mask(seq_length: int) -> torch.Tensor:
    """Create a causal mask to prevent attention to future tokens."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0  # True for allowed positions, False for masked
```

(sec-ch1-training-process)=

### 4.2 Training Process

Training maximizes the likelihood of the training data, which is the same as minimizing the mean negative log-likelihood per token over the $T$ tokens of the data, in nats:

$$\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(x_i \mid x_{<i})$$

The model outputs a probability distribution over the vocabulary:

$$p_\theta(\cdot \mid x_{<i}) = \text{softmax}(h^{(L)}_{i-1} W_{\text{out}})$$

where $h^{(L)}_{i-1}$ is the hidden state of the last layer $L$ at position $i-1$, which under causal masking has seen exactly $x_{<i}$. GPT-2 and later models have no bias in this projection.

(sec-ch1-perplexity)=

### 4.3 Evaluation Metrics

**Perplexity** measures how well the model predicts the test data:

$$\text{Perplexity} = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log p_\theta(x_i \mid x_{<i})\right) = \exp(\mathcal{L})$$

Perplexity was introduced as the effective branching factor of a language model {cite}`jelinek1977perplexity`; lower is better. It correlates only loosely with downstream task accuracy, so large-scale evaluations report both {cite}`liang2022holistic`.

```{admonition} Language Modeling in Practice
:class: note
The notebook ({ref}`nb-attn-lm`) builds the causal mask, applies it inside attention, adds a language-model head, and computes cross-entropy and perplexity for an untrained model so you can see what "no better than guessing" looks like. Optimization, schedules and gradient flow are in the Chapter 2 notebook ({ref}`nb-math-optimizers`, {ref}`nb-math-backprop`).
```

(sec-ch1-historical-context)=

## 5. Historical Context

(sec-ch1-evolution-rnns)=

### 5.1 Evolution from RNNs

The path to transformers involved several key developments:

**Recurrent Neural Networks** initially dominated sequence modeling but suffered from:

- Vanishing gradient problems in long sequences
- Sequential processing preventing parallelization
- Limited ability to capture long-range dependencies

**Long Short-Term Memory (LSTM)** {cite}`hochreiter1997long` networks addressed vanishing gradients but remained sequential.

**Attention mechanisms** {cite}`bahdanau2014neural` first appeared in machine translation, allowing models to focus on relevant source positions when generating target words.

(sec-ch1-transformer-breakthrough)=

### 5.2 The Transformer Breakthrough

"Attention Is All You Need" {cite}`vaswani2017attention` demonstrated that attention alone could achieve state-of-the-art results without recurrence or convolution. Key innovations included:

- Self-attention for capturing dependencies regardless of distance
- Parallel processing enabling efficient training
- Multi-head attention for capturing diverse relationships

(sec-ch1-scaling-language-models)=

### 5.3 Scaling to Language Models

The transformer's success in machine translation led to its adoption for language modeling:

- **GPT** (2018) showed the power of unsupervised pre-training
- **BERT** (2018) demonstrated bidirectional representations
- **GPT-2** (2019) showed zero-shot transfer to many tasks from language modeling alone {cite}`radford2019language`
- **GPT-3** (2020) achieved few-shot learning with 175B parameters {cite}`brown2020language`

This progression established the foundation for today's LLMs and their remarkable capabilities.

(sec-ch1-exercises)=

## Exercises

1. **Tokenization Analysis**: Compare BPE and unigram tokenization (both available in SentencePiece) on multilingual text. How do vocabulary sizes affect model performance?

1. **Attention Visualization**: Implement attention map visualization and analyze patterns in a small transformer model.

1. **Positional Encoding Experiment**: Compare sinusoidal, learned, and relative positional encodings on a language modeling task.

1. **Architecture Comparison**: Implement encoder-only, decoder-only, and encoder-decoder variants. When might each be preferred?

(sec-ch1-further-reading)=

## Further Reading

(sec-ch1-further-reading-learning-path-beginners)=

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar - Visual explanations of transformer concepts
1. [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng - Comprehensive attention mechanisms guide
1. [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Video series building transformers from scratch

**Mathematical Foundations:**

4. [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) by Sebastian Raschka
5. [Linear Algebra for Machine Learning](https://www.deeplearningbook.org/contents/linear_algebra.html) - Deep Learning Book Chapter 2

(sec-ch1-further-reading-foundational-papers)=

### Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite}`vaswani2017attention`: The original transformer paper
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) {cite}`radford2018improving`: GPT and the decoder-only architecture
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) {cite}`devlin2018bert`: BERT and bidirectional representations

(sec-ch1-further-reading-technical-implementation-resources)=

### Technical Implementation Resources

**Code Implementations:**

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - Line-by-line implementation with explanations
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) - Production-ready transformer implementations
- [MinGPT](https://github.com/karpathy/minGPT) by Andrej Karpathy - Minimal GPT implementation for learning

**Interactive Learning:**

- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) - Interactive visualization tool
- [Tiktokenizer](https://tiktokenizer.vercel.app/) - Explore how GPT and other tokenizers split text
- [Attention Patterns](https://github.com/jessevig/bertviz) - Visualize attention in BERT and GPT models

(sec-ch1-further-reading-advanced-topics-recent-research)=

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

(sec-ch1-further-reading-books-comprehensive-resources)=

### Books and Comprehensive Resources

1. **[Natural Language Processing with Transformers](https://github.com/nlp-with-transformers/notebooks)** by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
1. **[Deep Learning](https://www.deeplearningbook.org/)** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Mathematical foundations
1. **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** by Dan Jurafsky and James H. Martin - NLP fundamentals

(sec-ch1-further-reading-practical-learning-projects)=

### Practical Learning Projects

1. **Build a Miniature GPT** - Implement transformer from scratch using the notebook examples
1. **Tokenization Comparison** - Compare different tokenization strategies on multilingual text
1. **Attention Visualization** - Create tools to visualize attention patterns in pre-trained models
1. **Scaling Experiments** - Study how model performance changes with size using smaller versions

(sec-ch1-further-reading-documentation-apis)=

### Documentation and APIs

- [PyTorch Transformer Tutorial](https://docs.pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)

(sec-ch1-further-reading-research-communities-updates)=

### Research Communities and Updates

- [Hugging Face Papers](https://huggingface.co/papers) - Curated ML research papers

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
