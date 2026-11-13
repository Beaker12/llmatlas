# LLM Foundations

Large language models are transformer networks trained on very large text corpora to predict the next token. This chapter establishes the foundational concepts necessary to understand how these models work, their historical development, and the principles behind their performance.

(sec-ch1-chapter-overview)=

## Chapter Overview

A language model reads text as a sequence of integer token ids and outputs, at every position, a probability distribution over the next token. The chapter follows that path in order:

- **Tokenization** ({ref}`sec-ch1-tokenization`) cuts text into pieces from a fixed vocabulary, such as "un", "happ" and "iness", and maps each to an id. Subword pieces keep the vocabulary to tens or hundreds of thousands of entries while still spelling out any word.
- **Embeddings and positions** ({ref}`sec-ch1-embeddings-positional-encoding`) turn each id into a vector and give the model a way to tell positions apart. The relative schemes among them act on attention scores, so their subsection defines the few terms it needs and points ahead to the attention section.
- **Historical context** ({ref}`sec-ch1-historical-context`) traces language modeling from n-gram counts and recurrent networks to the transformer and the pretrained models built on it.
- **Attention** ({ref}`sec-ch1-attention-mechanisms`) lets every position compute a weighted average of the others, with weights that depend on how well their vectors match.
- **The transformer** ({ref}`sec-ch1-transformer-architecture`) stacks blocks of attention and feed-forward layers on a residual stream, and its parameters can be counted from four numbers.
- **The language-modeling objective** ({ref}`sec-ch1-language-modeling-objective`) trains the stack to predict each next token under a causal mask, and perplexity and bits per byte measure how well it does.

A companion page, {doc}`tokenization-deep-dive`, goes further into tokenizers used in practice.

Every symbol the book uses is listed on the {ref}`notation page <sec-notation>`, with a link to the section that introduces each symbol used only in this chapter.

(sec-ch1-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will be able to:

- Train byte-pair-encoding merges on a small corpus by hand and in code, and explain how byte fallback ({ref}`sec-ch1-tokenization`) and byte-level vocabularies ({ref}`sec-tdd-byte-level-bpe`) remove the need for an unknown token
- Compare sinusoidal, learned, relative-bias and rotary position schemes, and show that RoPE scores depend only on the offset between positions
- Place the transformer in the history of language modeling, from n-grams and recurrent networks to pretrained, instruction-tuned models
- Compute scaled dot-product and multi-head attention by hand, including the causal mask, and state their cost in time and memory
- Assemble a transformer block and a whole decoder-only model with their tensor shapes, and count their parameters
- Define the next-token loss, perplexity and bits per byte, convert between them, and choose a decoding rule

(sec-ch1-hands-on-learning)=

## Hands-On Learning

For interactive implementations and visualizations of the foundational concepts in this chapter, see the accompanying Jupyter notebook: [Attention Mechanisms Demo](examples/attention_mechanisms_demo.ipynb)

The notebook provides executable, assertion-checked code for:

- A word-level tokenizer and the BPE merge loop ({ref}`nb-attn-tokenization`)
- Token embeddings with sinusoidal, learned, relative-bias and rotary positions ({ref}`nb-attn-positional`)
- Scaled dot-product and multi-head attention with attention maps ({ref}`nb-attn-attention`)
- A post-LN transformer block ({ref}`nb-attn-transformer`)
- The causal mask, the next-token objective with perplexity, and a short character-level training run ({ref}`nb-attn-lm`)

(sec-ch1-tokenization)=

## 1. Tokenization

(sec-ch1-understanding-tokenization)=

### 1.1 Understanding Tokenization

A language model computes with numbers, so text is first split by **tokenization** into discrete units called tokens, each mapped to an integer id. The choice of units fixes the vocabulary size $|\mathcal{V}|$, the number of tokens every text costs, and which strings the model sees as single units.

{numref}`fig-ch1-tokenization-pipeline` follows one sentence through every stage of this chapter's first two sections.

```{figure} figures/tokenization_pipeline.*
:name: fig-ch1-tokenization-pipeline
:width: 87%
:alt: Five stages from top to bottom: normalization, pre-tokenization, subword model, vocabulary lookup and embedding lookup, each beside its result for the text The cat sat: three chunks, the tokens The, cat and sat, the ids 464, 3797 and 3332, and a 3 by 768 matrix.

From text to the model's input. Left, the stages; right, what each gives for "The cat
sat" with GPT-2's tokenizer (ids from tiktoken's `gpt2` encoding; Ġ is how GPT-2's
vocabulary writes a leading space; $d = 768$ is the smallest GPT-2).
```

(sec-ch1-token-definition)=

#### What Are Tokens?

A **token** is the fundamental unit of text that a language model processes. Tokens can represent:

- **Complete words**: "hello", "world", "artificial"
- **Subwords**: "hell", "o", "artific", "ial"
- **Characters**: "h", "e", "l", "l", "o"
- **Bytes**: Individual byte values (0-255)
- **Special symbols**: `<BOS>`, `<EOS>`, `<UNK>`, `<PAD>`

The choice of tokenization granularity creates fundamental trade-offs:

```{tabularcolumns} \Y{0.20}\Y{0.23}\Y{0.18}\Y{0.21}\Y{0.18}
```

| Granularity | Vocabulary Size    | Sequence Length | OOV Handling | Semantic Coherence |
| ----------- | ------------------ | --------------- | ------------ | ------------------ |
| Word-level  | Very Large (>100k) | Short           | Poor         | High               |
| Subword     | 32k-256k           | Medium          | Good         | Medium             |
| Character   | ~10²–10⁴ (script dependent) | Very Long | Good (unseen characters are still OOV) | Low |
| Byte-level  | Fixed (256)        | Longest         | Perfect      | Lowest             |

(sec-ch1-vocabulary-size-dilemma)=

#### The Vocabulary Size Dilemma

Vocabulary size directly impacts:

1. **Model parameters**: the embedding matrix holds $|\mathcal{V}|\,d$ parameters
1. **Computational cost**: Output projection scales with vocabulary size
1. **Memory requirements**: Larger vocabularies require more memory
1. **Out-of-vocabulary handling**: Smaller vocabularies struggle with rare words

For example, $|\mathcal{V}| = 50{,}000$ and $d = 4{,}096$ give $|\mathcal{V}|\,d = 204{,}800{,}000$ embedding parameters before any transformer block.

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
print(tokens)
```

% output

```text
['hello', ',', 'world', '!', 'how', 'are', 'you', '?']
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
print(tokens)
```

% output

```text
['H', 'e', 'l', 'l', 'o']
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

Byte pair encoding began as a data-compression technique {cite}`gage1994new` that, in the words of {cite:t}`sennrich2015neural`, "iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte"; they adapted it to word segmentation. It starts from characters and iteratively merges the most frequent adjacent pair:

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
# an unseen word splits into learned subwords and characters
print(f"Tokens: {bpe.encode('helper')}")
```

% output

```text
Merges: [('h', 'e'), ('he', 'l'), ('hel', 'l'), ('hell', 'o')]
Tokens: ['hel', 'p', 'e', 'r', '</w>']
```

{numref}`fig-ch1-bpe-merges` shows the corpus after each merge: frequent pairs grow into whole words (hello), while a word seen once (world) stays split into characters.

```{figure} figures/bpe-merges.*
:name: fig-ch1-bpe-merges
:width: 82%
:alt: Five rows of boxed symbols for the words hello, world, help and held, each ending in an end-of-word marker. Row by row the highlighted symbol grows from he to hel to hell to hello, while world never changes.

The four merges of the listing on its corpus (hello twice, world, help, held), drawn by
`figures/make_figures.py` from the listing's own `SimpleBPE`. Each row is the corpus
after one more merge; the symbol just created is highlighted. Merge 1 picks h + e over
e + l, which also occurs 4 times, because `max` returns the first pair counted.
```

**Other subword algorithms**:

1. **Original BPE** {cite}`sennrich2015neural`: merges the most frequent adjacent pair
1. **WordPiece** {cite}`schuster2012japanese`: Google's variant, later used in its translation system {cite}`wu2016google`, that merges the pair which most increases the training corpus likelihood
1. **Unigram LM** {cite}`kudo2018subword`: starts from a large candidate vocabulary and prunes it, keeping the segmentation with the highest probability
1. **SentencePiece** {cite}`kudo2018sentencepiece` is a *toolkit*, not an algorithm: it implements both BPE and unigram on raw text without pre-tokenization

(sec-ch1-sentencepiece)=

#### SentencePiece

SentencePiece {cite}`kudo2018sentencepiece` treats text as raw Unicode sequences without pre-tokenization:

```python
import os
import tempfile
import textwrap

import sentencepiece as spm

# A small corpus: a few English sentences repeated, plus one line of Chinese
# the model will rarely see.
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
        # characters outside the learned set become UTF-8 byte pieces, not <unk>
        byte_fallback=True,
        normalization_rule_name="identity",
        num_threads=1,
        minloglevel=2,
    )
    sp = spm.SentencePieceProcessor(model_file=os.path.join(tmp, "tokenizer.model"))

text = "Hello, world! 你好世界"
pieces = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)
# lossless, because unseen characters fall back to bytes
assert sp.decode_ids(ids) == text
assert sp.unk_id() not in ids
print(textwrap.fill(f"Pieces: {pieces}", width=88, subsequent_indent="        "))
print(f"{len(ids)} ids; decoded text equals the input: {sp.decode_ids(ids) == text}")
```

% output

```text
Pieces: ['▁He', 'llo', ',', '▁w', 'orld', '!', '▁', '<0xE4>', '<0xBD>', '<0xA0>',
        '<0xE5>', '<0xA5>', '<0xBD>', '<0xE4>', '<0xB8>', '<0x96>', '<0xE7>', '<0x95>',
        '<0x8C>']
19 ids; decoded text equals the input: True
```

With byte fallback, the four Chinese characters, absent from the learned pieces, become twelve UTF-8 byte pieces instead of `<unk>`, and decoding returns the input. Without it, the same model would map them to the unknown id. SentencePiece refuses a vocabulary size larger than the corpus can fill. The toy corpus supports a few hundred pieces. For the common 32,000, the first 100,000 bytes of the raw WikiText-103 training split (English Wikipedia articles; Hugging Face dataset `Salesforce/wikitext`, configuration `wikitext-103-raw-v1`, rows of the first shard concatenated) are refused by both model types. With SentencePiece 0.2.3, `vocab_size=32000` and every other setting at its default, the error asks for at most 14,602 pieces with `bpe` and 2,743 with `unigram`. The first 1,000,000 bytes are enough for `bpe` and still refused for `unigram` (at most 10,599). These limits depend on the file variant, the cut and the trainer settings, so other choices give somewhat different values; the live test `test_sentencepiece_wikitext_vocabulary_limits` in `tests/test_ch01_snippets.py` repeats the four runs and asserts the numbers. Production tokenizers train on far more text, so that the pieces reflect real frequencies.

**Key advantages**:

- Language-agnostic (no pre-tokenization assumptions)
- Handles any Unicode text, losslessly only with byte fallback or full character coverage
- Consistent treatment of whitespace, written as the visible symbol `▁`
- Reversible tokenization under the same condition

For vocabulary construction, special tokens, cross-lingual issues and a full tokenizer implementation, see {doc}`tokenization-deep-dive`.

```{admonition} Tokenization Implementation
:class: tip
The notebook's tokenization section ({ref}`nb-attn-tokenization`) builds a tiny
word-level vocabulary and runs the BPE merge loop step by step, printing the splits
after every merge. For WordPiece, unigram, byte-level BPE and SentencePiece in practice,
see {ref}`sec-tdd-subword-algorithms`.
```

(sec-ch1-embeddings-positional-encoding)=

## 2. Embeddings and Positional Encoding

How the embedding width is chosen, how to inspect a trained embedding table, what each positional scheme costs and a complete embedding layer are in {ref}`sec-tdd-embeddings-in-practice`.

(sec-ch1-embedding-spaces)=

### 2.1 Embedding Spaces

Once tokenized, each token is mapped to a high-dimensional vector through an **embedding matrix** $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$, where $\mathcal{V}$ is the vocabulary, $|\mathcal{V}|$ its size and $d$ the embedding dimension. The vector for token $x_i$ is row $x_i$ of $W_E$:

$$W_E[x_i] \in \mathbb{R}^{d}$$

These embeddings are learned during training to capture semantic relationships. Similar tokens develop similar vector representations, enabling the model to generalize across related concepts {cite}`mikolov2013efficient`.

(sec-ch1-embedding-initialization-strategies)=

#### Embedding Initialization Strategies

The initialization and any scaling in the forward pass have to be chosen together. The original transformer multiplies the embedding by $\sqrt{d}$ before adding the sinusoidal encoding {cite}`vaswani2017attention`. If the table is drawn from $\mathcal{N}(0, 1/d)$, each coordinate of the scaled embedding has variance $1/d \cdot d = 1$, the same order as the sinusoidal encoding, whose coordinates average $\sin^2 = \cos^2 = 1/2$. Drawing the table with Xavier's rule instead, whose variance $2/(|\mathcal{V}| + d)$ is tiny for a 50,000-row table, would leave the position signal $(|\mathcal{V}| + d)/(4d) \approx 24.7$ times stronger in power than the token signal for $|\mathcal{V}| = 50{,}000$ and $d = 512$; drawing it from $\mathcal{N}(0, 1)$ and then scaling would make the token signal $2d = 1{,}024$ times stronger.

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

Many architectures share the input embedding matrix with the output projection {cite}`press2017using`. GPT-2's released code computes the output scores with its token-embedding matrix `wte` {cite}`radford2019language`, and Gemma 2's architecture table lists "Tied embedding: yes" for all three sizes {cite}`gemma2024gemma2`. Mistral 7B keeps the two separate (its released configuration sets `tie_word_embeddings` to false) {cite}`jiang2023mistral`, and so does the original LLaMA, whose reference code builds them as separate modules {cite}`touvron2023llama`; by contrast, the Llama 3.2 model card marks "Shared Embeddings: Yes" for its 1B and 3B text models {cite}`meta2024llama32card`:

```python
# pseudo-code: `TransformerStack` stands for the model body;
# illustrates weight tying only
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

- Saves $|\mathcal{V}|\,d$ parameters, 38,597,376 for GPT-2 small ($|\mathcal{V}| = 50{,}257$, $d = 768$)
- Enforces consistency between input and output representations
- Often improves performance on language modeling tasks

(sec-ch1-subword-embedding-composition)=

#### Subword Embedding Composition

For subword tokens, embeddings must compose meaningfully:

```python
# pseudo-code: `embedding_matrix` and `token_to_id` come from a trained model
# Example: How "unhappiness" might be tokenized and embedded
text = "unhappiness"
# tiktoken's gpt2, cl100k_base and o200k_base all give this split
tokens = ["un", "h", "appiness"]

# Each token gets its own embedding
# 3 vectors of size [d_model]
token_embeddings = [embedding_matrix[token_to_id[t]] for t in tokens]

# The model sees these as separate tokens in sequence
# The split does not follow the morphemes un + happi + ness;
# the model must still learn that "un" negates
# This happens through attention and context
```

(sec-ch1-positional-encodings)=

### 2.2 Positional Encodings

Attention by itself does not see the order of its inputs, so positional information must be supplied. Word order matters in language:

- "The cat sat on the mat"
- "The mat sat on the cat"

Without positional information and without a mask, self-attention treats each sentence as a multiset of tokens: permuting the inputs permutes the outputs in the same way and changes nothing else, so these two sentences, which hold the same tokens, get the same set of output vectors. A causal decoder is not completely order-blind even without positions, because its causal mask ({ref}`sec-ch1-causal-masking`) lets position $m$ attend to exactly $m + 1$ tokens, positions $0$ to $m$; but that is a weak signal, and every model below supplies positions explicitly.

The sinusoidal and learned schemes below both add a position vector to each token vector, as in {numref}`fig-ch1-embedding-composition`.

```{figure} figures/embedding_composition.*
:name: fig-ch1-embedding-composition
:width: 94%
:alt: Token ids go through a lookup in the token embedding matrix and positions go through a positional encoding; the two n by d results are added to give the input X.

Absolute positions are added to the token embeddings before the first block. RoPE
instead acts inside attention ({ref}`sec-ch1-rope`), and the relative bias adds to the
attention scores.
```

(sec-ch1-sinusoidal-positions)=

#### Sinusoidal Positional Encoding

The original transformer uses sinusoidal positional encodings {cite}`vaswani2017attention`:

$$\operatorname{PE}_{(m, 2i)} = \sin\left(\frac{m}{10000^{2i/d}}\right)$$
$$\operatorname{PE}_{(m, 2i+1)} = \cos\left(\frac{m}{10000^{2i/d}}\right)$$

where:

- $\operatorname{PE}_{(m, j)}$ is component $j$ of the positional encoding added to the token embedding at position $m$
- $m$ is the position in the sequence
- $i$ indexes the sine-cosine pairs, $0 \le i < d/2$
- $d$ is the embedding dimension

**Key properties**:

- Deterministic (same encoding for same position)
- Bounded values (between -1 and 1)
- Different frequencies for different dimensions
- A fixed offset is a linear map, the property the paper gives for choosing this encoding {cite}`vaswani2017attention`. With $\omega_i = 10000^{-2i/d}$, pair $i$ at position $m'$ is pair $i$ at position $m$ rotated by the angle $(m' - m)\,\omega_i$, a rotation that depends on the offset and not on $m$: $\bigl(\sin m'\omega_i,\ \cos m'\omega_i\bigr) = \bigl(\sin m\omega_i,\ \cos m\omega_i\bigr)\begin{pmatrix}\cos (m'-m)\omega_i & -\sin (m'-m)\omega_i \\ \sin (m'-m)\omega_i & \cos (m'-m)\omega_i\end{pmatrix}$, by the angle-addition formulas. RoPE ({ref}`sec-ch1-rope`) applies the same rotations to queries and keys instead of adding them to the input
- Defined for any position, so a model *can* be run past its training length, although in practice quality degrades quickly there {cite}`press2022train`

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
    print(
        f"encoding table: {tuple(pe.shape)}, "
        f"values from {pe.min():.4f} to {pe.max():.4f}"
    )
    print(f"position 0: {pe[0, :4].tolist()}")

    plt.figure(figsize=(7, 3.6))
    plt.imshow(pe.T.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Encoding value")
    plt.title("Sinusoidal positional encodings")
    plt.xlabel("Position $m$")
    plt.ylabel("Dimension")
    plt.show()

    # Show how encoding changes with position:
    # the frequency falls as the dimension grows
    plt.figure(figsize=(6, 3.2))
    for dim, style in ((0, "-"), (8, "--"), (16, "-."), (32, ":")):
        plt.plot(pe[:50, dim].numpy(), linestyle=style, label=f"Dimension {dim}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=11)
    plt.tick_params(labelsize=11)
    plt.title("Encoding value by position", fontsize=12)
    plt.xlabel("Position $m$", fontsize=11.5)
    plt.ylabel("Encoding value", fontsize=11.5)
    plt.show()


# Visualize the encodings
visualize_positional_encoding()
```

% output

```text
encoding table: (100, 64), values from -1.0000 to 1.0000
position 0: [0.0, 1.0, 0.0, 1.0]
```

```{figure} figures/sinusoidal-heatmap.*
:name: fig-ch1-sinusoidal-heatmap
:width: 83%
:alt: Heatmap of 64 dimensions by 100 positions: the lowest dimensions alternate red and blue every few positions, higher dimensions change ever more slowly, and the bottom rows, the highest dimensions, are almost constant.

The encoding table the listing prints, $m < 100$ and $d = 64$, drawn by its first plot.
Each pair of dimensions is a sine and cosine whose wavelength grows with the dimension
index.
```

```{figure} figures/sinusoidal-lines.*
:name: fig-ch1-sinusoidal-lines
:width: 98%
:alt: Four curves over positions 0 to 49: dimension 0 oscillates fast, dimension 8 more slowly, dimension 16 completes less than one cycle and dimension 32 rises almost linearly.

The listing's second plot: dimensions 0, 8, 16 and 32 over the first 50 positions. Their
angular frequencies are 1, 0.32, 0.1 and 0.01 radians per position.
```

Position 0 is $(0, 1, 0, 1, \dots)$ because $\sin 0 = 0$ and $\cos 0 = 1$, and every entry stays within $[-1, 1]$.

{numref}`fig-ch1-sinusoidal-heatmap` shows the whole table and {numref}`fig-ch1-sinusoidal-lines` a few of its rows: low dimensions change fast enough to tell neighboring positions apart, high ones slowly enough to separate distant ones.


(sec-ch1-learned-positions)=

#### Learned Positional Embeddings

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


# GPT-2 small's sizes: 1,024 positions of width 768
pos = LearnedPositionalEmbedding(1024, 768)
print(f"position parameters: {sum(p.numel() for p in pos.parameters()):,}")
x = torch.zeros(2, 10, 768)  # a batch of 2 sequences of 10 token embeddings
print("output shape:", tuple(pos(x).shape))
try:
    pos(torch.zeros(1, 1025, 768))
except IndexError as err:
    print("a sequence of 1,025:", type(err).__name__)
```

% output

```text
position parameters: 786,432
output shape: (2, 10, 768)
a sequence of 1,025: IndexError
```


**Learned vs. Sinusoidal**:

| Aspect         | Sinusoidal    | Learned                     |
| -------------- | ------------- | --------------------------- |
| Parameters     | 0 additional  | $n_{ctx}\,d$               |
| Beyond training length | Defined, but degrades in practice | Undefined (no embedding) |
| Used by        | Original transformer | BERT {cite}`devlin2018bert`, GPT-2 (its code's `wpe` table) {cite}`radford2019language` |
| Flexibility    | Fixed pattern | Adaptive to data            |

(sec-ch1-relative-bias)=

#### Relative Position Bias (T5)

The relative schemes of this subsection and the next two act on attention scores, which {ref}`sec-ch1-self-attention-fundamentals` defines in full. Three facts suffice here. Each token position $m$ has a query row $q_m$ and a key row $k_m$ of width $d_k$, and a value row $v_m$, all computed from that token's vector by learned linear maps. The attention score of query position $m$ for key position $m'$ is the dot product $q_m \cdot k_{m'}$, divided by $\sqrt{d_k}$ in the standard form, and a softmax over $m'$ turns the scores of one query into weights that sum to 1; the output is the weighted average of the value rows. A head is one such set of queries, keys, values and scores, and a layer runs $n_h$ heads side by side ({ref}`sec-ch1-multi-head-attention`), each with its own $n \times n$ score matrix for $n$ tokens.

Instead of adding absolute positions to the input, a relative scheme makes attention depend on the offset $m - m'$ between the query position $m$ and the key position $m'$. {cite:t}`shaw2018self` add learned relative-position *vectors* to the keys and values. T5 {cite}`raffel2019exploring` simplifies this to one learned scalar per head, added to the attention score of every query-key pair:

$$S_{mm'} = q_m \cdot k_{m'} + \operatorname{rb}(\operatorname{bucket}(m - m'))$$

Here $\operatorname{rb}$ is a learned table of scalars and $\operatorname{bucket}$ maps an offset to a row of it; both are described below. T5 does not divide the dot product by $\sqrt{d_k}$; its reference implementation in the `transformers` library {cite}`wolf2020transformers` sets the score scale of `T5Attention` to 1.

The function $\operatorname{bucket}$ maps an offset to one of 32 buckets. Small offsets get a bucket each; larger ones share buckets whose width grows logarithmically, up to an offset of 128, beyond which all offsets share the last bucket. In the encoder, which attends in both directions, half of the buckets are for keys before or at the query and half for keys after it. For keys at or before the query, offsets 0 to 7 have their own buckets, and an offset $8 \le m - m' < 128$ goes to bucket

$$8 + \left\lfloor 8 \, \frac{\ln\big((m - m')/8\big)}{\ln 16} \right\rfloor,$$

so offset 12 lands in bucket $8 + \lfloor 8 \times 0.146 \rfloor = 9$, offset 16 in bucket 10, offset 64 in bucket 14, and every offset from 128 on in bucket 15. A key after the query uses the same rule on $m' - m$ plus 16. A causal decoder never sees keys after the query, so it spends all 32 buckets on the past: offsets 0 to 15 exactly, then 16 log-spaced buckets up to 128. The table $\operatorname{rb}$ holds one scalar per bucket and head, and within a stack it is *shared across all layers*: T5 computes the bias once in the first layer of the encoder and once in the first layer of the decoder, and adds the same $n_h \times n \times n$ tensor in every layer of that stack {cite}`raffel2019exploring`. T5-small's released configuration has 32 buckets and $n_h = 8$ heads, so each stack spends $32 \times 8 = 256$ parameters on position and the encoder-decoder model $2 \times 256 = 512$.

```python
import math

import torch
import torch.nn as nn


def relative_position_bucket(
    offset, bidirectional=True, num_buckets=32, max_distance=128
):
    """T5's bucket for the offset m - m' of query position m and key position m'."""
    buckets = torch.zeros_like(offset)
    n = offset
    if bidirectional:
        num_buckets //= 2
        # keys after the query use the upper half
        buckets = buckets + (n < 0).long() * num_buckets
        n = n.abs()
    else:
        n = n.clamp(min=0)  # a causal model only sees keys at or before the query
    max_exact = num_buckets // 2
    is_small = n < max_exact
    log_ratio = torch.log(n.float() / max_exact) / math.log(max_distance / max_exact)
    log_spaced = max_exact + (log_ratio * (num_buckets - max_exact)).long()
    log_spaced = log_spaced.clamp(max=num_buckets - 1)
    return buckets + torch.where(is_small, n, log_spaced)


class RelativePositionBias(nn.Module):
    """T5 relative position bias: one learned scalar per (bucket, head).

    The scalars are added to the attention scores. T5-style scalar bias (Shaw et al.
    use vectors). A model builds one instance and passes its output to every layer, so
    the table is shared across layers.
    """

    def __init__(self, num_heads, num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.table = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        # [seq, seq], entry (m, m') = m - m'
        offset = positions[:, None] - positions[None, :]
        buckets = relative_position_bucket(
            offset, self.bidirectional, self.num_buckets, self.max_distance
        )
        return self.table(buckets).permute(2, 0, 1)  # [heads, seq, seq]


offsets = torch.tensor([0, 1, 7, 8, 12, 16, 64, 127, 128, 1000, -1, -12])
print("offset m - m':      ", offsets.tolist())
print("encoder bucket:     ", relative_position_bucket(offsets).tolist())
decoder_buckets = relative_position_bucket(offsets, bidirectional=False)
print("decoder bucket:     ", decoder_buckets.tolist())
rpb = RelativePositionBias(num_heads=8)
# inside T5's attention: scores = Q @ K.transpose(-2, -1) + bias, with no 1/sqrt(d_k)
bias = rpb(seq_len=16)
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

Rotary position embedding {cite}`su2021roformer` is used by LLaMA {cite}`touvron2023llama`, Qwen2.5 {cite}`qwen2024qwen25` and Gemma 2 {cite}`gemma2024gemma2`, and Mistral 7B's released configuration sets its base (`rope_theta`) {cite}`jiang2023mistral`. It has no parameters, and like the relative bias it acts *inside* attention: it rotates the queries and keys, never the values or the input embeddings. Split a query or key row vector of width $d_k$ into $d_k/2$ consecutive pairs of components: pair $i$ is components $2i$ and $2i+1$ of the row (counting from 0), $0 \le i < d_k/2$. Rotate pair $i$ at position $m$ by the angle $m\,\omega_i$, with the frequencies $\omega_i = b^{-2i/d_k}$ of the sinusoidal encoding and base $b = 10000$. As a matrix, the rotation at position $m$ is block diagonal:

```{math}
:label: eq-ch1-rope
\mathcal{R}_m = \begin{pmatrix}
\cos m\omega_0 & -\sin m\omega_0 & & & \\
\sin m\omega_0 & \cos m\omega_0 & & & \\
& & \ddots & & \\
& & & \cos m\omega_{d_k/2-1} & -\sin m\omega_{d_k/2-1} \\
& & & \sin m\omega_{d_k/2-1} & \cos m\omega_{d_k/2-1}
\end{pmatrix},
\qquad \tilde{q}_m = q_m \mathcal{R}_m^\top,
\quad \tilde{k}_{m'} = k_{m'} \mathcal{R}_{m'}^\top
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


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """Rotary position embedding (Su et al., 2021), applied to queries and keys.

    x: [..., seq, d]; positions: [seq]. Pair i of features (features 2i and 2i+1 of
    the last dimension) is rotated by the angle m * omega_i, where m is the position
    and omega_i = base^(-2i/d). Because a rotation preserves norms and q_m . k_m' then
    depends only on m - m', RoPE encodes relative position without any parameters.
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
    q_rot = apply_rope(q[None], torch.tensor([m]))
    k_rot = apply_rope(k[None], torch.tensor([m_prime]))
    return (q_rot @ k_rot.T).item()


print(f"score at (3, 5):   {rotated_score(3, 5):.2f}")
print(f"score at (10, 12): {rotated_score(10, 12):.2f}")
print(f"score at (3, 6):   {rotated_score(3, 6):.2f}")
q_at_7 = apply_rope(q[None], torch.tensor([7]))
print(f"norm before and after rotation: {q.norm():.3f} {q_at_7.norm():.3f}")
```

% output

```text
score at (3, 5):   -10.28
score at (10, 12): -10.28
score at (3, 6):   -10.75
norm before and after rotation: 8.370 8.370
```

{numref}`fig-ch1-rope-rotation` draws the $d_k = 2$ example: moving both positions by 7 turns both vectors and leaves the angle between them alone.

```{figure} figures/rope-rotation.*
:name: fig-ch1-rope-rotation
:width: 80%
:alt: Two unit circles. In each, a query arrow and a key arrow are separated by an arc labeled 2: on the left at angles 3 and 5, on the right at angles 10 and 12.

The worked number above with $d_k = 2$ and $\omega_0 = 1$: the rotated query and key at
$m = 3, m' = 5$ (left) and $m = 10, m' = 12$ (right) point in different directions, but
the angle between them is the same, so the score is the same.
```

(sec-ch1-position-scheme-choice)=

#### Which Scheme Models Use

Most open-weight models since 2022 use RoPE {cite}`su2021roformer`; a few use ALiBi {cite}`press2022train`. ALiBi adds no vector and learns nothing: each head adds to the score of query $m$ and key $m' \le m$ the penalty $-a_h (m - m')$, a fixed slope times the distance, so that attention fades with distance. With 8 heads the slopes are $1/2, 1/4, \dots, 1/2^8$, and the bias is not divided by $\sqrt{d_k}$; models trained on 512 tokens kept their perplexity on longer inputs, where sinusoidal, rotary and T5 positions degraded {cite}`press2022train`. Both encode relative position inside attention without per-position parameters; learned absolute embeddings are now mostly found in BERT- and GPT-2-era models. The parameters each scheme adds are plotted in {ref}`sec-tdd-position-encoding-trade-offs`.

```{admonition} Embeddings and Positional Encoding
:class: note
The notebook ({ref}`nb-attn-positional`) visualizes sinusoidal and learned encodings,
implements a learned relative-position bias and a rotary (RoPE) transform, and asserts
RoPE's defining properties: rotation preserves vector norms, and query-key dot products
depend only on the relative offset.
```

(sec-ch1-historical-context)=

## 3. Historical Context

The pieces of the previous two sections, and the attention of the next, were not designed at once. This section follows them from counting word sequences to the open-weight models of 2024.

{numref}`fig-ch1-history-timeline` places the milestones on one time axis.

```{figure} figures/history-timeline.*
:name: fig-ch1-history-timeline
:width: 86%
:alt: A vertical year axis from 1995 to 2025 with a dot per milestone, each joined by a leader line to a row of text: LSTM in 1997 and the neural language model in 2003, then word2vec, seq2seq and attention in 2013 to 2014, the transformer in 2017, ELMo, GPT and BERT in 2018, GPT-2, T5 and BART in 2019, GPT-3 in 2020, InstructGPT and ChatGPT in 2022, LLaMA and Mistral 7B in 2023 and Llama 3 in 2024.

The milestones this section discusses (sources in the text). Left, each milestone's year
on a time axis to scale; right, one row per milestone. Fifteen of the seventeen fall
after 2012.
```

(sec-ch1-ngram-neural-lm)=

### 3.1 From n-grams to Neural Language Models

For decades a language model was an **n-gram** model: the probability of the next word given a fixed number of previous words (two in a trigram model), estimated by counting how often each such sequence of words occurs in a corpus. Counting cannot generalize to sequences it has never seen, and almost every long sequence is unseen. Bengio et al. put a number on this "curse of dimensionality": to model 10 consecutive words from a vocabulary of 100,000 there are "potentially $100{,}000^{10} - 1 = 10^{50} - 1$ free parameters" {cite}`bengio2003neural`. Their answer was the **neural language model**: give every word a learned vector, a "distributed representation", and let a neural network map the vectors of the context words to the next-word distribution, so that what is learned about one sentence transfers to sentences with similar words. The embedding tables of {ref}`sec-ch1-embedding-spaces` descend from this idea. The difference in size is the whole argument:

```python
vocab, context = 100_000, 10
table = vocab**context - 1
print(f"joint table over {context} words: {table:.3e} free parameters")
d = 100
print(f"one {d}-dimensional vector per word: {vocab * d:,} parameters")
```

% output

```text
joint table over 10 words: 1.000e+50 free parameters
one 100-dimensional vector per word: 10,000,000 parameters
```

**word2vec** made such vectors cheap. Mikolov et al. dropped the hidden layers and trained the vectors alone, so that "it takes less than a day to learn high quality word vectors from a 1.6 billion words data set" {cite}`mikolov2013efficient`. These vectors were static, one per word whatever its context; the models below made them depend on the sentence.

(sec-ch1-evolution-rnns)=

### 3.2 Evolution from RNNs

**Recurrent Neural Networks** initially dominated sequence modeling. They read a sequence one token at a time, computing a new hidden state from the previous state and the current input, and this has two costs:

- **Vanishing and exploding gradients.** The gradient that reaches a token many steps back has been multiplied by one Jacobian per step in between, so long-range components "go exponentially fast to norm 0" or grow exponentially {cite}`pascanu2013difficulty`. If each step scales the gradient by 0.9, the signal from 50 tokens back arrives scaled by $0.9^{50} \approx 0.005$; at 1.1 it is multiplied by $1.1^{50} \approx 117$. Either way the network learns little about dependencies that span many tokens.
- **Sequential computation.** Each state needs the one before it, so processing $n$ tokens takes $n$ sequential steps however many processors are available, and a signal between two tokens $n$ positions apart passes through $n$ steps. A self-attention layer needs a constant number of sequential operations and connects any two positions directly (Table 1 of {cite:t}`vaswani2017attention`).

**Long Short-Term Memory (LSTM)** {cite}`hochreiter1997long` networks addressed vanishing gradients but remained sequential.

**Sequence to sequence.** Sutskever et al. used one deep LSTM "to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector" {cite}`sutskever2014sequence`. On English-to-French translation (WMT'14) it reached a BLEU score of 34.8, against 33.3 for a phrase-based statistical system. This encoder-decoder shape is the one the transformer kept ({ref}`sec-ch1-encoder-decoder`).

**Attention mechanisms** {cite}`bahdanau2014neural` first appeared in machine translation, allowing models to focus on relevant source positions when generating target words, instead of squeezing the whole source into one fixed-size vector.

(sec-ch1-transformer-breakthrough)=

### 3.3 The Transformer Breakthrough

{cite:t}`vaswani2017attention` kept the encoder-decoder shape but built both halves from multi-head attention and feed-forward layers alone, without recurrence or convolution, so all positions of a training sequence are processed in parallel. Their big model reached 28.4 BLEU on WMT 2014 English-to-German, "improving over the existing best results, including ensembles, by over 2 BLEU", and trained in 3.5 days on 8 P100 GPUs, which the authors call a small fraction of the training cost of the best earlier models.

(sec-ch1-scaling-language-models)=

### 3.4 Pretraining and Scaling

The transformer's success in machine translation led to its adoption for language modeling, and to the idea of pretraining one model on unlabeled text and reusing it for many tasks:

- **ELMo** (2018) introduced "deep contextualized word representations": each token's vector is "a function of the entire input sentence", computed by a bidirectional LSTM trained as a language model, so the same word gets different vectors in different sentences {cite}`peters2018deep`.
- **GPT** (2018) pretrained a transformer decoder as a language model on unlabeled text and fine-tuned it for each task {cite}`radford2018improving`.
- **BERT** (2018) pretrained a transformer encoder with the masked-language-model objective of {ref}`sec-ch1-masked-lm`, giving bidirectional representations {cite}`devlin2018bert`.
- **GPT-2** (2019) scaled the decoder to 1.5 billion parameters (1,542M in its paper's table) and showed zero-shot transfer to many tasks from language modeling alone {cite}`radford2019language`.
- **T5** (2019) cast every task as text-to-text with an encoder-decoder {cite}`raffel2019exploring`, and **BART** (2019) pretrained an encoder-decoder as "a denoising autoencoder", reconstructing text from corrupted input {cite}`lewis2020bart`.
- **GPT-3** (2020) achieved few-shot learning with 175B parameters {cite}`brown2020language`.

(sec-ch1-instruction-tuning-open-weights)=

### 3.5 Instruction Tuning and Open Weights

A pretrained language model continues text; it does not follow instructions. **Instruction tuning** closes the gap. InstructGPT fine-tuned GPT-3 on demonstrations written by human labelers, then further with reinforcement learning from human feedback on their rankings of model outputs. Labelers preferred the outputs of the 1.3B-parameter InstructGPT to those of the 175B GPT-3, "despite having 100x fewer parameters" {cite}`ouyang2022training`. On 30 November 2022 OpenAI released **ChatGPT**, described as "a sibling model to InstructGPT" trained for dialogue, and brought instruction-tuned models to a general audience {cite}`openai2022chatgpt`.

The weights of these models were not released. The **open-weight** line that followed published them:

- **LLaMA** (February 2023): models "ranging from 7B to 65B parameters", trained on publicly available data only {cite}`touvron2023llama`.
- **Mistral 7B** (October 2023): a 7-billion-parameter model that outperforms "the best open 13B model (Llama 2) across all evaluated benchmarks" {cite}`jiang2023mistral`.
- **Llama 3** (2024): up to "a dense Transformer with 405B parameters and a context window of up to 128K tokens" {cite}`grattafiori2024llama`.

All three are decoder-only transformers with the changes of {ref}`sec-ch1-modern-block`. From GPT-2's 1.5 billion parameters in 2019 to Llama 3's 405 billion in 2024 is a factor of 270 in five years.

(sec-ch1-attention-mechanisms)=

## 4. Attention Mechanisms

(sec-ch1-attention-revolution)=

### 4.1 The Attention Revolution

Unlike an RNN, which processes a sequence one position after another, attention computes all positions in parallel and connects any two positions directly.

```{admonition} Interactive Exploration
:class: note
The notebook implements scaled dot-product and multi-head attention in eval mode,
asserts that every row of attention weights sums to one, and plots per-head attention
maps and entropies ({ref}`nb-attn-attention`).
```

(sec-ch1-self-attention-fundamentals)=

#### Self-Attention Fundamentals

Self-attention computes a weighted representation of the input sequence, where weights indicate the relevance of each position to the current position. Given input embeddings $X \in \mathbb{R}^{n \times d}$:

1. **Query, Key, Value projections**:

   $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. **Attention output**:

   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The scaling factor $\sqrt{d_k}$, where $d_k$ is the per-head key dimension, keeps the dot products at unit variance so the softmax does not saturate, assuming the components of $q$ and $k$ are independent with mean 0 and variance 1 (their dot product then has variance $d_k$; {ref}`sec-ch2-sqrt-dk` derives this) {cite}`vaswani2017attention`.

{numref}`fig-ch1-scaled-dot-product-attention` follows the shapes through the formula.

```{figure} figures/scaled_dot_product_attention.*
:name: fig-ch1-scaled-dot-product-attention
:width: 82%
:alt: X feeds three projections Q, K and V; Q and K give the n by n scores divided by the square root of d_k, an optional mask adds minus infinity, the softmax gives weights A, and A times V gives the n by d_v output O.

Scaled dot-product attention with the shape of every intermediate. The $n \times n$
score and weight matrices are what makes attention quadratic in $n$
({ref}`sec-ch1-attention-complexity`).
```

(sec-ch1-attention-worked-example)=

#### A Worked Example

Take $n = 3$ tokens with $d = d_k = 2$, embeddings $(1, 0)$, $(0, 1)$ and $(1, 1)$ as the rows of $X$, and let $W_Q$, $W_K$ and $W_V$ be the identity so that $Q = K = V = X$. Token 1's scores are its dot products with every key, divided by $\sqrt{2} \approx 1.4142$: $(1, 0, 1)/\sqrt{2} = (0.7071, 0, 0.7071)$. The softmax turns them into weights: $e^{0.7071} = 2.028$ and $e^{0} = 1$, so the weights are $(2.028, 1, 2.028)/5.056 = (0.4011, 0.1978, 0.4011)$. Token 1's output is the weighted sum of the values, $0.4011\,(1, 0) + 0.1978\,(0, 1) + 0.4011\,(1, 1) = (0.8022, 0.5989)$. It has moved from $(1, 0)$ toward the tokens it resembles; the orthogonal token 2 gets the smallest weight but not zero. The listing computes all three rows:

```python
import torch

X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # n = 3 tokens, d = 2
Q, K, V = X, X, X  # W_Q = W_K = W_V = identity, so the numbers stay small
scores = Q @ K.T / 2 ** 0.5
weights = torch.softmax(scores, dim=-1)
out = weights @ V
torch.set_printoptions(precision=4)
print("scores:\n", scores)
print("weights:\n", weights)
print("row sums:", weights.sum(dim=-1))
print("output:\n", out)
```

% output

```text
scores:
 tensor([[0.7071, 0.0000, 0.7071],
        [0.0000, 0.7071, 0.7071],
        [0.7071, 0.7071, 1.4142]])
weights:
 tensor([[0.4011, 0.1978, 0.4011],
        [0.1978, 0.4011, 0.4011],
        [0.2483, 0.2483, 0.5035]])
row sums: tensor([1., 1., 1.])
output:
 tensor([[0.8022, 0.5989],
        [0.5989, 0.8022],
        [0.7517, 0.7517]])
```

{numref}`fig-ch1-attention-worked-example` shows the same two matrices as heatmaps.

```{figure} figures/attention-worked-example.*
:name: fig-ch1-attention-worked-example
:width: 86%
:alt: Two 3 by 3 heatmaps. Scores: 0.7071 where two tokens share a coordinate, 0 between tokens 1 and 2, 1.4142 for token 3 with itself. Weights: rows 0.4011, 0.1978, 0.4011; 0.1978, 0.4011, 0.4011; 0.2483, 0.2483, 0.5035.

The listing's scores and weights. The softmax keeps the order of each row's scores but
makes them positive and sum to 1; the zero score still gets weight 0.1978.
```

(sec-ch1-attention-complexity)=

#### Cost in Time and Memory

The score matrix $QK^\top$ has one entry for every pair of positions, $n \times n$ of them, and each entry is a dot product of length $d_k$. Over all heads that is $n^2 d$ multiply-adds for the scores and as many again for multiplying the weights by $V$, so self-attention costs $\mathcal{O}(n^2 d)$ time per layer, against $\mathcal{O}(n d^2)$ for the projections and the feed-forward network {cite}`vaswani2017attention`. The weights also have to be stored, $n^2$ numbers per head, so memory grows as $\mathcal{O}(n^2)$. Doubling the context quadruples both. The four projections cost $4nd^2$ and a feed-forward network of width $4d$ costs $8nd^2$, so the $2n^2 d$ of attention overtakes their $12nd^2$ only when $n > 6d$, which is 4,608 tokens for $d = 768$. Below that the $nd^2$ terms dominate; above it the $n^2$ terms do:

```python
for n in (1_024, 8_192, 131_072):
    scores = n * n  # one score per (query, key) pair, per head and layer
    mib = 4 * scores / 2**20
    print(f"n = {n:>7,}: {scores:>14,} scores, {mib:>9,.0f} MiB in fp32 per head")
```

% output

```text
n =   1,024:      1,048,576 scores,         4 MiB in fp32 per head
n =   8,192:     67,108,864 scores,       256 MiB in fp32 per head
n = 131,072: 17,179,869,184 scores,    65,536 MiB in fp32 per head
```

FlashAttention computes exactly the same output without ever storing the full $n \times n$ matrix, which removes the $\mathcal{O}(n^2)$ memory but not the $\mathcal{O}(n^2 d)$ time {cite}`dao2022flashattention`.

(sec-ch1-multi-head-attention)=

#### Multi-Head Attention

Multi-head attention runs multiple attention mechanisms in parallel, allowing the model to attend to different types of relationships:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_{n_h})W_O, \qquad W_O \in \mathbb{R}^{n_h d_v \times d}$$

where each head projects the input $X$ with its own matrices:

$$\text{head}_i = \text{Attention}(XW_Q^{(i)}, XW_K^{(i)}, XW_V^{(i)}), \qquad W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d \times d_k}, \quad W_V^{(i)} \in \mathbb{R}^{d \times d_v}, \quad 1 \le i \le n_h$$

with $d_k = d_v = d/n_h$, so the heads together cost as much as one head of width $d$. The single projection $W_Q$ of the previous equations is the $n_h$ matrices $W_Q^{(i)}$ placed side by side, $W_Q = [W_Q^{(1)} \cdots W_Q^{(n_h)}] \in \mathbb{R}^{d \times d}$, and likewise for $W_K$ and $W_V$; that is how the code below computes all heads with one matrix product and then splits the columns into heads. For cross-attention the queries come from one sequence and the keys and values from another, so the key length may differ from the query length; the listing below shows both cases.

{numref}`fig-ch1-multi-head-attention` shows the split and the concatenation.

```{figure} figures/multi_head_attention.*
:name: fig-ch1-multi-head-attention
:width: 84%
:alt: X is projected to Q, K and V of width d, the columns are split into n_h heads of width d_k, each head runs attention, the heads are concatenated back to width d and multiplied by W_O.

Multi-head attention: one projection of width $d$, split into $n_h$ heads of
$d_k = d/n_h$ columns, attention in each head, concatenation and the output projection
$W_O$.
```

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [..., n_q, d_k], e.g. [batch, heads, n_q, d_k]
            key: [..., n_k, d_k]
            value: [..., n_k, d_v]
            mask: True where attention is allowed, broadcastable to the scores
                (e.g. [batch, heads, n_q, n_k]), or None

        Returns:
            output: [..., n_q, d_v]
            attention_weights: [..., n_q, n_k]
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

    w_q holds the per-head matrices W_Q^(i) side by side as its columns (likewise w_k
    and w_v), so splitting X W_Q into n_h blocks of d_k columns gives X W_Q^(i) for
    every head at once.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V (all heads at once)
        # and the output projection W_O
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """query: [batch, n_q, d_model]; key and value: [batch, n_k, d_model].

        mask: True where attention is allowed, broadcastable to
        [batch, heads, n_q, n_k]; a causal mask is [n_q, n_k], a padding mask
        [batch, 1, 1, n_k].
        Returns the output [batch, n_q, d_model] and the weights
        [batch, heads, n_q, n_k].
        """
        batch_size, n_q, n_k = query.size(0), query.size(1), key.size(1)

        # 1. Project, then split the d_model columns into heads:
        #    [batch, heads, length, d_k]
        h, d_k = self.num_heads, self.d_k
        Q = self.w_q(query).view(batch_size, n_q, h, d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, n_k, h, d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, n_k, h, d_k).transpose(1, 2)

        # 2. Attention in every head at once
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 3. Concatenate the heads and apply W_O
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, n_q, self.d_model)
        return self.w_o(attn_output), attn_weights


torch.manual_seed(0)
mha = MultiHeadAttention(d_model=64, num_heads=8).eval()
x, memory = torch.randn(2, 5, 64), torch.randn(2, 7, 64)
out, weights = mha(x, x, x)  # self-attention
print("self-attention: ", tuple(out.shape), tuple(weights.shape))
# queries from x, keys and values from a longer sequence
out, weights = mha(x, memory, memory)
print("cross-attention:", tuple(out.shape), tuple(weights.shape))
```

% output

```text
self-attention:  (2, 5, 64) (2, 8, 5, 5)
cross-attention: (2, 5, 64) (2, 8, 5, 7)
```

(sec-ch1-attention-patterns-interpretability)=

### 4.2 Attention Patterns and Interpretability

An attention map can be read directly: for each token, which token does a head attend to most? {cite:t}`clark2019what` did this for all 144 heads of BERT-base (12 layers of 12 heads) and scored the most-attended word as a guess of the token's syntactic head in the Wall Street Journal dependency treebank. No head parses well overall (the best gets 34.5% of all relations right, against 26.3% for always choosing the next word), but single relations have specialists:

| Relation | Head (layer-head) | Accuracy | Best fixed-offset baseline |
| --- | --- | --- | --- |
| Determiner to its noun | 8-11 | 94.3% | 51.7% (next word) |
| Direct object to its verb | 8-10 | 86.8% | 40.0% (two words back) |
| Coreferent mention to its antecedent | 5-4 | 65% | 52% (same head word) |

The coreference head comes close to a rule-based system (69%). Other heads attend almost only to the previous or the next token, and over half of the attention in layers 6 to 10 goes to the separator token [SEP], which the authors argue serves as a no-op. Such maps say where a head looks, not why the model predicts what it does: {cite:t}`jain2019attention` find that attention weights "are frequently uncorrelated with gradient-based measures of feature importance", and that very different attention distributions can give the same prediction.

(sec-ch1-transformer-architecture)=

## 5. Transformer Architecture

(sec-ch1-building-blocks)=

### 5.1 Building Blocks

The transformer {cite}`vaswani2017attention` combines attention with residual connections, normalization and a position-wise feed-forward network; this subsection builds each and assembles them into a block.

(sec-ch1-layernorm)=

#### Layer Normalization

Layer normalization {cite}`ba2016layer` stabilizes training by normalizing inputs across the feature dimension:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed across the feature dimension (for numerical stability $\sigma = \sqrt{\operatorname{Var} + \epsilon}$, with a small $\epsilon$ added to the variance under the square root), and $\gamma$, $\beta$ are learned per-feature scale and shift.

(sec-ch1-residual-connections)=

#### Residual Connections

Residual connections {cite}`he2016deep` help gradient flow in deep networks:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

(sec-ch1-feed-forward-networks)=

#### Feed-Forward Networks

Each transformer layer includes a position-wise feed-forward network:

$$\text{FFN}(x) = \max(0, xW_{\text{up}})W_{\text{down}}$$

with $W_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$. Bias terms are left out of the formulas in this chapter. The network provides non-linear transformations, and the original transformer sets the hidden width $d_{ff} = 4d$.

The original activation is ReLU, $\max(0, \cdot)$. BERT and the GPT models use the **Gaussian error linear unit** instead {cite}`hendrycks2016gaussian`, applied elementwise: $\mathrm{GELU}(x) = \tfrac{x}{2}\bigl(1 + \operatorname{erf}(x/\sqrt{2})\bigr)$, which is $x$ times the probability that a standard normal variable lies below $x$. Instead of cutting at 0 it weights each input by that probability, so $\mathrm{GELU}(1) = 0.841$, $\mathrm{GELU}(-1) = -0.159$ and the curve is smooth where ReLU has a kink. BERT's authors "use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT" {cite}`devlin2018bert`, and GPT-2's released code (`src/model.py` of the `openai/gpt-2` repository) applies the tanh approximation of GELU in its feed-forward network. The gated variant used by later models is in {ref}`sec-ch1-modern-block`.

(sec-ch1-assembling-block)=

#### Assembling a Block

The original block applies multi-head attention and then the feed-forward network, each followed by a residual connection and layer normalization (post-LN, the formula above).

The `dropout` arguments in this and the attention listing are **dropout**: during training each component of its input is set to 0 with probability $p_{drop}$ and the others are multiplied by $1/(1 - p_{drop})$, so the expected value is unchanged; in evaluation mode (`model.eval()`) it passes its input through unchanged. The original transformer applies it "to the output of each sub-layer, before it is added to the sub-layer input and normalized" and to the sums of the token embeddings and positional encodings, and its base model uses $p_{drop} = 0.1$ {cite}`vaswani2017attention`. The listings use the same rate and also apply it to the attention weights:

```python
# pseudo-code: continues the attention listing of Section 4.1, which defines
# MultiHeadAttention; run after it in the same session
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

Run after the attention listing with $d = 64$, $n_h = 8$ and $d_{ff} = 256 = 4d$, `TransformerBlock` maps a $2 \times 5 \times 64$ input to a $2 \times 5 \times 64$ output and has 49,984 parameters: 16,640 in attention, 33,088 in the feed-forward network and 256 in the two LayerNorms, which is $12d^2 + 13d$ of {ref}`sec-ch1-parameter-count`. The test `test_transformer_block_shape_and_parameter_count` asserts these numbers.

```{admonition} Transformer Architecture Implementation
:class: tip
The notebook ({ref}`nb-attn-transformer`) assembles one post-LN transformer block from
multi-head attention, a position-wise feed-forward network, layer normalization and
residual connections, and shows what layer normalization does to activations of
different scales. The notebook's language-model section ({ref}`nb-attn-lm`) then trains
a small character-level model and checks that its loss falls.
```

(sec-ch1-end-to-end)=

#### The Model End to End

A decoder-only language model is these pieces in a fixed order. For a batch of $B$ sequences of $n$ token ids each:

1. **Token ids**, shape $B \times n$, integers below $|\mathcal{V}|$.
2. **Embedding**: each id selects its row of $W_E$, and the positional encoding of {ref}`sec-ch1-positional-encodings` is added (or, with RoPE, applied later inside attention). Shape $B \times n \times d$.
3. **$L$ blocks**, each attention followed by the feed-forward network, each with its residual connection and normalization. The causal mask of {ref}`sec-ch1-causal-masking` keeps position $i$ from seeing later positions. Every block maps $B \times n \times d$ to $B \times n \times d$, so blocks can be stacked to any depth.
4. **Final normalization** of the last block's output (pre-LN models need it, because their residual stream is never normalized inside a block). Shape still $B \times n \times d$.
5. **Unembedding**: multiplying by $W_{\text{out}} \in \mathbb{R}^{d \times |\mathcal{V}|}$ gives one score per vocabulary entry, shape $B \times n \times |\mathcal{V}|$. The softmax of row $i$ is the model's distribution over the token at position $i + 1$ ({ref}`sec-ch1-training-process`).

The smallest GPT-2, for example, has $|\mathcal{V}| = 50{,}257$, $d = 768$, $L = 12$ blocks and a context of 1,024 tokens {cite}`radford2019language`, so a batch of 8 sequences of 1,024 tokens goes from an $8 \times 1024$ array of ids to $8 \times 1024 \times 768$ hidden states and finally to $8 \times 1024 \times 50{,}257 = 411{,}705{,}344$ scores. The listing runs the same path at toy size. Its blocks are PyTorch's `nn.TransformerEncoderLayer` with `norm_first=False`, which computes the same post-LN block as `TransformerBlock` above (attention, then a ReLU feed-forward network of width $4d$, each followed by a residual connection and LayerNorm). It stands in for `TransformerBlock` only because every listing on this page runs on its own, and repeating the attention and block classes here would add over a hundred lines:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
vocab_size, d_model, num_heads, num_layers, max_len = 1000, 64, 4, 2, 16

embed = nn.Embedding(vocab_size, d_model)  # W_E
positions = nn.Embedding(max_len, d_model)  # learned absolute positions
blocks = nn.ModuleList(
    nn.TransformerEncoderLayer(
        d_model, num_heads, 4 * d_model,
        dropout=0.0, batch_first=True, norm_first=False,
    )
    for _ in range(num_layers)
)
# needed by pre-LN models; kept so the path matches step 4
final_norm = nn.LayerNorm(d_model)
unembed = nn.Linear(d_model, vocab_size, bias=False)  # W_out

ids = torch.randint(0, vocab_size, (2, 10))  # B = 2 sequences of n = 10 tokens
n = ids.size(1)
causal = nn.Transformer.generate_square_subsequent_mask(n)
x = embed(ids) + positions(torch.arange(n))
print("token ids:   ", tuple(ids.shape))
print("embedded:    ", tuple(x.shape))
for block in blocks:
    x = block(x, src_mask=causal, is_causal=True)
print("after blocks:", tuple(x.shape))
logits = unembed(final_norm(x))
print("logits:      ", tuple(logits.shape))
modules = [embed, positions, blocks, final_norm, unembed]
print("parameters:  ", f"{sum(p.numel() for m in modules for p in m.parameters()):,}")
```

% output

```text
token ids:    (2, 10)
embedded:     (2, 10, 64)
after blocks: (2, 10, 64)
logits:       (2, 10, 1000)
parameters:   229,120
```

(sec-ch1-parameter-count)=

#### Counting Parameters

Every block of the model above has the same parameters, so the count is a short formula. With biases, LayerNorm and $d_{ff} = 4d$, one block holds:

- attention: four $d \times d$ projections and their biases, $4d^2 + 4d$;
- feed-forward network: $d \times 4d$ and $4d \times d$ matrices with biases, $8d^2 + 5d$;
- two LayerNorms with $\gamma$ and $\beta$, $4d$.

That is $12d^2 + 13d$ per block. The whole model adds the token and position tables, the final LayerNorm and, unless it is tied to $W_E$ ({ref}`sec-ch1-embedding-sharing-tying`), the unembedding:

$$|\mathcal{V}|\,d + n_{ctx}\,d + L\,(12d^2 + 13d) + 2d + d\,|\mathcal{V}|$$

For the listing's toy model this is $64{,}000 + 1{,}024 + 2 \times 49{,}984 + 128 + 64{,}000 = 229{,}120$, exactly what it printed. For large models the $12Ld^2$ term is almost everything, so $12Ld^2$ alone is a good estimate:

```python
def count_parameters(vocab, d, layers, context, tied):
    # attention 4d^2 + 4d, FFN (width 4d) 8d^2 + 5d, two LayerNorms 4d
    per_block = 12 * d * d + 13 * d
    embeddings = vocab * d + context * d  # W_E and learned positions
    unembedding = 0 if tied else d * vocab
    return embeddings + layers * per_block + 2 * d + unembedding  # 2d: final LayerNorm

toy = count_parameters(1000, 64, 2, 16, tied=False)
print(f"toy model of the listing above: {toy:,}")
gpt3 = count_parameters(50257, 12288, 96, 2048, tied=True)
print(f"GPT-3 (96 layers, d = 12288):   {gpt3:,} = {gpt3 / 1e9:.1f}B")
print(f"of which 12 L d^2:              {12 * 96 * 12288**2 / gpt3:.1%}")
```

% output

```text
toy model of the listing above: 229,120
GPT-3 (96 layers, d = 12288):   174,604,259,328 = 174.6B
of which 12 L d^2:              99.6%
```

GPT-3's 96 layers of width 12,288 come from its paper, which lists the model as 175.0B parameters {cite}`brown2020language`; the formula lands within 0.3% of that. Modern blocks change the constants (a SwiGLU network of width about $\tfrac{8}{3}d$ has three matrices, $8d^2$, the same as the ReLU network of width $4d$; grouped-query attention shrinks $W_K$ and $W_V$), and chapter 6 counts them for real models.

(sec-ch1-encoder-decoder)=

#### The Encoder-Decoder Stack and Cross-Attention

The original transformer is two stacks {cite}`vaswani2017attention`. The **encoder** reads the whole source sentence of $n_{src}$ tokens with unmasked self-attention and outputs one vector per source token, $H_{enc} \in \mathbb{R}^{n_{src} \times d}$. The **decoder** generates the target one token at a time. Each decoder block has three sublayers instead of two: masked self-attention over the target so far, then **cross-attention**, then the feed-forward network. In cross-attention, in the paper's words, "the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder":

$$\text{CrossAttention}(Y, H_{enc}) = \text{softmax}\left(\frac{(YW_Q)(H_{enc}W_K)^\top}{\sqrt{d_k}}\right) H_{enc}W_V$$

where $Y \in \mathbb{R}^{n \times d}$ holds the decoder's states. The score matrix is $n \times n_{src}$, so every target position can look at every source position, and no mask is needed because the source is fully known. $H_{enc}$ is computed once per source sentence and reused at every decoding step and in every decoder block:

```python
import torch

torch.manual_seed(0)
n, n_src, d = 3, 5, 4  # 3 target positions, 5 source positions
H_enc = torch.randn(n_src, d)  # encoder output, computed once per source sentence
y = torch.randn(n, d)  # decoder states after its own masked self-attention
W_Q, W_K, W_V = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
# queries from the decoder, keys and values from the encoder
Q, K, V = y @ W_Q, H_enc @ W_K, H_enc @ W_V
weights = torch.softmax(Q @ K.T / d ** 0.5, dim=-1)
out = weights @ V
row_sums = [round(s, 4) for s in weights.sum(-1).tolist()]
print("weights:", tuple(weights.shape), "row sums:", row_sums)
print("output: ", tuple(out.shape))
```

% output

```text
weights: (3, 5) row sums: [1.0, 1.0, 1.0]
output:  (3, 4)
```

The decoder's output then goes through the same final normalization and unembedding as in {ref}`sec-ch1-end-to-end`. The `MultiHeadAttention` listing in {ref}`sec-ch1-multi-head-attention` runs the multi-head version of this case when it is called with a second sequence as keys and values.

{numref}`fig-ch1-encoder-decoder` puts the two stacks side by side.

```{figure} figures/encoder_decoder.*
:name: fig-ch1-encoder-decoder
:width: 96%
:alt: Left, the encoder: source tokens through self-attention and a feed-forward network, repeated block after block, giving the encoder output H_enc. Right, the decoder: target tokens through masked self-attention, cross-attention that takes keys and values from H_enc, and a feed-forward network, repeated block after block, then the next-token distribution.

The encoder-decoder transformer of {cite:t}`vaswani2017attention`. $H_{enc}$ is computed
once and feeds the cross-attention of every decoder block.
```

(sec-ch1-architectural-variants)=

### 5.2 Architectural Variants

While the original transformer was designed for machine translation, different variants emerged for language modeling:

- **GPT** {cite}`radford2018improving`: Decoder-only architecture with causal masking
- **BERT** {cite}`devlin2018bert`: Encoder-only architecture with bidirectional attention
- **T5** {cite}`raffel2019exploring`: Encoder-decoder architecture treating all tasks as text-to-text

The three differ in which positions may attend to which, as {numref}`fig-ch1-mask-patterns` shows.

```{figure} figures/mask-patterns.*
:name: fig-ch1-mask-patterns
:width: 96%
:alt: Four grids of allowed query-key pairs: encoder-only is fully dark; decoder-only is dark on and below the diagonal; the encoder-decoder's masked self-attention grid, three target queries by three target keys, is dark on and below the diagonal; its cross-attention grid, three target queries by four source keys, is fully dark.

Allowed (dark) and masked (light) query-key pairs for four source tokens s1 to s4 and
three target tokens t1 to t3. The decoder of an encoder-decoder has two attention
sublayers, each with its own softmax: masked self-attention over the target keys, then
cross-attention over the source keys, so it gets two grids, in that order.
```

(sec-ch1-masked-lm)=

#### The Masked-Language-Model Objective

An encoder sees the whole sequence, so it cannot be trained to predict the next token: the answer would be visible. BERT instead hides some tokens and predicts them from both sides {cite}`devlin2018bert`. It chooses 15% of the token positions at random as targets, the set $\mathcal{M}$, and replaces each chosen token with `[MASK]` 80% of the time, with a random token 10% of the time, and leaves it unchanged 10% of the time, so the model cannot rely on `[MASK]` marking what to predict and the inputs at fine-tuning time, which have no `[MASK]`, look like training inputs. The loss is the cross-entropy at the chosen positions only:

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid \tilde{x})$$

where $\tilde{x}$ is the corrupted sequence. The listing applies the procedure to 10,000 tokens:

```python
import torch

torch.manual_seed(0)
n, vocab, mask_id = 10_000, 30_000, 103  # 103 is [MASK] in BERT's vocabulary
ids = torch.randint(1_000, vocab, (n,))
# 15% of positions become prediction targets
chosen = torch.randperm(n)[: int(0.15 * n)]
k = len(chosen)
# 80% / 10%; the last 10% stay
to_mask, to_random = chosen[: int(0.8 * k)], chosen[int(0.8 * k): int(0.9 * k)]
corrupted = ids.clone()
corrupted[to_mask] = mask_id
corrupted[to_random] = torch.randint(1_000, vocab, (len(to_random),))
print(f"targets (loss terms): {k:,} of {n:,}")
print(f"[MASK] inputs:        {int((corrupted == mask_id).sum()):,}")
print(f"left unchanged:       {int((corrupted[chosen] == ids[chosen]).sum()):,}")
print(f"causal LM targets:    {n - 1:,}")
```

% output

```text
targets (loss terms): 1,500 of 10,000
[MASK] inputs:        1,200
left unchanged:       150
causal LM targets:    9,999
```

The last line is the contrast that matters. A causal language model gets a training signal from every position, 9,999 terms for 10,000 tokens, while the masked model gets 1,500; BERT's authors note that because the masked model "only make[s] predictions on 15% of tokens in each batch", "more pre-training steps may be required" {cite}`devlin2018bert`.

(sec-ch1-why-decoder-only)=

#### Why Decoder-Only Models Won

Encoders trained this way are strong at understanding tasks such as classification, but they do not generate text, and an encoder-decoder needs two stacks and a split of the input into source and target. A decoder-only model needs neither. Its objective uses every token, as the count above shows; it needs no task-specific output layer, because any task can be posed as text to continue; and the same next-token prediction that trains it is how it generates. GPT-3 showed that such a model, made large enough, performs new tasks from a few examples placed in its prompt {cite}`brown2020language`, and the widely used LLMs since, from LLaMA to Llama 3, are decoder-only {cite}`touvron2023llama,grattafiori2024llama`. Encoder-only and encoder-decoder models remain in use where their strengths matter, for example for retrieval embeddings and for translation.

(sec-ch1-modern-block)=

### 5.3 What Changed Since 2017

The block above is the 2017 design. Later models change it in some of the following ways, aimed at training stability or inference cost; not every model adopts every change (GPT-2 and GPT-3, for example, adopt only the first):

- **Pre-LN instead of post-LN.** Normalizing each sublayer's input, $x \leftarrow x + \mathrm{MHA}(\mathrm{Norm}(x))$, instead of the sum makes training less sensitive to the learning rate: post-LN can fail at learning rates where pre-LN trains ({ref}`sec-ch2-pre-post-ln`).
- **RMSNorm instead of LayerNorm.** Dropping the mean subtraction and the bias, $\mathrm{RMSNorm}(x) = \gamma \odot x / \sqrt{\lVert x \rVert_2^2 / d + \epsilon}$, is cheaper and trains just as well {cite}`zhang2019root`. LLaMA {cite}`touvron2023llama`, Qwen2.5 {cite}`qwen2024qwen25` and Gemma 2 {cite}`gemma2024gemma2` use it, and Mistral 7B's released configuration sets its $\epsilon$ (`rms_norm_eps`) {cite}`jiang2023mistral`.
- **Gated activations.** The feed-forward network becomes $\mathrm{FFN}(x) = (\mathrm{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}})W_{\text{down}}$ (SwiGLU), with the hidden width reduced to about $\tfrac{8}{3}d$ to keep the parameter count unchanged {cite}`shazeer2020glu`.
- **Rotary position embeddings.** Positions enter through RoPE on the queries and keys ({ref}`sec-ch1-rope`), not through vectors added to the input {cite}`su2021roformer`.
- **Grouped-query attention.** Several query heads share one key/value head, shrinking the KV cache that dominates inference memory at long context {cite}`ainslie2023gqa`.
- **No biases, tied or untied embeddings, larger vocabularies.** Linear layers usually drop their bias terms; vocabularies grew from 32k to 128k-256k (see {ref}`sec-tdd-choosing-vocabulary-size`).

{numref}`fig-ch1-transformer-block-norms` shows the first change for the attention sublayer; the feed-forward sublayer repeats the pattern.

```{figure} figures/transformer_block_norms.*
:name: fig-ch1-transformer-block-norms
:width: 89%
:alt: Left, post-LN: x goes through multi-head attention, is added to the residual x, then normalized. Right, pre-LN: x is normalized, goes through attention and is added to the unnormalized x.

Post-LN normalizes the sum, so the residual stream passes through a normalization in
every block; pre-LN normalizes only the sublayer's input, so the residual path from
input to output is the identity.
```

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


class SelfAttention(nn.Module):
    """Causal multi-head attention.

    Its forward(q, k, v) is the interface PreLNBlock expects.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, bias=False, batch_first=True)

    def forward(self, q, k, v):
        n = q.size(1)
        future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        return self.mha(q, k, v, attn_mask=future, need_weights=False)[0]


torch.manual_seed(0)
d_model = 64
block = PreLNBlock(d_model, SelfAttention(d_model, n_heads=4))
count = lambda m: sum(p.numel() for p in m.parameters())
hidden = block.ffn.w1.out_features
print(f"SwiGLU hidden width: {hidden} (8/3 x {d_model}, rounded down)")
two_matrix = 2 * d_model * 4 * d_model
print(f"SwiGLU parameters: {count(block.ffn):,}; a 4x two-matrix FFN: {two_matrix:,}")
norms = count(block.norm1) + count(block.norm2)
print(f"attention {count(block.attention):,} + norms {norms} "
      f"+ SwiGLU {count(block.ffn):,} = block {count(block):,}")
x = torch.randn(2, 5, d_model)
print("output shape:", tuple(block(x).shape))
```

% output

```text
SwiGLU hidden width: 170 (8/3 x 64, rounded down)
SwiGLU parameters: 32,640; a 4x two-matrix FFN: 32,768
attention 16,384 + norms 128 + SwiGLU 32,640 = block 49,152
output shape: (2, 5, 64)
```

SwiGLU has three weight matrices where the original feed-forward layer has two, so its hidden width is cut from $4d$ to $\tfrac{8}{3} d$: at model width $d = 64$ that is 170 units and 32,640 parameters, within 0.4% of the 32,768 of the two-matrix layer. The block keeps the shape of its input, so blocks stack.

(sec-ch1-language-modeling-objective)=

## 6. Language Modeling Objective

(sec-ch1-next-token-prediction)=

### 6.1 Next-Token Prediction

Modern LLMs are trained on the **causal language modeling** objective, predicting the next token given previous tokens:

$$p_\theta(x) = \prod_{i=1}^{n} p_\theta(x_i \mid x_1, x_2, \dots, x_{i-1}) = \prod_{i=1}^{n} p_\theta(x_i \mid x_{<i})$$

where $x_{<i}$ abbreviates $x_1, \dots, x_{i-1}$ and the model supplies each factor. The product is the chain rule of probability, so the factorization is exact (a model may still fit its factors badly), and it gives a generation procedure: draw $x_i$ from $p_\theta(\cdot \mid x_{<i})$, append it and repeat ({ref}`sec-ch1-decoding`).

(sec-ch1-causal-masking)=

#### Causal Masking

To prevent the model from "seeing the future," causal masking ensures attention only flows from earlier to later positions:

```python
import torch

def create_causal_mask(seq_length: int) -> torch.Tensor:
    """Create a causal mask to prevent attention to future tokens."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask == 0  # True for allowed positions, False for masked


mask = create_causal_mask(4)
print(mask.int())  # row = query position, column = key position
print("keys allowed per query:", mask.sum(dim=1).tolist())
```

% output

```text
tensor([[1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]], dtype=torch.int32)
keys allowed per query: [1, 2, 3, 4]
```

For the four inputs of the next listing the mask is {numref}`fig-ch1-causal-mask`. Mathematically a masked score is $-\infty$, so that its weight $e^{-\infty}$ is exactly 0. The attention listings fill masked scores with `torch.finfo(scores.dtype).min`, the most negative finite number, instead: the weight still underflows to exactly 0, and a row whose keys are all masked (a padding query) gets uniform weights instead of the NaN that $-\infty - (-\infty)$ gives inside the softmax ({ref}`sec-ch1-exercises` works both cases).

```{figure} figures/causal-mask.*
:name: fig-ch1-causal-mask
:width: 60%
:alt: A 4 by 4 grid over the tokens bos, the, cat, sat: zeros on and below the diagonal, minus infinity above it.

The causal mask as it is added to the scores: 0 where a query may attend, $-\infty$
where the key is later, so the softmax gives those keys weight 0.
`create_causal_mask(4)` is `True` exactly where this grid shows 0.
```

(sec-ch1-teacher-forcing)=

#### Teacher Forcing, Shifted Targets, BOS and EOS

During training the model never feeds its own predictions back in. The whole true sequence goes in at once, and position $i$ is asked to predict token $i + 1$; this is called **teacher forcing**. In code the targets are the inputs shifted left by one, and the causal mask makes the $n$ predictions independent of each other's answers, so one forward pass yields $n$ loss terms. Two special tokens frame each sequence: a **beginning-of-sequence** token (BOS) occupies position 0, so that the first real token, at position 1, is predicted from something, and an **end-of-sequence** token (EOS) is a target like any other, so the model learns when to stop. GPT-2 uses one token, `<|endoftext|>` (id 50,256), for both roles, placed between documents:

```python
tokens = ["<bos>", "the", "cat", "sat", "<eos>"]
# the targets are the inputs shifted left by one
inputs, targets = tokens[:-1], tokens[1:]
for i, target in enumerate(targets):  # BOS is position 0
    print(f"position {i}: sees {inputs[:i + 1]} -> predicts {target!r}")
```

% output

```text
position 0: sees ['<bos>'] -> predicts 'the'
position 1: sees ['<bos>', 'the'] -> predicts 'cat'
position 2: sees ['<bos>', 'the', 'cat'] -> predicts 'sat'
position 3: sees ['<bos>', 'the', 'cat', 'sat'] -> predicts '<eos>'
```

Real batches hold sequences of different lengths, so the shorter ones are filled up to the longest with a padding token (`[PAD]` or `<pad>`, id `pad_id` below) that must change nothing. Two things enforce that. The attention mask allows a key only if the causal mask allows it *and* the key is a real token, so no query gives weight to padding. The loss skips every target equal to `pad_id` (`ignore_index`), so padded positions add no loss terms and the mean runs over real targets only. The mask below has the shape `[B, 1, n, n]` that `MultiHeadAttention` accepts:

```python
import torch
import torch.nn.functional as F

pad_id = 0
# two sequences of different length, padded on the right to n = 5 tokens
ids = torch.tensor([[1, 2, 3, 4, 5],
                    [1, 2, 3, pad_id, pad_id]])
keep = ids != pad_id  # [B, n]: True for real tokens
n = ids.size(1)
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
# allowed = causal AND the key is a real token; broadcastable to [B, heads, n_q, n_k]
mask = causal[None, None] & keep[:, None, None, :]
print("real keys per query, sequence 2:", mask[1, 0].sum(dim=1).tolist())

torch.manual_seed(0)
scores = torch.randn(2, 1, n, n)
weights = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1)
print("weight on pad keys, sequence 2:", weights[1, 0, :, 3:].sum().item())

vocab = 10
logits = torch.randn(2, n, vocab)
targets = ids[:, 1:]  # shifted by one; padded targets are skipped
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab), targets.reshape(-1), ignore_index=pad_id
)
terms = (targets != pad_id).sum().item()
print(f"loss terms: {terms} of {targets.numel()}; mean loss {loss.item():.4f}")
per_token = F.cross_entropy(
    logits[:, :-1].reshape(-1, vocab), targets.reshape(-1), reduction="none"
)
assert torch.allclose(loss, per_token[targets.reshape(-1) != pad_id].mean())
```

% output

```text
real keys per query, sequence 2: [1, 2, 3, 3, 3]
weight on pad keys, sequence 2: 0.0
loss terms: 6 of 8; mean loss 3.6028
```

The second sequence has three real tokens, so each of its queries sees at most three keys and the two padded keys get weight exactly 0. Of the eight shifted targets, two are padding, leaving six loss terms; the assertion shows that the loss is the plain mean over those six. With right padding the padded positions are the last ones, so the causal mask alone already hides them from every real query, but the combined mask is what keeps left-padded batches correct too.

(sec-ch1-decoding)=

#### Decoding: Greedy, Temperature, Top-k and Top-p

At generation time there is no true sequence. The model computes the distribution for the next token, one token is chosen and appended, and the loop repeats until EOS or a length limit. How the token is chosen is the **decoding** strategy:

- **Greedy** decoding takes the most probable token every time. It is deterministic and tends to repeat itself in long text.
- **Temperature** $\tau$ divides the logits before the softmax, $\text{softmax}(z/\tau)$. $\tau < 1$ sharpens the distribution toward greedy, $\tau > 1$ flattens it toward uniform, and $\tau = 1$ samples from the model's own distribution.
- **Top-k** sampling keeps only the $k_{top}$ most probable tokens, renormalizes and samples; {cite:t}`fan2018hierarchical` sampled from the $k_{top} = 10$ most likely candidates.
- **Top-p** (nucleus) sampling keeps the smallest set of most probable tokens whose total probability reaches $p_{top}$, so the number kept adapts to how confident the model is {cite}`holtzman2020curious`.
- **Beam search** searches for a likely *sequence* rather than choosing one token at a time; it is described after the listing.

The listing applies each rule to the same five logits:

```python
import torch

vocab = ["the", "a", "cat", "dog", "pizza"]
logits = torch.tensor([2.0, 1.5, 1.0, 0.5, -1.0])


def show(name, probs):
    shown = " ".join(f"{w}={p:.3f}" for w, p in zip(vocab, probs.tolist()))
    print(f"{name:<15}", shown)


print("greedy:        ", vocab[int(logits.argmax())])
for tau in (1.0, 0.5, 2.0):
    show(f"tau = {tau}", torch.softmax(logits / tau, dim=-1))

probs = torch.softmax(logits, dim=-1)
k_top = 2
kth_largest = logits.topk(k_top).values[-1]
top_k = torch.where(logits >= kth_largest, probs, torch.zeros_like(probs))
show("top-k, k = 2", top_k / top_k.sum())

p_top = 0.9
sorted_p, order = probs.sort(descending=True)
# keep tokens until the kept mass first reaches p_top
keep = sorted_p.cumsum(0) - sorted_p < p_top
nucleus = torch.zeros_like(probs)
nucleus[order[keep]] = probs[order[keep]]
show("top-p, p = 0.9", nucleus / nucleus.sum())
```

% output

```text
greedy:         the
tau = 1.0       the=0.445 a=0.270 cat=0.164 dog=0.099 pizza=0.022
tau = 0.5       the=0.643 a=0.237 cat=0.087 dog=0.032 pizza=0.002
tau = 2.0       the=0.325 a=0.253 cat=0.197 dog=0.153 pizza=0.072
top-k, k = 2    the=0.622 a=0.378 cat=0.000 dog=0.000 pizza=0.000
top-p, p = 0.9  the=0.455 a=0.276 cat=0.167 dog=0.102 pizza=0.000
```

Halving the temperature raises the top token from 0.445 to 0.643. Top-k with $k_{top} = 2$ keeps "the" and "a" and rescales them by $1/(0.445 + 0.270)$. Top-p with $p_{top} = 0.9$ needs four tokens, because the first three hold only $0.445 + 0.270 + 0.164 = 0.879$; on a more confident distribution the same $p_{top}$ would keep fewer. In practice temperature is combined with top-k or top-p, and the chosen token is then drawn from the result.

**Beam search** keeps the best few prefixes instead of one. With beam width $b_{beam}$ it extends each of the $b_{beam}$ kept prefixes by every token, scores each extension by its summed log-probability, keeps the best $b_{beam}$ and repeats; greedy decoding is beam width 1. A sum of log-probabilities falls with every token added, so finished hypotheses of different lengths are compared with a **length penalty** that divides the sum by a term growing with the length. The original transformer translated with "beam search with a beam size of 4 and length penalty α = 0.6", using the penalty of {cite:t}`wu2016google` {cite}`vaswani2017attention`. The listing runs two steps of a toy model in which the most probable first token leads to a weak continuation:

```python
import math

first = {"the": 0.5, "a": 0.4, "cat": 0.1}
second = {
    "the": {"cat": 0.3, "dog": 0.3, "pizza": 0.4},
    "a": {"cat": 0.9, "dog": 0.05, "pizza": 0.05},
    "cat": {"cat": 0.2, "dog": 0.4, "pizza": 0.4},
}


def beam_search(width):
    beams = sorted(((math.log(p), [w]) for w, p in first.items()), reverse=True)[:width]
    candidates = [
        (s + math.log(p), seq + [w])
        for s, seq in beams
        for w, p in second[seq[-1]].items()
    ]
    # best two-token sequence among the extensions of the kept prefixes
    return max(candidates)


for width in (1, 2):
    score, seq = beam_search(width)
    print(f"beam width {width}: {' '.join(seq)}  p = {math.exp(score):.2f}")
```

% output

```text
beam width 1: the pizza  p = 0.20
beam width 2: a cat  p = 0.36
```

Width 1 commits to "the" (0.5) and ends with $0.5 \times 0.4 = 0.20$; width 2 also keeps "a" (0.4), whose continuation "cat" gives $0.4 \times 0.9 = 0.36$. Beam search suits tasks with one correct output for an input, such as translation. For open-ended text {cite:t}`holtzman2020curious` find that "maximization-based decoding methods such as beam search lead to degeneration", output that is "bland, incoherent, or gets stuck in repetitive loops", which is why chat and story generation sample with temperature, top-k or top-p.

Each pass of the loop runs the model on the whole sequence so far, but under the causal mask the keys and values of the earlier positions do not change when a token is appended: the key and value at position $m$ depend only on the tokens at positions $0$ to $m$. Implementations therefore keep them in a **key-value (KV) cache**, which holds, for every layer and every key/value head, one key vector and one value vector per token read or generated so far. Each new token computes its own query, key and value, appends its key and value to the cache and attends over the cache, so the projections and the feed-forward network run for one position instead of the whole prefix; only the attention step still reads every cached position. The price is memory that grows with the sequence: with $L$ layers, key/value width $n_h d_k$ and 2-byte numbers the cache takes $2 \times L \times n_h d_k \times 2$ bytes per token, which for GPT-2 small ($L = 12$, $n_h d_k = 768$) is 36,864 bytes per token and 37,748,736 bytes for a full context of 1,024 tokens. {ref}`sec-ch6-inference-drives-design` compares the cache sizes of modern models.

{numref}`fig-ch1-autoregressive-loop` summarizes the generation loop that any of these rules plugs into.

```{figure} figures/autoregressive_loop.*
:name: fig-ch1-autoregressive-loop
:width: 76%
:alt: The prompt goes into the model, which gives a next-token distribution; a token is chosen; if it is EOS or the length limit is reached generation stops, otherwise the token is appended and the model runs again.

Generation: one forward pass per new token. Training does not loop, because teacher
forcing supplies the true tokens ({ref}`sec-ch1-teacher-forcing`).
```

(sec-ch1-training-process)=

### 6.2 Training Process

Training maximizes the likelihood of the training data, which is the same as minimizing the mean negative log-likelihood per token over the $T$ tokens of the data, in nats:

$$\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p_\theta(x_i \mid x_{<i})$$

The model outputs a probability distribution over the vocabulary:

$$p_\theta(\cdot \mid x_{<i}) = \text{softmax}(h^{(L)}_{i-1} W_{\text{out}})$$

where $h^{(L)}_{i-1}$ is the hidden state of the last layer $L$ at position $i-1$, which under causal masking has seen exactly $x_{<i}$. Positions count from 0 and the BOS token prepended to the sequence occupies position 0, so the real tokens $x_1, \dots, x_n$ sit at positions 1 to $n$ and the first of them is predicted from $h^{(L)}_{i-1}$ with $i = 1$, the hidden state of the BOS token ({ref}`sec-ch1-teacher-forcing`). GPT-2 and later models have no bias in this projection.

(sec-ch1-perplexity)=

### 6.3 Evaluation Metrics

**Perplexity** measures how well the model predicts the test data:

$$\text{Perplexity} = \exp\left(-\frac{1}{T} \sum_{i=1}^{T} \log p_\theta(x_i \mid x_{<i})\right) = \exp(\mathcal{L})$$

Perplexity was introduced as a measure of the difficulty of speech recognition tasks {cite}`jelinek1977perplexity`; it is the number of equally likely tokens that would give the same loss, and lower is better. Across model families it predicts downstream accuracy poorly: in HELM, bits per byte on the Pile was a poor predictor of downstream accuracy {cite}`liang2022holistic`, so evaluations report both.

(sec-ch1-bits-per-byte)=

#### Perplexity Depends on the Tokenizer; Bits per Byte Does Not

Perplexity is an average per token, so two models can only be compared by it if they cut the text into the same tokens. Suppose two models assign exactly the same probability to a 1 MB test text, a total of 700,000 nats, but model A's tokenizer splits it into 250,000 tokens and model B's into 200,000. Model A's loss is $700{,}000/250{,}000 = 2.80$ nats per token, a perplexity of $e^{2.80} = 16.44$; model B's is 3.50 nats, a perplexity of 33.12. B looks twice as bad while predicting the text exactly as well; it simply has fewer, larger tokens, each of which is harder to guess.

**Bits per byte** removes the tokenizer by dividing the total by the number of UTF-8 bytes $n_{bytes}$ instead of the number of tokens, and converting nats to bits {cite}`gao2020pile`:

$$\text{BPB} = \frac{T}{n_{bytes}} \cdot \frac{\mathcal{L}}{\ln 2}$$

where $T\mathcal{L}$ is the total loss in nats. The Pile's authors prefer it "due to its invariance to different tokenization schemes". Both models above score 1.0099 bits per byte:

```python
import math

n_bytes = 1_000_000  # the same 1 MB evaluation text
total_nll = 700_000.0  # summed -ln p over the text, in nats, for both models
for name, T in [("tokenizer A", 250_000), ("tokenizer B", 200_000)]:
    loss = total_nll / T
    ppl = math.exp(loss)
    print(f"{name}: {T:,} tokens, loss {loss:.2f} nats/token, perplexity {ppl:.2f}")
print(f"bits per byte, either tokenizer: {total_nll / (n_bytes * math.log(2)):.4f}")
```

% output

```text
tokenizer A: 250,000 tokens, loss 2.80 nats/token, perplexity 16.44
tokenizer B: 200,000 tokens, loss 3.50 nats/token, perplexity 33.12
bits per byte, either tokenizer: 1.0099
```

Perplexities of models with the same tokenizer, such as checkpoints of one training run, can be compared directly; across tokenizers, compare bits per byte.

```{admonition} Language Modeling in Practice
:class: note
The notebook ({ref}`nb-attn-lm`) builds the causal mask, applies it inside attention,
adds a language-model head, and computes cross-entropy and perplexity for an untrained
model so you can see what "no better than guessing" looks like. Optimization, schedules
and gradient flow are in the Chapter 2 notebook ({ref}`nb-math-optimizers`,
{ref}`nb-math-backprop`).
```

(sec-ch1-summary)=

## Summary

- A tokenizer turns text into integer ids. BPE starts from single characters and adds merged pieces in the order they were learned from the training text, so frequent words become single tokens; byte fallback spells any character outside the vocabulary as its UTF-8 bytes, so no input needs an unknown token ({ref}`sec-ch1-tokenization`). Byte-level BPE, which starts from the 256 byte values themselves, is taught in {ref}`sec-tdd-byte-level-bpe`.
- An embedding table maps each id to a vector of width $d$. Attention by itself ignores order, so position enters as an added sinusoidal or learned vector, as a bias on the scores (T5, ALiBi) or as a rotation of the queries and keys (RoPE), whose scores depend only on the offset $m' - m$ ({ref}`sec-ch1-rope`).
- Language modeling went from n-gram counts through recurrent networks to the transformer, whose attention lets every position read any other in a single step; pretraining at scale and instruction tuning followed ({ref}`sec-ch1-historical-context`).
- Scaled dot-product attention computes $\text{softmax}(QK^\top/\sqrt{d_k})\,V$: every row of weights is positive and sums to 1, and each output is a weighted average of the value vectors. The scores take $\mathcal{O}(n^2)$ memory per head ({ref}`sec-ch1-attention-complexity`). Multi-head attention runs several such maps on slices of width $d_k$ and concatenates them.
- A transformer block adds an attention sublayer and a feed-forward sublayer to a residual stream, each with a normalization. With biases and $d_{ff} = 4d$ a block holds $12d^2 + 13d$ parameters, which gives GPT-2 small its 124,439,808 ({ref}`sec-ch1-parameter-count`, worked in {ref}`sec-ch1-exercise-7`).
- A decoder-only model is trained under a causal mask to minimize the mean negative log-likelihood of the next token. Perplexity, $\exp(\mathcal{L})$, compares models that share a tokenizer; bits per byte compares models that do not ({ref}`sec-ch1-perplexity`).
- An encoder such as BERT is trained instead with the masked-language-model objective: some input tokens are hidden or replaced, and the loss is the mean negative log-likelihood of the original tokens at those positions only ({ref}`sec-ch1-masked-lm`).
- Text is generated one token at a time: greedy decoding takes the most probable token, beam search keeps the $b_{beam}$ most probable prefixes, and temperature, top-k and top-p sampling trade fidelity for variety; a KV cache keeps the keys and values of earlier positions so that each step processes only the new token ({ref}`sec-ch1-decoding`).

(sec-ch1-exercises)=

## Exercises

The exercises are graded: level 1 applies one definition by hand, level 2 combines two ideas, level 3 asks for a short proof. The test suite checks every numerical answer.

(sec-ch1-exercise-1)=

### Exercise 1: Two BPE merges by hand (level 1)

A corpus holds the word "low" 5 times, "lower" 2 times, "newest" 6 times and "widest" 3 times. Split each word into characters followed by the end-of-word symbol `</w>`, as `SimpleBPE` does. Count every adjacent pair, weighted by word frequency, and find the first two merges. When several pairs tie, take the one that occurs first in the corpus, which is what `max` does in `SimpleBPE.train`. What is the third merge, and how is "lowest" segmented after three merges?

```{admonition} Solution 1
:class: dropdown
The pair counts are: (l, o) 5 + 2 = 7; (o, w) 7; (w, `</w>`) 5; (w, e) 2 + 6 = 8; (e,
r) 2; (r, `</w>`) 2; (n, e) 6; (e, w) 6; (e, s) 6 + 3 = 9; (s, t) 9; (t, `</w>`) 9;
(w, i) 3; (i, d) 3; (d, e) 3. Three pairs tie at 9 and (e, s) occurs first, in
"newest", so **merge 1 is (e, s)**. The words become n e w es t `</w>` and w i d es t
`</w>`, which gives (es, t) 9 and (t, `</w>`) 9, with (es, t) first: **merge 2 is (es,
t)**. The only pair left at 9 is (est, `</w>`), so **merge 3 is (est, `</w>`)**.
"lowest" then splits as l, o, w, est`</w>`: the suffix learned from "newest" and
"widest" applies to a word the corpus never contained, while "low" stays in pieces
because (l, o), at 7, never reached the top. Training `SimpleBPE` on this corpus for
three merges gives the same list.
```

(sec-ch1-exercise-2)=

### Exercise 2: A 2×2 attention output (level 1)

With $n = 2$ and $d_k = 2$, let the rows of $Q$ be $(1, 0)$ and $(1, 1)$, the rows of $K$ be $(2, 0)$ and $(0, 2)$, and the rows of $V$ be $(1, 2)$ and $(3, 0)$. Compute the scores $QK^\top/\sqrt{d_k}$, the attention weights and the output.

```{admonition} Solution 2
:class: dropdown
$QK^\top$ has rows $(2, 0)$ and $(2, 2)$; dividing by $\sqrt{2}$ gives the scores
$(1.4142, 0)$ and $(1.4142, 1.4142)$. Row 1: $e^{1.4142} = 4.113$, so the weights are
$4.113/5.113 = 0.8044$ and $1/5.113 = 0.1956$. Row 2 has equal scores, so its weights
are $(0.5, 0.5)$. The outputs are $0.8044\,(1, 2) + 0.1956\,(3, 0) = (1.3911, 1.6089)$
and $0.5\,(1, 2) + 0.5\,(3, 0) = (2, 1)$. Query 2 matches both keys equally and gets the
plain average of the values; query 1 points along key 1 and is pulled toward value 1.
```

(sec-ch1-exercise-3)=

### Exercise 3: The perplexity of a uniform model (level 1)

A model assigns probability $1/|\mathcal{V}|$ to every token of a vocabulary of $|\mathcal{V}| = 50{,}257$ at every position. What are its loss $\mathcal{L}$ in nats per token and its perplexity, on any text?

```{admonition} Solution 3
:class: dropdown
Every term of the loss is $-\log(1/|\mathcal{V}|) = \ln |\mathcal{V}|$, so the mean over
the $T$ tokens is $\mathcal{L} = \ln 50{,}257 = 10.82$ nats per token, whatever the
text. The perplexity is $\exp(\mathcal{L}) = |\mathcal{V}| = 50{,}257$: the model is as
uncertain as a uniform choice among all 50,257 tokens, which is the reading of
perplexity given in {ref}`sec-ch1-perplexity`.
```

(sec-ch1-exercise-4)=

### Exercise 4: Masked scores (level 2)

In a sequence of $n = 3$ tokens, query 2 has the scores $(1, 2, 3)$ against keys 1, 2 and 3. (a) Which key does the causal mask hide from it, and what are its attention weights? (b) How many scores does the causal mask hide in a sequence of $n = 1{,}024$ tokens? (c) If every key of a row is masked, what does `torch.softmax` return when the masked scores are $-\infty$, and what when they are `torch.finfo(torch.float32).min`?

```{admonition} Solution 4
:class: dropdown
(a) Key 3 comes after query 2, so its score becomes $-\infty$ and its weight
$e^{-\infty} = 0$. The other two scores give $e^1/(e^1 + e^2) = 1/(1 + e) = 0.2689$ and
$e/(1 + e) = 0.7311$: the weights are $(0.2689, 0.7311, 0)$.

(b) The mask hides the entries above the diagonal of the $n \times n$ score matrix,
$n(n - 1)/2 = 1{,}024 \times 1{,}023 / 2 = 523{,}776$ of them, just under half of the
$1{,}048{,}576$ scores.

(c) `torch.softmax` subtracts the row maximum before it exponentiates. With every score
$-\infty$ the maximum is $-\infty$, and $-\infty - (-\infty)$ is NaN, so all three
weights are NaN. With the finite minimum every score is the same number, the differences
are 0, and the weights are uniform, $(1/3, 1/3, 1/3)$. This is why the chapter's
listings use the finite value: a padding query whose keys are all masked produces a
harmless average instead of a NaN that spreads through the batch.
```

(sec-ch1-exercise-5)=

### Exercise 5: Temperature (level 2)

The logits over a three-token vocabulary are $z = (2, 1, 0)$. Compute the sampling distribution $\text{softmax}(z/\tau)$ for $\tau = 1$ and for $\tau = 0.5$, and say what happens to the probability of the most likely token as $\tau \to 0$.

```{admonition} Solution 5
:class: dropdown
For $\tau = 1$: $e^2 = 7.389$, $e^1 = 2.718$ and $e^0 = 1$ sum to 11.11, giving
$(0.6652, 0.2447, 0.0900)$. For $\tau = 0.5$ the logits double to $(4, 2, 0)$:
$e^4 = 54.60$, $e^2 = 7.389$ and $1$ sum to 62.99, giving $(0.8668, 0.1173, 0.0159)$.
Halving the temperature doubles every gap between logits, so every ratio of two
probabilities is squared: the ratio of the first two goes from $e^1 = 2.718$ to
$e^2 = 7.389$. As $\tau \to 0$ the gaps grow without bound and the top token's
probability goes to 1: sampling becomes greedy decoding.
```

(sec-ch1-exercise-6)=

### Exercise 6: From perplexity to bits per byte (level 2)

A model's loss on a test text is 2.0 nats per token, and its tokenizer produces one token per 4.0 UTF-8 bytes of that text on average. (a) What are its perplexity and its bits per byte? (b) A second model has the same bits per byte with a tokenizer that produces one token per 3.0 bytes. What are its loss and its perplexity?

```{admonition} Solution 6
:class: dropdown
(a) The perplexity is $e^{2.0} = 7.389$. With $T/n_{bytes} = 1/4$, bits per byte is
$\frac{1}{4} \cdot \frac{2.0}{\ln 2} = 0.7213$.

(b) The same bits per byte with $T/n_{bytes} = 1/3$ needs
$\frac{1}{3} \cdot \frac{\mathcal{L}}{\ln 2} = 0.7213$, so
$\mathcal{L} = 3 \times 0.7213 \times \ln 2 = 1.5$ nats per token and a perplexity of
$e^{1.5} = 4.482$. The second model looks better by perplexity (4.482 against 7.389)
although both predict the text equally well: its tokens are shorter, so each is easier
to guess.
```

(sec-ch1-exercise-7)=

### Exercise 7: GPT-2 small's parameters (level 2)

GPT-2 small has $|\mathcal{V}| = 50{,}257$, $d = 768$, $L = 12$ blocks, $n_{ctx} = 1{,}024$ learned positions {cite}`radford2019language`. Assume biases, LayerNorm, $d_{ff} = 4d$ and an output layer tied to $W_E$. Count its parameters with the formula of {ref}`sec-ch1-parameter-count`, and say how many it would have with an untied output layer.

```{admonition} Solution 7
:class: dropdown
Token embeddings: $50{,}257 \times 768 = 38{,}597{,}376$. Positions:
$1{,}024 \times 768 = 786{,}432$. One block:
$12 \times 768^2 + 13 \times 768 = 7{,}077{,}888 + 9{,}984 = 7{,}087{,}872$, and twelve
blocks hold $85{,}054{,}464$. The final LayerNorm adds $2 \times 768 = 1{,}536$. The
total is $38{,}597{,}376 + 786{,}432 + 85{,}054{,}464 + 1{,}536 = 124{,}439{,}808$, and
`count_parameters(50257, 768, 12, 1024, tied=True)` returns the same number. An untied
output layer would add another $768 \times 50{,}257 = 38{,}597{,}376$, for
$163{,}037{,}184$. In this small model the two embedding tables are 32% of the total, so
the $12Ld^2$ estimate that serves for GPT-3 would be far off here.
```

(sec-ch1-exercise-8)=

### Exercise 8: The memory of the score matrix (level 2)

A layer with 32 heads attends over $n = 4{,}096$ tokens and stores its attention weights in fp16, 2 bytes per number. How many bytes do the weights of one layer take for one sequence? What happens when $n$ doubles?

```{admonition} Solution 8
:class: dropdown
Each head has $n^2 = 4{,}096^2 = 16{,}777{,}216$ weights, so 32 heads hold
$536{,}870{,}912$ numbers, or $1{,}073{,}741{,}824$ bytes: exactly 1 GiB per layer and
sequence. Doubling $n$ multiplies the count by four, to 4 GiB. This quadratic memory is
what FlashAttention avoids by never storing the full matrix
({ref}`sec-ch1-attention-complexity`).
```

(sec-ch1-exercise-9)=

### Exercise 9: RoPE's offset property from two rules (level 3)

Take $d_k = 2$, so that the single pair has frequency $b^0 = 1$ and $\mathcal{R}_m$ of {eq}`eq-ch1-rope` is the rotation by angle $m$. (a) Multiply out $\mathcal{R}_m \mathcal{R}_{m'}$ and show that it equals $\mathcal{R}_{m + m'}$; show that $\mathcal{R}_m^\top = \mathcal{R}_{-m}$. (b) Use only these two rules, with no further trigonometry, to show that the score $\tilde{q}_m \tilde{k}_{m'}^\top$ depends on $m$ and $m'$ only through $m' - m$. (c) With $q = (1, 2)$ and $k = (3, -1)$, compute the score at $m = 0$, $m' = 1$ and at $m = 4$, $m' = 5$.

```{admonition} Solution 9
:class: dropdown
(a) The top-left entry of $\mathcal{R}_m \mathcal{R}_{m'}$ is
$\cos m \cos m' - \sin m \sin m' = \cos(m + m')$ and the bottom-left entry is
$\sin m \cos m' + \cos m \sin m' = \sin(m + m')$; the other two entries follow in the
same way, so the product is $\mathcal{R}_{m + m'}$. Transposing $\mathcal{R}_m$ swaps
the signs of its two sine entries, and since $\cos(-m) = \cos m$ and
$\sin(-m) = -\sin m$, that is $\mathcal{R}_{-m}$.

(b)

$$\tilde{q}_m \tilde{k}_{m'}^\top = q\,\mathcal{R}_m^\top \mathcal{R}_{m'}\,k^\top
= q\,\mathcal{R}_{-m} \mathcal{R}_{m'}\,k^\top = q\,\mathcal{R}_{m' - m}\,k^\top :$$

The first step is the second rule, the second step the first rule with $-m$ in place of
$m$. Only $m' - m$ is left.

(c) Both pairs have offset 1, so both scores are $q\,\mathcal{R}_1 k^\top$.
$\mathcal{R}_1 k^\top = (3\cos 1 + \sin 1,\; 3\sin 1 - \cos 1)^\top$, and the dot
product with $q = (1, 2)$ is
$3\cos 1 + \sin 1 + 6\sin 1 - 2\cos 1 = \cos 1 + 7\sin 1 = 0.5403 + 5.8903 = 6.4306$.
The unrotated score $q k^\top$ is $3 - 2 = 1$. `apply_rope` gives 6.4306 at both pairs
of positions.
```

(sec-ch1-further-reading)=

## Further Reading

(sec-ch1-further-reading-learning-path-beginners)=

### Learning Path for Beginners

**Start Here - Intuitive Explanations:**

1. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) {cite}`alammar2018illustrated` - Visual explanations of transformer concepts
1. [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) {cite}`weng2018attention` - A survey of attention mechanisms up to and including the transformer
1. [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) {cite}`karpathy2022zerotohero` - Video series building transformers from scratch

**Mathematical Foundations:**

1. [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) {cite}`raschka2023understanding` - A reading list of the papers behind LLMs
1. [Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html), chapter 2 of *Deep Learning* {cite}`goodfellow2016deep`

(sec-ch1-further-reading-foundational-papers)=

### Foundational Papers

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) {cite}`vaswani2017attention`: The original transformer paper
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) {cite}`radford2018improving`: GPT and the decoder-only architecture
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) {cite}`devlin2018bert`: BERT and bidirectional representations

(sec-ch1-further-reading-technical-implementation-resources)=

### Technical Implementation Resources

**Code Implementations:**

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) {cite}`rush2018annotated` - Line-by-line implementation with explanations
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) {cite}`wolf2020transformers` - A library of pretrained transformer implementations
- [minGPT](https://github.com/karpathy/minGPT) {cite}`karpathy2020mingpt` - Minimal GPT implementation for learning

**Interactive Learning:**

- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) {cite}`cho2024transformer` - Interactive visualization of GPT-2 running in the browser
- [Tiktokenizer](https://tiktokenizer.vercel.app/) {cite}`tiktokenizer` - Explore how GPT and other tokenizers split text
- [BertViz](https://github.com/jessevig/bertviz) {cite}`vig2019multiscale` - Visualize attention in BERT and GPT models

(sec-ch1-further-reading-advanced-topics-recent-research)=

### Advanced Topics and Recent Research

**Scaling Laws and Training:**

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) {cite}`kaplan2020scaling` - Understanding model scaling behavior
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) {cite}`hoffmann2022training` - Chinchilla scaling laws

**Technical Deep Dives:**

- [A Primer in BERTology](https://arxiv.org/abs/2002.12327) {cite}`rogers2020primer` - What probing studies found about how BERT works
- [Pre-trained Models for Natural Language Processing](https://arxiv.org/abs/2003.08271) {cite}`qiu2020pre` - Survey of pre-trained models
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) {cite}`tenney2019bert` - Probing which layers of BERT encode which linguistic structure

**Architecture and Efficiency:**

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) {cite}`su2021roformer` - RoPE positional encodings
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) {cite}`dao2022flashattention` - Efficient attention computation
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) {cite}`touvron2023llama` - Modern transformer design choices

(sec-ch1-further-reading-books-comprehensive-resources)=

### Books and Comprehensive Resources

1. **[Natural Language Processing with Transformers](https://github.com/nlp-with-transformers/notebooks)** {cite}`tunstall2022natural` - Building and fine-tuning transformer models with the Hugging Face libraries
1. **[Deep Learning](https://www.deeplearningbook.org/)** {cite}`goodfellow2016deep` - Mathematical foundations
1. **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** {cite}`jurafsky2026speech` - NLP fundamentals

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
