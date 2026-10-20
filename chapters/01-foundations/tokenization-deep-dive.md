# Tokenization in Depth

This page collects the tokenizer material from Chapter 1 that goes beyond what you need to follow the transformer: modern tokenizers, special tokens, vocabulary construction, cross-lingual and domain issues, quality metrics, a complete BPE tokenizer, and analysis tools. Read {doc}`index` first.

## T.1 Modern Tokenization Strategies

### Tiktoken (OpenAI's Tokenizer)

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

### Multi-Modal Tokenization

Modern models handle multiple modalities:

- **Text tokens**: Standard subword tokens
- **Image tokens**: Patch embeddings or discrete visual tokens
- **Audio tokens**: Spectrogram patches or audio codebook tokens
- **Special tokens**: Task indicators, separators, padding

### Vocabulary Construction Strategies

**Frequency-based**: Include most common tokens

```python
# pseudo-code: `tokenize` stands for any tokenizer function
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
# pseudo-code: `tokenize` stands for any tokenizer function
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

## T.2 Tokenization Impact on Model Behavior

### Length and Efficiency

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

### Linguistic Bias

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

### Out-of-Vocabulary Handling

Different strategies for unknown tokens:

1. **UNK token**: Replace unknown words with `<UNK>`
1. **Subword fallback**: Break unknown words into subwords
1. **Character fallback**: Fall back to character-level for unknown segments
1. **Byte fallback**: Ultimate fallback to byte representation

## T.3 Special Tokens and Their Roles

### Common Special Tokens

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

### Token IDs and Embeddings

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

## T.4 Practical Considerations

### Choosing Vocabulary Size

Common vocabulary sizes and their trade-offs:

- **Small (1K-8K)**: character-like; used for code-only or single-domain models
- **Medium (32K-50K)**: GPT-2 (50k), the original LLaMA (32k); the usual choice for research-scale models
- **Large (100K-150K)**: GPT-4's `cl100k_base` (~100k), Llama 3 (128k), Qwen (~150k); better compression of multilingual text and code
- **Very large (250K+)**: Gemma (256k) and multilingual models; the embedding and output matrices become a noticeable share of a small model's parameters

### Tokenization Efficiency

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

### Reversibility and Information Loss

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

## T.5 Tokenization Challenges and Solutions

### Cross-Lingual Tokenization

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

### Handling Special Cases

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

### Domain Adaptation

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

### Quality Metrics for Tokenization

```python
def evaluate_tokenization_quality(tokenizer, corpus):
    """Evaluate a ComprehensiveTokenizer-style tokenizer (encode/decode/special_tokens)."""

    # Only metrics this function actually measures are reported. Morphological consistency
    # and cross-lingual equity need linguistic annotations or parallel corpora.
    metrics = {"compression_ratio": 0, "oov_rate": 0, "reversibility_errors": 0}

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
        oov_count += sum(1 for token in tokens if token == tokenizer.special_tokens["<UNK>"])

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

### Best Practices for Tokenization

**1. Vocabulary Size Selection**:

```python
# pseudo-code: `train_bpe_tokenizer` and `evaluate_tokenization_quality` are defined elsewhere in this section
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
import re

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
import matplotlib.pyplot as plt

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
# pseudo-code: `train_tokenizer` stands for any tokenizer trainer
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

## T.6 Practical Implementation

This section provides hands-on implementation of tokenization and embedding concepts:

### Complete Tokenization Pipeline

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
        """Count word frequencies in corpus (case is preserved so decoding is exact)."""
        word_freq = Counter()

        for text in corpus:
            word_freq.update(self.preprocess_text(text).split())

        return word_freq

    def train_bpe(self, corpus: List[str]) -> None:
        """Train Byte Pair Encoding tokenizer."""
        print("Training BPE tokenizer...")

        # Get word frequencies
        word_freq = self.get_word_frequencies(corpus)

        # Filter by frequency
        word_freq = {
            word: freq for word, freq in word_freq.items() if freq >= self.min_frequency
        }

        # Convert words to symbol sequences with an end-of-word marker
        word_splits = {}
        for word, freq in word_freq.items():
            word_splits[tuple(word) + ("</w>",)] = freq

        # Initial vocabulary: every symbol that can appear in a split, including the
        # end-of-word marker itself (otherwise an unmerged '</w>' would decode as <UNK>).
        # Special tokens are NOT added here; they get the fixed ids 0..4 below.
        vocab = set()
        for word in word_splits:
            vocab.update(word)

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

        # Final mapping: special tokens keep ids 0..4, then the learned symbols
        vocab_list = list(self.special_tokens.keys()) + sorted(vocab)
        self.vocab = {token: i for i, token in enumerate(vocab_list)}
        assert all(self.vocab[t] == i for t, i in self.special_tokens.items())
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
        words = self.preprocess_text(text).split()

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


# Example usage. Printed output: 'Training complete. Final vocabulary size: 153', then
# Encoded: [2, 12, 110, 26, 49, 3] and an exact round trip of 'The quick brown fox'
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

# Test encoding/decoding: the round trip must be exact, including case
test_text = "The quick brown fox"
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, (decoded, test_text)

print(f"Original: {test_text}")
print(f"Encoded: {encoded}")      # [2, ..., 3]: <BOS> ... <EOS>
print(f"Decoded: {decoded}")
```

### Tokenization Analysis Tools

```python
# pseudo-code: uses `tokenizer` and `sample_corpus` from the previous cell
import numpy as np
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

