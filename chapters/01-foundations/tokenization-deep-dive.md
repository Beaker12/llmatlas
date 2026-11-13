# Tokenization in Depth

This page collects the tokenizer material from Chapter 1 that goes beyond what you need to follow the transformer: modern tokenizers, special tokens, vocabulary construction, cross-lingual and domain issues, quality metrics, a complete BPE tokenizer, and analysis tools. Read {doc}`index` first.

(sec-tdd-modern-tokenization-strategies)=

## T.1 Modern Tokenization Strategies

(sec-tdd-tiktoken-tokenizer)=

### Tiktoken (OpenAI's Tokenizer)

GPT models use a sophisticated tokenizer that:

- Handles regex patterns for different text types
- Uses byte-level BPE
- Includes special tokens for different contexts

```python
# network
import hashlib
import os
import tempfile
import urllib.request

import tiktoken

# tiktoken downloads its encoding files without a User-Agent we can set, so fetch them ourselves into the
# cache directory tiktoken reads (file name: SHA-1 of the URL) and send LLMATLAS_USER_AGENT.
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; reader)")
BLOB = "https://openaipublic.blob.core.windows.net/"
FILES = {
    "gpt2": ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"],
    "cl100k_base": ["encodings/cl100k_base.tiktoken"],
    "o200k_base": ["encodings/o200k_base.tiktoken"],
}
CACHE = os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tiktoken-cache"))


def get_encoding(name):
    os.makedirs(CACHE, exist_ok=True)
    for path in FILES[name]:
        target = os.path.join(CACHE, hashlib.sha1((BLOB + path).encode()).hexdigest())
        if not os.path.exists(target):
            request = urllib.request.Request(BLOB + path, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as f:
                f.write(response.read())
    return tiktoken.get_encoding(name)


enc = get_encoding("cl100k_base")  # the GPT-4 encoding

text = "Hello, world! Here's some code: `print('hi')`"
tokens = enc.encode(text)
assert enc.decode(tokens) == text  # byte-level BPE is lossless
print(f"Token count: {len(tokens)}")
print(f"Pieces: {[enc.decode([i]) for i in tokens]}")
```

% output

```text
Token count: 15
Pieces: ['Hello', ',', ' world', '!', ' Here', "'s", ' some', ' code', ':', ' `', 'print', "('", 'hi', "')", '`']
```

(sec-tdd-multi-modal-tokenization)=

### Multi-Modal Tokenization

Modern models handle multiple modalities:

- **Text tokens**: Standard subword tokens
- **Image tokens**: Patch embeddings or discrete visual tokens
- **Audio tokens**: Spectrogram patches or audio codebook tokens
- **Special tokens**: Task indicators, separators, padding

(sec-tdd-vocabulary-construction-strategies)=

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

(sec-tdd-tokenization-impact-model-behavior)=

## T.2 Tokenization Impact on Model Behavior

(sec-tdd-sequence-length-cost)=

### Length and Efficiency

Different tokenization schemes produce different sequence lengths:

```python
# Example text
text = "The transformer architecture revolutionized natural language processing."

# Word-level: split on spaces
word_tokens = text.lower().split()
print(f"Word tokens ({len(word_tokens)}): {word_tokens}")

# Character-level
char_tokens = list(text)
print(f"Character tokens ({len(char_tokens)}): {char_tokens[:20]}...")

# Subword: the output of tiktoken's gpt2, cl100k_base and o200k_base encodings, which all give the same
# 9 tokens here (run the network fence under "Linguistic Bias" to reproduce it)
subword_tokens = [
    "The", " transformer", " architecture", " revolution", "ized", " natural", " language", " processing", ".",
]
assert "".join(subword_tokens) == text
print(f"Subword tokens ({len(subword_tokens)}): {subword_tokens}")
```

% output

```text
Word tokens (7): ['the', 'transformer', 'architecture', 'revolutionized', 'natural', 'language', 'processing.']
Character tokens (72): ['T', 'h', 'e', ' ', 't', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r', ' ', 'a', 'r', 'c', 'h']...
Subword tokens (9): ['The', ' transformer', ' architecture', ' revolution', 'ized', ' natural', ' language', ' processing', '.']
```

**Computational implications**:

- Sequence length affects attention computation: $\mathcal{O}(n^2)$ for self-attention over $n$ tokens
- Longer sequences require more memory
- Processing time scales with sequence length

(sec-tdd-linguistic-bias)=

### Linguistic Bias

Tokenization can introduce linguistic biases. A tokenizer trained mostly on English text spends more tokens on other languages, so the same content costs more compute and fits less into the context window. Larger vocabularies trained on more languages narrow the gap; the Japanese greeting below takes 10 tokens in GPT-2's encoding, 4 in `cl100k_base` and 2 in `o200k_base`:

```python
# network
import hashlib
import os
import tempfile
import urllib.request

import tiktoken

# tiktoken downloads its encoding files without a User-Agent we can set, so fetch them ourselves into the
# cache directory tiktoken reads (file name: SHA-1 of the URL) and send LLMATLAS_USER_AGENT.
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; reader)")
BLOB = "https://openaipublic.blob.core.windows.net/"
FILES = {
    "gpt2": ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"],
    "cl100k_base": ["encodings/cl100k_base.tiktoken"],
    "o200k_base": ["encodings/o200k_base.tiktoken"],
}
CACHE = os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tiktoken-cache"))


def get_encoding(name):
    os.makedirs(CACHE, exist_ok=True)
    for path in FILES[name]:
        target = os.path.join(CACHE, hashlib.sha1((BLOB + path).encode()).hexdigest())
        if not os.path.exists(target):
            request = urllib.request.Request(BLOB + path, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as f:
                f.write(response.read())
    return tiktoken.get_encoding(name)


# The same greeting in five languages, counted by three OpenAI encodings (2019, 2023, 2024)
texts = {
    "English": "Hello world",
    "French": "Bonjour le monde",
    "Spanish": "Hola mundo",
    "Japanese": "こんにちは世界",
    "Arabic": "مرحبا بالعالم",
}
names = ["gpt2", "cl100k_base", "o200k_base"]
encodings = {name: get_encoding(name) for name in names}
print(f"{'Language':<10}" + "".join(f"{name:>13}" for name in names))
for language, text in texts.items():
    print(f"{language:<10}" + "".join(f"{len(encodings[n].encode(text)):>13}" for n in names))
```

% output

```text
Language           gpt2  cl100k_base   o200k_base
English               2            2            2
French                6            3            3
Spanish               4            2            2
Japanese             10            4            2
Arabic               12           10            4
```

(sec-tdd-out-of-vocabulary-handling)=

### Out-of-Vocabulary Handling

Different strategies for unknown tokens:

1. **UNK token**: Replace unknown words with `<UNK>`
1. **Subword fallback**: Break unknown words into subwords
1. **Character fallback**: Fall back to character-level for unknown segments
1. **Byte fallback**: Ultimate fallback to byte representation

(sec-tdd-special-tokens-their-roles)=

## T.3 Special Tokens and Their Roles

(sec-tdd-common-special-tokens)=

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

(sec-tdd-token-ids-embeddings)=

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

(sec-tdd-practical-considerations)=

## T.4 Practical Considerations

(sec-tdd-choosing-vocabulary-size)=

### Choosing Vocabulary Size

Common vocabulary sizes and their trade-offs:

- **Small (1K-8K)**: character-like; used for code-only or single-domain models
- **Medium (32K-50K)**: GPT-2 (50k), the original LLaMA (32k); the usual choice for research-scale models
- **Large (100K-150K)**: GPT-4's `cl100k_base` (~100k), Llama 3 (128k), Qwen (~150k); better compression of multilingual text and code
- **Very large (250K+)**: Gemma (256k) and multilingual models; the embedding and output matrices become a noticeable share of a small model's parameters

(sec-tdd-efficiency-considerations)=

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

(sec-tdd-reversibility-information-loss)=

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

(sec-tdd-tokenization-challenges-solutions)=

## T.5 Tokenization Challenges and Solutions

(sec-tdd-cross-lingual-tokenization)=

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

(sec-tdd-handling-special-cases)=

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

(sec-tdd-domain-adaptation)=

### Domain Adaptation

**Medical Text**:

```python
medical_text = (
    "Patient presents with myocardial infarction and diabetes mellitus type 2"
)

# A domain vocabulary with every term as one token (hypothetical):
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

# GPT-2's encoding (tiktoken "gpt2") splits the medical terms; cl100k_base needs 15 tokens, o200k_base 12
generic_tokens = [
    "Pat", "ient", " presents", " with", " my", "ocard", "ial", " inf", "ar", "ction",
    " and", " diabetes", " mell", "itus", " type", " 2",
]
assert "".join(generic_tokens) == "".join(medical_tokens) == medical_text
print(f"Domain vocabulary: {len(medical_tokens)} tokens; gpt2: {len(generic_tokens)} tokens")
```

% output

```text
Domain vocabulary: 10 tokens; gpt2: 16 tokens
```

**Legal Text**:

```python
legal_text = "The aforementioned clause notwithstanding the heretofore provisions"

# Legal language has specific terminology patterns
# Need specialized vocabulary for legal corpus
```

(sec-tdd-quality-metrics-tokenization)=

### Quality Metrics for Tokenization

```python
def evaluate_tokenization_quality(tokenizer, corpus):
    """Evaluate a tokenizer with encode/decode and special_tokens, such as ComprehensiveTokenizer below."""

    # Only metrics this function actually measures are reported. A byte-level tokenizer has no
    # out-of-vocabulary tokens, so there is no OOV rate. Morphological consistency and cross-lingual
    # equity need linguistic annotations or parallel corpora.
    special = set(tokenizer.special_tokens.values())
    total_chars = total_bytes = total_tokens = reversibility_errors = 0

    for text in corpus:
        tokens = tokenizer.encode(text)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += sum(1 for token in tokens if token not in special)
        if tokenizer.decode(tokens) != text:
            reversibility_errors += 1

    return {
        "chars_per_token": total_chars / total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "reversibility_errors": reversibility_errors / len(corpus),
    }


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

(sec-tdd-best-practices-tokenization)=

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

(sec-tdd-implementation-walkthrough)=

## T.6 Practical Implementation

This section implements a tokenizer end to end. It is byte-level BPE, the scheme GPT-2 introduced: the base vocabulary is the 256 byte values, so any UTF-8 text can be encoded without an unknown token; whitespace stays inside the tokens (a token can start with a space, and runs of spaces or newlines are tokens of their own), so decoding returns the input exactly; and every chunk of the training text counts, however rare. With only 39 merges on a five-sentence corpus, frequent words such as `The` become one token while an unseen word falls back to short pieces.

(sec-tdd-complete-tokenization-pipeline)=

### Complete Tokenization Pipeline

```python
import re
from collections import Counter
from typing import Dict, List, Tuple

# Pre-tokenization in the style of GPT-2: a chunk is an optional leading space followed by letters, digits or
# punctuation, or a run of whitespace. Joining the chunks gives back the text exactly, so whitespace is kept
# inside the tokens instead of being thrown away.
CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


class ComprehensiveTokenizer:
    """Byte-level BPE: 256 byte tokens, five special tokens, then learned merges. Every text round-trips."""

    def __init__(self, vocab_size: int = 300):
        assert vocab_size >= 256 + 5
        self.vocab_size = vocab_size
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # id -> the bytes it stands for
        names = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
        self.special_tokens = {name: 256 + i for i, name in enumerate(names)}  # ids 256..260
        self.merges: Dict[Tuple[int, int], int] = {}  # (left id, right id) -> merged id, in training order

    @staticmethod
    def _merge(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train_bpe(self, corpus: List[str]) -> None:
        """Learn vocab_size - 261 merges. Every chunk is counted; nothing is dropped for being rare."""
        chunk_freq = Counter(c for text in corpus for c in CHUNK.findall(text))
        words = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_freq}
        next_id = 256 + len(self.special_tokens)
        while next_id < self.vocab_size:
            pairs = Counter()
            for chunk, ids in words.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += chunk_freq[chunk]
            if not pairs:
                break
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))  # ties go to the smaller ids
            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = {chunk: self._merge(ids, best, next_id) for chunk, ids in words.items()}
            next_id += 1
        print(f"Training complete. {len(self.merges)} merges, vocabulary size {len(self.vocab) + 5}")

    def encode_chunk(self, chunk: str) -> List[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge present, as the trainer did
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> List[int]:
        ids = [self.special_tokens["<BOS>"]]
        for chunk in CHUNK.findall(text):
            ids.extend(self.encode_chunk(chunk))
        return ids + [self.special_tokens["<EOS>"]]

    def decode(self, ids: List[int]) -> str:
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab)  # special ids are not in vocab
        return data.decode("utf-8", errors="replace")

    def pieces(self, text: str) -> List[str]:
        """The tokens of `text` as strings, for display (special tokens left out)."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)[1:-1]]


sample_corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "The only thing we have to fear is fear itself",
    "Ask not what your country can do for you",
] * 100

tokenizer = ComprehensiveTokenizer(vocab_size=300)  # 39 merges: small enough that words still split
tokenizer.train_bpe(sample_corpus)

for text in ["The quick brown fox.", "Hello, world!", "two  spaces\nand a newline", "unthinkable", "你好"]:
    assert tokenizer.decode(tokenizer.encode(text)) == text, text  # exact round trip, including whitespace
for text in ["The quick brown fox.", "two  spaces\nand a newline", "unthinkable"]:  # the last is unseen
    print(f"{text!r} -> {tokenizer.pieces(text)}")
```

% output

```text
Training complete. 39 merges, vocabulary size 300
'The quick brown fox.' -> ['The', ' qu', 'i', 'c', 'k', ' b', 'r', 'o', 'w', 'n', ' fo', 'x', '.']
'two  spaces\nand a newline' -> ['t', 'w', 'o', ' ', ' s', 'p', 'a', 'c', 'es', '\n', 'an', 'd', ' a', ' n', 'e', 'w', 'l', 'in', 'e']
'unthinkable' -> ['u', 'n', 't', 'h', 'in', 'k', 'a', 'b', 'l', 'e']
```

(sec-tdd-tokenization-analysis-tools)=

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
        tokens = [tokenizer.vocab.get(tid, b"").decode("utf-8", "replace") for tid in token_ids]

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

