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

(sec-tdd-subword-algorithms)=

## T.2 Subword Algorithms in Practice

The chapter builds byte pair encoding (BPE) by hand. Production tokenizers use four related schemes: WordPiece, the unigram language model, byte-level BPE and the SentencePiece library, which implements BPE and unigram. They differ in how they choose the vocabulary and in how they split a word once the vocabulary is fixed. The listing at the end of this section runs a small version of each.

(sec-tdd-wordpiece)=

### WordPiece

WordPiece comes from Google's speech and translation systems {cite}`wu2016google` and is the tokenizer of BERT {cite}`devlin2018bert`. Training starts, as BPE does, from single characters and merges pairs, but it does not merge the most frequent pair. It merges the pair that most increases the likelihood of the training data {cite}`kudo2018subword`. Google never published its training algorithm; the score below is the reconstruction in the Hugging Face course {cite}`huggingface2026wordpiece`, which merges the pair with the highest

$$\operatorname{score}(\text{pair}) = \frac{\operatorname{count}(\text{pair})}{\operatorname{count}(\text{first piece}) \cdot \operatorname{count}(\text{second piece})}.$$

Dividing by the counts of the parts favors pairs whose parts rarely occur apart. In the listing's corpus the pair `##u ##g` occurs 20 times, the most of any pair, but `##u` and `##g` are both common on their own, so its score is $20 / (36 \cdot 20) = 0.0278$; `##g ##s` occurs only 5 times, but `##s` never occurs without `##g` before it, so its score is $5 / (20 \cdot 5) = 0.05$ and WordPiece merges it first.

Pieces that continue a word carry the prefix `##`, so `##ing` and `ing` are different tokens. WordPiece keeps only the final vocabulary, not the merge order, and encodes a word greedily: take the longest vocabulary entry that starts the word, then the longest `##` entry that continues it, and so on. If at some point no entry fits, the whole word becomes `[UNK]`, not just the character that failed.

(sec-tdd-unigram)=

### Unigram Language Model

The unigram model {cite}`kudo2018subword` works in the opposite direction. It starts from a large seed vocabulary (all characters plus the most frequent substrings of the corpus) and shrinks it. The probability of a segmentation of a word is the product of the probabilities of its pieces, which the model treats as independent. Training repeats three steps until the vocabulary has the target size:

1. With the vocabulary fixed, estimate the piece probabilities with the EM algorithm, treating the segmentation of each sentence as hidden.
2. For each piece, compute the loss: how much the likelihood of the corpus falls if that piece is removed.
3. Keep the pieces with the smallest loss, for example the best 80%, but always keep single characters so that no input becomes unrepresentable.

Encoding picks the most probable segmentation with the Viterbi algorithm, a dynamic program over the end positions of the word. Because the model assigns a probability to every segmentation, it can also *sample* one: subword regularization trains a translation model on segmentations sampled afresh at each update, so the model sees a word split in several ways {cite}`kudo2018subword`. In the listing, "unhappiness" has 12 segmentations; Viterbi returns `un happiness`, and sampling returns `un happi ness` one time in five.

(sec-tdd-byte-level-bpe)=

### Byte-Level BPE

A BPE vocabulary over Unicode characters would need over 130,000 base symbols to cover every string. GPT-2 runs BPE over UTF-8 bytes instead, so the base vocabulary is 256 symbols and every string can be encoded {cite}`radford2019language`. Two details make this work:

- **Pre-tokenization.** Before any merge, a regular expression splits the text into chunks: common English contractions, runs of letters, runs of digits, runs of other symbols, and whitespace. A space stays attached to the start of the following chunk (`" world"`). Merges never cross chunk boundaries, which keeps BPE from learning separate tokens for `dog.`, `dog!` and `dog?` {cite}`radford2019language`.
- **Printable bytes.** The released code maps each of the 256 bytes to a printable Unicode character, so the BPE code only ever handles strings without whitespace or control characters. Printable ASCII and Latin-1 bytes map to themselves; the others, including the space (byte 32), map to characters from U+0100 upward. The space becomes `Ġ` (U+0120), which is why GPT-2 vocabularies are full of tokens like `Ġworld`. A non-ASCII character spans several bytes and so several symbols: `é` is the two bytes `0xC3 0xA9`, shown as `Ã©` until a merge joins them.
(sec-tdd-sentencepiece-library)=

### SentencePiece

SentencePiece {cite}`kudo2018sentencepiece` is a library that trains and runs BPE or unigram models directly on raw text, with no language-specific pre-tokenization. It treats whitespace as an ordinary symbol, writing it as `▁` (U+2581), so `Hello world.` becomes `[Hello] [▁wor] [ld] [.]` and decoding is a matter of joining the pieces and turning `▁` back into a space. Three training options matter most in practice ([option reference](https://github.com/google/sentencepiece/blob/master/doc/options.md)):

- `normalization_rule_name`: the input is Unicode-normalized before training and encoding, by default with a variant of Unicode NFKC normalization (`nmt_nfkc`).
- `character_coverage`: the fraction of the corpus's characters that the base alphabet must cover; rarer characters are dropped from it. The documentation recommends 1.0 for languages with small alphabets and the default 0.9995 for Chinese, Japanese and Korean.
- `byte_fallback`: a character outside the vocabulary is encoded as its UTF-8 bytes, each a token of its own, instead of as `<unk>`. LLaMA's tokenizer uses SentencePiece's BPE with this option and also splits every number into single digits {cite}`touvron2023llama`.

The chapter's {ref}`sec-ch1-sentencepiece` trains such a model with `byte_fallback=True`.

(sec-tdd-which-model-which-tokenizer)=

### Which Model Uses Which

| Model | Tokenizer | Vocabulary | Source |
| ----- | --------- | ---------- | ------ |
| BERT | WordPiece | 30,000 | {cite}`devlin2018bert` |
| GPT-2 | Byte-level BPE | 50,257 | {cite}`radford2019language` |
| T5 | SentencePiece | 32,000 | {cite}`raffel2019exploring` |
| Llama 2 | SentencePiece BPE, digits split, byte fallback | 32k | {cite}`touvron2023llama2` |

T5's paper calls its pieces "wordpieces" but builds them with SentencePiece. Larger vocabularies and the tokenizers of more recent models are in {ref}`sec-tdd-choosing-vocabulary-size`.

(sec-tdd-subword-listing)=

### The Four Schemes in Code

The listing trains one WordPiece step on a toy word-count corpus, encodes words greedily, runs Viterbi and sampling over a toy unigram vocabulary, and shows GPT-2's pre-tokenization and byte characters. The byte map is GPT-2's own `bytes_to_unicode`, rewritten more compactly; the pattern is GPT-2's with its Unicode letter and digit classes narrowed to ASCII, because Python's `re` module has no `\p{L}`.

```python
import math
import random
import re
from collections import Counter

# --- WordPiece training: score = count(ab) / (count(a) * count(b)) ---
word_counts = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}
splits = {w: [w[0]] + ["##" + c for c in w[1:]] for w in word_counts}  # h ##u ##g


def wordpiece_scores(splits):
    unit, pair = Counter(), Counter()
    for word, pieces in splits.items():
        for p in pieces:
            unit[p] += word_counts[word]
        for a, b in zip(pieces, pieces[1:]):
            pair[a, b] += word_counts[word]
    return pair, {ab: n / (unit[ab[0]] * unit[ab[1]]) for ab, n in pair.items()}


pair_counts, scores = wordpiece_scores(splits)
frequent, best = max(pair_counts, key=pair_counts.get), max(scores, key=scores.get)
print(f"BPE would merge:  {frequent} (count {pair_counts[frequent]}, score {scores[frequent]:.4f})")
print(f"WordPiece merges: {best} (count {pair_counts[best]}, score {scores[best]:.4f})")


# --- WordPiece encoding: greedy longest match, continuation pieces start with ## ---
def wordpiece_encode(word, vocab):
    pieces, start = [], 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                pieces.append(piece)
                start = end
                break
        else:
            return ["[UNK]"]  # no piece fits: the whole word is unknown
    return pieces


vocab = {"un", "##happ", "##i", "##ness", "##y", "h", "##a", "##p"}
print("WordPiece:", wordpiece_encode("unhappiness", vocab), wordpiece_encode("unhappy", vocab),
      wordpiece_encode("sad", vocab))

# --- Unigram LM: Viterbi picks the most probable segmentation; sampling gives others ---
probs = {"un": 0.08, "happi": 0.02, "ness": 0.06, "happiness": 0.002, "u": 0.03, "n": 0.05,
         "h": 0.03, "a": 0.06, "p": 0.03, "i": 0.05, "e": 0.08, "s": 0.06, "unh": 0.001}
logp = {piece: math.log(p) for piece, p in probs.items()}


def segmentations(text):
    if not text:
        return [[]]
    return [[text[:k]] + rest for k in range(1, len(text) + 1) if text[:k] in logp
            for rest in segmentations(text[k:])]


def viterbi(text):
    best = [(0.0, [])] + [(-math.inf, None)] * len(text)  # best[i]: best score and pieces for text[:i]
    for end in range(1, len(text) + 1):
        for start in range(end):
            piece = text[start:end]
            if piece in logp and best[start][1] is not None:
                score = best[start][0] + logp[piece]
                if score > best[end][0]:
                    best[end] = (score, best[start][1] + [piece])
    return best[-1]


score, pieces = viterbi("unhappiness")
print(f"Unigram Viterbi: {pieces} (log p = {score:.3f})")
candidates = segmentations("unhappiness")
assert max(sum(logp[p] for p in seg) for seg in candidates) == score  # Viterbi found the best of all
weights = [math.exp(sum(logp[p] for p in seg)) for seg in candidates]
rng = random.Random(0)
print(f"{len(candidates)} segmentations; five samples:",
      [" ".join(rng.choices(candidates, weights)[0]) for _ in range(5)])


# --- Byte-level BPE as in GPT-2: every byte maps to a printable character, so a space shows as G-dot ---
def bytes_to_unicode():
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, chars)))


byte_char = bytes_to_unicode()
# GPT-2's pre-tokenization pattern with \p{L} and \p{N} narrowed to ASCII letters and digits
pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")
chunks = pattern.findall("Hello world, it's 2024!")
print("pre-tokenized:", chunks)
print("as byte characters:", ["".join(byte_char[b] for b in c.encode("utf-8")) for c in chunks])
print("base vocabulary:", len(byte_char), "| é becomes", "".join(byte_char[b] for b in "é".encode("utf-8")))
```

% output

```text
BPE would merge:  ('##u', '##g') (count 20, score 0.0278)
WordPiece merges: ('##g', '##s') (count 5, score 0.0500)
WordPiece: ['un', '##happ', '##i', '##ness'] ['un', '##happ', '##y'] ['[UNK]']
Unigram Viterbi: ['un', 'happiness'] (log p = -8.740)
12 segmentations; five samples: ['un happiness', 'un happiness', 'un happiness', 'un happi ness', 'un happiness']
pre-tokenized: ['Hello', ' world', ',', ' it', "'s", ' 2024', '!']
as byte characters: ['Hello', 'Ġworld', ',', 'Ġit', "'s", 'Ġ2024', '!']
base vocabulary: 256 | é becomes Ã©
```

(sec-tdd-tokenization-impact-model-behavior)=

## T.3 Tokenization Impact on Model Behavior

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

Tokenization can introduce linguistic biases. A tokenizer trained mostly on English text spends more tokens on other languages, so the same content costs more compute and fits less into the context window {cite}`petrov2023language`. Larger vocabularies trained on more languages narrow the gap; the Japanese greeting below takes 10 tokens in GPT-2's encoding, 4 in `cl100k_base` and 2 in `o200k_base`:

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

# With LLMATLAS_WRITE_DATA=1 (run from the chapter's figures/data directory) also save the counts for the figure
if os.environ.get("LLMATLAS_WRITE_DATA") == "1":
    import json

    counts = {lang: {"characters": len(text), **{n: len(encodings[n].encode(text)) for n in names}}
              for lang, text in texts.items()}
    with open("tokens_per_language.json", "w", encoding="utf-8") as f:
        json.dump({"encodings": names, "counts": counts}, f, indent=1, ensure_ascii=False)
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

{numref}`fig-ch1-tokens-per-language` divides each count by the text's length in characters.

```{figure} figures/tokens-per-language.*
:name: fig-ch1-tokens-per-language
:width: 83%
:alt: Grouped bars for English, French, Spanish, Japanese and Arabic. English is 18 tokens per 100 characters in all three encodings; Japanese falls from 143 in gpt2 to 57 in cl100k_base and 29 in o200k_base; Arabic from 92 to 77 to 31.

Tokens per 100 characters for the five greetings above, from the counts the listing prints (saved in `figures/data/tokens_per_language.json`). The greetings are 7 to 16 characters long, so each bar rests on a handful of tokens.
```

(sec-tdd-out-of-vocabulary-handling)=

### Out-of-Vocabulary Handling

Different strategies for unknown tokens:

1. **UNK token**: Replace unknown words with `<UNK>`
1. **Subword fallback**: Break unknown words into subwords
1. **Character fallback**: Fall back to character-level for unknown segments
1. **Byte fallback**: Ultimate fallback to byte representation

(sec-tdd-special-tokens-their-roles)=

## T.4 Special Tokens and Their Roles

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

(sec-tdd-chat-templates)=

### Chat Templates

A model fine-tuned for conversation does not see a list of messages; it sees one string, in which special tokens or markers say where each turn begins and ends and who is speaking. The rule that turns messages into that string is the model's **chat template**. It is fixed by the fine-tuning data, so a prompt in another model's format looks to the model like ordinary text, and as the Hugging Face documentation warns, "different models may use different formats or control tokens, even if they were fine-tuned from the same base model". The `transformers` library stores each model's template with its tokenizer and applies it with `apply_chat_template`. The listing reproduces, by hand, the format of a Mistral instruction-tuned model, which wraps each user turn in `[INST]` and `[/INST]` and ends each assistant turn with the end-of-sequence token `</s>`:

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]


def mistral_template(messages):
    """The format Mistral's instruction-tuned models expect: [INST] ... [/INST] around user turns."""
    text = "<s>"
    for m in messages:
        if m["role"] == "user":
            text += f"[INST] {m['content']} [/INST]"
        else:
            text += f"{m['content']}</s> "
    return text


print(mistral_template(messages))
```

% output

```text
<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
```

The output is the string the chat templating documentation {cite}`huggingface2026chattemplates` shows for this conversation. The model's reply is generated after the final `[/INST]`, and generation stops when the model emits `</s>`.

(sec-tdd-token-ids-embeddings)=

### Token IDs and Embeddings

Each token maps to a unique integer ID. The class below splits on whitespace to keep the mapping visible; a real tokenizer replaces `tokenize` and `detokenize` with one of the algorithms above:

```python
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def encode(self, text):
        tokens = self.tokenize(text)
        return [
            self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens
        ]

    def decode(self, ids):
        tokens = [self.id_to_token[i] for i in ids]
        return self.detokenize(tokens)


tokenizer = Tokenizer(["<UNK>", "the", "cat", "sat"])
ids = tokenizer.encode("the cat sat on the mat")
print(ids)
print(tokenizer.decode(ids))
```

% output

```text
[1, 2, 3, 0, 1, 0]
the cat sat <UNK> the <UNK>
```

Words outside the vocabulary map to the `<UNK>` id, so decoding cannot recover them; the byte-level and byte-fallback tokenizers above avoid this.

(sec-tdd-practical-considerations)=

## T.5 Practical Considerations

(sec-tdd-choosing-vocabulary-size)=

### Choosing Vocabulary Size

Common vocabulary sizes and their trade-offs:

- **Small (1K-8K)**: character-like; used for code-only or single-domain models
- **Medium (32K-50K)**: GPT-2 (50k), the original LLaMA (32k); the usual choice for research-scale models
- **Large (100K-160K)**: GPT-4's `cl100k_base` (~100k), Llama 3 (128k), Qwen (~152k); better compression of multilingual text and code
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

(sec-tdd-unicode-normalization)=

### Unicode Normalization

The same visible text can be stored as different code points. "café" may end in the single code point U+00E9 or in "e" followed by the combining accent U+0301; a tokenizer that sees only code points or bytes would give the two spellings different tokens. **Normalization** maps such variants to one form before tokenization. Unicode defines two kinds. The canonical forms (NFC composes, NFD decomposes) only merge sequences that are the same character, so they lose nothing a reader could see. The compatibility forms (NFKC, NFKD) also fold characters that merely look alike or play the same role: the ligature "ﬁ" becomes "fi", fullwidth letters become ASCII, a superscript 2 becomes a plain 2. The listing applies NFC and NFKC to a few strings, written as escapes so the output stays in ASCII:

```python
import unicodedata

for text in ["caf\u00e9", "cafe\u0301", "\ufb01le", "\uff21\uff22\uff23", "x\u00b2", "\u2460"]:
    nfc = unicodedata.normalize("NFC", text)
    nfkc = unicodedata.normalize("NFKC", text)
    print(f"{ascii(text):<22} {len(text)} code points | NFC {ascii(nfc):<16} | NFKC {ascii(nfkc)}")
```

% output

```text
'caf\xe9'              4 code points | NFC 'caf\xe9'        | NFKC 'caf\xe9'
'cafe\u0301'           5 code points | NFC 'caf\xe9'        | NFKC 'caf\xe9'
'\ufb01le'             3 code points | NFC '\ufb01le'       | NFKC 'file'
'\uff21\uff22\uff23'   3 code points | NFC '\uff21\uff22\uff23' | NFKC 'ABC'
'x\xb2'                2 code points | NFC 'x\xb2'          | NFKC 'x2'
'\u2460'               1 code points | NFC '\u2460'         | NFKC '1'
```

Both spellings of "café" become the same four code points under either form. NFKC goes further, and that is the trade-off: "x²" and "x2" become the same text, so a tokenizer that applies NFKC is not reversible in the sense of the test above. SentencePiece applies an NFKC-based normalization by default ({ref}`sec-tdd-sentencepiece-library`); GPT-2's byte-level BPE works on the bytes as given, so it never merges variants but always decodes back to the exact input ({ref}`sec-tdd-byte-level-bpe`).

(sec-tdd-tokenization-challenges-solutions)=

## T.6 Tokenization Challenges and Solutions

(sec-tdd-cross-lingual-tokenization)=

### Cross-Lingual Tokenization

Different languages present unique tokenization challenges:

**Compounding Languages** (German):

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

(sec-tdd-tokenizer-free)=

### Tokenizer-Free Models

Every problem in this section comes from the tokenizer, so one answer is to drop it. **Tokenizer-free** (or token-free) models read raw UTF-8 bytes, a vocabulary of 256 values plus a few special ids (ByT5 reserves three: padding, end of sequence and an unused unknown token), with no merges to learn and no text that cannot be encoded. ByT5 is such a model: its authors list the benefits that such models "can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines" {cite}`xue2022byt5`. The cost is length. "The main drawback of byte-level models is that byte sequences tend to be significantly longer than token sequences", and attention's cost grows with the square of the length ({ref}`sec-ch1-attention-complexity`). The listing counts bytes per character:

```python
for text in ["Hello world", "caf\u00e9", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627"]:
    n_chars, n_bytes = len(text), len(text.encode("utf-8"))
    print(f"{n_chars:>2} characters -> {n_bytes:>2} UTF-8 bytes ({n_bytes / n_chars:.2f} per character)")
```

% output

```text
11 characters -> 11 UTF-8 bytes (1.00 per character)
 4 characters ->  5 UTF-8 bytes (1.25 per character)
 7 characters -> 21 UTF-8 bytes (3.00 per character)
 5 characters -> 10 UTF-8 bytes (2.00 per character)
```

English costs one byte per character, Arabic two and Japanese three. The Japanese greeting in the third line is 21 bytes, where GPT-2's tokenizer needs 10 tokens and `o200k_base` needs 2 ({ref}`sec-tdd-linguistic-bias`), so a byte-level model processes about ten times as many positions as a model with a large modern vocabulary for the same text. ByT5 shifts capacity to an encoder three times as deep as its decoder and is, in its authors' words, "competitive with a subword-level baseline" {cite}`xue2022byt5`; the widely used LLMs of this chapter still use subword tokenizers.

(sec-tdd-tokenization-quality-practice)=

## T.7 Measuring and Improving Tokenization

The challenges above call for a way to tell a good tokenizer from a poor one. This section gives the measurements and then the practices they support.

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
    plt.figure(figsize=(7, 3.6))
    ranks = range(1, len(frequencies) + 1)
    plt.loglog(ranks, frequencies, marker=".", linestyle="none", label="Observed")
    # Zipf's law: frequency proportional to 1 / rank, a line of slope -1 on log-log axes
    plt.loglog(ranks, [frequencies[0] / r for r in ranks], linestyle="--", color="gray", label="Slope -1 (Zipf)")
    plt.legend(loc="upper right")
    plt.xlabel("Token rank")
    plt.ylabel("Frequency")
    plt.title("Token frequency distribution (Zipf's law)")
    plt.grid(True)
    plt.show()

    return token_counts
```

```{figure} figures/word-frequency.*
:name: fig-ch1-word-frequency
:width: 83%
:alt: Log-log scatter of word frequency against rank: about 440 occurrences for the most frequent word, falling roughly along a straight line to a long tail of words that occur once.

`analyze_token_frequency` applied to the prose of this chapter's main page (its paragraphs, without code and display math), split into lowercase words of letters only; `figures/make_figures.py` runs it. The dashed line has slope −1, Zipf's law anchored at the most frequent word. The counts fall more slowly than the line over the first few hundred ranks and then end in a long tail of words seen once or twice.
```

{numref}`fig-ch1-word-frequency` shows the plot for this chapter's own text.


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

## T.8 Practical Implementation

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


(sec-tdd-embeddings-in-practice)=

## T.9 Embeddings in Practice

This section extends the chapter's {ref}`sec-ch1-embeddings-positional-encoding`: how the embedding width is chosen, how to inspect a trained embedding table, what each positional scheme costs in parameters, and a complete embedding layer.

(sec-tdd-embedding-dimensions)=

### Understanding Embedding Dimensions

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

(sec-tdd-embedding-visualization)=

### Embedding Visualization and Analysis

To see whether related tokens sit close together, project their rows of a trained embedding table to two dimensions:

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

(sec-tdd-position-encoding-trade-offs)=

### Position Encoding Trade-offs

The listing plots the extra parameters each scheme of {ref}`sec-ch1-positional-encodings` adds as the maximum length grows:

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
    for method, params in results.items():
        print(f"{method:<19}", " ".join(f"{p:>9,}" for p in params))

    plt.figure(figsize=(7, 3.6))
    styles = {"Sinusoidal": ("o", "-"), "Learned absolute": ("s", "-"), "Relative bias (T5)": ("^", "-"),
              "RoPE": ("x", "--")}  # sinusoidal and RoPE overlap at zero, so they differ in marker and dash
    for method, params in results.items():
        marker, dash = styles[method]
        plt.plot(seq_lens, [p / 1e6 for p in params], marker=marker, linestyle=dash, label=method)
    plt.xlabel("Maximum sequence length")
    plt.ylabel("Additional parameters (millions)")
    plt.title("Positional encoding parameter cost (sinusoidal and RoPE add none)")
    plt.legend()
    plt.grid(True)
    plt.show()


compare_position_encodings()
```

% output

```text
Sinusoidal                  0         0         0         0         0
Learned absolute       65,536   262,144   524,288 1,048,576 2,097,152
Relative bias (T5)        256       256       256       256       256
RoPE                        0         0         0         0         0
```

```{figure} figures/position-parameter-cost.*
:name: fig-ch1-position-parameter-cost
:width: 83%
:alt: Four lines over maximum lengths 128 to 4096: learned absolute rises to about 2.1 million parameters; relative bias stays at 256; sinusoidal and RoPE stay at 0 and overlap.

The listing's plot, $d = 512$, in millions of parameters. Only learned absolute positions grow with the length. Three lines lie on the axis: T5's relative bias is a constant 256 parameters (32 buckets times 8 heads), too few to leave the axis at this scale, and sinusoidal and RoPE add none.
```

{numref}`fig-ch1-position-parameter-cost` plots these numbers. 
Only learned absolute positions grow with the maximum length, by $d$ parameters per position; T5's bias table is fixed at 32 buckets times 8 heads.

(sec-tdd-advanced-embedding-implementation)=

### Advanced Embedding Implementation

The layer below combines the scaled token embedding of {ref}`sec-ch1-embedding-initialization-strategies` with learned absolute positions, an optional layer normalization and dropout. For sinusoidal positions, add the table of `create_sinusoidal_encoding` from {ref}`sec-ch1-sinusoidal-positions` instead; relative schemes act on the attention scores and are not part of this layer.

```python
import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Enhanced positional encoding with multiple strategies."""

    def __init__(self, d_model, max_seq_length=5000, encoding_type="learned"):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type

        if encoding_type == "learned":
            self.position_embeddings = nn.Embedding(max_seq_length, d_model)

        elif encoding_type == "relative":
            raise ValueError(
                "relative position bias is added to the attention scores, not to the input; "
                "see RelativePositionBias in the chapter"
            )

    def forward(self, x, position_ids=None):
        if self.encoding_type == "learned":
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
        position_encoding="learned",
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
torch.manual_seed(0)
vocab_size = 50000
d_model = 512
max_seq_length = 2048

embedding_layer = TokenEmbeddingLayer(
    vocab_size=vocab_size,
    d_model=d_model,
    max_seq_length=max_seq_length,
    position_encoding="learned",
    layer_norm=True,
    scale_embeddings=True,
).eval()  # eval mode switches dropout off, so the printed norm is reproducible

# Test with sample input
batch_size = 4
seq_length = 256
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

embeddings = embedding_layer(input_ids)
print(f"Input shape: {input_ids.shape}")
print(f"Output shape: {embeddings.shape}")
print(f"Embedding norm: {embeddings.norm(dim=-1).mean():.3f}")
```

% output

```text
Input shape: torch.Size([4, 256])
Output shape: torch.Size([4, 256, 512])
Embedding norm: 22.627
```

The norm is $\sqrt{512} = 22.627$ because the final LayerNorm rescales every vector to zero mean and unit variance across its 512 features (its $\gamma$ starts at 1 and $\beta$ at 0).
