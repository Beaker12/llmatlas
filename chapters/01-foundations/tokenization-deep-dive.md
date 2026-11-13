# Tokenization in Depth

The chapter ({doc}`index`) builds byte pair encoding by hand and then moves on to the transformer. This page stays with tokenizers and covers what the chapter leaves out: the subword algorithms production models use and how real tokenizers split text ({ref}`sec-tdd-subword-algorithms`, {ref}`sec-tdd-modern-tokenization-strategies`), what the choice of tokenizer costs in length and fairness across languages ({ref}`sec-tdd-tokenization-impact-model-behavior`), special tokens and chat templates ({ref}`sec-tdd-special-tokens-their-roles`), vocabulary size, reversibility and Unicode normalization ({ref}`sec-tdd-practical-considerations`), hard inputs such as compounds, numbers and code ({ref}`sec-tdd-tokenization-challenges-solutions`), a complete byte-level BPE tokenizer ({ref}`sec-tdd-implementation-walkthrough`), how to measure a tokenizer ({ref}`sec-tdd-tokenization-quality-practice`) and the embedding layer that turns token ids into vectors ({ref}`sec-tdd-embeddings-in-practice`). It ends with a summary and four exercises. Read the chapter's tokenization section first.

(sec-tdd-learning-objectives)=

## Learning Objectives

After this page you should be able to:

- compute a WordPiece merge score and a unigram segmentation's probability by hand, and say how the two algorithms build their vocabularies;
- explain how byte-level BPE covers every string with 256 base symbols, and why a space shows up as `Ġ` in GPT-2 vocabularies;
- read the token counts of real encodings and say what they cost in sequence length and across languages;
- train a small byte-level BPE tokenizer, check that it decodes its input exactly, and measure its bytes per token;
- count the parameters that a vocabulary and a positional scheme add to a model.

(sec-tdd-subword-algorithms)=

## T.1 Subword Algorithms in Practice

The chapter builds byte pair encoding (BPE) by hand. Production tokenizers use four related schemes: WordPiece, the unigram language model, byte-level BPE and the SentencePiece library, which implements BPE and unigram. They differ in how they choose the vocabulary and in how they split a word once the vocabulary is fixed. The listing at the end of this section does not train full tokenizers. For WordPiece it scores the candidate pairs of a small word list and picks the first merge, then encodes words greedily with a fixed toy vocabulary; for the unigram model it runs Viterbi, sampling and one pruning step over a fixed toy vocabulary with given probabilities; for byte-level BPE it shows only the pre-tokenization and the byte mapping, and learns no merges (a complete byte-level BPE tokenizer is trained in {ref}`sec-tdd-implementation-walkthrough`). The chapter's {ref}`sec-ch1-sentencepiece` trains a real SentencePiece model.

(sec-tdd-wordpiece)=

### WordPiece

WordPiece was introduced by {cite:t}`schuster2012japanese` for Japanese and Korean voice search, was later used in Google's neural translation system {cite}`wu2016google`, and is the tokenizer of BERT {cite}`devlin2018bert`. Training starts, as BPE does, from single characters and merges pairs, but the 2012 paper chooses each new unit "in a greedy way ... to maximize likelihood on the language model training data", not by frequency. The paper describes this rule but gives no exact scoring formula, and Google never released its implementation; the score below is the reconstruction in the Hugging Face course {cite}`huggingface2026wordpiece`, which merges the pair with the highest

$$\operatorname{score}(\text{pair}) = \frac{\operatorname{count}(\text{pair})}{\operatorname{count}(\text{first piece}) \cdot \operatorname{count}(\text{second piece})}.$$

Dividing by the counts of the parts favors pairs whose parts rarely occur apart. In the listing's corpus the pair `##u ##g` occurs 20 times, the most of any pair, but `##u` and `##g` are both common on their own, so its score is $20 / (36 \cdot 20) = 0.0278$; `##g ##s` occurs only 5 times, but `##s` never occurs without `##g` before it, so its score is $5 / (20 \cdot 5) = 0.05$ and WordPiece merges it first.

Pieces that continue a word carry the prefix `##`, so `##ing` and `ing` are different tokens. WordPiece keeps only the final vocabulary, not the merge order, and encodes a word greedily: take the longest vocabulary entry that starts the word, then the longest `##` entry that continues it, and so on. If at some point no entry fits, the whole word becomes `[UNK]`, not just the character that failed.

(sec-tdd-unigram)=

### Unigram Language Model

The unigram model {cite}`kudo2018subword` works in the opposite direction. It starts from a large seed vocabulary (all characters plus the most frequent substrings of the corpus) and shrinks it. The probability of a segmentation of a word is the product of the probabilities of its pieces, which the model treats as independent. Training repeats three steps until the vocabulary has the target size:

1. With the vocabulary fixed, estimate the piece probabilities with the EM algorithm, treating the segmentation of each sentence as hidden.
2. For each piece, compute its loss: how much the likelihood of the corpus falls if that piece is removed.
3. Sort the pieces by loss and keep those with the *largest* loss, for example the top 80%; the pieces whose removal costs least are dropped. Single characters are always kept, so that no input becomes unrepresentable.

In the paper's words, "Sort the symbols by $\text{loss}_i$ and keep top η % of subwords (η is 80, for example)". SentencePiece's `shrinking_factor` option, 0.75 by default, is the fraction of pieces each round keeps, so a round keeps the top 75% of pieces by loss ([option reference](https://github.com/google/sentencepiece/blob/master/doc/options.md)). The listing applies step 3 to a two-word corpus, holding the other probabilities fixed instead of re-estimating them, and asserts that the pieces it keeps all have a larger loss than the one it drops.

Encoding picks the most probable segmentation with the Viterbi algorithm, a dynamic program over the end positions of the word. Because the model assigns a probability to every segmentation, it can also *sample* one: subword regularization trains a translation model on segmentations sampled afresh at each update, so the model sees a word split in several ways {cite}`kudo2018subword`. In the listing, "unhappiness" has 12 segmentations. Viterbi returns `un happiness`, which carries 0.613 of the probability; `un happi ness` carries 0.368, so sampling returns it a little more than one time in three (once in the five samples shown).

(sec-tdd-byte-level-bpe)=

### Byte-Level BPE

A BPE vocabulary over Unicode characters would need over 130,000 base symbols to cover every string. GPT-2 runs BPE over UTF-8 bytes instead, so the base vocabulary is 256 symbols and every string can be encoded {cite}`radford2019language`. Two details make this work:

- **Pre-tokenization.** Before any merge the text is split into chunks, and merges never cross a chunk boundary. The GPT-2 paper states the rule as preventing BPE from "merging across character categories for any byte sequence", with an exception for spaces, which keeps BPE from filling the vocabulary with variants of a word such as `dog.`, `dog!` and `dog?` {cite}`radford2019language`. A regular expression that separates runs of letters, digits, other symbols and whitespace implements such a rule, and the pipeline listing of {ref}`sec-tdd-complete-tokenization-pipeline` uses one in that style. A space stays attached to the start of the following chunk (`" world"`).
- **Printable bytes.** The released code maps each of the 256 bytes to a printable Unicode character, so the BPE code only ever handles strings without whitespace or control characters. The 188 bytes that are already printable characters (`!` to `~`, `¡` to `¬` and `®` to `ÿ`) map to themselves. The other 68, the control characters, the space (byte 32), the no-break space (0xA0) and the soft hyphen (0xAD), map to characters from U+0100 upward. The space becomes `Ġ` (U+0120), which is why GPT-2 vocabularies are full of tokens like `Ġworld`. A non-ASCII character spans several bytes and so several symbols: `é` is the two bytes `0xC3 0xA9`, shown as `Ã©` until a merge joins them.

(sec-tdd-sentencepiece-library)=

### SentencePiece

SentencePiece {cite}`kudo2018sentencepiece` is a library that trains and runs BPE or unigram models directly on raw text, with no language-specific pre-tokenization. It treats whitespace as an ordinary symbol, writing it as `▁` (U+2581), and by default (`add_dummy_prefix`) adds a space before the text, so the first word looks like every other word. In a model whose vocabulary holds `▁Hello` and `▁wor` but not `▁world`, `Hello world.` becomes `[▁Hello] [▁wor] [ld] [.]`, and decoding joins the pieces, turns `▁` back into spaces and drops the leading one. Three training options matter most in practice ([option reference](https://github.com/google/sentencepiece/blob/master/doc/options.md)):

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

### Three Schemes in Code

The listing trains one WordPiece step on a toy word-count corpus, encodes words greedily, runs Viterbi, sampling and one pruning step over a toy unigram vocabulary, and shows GPT-2's pre-tokenization and byte characters. The byte map is GPT-2's own `bytes_to_unicode`, rewritten more compactly; the pattern is GPT-2's with its Unicode letter and digit classes narrowed to ASCII, because Python's `re` module has no `\p{L}`.

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
for label, ab in [("BPE would merge: ", frequent), ("WordPiece merges:", best)]:
    print(f"{label} {ab} (count {pair_counts[ab]}, score {scores[ab]:.4f})")


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
words = ["unhappiness", "unhappy", "sad"]
print("WordPiece:", *(wordpiece_encode(w, vocab) for w in words))

# --- Unigram LM: Viterbi picks the most probable segmentation; sampling gives others
probs = {"un": 0.08, "happi": 0.02, "ness": 0.06, "happiness": 0.002, "unh": 0.001,
         "u": 0.03, "n": 0.05, "h": 0.03, "a": 0.06, "p": 0.03, "i": 0.05, "e": 0.08,
         "s": 0.06}
logp = {piece: math.log(p) for piece, p in probs.items()}


def segmentations(text, vocab=logp):
    if not text:
        return [[]]
    return [[text[:k]] + rest for k in range(1, len(text) + 1) if text[:k] in vocab
            for rest in segmentations(text[k:], vocab)]


def viterbi(text):
    # best[i]: best score and pieces for text[:i]
    best = [(0.0, [])] + [(-math.inf, None)] * len(text)
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
# Viterbi found the best of all
assert max(sum(logp[p] for p in seg) for seg in candidates) == score
weights = [math.exp(sum(logp[p] for p in seg)) for seg in candidates]
share = {" ".join(seg): w / sum(weights) for seg, w in zip(candidates, weights)}
print(f"{len(candidates)} segmentations; "
      f"P(un happiness) = {share['un happiness']:.3f}, "
      f"P(un happi ness) = {share['un happi ness']:.3f}")
rng = random.Random(0)
samples = [" ".join(rng.choices(candidates, weights)[0]) for _ in range(5)]
print("five samples:", ", ".join(samples))

# --- Unigram pruning: a piece's loss is how much the corpus log-likelihood
# falls without it
corpus = {"unhappiness": 5, "happiness": 3}


def log_likelihood(vocab):
    """Sum over words of count * log P(word).

    P(word) is summed over the word's segmentations in `vocab`.
    """
    total = 0.0
    for w, n in corpus.items():
        segs = segmentations(w, vocab)
        total += n * math.log(sum(math.exp(sum(logp[p] for p in seg)) for seg in segs))
    return total


full = log_likelihood(set(logp))
# single characters stay
loss = {x: full - log_likelihood(set(logp) - {x}) for x in logp if len(x) > 1}
ranked = sorted(loss, key=loss.get, reverse=True)
kept, dropped = ranked[: round(0.8 * len(ranked))], ranked[round(0.8 * len(ranked)):]
# keep the largest losses
assert min(loss[x] for x in kept) > max(loss[x] for x in dropped)
print("loss:", ", ".join(f"{x} {loss[x]:.3f}" for x in ranked))
print("keep the top 80%, drop:", dropped)


# --- Byte-level BPE as in GPT-2: every byte maps to a printable character,
# so a space shows as G-dot
def bytes_to_unicode():
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars, extra = printable[:], 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(printable, map(chr, chars)))


byte_char = bytes_to_unicode()
# GPT-2's pre-tokenization pattern with \p{L} and \p{N} narrowed to ASCII letters
# and digits
pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)
chunks = pattern.findall("Hello world, it's 2024!")
print("pre-tokenized:", chunks)
as_chars = ["".join(byte_char[b] for b in c.encode("utf-8")) for c in chunks]
print("as byte characters:", as_chars)
remapped = [b for b, ch in byte_char.items() if ord(ch) != b]
print("base vocabulary:", len(byte_char), "| remapped:", len(remapped), "| é becomes",
      "".join(byte_char[b] for b in "é".encode("utf-8")))
```

% output

```text
BPE would merge:  ('##u', '##g') (count 20, score 0.0278)
WordPiece merges: ('##g', '##s') (count 5, score 0.0500)
WordPiece: ['un', '##happ', '##i', '##ness'] ['un', '##happ', '##y'] ['[UNK]']
Unigram Viterbi: ['un', 'happiness'] (log p = -8.740)
12 segmentations; P(un happiness) = 0.613, P(un happi ness) = 0.368
five samples: un happiness, un happiness, un happiness, un happi ness, un happiness
loss: un 19.976, happiness 7.845, happi 3.761, ness 3.760, unh 0.000
keep the top 80%, drop: ['unh']
pre-tokenized: ['Hello', ' world', ',', ' it', "'s", ' 2024', '!']
as byte characters: ['Hello', 'Ġworld', ',', 'Ġit', "'s", 'Ġ2024', '!']
base vocabulary: 256 | remapped: 68 | é becomes Ã©
```

(sec-tdd-modern-tokenization-strategies)=

## T.2 A Production Tokenizer

(sec-tdd-tiktoken-tokenizer)=

### Tiktoken (OpenAI's Tokenizer)

OpenAI's `tiktoken` library runs the byte-level BPE of {ref}`sec-tdd-byte-level-bpe` with the vocabularies of its models: `gpt2` (50,257 entries), `cl100k_base` and `o200k_base`; the library's `MODEL_TO_ENCODING` table (`tiktoken/model.py`) maps `gpt-4` to `cl100k_base` and `gpt-4o` to `o200k_base`. Each encoding pairs a merge table with its own pre-tokenization pattern and a few special tokens such as `<|endoftext|>`. The listing downloads `cl100k_base` and splits a short string that mixes prose and code. Each `# network` listing on the page carries the same download helper, so that it runs on its own:

```python
# network
import hashlib
import os
import tempfile
import urllib.request

import tiktoken

# tiktoken downloads its encoding files without a User-Agent we can set, so fetch
# them ourselves into the cache directory tiktoken reads (file name: SHA-1 of the URL)
# and send LLMATLAS_USER_AGENT.
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; reader)")
BLOB = "https://openaipublic.blob.core.windows.net/"
FILES = {
    "gpt2": ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"],
    "cl100k_base": ["encodings/cl100k_base.tiktoken"],
    "o200k_base": ["encodings/o200k_base.tiktoken"],
}
CACHE = os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tiktoken-cache")
)


def get_encoding(name):
    os.makedirs(CACHE, exist_ok=True)
    for path in FILES[name]:
        target = os.path.join(CACHE, hashlib.sha1((BLOB + path).encode()).hexdigest())
        if not os.path.exists(target):
            headers = {"User-Agent": USER_AGENT}
            request = urllib.request.Request(BLOB + path, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(target, "wb") as f:
                f.write(data)
    return tiktoken.get_encoding(name)


import pprint

enc = get_encoding("cl100k_base")  # the GPT-4 encoding

text = "Hello, world! Here's some code: `print('hi')`"
tokens = enc.encode(text)
assert enc.decode(tokens) == text  # byte-level BPE is lossless
print(f"Token count: {len(tokens)}")
print("Pieces:")
pprint.pp([enc.decode([i]) for i in tokens], width=88, compact=True)
```

% output

```text
Token count: 15
Pieces:
['Hello', ',', ' world', '!', ' Here', "'s", ' some', ' code', ':', ' `', 'print', "('",
 'hi', "')", '`']
```

(sec-tdd-tokenization-impact-model-behavior)=

## T.3 Tokenization Impact on Model Behavior

(sec-tdd-sequence-length-cost)=

### Length and Efficiency

Different tokenization schemes produce different sequence lengths:

```python
import pprint

# Example text
text = "The transformer architecture revolutionized natural language processing."

# Word-level: split on spaces
word_tokens = text.lower().split()
print(f"Word tokens ({len(word_tokens)}):")
pprint.pp(word_tokens, width=88, compact=True)

# Character-level
char_tokens = list(text)
print(f"Character tokens ({len(char_tokens)}), the first 20:")
pprint.pp(char_tokens[:20], width=88, compact=True)

# Subword: the pieces that tiktoken's gpt2, cl100k_base and o200k_base encodings all
# give for this sentence (the network listing under "Real Tokenizers on Hard Inputs"
# prints the count 9 for each)
subword_tokens = [
    "The", " transformer", " architecture", " revolution", "ized",
    " natural", " language", " processing", ".",
]
assert "".join(subword_tokens) == text
print(f"Subword tokens ({len(subword_tokens)}):")
pprint.pp(subword_tokens, width=88, compact=True)
```

% output

```text
Word tokens (7):
['the', 'transformer', 'architecture', 'revolutionized', 'natural', 'language',
 'processing.']
Character tokens (72), the first 20:
['T', 'h', 'e', ' ', 't', 'r', 'a', 'n', 's', 'f', 'o', 'r', 'm', 'e', 'r', ' ', 'a',
 'r', 'c', 'h']
Subword tokens (9):
['The', ' transformer', ' architecture', ' revolution', 'ized', ' natural', ' language',
 ' processing', '.']
```

Self-attention compares every position with every other, so its score matrix has $n^2$ entries for $n$ tokens: $72^2 = 5{,}184$ for the characters, $7^2 = 49$ for the words and $9^2 = 81$ for the subword tokens ({ref}`sec-ch1-attention-complexity`). The character sequence costs 64 times as many attention scores as the subword one, and every per-token cost of the model (the feed-forward layers, the stored keys and values) grows eightfold with its length. Words give the shortest sequence, but a word-level vocabulary needs an entry for every word form and still meets words it has never seen; subwords sit between the two.

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

# tiktoken downloads its encoding files without a User-Agent we can set, so fetch
# them ourselves into the cache directory tiktoken reads (file name: SHA-1 of the URL)
# and send LLMATLAS_USER_AGENT.
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; reader)")
BLOB = "https://openaipublic.blob.core.windows.net/"
FILES = {
    "gpt2": ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"],
    "cl100k_base": ["encodings/cl100k_base.tiktoken"],
    "o200k_base": ["encodings/o200k_base.tiktoken"],
}
CACHE = os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tiktoken-cache")
)


def get_encoding(name):
    os.makedirs(CACHE, exist_ok=True)
    for path in FILES[name]:
        target = os.path.join(CACHE, hashlib.sha1((BLOB + path).encode()).hexdigest())
        if not os.path.exists(target):
            headers = {"User-Agent": USER_AGENT}
            request = urllib.request.Request(BLOB + path, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(target, "wb") as f:
                f.write(data)
    return tiktoken.get_encoding(name)


# The same greeting in five languages, counted by the encodings of GPT-2, GPT-4
# and GPT-4o
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
    row = "".join(f"{len(encodings[n].encode(text)):>13}" for n in names)
    print(f"{language:<10}" + row)

# With LLMATLAS_WRITE_DATA=1 (run from the chapter's figures/data directory)
# also save the counts for the figure
if os.environ.get("LLMATLAS_WRITE_DATA") == "1":
    import json

    counts = {
        lang: {
            "characters": len(text),
            **{n: len(encodings[n].encode(text)) for n in names},
        }
        for lang, text in texts.items()
    }
    with open("tokens_per_language.json", "w", encoding="utf-8") as f:
        data = {"encodings": names, "counts": counts}
        json.dump(data, f, indent=1, ensure_ascii=False)
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
:alt: Grouped bars for English, French, Spanish, Japanese and Arabic. English is 18
  tokens per 100 characters in all three encodings; Japanese falls from 143 in gpt2 to
  57 in cl100k_base and 29 in o200k_base; Arabic from 92 to 77 to 31.

Tokens per 100 characters for the five greetings above, from the counts the listing
prints (saved in `figures/data/tokens_per_language.json`). The greetings are 7 to 16
characters long, so each bar rests on a handful of tokens.
```

(sec-tdd-out-of-vocabulary-handling)=

### Out-of-Vocabulary Handling

A tokenizer meets text its vocabulary does not cover in one of four ways, and each algorithm above picks one:

1. **Unknown token.** A word-level vocabulary maps every unseen word to `<UNK>` (the toy tokenizer in {ref}`sec-tdd-token-ids-embeddings`), and WordPiece maps a word to `[UNK]` when no sequence of its pieces covers it ({ref}`sec-tdd-wordpiece`). The text is lost.
1. **Subword fallback.** BPE and unigram split an unseen word into known pieces, down to single characters if necessary; the word survives as long as its characters are in the base vocabulary.
1. **Byte fallback.** SentencePiece with `byte_fallback` spells a character outside the vocabulary as its UTF-8 bytes ({ref}`sec-tdd-sentencepiece-library`).
1. **Bytes as the base.** Byte-level BPE starts from all 256 byte values, so no input is unknown at all ({ref}`sec-tdd-byte-level-bpe`).

(sec-tdd-special-tokens-their-roles)=

## T.4 Special Tokens and Their Roles

(sec-tdd-common-special-tokens)=

### Common Special Tokens

Special tokens are vocabulary entries that never come from splitting text: the tokenizer or the training pipeline inserts them. Each family of models has its own set:

| Token | Role | Used by |
| ----- | ---- | ------- |
| `<s>`, `</s>` (generically `<BOS>`, `<EOS>`) | start and end of a sequence; generation stops at the end token | Mistral ({ref}`sec-tdd-chat-templates`) |
| `<\|endoftext\|>` | boundary between documents packed into one training sequence | GPT-2's `gpt2` encoding |
| `[CLS]` | first token of the input; its final vector represents the whole input for classification | BERT {cite}`devlin2018bert` |
| `[SEP]` | separator between the two segments of an input pair | BERT |
| `[MASK]` | replaces each token a masked language model must predict | BERT |
| `[PAD]`, `<PAD>` | fills short sequences up to the batch length; masked out of attention and of the loss | batched training |
| `[UNK]`, `<UNK>` | stands for input the vocabulary cannot represent | WordPiece and word-level vocabularies |
| `<\|im_start\|>`, `<\|im_end\|>` | start and end of one chat message, with the speaker's role after the start token | the ChatML chat format of OpenAI's chat models {cite}`openai2023chatml` |

The byte-level tokenizer of {ref}`sec-tdd-implementation-walkthrough` reserves ids 256 to 260 for `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` and `<MASK>`.

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
    """The format Mistral's instruction-tuned models expect.

    [INST] ... [/INST] goes around user turns.
    """
    text = "<s>"
    for m in messages:
        if m["role"] == "user":
            text += f"[INST] {m['content']} [/INST]"
        else:
            text += f"{m['content']}</s> "
    return text


# one line per turn for display: a line break after each [/INST] and each </s>
formatted = mistral_template(messages)
print(formatted.replace("[/INST]", "[/INST]\n").replace("</s>", "</s>\n").rstrip())
```

% output

```text
<s>[INST] Hello, how are you? [/INST]
I'm doing great. How can I help you today?</s>
 [INST] I'd like to show off how chat templating works! [/INST]
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

A larger vocabulary shortens every sequence, most of all for the languages and scripts a smaller one handles poorly ({ref}`sec-tdd-linguistic-bias`), but each entry adds a row of $d$ parameters to the embedding matrix and, unless the two are tied, to the output projection, and widens the softmax at every position. Vocabularies have grown over time:

| Tokenizer | Entries | Source |
| --------- | ------- | ------ |
| LLaMA, Llama 2 | 32k | {cite}`touvron2023llama`, {cite}`touvron2023llama2` |
| `gpt2` (GPT-2) | 50,257 | `n_vocab`, printed in {ref}`sec-tdd-hard-inputs` |
| `cl100k_base` (GPT-4) | 100,277 | `n_vocab`, printed in {ref}`sec-tdd-hard-inputs` |
| Llama 3 | 128K: 100K tokens from `tiktoken` plus 28K for other languages | {cite}`grattafiori2024llama` |
| `o200k_base` (GPT-4o) | 200,019 | `n_vocab`, printed in {ref}`sec-tdd-hard-inputs` |
| Gemma 2 | 256k, inherited from Gemini | {cite}`gemma2024gemma2` |

The cost shows in Gemma 2's parameter table: its 27B model spends 1,180,237,824 parameters on the embedding against 26,047,480,320 on everything else, about 4.3% of the total {cite}`gemma2024gemma2`. {ref}`sec-tdd-exercise-4` works out the cost for a model of width 4,096.

(sec-tdd-reversibility-information-loss)=

### Reversibility and Information Loss

A tokenizer is *reversible* when decoding the ids of any text gives back exactly that text. Byte-level tokenizers are reversible by construction. A tokenizer that changes the text before splitting it (lowercasing, collapsing whitespace, Unicode normalization) is not, and the change stays invisible until a user's text comes back different. The check is a round trip over a varied sample, which the listing in the next subsection runs.

(sec-tdd-unicode-normalization)=

### Unicode Normalization

The same visible text can be stored as different code points. "café" may end in the single code point U+00E9 or in "e" followed by the combining accent U+0301; a tokenizer that sees only code points or bytes would give the two spellings different tokens. **Normalization** maps such variants to one form before tokenization. Unicode defines two kinds. The canonical forms (NFC composes, NFD decomposes) only merge sequences that stand for the same character. The compatibility forms (NFKC, NFKD) also replace a character by a compatibility equivalent that plays the same role in a different shape: the ligature "ﬁ" becomes "fi", fullwidth letters become ASCII, a superscript 2 becomes a plain 2. They do not merge characters that merely look alike: the Cyrillic "а" (U+0430) stays distinct from the Latin "a". The listing normalizes a few strings, written as escapes so the output stays in ASCII, and then runs the round-trip test on three tokenizers: UTF-8 bytes, and code points after NFC or after NFKC.

```python
import unicodedata

texts = [
    "caf\u00e9", "cafe\u0301", "\ufb01le", "\uff21\uff22\uff23",
    "x\u00b2", "\u2460", "\u0430"
]
for text in texts:
    nfc, nfkc = (unicodedata.normalize(form, text) for form in ("NFC", "NFKC"))
    print(f"{ascii(text):<22} length {len(text)} | NFC {ascii(nfc):<16} "
          f"| NFKC {ascii(nfkc)}")


def test_reversibility(encode, decode, texts):
    """The texts that do not come back unchanged from decode(encode(text))."""
    return [text for text in texts if decode(encode(text)) != text]


def code_points(form):
    return lambda text: [ord(c) for c in unicodedata.normalize(form, text)]


def from_code_points(ids):
    return "".join(map(chr, ids))


tokenizers = {
    "UTF-8 bytes": (
        lambda text: list(text.encode("utf-8")),
        lambda ids: bytes(ids).decode("utf-8"),
    ),
    "NFC code points": (code_points("NFC"), from_code_points),
    "NFKC code points": (code_points("NFKC"), from_code_points),
}
for name, (encode, decode) in tokenizers.items():
    changed = test_reversibility(encode, decode, texts)
    print(f"{name:<17} {len(changed)} of {len(texts)} texts changed:")
    if changed:
        print("    " + " ".join(ascii(t) for t in changed))
```

% output

```text
'caf\xe9'              length 4 | NFC 'caf\xe9'        | NFKC 'caf\xe9'
'cafe\u0301'           length 5 | NFC 'caf\xe9'        | NFKC 'caf\xe9'
'\ufb01le'             length 3 | NFC '\ufb01le'       | NFKC 'file'
'\uff21\uff22\uff23'   length 3 | NFC '\uff21\uff22\uff23' | NFKC 'ABC'
'x\xb2'                length 2 | NFC 'x\xb2'          | NFKC 'x2'
'\u2460'               length 1 | NFC '\u2460'         | NFKC '1'
'\u0430'               length 1 | NFC '\u0430'         | NFKC '\u0430'
UTF-8 bytes       0 of 7 texts changed:
NFC code points   1 of 7 texts changed:
    'cafe\u0301'
NFKC code points  5 of 7 texts changed:
    'cafe\u0301' '\ufb01le' '\uff21\uff22\uff23' 'x\xb2' '\u2460'
```

Both spellings of "café" become the same four code points under either form, so even NFC breaks a byte-exact round trip: the decomposed spelling comes back composed. NFKC changes five of the seven strings, and some of what it removes is visible to a reader: "x²" and "x2" become the same text. SentencePiece applies an NFKC-based normalization by default ({ref}`sec-tdd-sentencepiece-library`); GPT-2's byte-level BPE works on the bytes as given, so it never merges variants but always decodes back to the exact input ({ref}`sec-tdd-byte-level-bpe`).

(sec-tdd-tokenization-challenges-solutions)=

## T.6 Tokenization Challenges and Solutions

(sec-tdd-hard-inputs)=

### Real Tokenizers on Hard Inputs

The subsections below discuss inputs that strain a tokenizer: compounds, words built from many suffixes, text without spaces, code, URLs, numbers and specialist vocabulary. The listing encodes one example of each with the three OpenAI encodings of {ref}`sec-tdd-linguistic-bias`, prints each vocabulary's size and the token counts, and shows the pieces of the examples discussed in most detail. A `�` marks a token that holds only part of a character's UTF-8 bytes.

```python
# network
import hashlib
import os
import tempfile
import urllib.request

import tiktoken

# tiktoken downloads its encoding files without a User-Agent we can set, so fetch
# them ourselves into the cache directory tiktoken reads (file name: SHA-1 of the URL)
# and send LLMATLAS_USER_AGENT.
USER_AGENT = os.environ.get("LLMATLAS_USER_AGENT", "Mozilla/5.0 (compatible; reader)")
BLOB = "https://openaipublic.blob.core.windows.net/"
FILES = {
    "gpt2": ["gpt-2/encodings/main/vocab.bpe", "gpt-2/encodings/main/encoder.json"],
    "cl100k_base": ["encodings/cl100k_base.tiktoken"],
    "o200k_base": ["encodings/o200k_base.tiktoken"],
}
CACHE = os.environ.setdefault(
    "TIKTOKEN_CACHE_DIR", os.path.join(tempfile.gettempdir(), "tiktoken-cache")
)


def get_encoding(name):
    os.makedirs(CACHE, exist_ok=True)
    for path in FILES[name]:
        target = os.path.join(CACHE, hashlib.sha1((BLOB + path).encode()).hexdigest())
        if not os.path.exists(target):
            headers = {"User-Agent": USER_AGENT}
            request = urllib.request.Request(BLOB + path, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(target, "wb") as f:
                f.write(data)
    return tiktoken.get_encoding(name)


names = ["gpt2", "cl100k_base", "o200k_base"]
encodings = {name: get_encoding(name) for name in names}
samples = {
    "German": "Donaudampfschifffahrtsgesellschaftskapitän",
    "Turkish": "evlerimizde",
    "Chinese": "我喜欢学习自然语言处理",
    "English": (
        "The transformer architecture revolutionized natural language processing."
    ),
    "Medical": (
        "Patient presents with myocardial infarction and diabetes mellitus type 2"
    ),
    "Legal": "The aforementioned clause notwithstanding the heretofore provisions",
    "URL": "https://www.example.com/path/to/page?param=value",
    "Code": (
        "def fibonacci(n):\n    if n <= 1:\n        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)\n"
    ),
    "Digits": "12345678",
    "Decimal": "1,234,567.89",
    "ISO date": "2023-12-25",
}
print(f"{'vocabulary':<10}" + "".join(f"{encodings[n].n_vocab:>13,}" for n in names))
print(f"{'tokens':<10}" + "".join(f"{n:>13}" for n in names))
for label, text in samples.items():
    row = "".join(f"{len(encodings[n].encode(text)):>13}" for n in names)
    print(f"{label:<10}" + row)
shown_labels = [
    "German", "Turkish", "Chinese", "Legal", "URL",
    "Code", "Digits", "Decimal", "ISO date"
]
for label in shown_labels:
    for n in names:
        pieces = [encodings[n].decode([i]) for i in encodings[n].encode(samples[label])]
        # show a space as · and a newline as \n so that whitespace tokens can be seen
        # and counted
        shown = [p.replace(" ", "·").replace("\n", "\\n") for p in pieces]
        print(f"{label} / {n}:")
        # 16 pieces per line, so that long rows fit a printed page
        for j in range(0, len(shown), 16):
            print("    " + "|".join(shown[j : j + 16]))
```

% output

```text
vocabulary       50,257      100,277      200,019
tokens             gpt2  cl100k_base   o200k_base
German               19           17           13
Turkish               5            5            4
Chinese              26           13            6
English               9            9            9
Medical              16           15           12
Legal                 9           10           10
URL                  17           11           11
Code                 49           28           28
Digits                3            3            3
Decimal               8            7            7
ISO date              6            6            6
German / gpt2:
    Don|aud|amp|fs|ch|if|ff|ah|r|ts|ges|ells|cha|ft|sk|ap
    it|ä|n
German / cl100k_base:
    D|ona|ud|amp|fs|ch|if|ff|ah|rt|sg|es|ellschaft|sk|ap|it
    än
German / o200k_base:
    D|ona|ud|amp|fsch|if|ff|ahr|ts|gesellschaft|skap|it|än
Turkish / gpt2:
    ev|ler|im|iz|de
Turkish / cl100k_base:
    ev|ler|im|iz|de
Turkish / o200k_base:
    ev|ler|imiz|de
Chinese / gpt2:
    �|�|�|�|�|�|�|�|�|�|�|�|�|�|�|�
    �|�|�|�|�|�|�|�|�|�
Chinese / cl100k_base:
    我|�|�|�|�|学|�|�|自|然|语|言|处理
Chinese / o200k_base:
    我|喜欢|学习|自然|语言|处理
Legal / gpt2:
    The|·aforementioned|·clause|·notwithstanding|·the|·here|to|fore|·provisions
Legal / cl100k_base:
    The|·aforementioned|·clause|·notwithstanding|·the|·her|et|of|ore|·provisions
Legal / o200k_base:
    The|·aforementioned|·clause|·notwithstanding|·the|·her|et|of|ore|·provisions
URL / gpt2:
    https|://|www|.|example|.|com|/|path|/|to|/|page|?|param|=
    value
URL / cl100k_base:
    https|://|www|.example|.com|/path|/to|/page|?|param|=value
URL / o200k_base:
    https|://|www|.example|.com|/path|/to|/page|?|param|=value
Code / gpt2:
    def|·fib|on|acci|(|n|):|\n|·|·|·|·if|·n|·<=|·1|:
    \n|·|·|·|·|·|·|·|·return|·n|\n|·|·|·|·return|·fib
    on|acci|(|n|-|1|)|·+|·fib|on|acci|(|n|-|2|)
    \n
Code / cl100k_base:
    def|·fibonacci|(n|):\n|···|·if|·n|·<=|·|1|:\n|·······|·return|·n|\n|···
    ·return|·fibonacci|(n|-|1|)|·+|·fibonacci|(n|-|2|)\n
Code / o200k_base:
    def|·fibonacci|(n|):\n|···|·if|·n|·<=|·|1|:\n|·······|·return|·n|\n|···
    ·return|·fibonacci|(n|-|1|)|·+|·fibonacci|(n|-|2|)\n
Digits / gpt2:
    123|45|678
Digits / cl100k_base:
    123|456|78
Digits / o200k_base:
    123|456|78
Decimal / gpt2:
    1|,|234|,|5|67|.|89
Decimal / cl100k_base:
    1|,|234|,|567|.|89
Decimal / o200k_base:
    1|,|234|,|567|.|89
ISO date / gpt2:
    20|23|-|12|-|25
ISO date / cl100k_base:
    202|3|-|12|-|25
ISO date / o200k_base:
    202|3|-|12|-|25
```

(sec-tdd-cross-lingual-tokenization)=

### Cross-Lingual Tokenization

**Compounding languages.** German writes compounds as one word. *Donaudampfschifffahrtsgesellschaftskapitän* ("Danube steamship company captain") is built from Donau, Dampf, Schiff, Fahrt, Gesellschaft and Kapitän, but no encoding finds those parts: `o200k_base` needs 13 tokens, of which only `gesellschaft` is a whole part, and `gpt2` needs 19. BPE merges what was frequent in its training data, not what is meaningful.

**Agglutinative languages.** Turkish *evlerimizde* ("in our houses") is ev (house) + ler (plural) + imiz (our) + de (in). `o200k_base` splits it exactly so, `ev|ler|imiz|de`; the two older encodings split *imiz* into `im|iz`.

**Languages without spaces.** Chinese writes no spaces between words. In *我喜欢学习自然语言处理* ("I like learning natural language processing") `o200k_base` finds the six words 我, 喜欢, 学习, 自然, 语言 and 处理. `cl100k_base` needs 13 tokens, several of them fragments of a character's three UTF-8 bytes, and `gpt2` needs 26 for 11 characters, none of them a whole character.

(sec-tdd-handling-special-cases)=

### Handling Special Cases

**Code.** Indentation is syntax in Python. The listing shows a space as `·`. In `gpt2` every indenting space but the last is a token of its own (the eight spaces before `return n` are seven `·` tokens and `·return`), the name `fibonacci` is three tokens, `·fib|on|acci`, and each line break is a separate `\n`; the four-line `fibonacci` function costs 49 tokens. `cl100k_base` and `o200k_base` give the same 28 pieces: a run of spaces is one token (`···` before `·if`, `·······` before `·return`), `·fibonacci` is one token, and a line break joins the punctuation before it (`):\n`).

**URLs and email addresses.** `gpt2` splits `https://www.example.com/path/to/page?param=value` at every punctuation mark, 17 tokens. The newer encodings keep `https`, `://`, `www` and `?` apart but attach a dot, slash or equals sign to the word that follows it (`.example`, `/path`, `=value`) and need 11.

**Numbers and dates.** How a model sees digits decides how hard arithmetic is for it. GPT-2 has no rule for digits: runs of digits are merged like any other text, so `12345678` becomes `123|45|678` and the year in `2023-12-25` becomes `20|23`. `cl100k_base` and `o200k_base` cut a run of digits into groups of at most three from the left, `123|456|78`, so the same year becomes `202|3`, and `1,234,567.89` keeps each comma group as one token, where `gpt2` splits 567 into `5|67`. LLaMA goes further and splits every number into single digits {cite}`touvron2023llama`, so that a number is always spelled the same way.

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

# GPT-2's encoding (tiktoken "gpt2") splits the medical terms;
# the network listing above prints 16 tokens
# for gpt2, 15 for cl100k_base and 12 for o200k_base
generic_tokens = [
    "Pat", "ient", " presents", " with", " my", "ocard", "ial", " inf", "ar", "ction",
    " and", " diabetes", " mell", "itus", " type", " 2",
]
assert "".join(generic_tokens) == "".join(medical_tokens) == medical_text
print(f"Domain vocabulary: {len(medical_tokens)} tokens; "
      f"gpt2: {len(generic_tokens)} tokens")
```

% output

```text
Domain vocabulary: 10 tokens; gpt2: 16 tokens
```

**Legal Text**: in the listing, `gpt2` splits *heretofore* as `here|to|fore`, its three parts, while both newer encodings give `her|et|of|ore`; every other word of the sentence is one token in all three encodings, so the legal sentence of the listing costs 9 tokens in `gpt2` and 10 in the others. A larger vocabulary spends its extra entries on frequent strings, so it does not guarantee better splits of rare words.

(sec-tdd-tokenizer-free)=

### Tokenizer-Free Models

Every problem in this section comes from the tokenizer, so one answer is to drop it. **Tokenizer-free** (or token-free) models read raw UTF-8 bytes, a vocabulary of 256 values plus a few special ids (ByT5 reserves three: padding, end of sequence and an unused unknown token), with no merges to learn and no text that cannot be encoded. ByT5 is such a model: its authors list the benefits that such models "can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines" {cite}`xue2022byt5`. The cost is length. "The main drawback of byte-level models is that byte sequences tend to be significantly longer than token sequences", and attention's cost grows with the square of the length ({ref}`sec-ch1-attention-complexity`). The listing counts bytes per character:

```python
samples = [
    "Hello world",
    "caf\u00e9",
    "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c",
    "\u0645\u0631\u062d\u0628\u0627",
]
for text in samples:
    n_chars, n_bytes = len(text), len(text.encode("utf-8"))
    per_char = n_bytes / n_chars
    print(f"{n_chars:>2} characters -> {n_bytes:>2} UTF-8 bytes "
          f"({per_char:.2f} per character)")
```

% output

```text
11 characters -> 11 UTF-8 bytes (1.00 per character)
 4 characters ->  5 UTF-8 bytes (1.25 per character)
 7 characters -> 21 UTF-8 bytes (3.00 per character)
 5 characters -> 10 UTF-8 bytes (2.00 per character)
```

English costs one byte per character, Arabic two and Japanese three. The Japanese greeting in the third line is 21 bytes, where GPT-2's tokenizer needs 10 tokens and `o200k_base` needs 2 ({ref}`sec-tdd-linguistic-bias`), so a byte-level model processes about ten times as many positions as a model with a large modern vocabulary for the same text. ByT5 shifts capacity to an encoder three times as deep as its decoder and is, in its authors' words, "competitive with a subword-level baseline" {cite}`xue2022byt5`; the widely used LLMs of this chapter still use subword tokenizers.

(sec-tdd-implementation-walkthrough)=

## T.7 Practical Implementation

This section implements a tokenizer end to end. It is byte-level BPE, the scheme GPT-2 introduced: the base vocabulary is the 256 byte values, so any UTF-8 text can be encoded without an unknown token; whitespace stays inside the tokens (a token can start with a space, and runs of spaces or newlines are tokens of their own), so decoding returns the input exactly; and every chunk of the training text counts, however rare. With only 39 merges on a five-sentence corpus, frequent words such as `The` become one token while an unseen word falls back to short pieces.

(sec-tdd-complete-tokenization-pipeline)=

### Complete Tokenization Pipeline

```python
import pprint
import re
from collections import Counter
from typing import Dict, List, Tuple

# Pre-tokenization in the style of GPT-2: a chunk is an optional leading space
# followed by letters, digits or punctuation, or a run of whitespace. Joining the
# chunks gives back the text exactly, so whitespace is kept inside the tokens
# instead of being thrown away.
CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")


class ComprehensiveTokenizer:
    """Byte-level BPE: 256 byte tokens, five special tokens, then learned merges.

    Every text round-trips.
    """

    def __init__(self, vocab_size: int = 300):
        assert vocab_size >= 256 + 5
        self.vocab_size = vocab_size
        # id -> the bytes it stands for
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        names = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"]
        # ids 256..260
        self.special_tokens = {name: 256 + i for i, name in enumerate(names)}
        # (left id, right id) -> merged id, in training order
        self.merges: Dict[Tuple[int, int], int] = {}

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

    def train_bpe(self, corpus: List[str], verbose: bool = True) -> None:
        """Learn vocab_size - 261 merges.

        Every chunk is counted; nothing is dropped for being rare.
        """
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
            # ties go to the smaller ids
            best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
            self.merges[best] = next_id
            self.vocab[next_id] = self.vocab[best[0]] + self.vocab[best[1]]
            words = {
                chunk: self._merge(ids, best, next_id) for chunk, ids in words.items()
            }
            next_id += 1
        if verbose:
            print(f"Training complete. {len(self.merges)} merges, "
                  f"vocabulary size {len(self.vocab) + 5}")

    def encode_chunk(self, chunk: str) -> List[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # apply the earliest-learned merge present, as the trainer did
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.merges.get(p, float("inf")))
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
        # special ids are not in vocab
        data = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return data.decode("utf-8", errors="replace")

    def pieces(self, text: str) -> List[str]:
        """The tokens of `text` as strings, for display (special tokens left out)."""
        ids = self.encode(text)[1:-1]
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]


sample_corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question",
    "The only thing we have to fear is fear itself",
    "Ask not what your country can do for you",
] * 100

# 39 merges: small enough that words still split
tokenizer = ComprehensiveTokenizer(vocab_size=300)
tokenizer.train_bpe(sample_corpus)

checks = ["The quick brown fox.", "Hello, world!", "two  spaces\nand a newline",
          "unthinkable", "你好"]
for text in checks:
    # exact round trip, including whitespace
    assert tokenizer.decode(tokenizer.encode(text)) == text, text
# the last is unseen
for text in ["The quick brown fox.", "two  spaces\nand a newline", "unthinkable"]:
    print(f"{text!r} ->")
    pprint.pp(tokenizer.pieces(text), width=88, compact=True)


# --- Measuring the tokenizer on text it was not trained on ---
def evaluate_tokenization_quality(tokenizer, corpus):
    """Characters and bytes per token (special ids not counted).

    Also the share of texts that do not round-trip.
    """
    special = set(tokenizer.special_tokens.values())
    n_chars = n_bytes = n_tokens = errors = 0
    for text in corpus:
        ids = tokenizer.encode(text)
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))
        n_tokens += sum(1 for i in ids if i not in special)
        errors += tokenizer.decode(ids) != text
    return {
        "chars_per_token": n_chars / n_tokens,
        "bytes_per_token": n_bytes / n_tokens,
        "reversibility_errors": errors / len(corpus),
    }


def analyze_tokenization_efficiency(tokenizer, corpus):
    """Token lengths in bytes, and how many tokens are of each kind.

    The kinds are single bytes, whole pre-tokenized chunks or parts of one.
    """
    special = set(tokenizer.special_tokens.values())
    lengths, kinds = Counter(), Counter()
    for text in corpus:
        chunks = set(CHUNK.findall(text))
        for i in tokenizer.encode(text):
            if i not in special:
                piece = tokenizer.vocab[i]
                lengths[len(piece)] += 1
                whole = piece.decode("utf-8", "replace") in chunks
                if len(piece) == 1:
                    kinds["single byte"] += 1
                else:
                    kinds["whole chunk" if whole else "part of a chunk"] += 1
    return dict(sorted(lengths.items())), dict(kinds)


held_out = ["The question is what the fox can do for the dog", "Ask a thousand times"]
quality = evaluate_tokenization_quality(tokenizer, held_out)
print("held out:")
pprint.pp({k: round(v, 3) for k, v in quality.items()}, width=88)
lengths, kinds = analyze_tokenization_efficiency(tokenizer, held_out)
print("token lengths in bytes:", lengths)
print("kinds:", kinds)
for size in (261, 300, 400):
    sized = ComprehensiveTokenizer(vocab_size=size)
    sized.train_bpe(sample_corpus, verbose=False)
    metrics = evaluate_tokenization_quality(sized, held_out)
    print(f"vocabulary {size}: {len(sized.merges):>3} merges, "
          f"{metrics['bytes_per_token']:.3f} bytes per token held out")
```

% output

```text
Training complete. 39 merges, vocabulary size 300
'The quick brown fox.' ->
['The', ' qu', 'i', 'c', 'k', ' b', 'r', 'o', 'w', 'n', ' fo', 'x', '.']
'two  spaces\nand a newline' ->
['t', 'w', 'o', ' ', ' s', 'p', 'a', 'c', 'es', '\n', 'an', 'd', ' a', ' n', 'e', 'w',
 'l', 'in', 'e']
'unthinkable' ->
['u', 'n', 't', 'h', 'in', 'k', 'a', 'b', 'l', 'e']
held out:
{'chars_per_token': 1.914, 'bytes_per_token': 1.914, 'reversibility_errors': 0.0}
token lengths in bytes: {1: 15, 2: 10, 3: 8, 4: 2}
kinds: {'whole chunk': 7, 'part of a chunk': 13, 'single byte': 15}
vocabulary 261:   0 merges, 1.000 bytes per token held out
vocabulary 300:  39 merges, 1.914 bytes per token held out
vocabulary 400: 115 merges, 3.722 bytes per token held out
```

(sec-tdd-tokenization-analysis-tools)=

### Tokenization Analysis Tools

The last block of the listing measures the trained tokenizer on two sentences it has not seen. With 39 merges it averages 1.914 bytes per token (the same number of characters, since the text is ASCII) and decodes both sentences exactly. `analyze_tokenization_efficiency` sorts the 35 tokens: 15 are single bytes, 13 are pieces of a pre-tokenized chunk and 7 are whole chunks, the longest 4 bytes. Words that are frequent in the five training sentences come out whole and other words fall apart into bytes, so on real text this breakdown shows where a vocabulary is too small for the domain. The loop at the end retrains at three vocabulary sizes; {ref}`sec-tdd-best-practices-tokenization` reads its result.

(sec-tdd-tokenization-quality-practice)=

## T.8 Measuring and Improving Tokenization

The challenges of {ref}`sec-tdd-tokenization-challenges-solutions` call for a way to tell a good tokenizer from a poor one. This section names the measurements and then the practices they support.

(sec-tdd-quality-metrics-tokenization)=

### Quality Metrics for Tokenization

Three numbers can be measured on any tokenizer without annotated data: characters per token and bytes per token, which measure compression (more per token means shorter sequences, {ref}`sec-tdd-sequence-length-cost`), and the share of texts that fail the round trip of {ref}`sec-tdd-reversibility-information-loss`. `evaluate_tokenization_quality` in {ref}`sec-tdd-complete-tokenization-pipeline` computes all three and leaves the special ids out of the count. A byte-level tokenizer has no out-of-vocabulary rate to report, because every input is covered. Two further properties are often wanted but need more data: whether splits follow word parts (the Turkish example of {ref}`sec-tdd-cross-lingual-tokenization`) needs morphological annotation, and equal cost across languages needs parallel text, as in {ref}`sec-tdd-linguistic-bias`.

(sec-tdd-best-practices-tokenization)=

### Best Practices for Tokenization

**1. Vocabulary Size Selection**: Train at several sizes and compare the metrics on held-out text. The last lines of the listing in {ref}`sec-tdd-complete-tokenization-pipeline` do this with `ComprehensiveTokenizer.train_bpe`: with no merges every byte is a token (1.000 bytes per token on the held-out sentences), 39 merges give 1.914, and 115 merges, after which every chunk of the training text is a single token, give 3.722. On a real corpus the sizes tried run from a few thousand entries to the hundreds of thousands of {ref}`sec-tdd-choosing-vocabulary-size`, and the gain in bytes per token is weighed against the parameters each size costs.

**2. Preprocessing**: Any change to the text before tokenization is a change the decoder cannot undo. The listing compares a conservative strategy (strip the ends) with an aggressive one (collapse every run of whitespace, remove the space before punctuation) on a line of prose and on two lines of Python:

```python
import re


def preprocess(text, strategy):
    if strategy == "aggressive":
        # collapse every run of whitespace, newlines included
        text = re.sub(r"\s+", " ", text)
        # remove the space before punctuation
        text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


prose = "  Wait , what ?  Really !"
code = "def f(x):\n    return x + 1\n"
for strategy in ["conservative", "aggressive"]:
    clean_prose = preprocess(prose, strategy)
    clean_code = preprocess(code, strategy)
    print(f"{strategy:<12} {clean_prose!r:<26} {clean_code!r}")
```

% output

```text
conservative 'Wait , what ?  Really !'  'def f(x):\n    return x + 1'
aggressive   'Wait, what? Really!'      'def f(x): return x + 1'
```

The aggressive strategy tidies the prose but deletes the newline and the indentation that are the code's syntax, so it suits neither a code model nor any model whose output must reproduce its input. Byte-level tokenizers such as GPT-2's apply no preprocessing at all.

**3. Word Frequency and Zipf's Law**: Most entries of any vocabulary are rare, and the listing shows why on a fixed sample. It splits the Gettysburg Address on everything but letters, counts the lowercase words, not tokens of any tokenizer, and plots their counts against their rank:

```python
import re

import matplotlib.pyplot as plt


def analyze_token_frequency(corpus, tokenizer):
    """Count the units tokenizer.encode returns over the corpus.

    Print the counts and plot them by rank.
    """
    from collections import Counter

    all_tokens = []
    for text in corpus:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

    token_counts = Counter(all_tokens)

    # Calculate statistics
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counts)

    # distinct units seen fewer than 5 times
    rare_threshold = 5
    rare_tokens = sum(1 for count in token_counts.values() if count < rare_threshold)

    print(f"Units counted: {total_tokens:,}")
    print(f"Distinct units: {unique_tokens:,}")
    print(
        f"Distinct units seen fewer than {rare_threshold} times: "
        f"{rare_tokens:,} ({rare_tokens/unique_tokens:.1%})"
    )

    # Plot frequency distribution
    frequencies = sorted(token_counts.values(), reverse=True)
    plt.figure(figsize=(7, 3.6))
    ranks = range(1, len(frequencies) + 1)
    plt.loglog(ranks, frequencies, marker=".", linestyle="none", label="Observed")
    # Zipf's law: frequency proportional to 1 / rank, a line of slope -1 on log-log axes
    zipf = [frequencies[0] / r for r in ranks]
    plt.loglog(ranks, zipf, linestyle="--", color="gray", label="Slope −1 (Zipf)")
    plt.legend(loc="upper right")
    plt.xlabel("Word rank")
    plt.ylabel("Occurrences")
    plt.title("Word frequency by rank, with a slope −1 reference line")
    plt.grid(True)
    plt.show()

    return token_counts


class WordTokenizer:
    """Lowercase words of letters only; enough to count word frequencies."""

    def encode(self, text):
        return re.findall(r"[a-z]+", text.lower())


# A fixed sample: Lincoln's Gettysburg Address (1863), public domain,
# as in Project Gutenberg eBook #4
GETTYSBURG = """Four score and seven years ago, our fathers brought forth upon this
continent a new nation: conceived in liberty, and dedicated to the proposition that all
men are created equal.

Now we are engaged in a great civil war. . .testing whether that nation, or any nation
so conceived and so dedicated. . . can long endure. We are met on a great battlefield of
that war.

We have come to dedicate a portion of that field as a final resting place for those who
here gave their lives that this nation might live. It is altogether fitting and proper
that we should do this.

But, in a larger sense, we cannot dedicate. . .we cannot consecrate. . . we cannot
hallow this ground. The brave men, living and dead, who struggled here have consecrated
it, far above our poor power to add or detract. The world will little note, nor long
remember, what we say here, but it can never forget what they did here.

It is for us the living, rather, to be dedicated here to the unfinished work which they
who fought here have thus far so nobly advanced. It is rather for us to be here
dedicated to the great task remaining before us. . .that from these honored dead we take
increased devotion to that cause for which they gave the last full measure of devotion.
. . that we here highly resolve that these dead shall not have died in vain. . . that
this nation, under God, shall have a new birth of freedom. . . and that government of
the people. . .by the people. . .for the people. . . shall not perish from this
earth."""

counts = analyze_token_frequency(GETTYSBURG.split("\n\n"), WordTokenizer())
print("most frequent:", ", ".join(f"{w} {c}" for w, c in counts.most_common(5)))
```

% output

```text
Units counted: 268
Distinct units: 140
Distinct units seen fewer than 5 times: 127 (90.7%)
most frequent: that 12, the 10, we 10, to 8, here 8
```

```{figure} figures/word-frequency.*
:name: fig-ch1-word-frequency
:width: 84%
:alt: Log-log scatter of word frequency against rank for the 140 distinct words of the
  Gettysburg Address: 12 occurrences for the most frequent word, falling more slowly
  than a dashed line of slope −1 to a flat tail of 93 words that occur once.

The listing's plot for its fixed sample, the 268 words of the Gettysburg Address split
into lowercase words of letters only; `figures/make_figures.py` runs the listing to draw
it. The dashed line has slope −1, Zipf's law anchored at the most frequent word ("that",
12 times). The counts stay above the line (rank 10 has 5 occurrences where the line
gives 1.2), and 93 of the 140 distinct words occur once, which makes the flat tail on
the right.
```

{numref}`fig-ch1-word-frequency` shows the plot. A text this short shows the shape only roughly: Zipf's law describes large corpora, and a few hundred words cannot hold a long sequence of ranks with distinct counts. Even here 127 of the 140 distinct words occur fewer than five times. A word-level vocabulary would have to hold this long tail or map it to an unknown token; a subword vocabulary instead spends its entries on frequent strings and spells a rare word from smaller pieces, down to single bytes in a byte-level tokenizer ({ref}`sec-tdd-out-of-vocabulary-handling`).

**4. Multi-Domain Tokenization**: A tokenizer trained on one mixed corpus learns the merges of whichever domain dominates it. To serve several domains, sample the training text so that each domain contributes the same amount, train once, and report the metrics of `evaluate_tokenization_quality` for each domain separately, so that a domain the tokenizer serves poorly shows up as a low bytes-per-token figure. The medical and code examples of {ref}`sec-tdd-hard-inputs` show the size of the effect: the same text costs from 12 to 16 tokens, or from 28 to 49, depending on what the vocabulary was trained on.

(sec-tdd-embeddings-in-practice)=

## T.9 Embeddings in Practice

This section extends the chapter's {ref}`sec-ch1-embeddings-positional-encoding`: how the embedding width is chosen, how to inspect a trained embedding table, what each positional scheme costs in parameters, and a complete embedding layer.

(sec-tdd-embedding-dimensions)=

### Understanding Embedding Dimensions

The width $d$ of the embedding is also the width of every vector in the residual stream, so it sets the size of every weight matrix: each projection in attention and in the feed-forward network has $d$ rows or columns, and its parameters and multiply-adds grow with $d^2$ when its other dimension grows with $d$. The embedding table itself holds $|\mathcal{V}| \times d$ parameters. The widths of some published models:

| Model | $d$ | Source |
| ----- | --- | ------ |
| T5-small | 512 | its released configuration (`d_model`) |
| GPT-2 small | 768 | {cite}`radford2019language` |
| Llama 3 8B | 4,096 | {cite}`grattafiori2024llama` |
| Llama 3 70B | 8,192 | {cite}`grattafiori2024llama` |
| Llama 3 405B | 16,384 | {cite}`grattafiori2024llama` |

Width grows with model size, but much more slowly than the parameter count: from Llama 3 8B to 405B the width grows four times while the parameter count grows about fifty times.

(sec-tdd-embedding-visualization)=

### Embedding Visualization and Analysis

To see whether related tokens sit close together, take their rows of a trained embedding table, one vector of width $d$ per token, and project them onto two dimensions. Principal component analysis (PCA) is the usual choice: subtract the mean row, then keep the two directions along which the rows vary most, which are the first two right singular vectors of the centered matrix. Each token becomes a point in the plane, and the share of the variance the two directions carry says how much of the table the picture shows; when that share is small, two points that look close may be far apart in the full $d$ dimensions, so compare suspected neighbors by their cosine similarity in the full space as well. A table worth inspecting must be trained first: the chapter's notebook (`examples/attention_mechanisms_demo.ipynb`) trains a small character-level language model, and its embedding table, `char_model.embed.embedding.weight` (64 numbers per character), is a matrix to which this method applies.

(sec-tdd-position-encoding-trade-offs)=

### Position Encoding Trade-offs

The listing compares the extra parameters each scheme of {ref}`sec-ch1-positional-encodings` adds as the maximum length grows, at the sizes of T5-small ($d = 512$, 8 heads, 32 buckets). T5 keeps one bias table in the encoder's self-attention and one in the decoder's, each shared by all the layers of its stack; T5-small's released configuration (`relative_attention_num_buckets` 32, `num_heads` 8) gives $2 \times 32 \times 8 = 512$ parameters. A token embedding table for a 32,000-entry vocabulary is drawn for scale. The vertical axis is logarithmic, so lines that differ by orders of magnitude stay apart; the schemes that add no parameters (sinusoidal, RoPE, ALiBi) cannot be drawn on it and are listed instead.

```python
import matplotlib.pyplot as plt


def compare_position_encodings():
    """Extra parameters each positional scheme adds, by maximum length.

    The sizes are T5-small's.
    """
    seq_lens = [128, 256, 512, 1024, 2048, 4096]
    d_model, num_heads, num_buckets, num_stacks, vocab = 512, 8, 32, 2, 32_000
    results = {
        # for scale; not a position cost
        "Token embedding, 32,000 entries": [vocab * d_model for _ in seq_lens],
        # one d-vector per position
        "Learned absolute": [n * d_model for n in seq_lens],
        # a table per stack
        "Relative bias (T5, 2 stacks)": [
            num_stacks * num_buckets * num_heads for _ in seq_lens
        ],
    }
    labels = ["Token embedding (32k)", "Learned absolute", "T5 bias, 2 stacks"]
    print(f"{'max length':<21}" + "".join(f"{n:>11,}" for n in seq_lens))
    for label, params in zip(labels, results.values()):
        print(f"{label:<21}" + "".join(f"{p:>11,}" for p in params))
    print("no parameters: sinusoidal, RoPE, ALiBi")

    plt.figure(figsize=(6.4, 3.8))
    styles = [("", "--"), ("s", "-"), ("^", "-")]
    for (method, params), (marker, dash) in zip(results.items(), styles):
        plt.plot(seq_lens, params, marker=marker or None, linestyle=dash, label=method)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Maximum sequence length", fontsize=11.5)
    plt.ylabel("Extra parameters", fontsize=11.5)
    plt.title("Parameters spent on positions (log scale)", fontsize=12)
    plt.tick_params(labelsize=11)
    plt.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.0, 0.1))
    plt.grid(True)
    plt.show()


compare_position_encodings()
```

% output

```text
max length                   128        256        512      1,024      2,048      4,096
Token embedding (32k) 16,384,000 16,384,000 16,384,000 16,384,000 16,384,000 16,384,000
Learned absolute          65,536    131,072    262,144    524,288  1,048,576  2,097,152
T5 bias, 2 stacks            512        512        512        512        512        512
no parameters: sinusoidal, RoPE, ALiBi
```

```{figure} figures/position-parameter-cost.*
:name: fig-ch1-position-parameter-cost
:width: 77%
:alt: Log-log plot over maximum lengths 128 to 4096. A dashed flat line for the token
  embedding sits at 16.4 million; the learned absolute line rises from about 65 thousand
  to 2.1 million; the T5 relative bias line stays flat at 512.

The listing's plot on logarithmic axes, at T5-small's sizes. Learned absolute positions
grow in proportion to the maximum length, from 65,536 parameters at 128 to 2,097,152 at
4,096, an eighth of the token embedding table (dashed, 16,384,000). T5's two bias tables
stay at 512 parameters at any length. Sinusoidal positions, RoPE and ALiBi add none and
do not appear on a logarithmic axis.
```

{numref}`fig-ch1-position-parameter-cost` shows that only learned absolute positions grow with the maximum length, by $d$ parameters per position, and that even at 4,096 positions they are small next to the vocabulary. Their real cost is that the model has no vector at all for a position past the end of the table, the extrapolation problem the relative schemes of {ref}`sec-ch1-positional-encodings` avoid.

(sec-tdd-advanced-embedding-implementation)=

### Advanced Embedding Implementation

The layer below combines the scaled token embedding of {ref}`sec-ch1-embedding-initialization-strategies` with learned absolute positions, an optional layer normalization and dropout. For sinusoidal positions, add the table of `create_sinusoidal_encoding` from {ref}`sec-ch1-sinusoidal-positions` instead; relative schemes act on the attention scores and are not part of this layer.

```python
import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Learned absolute positions ("learned") or none ("none").

    Other schemes do not belong in this layer.
    """

    def __init__(self, d_model, max_seq_length=5000, encoding_type="learned"):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type

        if encoding_type == "learned":
            self.position_embeddings = nn.Embedding(max_seq_length, d_model)
        elif encoding_type != "none":
            raise ValueError(
                f"{encoding_type!r}: add a table from create_sinusoidal_encoding "
                "to the input instead; relative schemes (RelativePositionBias, "
                "RoPE, ALiBi) act on the attention scores"
            )

    def forward(self, x, position_ids=None):
        if self.encoding_type == "learned":
            if position_ids is None:
                position_ids = torch.arange(x.size(1), device=x.device)
            return x + self.position_embeddings(position_ids)
        return x  # "none": no positions added


class TokenEmbeddingLayer(nn.Module):
    """Token ids to vectors.

    Embedding lookup, optional sqrt(d) scaling, learned positions, LayerNorm, dropout.
    """

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

        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        self.position_encoding = PositionalEncoding(
            d_model, max_seq_length, position_encoding
        )

        self.layer_norm = nn.LayerNorm(d_model) if layer_norm else None

        self.dropout = nn.Dropout(dropout)

        self._init_embeddings()

    def _init_embeddings(self):
        """N(0, 1/d), so the sqrt(d)-scaled embedding has unit variance.

        The positional encoding is of the same scale.
        """
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=self.d_model**-0.5)

    def forward(self, input_ids, position_ids=None):
        token_embeds = self.token_embeddings(input_ids)

        # scale by sqrt(d), as in the original transformer
        if self.scale_embeddings:
            token_embeds = token_embeds * math.sqrt(self.d_model)

        embeddings = self.position_encoding(token_embeds, position_ids)

        if self.layer_norm:
            embeddings = self.layer_norm(embeddings)

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

(sec-tdd-summary)=

## Summary

- WordPiece merges the pair that most raises the training likelihood; the unigram model starts large and prunes, keeping the pieces with the largest loss; byte-level BPE starts from the 256 byte values; SentencePiece trains BPE or unigram on raw text with whitespace as a symbol ({ref}`sec-tdd-subword-algorithms`).
- Real encodings differ widely on the same text: the Japanese greeting takes 10 tokens in `gpt2` and 2 in `o200k_base`, the German compound 19 and 13 ({ref}`sec-tdd-linguistic-bias`, {ref}`sec-tdd-hard-inputs`).
- Special tokens and chat templates are part of a model's input format, fixed by its training data ({ref}`sec-tdd-special-tokens-their-roles`).
- A larger vocabulary shortens sequences but costs $|\mathcal{V}| \times d$ parameters per embedding matrix; any normalization, even NFC, breaks the byte-exact round trip ({ref}`sec-tdd-practical-considerations`).
- Tokenizers split numbers by different rules: learned merges in `gpt2`, groups of at most three digits in `cl100k_base` and `o200k_base`, single digits in LLaMA ({ref}`sec-tdd-handling-special-cases`).
- A complete byte-level BPE tokenizer fits in one short class and round-trips every text; bytes per token on held-out text measures it ({ref}`sec-tdd-implementation-walkthrough`, {ref}`sec-tdd-tokenization-quality-practice`).
- Only learned absolute positions cost parameters that grow with the length; T5's bias costs a fixed 512 in T5-small ({ref}`sec-tdd-embeddings-in-practice`).

(sec-tdd-exercises)=

## Exercises

The exercises use the page's listings and are graded as in the chapter: level 1 applies one definition, level 2 combines two. The test suite checks every numerical answer.

(sec-tdd-exercise-1)=

### Exercise T1: A WordPiece score (level 1)

In the WordPiece corpus of {ref}`sec-tdd-subword-listing` ("hug" 10 times, "pug" 5, "pun" 12, "bun" 4, "hugs" 5), compute the count and the score of the pair `##u ##n`. Is it merged before `##g ##s`?

```{admonition} Solution T1
:class: dropdown
`##u` occurs in every word, $10 + 5 + 12 + 4 + 5 = 36$ times. `##n` occurs in "pun" and
"bun", $12 + 4 = 16$ times, always after `##u`, so the pair also occurs 16 times. Its
score is $16 / (36 \cdot 16) = 1/36 = 0.0278$, the same as `##u ##g`: when the second
piece never occurs without the first, the score reduces to
$1/\operatorname{count}(\text{first piece})$, and `##u` is common. `##g ##s` scores 0.05
and is merged first, although BPE would rank `##u ##n` (16 occurrences) above it (5).
```

(sec-tdd-exercise-2)=

### Exercise T2: Viterbi against sampling (level 1)

With the unigram probabilities of {ref}`sec-tdd-subword-listing`, compute the probabilities of the segmentations `un happiness` and `un happi ness` and their ratio. The 12 segmentations of "unhappiness" have total probability $2.608 \times 10^{-4}$. What share of it does each of the two carry?

```{admonition} Solution T2
:class: dropdown
$p(\text{un happiness}) = 0.08 \times 0.002 = 1.6 \times 10^{-4}$ and
$p(\text{un happi ness}) = 0.08 \times 0.02 \times 0.06 = 9.6 \times 10^{-5}$, so the
ratio is $5/3 = 1.667$. Their shares are $1.6 / 2.608 = 0.613$ and
$0.96 / 2.608 = 0.368$, the values the listing prints. Viterbi always returns the first;
sampling returns the second about 0.368 of the time, which is why subword regularization
shows the model more than one segmentation of a frequent word.
```

(sec-tdd-exercise-3)=

### Exercise T3: Bytes of an unseen script (level 1)

The `ComprehensiveTokenizer` of {ref}`sec-tdd-complete-tokenization-pipeline` was trained on five English sentences. How many ids does `tokenizer.encode("你好")` return, and what does `decode` give back?

```{admonition} Solution T3
:class: dropdown
Each of the two characters is three UTF-8 bytes, 6 bytes in all. The training text is
ASCII, so no merge involves these bytes and each stays a token of its own: 6 byte ids
plus `<BOS>` and `<EOS>`, 8 ids. `decode` joins the bytes of the ids that are not
special and returns "你好" exactly; the listing asserts this round trip.
```

(sec-tdd-exercise-4)=

### Exercise T4: What a vocabulary costs (level 2)

A model of width $d = 4096$ does not tie its input embedding and output projection. How many parameters do the two matrices hold together with the 100,277 entries of `cl100k_base`, and with the 200,019 entries of `o200k_base`? What fraction of an 8-billion-parameter model is the difference?

```{admonition} Solution T4
:class: dropdown
Each matrix holds $|\mathcal{V}| \times d$ parameters, so the pair holds
$2\,|\mathcal{V}|\,d$: $2 \times 100{,}277 \times 4{,}096 = 821{,}469{,}184$ and
$2 \times 200{,}019 \times 4{,}096 = 1{,}638{,}555{,}648$. The difference is
$817{,}086{,}464$ parameters, $0.817 / 8 = 0.102$ of the model. In return, text in the
languages of {ref}`sec-tdd-linguistic-bias` gets shorter: the Arabic greeting falls from
10 tokens in `cl100k_base` to 4 in `o200k_base`.
```

(sec-tdd-further-reading)=

## Further Reading

- {cite:t}`kudo2018subword` introduces the unigram model and subword regularization, with the pruning algorithm of {ref}`sec-tdd-unigram`.
- {cite:t}`kudo2018sentencepiece` describes the SentencePiece library.
- The Hugging Face course's tokenizer chapter {cite}`huggingface2026wordpiece` implements BPE, WordPiece and unigram training step by step.
- {cite:t}`petrov2023language` measure how much more some languages cost under common tokenizers.
- {cite:t}`xue2022byt5` train a tokenizer-free byte-level model and compare it with subword models.
