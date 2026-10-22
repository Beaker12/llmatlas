# Dataset Management and Versioning

A pretraining corpus is not a file. It is tens of thousands of shards, produced by a pipeline from sources that change, consumed by training runs that must be reproducible, audited after the fact for what they contained, and edited when something has to come out. This chapter covers the engineering that makes that manageable: how the data is laid out on disk, how a version is named and verified, how documents are tokenized, packed and mixed into training batches, how a run resumes without repeating or skipping data, and how a dataset is documented and governed over its life. Chapter 3 acquired the text and Chapter 4 cleaned it; this chapter turns the result into an artifact that a training job can depend on.

## Learning Objectives

By the end of this chapter, you will understand:

- The storage formats used for training data (JSON Lines, Parquet, tar shards, tokenized binaries) and when each is the right choice
- How a dataset version is defined by content hashes and a manifest, and how to diff and verify versions
- How documents are tokenized, packed into fixed-length sequences and masked so that unrelated documents do not attend to each other
- How sources are mixed by weight and temperature, with epoch caps, and how a sampler is made deterministic and resumable
- How data is sharded across many training processes without duplication, and why the global batch should not depend on the number of processes
- What statistics, tests and documents accompany a dataset version, and how removal requests are handled

## Hands-On Learning

The accompanying notebook, [Dataset Management Demo](examples/dataset_management_demo.ipynb), builds a small corpus into a versioned, tokenized training dataset and checks every step with an assertion:

- Writing JSON Lines shards and a tokenized binary with a document index, and reading both back ({ref}`nb-dm-formats`)
- Building a manifest of content hashes, computing a version root, and diffing two versions ({ref}`nb-dm-versioning`)
- Packing documents into sequences, measuring truncation and padding, and building the document attention mask ({ref}`nb-dm-packing`)
- A deterministic, resumable mixture sampler with temperature and epoch caps ({ref}`nb-dm-sampler`)
- Sharding across ranks and showing that the global batch is independent of the world size ({ref}`nb-dm-distributed`)
- Dataset statistics, a data contract, and a data card generated from the manifest ({ref}`nb-dm-governance`)

Everything runs on a CPU in seconds and uses only the standard library and NumPy.

## Chapter Overview

Think of a training dataset the way a software team thinks of a release. Source code is edited constantly, but what ships is a build: a fixed set of files with a version number, produced by a pinned build script, with a changelog and a way to verify that the bytes you downloaded are the bytes that were released. Nobody debugs a production incident against "whatever was in the repository on Tuesday".

**Storage formats** are the file types of the build. Text is kept as compressed JSON Lines or Parquet for inspection and reprocessing, and as flat token arrays for training, because a training loop that parses JSON on every step is wasting the accelerator.

**Versioning** gives the build its number. A version is the set of content hashes of its shards; two versions can be diffed shard by shard, and a downloaded copy can be checked against the manifest. Nothing in a released version is ever modified in place.

**Tokenization and packing** turn documents into the fixed-length sequences the model consumes. Documents are rarely the right length, so they are concatenated and cut, or fitted into sequences like boxes into a container, and the model is told where one document ends and the next begins.

**Mixing and sampling** decide which documents, in which order, from which sources. The sampler is a pure function of a seed and a step number, so a run that crashes at step 40,000 restarts at step 40,001 with exactly the data it would have seen.

**Distributed loading** splits the work across hundreds or thousands of processes without two of them reading the same shard, and without the data order changing when the number of processes changes.

**Monitoring and governance** keep the record: how many tokens from each source, what the tests checked, who may use the dataset, and how a document is found and removed when someone asks.

The theme is that reproducibility is cheap when it is designed in and impossible to retrofit. A dataset built by hand in a notebook, shuffled with an unseeded call and written over an older copy, cannot be reproduced, diffed, or audited, and every experiment trained on it is uninterpretable.

## 1. The Dataset as a Build Artifact

### 1.1 What has to be reproducible

Three distinct things are called "the dataset" and each needs its own handling:

- **The sources**: crawl snapshots, dumps and repositories, as acquired in Chapter 3. They are large, mostly immutable once fetched, and recorded by provenance manifest.
- **The processed corpus**: the output of the Chapter 4 pipeline, a set of cleaned document shards with metadata. This is what is versioned, documented and released.
- **The training stream**: the sequence of token batches a particular run consumed, which depends on the corpus version, the tokenizer, the packing method, the mixture weights, the seed and the number of steps. It is not stored; it is recomputed from those inputs.

Reproducing a model means reproducing the third from the second, which means the second has to be fixed and the transformation has to be deterministic. The practical rule used by every serious training codebase: a run's configuration names a corpus version by hash, a tokenizer by hash, and a data seed, and the loader is a pure function of those.

### 1.2 The scale of the problem

A 15-trillion-token corpus is roughly 60 TB of UTF-8 text, 30 TB as 16-bit token ids, or 60 TB as the 32-bit ids that vocabularies above 65,536 entries need. Stored as 100 MB shards, it is a few hundred thousand files. A training run over it with a 4-million-token global batch takes nearly four million steps, each of which must fetch the right batch in well under the time the accelerators take to process the previous one, from storage that is usually a network object store rather than a local disk.

Those numbers drive the design choices in the rest of the chapter: shards rather than files small enough to list, token arrays rather than text that needs parsing, indexes that locate a document without scanning, and sampling that computes the next batch's location arithmetically rather than by maintaining a queue.

## 2. Storage Formats and Layout

### 2.1 The formats in use

| Format                       | Unit            | Used for                                | Strengths                                           | Weaknesses                                        |
| ---------------------------- | --------------- | --------------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| JSON Lines (gzip or zstd)    | One document per line | Pipeline input and output, inspection | Human-readable, streamable, any tool reads it        | No random access, slow to parse, no schema        |
| Parquet                      | Column chunks   | Processed corpora, metadata queries     | Columnar compression, schema, predicate push-down    | Needs a library, row access is by row group       |
| Tar shards (WebDataset)      | Files in a tar  | Streaming from object stores            | Sequential reads, any payload type, simple           | No random access, no schema                       |
| Tokenized binary plus index  | Token arrays    | Training input                          | Zero parsing, memory-mapped random access            | Tied to one tokenizer, opaque to inspection       |
| Arrow (memory-mapped)        | Record batches  | Local caches (Hugging Face `datasets`)  | Zero-copy access, typed columns                      | Large on disk, not a transport format             |

Most pipelines use two of these: a text format for everything up to and including the released corpus, and a tokenized binary for training. The text format is what gets versioned and documented; the binary is derived from it and the tokenizer, and is rebuilt rather than versioned.

The Hugging Face hub converts hosted corpora to Parquet for its viewer and `datasets` streams Parquet, JSON Lines and other formats from the hub without downloading everything {cite}`lhoest2021datasets`. The Megatron-LM codebase {cite}`shoeybi2019megatron` (the paper is about model parallelism; the data format lives in the repository), GPT-NeoX and their descendants train from an indexed binary format, a flat array of token ids alongside an array of document offsets. Mosaic's streaming format and WebDataset shards exist for the case where training reads directly from object storage and sequential throughput matters more than random access.

### 2.2 Sharding

A shard is the unit of parallelism, of integrity checking and of versioning. The conventions that have settled:

- **Size** between about 100 MB and 1 GB. Smaller shards make listing and manifests unwieldy; larger ones make a single failed download or corrupted file expensive and limit parallelism.
- **Naming** that sorts correctly and encodes the source: `fineweb/CC-MAIN-2024-10/000_00042.jsonl.zst`. Zero-padded, fixed-width indices are not optional; a tool that lists shards in lexical order must get them in numerical order.
- **Self-describing contents**: every record carries its source, its identifier within the source and the pipeline version that produced it, so that a shard found on a disk can be traced without its manifest.
- **One writer per shard**, written to a temporary name and renamed on completion, so that a partially written shard is never mistaken for a complete one.

```python
import gzip
import hashlib
import json
from pathlib import Path


def write_jsonl_shards(docs, out_dir: Path, docs_per_shard: int, prefix: str = "shard"):
    """Write documents to gzip-compressed JSON Lines shards; return a manifest.

    Each manifest entry records the shard's path, record count, uncompressed
    byte count and SHA-256 of the compressed file, which is what later stages
    verify against.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    batch, shard_idx = [], 0

    def flush():
        nonlocal batch, shard_idx
        if not batch:
            return
        path = out_dir / f"{prefix}_{shard_idx:05d}.jsonl.gz"
        tmp = path.with_suffix(".tmp")
        raw = b"".join(json.dumps(d, ensure_ascii=False).encode("utf-8") + b"\n" for d in batch)
        # mtime=0: without it the gzip header records the time of writing
        with open(tmp, "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=0) as f:
            f.write(raw)
        tmp.rename(path)  # a shard exists only once it is complete
        manifest.append(
            {
                "path": path.name,
                "records": len(batch),
                "bytes_uncompressed": len(raw),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        batch, shard_idx = [], shard_idx + 1

    for doc in docs:
        batch.append(doc)
        if len(batch) == docs_per_shard:
            flush()
    flush()
    return manifest


def read_jsonl_shards(out_dir: Path, manifest):
    for entry in manifest:
        path = out_dir / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise IOError(f"{path} does not match its manifest hash")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


import tempfile

docs = [{"id": f"doc{i}", "source": "demo", "text": f"Document number {i}. " * (i + 1)} for i in range(10)]
with tempfile.TemporaryDirectory() as tmp:
    manifest = write_jsonl_shards(docs, Path(tmp), docs_per_shard=4)
    assert [m["records"] for m in manifest] == [4, 4, 2]
    assert [d["id"] for d in read_jsonl_shards(Path(tmp), manifest)] == [d["id"] for d in docs]
    assert not list(Path(tmp).glob("*.tmp"))
```

The `mtime=0` argument matters more than it looks. A gzip header records a modification time, and `gzip.open` fills it with the current time, so the same documents compressed a second apart produce different bytes and different hashes. A shard's hash has to be a function of its content alone, or every rebuild looks like a new version. Zstandard frames carry no timestamp; tar and zip archives do, and need the same treatment.

### 2.3 Tokenized binaries and the document index

Training reads tokens, not text. Tokenizing on the fly costs CPU time per step that is better spent once, and makes the stream depend on the tokenizer library's version at run time. The standard layout, shared in spirit by Megatron-LM's indexed dataset and the `.bin` files that nanoGPT-style codebases use, is:

- A flat array of token ids, `uint16` for vocabularies under 65,536 and `uint32` above, for all documents in the shard concatenated with an end-of-document token after each.
- An offsets array giving the start of each document, so that document $i$ is `tokens[offsets[i]:offsets[i + 1]]`.
- A header or sidecar naming the tokenizer by hash and the dtype, so the file cannot be read with the wrong vocabulary by accident.

Memory-mapping the token array lets a process read any document, or any window of the concatenated stream, with no parsing and no copy. Random access to the stream is an array slice.

```python
import hashlib
import json
from pathlib import Path

import numpy as np

EOD = 0  # end-of-document token id; real tokenizers reserve one


class TokenShard:
    """Flat token array plus document offsets, memory-mapped on read."""

    def __init__(self, path: Path):
        self.path = path
        meta = json.loads(path.with_suffix(".json").read_text())
        self.dtype = np.dtype(meta["dtype"])
        self.tokenizer_hash = meta["tokenizer_hash"]
        self.offsets = np.load(path.with_suffix(".idx.npy"))
        self.tokens = np.memmap(path, dtype=self.dtype, mode="r")

    @classmethod
    def write(cls, path: Path, token_docs, tokenizer_hash: str, vocab_size: int):
        dtype = np.uint16 if vocab_size < 2**16 else np.uint32
        offsets = [0]
        chunks = []
        for toks in token_docs:
            arr = np.asarray(list(toks) + [EOD], dtype=dtype)
            chunks.append(arr)
            offsets.append(offsets[-1] + len(arr))
        np.concatenate(chunks).tofile(path)
        np.save(path.with_suffix(".idx.npy"), np.asarray(offsets, dtype=np.int64))
        path.with_suffix(".json").write_text(
            json.dumps({"dtype": dtype.__name__ if hasattr(dtype, "__name__") else np.dtype(dtype).name,
                        "tokenizer_hash": tokenizer_hash, "num_docs": len(chunks)})
        )
        return cls(path)

    def __len__(self):
        return len(self.offsets) - 1

    def document(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[self.offsets[i] : self.offsets[i + 1]])

    def window(self, start: int, length: int) -> np.ndarray:
        return np.asarray(self.tokens[start : start + length])


def toy_tokenize(text: str):
    """Deterministic stand-in for a real tokenizer: one id per word, 1..9999."""
    return [1 + (int(hashlib.md5(w.encode()).hexdigest(), 16) % 9999) for w in text.split()]


import tempfile

texts = ["the cat sat", "on the mat and slept", "fin"]
with tempfile.TemporaryDirectory() as tmp:
    shard = TokenShard.write(Path(tmp) / "demo.bin", (toy_tokenize(t) for t in texts),
                             tokenizer_hash="toy-v1", vocab_size=10_000)
    assert len(shard) == 3
    assert shard.document(0).tolist() == toy_tokenize("the cat sat") + [EOD]
    assert shard.offsets.tolist() == [0, 4, 10, 12]
    assert shard.window(2, 4).tolist() == [toy_tokenize("the cat sat")[2], EOD] + toy_tokenize("on the mat and slept")[:2]
    assert shard.tokens.dtype == np.uint16
```

The tokenizer hash is the part people skip and regret. Two tokenizers with the same name and vocabulary size but different normalisation or merge order produce different id streams, and a model evaluated with the wrong one is silently broken. Hash the tokenizer's serialised files and store the hash with every binary that depends on it.

```{admonition} Formats on Disk
:class: tip
The notebook section {ref}`nb-dm-formats` writes a corpus as JSON Lines shards and as a tokenized binary with offsets, verifies both against their manifests, and compares the bytes per token of each layout.
```

## 3. Versioning

### 3.1 A version is a set of hashes

Git works for source code because a commit is a hash of a tree of hashes of file contents. The same structure works for datasets at any scale, and the tools that do it well expose the same model, whether they store the objects themselves by content hash (DVC, Git LFS, the Hugging Face hub's storage layer) or keep content-addressed commit metadata over ordinary object storage (lakeFS):

- Each shard is content-addressed: its identity is the hash of its bytes, and it is stored under that hash.
- A **manifest** lists the shards of a version by path, hash and size. The manifest is small enough to keep in Git.
- The **version root** is the hash of the manifest, or a Merkle root over the shard hashes, so that one short string identifies the whole dataset.

Because shards are stored by hash, two versions that share most of their shards share the storage, and the diff between them is the diff between their manifests. A corpus rebuilt with one filter changed differs in the shards that filter touched and nowhere else, and the manifest diff says exactly which. This holds only if shard membership is stable: a document must land in the same shard in both builds. Shards cut by running count do not have that property, since removing one document shifts every later shard by one; shards cut by source and by a hash of the document identifier do, which is why pipelines bucket by key rather than by count.

```python
import hashlib
import json


def manifest_root(manifest) -> str:
    """Hash of the sorted (path, sha256) pairs; identifies the version."""
    canonical = json.dumps(sorted((m["path"], m["sha256"]) for m in manifest))
    return hashlib.sha256(canonical.encode()).hexdigest()


def diff_manifests(old, new):
    old_by_path = {m["path"]: m["sha256"] for m in old}
    new_by_path = {m["path"]: m["sha256"] for m in new}
    return {
        "added": sorted(set(new_by_path) - set(old_by_path)),
        "removed": sorted(set(old_by_path) - set(new_by_path)),
        "changed": sorted(p for p in old_by_path.keys() & new_by_path.keys() if old_by_path[p] != new_by_path[p]),
        "unchanged": sum(1 for p in old_by_path.keys() & new_by_path.keys() if old_by_path[p] == new_by_path[p]),
    }


v1 = [{"path": f"shard_{i:03d}", "sha256": f"h{i}"} for i in range(5)]
v2 = [dict(m) for m in v1]
v2[2]["sha256"] = "h2-refiltered"
v2.append({"path": "shard_005", "sha256": "h5"})
del v2[0]

assert manifest_root(v1) != manifest_root(v2)
assert manifest_root(v1) == manifest_root(list(reversed(v1)))  # order-independent
assert diff_manifests(v1, v2) == {
    "added": ["shard_005"],
    "removed": ["shard_000"],
    "changed": ["shard_002"],
    "unchanged": 3,
}
```

### 3.2 Immutability and naming

A released version is never modified. If a shard is found to be corrupt or to contain something that must be removed, a new version is cut without it, and the old version's manifest is kept so that models trained on it can be audited. The two conventions that make this workable:

- **Human-readable tags** on top of hashes: `fineweb-edu-v1.2.0` rather than a 64-character string. A tag is a pointer to a root hash and is itself recorded in the version history, as a Git tag is.
- **Semantic-style versioning** for the tags, with an agreed meaning: a patch change removes or corrects shards without changing the pipeline; a minor change reruns the pipeline with changed parameters; a major change alters the sources or the schema. The point is less the numbers than having a rule for which experiments remain comparable.

A dataset tag is the thing a training configuration references. A configuration that says `data: fineweb-edu-v1.2.0` with the tag resolved to a root hash at launch and recorded in the run's metadata is reproducible; one that says `data: /mnt/data/fineweb-edu/latest` is not.

### 3.3 Lineage

Each version records how it was made: the versions of its inputs, the pipeline code commit, the configuration, and the tokenizer if it is a tokenized build. This forms a directed graph from raw sources through cleaned corpora to tokenized datasets to models, and it is the graph that answers the questions that come up later: which models were trained on data that included this document, which runs would change if this filter were fixed, and whether two experiments differ in their data or only in their seeds. Tools such as DVC store the graph as pipeline stages with declared inputs and outputs; others store it as metadata on each version. Either way the graph is only as good as the discipline of never producing a version outside it.

```{admonition} Versions and Diffs
:class: note
The notebook section {ref}`nb-dm-versioning` builds a manifest for the corpus written earlier, rebuilds it with one document changed and one removed, and shows that the version root changes while the diff names exactly the affected shards. It then verifies a downloaded copy against the manifest and detects a corrupted shard.
```

## 4. Tokenization and Packing

### 4.1 From documents to sequences

A transformer trains on sequences of a fixed length $L$ (2,048 to 128,000 tokens in current models), and documents are whatever length they are. Three ways to reconcile them:

- **Concatenate and chunk.** Join all documents with an end-of-document token between them into one stream, and cut it into consecutive windows of length $L$. No padding, no waste, and the method the GPT-3 paper describes and most pretraining codebases default to. The cost: a document is split across windows wherever the cut falls, and a window usually contains the tail of one document and the head of the next.
- **Pad.** One document per sequence, padded to $L$. No cross-document contamination, but with a long-tailed length distribution most of each sequence is padding, which is compute spent on nothing. Used for fine-tuning on short examples, rarely for pretraining.
- **Pack.** Fit whole documents into sequences like boxes into a container (a bin-packing problem) so that most sequences are nearly full and documents are rarely cut. Best-fit packing {cite}`ding2024fewer` first splits any document longer than $L$ into pieces of at most $L$, then places each piece into the sequence with the least remaining room that still fits it, processing pieces longest first. It reduced truncation substantially relative to chunking, with measurable gains in reading comprehension and faithfulness and less hallucination, at the cost of a sort and an index.

```python
import numpy as np


def concat_and_chunk(token_docs, seq_len: int, eod: int = 0):
    """Join documents with EOD and cut into windows. Returns sequences and the
    fraction of documents that were split across a window boundary."""
    stream, boundaries, pos = [], [], 0
    for doc in token_docs:
        stream.extend(doc)
        stream.append(eod)
        boundaries.append((pos, pos + len(doc)))
        pos += len(doc) + 1
    n_seq = len(stream) // seq_len
    sequences = np.asarray(stream[: n_seq * seq_len]).reshape(n_seq, seq_len)
    split = sum((start // seq_len) != ((end - 1) // seq_len) for start, end in boundaries)
    return sequences, split / len(boundaries)


def best_fit_pack(token_docs, seq_len: int, eod: int = 0, pad: int = -1):
    """Best-fit-decreasing packing: documents longer than seq_len are split into
    pieces first; each piece (plus EOD) then goes into the sequence with the least
    remaining room that still fits it, longest pieces first."""
    pieces = []
    for d in token_docs:
        d = list(d)
        for start in range(0, max(1, len(d)), seq_len - 1):
            pieces.append(d[start : start + seq_len - 1] + [eod])
    docs = sorted(pieces, key=len, reverse=True)
    bins, free = [], []
    for doc in docs:
        best = None
        for i, space in enumerate(free):
            if len(doc) <= space and (best is None or space < free[best]):
                best = i
        if best is None:
            bins.append(list(doc))
            free.append(seq_len - len(doc))
        else:
            bins[best].extend(doc)
            free[best] -= len(doc)
    sequences = np.full((len(bins), seq_len), pad, dtype=np.int64)
    for i, b in enumerate(bins):
        sequences[i, : len(b)] = b
    padding_frac = float((sequences == pad).mean())
    return sequences, padding_frac


rng = np.random.default_rng(0)
lengths = rng.geometric(1 / 60, size=400)  # long-tailed document lengths, mean 60
docs = [list(rng.integers(1, 1000, size=n)) for n in lengths]

chunked, split_frac = concat_and_chunk(docs, seq_len=256)
packed, pad_frac = best_fit_pack(docs, seq_len=256)
assert split_frac > 0.1  # chunking splits many documents
assert pad_frac < 0.05  # packing wastes little
assert packed.shape[1] == 256 and chunked.shape[1] == 256
# Packing needs more sequences than chunking, by about the padding fraction (nothing is dropped)
assert chunked.shape[0] <= packed.shape[0] <= chunked.shape[0] * 1.1
n_pieces = sum(max(1, -(-len(d) // 255)) for d in docs)  # pieces of at most 255 tokens plus EOD
assert (packed != -1).sum() == sum(len(d) for d in docs) + n_pieces  # every token kept, one EOD per piece
```

### 4.2 Cross-document attention

When a sequence holds several documents, causal attention lets a token in the second document attend to the first, which is both a waste (the first document is irrelevant) and a source of the model learning spurious dependencies. The fix is a document mask: a token attends only to earlier tokens of its own document. With a segment id per position, the mask is $M_{ij} = [j \le i] \cdot [s_i = s_j]$, block-diagonal within the causal triangle. Efficient attention kernels accept this as a list of segment boundaries rather than a full matrix.

```python
import numpy as np


def segment_ids(sequence, eod: int = 0) -> np.ndarray:
    """Document index of each position; the EOD token belongs to the document it ends."""
    ends = np.asarray(sequence) == eod
    return np.concatenate([[0], np.cumsum(ends)[:-1]])


def document_causal_mask(sequence, eod: int = 0) -> np.ndarray:
    seg = segment_ids(sequence, eod)
    causal = np.tril(np.ones((len(seg), len(seg)), dtype=bool))
    same_doc = seg[:, None] == seg[None, :]
    return causal & same_doc


seq = [5, 6, 0, 7, 8, 9, 0, 1]
assert segment_ids(seq).tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
mask = document_causal_mask(seq)
assert mask[4, 3] and mask[4, 4]  # within document 1, causal
assert not mask[4, 1]  # document 1 cannot see document 0
assert not mask[3, 4]  # and not the future
assert mask.sum() == 3 + 2 + 1 + 4 + 3 + 2 + 1 + 1
```

Whether to mask is a choice. Many pretraining runs did not, on the grounds that the kernel was simpler and the effect small; recent ones increasingly do, since efficient variable-length attention makes it free. Fine-tuning with packed examples should always mask, because the examples are short and the leakage between them is large.

```{admonition} Packing and Masking
:class: tip
The notebook section {ref}`nb-dm-packing` packs the same corpus by chunking and by best-fit, reports the split and padding rates, plots the distribution of documents per sequence, and renders the document-causal mask for the fullest packed sequence.
```

## 5. Mixing and Sampling

### 5.1 Weights, temperature and epochs

A corpus has several sources with very different sizes, and the fraction of training tokens drawn from each is a decision, not a consequence of size. Chapter 3 covered how the weights are chosen; here the question is how a sampler realises them.

Given weights $w_s$ that sum to one, the sampler draws each document's source with probability $w_s$ and then the next unseen document from that source. Two refinements:

- **Temperature** flattens or sharpens the weights: $p_s \propto w_s^{1/T}$. With $T = 1$ the weights are used as given; $T > 1$ upsamples small sources, which is how multilingual models give low-resource languages more than their share {cite}`xue2021mt5`. The same scheme is used on domain weights.
- **Epoch caps** bound how many times a source is repeated. If source $s$ has $N_s$ tokens and the run needs $D$ tokens, source $s$ is seen $p_s D / N_s$ times; beyond about four repetitions the marginal token is worth much less {cite}`muennighoff2023scaling`. A cap re-normalises the weights of the uncapped sources once a capped one has used its budget.

Learned weights (DoReMi {cite}`xie2023doremi`, which fits static $w_s$ with a small proxy model) and curricula that change the mixture over training are both consumed by the same sampler, the latter with $w_s$ a function of the step. The cap is stated in tokens; the sampler below counts documents, which production samplers replace with token counts from the index.

### 5.2 Determinism and resumption

The sampler must be a pure function of `(seed, step)`. The reason is resumption: a run that fails at step $k$ and restarts must continue from step $k + 1$ with the data it would have seen, not restart its epoch, and not replay the batch it just trained on. A sampler that keeps a queue or an iterator in memory cannot do this; one that computes the batch at step $k$ arithmetically can.

The construction that achieves it: for each source, a seeded permutation of its document indices defines the order within an epoch; a seeded sequence of source draws decides which source supplies each slot; the document for slot $t$ is the next unused index of the chosen source. All of this can be recomputed from the seed in $O(t)$ time for the draws, or in $O(1)$ with per-epoch permutations generated by a counter-based random number generator.

```python
import numpy as np


class MixtureSampler:
    """Deterministic, resumable sampler over several sources.

    State is (seed, step). Restarting at any step reproduces the same stream.
    """

    def __init__(self, sizes: dict, weights: dict, seed: int, temperature: float = 1.0, max_epochs: float = 4.0):
        self.names = sorted(sizes)
        self.sizes = np.array([sizes[n] for n in self.names], dtype=np.int64)
        w = np.array([weights[n] for n in self.names], dtype=np.float64) ** (1.0 / temperature)
        self.probs = w / w.sum()
        self.seed = seed
        self.max_epochs = max_epochs
        self.reset()

    def reset(self):
        self.step = 0
        self.cursor = np.zeros(len(self.names), dtype=np.int64)  # documents consumed per source
        self.rng = np.random.default_rng(self.seed)
        self._perm = [None] * len(self.names)

    def _permutation(self, s: int, epoch: int):
        if self._perm[s] is None or self._perm[s][0] != epoch:
            prng = np.random.default_rng([self.seed, s, epoch])  # counter-based seeding
            self._perm[s] = (epoch, prng.permutation(self.sizes[s]))
        return self._perm[s][1]

    def __next__(self):
        exhausted = self.cursor >= self.sizes * self.max_epochs
        probs = np.where(exhausted, 0.0, self.probs)
        if probs.sum() == 0:
            raise StopIteration
        s = self.rng.choice(len(self.names), p=probs / probs.sum())
        epoch, within = divmod(int(self.cursor[s]), int(self.sizes[s]))
        doc = int(self._permutation(s, epoch)[within])
        self.cursor[s] += 1
        self.step += 1
        return self.names[s], doc

    def __iter__(self):
        return self

    def seek(self, step: int):
        """Resume at `step` by replaying the cheap draw sequence."""
        self.reset()
        for _ in range(step):
            next(self)
        return self


sizes = {"web": 1000, "wiki": 50, "code": 200}
weights = {"web": 0.7, "wiki": 0.15, "code": 0.15}

a = MixtureSampler(sizes, weights, seed=1)
stream = [next(a) for _ in range(500)]
b = MixtureSampler(sizes, weights, seed=1).seek(300)
assert [next(b) for _ in range(200)] == stream[300:]  # resumption is exact

# Within a source, every document is seen once before any is repeated
wiki_docs = [d for s, d in stream if s == "wiki"]
assert len(wiki_docs) >= 50  # about 75 draws expected from 500 at weight 0.15
assert sorted(wiki_docs[:50]) == list(range(50))

# Temperature > 1 flattens the mixture toward the small sources
flat = MixtureSampler(sizes, weights, seed=1, temperature=3.0)
assert flat.probs[flat.names.index("wiki")] > a.probs[a.names.index("wiki")]
```

The `seek` method replays draws, which is fine for millions of steps and a few sources. Production samplers are designed differently so that no replay is needed: the mixture is fixed in advance as a schedule of phases (the caps are applied when the schedule is computed, not as the run proceeds), the source for slot $t$ is a hash of `(seed, t)` under that schedule, and the within-source cursor at step $t$ is a count obtainable from a prefix sum, so that any step is $O(1)$. The price is that the schedule must be recomputed if the weights change mid-run.

### 5.3 Shuffling at the right granularity

Three levels of shuffling exist and they are not interchangeable:

- **Across shards**: the order in which shards are visited. Cheap, and sufficient to mix sources if each shard is single-source.
- **Within a shard**: the order of documents inside it. Needed because shards are often written in crawl or alphabetical order, which correlates documents.
- **Across the whole corpus**: a global permutation, which is what the sampler above does with per-source permutations. Ideal, but requires random access to any document, which object storage and sequential formats do not give cheaply.

Streaming loaders compromise with a shuffle buffer: read shards in a seeded order, hold a few thousand documents in memory, and emit from the buffer at random. The quality of the mix depends on the buffer size relative to the shard size, and a buffer smaller than a shard barely shuffles at all. Mosaic's streaming format and WebDataset both document their shuffle schemes precisely because an insufficiently shuffled stream shows up as a periodic loss curve, which is hard to diagnose after the fact.

```{admonition} The Sampler
:class: note
The notebook section {ref}`nb-dm-sampler` runs the mixture sampler, plots the realised source fractions at several temperatures with the tempered target weights marked, shows the epoch cap re-normalising the mixture once the smallest source is exhausted, and checks resumption at a hundred random steps.
```

## 6. Distributed Loading

### 6.1 Sharding across ranks

A training job with $W$ processes (data-parallel ranks) needs each step's global batch of $B$ sequences split into $W$ micro-batches of $B / W$, with every rank reading different data and no rank reading what another reads. Two designs:

- **Shard assignment.** Shards are partitioned among ranks, rank $r$ gets every $W$-th shard, and each rank samples within its own shards. Simple, and what a WebDataset-style loader does. The drawback is that the global batch at step $t$ depends on $W$: change the number of GPUs and the data order changes, so a run resumed on a different cluster size is not the same run.
- **Global indexing.** The sampler produces a single global sequence of samples; the sample at global position $t B + i$ goes to rank $i \bmod W$ (or rank $\lfloor i W / B \rfloor$). Every rank runs the same sampler and takes its slice. The global batch is then independent of $W$, and a run can be resumed with a different number of ranks and see the same data in the same order.

The second property, that the stream is a function of `(seed, step)` and nothing else, is what makes a large run reproducible across hardware changes, and it costs nothing beyond requiring that every rank can read any shard. Mosaic Streaming is built around it: its sample order is fixed by the seed and does not change with the number of GPUs.

```python
import numpy as np


def rank_shards(num_shards: int, rank: int, world_size: int):
    """Shard assignment: rank r takes shards r, r + W, r + 2W, ..."""
    return list(range(rank, num_shards, world_size))


def global_batch_slice(step: int, global_batch: int, rank: int, world_size: int):
    """Global indexing: the sample ids rank r processes at `step`."""
    assert global_batch % world_size == 0
    per_rank = global_batch // world_size
    start = step * global_batch + rank * per_rank
    return np.arange(start, start + per_rank)


def shard_assignment_batch(step, global_batch, world_size, num_shards, docs_per_shard):
    """The set of document ids the whole job sees at `step` under shard assignment:
    each rank reads its own shards in order and takes the next per-rank slice."""
    per_rank = global_batch // world_size
    batch = []
    for r in range(world_size):
        mine = [s * docs_per_shard + i for s in rank_shards(num_shards, r, world_size) for i in range(docs_per_shard)]
        batch += mine[step * per_rank : (step + 1) * per_rank]
    return sorted(batch)


# Shard assignment covers every shard exactly once ...
assigned = [s for r in range(4) for s in rank_shards(10, r, 4)]
assert sorted(assigned) == list(range(10))

# ... but the documents in the global batch at a step depend on the world size
assert shard_assignment_batch(3, 8, 2, 10, 16) != shard_assignment_batch(3, 8, 4, 10, 16)

# Global indexing: the union over ranks at step t is the same set for any world size
step, B = 7, 64
for W in (1, 2, 4, 8, 16):
    ids = np.concatenate([global_batch_slice(step, B, r, W) for r in range(W)])
    assert sorted(ids.tolist()) == list(range(step * B, (step + 1) * B))
```

### 6.2 Throughput

The loader's job is to never make the accelerator wait. The arithmetic for a modern run: a step over a 4-million-token batch on a large cluster takes a few seconds; the batch is 8 MB of `uint16` ids (16 MB as `uint32`), spread over all ranks, so each rank reads kilobytes per step, trivial for local disk and comfortable for object storage with a few requests in flight. Text formats change the picture less in volume than in work: the same batch is some 16 MB of UTF-8 that must be decompressed, parsed and tokenized, which is CPU time on the step's critical path unless it is prefetched, and whose output depends on the tokenizer library's version at run time. This is the practical argument for pre-tokenized binaries: not storage (gzip-compressed text is comparable in size to 16-bit ids), but zero parsing, random access, and a token stream fixed when the dataset is built rather than when it is read.

The remaining ingredients are standard: a few worker processes per rank that prefetch the next batches, reads of whole shards or large ranges rather than per-document requests, and a local cache for shards that will be read again in a later epoch.

```{admonition} Sharding Across Ranks
:class: tip
The notebook section {ref}`nb-dm-distributed` simulates eight ranks under both designs, checks that each covers the corpus exactly once, and shows the global batch at a fixed step staying identical as the world size changes under global indexing and changing under shard assignment.
```

## 7. Statistics, Tests and Monitoring

### 7.1 What to compute for every version

A dataset version ships with its statistics, computed by the build and stored next to the manifest, because the questions they answer come up in every training post-mortem:

- Tokens and documents per source, and the realised mixture (which differs from the planned one once epoch caps and filtering have acted)
- Document length distribution per source (mean, median, percentiles, maximum); a change in the tail between versions usually means an extraction change
- Per-stage drop counts from the Chapter 4 pipeline, so that a version's keep rate can be compared with its predecessor's
- Deduplication rate within the version and overlap with the previous version
- Language distribution, and for code the distribution over languages and licences
- Overlap with the evaluation suite after decontamination, per benchmark, reported as a number even when it is zero

Elazar et al. {cite}`elazar2024whats` built exactly this kind of inspection tooling for the public corpora and found, among other things, that benchmark contamination and duplicated content were common across them. Statistics that are computed rather than asserted are the defence.

### 7.2 Data contracts

The analogue of a unit test for data is a contract: a set of assertions about a version that the build runs before publishing it. Typical checks:

- Schema: every record has the required fields with the right types; identifiers are unique within the version
- Ranges: no empty documents, no document above the length cap, token ids below the vocabulary size
- Composition: each source's token share within a tolerance of the plan; no source silently empty
- Hygiene: canary strings absent, PII patterns below the pipeline's residual rate, decontamination index overlap zero
- Integrity: every shard's hash matches the manifest, every shard is readable to the end

```python
import numpy as np


def dataset_stats(records):
    """Per-source token and document counts and length percentiles."""
    by_source = {}
    for rec in records:
        s = by_source.setdefault(rec["source"], [])
        s.append(rec["num_tokens"])
    stats = {}
    total = sum(sum(v) for v in by_source.values())
    for source, lengths in by_source.items():
        arr = np.asarray(lengths)
        stats[source] = {
            "documents": int(arr.size),
            "tokens": int(arr.sum()),
            "share": float(arr.sum() / total),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "max": int(arr.max()),
        }
    return stats


def check_contract(stats, plan: dict, tolerance: float = 0.05, max_len: int = 100_000):
    """Return a list of violated expectations; empty means the version passes."""
    failures = []
    for source, target in plan.items():
        if source not in stats:
            failures.append(f"{source}: missing from dataset")
            continue
        if abs(stats[source]["share"] - target) > tolerance:
            failures.append(f"{source}: share {stats[source]['share']:.3f} vs plan {target:.3f}")
        if stats[source]["max"] > max_len:
            failures.append(f"{source}: document of {stats[source]['max']} tokens exceeds {max_len}")
    for source in stats:
        if source not in plan:
            failures.append(f"{source}: present but not in plan")
    return failures


rng = np.random.default_rng(3)
records = (
    [{"source": "web", "num_tokens": int(n)} for n in rng.geometric(1 / 500, 7000)]
    + [{"source": "code", "num_tokens": int(n)} for n in rng.geometric(1 / 800, 1500)]
    + [{"source": "wiki", "num_tokens": int(n)} for n in rng.geometric(1 / 400, 1000)]
)
stats = dataset_stats(records)
plan = {"web": 0.70, "code": 0.22, "wiki": 0.08}
assert check_contract(stats, plan, tolerance=0.08) == []
assert any("books" in f for f in check_contract(stats, {**plan, "books": 0.1}, tolerance=0.08))
assert any("exceeds" in f for f in check_contract(stats, plan, tolerance=0.08, max_len=1000))
```

### 7.3 Canaries and drift

Two monitoring practices borrowed from other fields:

- **Canaries.** Insert a handful of unique, random strings into the corpus at known positions and with known repetition counts. After training, test whether the model can complete them. Canaries measure memorisation directly {cite}`carlini2019secret`, and a canary that appears in a version where it should not have is proof that an upstream shard leaked.
- **Drift.** Compare every statistic in Section 7.1 between consecutive versions and flag changes beyond a threshold. Most data bugs (an extractor update that drops tables, a language model update that rejects more of one language, a source that silently went offline) show up as a drift in a per-source statistic before they show up in a loss curve.

```{admonition} Statistics and Contracts
:class: note
The notebook section {ref}`nb-dm-governance` computes per-source statistics for the demo corpus, runs a contract against a plan, injects a canary and finds it, and renders a data card from the manifest and statistics.
```

## 8. Documentation and Governance

### 8.1 Data cards and datasheets

A dataset version is accompanied by a document that says what it is for someone who was not there when it was built. The two established templates:

- **Datasheets for Datasets** {cite}`gebru2021datasheets`: a questionnaire covering motivation, composition, collection process, preprocessing, uses, distribution and maintenance. It was written to make the people assembling a dataset answer the questions a user will have, and it remains the reference.
- **Data cards** {cite}`pushkarna2022data`: a structured summary intended to be published with the dataset, with an emphasis on what a downstream team needs to decide whether to use it.

The pretraining corpora that are well documented (The Pile {cite}`gao2020pile`, Dolma {cite}`soldaini2024dolma`, FineWeb {cite}`penedo2024fineweb`) publish a datasheet or card, the per-source statistics, the pipeline code and the ablation results together, and that combination is what allows anyone else to build on them. C4's post-hoc documentation {cite}`dodge2021documenting`, written by a different team about eighteen months after release, found that the corpus contained machine-generated text, benchmark test sets and a skew in whose text had been filtered out, none of which its original description mentioned.

Croissant, a machine-readable metadata vocabulary built on schema.org, is becoming the common format for the card's structured part (sources, licences, fields, splits) so that hubs and loaders can read it; the prose part still needs a person.

### 8.2 Licences and the provenance record

A corpus mixes sources under different terms, and the terms travel with the data. The version's metadata records, per source, the licence or terms under which it was obtained, the opt-out policy applied and the date, and any restrictions that propagate to models trained on it. Longpre et al. {cite}`longpre2023data` audited the licences attached to popular fine-tuning datasets and found that a majority were mislabelled or had their restrictions dropped as they were repackaged; the only remedy is to carry the record forward from acquisition (Chapter 3) through every version rather than reconstructing it later.

Access control follows from the record. A corpus with a non-commercial source cannot be released under a commercial licence as a whole, so either the source is separated into its own shards with their own terms, or the released version excludes it. Keeping sources in separate shards, which the naming convention in Section 2.2 already does, is what makes this a manifest operation rather than a rebuild.

### 8.3 Removal

Requests to remove content arrive after release: an opt-out, a takedown, a discovered privacy problem, a licence that turned out not to permit the use. Handling them is where the earlier design pays off:

1. Locate the document. A content-addressed index from document hash (or URL, or source identifier) to the shards and versions containing it answers this in one lookup. Without it, removal means scanning every version.
1. Cut a new version without the document, with the manifest diff naming the shards that changed, and record the reason.
1. Record which models were trained on versions containing it, from the lineage graph. What is done about those models is a policy question (retraining, machine unlearning, or disclosure) that the data system cannot answer, but it can say exactly which models are affected.

```python
import hashlib


def document_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DocumentIndex:
    """Maps a document hash to every (version, shard) that contains it."""

    def __init__(self):
        self.locations = {}

    def add_version(self, version: str, shard_docs: dict):
        for shard, docs in shard_docs.items():
            for text in docs:
                self.locations.setdefault(document_hash(text), set()).add((version, shard))

    def locate(self, text: str):
        return sorted(self.locations.get(document_hash(text), ()))


index = DocumentIndex()
index.add_version("v1.0", {"s0": ["alpha", "beta"], "s1": ["gamma"]})
index.add_version("v1.1", {"s0": ["alpha", "beta"], "s1": ["gamma-revised"], "s2": ["delta"]})
assert index.locate("gamma") == [("v1.0", "s1")]
assert index.locate("alpha") == [("v1.0", "s0"), ("v1.1", "s0")]
assert index.locate("not present") == []
```

The index is cheap: one hash per document, which the build computes anyway for deduplication, and a key-value store that fits on one machine even at tens of billions of entries.

## 9. Putting It Together

### 9.1 The build

A dataset build, in the sense of a software build, is a script under version control that:

1. Resolves its inputs to hashes: the source manifests from acquisition, the pipeline commit, the configuration, the tokenizer.
1. Runs the pipeline, writing shards to temporary names and renaming on completion.
1. Computes the manifest, root hash, statistics and contract results.
1. Fails if the contract fails.
1. Publishes the shards to content-addressed storage, the manifest and statistics to the version history, and tags the version.
1. Records the lineage: inputs, outputs and the build's own identity.

Everything a training run needs is then a tag. A run's configuration names the tag, the loader resolves it to a manifest at launch and records the root hash in the run metadata, and from then on the run's data is a function of that hash, the seed and the step.

### 9.2 Tooling

The pieces exist as libraries, and most teams assemble rather than write them:

- **Hugging Face `datasets`** {cite}`lhoest2021datasets` for Parquet and Arrow handling, streaming from the hub, and memory-mapped local caches; the hub itself for hosting versioned datasets with per-commit manifests.
- **Mosaic Streaming (MDS)** for a shard format that streams from object storage with a deterministic, resumable sample order that does not depend on the number of ranks; **WebDataset** for tar shards streamed sequentially with a seeded shard order and a shuffle buffer.
- **Megatron-LM, GPT-NeoX and their derivatives** for the indexed tokenized binary and the blended-dataset sampler that implements weighted mixing; epoch caps are set when the blend is planned.
- **DVC** and **lakeFS** for content-addressed versioning with lineage, in a Git-like workflow; **Git LFS** when the data is small enough.
- **Great Expectations** and similar for data contracts, though a few hundred lines of assertions in the build script are usually enough.
- **Croissant** for machine-readable dataset metadata.

What no tool supplies is the discipline: that every version is produced by the build, that no shard is edited in place, that every run names a tag, and that the statistics and the card are updated with the data rather than after it.

## Exercises

1. **Bytes per token.** Take a few thousand documents and store them as gzip JSON Lines, as a `uint16` token binary, and as a `uint32` token binary. Report bytes per token for each and the time to read every document back. At what point does the text format's inspectability stop being worth its cost?
1. **Manifest diff.** Rebuild the notebook's corpus with the Chapter 4 repetition filter's thresholds changed by 10%. Diff the manifests. How many shards changed, and could you have predicted which ones from the shard naming scheme?
1. **Packing efficiency.** For document length distributions with means of 100, 1,000 and 10,000 tokens and a sequence length of 4,096, measure the truncation rate of chunking and the padding rate of best-fit packing. When does packing stop being worth its complexity?
1. **The mask's effect.** Train two small models on packed sequences with and without the document mask (the Chapter 1 notebook's transformer is enough). Compare the loss on the first tokens of each document, which are the ones the mask changes.
1. **O(1) seeking.** Design a sampler whose state at step $t$ is computed without replaying: a fixed schedule of mixture phases, a hash of `(seed, t)` for the source draw, and per-source cursors from prefix counts. Its stream will differ from `MixtureSampler`'s, so verify it against itself: iterating to step $t$ and seeking to step $t$ must agree at a thousand random steps.
1. **World-size invariance.** Implement a loader under both designs of Section 6.1 and resume a simulated run at step 500 with the world size changed from 8 to 4. Which design gives the same batches, and what does the other one skip or repeat?
1. **A removal request.** Pick a document in the notebook's corpus and carry out the three steps of Section 8.3: locate it, cut a new version without it, and list the versions (and, hypothetically, the models) affected. How long did each step take, and which would have been impossible without the index?

## Further Reading

### Formats, loaders and packing

- [Datasets: A Community Library for Natural Language Processing](https://arxiv.org/abs/2109.02846) {cite}`lhoest2021datasets` - the Arrow-backed library and hub conventions
- [Megatron-LM](https://arxiv.org/abs/1909.08053) {cite}`shoeybi2019megatron` - the indexed binary dataset and blended sampler used by many training codebases
- [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) {cite}`ding2024fewer` - best-fit packing and its measured effects
- [In-Context Pretraining](https://arxiv.org/abs/2310.10638) {cite}`shi2024incontext` - choosing which documents share a sequence
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) {cite}`muennighoff2023scaling` - epoch caps
- [DoReMi](https://arxiv.org/abs/2305.10429) {cite}`xie2023doremi` - learned mixture weights
- [mT5](https://arxiv.org/abs/2010.11934) {cite}`xue2021mt5` - temperature sampling across languages

### Documentation, auditing and governance

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) {cite}`gebru2021datasheets` and [Data Cards](https://arxiv.org/abs/2204.01075) {cite}`pushkarna2022data` - the documentation templates
- [What's In My Big Data?](https://arxiv.org/abs/2310.20707) {cite}`elazar2024whats` - inspection tooling and what it found in the public corpora
- [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) {cite}`dodge2021documenting` - C4 audited after the fact
- [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787) {cite}`longpre2023data` - licence tracking across repackaged datasets
- [The Secret Sharer](https://arxiv.org/abs/1802.08232) {cite}`carlini2019secret` - canaries for measuring memorisation
- [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149) {cite}`carlini2024poisoning` - why distributed datasets need content hashes
- [The Pile](https://arxiv.org/abs/2101.00027) {cite}`gao2020pile`, [Dolma](https://arxiv.org/abs/2402.00159) {cite}`soldaini2024dolma` and [FineWeb](https://arxiv.org/abs/2406.17557) {cite}`penedo2024fineweb` - corpora released with their documentation, statistics and code

### Tools

- [Hugging Face Hub datasets](https://huggingface.co/docs/hub/datasets) - versioned hosting with Parquet conversion and the dataset viewer
- [Mosaic Streaming](https://github.com/mosaicml/streaming) - MDS shards with deterministic resumable shuffling
- [WebDataset](https://github.com/webdataset/webdataset) - tar-shard streaming
- [DVC](https://dvc.org/) and [lakeFS](https://lakefs.io/) - content-addressed data versioning with lineage
- [Croissant](https://github.com/mlcommons/croissant) - machine-readable dataset metadata
- [Great Expectations](https://greatexpectations.io/) - data contracts as tests

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
