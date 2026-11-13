# Dataset Management and Versioning

A pretraining corpus is not a file. It is tens of thousands of shards, produced by a pipeline from sources that change, consumed by training runs that must be reproducible, audited after the fact for what they contained, and edited when something has to come out. This chapter covers the engineering that makes that manageable: how the data is laid out on disk, how a version is named and verified, how documents are tokenized, packed and mixed into training batches, how a run resumes without repeating or skipping data, and how a dataset is documented and governed over its life. Chapter 3 acquired the text and Chapter 4 cleaned it; this chapter turns the result into an artifact that a training job can depend on.

(sec-ch5-learning-objectives)=

## Learning Objectives

By the end of this chapter, you will understand:

- The storage formats used for training data (JSON Lines, Parquet and Arrow, tokenized binaries), how each is written and read, and when each is the right choice; and to describe the streaming shard formats, WebDataset tar shards and Mosaic's MDS
- How a dataset version is defined by content hashes and a manifest, and how to diff and verify versions
- How documents are tokenized, packed into fixed-length sequences and masked so that unrelated documents do not attend to each other
- How sources are mixed by weight and temperature, with epoch caps, and how a sampler is made deterministic and resumable
- How data is sharded across many training processes without duplication, and why the global batch should not depend on the number of processes
- What statistics, tests and documents accompany a dataset version, and how removal requests are handled

(sec-ch5-hands-on-learning)=

## Hands-On Learning

The accompanying notebook, [Dataset Management Demo](examples/dataset_management_demo.ipynb), builds a small corpus into a versioned, tokenized training dataset and checks every step with an assertion:

- Writing JSON Lines shards and a tokenized binary with a document index, and reading both back ({ref}`nb-dm-formats`)
- Building a manifest of content hashes, computing a version root, and diffing two versions ({ref}`nb-dm-versioning`)
- Packing documents into sequences, measuring truncation and padding, and building the document attention mask ({ref}`nb-dm-packing`)
- A deterministic, resumable mixture sampler with temperature and epoch caps ({ref}`nb-dm-sampler`)
- Sharding across ranks and showing that the global batch is independent of the world size ({ref}`nb-dm-distributed`)
- Dataset statistics, a data contract, and a data card generated from the manifest ({ref}`nb-dm-governance`)

Everything runs on a CPU in seconds and uses only the standard library, NumPy, PyArrow for the Parquet and Arrow files and, for the plots, Matplotlib.

(sec-ch5-chapter-overview)=

## Chapter Overview

Think of a training dataset the way a software team thinks of a release. Source code is edited constantly, but what ships is a build: a fixed set of files with a version number, produced by a pinned build script, with a changelog and a way to verify that the bytes you downloaded are the bytes that were released. Nobody debugs a production incident against "whatever was in the repository on Tuesday".

**Storage formats** are the file types of the build. Text is kept as compressed JSON Lines or Parquet for inspection and reprocessing, and as flat token arrays for training, because a training loop that parses JSON on every step is wasting the accelerator.

**Versioning** gives the build its number. A version is the set of content hashes of its shards; two versions can be diffed shard by shard, and a downloaded copy can be checked against the manifest. Nothing in a released version is ever modified in place.

**Tokenization and packing** turn documents into the fixed-length sequences the model consumes. Documents are rarely the right length, so they are concatenated and cut, or fitted into sequences like boxes into a container, and the model is told where one document ends and the next begins.

**Mixing and sampling** decide which documents, in which order, from which sources. The sampler is a pure function of a seed and a step number, so a run that crashes at step 40,000 restarts at step 40,001 with exactly the data it would have seen.

**Distributed loading** splits the work across hundreds or thousands of processes without two of them reading the same shard, and without the data order changing when the number of processes changes.

**Monitoring and governance** keep the record: how many tokens from each source, what the tests checked, who may use the dataset, and how a document is found and removed when someone asks.

The theme is that reproducibility is cheap when it is designed in and impossible to retrofit. A dataset built by hand in a notebook, shuffled with an unseeded call and written over an older copy, cannot be reproduced, diffed, or audited, and every experiment trained on it is uninterpretable.

(sec-ch5-dataset-build-artifact)=

## 1. The Dataset as a Build Artifact

(sec-ch5-reproducibility-requirements)=

### 1.1 What has to be reproducible

Three distinct things are called "the dataset" and each needs its own handling:

- **The sources**: crawl snapshots, dumps and repositories, as acquired in Chapter 3. They are large, mostly immutable once fetched, and recorded by provenance manifest.
- **The processed corpus**: the output of the Chapter 4 pipeline, a set of cleaned document shards with metadata. This is what is versioned, documented and released.
- **The training stream**: the sequence of token batches a particular run consumed, which depends on the corpus version, the tokenizer, the packing method, the mixture weights, the seed and the number of steps. It is not stored; it is recomputed from those inputs.

Reproducing a model means reproducing the third from the second, which means the second has to be fixed and the transformation has to be deterministic. The practical rule used by every serious training codebase: a run's configuration names a corpus version by hash, a tokenizer by hash, and a data seed, and the loader is a pure function of those.

(sec-ch5-scale-problem)=

### 1.2 The scale of the problem

A 15-trillion-token corpus is roughly 60 TB of UTF-8 text, 30 TB as 16-bit token ids, or 60 TB as the 32-bit ids that vocabularies above 65,536 entries need. Stored as 100 MB shards, it is a few hundred thousand files. A training run over it with a 4-million-token global batch takes nearly four million steps, each of which must fetch the right batch in well under the time the accelerators take to process the previous one, from storage that is usually a network object store rather than a local disk.

Those numbers drive the design choices in the rest of the chapter: shards rather than files small enough to list, token arrays rather than text that needs parsing, indexes that locate a document without scanning, and sampling that computes the next batch's location arithmetically rather than by maintaining a queue.

(sec-ch5-storage-formats-layout)=

## 2. Storage Formats and Layout

(sec-ch5-formats-used)=

### 2.1 The formats in use

| Format                       | Unit            | Used for                                | Strengths                                           | Weaknesses                                        |
| ---------------------------- | --------------- | --------------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| JSON Lines (gzip or zstd)    | One document per line | Pipeline input and output, inspection | Human-readable, streamable, any tool reads it        | No random access, slow to parse, no schema        |
| Parquet {cite}`apache2026parquet` | Column chunks   | Processed corpora, metadata queries     | Columnar compression, schema, predicate push-down    | Needs a library, row access is by row group       |
| Tar shards (WebDataset {cite}`aizman2020high`) | Files in a tar  | Streaming from object stores            | Sequential reads, any payload type, simple           | No random access, no schema                       |
| Tokenized binary plus index  | Token arrays    | Training input                          | Zero parsing, memory-mapped random access            | Tied to one tokenizer, opaque to inspection       |
| Arrow (memory-mapped) {cite}`apache2026arrow` | Record batches  | Local caches (Hugging Face `datasets`)  | Zero-copy access, typed columns                      | Uncompressed by default; not an archival format             |

Most pipelines use two of these: a text format for everything up to and including the released corpus, and a tokenized binary for training. The text format is what gets versioned and documented; the binary is derived from it and the tokenizer, and is rebuilt rather than versioned.

The Hugging Face hub converts hosted corpora to Parquet for its viewer and `datasets` streams Parquet, JSON Lines and other formats from the hub without downloading everything {cite}`lhoest2021datasets`. The Megatron-LM codebase {cite}`shoeybi2019megatron` (the paper is about model parallelism; the data format lives in the repository), GPT-NeoX and their descendants train from an indexed binary format, a flat array of token ids alongside an array of document offsets. Mosaic's streaming format and WebDataset shards exist for the case where training reads directly from object storage and sequential throughput matters more than random access.

A Parquet file is a sequence of row groups, each holding one compressed chunk per column, followed by a footer with the schema, the location of every column chunk and per-chunk statistics such as each column's minimum and maximum; the footer is written last so that the file can be written in one pass {cite}`apache2026parquet`. A reader opens the footer first, and two savings follow. Column projection reads only the chunks of the columns asked for, so a query over `num_tokens` never decompresses `text`. Predicate push-down skips any row group whose statistics show it cannot match. The listing writes 1,000 documents in four row groups and reads the long ones back: the footer rules out three of the four row groups before any data is read. The same table written as an Arrow IPC file is the layout Hugging Face `datasets` caches on disk: memory-mapping it gives typed columns with no decoding step {cite}`apache2026arrow`.

```python
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# 1,000 documents written in order of length, as a build that sorts by length would write them
n = 1000
table = pa.table({
    "id": [f"doc{i:04d}" for i in range(n)],
    "source": ["code" if i % 4 == 0 else "web" for i in range(n)],
    "num_tokens": [10 + i for i in range(n)],
    "text": [" ".join(["word"] * (10 + i)) for i in range(n)],
})

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "part-00000.parquet"
    pq.write_table(table, path, row_group_size=250, compression="zstd")

    # The footer alone: schema, row groups, and per-column min and max for every row group
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    col = pf.schema_arrow.get_field_index("num_tokens")
    bounds = [(meta.row_group(g).column(col).statistics.min, meta.row_group(g).column(col).statistics.max)
              for g in range(meta.num_row_groups)]
    print(f"{meta.num_rows} rows, {meta.num_row_groups} row groups, {meta.num_columns} columns")
    print("num_tokens min and max per row group:", bounds)

    # Predicate push-down by hand: only row groups whose range can hold num_tokens >= 900 are read,
    # and only two of the four columns (column projection)
    wanted = [g for g, (lo, hi) in enumerate(bounds) if hi >= 900]
    long_docs = pf.read_row_groups(wanted, columns=["id", "num_tokens"])
    long_docs = long_docs.filter(pa.compute.greater_equal(long_docs["num_tokens"], 900))
    print(f"row groups read: {wanted}, columns read: {long_docs.column_names}, rows kept: {long_docs.num_rows}")
    assert wanted == [3] and long_docs.num_rows == 110
    # The library does the same from a filter expression
    assert pq.read_table(path, columns=["id"], filters=[("num_tokens", ">=", 900)]).num_rows == 110

    # Arrow IPC file: memory-mapped, read without decoding or copying the column buffers
    arrow_path = Path(tmp) / "cache.arrow"
    with pa.OSFile(str(arrow_path), "wb") as sink, pa.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    with pa.memory_map(str(arrow_path), "r") as source:
        cached = pa.ipc.open_file(source).read_all()
        assert cached.equals(table)
        print(f"Arrow cache: {cached.num_rows} rows, {cached.nbytes:,} bytes of buffers mapped from disk")
```

% output

```text
1000 rows, 4 row groups, 4 columns
num_tokens min and max per row group: [(10, 259), (260, 509), (510, 759), (760, 1009)]
row groups read: [3], columns read: ['id', 'num_tokens'], rows kept: 110
Arrow cache: 1000 rows, 2,576,750 bytes of buffers mapped from disk
```

(sec-ch5-sharding)=

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

    Each manifest entry records the shard's path, record count, uncompressed and
    compressed byte counts, and the SHA-256 of the uncompressed bytes: the
    shard's content identity, which later stages verify against.
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
        # filename="" and mtime=0 keep the temporary file name and the time of writing out of the gzip header
        with open(tmp, "wb") as out, gzip.GzipFile(filename="", fileobj=out, mode="wb", compresslevel=6, mtime=0) as f:
            f.write(raw)
        tmp.rename(path)  # a shard exists only once it is complete
        manifest.append(
            {
                "path": path.name,
                "records": len(batch),
                "bytes_uncompressed": len(raw),
                "bytes_compressed": path.stat().st_size,
                "sha256": hashlib.sha256(raw).hexdigest(),  # content identity: the uncompressed bytes
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
        raw = gzip.decompress((out_dir / entry["path"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise IOError(f"{entry['path']} does not match its manifest hash")
        for line in raw.split(b"\n")[:-1]:
            yield json.loads(line)


import tempfile

docs = [{"id": f"doc{i}", "source": "demo", "text": f"Document number {i}. " * (i + 1)} for i in range(10)]
with tempfile.TemporaryDirectory() as tmp:
    manifest = write_jsonl_shards(docs, Path(tmp), docs_per_shard=4)
    assert [m["records"] for m in manifest] == [4, 4, 2]
    assert [d["id"] for d in read_jsonl_shards(Path(tmp), manifest)] == [d["id"] for d in docs]
    assert not list(Path(tmp).glob("*.tmp"))
    # The same documents under another name: same content identity and, with the header fixed, the same bytes
    renamed = write_jsonl_shards(docs, Path(tmp) / "renamed", docs_per_shard=4, prefix="part")
    assert [m["sha256"] for m in renamed] == [m["sha256"] for m in manifest]
    assert (Path(tmp) / "renamed" / "part_00000.jsonl.gz").read_bytes() == (Path(tmp) / "shard_00000.jsonl.gz").read_bytes()
```

The `filename=""` and `mtime=0` arguments matter more than they look. A gzip header records a modification time, which `gzip.open` fills with the current time, and a file name, which `GzipFile` takes from the file it writes to (here the temporary name), so the same documents compressed a second apart or under another shard name produce different bytes. Fixing both makes the bytes repeatable with one compressor, but not everywhere: the compressed bytes also depend on the compression level and on the compression library. The DEFLATE specification defines the compressed format, not the algorithm that produces it, and says a compressor need not follow its suggested method to be compliant {cite}`deutsch1996deflate`; zlib-ng, a faster drop-in replacement for zlib, ships its own deflate algorithms {cite}`zlibng2026`. The same input can therefore compress to different bytes under a different library build. A shard's identity has to be a function of its content alone, or every rebuild looks like a new version, so the manifest records the SHA-256 of the uncompressed bytes and the reader decompresses before it checks; the gzip settings stay fixed so that a rebuild with the same library also reproduces the files. Zstandard frames carry no timestamp; tar and zip archives do, and need the same treatment.

(sec-ch5-tokenized-binaries-document-index)=

### 2.3 Tokenized binaries and the document index

Training reads tokens, not text. Tokenizing on the fly costs CPU time per step that is better spent once, and makes the stream depend on the tokenizer library's version at run time. The standard layout, shared in spirit by Megatron-LM's indexed dataset and the `.bin` files that nanoGPT-style codebases use, is:

- A flat array of token ids, `uint16` for vocabularies of at most 65,536 tokens (ids 0 to 65,535) and `uint32` above, for all documents in the shard concatenated with an end-of-document token after each.
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

    def __init__(self, path: Path, tokenizer_hash: str):
        self.path = path
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["tokenizer_hash"] != tokenizer_hash:
            raise ValueError(f"{path.name} was built with tokenizer {meta['tokenizer_hash']}, not {tokenizer_hash}")
        self.dtype = np.dtype(meta["dtype"])
        self.tokenizer_hash = meta["tokenizer_hash"]
        self.offsets = np.load(path.with_suffix(".idx.npy"))
        self.tokens = np.memmap(path, dtype=self.dtype, mode="r")

    @classmethod
    def write(cls, path: Path, token_docs, tokenizer_hash: str, vocab_size: int):
        dtype = np.uint16 if vocab_size <= 2**16 else np.uint32
        offsets = [0]
        chunks = []
        for toks in token_docs:
            arr = np.asarray(list(toks) + [EOD], dtype=dtype)
            chunks.append(arr)
            offsets.append(offsets[-1] + len(arr))
        np.concatenate(chunks).tofile(path)
        np.save(path.with_suffix(".idx.npy"), np.asarray(offsets, dtype=np.int64))
        path.with_suffix(".json").write_text(
            json.dumps({"dtype": np.dtype(dtype).name,
                        "tokenizer_hash": tokenizer_hash, "num_docs": len(chunks)})
        )
        return cls(path, tokenizer_hash)

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
    try:
        TokenShard(Path(tmp) / "demo.bin", tokenizer_hash="toy-v2")
        raise AssertionError("a binary was opened with the wrong tokenizer")
    except ValueError:
        pass
```

The tokenizer hash is the part people skip and regret. Two tokenizers with the same name and vocabulary size but different normalization or merge order produce different id streams, and a model evaluated with the wrong one is silently broken. Hash the tokenizer's serialized files and store the hash with every binary that depends on it.

The special-token policy is part of the same contract. The layout above puts an end-of-document token after each document, as GPT-3 did; many tokenizers also define a beginning-of-sequence token, and a build can put one before every document as well. Either way the rule is written into the binary's sidecar and applied identically at inference, because a model trained with a BOS token before every document and prompted without one sees a context it never saw in training. Two details follow from the policy. The BOS token is never a prediction target: it always follows an EOD, so there is nothing to learn, and the loss mask drops it. And the overhead is one or two tokens per document: for documents averaging 60 tokens, EOD alone is 1.62% of the stream and BOS plus EOD 3.19%, which the listing measures.

```python
import numpy as np

BOS, EOD = 1, 0


def with_special_tokens(token_docs, policy: str) -> np.ndarray:
    """Concatenate documents under a special-token policy: "eod" puts EOD after each document,
    "bos+eod" also puts BOS before each one."""
    stream = []
    for doc in token_docs:
        if policy == "bos+eod":
            stream.append(BOS)
        stream.extend(doc)
        stream.append(EOD)
    return np.asarray(stream)


def loss_mask(stream: np.ndarray) -> np.ndarray:
    """One flag per next-token target stream[1:]: BOS is never a prediction target, since it
    follows EOD by rule and carries nothing to learn."""
    return stream[1:] != BOS


rng = np.random.default_rng(0)
docs = [list(rng.integers(2, 1000, size=n)) for n in rng.geometric(1 / 60, size=2000)]
content = sum(len(d) for d in docs)
for policy in ("eod", "bos+eod"):
    stream = with_special_tokens(docs, policy)
    special = 1 - content / stream.size
    print(f"{policy:8s} {stream.size:,} tokens, {special:.2%} special, {int(loss_mask(stream).sum()):,} trained targets")
stream = with_special_tokens(docs, "bos+eod")
starts = np.flatnonzero(stream == BOS)
assert starts.size == len(docs) and np.all(stream[starts[1:] - 1] == EOD)  # every BOS follows an EOD
assert not loss_mask(stream)[starts[1:] - 1].any()  # and is never trained on as a target
```

% output

```text
eod      123,439 tokens, 1.62% special, 123,438 trained targets
bos+eod  125,439 tokens, 3.19% special, 123,439 trained targets
```

```{admonition} Formats on Disk
:class: tip
The notebook section {ref}`nb-dm-formats` writes a corpus as JSON Lines shards and as a tokenized binary with offsets, verifies both against their manifests, and compares the bytes per token of each layout.
```

(sec-ch5-codecs)=

### 2.4 Choosing a codec

Training data is written once and read many times, so a codec's decompression speed matters more than its compression speed, and its ratio matters through storage and network cost. gzip (DEFLATE) is readable by every tool; Zstandard {cite}`collet2021zstd`, LZ4 and Brotli are the usual alternatives, and Zstandard and Brotli offer a range of levels that trade compression time for size. The listing compresses 3.4 MB of synthetic text, Zipf-distributed words from a 20,000-word vocabulary, with each codec PyArrow ships. Ratios depend only on the codec and its version; throughputs depend on the machine, so the output below shows the shape of one run. On that run Zstandard at level 3 matched gzip's ratio within 3% while compressing 20 times as fast and decompressing twice as fast, and level 19 gave the best ratio at a decompression speed no worse than level 3. That is the case for `.jsonl.zst` in the shard names of {ref}`sec-ch5-sharding`: a build compresses once at a high level, and every later read pays only the fast decode.

```python
import time

import numpy as np
import pyarrow as pa

# Sizes and timings depend on the machine and the library builds; the order of the codecs is the result
rng = np.random.default_rng(0)
vocab = [f"w{i}" for i in range(20_000)]
zipf = rng.zipf(1.2, size=800_000)
text = " ".join(vocab[(z - 1) % len(vocab)] for z in zipf).encode()
results = {}
for name, level in [("lz4", None), ("zstd", 1), ("zstd", 3), ("zstd", 19), ("gzip", 6), ("brotli", 9)]:
    codec = pa.Codec(name, compression_level=level)
    t0 = time.perf_counter()
    packed = codec.compress(text, asbytes=True)
    t1 = time.perf_counter()
    assert codec.decompress(packed, decompressed_size=len(text), asbytes=True) == text
    t2 = time.perf_counter()
    label = name if level is None else f"{name}-{level}"
    results[label] = len(text) / len(packed)
    print(f"{label:9s} ratio {len(text) / len(packed):.3g}, compress {len(text) / (t1 - t0) / 1e6:.0f} MB/s, "
          f"decompress {len(text) / (t2 - t1) / 1e6:.0f} MB/s")
assert results["zstd-19"] > results["zstd-3"] > results["lz4"]
```

% output-shape

```text
lz4       ratio 1.73, compress 208 MB/s, decompress 636 MB/s
zstd-1    ratio 2.76, compress 162 MB/s, decompress 499 MB/s
zstd-3    ratio 2.77, compress 105 MB/s, decompress 379 MB/s
zstd-19   ratio 3.25, compress 1 MB/s, decompress 598 MB/s
gzip-6    ratio 2.85, compress 6 MB/s, decompress 200 MB/s
brotli-9  ratio 3.02, compress 4 MB/s, decompress 224 MB/s
```

(sec-ch5-versioning)=

## 3. Versioning

(sec-ch5-version-set-hashes)=

### 3.1 A version is a set of hashes

Git works for source code because a commit is a hash of a tree of hashes of file contents. The same structure works for datasets at any scale, and the tools that do it well expose the same model, whether they store the objects themselves by content hash (DVC {cite}`iterative2026dvc`, Git LFS, the Hugging Face hub's storage layer) or keep Git-like commits and branches as metadata over ordinary object storage (lakeFS {cite}`treeverse2026lakefs`):

- Each shard is content-addressed: its identity is the hash of its bytes, and it is stored under that hash. Publishing the hashes is also the defense {cite:t}`carlini2024poisoning` propose against split-view poisoning, in which the content behind a dataset's URLs changes after the dataset was indexed: a client that checks the hashes sees the data the maintainers saw.
- A **manifest** lists the shards of a version by path, hash and size. The manifest is small enough to keep in Git.
- The **version root** is the hash of the manifest, or a Merkle root over the shard hashes, so that one short string identifies the whole dataset.

A Merkle root does more than name the version. Pair the shard hashes, hash each pair, and repeat up to a single root: changing any one shard changes every hash on its path and so the root, and anyone who holds the root can check that a shard belongs to the version from the shard's hash and one sibling hash per level, four hashes for ten shards and about 20 for a million, without the rest of the manifest. Prefixing leaves and inner nodes with different bytes keeps a leaf from being passed off as an inner node.

```python
import hashlib


def _leaf(shard_hash: str) -> bytes:
    return hashlib.sha256(b"\x00" + bytes.fromhex(shard_hash)).digest()  # 0x00: a leaf


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()  # 0x01: an inner node


def merkle_levels(shard_hashes):
    """Every level of the tree, leaves first; an unpaired last node is carried up unchanged."""
    levels = [[_leaf(h) for h in shard_hashes]]
    while len(levels[-1]) > 1:
        level = levels[-1]
        up = [_node(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            up.append(level[-1])
        levels.append(up)
    return levels


def merkle_root(shard_hashes) -> str:
    return merkle_levels(shard_hashes)[-1][0].hex()


def inclusion_proof(shard_hashes, index: int):
    """The sibling hashes from leaf `index` to the root, each with the side it sits on."""
    proof = []
    for level in merkle_levels(shard_hashes)[:-1]:
        sibling = index ^ 1
        if sibling < len(level):
            proof.append(("left" if sibling < index else "right", level[sibling].hex()))
        index //= 2
    return proof


def verify_inclusion(shard_hash: str, proof, root: str) -> bool:
    h = _leaf(shard_hash)
    for side, sibling in proof:
        h = _node(bytes.fromhex(sibling), h) if side == "left" else _node(h, bytes.fromhex(sibling))
    return h.hex() == root


shards = [hashlib.sha256(f"shard {i}".encode()).hexdigest() for i in range(10)]
root = merkle_root(shards)
proof = inclusion_proof(shards, 6)
print(f"root of 10 shards: {root[:16]}..., proof for shard 6: {len(proof)} hashes")
assert verify_inclusion(shards[6], proof, root)
assert not verify_inclusion(shards[7], proof, root)  # the proof is for shard 6 only

# Change one shard: the root changes, and the old proof no longer verifies against the new root
edited = list(shards)
edited[6] = hashlib.sha256(b"shard 6, refiltered").hexdigest()
assert merkle_root(edited) != root and not verify_inclusion(shards[6], proof, merkle_root(edited))
# Order matters: the root commits to the sequence of shards, not just their set
assert merkle_root(shards[::-1]) != root
print(f"after editing shard 6: {merkle_root(edited)[:16]}..., old proof valid: {verify_inclusion(shards[6], proof, merkle_root(edited))}")
```

% output

```text
root of 10 shards: 56270bb62a689fd8..., proof for shard 6: 4 hashes
after editing shard 6: 667b16bc86d691d8..., old proof valid: False
```

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

(sec-ch5-immutability-naming)=

### 3.2 Immutability and naming

A released version is never modified. If a shard is found to be corrupt or to contain something that must be removed, a new version is cut without it, and the old version's manifest is kept so that models trained on it can be audited. The two conventions that make this workable:

- **Human-readable tags** on top of hashes: `fineweb-edu-v1.2.0` rather than a 64-character string. A tag is a pointer to a root hash and is itself recorded in the version history, as a Git tag is.
- **Semantic-style versioning** for the tags, with an agreed meaning: a patch change removes or corrects shards without changing the pipeline; a minor change reruns the pipeline with changed parameters; a major change alters the sources or the schema. The point is less the numbers than having a rule for which experiments remain comparable.

A dataset tag is the thing a training configuration references. A configuration that says `data: fineweb-edu-v1.2.0` with the tag resolved to a root hash at launch and recorded in the run's metadata is reproducible; one that says `data: /mnt/data/fineweb-edu/latest` is not.

The same rule applies to a corpus pulled from the Hugging Face Hub, where every dataset is a Git repository. Its branches and tags move when the maintainers push, so a configuration names the commit: the download functions take a `revision` argument that accepts a branch, a tag or a commit hash, and a commit must be given as the full 40-character hash, not the 7-character abbreviation {cite}`huggingface2026hubdownload`. A launcher can enforce this before any data is read, as the listing does; the resolved hash then goes into the run's metadata beside the corpus root.

```python
import re

FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def check_data_pin(config: dict) -> list:
    """Problems with a run configuration's data reference: the revision must be a full commit hash."""
    revision = config.get("revision", "main")
    if FULL_COMMIT.fullmatch(revision):
        return []
    if re.fullmatch(r"[0-9a-f]{7,39}", revision):
        return [f"{revision!r} is an abbreviated hash; the Hub needs all 40 characters"]
    return [f"{revision!r} is a branch or tag, which can move; pin the commit it points to"]


configs = {
    "pinned": {"repo_id": "org/corpus", "revision": "877b84a8f93f2d619faa2a6e514a32beef88ab0a"},
    "branch": {"repo_id": "org/corpus", "revision": "main"},
    "short": {"repo_id": "org/corpus", "revision": "877b84a"},
    "unset": {"repo_id": "org/corpus"},
}
for name, cfg in configs.items():
    print(f"{name:6s} -> {check_data_pin(cfg) or 'ok'}")
assert check_data_pin(configs["pinned"]) == [] and all(check_data_pin(configs[n]) for n in ("branch", "short", "unset"))
```

% output

```text
pinned -> ok
branch -> ["'main' is a branch or tag, which can move; pin the commit it points to"]
short  -> ["'877b84a' is an abbreviated hash; the Hub needs all 40 characters"]
unset  -> ["'main' is a branch or tag, which can move; pin the commit it points to"]
```

(sec-ch5-lineage)=

### 3.3 Lineage

Each version records how it was made: the versions of its inputs, the pipeline code commit, the configuration, and the tokenizer if it is a tokenized build. This forms a directed graph from raw sources through cleaned corpora to tokenized datasets to models, and it is the graph that answers the questions that come up later: which models were trained on data that included this document, which runs would change if this filter were fixed, and whether two experiments differ in their data or only in their seeds. Tools such as DVC store the graph as pipeline stages with declared inputs and outputs; others store it as metadata on each version. Either way the graph is only as good as the discipline of never producing a version outside it.

A lineage record is small. The listing below writes one per version: the output's root hash, the root of each input, the pipeline commit, the configuration and the tokenizer hash, all covered by a hash of the record. Verification recomputes that hash and checks that every input is a known version; the graph query that removal needs ({ref}`sec-ch5-removal`) is a walk from an input to everything built from it.

```python
import hashlib
import json


def lineage_record(output_root: str, inputs: dict, code_commit: str, config: dict, tokenizer_hash=None) -> dict:
    """How a version was made; the record hash covers every field, so an edited record is detected."""
    record = {"output": output_root, "inputs": dict(sorted(inputs.items())), "code": code_commit,
              "config": config, "tokenizer": tokenizer_hash}
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return {**record, "record_hash": hashlib.sha256(body.encode()).hexdigest()}


def verify_record(record: dict, known_roots: set) -> list:
    """Problems with a record: a hash that does not match its fields, or an input that is not a known version."""
    fields = {k: v for k, v in record.items() if k != "record_hash"}
    problems = []
    if lineage_record(**{"output_root": fields["output"], "inputs": fields["inputs"], "code_commit": fields["code"],
                         "config": fields["config"], "tokenizer_hash": fields["tokenizer"]})["record_hash"] != record["record_hash"]:
        problems.append("record hash does not match its contents")
    problems += [f"unknown input {name}={root}" for name, root in record["inputs"].items() if root not in known_roots]
    return problems


def descendants(records, root: str) -> set:
    """Every version built, directly or indirectly, from `root`."""
    found, frontier = set(), {root}
    while frontier:
        frontier = {r["output"] for r in records if set(r["inputs"].values()) & frontier} - found
        found |= frontier
    return found


raw = {"cc-2024-10": "r-cc10", "wiki-2024-03": "r-wiki"}
cleaned = lineage_record("c-v1.2.0", raw, code_commit="9f3c2e1", config={"min_words": 50, "lang": "en"})
tokens = lineage_record("t-v1.2.0", {"corpus": "c-v1.2.0"}, code_commit="9f3c2e1", config={"seq_len": 4096},
                        tokenizer_hash="tok-5d41")
model = lineage_record("m-run17", {"data": "t-v1.2.0"}, code_commit="a77b0c4", config={"seed": 17})
records = [cleaned, tokens, model]
known = {"r-cc10", "r-wiki", "c-v1.2.0", "t-v1.2.0"}

assert all(verify_record(r, known) == [] for r in records)
tampered = {**tokens, "config": {"seq_len": 8192}}
assert verify_record(tampered, known) == ["record hash does not match its contents"]
affected = descendants(records, "r-wiki")
print("record for t-v1.2.0:", json.dumps({k: tokens[k] for k in ("inputs", "code", "tokenizer")}))
print("versions built from wiki-2024-03:", sorted(affected))
assert affected == {"c-v1.2.0", "t-v1.2.0", "m-run17"}
```

% output

```text
record for t-v1.2.0: {"inputs": {"corpus": "c-v1.2.0"}, "code": "9f3c2e1", "tokenizer": "tok-5d41"}
versions built from wiki-2024-03: ['c-v1.2.0', 'm-run17', 't-v1.2.0']
```

```{admonition} Versions and Diffs
:class: note
The notebook section {ref}`nb-dm-versioning` builds a manifest for the corpus written earlier, rebuilds it with one document changed and one removed, and shows that the version root changes while the diff names exactly the affected shards. It then verifies a downloaded copy against the manifest and detects a corrupted shard.
```

(sec-ch5-splits-leakage)=

### 3.4 Splits and leakage

A version also fixes its splits: the training set, a validation set used to choose checkpoints and hyperparameters, and a held-out test set opened only for final numbers. A split is useful only if no document in it, and no near-copy of one, is also in training; otherwise the validation loss measures memorization. Such leakage is common in practice: {cite:t}`lee2022deduplicating` found train-test overlap affecting over 4% of the validation sets of standard datasets.

Versioning creates a second route for leakage. If each version draws its validation set afresh at random, documents that were held out in one version are trained on in the next, and a model fine-tuned from the earlier one has seen them. The fix is to assign the split by a hash of a stable key, so that a document keeps its split in every version, and to make that key the near-duplicate cluster from deduplication rather than the document, so that copies of a validation document cannot sit in training. The listing removes 5% of a 100,000-document version and adds 8,000 documents: a fresh 1% random split sends 94.2% of the old validation documents into the new training set, and the hash split sends none.

```python
import hashlib

import numpy as np


def split_of(key: str, salt: str = "split-v1", validation: float = 0.01) -> str:
    """Stable split by a hash of the key: the same key lands in the same split in every version."""
    u = int(hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()[:15], 16) / 16**15
    return "validation" if u < validation else "train"


rng = np.random.default_rng(1)
v1 = [f"doc{i}" for i in range(100_000)]
v2 = [d for d in v1 if rng.random() > 0.05] + [f"new{i}" for i in range(8000)]  # 5% removed, 8,000 added

# A fresh random 1% split per version puts most of v1's validation set into v2's training set
val_v1 = set(rng.choice(v1, size=len(v1) // 100, replace=False))
val_v2 = set(rng.choice(v2, size=len(v2) // 100, replace=False))
train_v2_random = set(v2) - val_v2
leak_random = len(val_v1 & train_v2_random) / len(val_v1)

# A hash split by key keeps every surviving validation document in validation
val_v1_hash = {d for d in v1 if split_of(d) == "validation"}
train_v2_hash = {d for d in v2 if split_of(d) == "train"}
leak_hash = len(val_v1_hash & train_v2_hash) / len(val_v1_hash)
print(f"v1 validation documents in v2 training: random split {leak_random:.1%}, hash split {leak_hash:.1%}")
assert leak_random > 0.9 and leak_hash == 0.0

# Near-duplicates must share a split: key the hash on the duplicate cluster, not the document
cluster = {"doc7": "c42", "doc7-mirror": "c42"}
assert split_of(cluster["doc7"]) == split_of(cluster["doc7-mirror"])
```

% output

```text
v1 validation documents in v2 training: random split 94.2%, hash split 0.0%
```

(sec-ch5-tokenization-packing)=

## 4. Tokenization and Packing

(sec-ch5-documents-sequences)=

### 4.1 From documents to sequences

A transformer trains on sequences of a fixed length $n$ (2,048 to 128,000 tokens in current models), and documents are whatever length they are. Three ways to reconcile them:

- **Concatenate and chunk.** Join all documents with an end-of-document token between them into one stream, and cut it into consecutive windows of length $n$. No padding, no waste apart from the final partial window, which is dropped, and the method the GPT-3 paper describes {cite}`brown2020language`: full 2,048-token sequences of packed documents separated by an end-of-text token, with no special masking. Most pretraining codebases default to it. The cost: a document is split across windows wherever the cut falls, and a window usually contains the tail of one document and the head of the next.
- **Pad.** One document per sequence, padded to $n$. No cross-document contamination, but with a long-tailed length distribution most of each sequence is padding, which is compute spent on nothing. Used for fine-tuning on short examples, rarely for pretraining.
- **Pack.** Fit whole documents into sequences like boxes into a container (a bin-packing problem) so that most sequences are nearly full and documents are rarely cut. Best-fit packing {cite}`ding2024fewer` first splits any document longer than $n$ into pieces of at most $n$, then places each piece into the sequence with the least remaining room that still fits it, processing pieces longest first. It reduced truncation substantially relative to chunking, with measurable gains in reading comprehension and faithfulness and less hallucination, at the cost of a sort and an index. Which documents share a sequence can itself be chosen: {cite:t}`shi2024incontext` order the corpus so that each context holds related documents, found by nearest-neighbor search, and report better language modeling than with randomly concatenated documents.

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
    """Best-fit-decreasing packing: each document gets one EOD at its true end and
    is split into pieces of at most seq_len tokens; each piece then goes into the
    sequence with the least remaining room that still fits it, longest pieces first."""
    pieces = []
    for d in token_docs:
        d = list(d) + [eod]  # one EOD, at the true end
        for start in range(0, len(d), seq_len):
            pieces.append(d[start : start + seq_len])
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
assert (packed != -1).sum() == sum(len(d) + 1 for d in docs)  # every token kept, one EOD per document
assert (packed == 0).sum() == len(docs)
long_doc, _ = best_fit_pack([list(range(1, 601))], seq_len=256)  # a 600-token document: three pieces, one EOD
assert long_doc.shape[0] == 3 and (long_doc == 0).sum() == 1
```

(sec-ch5-cross-document-attention)=

### 4.2 Cross-document attention

When a sequence holds several documents, causal attention lets a token in the second document attend to the first, which is both a waste (the first document is irrelevant) and a source of the model learning spurious dependencies. The fix is a document mask: a token attends only to earlier tokens of its own document. With a segment id $\operatorname{seg}(i)$ for each position $i$, the mask is $M_{ij} = [j \le i] \cdot [\operatorname{seg}(i) = \operatorname{seg}(j)]$, block-diagonal within the causal triangle. Efficient attention kernels accept this as a list of segment boundaries rather than a full matrix.

```python
import numpy as np


def segment_ids(sequence, eod: int = 0) -> np.ndarray:
    """Document index of each position; the EOD token belongs to the document it ends."""
    ends = np.asarray(sequence) == eod
    return np.concatenate([[0], np.cumsum(ends)[:-1]])


def document_causal_mask(sequence, eod: int = 0, pad: int = -1) -> np.ndarray:
    seg = segment_ids(sequence, eod)
    causal = np.tril(np.ones((len(seg), len(seg)), dtype=bool))
    same_doc = seg[:, None] == seg[None, :]
    valid = np.asarray(sequence) != pad  # padding attends to nothing and is attended by nothing
    return causal & same_doc & valid[:, None] & valid[None, :]


seq = [5, 6, 0, 7, 8, 9, 0, 1]
assert segment_ids(seq).tolist() == [0, 0, 0, 1, 1, 1, 1, 2]
mask = document_causal_mask(seq)
assert mask[4, 3] and mask[4, 4]  # within document 1, causal
assert not mask[4, 1]  # document 1 cannot see document 0
assert not mask[3, 4]  # and not the future
assert mask.sum() == 3 + 2 + 1 + 4 + 3 + 2 + 1 + 1
assert document_causal_mask(seq + [-1, -1]).sum() == mask.sum()  # padding adds no allowed pairs
```

Whether to mask is a choice. Many pretraining runs did not, on the grounds that the kernel was simpler and the effect small; recent ones increasingly do. Llama 3 masks attention between documents in the same sequence and reports that this had limited impact in standard pretraining but was important in continued pretraining on very long sequences {cite}`grattafiori2024llama`. Fine-tuning with packed examples should always mask, because the examples are short and the leakage between them is large.

```{admonition} Packing and Masking
:class: tip
The notebook section {ref}`nb-dm-packing` packs the same corpus by chunking and by best-fit, reports the split and padding rates, plots the distribution of documents per sequence, and renders the document-causal mask for the fullest packed sequence.
```

(sec-ch5-length-bucketing)=

### 4.3 Length bucketing for fine-tuning

Fine-tuning data is the case where padding is the norm: examples are short, of very different lengths, and often kept one per sequence so that the loss can be masked to the response. A batch is padded to its longest example, so a batch drawn at random from a long-tailed length distribution is mostly padding. Length bucketing groups examples of similar length into the same batch and then shuffles the batches, so that the order across steps stays random while the padding within a batch shrinks; the Hugging Face Trainer offers it as `group_by_length`, which groups "samples of roughly the same length together (to minimize padding and be more efficient)" {cite}`huggingface2026trainer`. The listing draws 32,000 example lengths with a median of 147 tokens, batches them 32 at a time, and sorts within pools of 50 batches: padding falls from 75.7% of the batch to 13.0%. The cost is a weak correlation between length and step within a pool, which is why the pool is shuffled before it is cut and the batches after.

```python
import numpy as np


def padding_fraction(lengths, batch_size: int, bucketed: bool, rng) -> float:
    """Fraction of a padded batch that is padding, each batch padded to its longest example."""
    order = rng.permutation(len(lengths))
    if bucketed:  # sort a pool of 50 batches by length, cut it into batches, shuffle the batches
        pool = batch_size * 50
        order = np.concatenate([chunk[np.argsort(lengths[chunk])] for chunk in np.array_split(order, max(1, len(order) // pool))])
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if bucketed:
        rng.shuffle(batches)
    padded = sum(len(b) * lengths[b].max() for b in batches)
    return 1 - lengths.sum() / padded


rng = np.random.default_rng(0)
lengths = np.clip(rng.lognormal(mean=5.0, sigma=0.8, size=32_000).astype(int), 8, 4096)  # fine-tuning examples
random_pad = padding_fraction(lengths, 32, bucketed=False, rng=np.random.default_rng(1))
bucket_pad = padding_fraction(lengths, 32, bucketed=True, rng=np.random.default_rng(1))
print(f"median length {np.median(lengths):.0f}, longest {lengths.max()}")
print(f"padding: random batches {random_pad:.1%}, length-bucketed {bucket_pad:.1%}")
assert bucket_pad < random_pad / 5
```

% output

```text
median length 147, longest 3485
padding: random batches 75.7%, length-bucketed 13.0%
```

(sec-ch5-mixing-sampling)=

## 5. Mixing and Sampling

(sec-ch5-weights-temperature-epochs)=

### 5.1 Weights, temperature and epochs

A corpus has several sources with very different sizes, and the fraction of training tokens drawn from each is a decision, not a consequence of size. Chapter 3 covered how the weights are chosen; here the question is how a sampler realizes them.

Given weights $w_s$ that sum to one, the sampler draws each document's source with probability $w_s$ and then the next unseen document from that source. Two refinements:

- **Temperature** flattens or sharpens the weights: $p_s \propto w_s^{1/\tau}$. With $\tau = 1$ the weights are used as given; $\tau > 1$ upsamples small sources, which is how multilingual models give low-resource languages more than their share: mT5 samples each language in proportion to its number of examples raised to the power 0.3, that is $\tau = 1/0.3 \approx 3.3$ {cite}`xue2021mt5`. The same scheme is used on domain weights.
- **Epoch caps** bound how many times a source is repeated. If source $s$ has $U_s$ tokens and the run needs $D$ tokens, source $s$ is seen $p_s D / U_s$ times; for a fixed compute budget, up to about four epochs of repeated data change the loss negligibly compared with unique data, and beyond that the value of further repetition decays toward zero {cite}`muennighoff2023scaling`. A cap renormalizes the weights of the uncapped sources once a capped one has used its budget.

Learned weights (DoReMi {cite}`xie2023doremi`, which sets static $w_s$ by training a small proxy model with group distributionally robust optimization over the domains) and curricula that change the mixture over training are both consumed by the same sampler, the latter with $w_s$ a function of the step. The cap is stated in tokens; the sampler below counts documents, which production samplers replace with token counts from the index.

(sec-ch5-determinism-resumption)=

### 5.2 Determinism and resumption

The sampler must be a pure function of `(seed, step)`. The reason is resumption: a run that fails at step $t$ and restarts must continue from step $t + 1$ with the data it would have seen, not restart its epoch, and not replay the batch it just trained on. A sampler that keeps a queue or an iterator in memory cannot do this; one that computes the batch at step $t$ arithmetically can.

A pure function is reproducible only while its code is fixed. NumPy's random number policy, NEP 19, keeps the legacy `RandomState` streams fixed but allows changes that alter the streams of the newer `Generator` in feature releases, and names pinning the versions of the whole software stack as the standard practice for bit-for-bit reproducibility {cite}`kern2018nep19`. The NumPy version therefore belongs to a training stream's identity, alongside the corpus version, the tokenizer and the compressor, and the run's configuration records it.

The construction that achieves it: for each source, a seeded permutation of its document indices defines the order within an epoch; a seeded sequence of source draws decides which source supplies each slot; the document for slot $t$ is the next unused index of the chosen source. The class below recomputes all of this from the seed by replaying the $t$ draws, which is $\mathcal{O}(t)$ work; the end of this section derives a construction that needs constant work per step.

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

The `seek` method replays draws, which is fine for millions of steps and a few sources. Constant work per step needs two further constructions, and it is worth seeing why seeding alone does not give it. Seeding each permutation by `(seed, source, epoch)`, as `_permutation` does, makes an epoch's order reproducible without replay, but `permutation` still builds the whole array, $\mathcal{O}(n_{\text{ex}})$ work and memory for a source of $n_{\text{ex}}$ documents, to read one element of it. And the cursor, $\operatorname{drawn}_s(t)$, the number of documents of source $s$ drawn before step $t$, is a sum over $t$ independent source draws: hashing each slot fixes every term, but it does not give the sum without adding up the terms.

- **Within-epoch order: a keyed bijection.** Replace the stored array by a function $\Pi$ on $\{0, \dots, n_{\text{ex}} - 1\}$ that can be evaluated at one point. A Feistel network does this. It splits an integer of an even number of bits into a left and a right half and, in each round, replaces (left, right) by (right, left XOR F(right)) with a keyed hash F. A round is undone from its output whatever F is, so the network is a bijection on its domain for any key. Taking the smallest even bit width that covers $n_{\text{ex}}$ gives a domain of fewer than $4 n_{\text{ex}}$ integers; a value that lands outside the range is encrypted again until it falls back in (cycle-walking). This is still a bijection, because each cycle of the network visits the in-range values in a fixed order and the walk moves each one to the next. Summed over all $n_{\text{ex}}$ starting points the walks touch each point of the domain at most once, so the mean cost is at most the domain size over $n_{\text{ex}}$: below four encryptions per index, and 1.024 for $n_{\text{ex}} = 1{,}000$ with its 1,024-point domain.
- **Cursors: a stride schedule.** Fix the mixture in advance as a repeating pattern of slots, here 20 slots in which each source holds its weight times 20, spread out by weighted round-robin. Then $\operatorname{drawn}_s(t)$ is the number of whole periods before $t$ times the source's slots per period, plus the source's count among the first $t \bmod 20$ slots, read from a table of 21 prefix counts. The document at step $t$ is $\Pi(\operatorname{drawn}_s(t) \bmod n_{\text{ex}})$, with $\Pi$ keyed by the seed, the source and the epoch $\lfloor \operatorname{drawn}_s(t) / n_{\text{ex}} \rfloor$.

Each step is then a table lookup, a division and a few hash evaluations, whatever $t$ is. The price is that the mixture is fixed when the schedule is computed: epoch caps and changes of weight between phases are applied there, and a change of weights mid-run starts a new schedule at a recorded step. A sampler that draws sources at random, like the one above, reaches constant work per step differently: it stores the cursors with each checkpoint and replays only the steps since the last one.

```python
import hashlib


class FeistelPermutation:
    """Keyed bijection on range(n): a balanced Feistel network on 2 * half bits, with cycle-walking."""

    def __init__(self, n: int, key: bytes, rounds: int = 4):
        self.n, self.key, self.rounds = n, key, rounds
        self.half = max(1, ((n - 1).bit_length() + 1) // 2)  # 4**half >= n, and 4**half < 4 * n
        self.mask = (1 << self.half) - 1
        self.evaluations = 0

    def _encrypt(self, x: int) -> int:
        self.evaluations += 1
        left, right = x >> self.half, x & self.mask
        for r in range(self.rounds):
            f = hashlib.blake2b(bytes([r]) + right.to_bytes(8, "little"), key=self.key, digest_size=8)
            left, right = right, left ^ (int.from_bytes(f.digest(), "little") & self.mask)
        return (left << self.half) | right

    def __call__(self, i: int) -> int:
        if not 0 <= i < self.n:
            raise IndexError(i)
        x = self._encrypt(i)
        while x >= self.n:  # cycle-walking: re-encrypt until the value falls back into range(n)
            x = self._encrypt(x)
        return x


def stride_schedule(weights: dict, period: int):
    """A fixed pattern of `period` slots, round(w * period) per source, spread out by smooth
    weighted round-robin, and for each source the prefix counts of its slots in the pattern."""
    counts = {s: round(w * period) for s, w in sorted(weights.items())}
    assert sum(counts.values()) == period
    credit = dict.fromkeys(counts, 0)
    pattern = []
    for _ in range(period):
        for s in counts:
            credit[s] += counts[s]
        chosen = max(counts, key=lambda s: credit[s])
        credit[chosen] -= period
        pattern.append(chosen)
    prefix = {s: [0] for s in counts}
    for slot in pattern:
        for s in counts:
            prefix[s].append(prefix[s][-1] + (slot == s))
    return pattern, prefix


class StrideSampler:
    """Random access to the stream: sample t is computed from (seed, t) alone, with no replay."""

    def __init__(self, sizes: dict, weights: dict, seed: int, period: int = 20):
        self.sizes, self.seed = sizes, seed
        self.pattern, self.prefix = stride_schedule(weights, period)

    def drawn(self, s: str, t: int) -> int:
        """Documents of source s drawn before step t: whole periods plus a prefix count."""
        periods, r = divmod(t, len(self.pattern))
        return periods * self.prefix[s][-1] + self.prefix[s][r]

    def permutation(self, s: str, epoch: int) -> FeistelPermutation:
        key = hashlib.blake2b(f"{self.seed}/{s}/{epoch}".encode(), digest_size=16).digest()
        return FeistelPermutation(self.sizes[s], key)

    def __getitem__(self, t: int):
        s = self.pattern[t % len(self.pattern)]
        epoch, within = divmod(self.drawn(s, t), self.sizes[s])
        return s, self.permutation(s, epoch)(within)


sizes = {"web": 1000, "wiki": 50, "code": 200}
weights = {"web": 0.7, "wiki": 0.15, "code": 0.15}

perm = FeistelPermutation(1000, key=b"demo-key")
order = [perm(i) for i in range(1000)]
assert sorted(order) == list(range(1000))  # a bijection on range(1000)
assert order[:20] != list(range(20))  # and not the identity
walk = perm.evaluations / 1000
assert walk < 4  # 4**half / n bounds the expected walk length; here 1024 / 1000
print(f"Feistel on range(1000): permutation {sorted(order) == list(range(1000))}, {walk:.3f} encryptions per index")

sampler = StrideSampler(sizes, weights, seed=1)
stream = [sampler[t] for t in range(2000)]
for s in sizes:  # the closed-form count agrees with counting the stream
    assert all(sampler.drawn(s, t) == sum(1 for x, _ in stream[:t] if x == s) for t in (0, 7, 333, 1999))
resumed = StrideSampler(sizes, weights, seed=1)  # a fresh process that knows only the step counter
assert [resumed[t] for t in range(1234, 1300)] == stream[1234:1300]
wiki = [d for s, d in stream if s == "wiki"]
assert sorted(wiki[:50]) == list(range(50)) and sorted(wiki[50:100]) == list(range(50))  # each epoch is a permutation
print("drawn by step 2000:", {s: sampler.drawn(s, 2000) for s in sorted(sizes)})
print("sample at step 1,000,000,007:", StrideSampler(sizes, weights, seed=1)[1_000_000_007])
```

% output

```text
Feistel on range(1000): permutation True, 1.024 encryptions per index
drawn by step 2000: {'code': 300, 'web': 1400, 'wiki': 300}
sample at step 1,000,000,007: ('web', 377)
```

(sec-ch5-shuffling-right-granularity)=

### 5.3 Shuffling at the right granularity

Three levels of shuffling exist and they are not interchangeable:

- **Across shards**: the order in which shards are visited. Cheap, and sufficient to mix sources if each shard is single-source.
- **Within a shard**: the order of documents inside it. Needed because shards are often written in crawl or alphabetical order, which correlates documents.
- **Across the whole corpus**: a global permutation, which is what the sampler above does with per-source permutations. Ideal, but requires random access to any document, which object storage and sequential formats do not give cheaply.

Streaming loaders compromise with a shuffle buffer: read shards in a seeded order, hold a few thousand documents in memory, and emit from the buffer at random. The quality of the mix depends on the buffer size relative to the shard size, and a buffer smaller than a shard barely shuffles at all. Mosaic's streaming format and WebDataset both document their shuffle schemes precisely because an insufficiently shuffled stream shows up as a periodic loss curve, which is hard to diagnose after the fact.

```{admonition} The Sampler
:class: note
The notebook section {ref}`nb-dm-sampler` runs the mixture sampler, plots the realized source fractions at several temperatures with the tempered target weights marked, shows the epoch cap re-normalizing the mixture once the smallest source is exhausted, and checks resumption at a hundred random steps.
```

(sec-ch5-distributed-loading)=

## 6. Distributed Loading

(sec-ch5-sharding-across-ranks)=

### 6.1 Sharding across ranks

A training job with $n_{\text{ranks}}$ processes (data-parallel ranks) needs each step's global batch of $B$ sequences split into $n_{\text{ranks}}$ micro-batches of $B / n_{\text{ranks}}$, with every rank reading different data and no rank reading what another reads. Two designs:

- **Shard assignment.** Shards are partitioned among ranks, each rank gets every $n_{\text{ranks}}$-th shard, and each rank samples within its own shards. Simple, and what a WebDataset-style loader does. The drawback is that the global batch at step $t$ depends on $n_{\text{ranks}}$: change the number of GPUs and the data order changes, so a run resumed on a different cluster size is not the same run.
- **Global indexing.** The sampler produces a single global sequence of samples; the sample at global position $t B + i$ goes to rank $i \bmod n_{\text{ranks}}$ (or rank $\lfloor i\, n_{\text{ranks}} / B \rfloor$). Every rank runs the same sampler and takes its slice. The global batch is then independent of $n_{\text{ranks}}$, and a run can be resumed with a different number of ranks and see the same data in the same order.

The second property, that the stream is a function of `(seed, step)` and nothing else, is what makes a large run reproducible across hardware changes, and it costs nothing beyond requiring that every rank can read any shard. Mosaic Streaming {cite}`mosaicml2022streaming` is built around it: its sample order is fixed by the seed and the number of canonical nodes, so with `num_canonical_nodes` pinned it does not change with the number of GPUs. The default is not pinned: it is derived from the number of physical nodes of the initial run (64 times that number for the `py1s` and `py2s` shuffles) {cite}`mosaicml2026streamingdocs`.

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

(sec-ch5-throughput)=

### 6.2 Throughput

The loader's job is to never make the accelerator wait. The arithmetic for a modern run: a step over a 4-million-token batch on a large cluster takes a few seconds; the batch is 8 MB of `uint16` ids (16 MB as `uint32`), spread over all ranks, so each rank reads kilobytes per step, trivial for local disk and comfortable for object storage with a few requests in flight. Text formats change the picture less in volume than in work: the same batch is some 16 MB of UTF-8 that must be decompressed, parsed and tokenized, which is CPU time on the step's critical path unless it is prefetched, and whose output depends on the tokenizer library's version at run time. This is the practical argument for pre-tokenized binaries: not storage (gzip-compressed text is comparable in size to 16-bit ids), but zero parsing, random access, and a token stream fixed when the dataset is built rather than when it is read.

The remaining ingredients are standard: a few worker processes per rank that prefetch the next batches, reads of whole shards or large ranges rather than per-document requests, and a local cache for shards that will be read again in a later epoch.

```{admonition} Sharding Across Ranks
:class: tip
The notebook section {ref}`nb-dm-distributed` simulates eight ranks under both designs, checks that each covers the corpus exactly once, and shows the global batch at a fixed step staying identical as the world size changes under global indexing and changing under shard assignment.
```

(sec-ch5-loader-workers)=

### 6.3 Worker processes within a rank

Each rank usually reads through $n_{\text{loader}}$ worker processes, which is a second level of splitting below the ranks, and the place where a common bug lives. With a map-style dataset the loader hands each worker a list of indices, and nothing is read twice. With an iterable dataset, the kind a streaming loader is, every worker receives a copy of the whole dataset object; the PyTorch documentation warns that "naive multi-process loading will often result in duplicated data" {cite}`pytorch2026data`. Each worker then yields the rank's entire stream, and the rank trains on every sample $n_{\text{loader}}$ times. The loss curve looks normal, only the effective dataset is smaller. The fix is to split the rank's stream again by worker inside the dataset's iterator, using the worker's id and the worker count that `torch.utils.data.get_worker_info()` reports. The listing simulates four ranks of three workers each over 1,200 samples: without the split each sample is seen three times, and with it once.

```python
def rank_stream(rank: int, world_size: int, n: int):
    """The sample ids a rank owns: every world_size-th id."""
    return range(rank, n, world_size)


def worker_items(rank, world_size, n, worker_id, num_workers, split_by_worker: bool):
    """What one DataLoader worker of one rank yields from an iterable dataset. Each worker holds
    a full replica, so without splitting each one yields the rank's whole stream."""
    stream = rank_stream(rank, world_size, n)
    return list(stream[worker_id::num_workers]) if split_by_worker else list(stream)


world_size, n_loader, n = 4, 3, 1200
for split in (False, True):
    seen = [x for r in range(world_size) for w in range(n_loader) for x in worker_items(r, world_size, n, w, n_loader, split)]
    print(f"split by worker {split!s:5s}: {len(seen):,} samples yielded, {len(set(seen)):,} distinct, "
          f"each seen {len(seen) / len(set(seen)):.0f} times")
naive = [x for w in range(n_loader) for x in worker_items(0, world_size, n, w, n_loader, False)]
fixed = [x for r in range(world_size) for w in range(n_loader) for x in worker_items(r, world_size, n, w, n_loader, True)]
assert len(naive) == n_loader * len(set(naive))  # every sample of the rank n_loader times
assert sorted(fixed) == list(range(n))  # each sample once across all ranks and workers
```

% output

```text
split by worker False: 3,600 samples yielded, 1,200 distinct, each seen 3 times
split by worker True : 1,200 samples yielded, 1,200 distinct, each seen 1 times
```

(sec-ch5-statistics-tests-monitoring)=

## 7. Statistics, Tests and Monitoring

(sec-ch5-statistics-per-version)=

### 7.1 What to compute for every version

A dataset version ships with its statistics, computed by the build and stored next to the manifest, because the questions they answer come up in every training post-mortem:

- Tokens and documents per source, and the realized mixture (which differs from the planned one once epoch caps and filtering have acted)
- Document length distribution per source (mean, median, percentiles, maximum); a change in the tail between versions usually means an extraction change
- Per-stage drop counts from the Chapter 4 pipeline, so that a version's keep rate can be compared with its predecessor's
- Deduplication rate within the version and overlap with the previous version
- Language distribution, and for code the distribution over languages and licenses
- Overlap with the evaluation suite after decontamination, per benchmark, reported as a number even when it is zero

{cite:t}`elazar2024whats` built exactly this kind of inspection tooling for the public corpora and found, among other things, that benchmark contamination and duplicated content were common across them. Statistics that are computed rather than asserted are the defense.

(sec-ch5-data-contracts)=

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

(sec-ch5-canaries-drift)=

### 7.3 Canaries and drift

Two monitoring practices borrowed from other fields:

- **Canaries.** Insert a handful of unique, random strings into the corpus at known positions and with known repetition counts. After training, test whether the model can complete them. Canaries measure memorization directly {cite}`carlini2019secret`, and a canary that appears in a version where it should not have is proof that an upstream shard leaked.
- **Drift.** Compare every statistic in Section 7.1 between consecutive versions and flag changes beyond a threshold. Most data bugs (an extractor update that drops tables, a language-identification model update that rejects more of one language, a source that silently went offline) show up as a drift in a per-source statistic before they show up in a loss curve.

A drift check needs a statistic and a threshold fixed before the comparison. For a length distribution, the two-sample Kolmogorov-Smirnov statistic, the largest gap between the two empirical distribution functions, has a large-sample critical value at significance level $\alpha$ of $\sqrt{-\ln(\alpha / 2) / 2}$ times the square root of the sum of the two sample sizes over their product. With 20,000 documents on each side and $\alpha = 0.001$ that is 0.0195. The listing compares a version with an unchanged rebuild and with a build in which an extractor update dropped the tables from 15% of documents: the first stays under the threshold and the second is flagged. On corpora of millions of documents the threshold becomes so small that real but harmless changes cross it, so in practice it is paired with a minimum effect size, such as a change in mean length of more than a few percent.

```python
import numpy as np


def ks_statistic(a, b) -> float:
    """Largest gap between the empirical distribution functions of two samples."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / a.size
                               - np.searchsorted(b, grid, side="right") / b.size)))


def ks_threshold(n: int, m: int, alpha: float = 0.001) -> float:
    """Large-sample critical value of the two-sample statistic at level alpha."""
    return float(np.sqrt(-np.log(alpha / 2) / 2) * np.sqrt((n + m) / (n * m)))


rng = np.random.default_rng(7)
v1 = rng.lognormal(mean=6.0, sigma=1.0, size=20_000).astype(int) + 1   # document lengths in tokens, version 1
v2_same = rng.lognormal(mean=6.0, sigma=1.0, size=20_000).astype(int) + 1  # a rebuild with nothing changed
v2_bug = v2_same.copy()
tables = rng.random(v2_bug.size) < 0.15   # an extractor update drops the tables from 15% of documents ...
v2_bug[tables] = (v2_bug[tables] * 0.6).astype(int) + 1  # ... which removes 40% of their tokens

limit = ks_threshold(v1.size, v2_same.size)
d_same, d_bug = ks_statistic(v1, v2_same), ks_statistic(v1, v2_bug)
print(f"threshold at alpha = 0.001: {limit:.4f}")
print(f"unchanged rebuild: D = {d_same:.4f}, flagged {d_same > limit}")
print(f"tables dropped:    D = {d_bug:.4f}, flagged {d_bug > limit}")
print(f"mean length {v1.mean():.4g} -> {v2_bug.mean():.4g} tokens")
assert d_same < limit < d_bug
```

% output

```text
threshold at alpha = 0.001: 0.0195
unchanged rebuild: D = 0.0049, flagged False
tables dropped:    D = 0.0356, flagged True
mean length 656.2 -> 617.9 tokens
```

```{admonition} Statistics and Contracts
:class: note
The notebook section {ref}`nb-dm-governance` computes per-source statistics for the demo corpus, runs a contract against a plan, injects a canary and finds it, and renders a data card from the manifest and statistics.
```

(sec-ch5-documentation-governance)=

## 8. Documentation and Governance

(sec-ch5-data-cards-datasheets)=

### 8.1 Data cards and datasheets

A dataset version is accompanied by a document that says what it is for someone who was not there when it was built. The two established templates:

- **Datasheets for Datasets** {cite}`gebru2021datasheets`: a questionnaire covering motivation, composition, collection process, preprocessing, uses, distribution and maintenance. It was written to make the people assembling a dataset answer the questions a user will have, and it remains the reference.
- **Data cards** {cite}`pushkarna2022data`: a structured summary intended to be published with the dataset, with an emphasis on what a downstream team needs to decide whether to use it.

The pretraining corpora that are well documented (The Pile {cite}`gao2020pile`, Dolma {cite}`soldaini2024dolma`, FineWeb {cite}`penedo2024fineweb`) publish a datasheet or card, the per-source statistics, the pipeline code and the ablation results together, and that combination is what allows anyone else to build on them. C4's post-hoc documentation {cite}`dodge2021documenting`, written about eighteen months after the T5 paper by authors none of whom had written it, found text from unexpected sources such as patents, machine-generated text, evaluation examples from benchmark datasets, and a blocklist filter that disproportionately removed text from and about minority individuals.

Croissant, a machine-readable metadata vocabulary built on schema.org, {cite}`akhtar2024croissant` is becoming the common format for the card's structured part (sources, licenses, fields, splits) so that hubs and loaders can read it; the prose part still needs a person.

(sec-ch5-licenses-provenance-record)=

### 8.2 Licenses and the provenance record

A corpus mixes sources under different terms, and the terms travel with the data. The version's metadata records, per source, the license or terms under which it was obtained, the opt-out policy applied and the date, and any restrictions that propagate to models trained on it. {cite:t}`longpre2023data` traced the lineage and licenses of more than 1,800 text datasets and found that widely used hosting sites omitted the license in more than 70% of cases and gave a wrong one in more than 50%; the only remedy is to carry the record forward from acquisition (Chapter 3) through every version rather than reconstructing it later.

Access control follows from the record. A corpus with a non-commercial source cannot be released under a commercial license as a whole, so either the source is separated into its own shards with their own terms, or the released version excludes it. Keeping sources in separate shards, which the naming convention in Section 2.2 already does, is what makes this a manifest operation rather than a rebuild.

(sec-ch5-removal)=

### 8.3 Removal

Requests to remove content arrive after release: an opt-out, a takedown, a discovered privacy problem, a license that turned out not to permit the use. Handling them is where the earlier design pays off:

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

The requests have legal drivers, and they differ in what they require. A license or terms of use that turn out not to permit training are a contract matter, settled by excluding the source (the record of {ref}`sec-ch5-licenses-provenance-record` says which shards). A website's opt-out is recorded with the date it was observed, so the record shows which crawls preceded it. Personal data is the strictest case: in the European Union, Article 17 of the General Data Protection Regulation gives a person "the right to obtain from the controller the erasure of personal data concerning him or her without undue delay" on grounds that include withdrawn consent, a successful objection and unlawful processing {cite}`eu2016gdpr`. Removing the document from the next version handles the data; whether a model trained on the old version must also forget it is the open question the lineage graph exposes.

Machine unlearning is the research answer to that question. {cite:t}`bourtoule2021machine` propose SISA training, which "reduces the computational overhead associated with unlearning": the training data is split into disjoint shards, a separate constituent model is trained on each, so a removal retrains only the model whose shard held the document. The same hash assignment used for splits decides the shard, and the cost of a batch of removals is the fraction of constituent models it touches. In the listing, five requests retrain all of one model, four of eight, and five of 64, 7.8% of the training compute. For a single large language model the analogue is retraining from the last checkpoint before the document was seen, which the data stream of {ref}`sec-ch5-determinism-resumption` makes possible to locate exactly.

```python
import hashlib

import numpy as np


def constituent(doc_id: str, n_shards: int) -> int:
    """SISA-style assignment: each document trains exactly one of n_shards constituent models."""
    return int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % n_shards


rng = np.random.default_rng(0)
requests = [f"doc{i}" for i in rng.choice(10_000_000, size=5, replace=False)]  # five removal requests
for n_shards in (1, 8, 64):
    hit = {constituent(d, n_shards) for d in requests}
    print(f"{n_shards:2d} constituent models: {len(hit)} retrained, {len(hit) / n_shards:.1%} of the training compute")
assert len({constituent(d, 64) for d in requests}) <= 5  # at most one model per request
```

% output

```text
 1 constituent models: 1 retrained, 100.0% of the training compute
 8 constituent models: 4 retrained, 50.0% of the training compute
64 constituent models: 5 retrained, 7.8% of the training compute
```

(sec-ch5-putting-it-together)=

## 9. Putting It Together

(sec-ch5-build)=

### 9.1 The build

A dataset build, in the sense of a software build, is a script under version control that:

1. Resolves its inputs to hashes: the source manifests from acquisition, the pipeline commit, the configuration, the tokenizer.
1. Runs the pipeline, writing shards to temporary names and renaming on completion.
1. Computes the manifest, root hash, statistics and contract results.
1. Fails if the contract fails.
1. Publishes the shards to content-addressed storage, the manifest and statistics to the version history, and tags the version.
1. Records the lineage: inputs, outputs and the build's own identity.

Everything a training run needs is then a tag. A run's configuration names the tag, the loader resolves it to a manifest at launch and records the root hash in the run metadata, and from then on the run's data is a function of that hash, the seed and the step.

(sec-ch5-tooling)=

### 9.2 Tooling

The pieces exist as libraries, and most teams assemble rather than write them:

- **Hugging Face `datasets`** {cite}`lhoest2021datasets` for Parquet and Arrow handling, streaming from the hub, and memory-mapped local caches; the hub itself for hosting versioned datasets with per-commit manifests.
- **Mosaic Streaming (MDS)** for a shard format that streams from object storage {cite}`mosaicml2022streaming` with a deterministic, resumable sample order that does not depend on the number of ranks once `num_canonical_nodes` is pinned; **WebDataset** {cite}`aizman2020high` for tar shards streamed sequentially with a seeded shard order and a shuffle buffer.
- **Megatron-LM, GPT-NeoX and their derivatives** for the indexed tokenized binary and the blended-dataset sampler that implements weighted mixing; epoch caps are set when the blend is planned.
- **DVC** {cite}`iterative2026dvc` and **lakeFS** {cite}`treeverse2026lakefs` for content-addressed versioning with lineage, in a Git-like workflow; **Git LFS** when the data is small enough.
- **Great Expectations** and similar for data contracts, though a few hundred lines of assertions in the build script are usually enough.
- **Croissant** {cite}`akhtar2024croissant` for machine-readable dataset metadata.

What no tool supplies is the discipline: that every version is produced by the build, that no shard is edited in place, that every run names a tag, and that the statistics and the card are updated with the data rather than after it.

(sec-ch5-exercises)=

## Exercises

1. **Bytes per token.** Take a few thousand documents and store them as gzip JSON Lines, as a `uint16` token binary, and as a `uint32` token binary. Report bytes per token for each and the time to read every document back. At what point does the text format's inspectability stop being worth its cost?
1. **Manifest diff.** Rebuild the notebook's corpus with the Chapter 4 repetition filter's thresholds changed by 10%. Diff the manifests. How many shards changed, and could you have predicted which ones from the shard naming scheme?
1. **Packing efficiency.** For document length distributions with means of 100, 1,000 and 10,000 tokens and a sequence length of 4,096, measure the truncation rate of chunking and the padding rate of best-fit packing. When does packing stop being worth its complexity?
1. **The mask's effect.** Train two small models on packed sequences with and without the document mask (the Chapter 1 notebook's transformer is enough). Compare the loss on the first tokens of each document, which are the ones the mask changes.
1. **$\mathcal{O}(1)$ seeking.** Design a sampler whose state at step $t$ is computed without replaying: a fixed schedule of mixture phases, a hash of `(seed, t)` for the source draw, and per-source cursors from prefix counts. Its stream will differ from `MixtureSampler`'s, so verify it against itself: iterating to step $t$ and seeking to step $t$ must agree at a thousand random steps.
1. **World-size invariance.** Implement a loader under both designs of Section 6.1 and resume a simulated run at step 500 with the world size changed from 8 to 4. Which design gives the same batches, and what does the other one skip or repeat?
1. **A removal request.** Pick a document in the notebook's corpus and carry out the three steps of Section 8.3: locate it, cut a new version without it, and list the versions (and, hypothetically, the models) affected. How long did each step take, and which would have been impossible without the index?

(sec-ch5-further-reading)=

## Further Reading

(sec-ch5-further-reading-formats-loaders-packing)=

### Formats, loaders and packing

- [Datasets: A Community Library for Natural Language Processing](https://arxiv.org/abs/2109.02846) {cite}`lhoest2021datasets` - the Arrow-backed library and hub conventions
- [Megatron-LM](https://arxiv.org/abs/1909.08053) {cite}`shoeybi2019megatron` - the indexed binary dataset and blended sampler used by many training codebases
- [Fewer Truncations Improve Language Modeling](https://arxiv.org/abs/2404.10830) {cite}`ding2024fewer` - best-fit packing and its measured effects
- [In-Context Pretraining](https://arxiv.org/abs/2310.10638) {cite}`shi2024incontext` - choosing which documents share a sequence
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) {cite}`muennighoff2023scaling` - epoch caps
- [DoReMi](https://arxiv.org/abs/2305.10429) {cite}`xie2023doremi` - learned mixture weights
- [mT5](https://arxiv.org/abs/2010.11934) {cite}`xue2021mt5` - temperature sampling across languages

(sec-ch5-further-reading-documentation-auditing-governance)=

### Documentation, auditing and governance

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) {cite}`gebru2021datasheets` and [Data Cards](https://arxiv.org/abs/2204.01075) {cite}`pushkarna2022data` - the documentation templates
- [What's In My Big Data?](https://arxiv.org/abs/2310.20707) {cite}`elazar2024whats` - inspection tooling and what it found in the public corpora
- [Documenting Large Webtext Corpora](https://arxiv.org/abs/2104.08758) {cite}`dodge2021documenting` - C4 audited after the fact
- [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787) {cite}`longpre2023data` - license tracking across repackaged datasets
- [The Secret Sharer](https://arxiv.org/abs/1802.08232) {cite}`carlini2019secret` - canaries for measuring memorization
- [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149) {cite}`carlini2024poisoning` - why distributed datasets need content hashes
- [The Pile](https://arxiv.org/abs/2101.00027) {cite}`gao2020pile`, [Dolma](https://arxiv.org/abs/2402.00159) {cite}`soldaini2024dolma` and [FineWeb](https://arxiv.org/abs/2406.17557) {cite}`penedo2024fineweb` - corpora released with their documentation, statistics and code

(sec-ch5-further-reading-tools)=

### Tools

- [Hugging Face Hub datasets](https://huggingface.co/docs/hub/datasets) - versioned hosting with Parquet conversion and the dataset viewer
- [Mosaic Streaming](https://github.com/mosaicml/streaming) - MDS shards with deterministic resumable shuffling
- [WebDataset](https://github.com/webdataset/webdataset) - tar-shard streaming
- [DVC](https://dvc.org/) and [lakeFS](https://lakefs.io/) - content-addressed data versioning with lineage
- [Croissant](https://github.com/mlcommons/croissant) - machine-readable dataset metadata
- [Great Expectations](https://greatexpectations.io/) - data contracts as tests

______________________________________________________________________

*Part of LLM Atlas: A Guide to Large Language Models*
