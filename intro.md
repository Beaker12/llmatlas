# LLM Atlas: A Guide to Large Language Models

LLM Atlas is a Jupyter Book about how large language models work, from tokenization and attention through the mathematics of training. Each written chapter pairs explanations and derivations with runnable PyTorch code and a companion notebook that executes on a CPU.

## Status

Chapters 1 to 5 are written. The remaining chapters are planned and not yet written; see the [Roadmap](roadmap.md).

**Chapter 1: LLM Foundations**

- Tokenization, embeddings and positional encodings
- Attention and multi-head attention
- Transformer architecture components
- The language modeling objective and perplexity
- From RNNs to transformers

**Chapter 2: Mathematical Foundations**

- Probability and information theory for language modeling
- Optimization: SGD, momentum, Adam, learning-rate schedules
- Backpropagation through layer normalization and attention
- Scaling laws and compute-optimal training
- Numerical stability: mixed precision, loss scaling, initialization

**Chapter 3: Data Acquisition and Sources**

- What the published pretraining mixtures are made of, and how much is enough
- Common Crawl: snapshots, WARC files and reading a slice of the crawl
- Crawling responsibly: robots.txt, politeness and the economics of fetching
- Curated sources: Wikipedia, books, papers, code and dialogue
- Rights, licences, consent and the shrinking data commons
- Multilingual and synthetic sources, mixture planning and provenance

**Chapter 4: Data Preprocessing and Cleaning**

- Why most raw web text is discarded, and what the retained fraction looks like
- Extraction, Unicode normalisation and language identification
- Heuristic, perplexity-based and classifier-based quality filters, and how to measure them
- Exact, near-duplicate (MinHash and LSH) and substring deduplication
- Personal data, toxicity and benchmark decontamination
- Assembling a reproducible pipeline with per-stage drop accounting

**Chapter 5: Dataset Management and Versioning**

- Storage formats: JSON Lines, Parquet, tar shards and tokenized binaries with a document index
- Versioning by content hash: manifests, version roots, diffs and lineage
- Packing documents into sequences and masking attention across document boundaries
- Deterministic, resumable mixture sampling with temperature and epoch caps
- Sharding across ranks without the global batch depending on the world size
- Statistics, data contracts, canaries, data cards and handling removal requests

## How to use this book

Read a chapter, then run its notebook (linked at the top of each chapter) and change the parameters. Exercises at the end of each chapter suggest experiments. Citations link to the [Bibliography](bibliography.md).

## Prerequisites

- Linear algebra and calculus at an introductory level
- Python and some PyTorch
- Basic machine learning concepts

## Contributing

Corrections and suggestions are welcome as issues or pull requests on [GitHub](https://github.com/Beaker12/llmatlas). The book is published under the MIT License.
