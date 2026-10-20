# LLM Atlas: A Guide to Large Language Models

LLM Atlas is a Jupyter Book about how large language models work, from tokenization and attention through the mathematics of training. Each written chapter pairs explanations and derivations with runnable PyTorch code and a companion notebook that executes on a CPU.

## Status

Chapters 1 and 2 are written. The remaining chapters are planned and not yet written; see the [Roadmap](roadmap.md).

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

## How to use this book

Read a chapter, then run its notebook (linked at the top of each chapter) and change the parameters. Exercises at the end of each chapter suggest experiments. Citations link to the [Bibliography](bibliography.md).

## Prerequisites

- Linear algebra and calculus at an introductory level
- Python and some PyTorch
- Basic machine learning concepts

## Contributing

Corrections and suggestions are welcome as issues or pull requests on [GitHub](https://github.com/Beaker12/llmatlas). The book is published under the MIT License.
